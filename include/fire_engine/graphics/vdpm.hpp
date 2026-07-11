#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// View-dependent progressive meshing (VDPM, rendering-spine #3 Phase 3). Promotes Phase 2's linear
// collapse stream into a vertex forest with per-split dependencies, so an active front can refine
// different regions of one mesh to different detail (screen-space error + silhouette) with a
// per-frame index buffer. A split fires only when its dependency neighbourhood is present, so
// adjacent regions at different detail meet without T-junction cracks. CPU-first (this module is
// Vulkan-free + headless-testable); a GPU-driven front is the planned follow-on.
//
// TOPOLOGY IDENTITY, like the simplifier: the forest and the active front live over the
// **canonical, position-welded** vertices (glTF splits a corner into several render wedges at UV /
// normal seams; those all weld to one canonical vertex, or the front would treat seam duplicates as
// permanently-independent topology and corrupt its invariants). Rendered indices map back to the
// original render wedges only at emit time — `emitActiveIndices` restores each corner's nearest
// wedge with the same pattern the simplifier's index emit uses, so UV/normal seams keep their
// identity.

// One vertex split — the inverse of collapse i, over canonical vertices. Splitting `parent`
// reintroduces `child` (the collapse's `removed`) between the two faces of the collapsed edge,
// whose far apex vertices are `vl` and `vr`. A boundary edge has only one adjacent face, so `vr ==
// kInvalidVertex`. Hoppe's fixed-size vsplit encoding — the split is legal iff `parent` and `vl`
// are active and (`vr == kInvalidVertex` || `vr` active), so no variable-length dependency list is
// needed. Coarsening additionally needs the split to be currently refined with `child` a leaf (no
// refined descendants); the ActiveFront tracks that. `error` is the simplifier's **cumulative
// geometric deviation radius** (MeshCollapse::deviationRadius) — a conservative estimate of how far
// the region this collapse subsumes sits from the original surface, accumulated up the collapse
// tree, which refineForView projects to screen pixels. `uvError` is the parallel cumulative UV
// deviation radius (MeshCollapse::uvDeviationRadius) — an independent texture-stretch channel
// refineForView projects on its own, so a texture-costly-but-geometrically-flat region still
// refines.
struct VertexSplit
{
    uint32_t parent{0};
    uint32_t child{0};
    uint32_t vl{0};
    uint32_t vr{0};
    float error{0.0f};
    float uvError{0.0f};
};

// Sentinel for a missing neighbour (boundary edge) in VertexSplit::vl / vr.
inline constexpr uint32_t kInvalidVertex = 0xFFFFFFFFu;

// The vertex forest built from a collapse stream: one VertexSplit per recorded collapse (finest ->
// coarsest), plus, per canonical vertex, the index of the split that removes it (kNoSplit if it
// survives to the coarsest level). Everything indexes the canonical (position-welded) vertex set;
// no vertex data moves (subset placement, inherited from the simplifier). `vertexCount` is the
// canonical count. Render-wedge restoration is a separate emit concern (see the module note).
struct VertexForest
{
    std::vector<VertexSplit> splits;     // one per collapse, in stream (finest -> coarsest) order
    std::vector<uint32_t> removingSplit; // per canonical vertex: split index that removes it
    std::size_t vertexCount{0};          // canonical vertex count
};

// Sentinel in VertexForest::removingSplit for a vertex never removed by the recorded stream.
inline constexpr uint32_t kNoSplit = 0xFFFFFFFFu;

// Build the vertex forest from the finest index buffer + the recorded collapse stream. Welds the
// input to canonical (position-welded) topology, then replays the collapses over an evolving
// adjacency view to recover each split's (vl, vr) apex dependencies. Each collapsing edge must have
// exactly one live adjacent face (boundary -> vr = kInvalidVertex) or two (interior -> the two
// opposite apexes); zero means the stream and the topology view diverged, and more than two is
// non-manifold — both are rejected. Vulkan-free + deterministic (headless-tested).
[[nodiscard]] VertexForest buildVertexForest(std::span<const Vertex> vertices,
                                             std::span<const uint32_t> indices,
                                             std::span<const MeshCollapse> collapses);

// A selectively-refinable active front over the vertex forest. Holds the current
// per-canonical-vertex active state and per-split refined state, exposes legal refine/coarsen
// (Hoppe vsplit/ecol), and emits the current active triangle set. CPU-side, Vulkan-free +
// headless-testable. Phase 3b drives it per frame from screen-space error + silhouette into a
// dynamic index buffer.
//
// Invariants: a root (never-removed) canonical vertex is always active. Refining a split needs its
// parent + vl (+ vr if not boundary) active and the split not already refined; it activates the
// child. Coarsening needs the split refined and the child a leaf — `dependents_` counts the refined
// splits that require a vertex as parent/vl/vr, so `dependents_[child] == 0` is exactly "no refined
// split depends on the child", which is the leaf condition.
class ActiveFront
{
public:
    [[nodiscard]] static ActiveFront build(std::span<const Vertex> vertices,
                                           std::span<const uint32_t> indices,
                                           std::span<const MeshCollapse> collapses);

    // Apply / undo one split. Returns false (leaving the front unchanged) if illegal in the current
    // state, so callers can attempt an op without pre-checking.
    bool refine(uint32_t splitIndex);
    bool coarsen(uint32_t splitIndex);

    // Drive the whole front to the finest / coarsest extreme.
    void refineAll();
    void coarsenAll();

    // Selectively refine the front for a camera view (the per-frame VDPM driver): reset to
    // coarsest, then refine every split (coarse-first, so dependencies stay satisfied) whose
    // world-space error projects beyond `pixelBudget` screen pixels. Silhouette regions — where the
    // vertex's world normal is near edge-on to the view — are held to a tighter budget via
    // `silhouetteBoost` (0 disables it), so contours stay dense. Clearly back-facing reps (signed
    // facing < -`backfaceThreshold`) skip *discretionary* refinement — they are back-face-culled,
    // so detail there is wasted — but can still be pulled in as a visible split's dependency. A
    // split also refines if its UV-deviation channel (`uvError · uvScale`, projected the same way)
    // exceeds the budget, so texture-costly-but-flat regions stay dense. `world` places the mesh;
    // `projScaleY = proj[1][1]`. The per-region result is the view-dependent LOD. Vulkan-free +
    // headless-testable.
    void refineForView(std::span<const Vertex> vertices, const Mat4& world, const Vec3& cameraPos,
                       float projScaleY, float viewportHeight, float pixelBudget,
                       float silhouetteBoost, float backfaceThreshold, float uvScale);

    [[nodiscard]] bool active(uint32_t canonicalVertex) const
    {
        return active_[canonicalVertex];
    }
    [[nodiscard]] bool refined(uint32_t splitIndex) const
    {
        return refined_[splitIndex];
    }
    [[nodiscard]] const VertexForest& forest() const noexcept
    {
        return forest_;
    }

    // The current active triangle set as canonical-vertex index triples: the finest faces mapped
    // through the front's collapses (each corner to its active ancestor), dropping any that
    // collapse to a degenerate. Render-wedge restoration is layered on top by emitActiveIndices
    // (3a.3).
    [[nodiscard]] std::vector<std::array<uint32_t, 3>> emitActiveCanonical() const;

    // The current active triangle set as RENDER indices into the original vertex array: each active
    // face's corners restored to the nearest render wedge at their active-ancestor position (the
    // same seam-preserving pattern the simplifier's index emit uses), so UV/normal seams keep their
    // identity. This is the index buffer a draw would use.
    [[nodiscard]] std::vector<uint32_t> emitActiveIndices(std::span<const Vertex> vertices,
                                                          std::span<const uint32_t> indices) const;

private:
    [[nodiscard]] uint32_t activeAncestor(uint32_t canonicalVertex) const;
    // Refine a split, first (recursively) force-refining any inactive dependency splits. vl/vr
    // errors are not monotone (see the [vdpm] probe), so a legal coarse-first refine can require a
    // lower-error neighbour split the budget alone wouldn't bring in.
    bool forceRefine(uint32_t splitIndex);

    VertexForest forest_;
    std::vector<bool> active_;                           // per canonical vertex
    std::vector<bool> refined_;                          // per split
    std::vector<uint32_t> dependents_;                   // per vertex: refined splits requiring it
    std::vector<std::array<uint32_t, 3>> finestFaces_;   // canonical, welded, deduped
    std::vector<uint32_t> weld_;                         // original vertex -> canonical
    std::vector<std::vector<uint32_t>> canonicalWedges_; // canonical -> original render wedges
};

} // namespace fire_engine
