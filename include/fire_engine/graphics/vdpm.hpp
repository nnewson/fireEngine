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
// tree, which refineForView projects to screen pixels. `uvError` is the parallel UV deviation
// radius (MeshCollapse::uvDeviationRadius, MAX-accumulated per-wedge — the worst texture jump in
// the region, not a compounding envelope) — an independent texture-stretch/seam channel
// refineForView projects on its own, so a texture-costly-but-geometrically-flat region still
// refines. `normalError` is the parallel cumulative shading-normal deviation
// (MeshCollapse::normalDeviationRadius, radians) — a third channel for lighting error a
// smooth-shaded curve carries even where its vertices stay near-coplanar (small geometric error).
// `tangentError` is the fourth channel (MeshCollapse::tangentDeviationRadius, radians): the tangent
// frame drives normal-map sampling, so a collapse that swings it tilts the mapped normals
// independently of the shading normal (0 on meshes without tangents).
struct VertexSplit
{
    uint32_t parent{0};
    uint32_t child{0};
    uint32_t vl{0};
    uint32_t vr{0};
    float error{0.0f};
    float uvError{0.0f};
    float normalError{0.0f};
    float tangentError{0.0f};
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

// Build the vertex forest from the recorded collapse stream. Each collapse already carries its
// (vl, vr) apex dependencies, recorded by the simplifier on the true canonical topology it
// coarsened, so this is a faithful transcription — no adjacency replay, no risk of diverging from
// the simplifier's decisions. A collapse whose position-welded edge was non-manifold carries
// kNoCollapseApex (the fixed-arity vsplit can't encode >2 apexes) and is skipped, leaving `removed`
// a root. `vertices` supplies only the canonical vertex count. Vulkan-free + deterministic.
[[nodiscard]] VertexForest buildVertexForest(std::span<const Vertex> vertices,
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
    // split also refines if its UV-deviation channel (`uvError · uvScale`, projected the same way),
    // its shading channel (`normalError · normalScale`, likewise), or its tangent channel
    // (`tangentError · tangentScale`) exceeds the budget, so texture-, shading-, and
    // normal-map-frame-costly-but-flat regions all stay dense. `world` places the mesh; `projScaleY
    // = proj[1][1]`. The per-region result is the view-dependent LOD. Vulkan-free +
    // headless-testable.
    void refineForView(std::span<const Vertex> vertices, const Mat4& world, const Vec3& cameraPos,
                       float projScaleY, float viewportHeight, float pixelBudget,
                       float silhouetteBoost, float backfaceThreshold, float uvScale,
                       float normalScale, float tangentScale);

    // Post-refinement COVERAGE repair (call after refineForView with the frame's proj*view). A
    // closed, non-folded front can still leak the background: at a silhouette the coarse
    // replacement recedes inside a fine FRONT-FACING triangle's projected footprint, so the
    // rasterised surface no longer covers it. Deviation/foldover criteria are blind to this — it is
    // purely a screen-space coverage property. For each front-facing finest face whose projected
    // centroid falls OUTSIDE its active-ancestor replacement in NDC, force-refine the collapsed
    // corner with the largest screen-space displacement; repeat to a fixed point. Monotone (only
    // force-refines), so it converges — at worst to full detail, which covers exactly. `viewProj`
    // is proj*view (world is applied separately, matching refineForView). `viewportWidth/Height`
    // turn the area gate into pixels (resolution-independent). A face straddling the near plane
    // can't be projected, so it is refined conservatively (see the .cpp).
    void repairCoverage(std::span<const Vertex> vertices, const Mat4& world, const Vec3& cameraPos,
                        const Mat4& viewProj, float viewportWidth, float viewportHeight);

    [[nodiscard]] bool active(uint32_t canonicalVertex) const
    {
        return active_[canonicalVertex] != 0;
    }
    [[nodiscard]] bool refined(uint32_t splitIndex) const
    {
        return refined_[splitIndex] != 0;
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
    // Same, but fills a caller-owned buffer (cleared first) instead of allocating — the per-frame
    // render path passes a reused scratch vector so emission does no heap allocation each frame.
    void emitActiveIndices(std::span<const Vertex> vertices, std::span<const uint32_t> indices,
                           std::vector<uint32_t>& out) const;

    // Per-frame repair diagnostics (overlay/regression watch): vertices each repair pass pulled
    // back in (successful force-refines) during the last refineForView + repair cycle — the
    // `active_==0` guard makes it a dedup'd per-frame work count. Reset at the top of
    // refineForView.
    [[nodiscard]] uint32_t foldoversRepaired() const noexcept
    {
        return foldoversRepaired_;
    }
    [[nodiscard]] uint32_t coverageRepaired() const noexcept
    {
        return coverageRepaired_;
    }

private:
    [[nodiscard]] uint32_t activeAncestor(uint32_t canonicalVertex) const;
    // Refine a split, first (recursively) force-refining any inactive dependency splits. vl/vr
    // errors are not monotone (see the [vdpm] probe), so a legal coarse-first refine can require a
    // lower-error neighbour split the budget alone wouldn't bring in.
    bool forceRefine(uint32_t splitIndex);
    // Post-refinement repair: force-refine any finest face whose active-ancestor replacement winds
    // AGAINST its original winding (a foldover). refineForView's per-vertex screen-space budget is
    // a linear-collapse criterion; a *selective* front is a non-prefix cut, so it can flip a
    // triangle the simplifier's linear wouldFlip() never saw — the rasteriser back-face-culls the
    // flipped replacement and punches a hole to the background. This drives such faces back toward
    // the original geometry (monotone: only force-refines, so it converges, at worst to full
    // detail).
    void repairFoldovers(std::span<const Vertex> vertices, const Mat4& world);

    VertexForest forest_;
    // uint8_t, not vector<bool>: this is per-frame mutation-heavy state, and the bit-proxy is
    // awkward to debug and no memory win at this scale. 0 = inactive/unrefined, 1 = active/refined.
    std::vector<std::uint8_t> active_;                   // per canonical vertex
    std::vector<std::uint8_t> refined_;                  // per split
    std::vector<uint32_t> dependents_;                   // per vertex: refined splits requiring it
    std::vector<std::array<uint32_t, 3>> finestFaces_;   // canonical, welded, deduped
    std::vector<uint32_t> weld_;                         // original vertex -> canonical
    std::vector<std::vector<uint32_t>> canonicalWedges_; // canonical -> original render wedges

    // Per-frame scratch, reused across frames so the per-frame path allocates nothing steady-state.
    // `facingCache_` memoises facingOf(v) within a refineForView call (a vertex is a witness of
    // many splits, so it is otherwise recomputed repeatedly); `ancestorCache_` memoises
    // activeAncestor(v) for emit (the front is settled by then, so it is stable). Both are pure
    // per-frame functions, so the cache is behaviour-identical to the inline computation. `mutable`
    // because they are filled by logically-const query methods.
    mutable std::vector<float> facingCache_;        // per canonical vertex (refineForView)
    mutable std::vector<std::uint8_t> facingValid_; // per canonical vertex: facingCache_ populated?
    mutable std::vector<uint32_t> ancestorCache_;   // per canonical vertex (emit)

    // Repair counters for the last refineForView + repair cycle (see the accessors above).
    uint32_t foldoversRepaired_{0};
    uint32_t coverageRepaired_{0};
};

} // namespace fire_engine
