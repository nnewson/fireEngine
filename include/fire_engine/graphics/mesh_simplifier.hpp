#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Sentinel for MeshCollapse::vl / vr when a collapse's edge has no second (boundary) face, or when
// the position-welded edge is non-manifold and the fixed-arity vsplit can't encode it. Numerically
// equal to vdpm.hpp's kInvalidVertex (the forest consumes these ids directly).
inline constexpr uint32_t kNoCollapseApex = 0xFFFFFFFFu;

// One recorded edge collapse: vertex `removed` is merged into `kept`. Phase 1 uses *subset*
// (endpoint) collapse — `kept` stays at its original position, so every simplified level indexes
// the unchanged original vertex buffer. `position` is `kept`'s position (recorded for the Phase 2/3
// progressive-mesh hierarchy, where a vertex split reintroduces `removed` at its original spot and
// geomorphs it out from `kept`). `error` is the quadric error this collapse incurs.
struct MeshCollapse
{
    uint32_t kept{0};
    uint32_t removed{0};
    Vec3 position{};
    float error{0.0f};
    // Cumulative geometric deviation radius for VDPM: a conservative screen-space estimate (not a
    // rigorous bound) of how far the surface subsumed by this collapse sits from the original,
    // accumulated up the collapse tree so a coarse collapse over a curved region carries the whole
    // region's deviation. `error` (RMS) is left untouched for discrete/VIPM selection.
    float deviationRadius{0.0f};
    // Cumulative EUCLIDEAN deviation radius — the SH-02 shadow-silhouette channel. Accumulated with
    // the same running-sum envelope as deviationRadius but from the Euclidean step, so an in-plane
    // collapse (which point-to-plane reads as ~0) still contributes. A conservative cumulative
    // ESTIMATE, explicitly NOT a Hausdorff bound: it measures the removed sample's displacement and
    // is one-sided, so simplified surface added outside the original outline isn't counted.
    float shadowDeviationRadius{0.0f};
    // UV deviation radius: the worst-case texture-space error the region will show, so VDPM can
    // refine regions whose parameterisation stretches — or whose welded seam wedges span charts —
    // even where the geometry is flat (the geometric radius alone can't see that). Per collapse it
    // is max(smooth stretch on a containing face, spread between the removed position's atlas
    // wedges); accumulated by MAX (not the geometric channel's running sum) because a UV error is a
    // screen-space discontinuity — the eye sees the single worst jump in the region, not a
    // compounding envelope. See the collapse() note.
    float uvDeviationRadius{0.0f};
    // Cumulative shading-normal deviation, in radians: the angle between the removed vertex's
    // normal and the normal the covering face interpolates to at its spot, accumulated up the
    // collapse tree the same way. Catches shading error the geometry and UV channels miss — a
    // smooth-shaded curved region whose vertices sit near-coplanar (small geometric δ) but whose
    // normals fan, so a coarse collapse there flattens the lighting. Projected on its own channel
    // in VDPM's refineForView.
    float normalDeviationRadius{0.0f};
    // Cumulative tangent deviation, in radians: the drift of the shader's TBN frame (the MAX of the
    // T- and B-axis angular deviation, T Gram-Schmidt'd against N, B = cross(N,T)·handedness), so
    // it sees both a tangent roll AND a handedness (w) flip — the raw tangent xyz, which the shader
    // never samples directly, would miss the flip. A tangent-space normal map is sampled in this
    // frame, so a coarse collapse that swings it tilts the mapped normals even where the vertex
    // normal + geometry barely move — a distinct error source from the shading-normal channel. Its
    // own VDPM channel.
    float tangentDeviationRadius{0.0f};
    // Spatial support radius (world/object units): the extent of the region this collapse subsumes,
    // a bounding sphere around `kept` grown to enclose the collapsed subtree. The angular channels
    // (normal/tangent, radians) multiply their chord by the support radius projected to pixels — an
    // angular deviation matters in proportion to the SCREEN FOOTPRINT of the region it affects, so
    // projecting a length (support) makes the shading score scale-invariant, exactly like the
    // geometric channel, instead of the old fixed-length projection. Conservative (a running
    // enclose-the-child bound, so it can over-estimate the region's true diameter — a real bounding
    // sphere would tighten it); never under-bounds.
    float supportRadius{0.0f};
    // CONSERVATIVE normal cone of the region this collapse subsumes: `normalConeAxis` (unit) bounds
    // every finest face normal in the collapse subtree within acos(`normalConeCos`) — a bounding
    // cap, NOT the exact normal set (intersecting its grazing band means a silhouette *may* exist,
    // not that one does). Accumulated bottom-up like supportRadius. Lets VDPM's refineForView do a
    // conservative per-split back-face / silhouette test — combined at runtime with the support
    // sphere's view-angle spread (a perspective camera's view direction varies across the region),
    // in object space so non-uniform world scale doesn't distort the circular cone — instead of the
    // per-frame coverage-repair sweep. RUNTIME CONTRACT: `normalConeCos <= 0` (half-angle >= 90°)
    // is the no-cull sentinel — treat the region as always-potentially-visible / -silhouette, never
    // cull (never feed it to a sin(halfAngle) test). `normalConeCos` 1 = flat region; lower = more
    // curved.
    Vec3 normalConeAxis{0.0f, 0.0f, 1.0f};
    float normalConeCos{1.0f};
    // The vsplit apexes for the VDPM vertex forest: the third vertices of the one (boundary) or two
    // (interior) live faces on the collapsing edge, recorded here on the true canonical topology
    // the simplifier collapses so the forest need not re-derive them by replaying the stream. `vr`
    // is `kNoCollapseApex` on a boundary edge; BOTH are `kNoCollapseApex` when the position-welded
    // edge is non-manifold (>2 incident faces) and Hoppe's fixed-arity vsplit can't represent it —
    // the forest then leaves `removed` a root (always active) at that spot rather than emit a
    // corrupt dependency, which isolates those few edges instead of cascading a topology desync.
    // Canonical (position-welded) vertex ids, matching `kept` / `removed`.
    uint32_t vl{kNoCollapseApex};
    uint32_t vr{kNoCollapseApex};
};

// Result of a simplification: an index buffer into the ORIGINAL vertex array (no vertex data
// moves), plus the largest collapse error applied to reach the requested ratio.
struct SimplifiedMesh
{
    std::vector<uint32_t> indices;
    float error{0.0f};
};

// One exact cut through a progressive collapse stream. `collapseCount` is the number of recorded
// collapses already replayed to produce `indices`; it is the topology identity for VIPM. `error`
// remains only the screen-space selection metric.
struct ProgressiveLod
{
    std::vector<uint32_t> indices;
    float error{0.0f};
    std::size_t collapseCount{0};
    // Max cumulative shadowDeviationRadius over this cut's EXACT collapse prefix (the first
    // `collapseCount` collapses — the authoritative topology identity, not an error-derived guess).
    // The shadow-LOD selector projects this into shadow-map texels; a conservative estimate, not a
    // bound. LOD0 is 0.
    float shadowDeviation{0.0f};
};

// The cut-level shadow-deviation policy, pure so it is directly testable and so buildProgressive
// stays declarative: the MAX cumulative `shadowDeviationRadius` over the first `collapseCount`
// collapses. The channel is cumulative and monotone up the collapse tree, so the prefix max is the
// cut's value; an empty prefix (LOD0) is 0.
//
// An INVALID prefix — `collapseCount` beyond the recorded stream — asserts in debug and returns
// infinity in release. Clamping instead would silently reduce over a shorter prefix and hand back a
// plausible UNDER-estimate for a cut whose real collapses are unknown; infinity forces LOD0, which
// is the only honest answer when the authoritative identity doesn't match the stream.
[[nodiscard]] float shadowDeviationForCut(std::span<const MeshCollapse> collapses,
                                          std::size_t collapseCount) noexcept;

// A single replay artifact for discrete LOD + VIPM. Every LOD index buffer and every geomorph
// target must be derived from this same collapse stream and these exact cuts; error values are not
// used to infer topology.
struct ProgressiveMesh
{
    std::vector<ProgressiveLod> lods;
    std::vector<MeshCollapse> collapses;
};

namespace detail
{

// One collapse's VDPM deviation, factored out of the simplifier's inner loop so the correspondence
// decision is unit-testable in isolation (it is otherwise buried in topology iteration). Each
// channel is the per-collapse *step*, before the tree accumulation the simplifier layers on top.
struct CollapseDeviation
{
    float geometry{
        0.0f}; // point-to-plane vs the nearest surviving triangle (0 across an in-plane gap)
    // EUCLIDEAN distance to the nearest surviving triangle — the shadow-silhouette channel (SH-02).
    // `geometry` above measures point-to-PLANE, which reads ~0 for a coplanar collapse even when
    // the removed vertex lands well outside every surviving triangle: exactly the in-plane
    // silhouette movement a shadow shows and the camera view barely does. This channel measures
    // that displacement.
    //
    // Scope, stated precisely: it measures the tangential displacement OF THE REMOVED SAMPLE, and
    // is one-sided. A coplanar collapse that bridges a concavity adds simplified surface outside
    // the original outline while the removed vertex stays close to the new surface — that addition
    // is not measured here. Conservative ESTIMATE, never a Hausdorff bound.
    //
    // Infinity when the inputs are non-finite, so a caller forces LOD0 rather than trusting a
    // number derived from NaNs.
    float shadow{0.0f};
    float normal{
        0.0f}; // shading-normal angular step, radians (vs the clamped closest-point interp)
    // Tangent-frame angular step, radians: the MAX of the T-axis and B-axis deviation of the
    // shader's TBN frame, so it catches a tangent roll AND a handedness (w) flip (which flips B); 0
    // without tangents. Independent of the shading-normal channel (which covers N).
    float tangent{0.0f};
    float uv{0.0f}; // smooth stretch on the containing face, MAX'd with the atlas wedge spread
    // Whether `removed` projected INSIDE some surviving face. The geometry/UV channels use it (UV
    // requires containment); the shading channels deliberately do NOT (they clamp to the nearest
    // point), so a false here with a non-zero `normal` is exactly the no-containing-face case the
    // decoupled correspondence must still measure — the property the regression test pins.
    bool hadContainingFace{false};
};

// Measure the deviation of collapsing `removed` into its surviving one-ring `faces` (vertex-id
// triples). The attribute spans are indexed by those ids: `nrm` shading normals, `tanT`/`tanB` the
// per-vertex TBN frame axes (shader construction — the tangent channel measures this frame, not raw
// tangent xyz, so a handedness flip is seen); `removedWedgeUv` holds `removed`'s render-wedge UVs
// (seam-spread term). Pure + free-standing, so a test can feed a hand-built one-ring — including
// one where `removed` sits outside every face, or a handedness-flipped frame — and assert the
// channels directly. Mirrors exactly what the simplifier's inner loop records per collapse.
// `kept` is the collapse's surviving representative: with no valid surviving triangle to measure
// against (every face degenerate, or an empty one-ring) the shadow channel falls back to
// |removed - kept|, because a subset collapse maps the removed vertex onto exactly that vertex —
// defaulting to zero there would claim a free collapse.
[[nodiscard]] CollapseDeviation
measureCollapseDeviation(std::uint32_t removed, std::uint32_t kept,
                         std::span<const std::array<std::uint32_t, 3>> faces,
                         std::span<const Vec3> pos, std::span<const Vec3> nrm,
                         std::span<const Vec3> tanT, std::span<const Vec3> tanB,
                         std::span<const Vec2> uv, std::span<const Vec2> removedWedgeUv) noexcept;

} // namespace detail

// Backend-swappable triangle-mesh simplifier. Our quadric-error-metric implementation is the first
// backend; the interface lets a different one (meshoptimizer, or a test oracle) drop in without
// touching call sites.
class MeshSimplifier
{
public:
    MeshSimplifier() = default;
    virtual ~MeshSimplifier() = default;

    MeshSimplifier(const MeshSimplifier&) = default;
    MeshSimplifier& operator=(const MeshSimplifier&) = default;
    MeshSimplifier(MeshSimplifier&&) = default;
    MeshSimplifier& operator=(MeshSimplifier&&) = default;

    // Simplify to roughly `targetRatio` (0..1) of the input triangle count. Returns indices into
    // the original vertices. Degenerate/boundary-locked meshes may simplify less than requested.
    [[nodiscard]] virtual SimplifiedMesh simplify(std::span<const Vertex> vertices,
                                                  std::span<const uint32_t> indices,
                                                  float targetRatio) const = 0;
};

// Garland–Heckbert quadric-error-metric edge-collapse simplifier (subset placement +
// boundary-quadric preservation + a normal-flip veto). Deterministic. Vulkan-free — the whole thing
// is CPU + headless- testable.
class QuadricSimplifier : public MeshSimplifier
{
public:
    [[nodiscard]] SimplifiedMesh simplify(std::span<const Vertex> vertices,
                                          std::span<const uint32_t> indices,
                                          float targetRatio) const override;

    // The full greedy coarsening stream (finest → coarsest) — the raw material for the Phase 2/3
    // progressive-mesh hierarchy. `simplify(ratio)` is exactly the mesh after replaying the prefix
    // of this stream that reaches the target triangle count.
    [[nodiscard]] std::vector<MeshCollapse>
    collapseSequence(std::span<const Vertex> vertices, std::span<const uint32_t> indices) const;

    // Build LOD0 plus one exact progressive cut per requested ratio in a single replay. This is the
    // authoritative path for runtime LODs: discrete index buffers, cut ordinals, and VIPM morph
    // data all consume the returned artifact.
    [[nodiscard]] ProgressiveMesh buildProgressive(std::span<const Vertex> vertices,
                                                   std::span<const uint32_t> indices,
                                                   std::span<const float> ratios) const;
};

} // namespace fire_engine
