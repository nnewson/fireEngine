#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat3.hpp>
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
    // Spatial support radius (MeshCollapse::supportRadius): the extent of the region this split
    // covers. refineForView projects it to a screen-space extent (a projected radius, not an area)
    // and multiplies the angular channels' chord by it, so a shading error refines in proportion to
    // the projected extent of the region it affects (and scales like the geometry channel — the fix
    // for the old fixed-length angular projection).
    float supportRadius{0.0f};
    // CONSERVATIVE normal cone of the region this split covers (MeshCollapse::normalCone*):
    // `normalConeAxis` (unit) bounds every finest face normal in the subtree within
    // acos(`normalConeCos`) — a bounding cap, not the exact set. refineForView uses it for a
    // conservative per-split back-face / silhouette test, combined with the support sphere's view-
    // direction spread. `normalConeCos <= 0` is the no-cull sentinel (never cull; see
    // mesh_simplifier).
    Vec3 normalConeAxis{0.0f, 0.0f, 1.0f};
    float normalConeCos{1.0f};
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

// ---- VDPM per-split scoring: the single Vulkan-free authority --------------------------------
// `refineForView` derives these per-instance params once, then scores every split through
// `scoreVdpmSplit`. The GPU port (render/vdpm_gpu) uploads the SAME params and its shader
// reproduces `scoreVdpmSplit`, and the GPU harness compares against it directly — so the oracle,
// the uploader, and the shader can't drift into three subtly different scorings.

// Everything a view contributes to a split's score, derived ONCE per instance per frame. The
// camera-relative affine (`worldLinear` · local + `worldTranslationMinusCamera`) reproduces the
// world-space distance `|world·local − camera|` EXACTLY under any linear transform — non-uniform
// scale, shear, reflection, even a singular world — which an object-space camera + a scalar cannot.
// The object-space `cameraObj` (+ `facingSign`, cone flags) drives the object-space cone predicate,
// and is deliberately unusable on a singular world (then `coneUsable` is false: never cull, and
// positional scoring still runs off the affine).
struct VdpmViewParams
{
    Mat3 worldLinear;                 // the world matrix's linear part (object → world direction)
    Vec3 worldTranslationMinusCamera; // world translation − camera (the affine's constant term)
    Vec3 cameraObj;                   // camera in object space (cone predicate; unused if singular)
    float worldLengthScale{1.0f};     // σ_max of worldLinear: bounds object-space radii into world
    float facingSign{1.0f};           // sign(det): folds a reflection into the cone facing
    float projScaleY{1.0f};           // |proj[1][1]|: world length → NDC height
    float halfViewport{0.0f};         // viewportHeight / 2: NDC → pixels
    float silhouetteBoost{0.0f};
    // (No pixelBudget here — scoring produces raw screen-space errors; thresholding against the
    // budget is the front-DECISION stage's job, in refineForView / applyView.)
    float uvScale{1.0f};
    float normalScale{1.0f};
    float tangentScale{1.0f};
    bool coneUsable{true};      // false on a near-singular world (no reliable inverse)
    bool coneCullEnabled{true}; // cone may prove back-facing (material culls AND cone usable)
};

// Per-split scoring result: the four independent screen-space channel errors (pixels), the cone
// silhouette straddle, and whether the split is provably raster back-face-culled. `score()` is the
// value `refineForView`/`applyView` threshold against; a back-facing split scores 0.
struct VdpmSplitScore
{
    float geometry{0.0f};
    float uv{0.0f};
    float normal{0.0f};
    float tangent{0.0f};
    float straddle{0.0f};
    std::uint8_t backface{0};

    [[nodiscard]] float score() const noexcept
    {
        return backface != 0 ? 0.0f : std::max({geometry, uv, normal, tangent});
    }
};

// Derive the per-instance view params (the conservative σ_max bound, the determinant conditioning
// test, the object-space camera). Pure + Vulkan-free.
[[nodiscard]] VdpmViewParams makeVdpmViewParams(const Mat4& world, const Vec3& cameraPos,
                                                float projScaleY, float viewportHeight,
                                                float silhouetteBoost, bool rasterBackfaceCulling,
                                                float uvScale, float normalScale,
                                                float tangentScale);

// Score one split against the params. `parentPos`/`childPos` are the split's OBJECT-space parent
// (kept, = the support-sphere centre) and child (removed) positions. Pure + Vulkan-free — the exact
// math the GPU shader reproduces.
[[nodiscard]] VdpmSplitScore scoreVdpmSplit(const VdpmViewParams& params, const VertexSplit& s,
                                            const Vec3& parentPos, const Vec3& childPos);

namespace detail
{

// Result of the per-split visibility test: whether the split's whole region is provably raster
// back- face-culled, and a [0,1] silhouette-straddle weight (1 = the cone is centred on edge-on).
struct ConeVisibility
{
    bool backFacing{false};
    float straddle{0.0f};
};

// EXACT evaluation of a CONSERVATIVE visibility bound, factored out of refineForView so it is unit-
// testable in isolation (all args OBJECT-space). The split's finest face normals are bounded by a
// cone (`coneAxis`, half-angle acos `coneCos`) and its region by a support sphere (`regionCenter`,
// `supportRadius`); the camera subtends a view-direction half-angle asin(r/d) over that sphere. The
// combined spread θn+θv is formed WITHOUT trig via the cosine sum identity (GPU-friendly). Results:
//   backFacing — the whole bounded surface faces away from every view direction (safe to skip:
//   raster
//     back-face-culled). Only ever true when `cullEnabled` (the material actually culls back-faces)
//     AND the transform is orientation-valid; it is a one-sided PROOF — false means "not provably
//     hidden", never "provably visible".
//   straddle — how centrally the edge-on direction sits inside the cone+spread (a boost heuristic,
//     not a guarantee a silhouette exists). 1 when the cone straddles edge-on or is wider than a
//     hemisphere (the `coneCos <= 0` no-cull sentinel) or the camera is inside the support sphere.
// `facingSign` folds a reflection (negative-determinant world flips winding ⇒ the culled side is
// the object-space FRONT) into the test exactly: pass sign(det), or +1 for a normal transform.
[[nodiscard]] ConeVisibility coneVisibility(const Vec3& coneAxis, float coneCos,
                                            float supportRadius, const Vec3& regionCenter,
                                            const Vec3& cameraObj, float facingSign,
                                            bool cullEnabled) noexcept;

// Shared per-face repair classifiers — pure geometry, so the sequential CPU sweeps AND the parallel
// snapshot detector run the IDENTICAL projection math and differ only in WHEN they apply the
// returned targets (the sweep mutates while walking; the snapshot marks a required set against a
// settled front, closes it, applies, and re-detects). All positions are WORLD-space; `viewProj` is
// the JITTER-FREE proj*view. This is the single source of truth for the P2 joint-repair per-face
// policy.

inline constexpr std::uint32_t kInvalidCorner = 0xFFFFFFFFu;
// Coverage repair below this SCREEN area isn't worth it (a couple of pixels); shared so the runtime
// and the test validator apply the same policy. Converted to NDC per viewport (area A covers
// A·(w/2)·(h/2) px²), so a fixed NDC constant would mean different pixel sizes per viewport.
inline constexpr float kMinCoverageScreenAreaPx = 2.0f;

// A finest face's active-ancestor replacement winds AGAINST the original ⇒ the rasteriser
// back-face- culls it into a hole. World-space winding (correct under non-uniform / reflected
// transforms). A degenerate replacement (two ancestors coincident) is NOT a foldover — a neighbour
// covers it.
[[nodiscard]] bool isFoldover(const std::array<Vec3, 3>& original,
                              const std::array<Vec3, 3>& replacement,
                              bool replacementDegenerate) noexcept;

enum class CoverageRepairKind : std::uint8_t
{
    None,               // covered / not visible / sub-pixel / already at full detail here
    AllInactiveCorners, // force-refine every inactive corner (near-plane / degenerate /
                        // unprojectable)
    WorstInactiveCorner // force-refine the single most screen-displaced inactive corner (centroid
                        // escape)
};
struct CoverageRepair
{
    CoverageRepairKind kind{CoverageRepairKind::None};
    std::uint32_t worstCorner{kInvalidCorner}; // LOCAL corner 0..2, only for WorstInactiveCorner
};

// Classify one finest face's screen-space coverage repair against a settled front. `original` /
// `replacement` are the corners' world positions and their active-ancestor world positions;
// `replacementDegenerate` is whether two ancestors coincide (the face dropped from the emit);
// `cornerInactive[k]` marks a corner not yet at finest (a refinable target). Follows the exact P2
// policy: back-face-culled / fully behind / sub-pixel / centroid-covered ⇒ None; near-plane
// straddle / degenerate / unprojectable replacement ⇒ AllInactiveCorners; ordinary centroid escape
// ⇒ the worst- displaced inactive corner. When there is no inactive corner (full detail here) the
// conservative AllInactiveCorners cases collapse to None — the clean full-detail no-op.
[[nodiscard]] CoverageRepair
classifyCoverageRepair(const std::array<Vec3, 3>& original, const std::array<Vec3, 3>& replacement,
                       bool replacementDegenerate, const std::array<bool, 3>& cornerInactive,
                       const Vec3& cameraPos, const Mat4& viewProj, float viewportWidth,
                       float viewportHeight, bool rasterBackfaceCulling) noexcept;

} // namespace detail

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

    // Refine a split, first (recursively) force-refining any inactive dependency splits. vl/vr
    // errors are not monotone (see the [vdpm] probe), so a legal coarse-first refine can require a
    // lower-error neighbour split the budget alone wouldn't bring in. This is the recursive
    // operation the GPU-shaped `ParallelFront` (vdpm_parallel.hpp) reproduces as rank-ordered
    // parallel passes.
    bool forceRefine(uint32_t splitIndex);

    // Drive the whole front to the finest / coarsest extreme.
    void refineAll();
    void coarsenAll();

    // Selectively refine the front for a camera view (the per-frame VDPM driver). The front
    // PERSISTS across frames — it is NOT reset to coarsest each call. Each split is scored (the max
    // of its four screen-space channels, silhouette-boosted); then a refine pass (coarse-first, so
    // dependencies stay satisfied) pulls in every front-facing split whose score exceeds
    // `pixelBudget`, and a coarsen pass drops every refined split whose score falls below
    // `kVdpmCoarsenRatio × pixelBudget` (or is back-face-culled) and whose child is a leaf. The
    // dead band between the two thresholds is the HYSTERESIS: a split whose score hovers at the
    // budget doesn't pop in and out under small camera moves / sub-pixel jitter. The four channels
    // are geometry (`error`), UV-stretch (`uvError · uvScale`), shading normal (`normalError ·
    // normalScale`) and tangent frame (`tangentError · tangentScale`), so texture-, shading-, and
    // normal-map-frame-costly-but-flat regions all stay dense. Visibility comes from each split's
    // precomputed CONSERVATIVE normal cone (see `detail::coneVisibility`): a split whose whole cone
    // provably faces away (over the support-sphere view spread) skips discretionary refinement
    // (raster back-face-culled) but can still be pulled in as a visible split's dependency; a cone
    // straddling edge-on is silhouette-boosted (`silhouetteBoost`; 0 disables).
    // `rasterBackfaceCulling` MUST reflect whether the draw actually culls back-faces — pass FALSE
    // for a double-sided or blended material (whose back-faces are visible), or refinement of
    // visible geometry would be wrongly suppressed. `world` places the mesh; `projScaleY =
    // proj[1][1]`. Vulkan-free + headless.
    void refineForView(std::span<const Vertex> vertices, const Mat4& world, const Vec3& cameraPos,
                       float projScaleY, float viewportHeight, float pixelBudget,
                       float silhouetteBoost, bool rasterBackfaceCulling, float uvScale,
                       float normalScale, float tangentScale);

    // JOINT post-refinement repair (call after refineForView, MANDATORY before emission). Closes
    // both failure classes a *selective* (non-prefix) front leaves: FOLDOVERS (a replacement
    // triangle wound against the original — the rasteriser back-face-culls it into a hole) and
    // COVERAGE holes (at a silhouette the coarse replacement recedes inside a VISIBLE finest face's
    // projected footprint). Neither the deviation nor the collapse-order criteria see either. The
    // two are fixed TOGETHER, to a JOINT fixed point: a coverage force-refine can re-fold a
    // neighbour and a foldover force-refine can open a coverage hole, so they must iterate until a
    // complete cycle changes NOTHING — a single foldover-then-coverage phase order does not leave
    // the front foldover-free (that was a real bug). The two sweeps are private so a caller cannot
    // run them separately or misorder them. `viewProj` is proj*view (world applied separately,
    // matching refineForView); pass the JITTER-FREE proj*view. `rasterBackfaceCulling` MUST reflect
    // the draw's cull mode (as refineForView) — with culling OFF a double-sided/blended material's
    // back-faces render too and need coverage. The result is the deterministic fixed point reached
    // by THIS sequential repair schedule (not a least/unique one). Refinement-only (each sweep only
    // ACTIVATES splits, never coarsens ⇒ inflationary) ⇒
    // terminates; `jointRepairSweeps()` reports the sweep count.
    void repairFront(std::span<const Vertex> vertices, const Mat4& world, const Vec3& cameraPos,
                     const Mat4& viewProj, float viewportWidth, float viewportHeight,
                     bool rasterBackfaceCulling);

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

    // Per-frame repair diagnostics (overlay/regression watch): vertices each repair sweep pulled
    // back in (successful force-refines) during the last `repairFront`, and the number of joint
    // foldover+coverage sweep cycles it took to converge (<= initially-unrefined splits + 1; the +1
    // is the final cycle that proves convergence). All reset at the top of `repairFront` — the
    // repair API owns its own diagnostics, so an independent call never reports stale counts.
    [[nodiscard]] uint32_t foldoversRepaired() const noexcept
    {
        return foldoversRepaired_;
    }
    [[nodiscard]] uint32_t coverageRepaired() const noexcept
    {
        return coverageRepaired_;
    }
    [[nodiscard]] uint32_t jointRepairSweeps() const noexcept
    {
        return jointRepairSweeps_;
    }

    // Per-frame metric instrumentation (reset at the top of refineForView). For every budget-driven
    // over-budget *trigger*, the split is attributed to its **winning** channel — the one whose
    // score/budget ratio is largest (the dominant reason it triggered). These are TRIGGER counts,
    // not resulting-refine counts: one metric decision legitimately refines several dependency
    // splits through forceRefine. `max*Ratio` is the largest score/budget seen on that channel
    // across ALL splits this frame (including under-budget ones) — how hard the channel is pushing
    // / how close it is to firing, which the counts alone can't say (a normal count of 0 can't
    // distinguish "genuinely zero" from "reached 99% of budget"). This exposes the metric itself,
    // which the foldover/coverage counters do not. See the VDPM metric arc.
    struct ChannelStats
    {
        uint32_t geometryTriggers{0}; // over-budget triggers whose winning channel was geometry
        uint32_t uvTriggers{0};
        uint32_t normalTriggers{0};
        uint32_t tangentTriggers{0};
        float maxGeometryRatio{0.0f}; // max screenError/budget over all splits this frame
        float maxUvRatio{0.0f};
        float maxNormalRatio{0.0f};
        float maxTangentRatio{0.0f};
    };
    [[nodiscard]] const ChannelStats& channelStats() const noexcept
    {
        return channelStats_;
    }

private:
    [[nodiscard]] uint32_t activeAncestor(uint32_t canonicalVertex) const;

    // Outcome of one repair sweep, for the joint `repairFront` loop. `changed` — a force-refine
    // succeeded (the loop must run another cycle). `failedToProgress` — a REPAIRABLE violation (a
    // finest face with an INACTIVE corner) could not be advanced because that corner's valid
    // removing split failed to force-refine: a forest/logic inconsistency, so `repairFront` throws.
    // (A face that no-ops because it is already at full detail — e.g. the coverage near-plane
    // conservative case — is CLEAN termination, NOT a failure.)
    struct RepairSweepResult
    {
        bool changed{false};
        bool failedToProgress{false};
    };
    // ONE sweep over the finest faces (the joint loop in repairFront repeats them). These MUTATE
    // the front as they walk (deterministic sequential sweeps — the later parallel detector will
    // instead detect against a settled snapshot). Foldover: force-refine any face whose
    // active-ancestor replacement winds against the original (the rasteriser would back-face-cull
    // it into a hole). Coverage: force-refine any VISIBLE face whose projected centroid escapes its
    // replacement in NDC.
    RepairSweepResult repairFoldoversSweep(std::span<const Vertex> vertices, const Mat4& world);
    RepairSweepResult repairCoverageSweep(std::span<const Vertex> vertices, const Mat4& world,
                                          const Vec3& cameraPos, const Mat4& viewProj,
                                          float viewportWidth, float viewportHeight,
                                          bool rasterBackfaceCulling);

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
    // `ancestorCache_` memoises activeAncestor(v) for emit (the front is settled by then, so it is
    // stable) — a pure per-frame function, so the cache is behaviour-identical to the inline
    // computation. `mutable` because it is filled by a logically-const query method.
    mutable std::vector<uint32_t> ancestorCache_; // per canonical vertex (emit)

    // Persistent-front hysteresis scratch (per split), filled by refineForView's score pass and
    // read by its refine + coarsen passes. `splitScore_` is the split's max screen-space channel
    // score (0 for a back-face-culled split); `splitBackface_` marks a split whose whole support is
    // clearly back-facing (skip refine, allow coarsen). The front persists across frames, so these
    // describe only THIS frame's view.
    std::vector<float> splitScore_;
    std::vector<std::uint8_t> splitBackface_;

    // Repair diagnostics for the last repairFront (see the accessors above).
    uint32_t foldoversRepaired_{0};
    uint32_t coverageRepaired_{0};
    uint32_t jointRepairSweeps_{0};
    // Per-channel metric attribution for the last refineForView (see channelStats()).
    ChannelStats channelStats_;
};

} // namespace fire_engine
