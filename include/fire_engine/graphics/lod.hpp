#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <fire_engine/graphics/gpu_handle.hpp>

namespace fire_engine
{

// LOD strategy, selectable at runtime so the modes coexist rather than supersede. Discrete = Phase
// 1 hard index-buffer swap. Continuous = VIPM geomorph (Phase 2): the level transitions are
// dissolved by sliding collapsing vertices onto their targets. ViewDependent = VDPM (Phase 3): a
// per-region active front refines different parts of one mesh to different detail into a per-frame
// index buffer.
enum class LodMode : uint8_t
{
    Discrete = 0,
    Continuous = 1,
    ViewDependent = 2,
};

// One discrete level of detail: an index buffer into the geometry's (shared, unchanged) vertex
// buffer, plus the world-space geometric deviation the simplifier introduced to reach it. LOD0 is
// the full mesh (error 0); coarser levels follow with increasing error.
struct GeometryLod
{
    BufferHandle indexBuffer{NullBuffer};
    uint32_t indexCount{0};
    float error{0.0f};
};

// Simplification ratios for the discrete LODs built beyond LOD0 (the full mesh), in order.
inline constexpr std::array<float, 2> kLodRatios{0.5f, 0.125f};

// Meshes with fewer triangles than this aren't worth simplifying (the selection + memory cost
// outweighs the saving).
inline constexpr std::size_t kMinLodTriangles = 512;

// Default screen-space error budget (pixels) a coarser LOD may introduce before it's rejected.
inline constexpr float kLodPixelErrorBudget = 2.0f;

// Shadow passes tolerate a coarser LOD than the main view (silhouette detail matters less in a
// shadow), so their pixel budget is scaled up by this factor.
inline constexpr float kShadowLodBias = 3.0f;

// VDPM silhouette boost: how much tighter the pixel budget is where a split's precomputed normal
// cone straddles the edge-on direction (0 = uniform screen-space error). Keeps contours dense.
inline constexpr float kVdpmSilhouetteBoost = 2.0f;

// VDPM GPU repair round budget: the number of snapshot detect→apply rounds the GPU repair fixpoint
// runs before its full-detail fallback (`vdpm_repair_fallback.comp`) seeds every unrefined split to
// guarantee a hole-free front. It is therefore a COST/QUALITY knob, not a correctness bound — the
// fallback closes any remainder regardless. Typical fronts converge in ≤ 2 rounds (docs/lod.md);
// the cap is set generously so the fallback almost never fires while still bounding worst-case
// cost.
inline constexpr std::uint32_t kVdpmGpuRepairRoundBudget = 24;

// VDPM UV channel scale: turns a collapse's cumulative UV-deviation radius into pixel-equivalent
// screen error against the same pixel budget (a texel-density stand-in — per-material texture
// resolution would refine it). The primary dial for texture fidelity vs triangle count; kept
// separate from the simplifier's kUvWeightFactor, which only orders collapses.
inline constexpr float kVdpmUvScale = 1.0f;

// VDPM shading channel scale: turns a collapse's cumulative shading-normal deviation (radians) into
// pixel-equivalent screen error against the same pixel budget, so a smooth-shaded curve that stays
// near-coplanar (invisible to the geometry channel) but whose normals fan still refines under
// magnification. The primary dial for shading fidelity vs triangle count; kept separate from the
// simplifier's collapse ordering. Radians are perceptually potent, so this rides below the UV/geom
// unit scale — tune it up if lighting still flattens at close range, down if the mesh over-refines.
inline constexpr float kVdpmNormalScale = 0.5f;

// VDPM tangent channel scale: as kVdpmNormalScale, but for the tangent-frame deviation (radians)
// that steers tangent-space normal-map sampling. Separate from the shading-normal dial so a
// normal-mapped asset's frame drift can be tuned independently of its interpolated-normal drift;
// reads 0 on meshes without tangents, so it costs nothing there.
inline constexpr float kVdpmTangentScale = 0.5f;

// VDPM refine/coarsen hysteresis: the active front persists across frames rather than rebuilding
// from coarsest every frame, so a split refines when its score exceeds the pixel budget but only
// coarsens once its score drops below this FRACTION of the budget. The dead band between the two
// keeps a split whose score hovers around the budget (small camera moves, sub-pixel jitter) from
// popping in and out each frame. 1.0 disables hysteresis (coarsen at the budget); lower = stickier
// / less popping but slightly more triangles held near the boundary.
inline constexpr float kVdpmCoarsenRatio = 0.6f;

// Material-aware multiplier on the shading-normal channel for a fully-glossy (roughness 0)
// material, ramped down to 1.0 at fully-rough. A shading-normal error reads far more strongly in a
// sharp specular highlight than on a diffuse surface, so a mirror-like material refines its normals
// harder. Applied per material in `vdpmChannelScales` (refine time), so the collapse stream stays
// material-agnostic. 1.0 = disable the gloss boost.
inline constexpr float kVdpmGlossyNormalBoost = 3.0f;

// Pick the coarsest LOD whose projected geometric error fits the pixel budget. `projScaleY` is the
// projection matrix's [1][1] term (= 1/tan(fovY/2)); a world deviation `e` at view distance `d`
// projects to `e·projScaleY·viewportHeight/(2d)` pixels. Returns 0 (the full mesh) when no coarser
// level is acceptable. Levels are ordered fine→coarse with non-decreasing error, so the first that
// overflows the budget ends the search.
[[nodiscard]] inline std::size_t selectLod(std::span<const GeometryLod> lods, float distance,
                                           float projScaleY, float viewportHeight,
                                           float pixelErrorBudget) noexcept
{
    std::size_t chosen = 0;
    const float d = distance > 1e-3f ? distance : 1e-3f;
    for (std::size_t i = 1; i < lods.size(); ++i)
    {
        const float projected = lods[i].error * projScaleY * viewportHeight / (2.0f * d);
        if (projected <= pixelErrorBudget)
        {
            chosen = i;
        }
        else
        {
            break;
        }
    }
    return chosen;
}

} // namespace fire_engine
