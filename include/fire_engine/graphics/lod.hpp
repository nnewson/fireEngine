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
