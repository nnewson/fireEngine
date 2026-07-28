#pragma once

#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/shadow_lod_resolver.hpp>
#include <fire_engine/graphics/shadow_render_view.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/gpu_profiler.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;

// Per-frame state for one active point shadow caster — the renderer hands one
// of these to recordPass for every point light that earned a shadow slot, so
// the shadow fragment shader can compute linear distance/range against the
// light's world position.
struct PointShadowCaster
{
    Vec3 worldPosition{};
    float range{0.0f};
};

class Shadows
{
public:
    Shadows(const Device& device, Resources& resources);
    ~Shadows() = default;

    Shadows(const Shadows&) = delete;
    Shadows& operator=(const Shadows&) = delete;
    Shadows(Shadows&&) noexcept = default;
    Shadows& operator=(Shadows&&) noexcept = default;

    [[nodiscard]] PipelineHandle pipelineHandle() const noexcept
    {
        return shadowPipelineHandle_;
    }

    // `views` is the frame's shadow view set and the ONLY source of each iteration's transform
    // (SH-03): every iteration takes one mandatory ShadowRenderView and uses its matrix to cull,
    // its projection descriptor to select a LOD, and its logical identity to key hysteresis. A
    // physical slot the set reports inactive is not rasterised — there is no second opinion to
    // consult, which is what makes "absent means inactive" true at the point of use.
    //
    // `resolver` is MUTATED: each accepted caster is resolved through it (per-frame cache + staged
    // hysteresis), so a caster's level is decided per VIEW rather than replayed from the camera.
    // Resolution happens AFTER the per-view filter, so a caster this view rejects acquires no
    // history against it. The caller commits or discards the staged history once the frame's fate
    // is known.
    //
    // `activeSelfShadowCasters` bounds the self-shadow slot loop (slots are assigned densely, and
    // an unassigned slot's layers are never sampled — no fragment carries that slot index — so they
    // need no clear). `renderWorldShadow` gates the world-only CSM: only skinned fragments sample
    // it (shader.frag gates on hasSkin), so a frame with no skinned draw skips those iterations,
    // and the frame that reintroduces one re-renders the map before anything samples it.
    //
    // `stats` (SH-01) is MUTATED: every iteration marks its view rasterised and observes every
    // command it walks, so a view that renders nothing is still reported. Rows are keyed by
    // PHYSICAL slot, which is stable across frames only while the dense light assignment is. The
    // observed verdict is the FILTER's alone, which is what keeps `candidateDraws - drawnDraws`
    // exactly the filter's yield.
    //
    // THROWS if an accepted caster resolves to no geometry — a corrupt unresolved command, which is
    // neither skippable (the caster would vanish from one shadow map silently) nor reportable as a
    // cull. Every recoverable case resolves to the whole mesh instead.
    void recordPass(vk::CommandBuffer cmd, std::span<const DrawCommand> shadowDraws,
                    std::span<const DrawCommand> worldOnlyShadowDraws,
                    std::span<const DrawCommand> selfShadowDraws, int activeSelfShadowCasters,
                    int activeSpotCasters, std::span<const PointShadowCaster> pointCasters,
                    const ShadowRenderViewSet& views, ShadowLodResolver& resolver,
                    float lodBudgetTexels, ShadowLodHysteresis hysteresis, bool cullingEnabled,
                    bool renderWorldShadow, ShadowFrameStats& stats, const GpuProfiler& profiler,
                    uint32_t frameIndex) const;

private:
    Resources* resources_{nullptr};
    Pipeline shadowPipeline_;
    Pipeline selfShadowFirstPipeline_;
    Pipeline selfShadowSecondPipeline_;
    PipelineHandle shadowPipelineHandle_{NullPipeline};
    PipelineHandle selfShadowFirstPipelineHandle_{NullPipeline};
    PipelineHandle selfShadowSecondPipelineHandle_{NullPipeline};
    TextureHandle shadowMapHandle_{NullTexture};
    TextureHandle worldShadowMapHandle_{NullTexture};
    TextureHandle selfShadowFirstMapHandle_{NullTexture};
    TextureHandle selfShadowMapHandle_{NullTexture};
    TextureHandle spotShadowMapHandle_{NullTexture};
    TextureHandle pointShadowMapHandle_{NullTexture};
};

} // namespace fire_engine
