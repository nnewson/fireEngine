#pragma once

#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/shadow_lod_resolver.hpp>
#include <fire_engine/graphics/shadow_map_validity.hpp>
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

// SH-05: which faces one shadow family keeps. A property of the PASS, not of the pipeline: the
// shadow pipelines declare cull mode dynamic, so this is set at record time and every family must
// name its policy — there is no static fallback to inherit if one forgets.
enum class ShadowFaceCull : std::uint8_t
{
    // Cascade / spot / point: the CASTER decides. Single-sided casters cull front faces (back faces
    // carry the depth, which is what keeps receiver acne off); a double-sided material culls
    // nothing, because front-culling a sheet authored face-on to the light discards the only faces
    // it has and it casts no shadow at all.
    PerCaster,
    // Self-shadow FIRST layer: keep everything, so the first light-facing surface is captured
    // whatever its winding. Was its own pipeline before SH-05 made cull mode dynamic.
    AllFaces,
    // Self-shadow SECOND layer: cull front faces so only back faces rasterise, which is what makes
    // the dual-depth rejection well-founded rather than a coin-flip on marginal fragments.
    BackFacesOnly,
};

// SH-05: one shadow family's two fragment paths. A draw picks between them by its caster's alpha
// classification, so the pair travels together — a recording site that could be handed the opaque
// path alone is a site where a cutout silently casts its quad.
struct ShadowPipelinePair
{
    PipelineHandle opaque{NullPipeline};
    PipelineHandle masked{NullPipeline};

    // Which path rasterises a caster with this classification. Deliberately a pure mapping on the
    // header rather than a branch buried in the recorder: reversing it would compile, rasterise,
    // and produce a solid shadow for every cutout in the scene with no test failing — so it is
    // pinned by tests/render/test_shadow_raster_policy.cpp.
    [[nodiscard]] PipelineHandle forCaster(ShadowCasterAlpha alpha) const noexcept
    {
        return alpha == ShadowCasterAlpha::Masked ? masked : opaque;
    }
};

// SH-05: the faces one draw keeps — the family's policy, resolved against the caster's own
// sidedness for the families where the caster decides.
//
// Pure, and public for the same reason `forCaster` is: every shadow pipeline declares cull mode
// dynamic, so this function IS the cull policy, and swapping two of its answers would silently
// restore the defect the item fixed (a double-sided sheet front-culled into casting nothing) or
// break the dual-depth self-shadow layer. Takes the caster's `doubleSided` flag rather than a
// DrawCommand so the mapping can be exercised exhaustively without building a draw.
[[nodiscard]] constexpr vk::CullModeFlags shadowCullMode(ShadowFaceCull policy,
                                                         bool casterIsDoubleSided) noexcept
{
    switch (policy)
    {
    case ShadowFaceCull::PerCaster:
        // A double-sided caster culls NOTHING. Front-culling one authored face-on to the light
        // discards the only faces it has, and it casts no shadow at all.
        return casterIsDoubleSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eFront;
    case ShadowFaceCull::AllFaces:
        return vk::CullModeFlagBits::eNone;
    case ShadowFaceCull::BackFacesOnly:
        return vk::CullModeFlagBits::eFront;
    }
    // Unreachable for a valid policy; the switch is exhaustive over the enum.
    return vk::CullModeFlagBits::eFront;
}

class Shadows
{
public:
    Shadows(const Device& device, Resources& resources);
    ~Shadows() = default;

    Shadows(const Shadows&) = delete;
    Shadows& operator=(const Shadows&) = delete;
    Shadows(Shadows&&) noexcept = default;
    Shadows& operator=(Shadows&&) noexcept = default;

    // The handle a shadow COMMAND carries (FrameInfo::shadowPipeline) — the marker that buckets a
    // draw into the shadow pass, not a promise about which pipeline rasterises it. Since SH-05 the
    // pass picks the fragment path per draw from the caster's alpha classification and the layer's
    // face policy, so this is the family's opaque path and deliberately nothing more: producers
    // must not be able to select a shadow pipeline, or the classification would have a second,
    // unchecked route into the pass.
    [[nodiscard]] PipelineHandle pipelineHandle() const noexcept
    {
        return shadowPipelines_.opaque;
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
    // need no clear).
    //
    // `validity` decides WHICH FAMILIES RECORD, and it is the same value the receiver read in
    // `LightUBO::shadowMapValidMask`. A family whose bit is clear draws nothing, clears nothing and
    // stamps no timestamp, so its diagnostic rows and its GPU time both stay at zero — that is the
    // honest report, since the views were not rasterised. Re-enabling is safe in the same frame:
    // this pass runs before anything samples a map, so the frame that turns a family back on
    // re-renders it before its first read. What makes SKIPPING safe is the other half of the same
    // value: the receiver is told the family is invalid and answers fully lit, rather than sampling
    // depth left behind by whichever frame last rendered it.
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
                    ShadowMapValidity validity, ShadowFrameStats& stats,
                    const GpuProfiler& profiler, uint32_t frameIndex) const;

private:
    Resources* resources_{nullptr};
    Pipeline shadowPipeline_;
    Pipeline shadowMaskedPipeline_;
    Pipeline selfShadowSecondPipeline_;
    Pipeline selfShadowSecondMaskedPipeline_;
    // The main (cascade / spot / point) pair, which the self-shadow FIRST layer also uses: since
    // SH-05 its cull-nothing policy is dynamic state rather than a distinct pipeline.
    ShadowPipelinePair shadowPipelines_{};
    ShadowPipelinePair selfShadowSecondPipelines_{};
    TextureHandle shadowMapHandle_{NullTexture};
    TextureHandle worldShadowMapHandle_{NullTexture};
    TextureHandle selfShadowFirstMapHandle_{NullTexture};
    TextureHandle selfShadowMapHandle_{NullTexture};
    TextureHandle spotShadowMapHandle_{NullTexture};
    TextureHandle pointShadowMapHandle_{NullTexture};
};

} // namespace fire_engine
