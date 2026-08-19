#pragma once

#include <cstddef>

#include <fire_engine/graphics/shadow_caster_alpha.hpp>
#include <fire_engine/graphics/shadow_diagnostics.hpp>
#include <fire_engine/graphics/shadow_face_cull.hpp>
#include <fire_engine/graphics/shadow_pass_plan.hpp>
#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/gpu_profiler.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;

// (There is no `PointShadowCaster` any more. A point light's world position and effective range are
// raster content of its six faces, so they live in the view set beside the matrices they belong to
// — `ShadowRenderView::pointLightDepth()` — and reach the pass inside the prepared view. The array
// that used to carry them beside the set was a second copy of a position the set already held, and
// the pass found it by arithmetic on the face slot.)

// `ShadowFaceCull` and `shadowEffectiveCull` now live in `graphics/shadow_face_cull.hpp` — the
// policy is Vulkan-free and the cache's content descriptor has to record the effective answer, so
// the mapping cannot live behind a Vulkan type. This header keeps only the translation to Vulkan.

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

// The Vulkan spelling of one effective cull answer. The POLICY decision is
// `shadowEffectiveCull` (graphics/shadow_face_cull.hpp) and this is only its translation, so there
// is one place that decides and one place that speaks Vulkan — the cache's content descriptor
// records the same effective value this converts, rather than a parallel derivation of it.
[[nodiscard]] constexpr vk::CullModeFlags shadowCullMode(ShadowEffectiveCull cull) noexcept
{
    return cull == ShadowEffectiveCull::None ? vk::CullModeFlagBits::eNone
                                             : vk::CullModeFlagBits::eFront;
}

// Convenience for the recorder, which holds the family policy and the caster's sidedness: one call
// instead of nesting the two. Pinned by tests/render/test_shadow_raster_policy.cpp, which is what
// keeps a reversed answer from silently restoring the defect SH-05 fixed (a double-sided sheet
// front-culled into casting nothing) or breaking the dual-depth self-shadow layer.
[[nodiscard]] constexpr vk::CullModeFlags shadowCullMode(ShadowFaceCull policy,
                                                         bool casterIsDoubleSided) noexcept
{
    return shadowCullMode(shadowEffectiveCull(policy, casterIsDoubleSided));
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

    // Records the frame's shadow work — and NOTHING ELSE decides what that work is (arc 2 #4).
    //
    // `plan` is the whole input. It carries every view's transform, extent, depth bias, depth mode
    // and light, the draws each of its layers rasterises in order, and what each view DOES this
    // frame. There is deliberately no draw span, no view set and no resolver here any more: the
    // decisions were all made in `prepareShadowFrame`, and a recorder that could still re-filter or
    // re-resolve would be a second answer to a question the cache has already answered. Whatever
    // the comparison decided was the content is exactly what gets recorded, because it is the only
    // description of the work that survives preparation.
    //
    // WHICH VIEWS RECORD is each entry's disposition. A view marked `Reused` rasterises nothing —
    // its image already holds the right depth — and a family with nothing to record stamps no
    // timestamp, so its GPU time reads zero rather than measuring an empty span. A family the plan
    // never prepared (suppressed by `--no-shadows`, or fitted to no light) draws nothing, clears
    // nothing and times nothing, and the receiver was told the same thing through
    // `LightUBO::shadowMapValidMask`, which is derived from this same plan.
    //
    // `stats` (SH-01) is MUTATED, but only with RASTER PASSES: preparation already claimed each row
    // and observed every draw it walked. The identity is re-checked against the claim at every
    // layer, so a recorder that rasterised view B into the row view A claimed is refused rather
    // than reported under A's name.
    //
    // THROWS if a recorded view's row was never claimed or holds a different identity — a
    // contradiction between the plan and the diagnostics, which is not a degraded frame.
    void recordPass(vk::CommandBuffer cmd, const ShadowFramePlan& plan, ShadowFrameStats& stats,
                    const GpuProfiler& profiler, uint32_t frameIndex) const;

private:
    // Where one prepared layer rasterises: the depth image, the single-layer attachment view, the
    // array layer its barriers target, and the fragment paths that write it.
    //
    // Resolved as ONE value from (family, layer kind) rather than four lookups at the call site.
    // The self-shadow pair is why: its two layers differ in image AND in pipelines, and a call site
    // free to pick them separately could bind the second layer's shaders to the first layer's image
    // — which rasterises the dual-depth rejection into the map it is meant to be sampling, with
    // every counter and timing still reading correctly.
    struct LayerTarget
    {
        vk::Image image{};
        vk::ImageView view{};
        uint32_t layer{0};
        ShadowPipelinePair pipelines{};
    };
    [[nodiscard]] LayerTarget layerTarget(ShadowViewGroup group, std::size_t slot,
                                          ShadowLayerKind kind) const;

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
