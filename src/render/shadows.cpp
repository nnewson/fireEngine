#include <fire_engine/render/shadows.hpp>

#include <cassert>
#include <format>
#include <optional>
#include <stdexcept>
#include <string_view>

#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

namespace
{

// One-subresource (single mip, single array layer) layout transition through
// synchronization2. Dynamic rendering does no implicit attachment transitions,
// so each shadow iteration brackets its draw with these.
void imageLayerBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageAspectFlags aspect,
                       uint32_t layer, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                       vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                       vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::ImageMemoryBarrier2 b{
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = aspect,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = layer,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}

struct ShadowDrawFilter
{
    const Frustum* frustum{nullptr};
    int selfShadowSlot{-1};

    [[nodiscard]] bool accepts(const DrawCommand& dc) const
    {
        if (selfShadowSlot >= 0 && dc.selfShadowSlot != selfShadowSlot)
        {
            return false;
        }
        if (frustum == nullptr)
        {
            return true;
        }
        // A caster whose bounds are STALE (cloth: a compute pass rewrites the vertices this box was
        // measured from) cannot be rejected by them. The box says roughly where the caster was in
        // its bind pose and nothing about where the drawn geometry is, so a frustum test against it
        // can only produce false rejections — a cloth that is genuinely in this view, dropped. It
        // is admitted until storage geometry carries a conservative envelope of its own.
        if (dc.shadowBoundsKind != ShadowCasterBoundsKind::Exact)
        {
            return true;
        }
        return frustum->intersects(dc.shadowBounds);
    }
};

// Which diagnostic view an iteration is rasterising (SH-01). MANDATORY on every recording call:
// an optional target would let a future shadow path silently skip instrumentation, and a missing
// row is indistinguishable from a view that legitimately drew nothing.
//
// `slot` is the PHYSICAL array slot — cascade index, self-shadow slot, spot slot, or
// `shadowPointViewSlot(p, face)`. Those slots are densely reassigned in scene-gather order each
// frame, so a row is a stable *array position*, NOT a stable light identity: if a light appears or
// disappears, later lights move rows. Panel labels must say "point slot 1, face 4" rather than
// naming a light. Cross-frame ownership would need a stable light ID, which fixed-capacity
// indexing cannot supply.
class ShadowViewTarget
{
public:
    // All four values are required and there is no default construction: `{}` would otherwise
    // compile into a null sink pointed at Cascade 0, which either crashes or — worse — silently
    // bills one view's work to another. The reference makes the sink's existence a type-level fact.
    ShadowViewTarget(ShadowFrameStats& stats, ShadowViewGroup group, std::size_t slot,
                     bool countSelection) noexcept
        : stats_{&stats},
          group_{group},
          slot_{slot},
          countSelection_{countSelection}
    {
    }
    ShadowViewTarget() = delete;

    [[nodiscard]] ShadowViewStats& view() const noexcept
    {
        return stats_->view(group_, slot_);
    }
    [[nodiscard]] ShadowViewGroup group() const noexcept
    {
        return group_;
    }
    [[nodiscard]] std::string_view groupName() const noexcept
    {
        return toString(group_);
    }
    [[nodiscard]] std::size_t slot() const noexcept
    {
        return slot_;
    }
    // False only for the self-shadow SECOND depth layer: it re-rasterises the same logical view, so
    // its cost counts but its LOD selection must not be counted twice.
    [[nodiscard]] bool countSelection() const noexcept
    {
        return countSelection_;
    }

private:
    ShadowFrameStats* stats_; // never null: bound from a reference
    ShadowViewGroup group_;
    std::size_t slot_;
    bool countSelection_;
};

// A diagnostic row that two logical views tried to claim in one frame, or a view rasterising with
// no identity at all. Terminal because the alternative is a row whose counters are the sum of two
// unrelated views under one of their names — worse than no measurement, because it reads like one.
[[noreturn]] void contradictoryShadowViewRow(std::string_view group, std::size_t slot)
{
    throw std::runtime_error(
        std::format("shadow view row {} slot {} was claimed by two different logical views in one "
                    "frame (or by a view with no identity)",
                    group, slot));
}

// A shadow command that could not be resolved into geometry (SH-03). TERMINAL, on the same
// reasoning as the view set's rejections: the request is corrupt render input, and both ways of
// continuing are worse than stopping — dropping the draw leaves a caster missing from one shadow
// map with nothing to say so, and counting it as filtered corrupts the one metric the per-view
// diagnostics promise. In a Dev build the resolver's own assertion fires first, at the request that
// was malformed; under NDEBUG this throw carries the same refusal to main().
[[noreturn]] void unresolvableShadowCaster(std::uint32_t objectId)
{
    throw std::runtime_error(
        std::format("shadow caster (objectId {}) resolved to no geometry — its unresolved command "
                    "carries no drawable base mesh",
                    objectId));
}

// Everything one iteration needs to turn unresolved casters into its own draws (SH-03).
//
// REFERENCE-BOUND with no default: `{}` would otherwise compile into a context with no view and no
// resolver, which in release resolves every caster to an empty draw — a shadow map that renders
// nothing, reported as if it had. Making the two mandatory at construction is the same argument
// ShadowViewTarget above makes for its stats sink.
class ShadowLodContext
{
public:
    ShadowLodContext(const ShadowRenderView& view, ShadowLodResolver& resolver, float budgetTexels,
                     ShadowLodHysteresis hysteresis) noexcept
        : view_{&view},
          resolver_{&resolver},
          budgetTexels_{budgetTexels},
          hysteresis_{hysteresis}
    {
    }
    ShadowLodContext() = delete;

    [[nodiscard]] ResolvedShadowDraw resolve(const DrawCommand& dc) const noexcept
    {
        return resolver_->resolve(dc.shadowRequest, *view_, dc.shadowBounds, budgetTexels_,
                                  hysteresis_);
    }
    // The identity this iteration is rasterising — the same one its resolutions are keyed on, so a
    // diagnostic row and a hysteresis entry can never name different views.
    [[nodiscard]] const ShadowLogicalViewId& logicalId() const noexcept
    {
        return view_->logicalId();
    }
    // Records that this family drew this caster for this view. Called only where the draw is
    // actually recorded, so membership means "rasterised", not "considered".
    void noteDrawn(ShadowViewGroup group, const ShadowGeometryRequest& request) const noexcept
    {
        resolver_->noteDrawn(group,
                             ShadowLodStateKey{request.casterId, request.generation, logicalId()});
    }

private:
    const ShadowRenderView* view_; // never null: bound from a reference
    ShadowLodResolver* resolver_;  // never null: bound from a reference
    float budgetTexels_;
    ShadowLodHysteresis hysteresis_;
};

void recordShadowDrawBucket(vk::CommandBuffer cmd, std::span<const DrawCommand> shadowDraws,
                            const Resources& resources, ShadowPipelinePair pipelines,
                            ShadowFaceCull cullPolicy, const ShadowPushConstants& viewConstants,
                            ShadowDrawFilter filter, const ShadowLodContext& lod,
                            const ShadowViewTarget& target)
{
    // Before walking the span, so a view that renders and clears with nothing to draw still
    // reports as rasterised — that empty-but-rendered view is itself a finding.
    ShadowViewStats& viewStats = target.view();
    if (!viewStats.beginRasterPass(lod.logicalId()))
    {
        // The row already belongs to a different logical view this frame, or the view arrived with
        // no identity. Continuing would blend two views' counters into one row and label it with
        // one of their names — evidence that looks like a measurement and is not. Terminal, like
        // every other contradiction between what the renderer thinks it is drawing and what the
        // shadow state says.
        contradictoryShadowViewRow(target.groupName(), target.slot());
    }

    // The pipeline currently bound, so the two fragment paths can interleave freely within one
    // iteration: a family's casters are one span, and splitting it by material would either reorder
    // the draws or walk it twice. NullPipeline means "nothing bound yet in this iteration".
    PipelineHandle boundPipeline = NullPipeline;
    for (const auto& dc : shadowDraws)
    {
        // FILTER FIRST, resolve second. Selecting for a caster this view is about to drop would
        // give it a dead band against a view it never appears in — a skinned caster would
        // accumulate hysteresis against every other object's self-shadow map — and would evaluate
        // wholly-rejected perspective casters outside the domain the projection model is good for.
        const bool accepted = filter.accepts(dc);
        const ResolvedShadowDraw resolved = accepted ? lod.resolve(dc) : ResolvedShadowDraw{};
        // TERMINAL, before anything is counted. The resolver returns drawable geometry for any
        // request that carries base geometry — including every recoverable fallback — so a
        // non-drawable result means the producer emitted a caster it could not describe. Skipping
        // it would drop the caster from this shadow map silently, and folding it into the observed
        // verdict would make it indistinguishable from a cull rejection, breaking the promise that
        // `candidateDraws - drawnDraws` is exactly the filter's yield.
        if (accepted && !resolved.drawable())
        {
            unresolvableShadowCaster(dc.objectId);
        }
        // One observation per walked command, carrying the FILTER's verdict — nothing else. The
        // full-detail count is what this view was OFFERED; the resolved count is what it will pay.
        viewStats.observe(dc.shadowRequest.baseIndexCount / 3, accepted, resolved.indexCount / 3,
                          static_cast<std::uint32_t>(resolved.level), resolved.reason,
                          target.countSelection());
        if (!accepted)
        {
            continue;
        }
        // SH-05: the fragment path this caster needs, from the classification on its REQUEST — the
        // single place that fact is stored, and the same field the resolver just read to decide the
        // level. Never from anything the producer could point at a pipeline with directly.
        const PipelineHandle pipelineHandle = pipelines.forCaster(dc.shadowRequest.alpha);
        const bool pipelineChanged = pipelineHandle != boundPipeline;
        if (pipelineChanged)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             resources.vulkanPipeline(pipelineHandle));
            boundPipeline = pipelineHandle;
        }
        const vk::PipelineLayout layout = resources.vulkanPipelineLayout(pipelineHandle);
        // Cull mode is dynamic on every shadow pipeline (SH-05), so it is set per draw and not once
        // per pass: within one iteration a single-sided and a double-sided caster need different
        // answers, and the pipeline carries none.
        cmd.setCullMode(shadowCullMode(cullPolicy, dc.doubleSided));
        if (dc.vertexBuffer != NullBuffer)
        {
            cmd.bindVertexBuffers(0, resources.vulkanBuffer(dc.vertexBuffer), {vk::DeviceSize{0}});
        }

        vk::IndexType indexType =
            dc.indexType == DrawIndexType::UInt32 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
        // The RESOLVED buffer, never the command's: a shadow command carries none.
        cmd.bindIndexBuffer(resources.vulkanBuffer(resolved.indexBuffer), 0, indexType);

        // Shadow set 0 is pushed inline (core 1.4 push descriptors) — no allocated
        // per-object descriptor set, mirroring the forward pass.
        pushShadowObjectDescriptors(cmd, resources, layout, dc);
        if (pipelineChanged)
        {
            // Bindless materials (set 2) for the masked fragment path. Bound AFTER the push
            // descriptors that establish set 0, matching the ordering the forward pass documents
            // (Vulkan layout compatibility preserves set 0, and this order also avoids a validation
            // -layer first-use state-tracking defect). Bound for the opaque path too: all four
            // shadow pipelines share one layout, so this costs one call per pipeline change and
            // removes any question of which path had the set.
            //
            // The SET itself is the one Resources allocated from the FORWARD pipeline's bindless
            // layout — there is a single global materials set, not one per pipeline. That is legal
            // because the layouts are IDENTICALLY DEFINED (same bindings, same binding flags, from
            // Pipeline::createBindlessDescriptorSetLayout), which Vulkan treats as the same layout
            // for compatibility. Change those bindings and every pipeline that opts in changes with
            // them, which is exactly the coupling we want here.
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 2,
                                   resources.bindlessDescriptorSet(), {});
        }
        // Per draw, because materialIndex varies per draw: the view's constants supply everything
        // else. One struct assembled in one place, so the masked path cannot read a material index
        // that belongs to the previously recorded caster.
        ShadowPushConstants pc = viewConstants;
        pc.materialIndex = dc.materialIndex;
        cmd.pushConstants<ShadowPushConstants>(
            layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.drawIndexed(resolved.indexCount, 1, 0, 0, 0);
        // Recorded HERE, beside the draw itself, so "this family drew this caster" cannot become
        // true for a caster that was only considered.
        lod.noteDrawn(target.group(), dc.shadowRequest);
    }
}

} // namespace

Shadows::Shadows(const Device& device, Resources& resources)
    : resources_{&resources},
      shadowPipeline_(device, Pipeline::shadowConfig()),
      shadowMaskedPipeline_(device, Pipeline::shadowMaskedConfig()),
      selfShadowSecondPipeline_(device, Pipeline::selfShadowSecondConfig()),
      selfShadowSecondMaskedPipeline_(device, Pipeline::selfShadowSecondMaskedConfig())
{
    // Four pipelines for the four shadow material modes the plan names, not eight: the ALPHA half
    // (opaque / masked) needs a different fragment shader and so a different pipeline, while the
    // SIDEDNESS half is dynamic cull state on all of them. The self-shadow FIRST layer needs no
    // pipeline of its own any more — it is the main pair recorded with AllFaces.
    shadowPipelines_ = ShadowPipelinePair{
        .opaque = resources_->registerPipeline(shadowPipeline_.pipeline(),
                                               shadowPipeline_.pipelineLayout()),
        .masked = resources_->registerPipeline(shadowMaskedPipeline_.pipeline(),
                                               shadowMaskedPipeline_.pipelineLayout()),
    };
    selfShadowSecondPipelines_ = ShadowPipelinePair{
        .opaque = resources_->registerPipeline(selfShadowSecondPipeline_.pipeline(),
                                               selfShadowSecondPipeline_.pipelineLayout()),
        .masked = resources_->registerPipeline(selfShadowSecondMaskedPipeline_.pipeline(),
                                               selfShadowSecondMaskedPipeline_.pipelineLayout()),
    };

    // Dynamic rendering needs no framebuffers: recordPass binds each per-layer
    // depth view straight into vk::RenderingInfo (depth-only — no colour
    // attachment). We retain only the texture handles; per-layer views are
    // fetched from Resources at record time. Every shadow depth image is created
    // already transitioned to DepthStencilReadOnlyOptimal across all layers
    // (createShadowMap), the resting layout the forward sampler expects and the
    // per-iteration barriers cycle through.
    shadowMapHandle_ = resources_->createShadowMap(kShadowMapExtent, kShadowCascadeCount);
    worldShadowMapHandle_ = resources_->createShadowMap(kShadowMapExtent, kShadowCascadeCount);

    selfShadowFirstMapHandle_ =
        resources_->createShadowMap(kSkinnedSelfShadowMapExtent, kMaxSkinnedSelfShadowCasters);
    selfShadowMapHandle_ =
        resources_->createShadowMap(kSkinnedSelfShadowMapExtent, kMaxSkinnedSelfShadowCasters);

    // Spot casters share a 2D-array depth image, one layer per caster.
    spotShadowMapHandle_ = resources_->createShadowMap(kSpotShadowMapExtent, kMaxSpotShadowCasters);

    // Point casters: one cubemap-array depth image, kCubeFaceCount faces per caster. Layout
    // `kCubeFaceCount * cube + face` matches Resources::vulkanPointShadowFaceView and the
    // matrixIndex layout in ShadowUBO::lightViewProj.
    pointShadowMapHandle_ =
        resources_->createPointShadowMap(kPointShadowMapExtent, kMaxPointShadowCasters);

    auto& shared = resources_->sharedTextures();
    shared.shadowMap = shadowMapHandle_;
    shared.worldShadowMap = worldShadowMapHandle_;
    shared.selfShadowFirstMap = selfShadowFirstMapHandle_;
    shared.selfShadowMap = selfShadowMapHandle_;
    shared.spotShadowMap = spotShadowMapHandle_;
    shared.pointShadowMap = pointShadowMapHandle_;
    // Debug ShadowDepth view samples the CSM depth map directly (raw depth ==
    // the gl_FragCoord.z the old throwaway colour attachment stored).
    shared.shadowDebugImage = shadowMapHandle_;
}

void Shadows::recordPass(vk::CommandBuffer cmd, std::span<const DrawCommand> shadowDraws,
                         std::span<const DrawCommand> worldOnlyShadowDraws,
                         std::span<const DrawCommand> selfShadowDraws, int activeSelfShadowCasters,
                         int activeSpotCasters, std::span<const PointShadowCaster> pointCasters,
                         const ShadowRenderViewSet& views, ShadowLodResolver& resolver,
                         float lodBudgetTexels, ShadowLodHysteresis hysteresis, bool cullingEnabled,
                         ShadowMapValidity validity, ShadowFrameStats& stats,
                         const GpuProfiler& profiler, uint32_t frameIndex) const
{
    // Nothing to record at all — `--no-shadows`, or a scene with no light any family is fitted to.
    // Returning here is what makes suppression OBSERVABLE: no draws, no clears, no timestamps, so
    // every shadow row in the diagnostics and every shadow group in the GPU timings reads zero.
    if (validity.none())
    {
        return;
    }
    // Bottom-to-bottom group timing: both boundaries are bottom-of-pipe stamps, so adjacent
    // sub-millisecond groups cannot overlap and inflate each other the way top-to-bottom spans do.
    // Every pass in the engine stamps this way now — begin() itself is bottom-of-pipe — so this is
    // no longer a shadow-only convention, just the convention. A group that records nothing leaves
    // its two stamps unwritten and reports 0, rather than an empty span that reads as a small real
    // cost and inflates the frame total. An active family with zero candidate draws is still timed
    // — its clears and layout barriers are real GPU work.
    //
    // `recording` gates the body AND the stamps together, which is the point: a family this frame
    // does not record must draw nothing, clear nothing and time nothing, and one gate is what makes
    // those three the same answer. It comes from `ShadowMapValidity`, the same value the receiver
    // was told, so a skipped family's diagnostics read zero and its shader path reads "fully lit".
    const auto timeGroup = [&](ProfilePass pass, bool recording, auto&& body)
    {
        if (!recording)
        {
            return;
        }
        profiler.begin(cmd, frameIndex, pass);
        body();
        profiler.end(cmd, frameIndex, pass);
    };

    const vk::ClearValue depthClear{.depthStencil =
                                        vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0}};

    // ONE lookup per iteration, from the frame's view set: the matrix that culls, the descriptor
    // that selects, and the identity that keys hysteresis all come from the same entry, so they
    // cannot describe different fits. A null return means the set says this physical view is not
    // active — the iteration is skipped rather than rasterised from a matrix nobody vouched for.
    const auto viewFor = [&](ShadowViewGroup group, std::size_t slot) -> const ShadowRenderView*
    { return views.find(group, slot); };

    // Culling frustum from the view's OWN matrix. Self-shadow layers pass everything through: they
    // are already restricted to one caster by the slot filter, so a frustum test would only repeat
    // it. Disabled culling passes everything through too.
    const auto frustumFor = [&](const ShadowRenderView& view, bool cull) -> std::optional<Frustum>
    {
        if (!cull)
        {
            return std::nullopt;
        }
        return Frustum::fromViewProj(view.viewProj());
    };

    const auto lodContextFor = [&](const ShadowRenderView& view)
    { return ShadowLodContext{view, resolver, lodBudgetTexels, hysteresis}; };

    // Renders one shadow layer with depth-only dynamic rendering. depthLayer is
    // the array-layer subresource the barriers target; depthView is the matching
    // single-layer attachment view. Depth rests in DepthStencilReadOnlyOptimal
    // (the forward-sampler layout) between frames, so we cycle it
    // ReadOnly → Attachment → ReadOnly. No colour attachment: current MoltenVK
    // commits depth-only stores under dynamic rendering.
    auto recordShadowIteration =
        [&](vk::Image depthImage, uint32_t depthLayer, vk::ImageView depthView, uint32_t extent,
            const ShadowPushConstants& pc, std::span<const DrawCommand> draws,
            ShadowDrawFilter filter, const ShadowLodContext& lod, ShadowPipelinePair pipelines,
            ShadowFaceCull cullPolicy, float depthBiasConstant, float depthBiasSlope,
            const ShadowViewTarget& target)
    {
        vk::Viewport vp = makeFullViewport(static_cast<float>(extent), static_cast<float>(extent));
        vk::Rect2D scissor{
            .offset = vk::Offset2D{.x = 0, .y = 0},
            .extent = vk::Extent2D{.width = extent, .height = extent},
        };

        imageLayerBarrier(cmd, depthImage, vk::ImageAspectFlagBits::eDepth, depthLayer,
                          vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal,
                          vk::PipelineStageFlagBits2::eFragmentShader,
                          vk::AccessFlagBits2::eShaderRead,
                          vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                          vk::AccessFlagBits2::eDepthStencilAttachmentWrite);

        vk::RenderingAttachmentInfo depth{
            .imageView = depthView,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = depthClear,
        };
        cmd.beginRendering(makeRenderingInfo(scissor, {}, &depth));
        cmd.setViewport(0, vp);
        cmd.setScissor(0, scissor);
        cmd.setDepthBias(depthBiasConstant, 0.0f, depthBiasSlope);
        // The push constants are pushed PER DRAW inside the bucket (SH-05 added a per-draw
        // materialIndex to the block), so `pc` travels in as this view's part of them rather than
        // being pushed here — one struct, assembled in one place, instead of a view-level push a
        // per-draw push would then have to agree with.
        recordShadowDrawBucket(cmd, draws, *resources_, pipelines, cullPolicy, pc, filter, lod,
                               target);
        cmd.endRendering();

        imageLayerBarrier(cmd, depthImage, vk::ImageAspectFlagBits::eDepth, depthLayer,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal,
                          vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                          vk::PipelineStageFlagBits2::eLateFragmentTests,
                          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                          vk::PipelineStageFlagBits2::eFragmentShader,
                          vk::AccessFlagBits2::eShaderRead);
    };

    // Layered maps (CSM/world/self): per-layer depth attachment view.
    auto layeredIteration = [&](TextureHandle depthHandle, uint32_t layer, uint32_t extent,
                                const ShadowPushConstants& pc, std::span<const DrawCommand> draws,
                                ShadowDrawFilter filter, const ShadowLodContext& lod,
                                ShadowPipelinePair pipelines, ShadowFaceCull cullPolicy,
                                float depthBiasConstant, float depthBiasSlope,
                                const ShadowViewTarget& target)
    {
        recordShadowIteration(resources_->vulkanImage(depthHandle), layer,
                              resources_->vulkanShadowMapLayerView(depthHandle, layer), extent, pc,
                              draws, filter, lod, pipelines, cullPolicy, depthBiasConstant,
                              depthBiasSlope, target);
    };

    // The main CSM and the world-only CSM are recorded as CONTIGUOUS groups rather than interleaved
    // per cascade. Nothing depends on the interleaving (each layer is independently barriered), and
    // grouping them is what lets each family carry one bottom-to-bottom timestamp boundary in the
    // per-group GPU timing — interleaved, the two families' costs could not be separated at all.
    timeGroup(ProfilePass::ShadowCascades, validity.cascades,
              [&]
              {
                  for (uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
                  {
                      const ShadowRenderView* view = viewFor(ShadowViewGroup::Cascade, cascade);
                      if (view == nullptr)
                      {
                          continue;
                      }
                      ShadowPushConstants pc{};
                      pc.matrixIndex = kShadowCascadeMatrixBase + static_cast<int>(cascade);
                      const std::optional<Frustum> frustum = frustumFor(*view, cullingEnabled);
                      const ShadowDrawFilter filter{.frustum = frustum ? &*frustum : nullptr};
                      layeredIteration(
                          shadowMapHandle_, cascade, kShadowMapExtent, pc, shadowDraws, filter,
                          lodContextFor(*view), shadowPipelines_, ShadowFaceCull::PerCaster,
                          kDirectionalShadowRasterBiasConstant, kDirectionalShadowRasterBiasSlope,
                          ShadowViewTarget{stats, ShadowViewGroup::Cascade, cascade, true});
                  }
              });

    // The world-only CSM exists so skinned receivers can sample a cascade without their own
    // geometry; with no skinned draw this frame no world-only view is enabled, so the duplicate
    // 4-cascade render is skipped entirely. Skipping leaves its diagnostic rows untouched, which is
    // the honest report: the views were not rasterised. The receiver is told (the WORLD_ONLY bit)
    // rather than left to rely on nothing sampling it.
    timeGroup(ProfilePass::ShadowWorldOnly, validity.worldOnly,
              [&]
              {
                  for (uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
                  {
                      // The set's world-only entry ALIASES the cascade's, so this iteration
                      // resolves against the same logical view — the resolver returns the cascade's
                      // cached answer, which is what makes the two CSMs agree for a rigid caster
                      // rather than agreeing by coincidence.
                      const ShadowRenderView* view = viewFor(ShadowViewGroup::WorldOnly, cascade);
                      if (view == nullptr)
                      {
                          continue;
                      }
                      ShadowPushConstants pc{};
                      pc.matrixIndex = kShadowCascadeMatrixBase + static_cast<int>(cascade);
                      const std::optional<Frustum> frustum = frustumFor(*view, cullingEnabled);
                      const ShadowDrawFilter filter{.frustum = frustum ? &*frustum : nullptr};
                      layeredIteration(
                          worldShadowMapHandle_, cascade, kShadowMapExtent, pc,
                          worldOnlyShadowDraws, filter, lodContextFor(*view), shadowPipelines_,
                          ShadowFaceCull::PerCaster, kDirectionalShadowRasterBiasConstant,
                          kDirectionalShadowRasterBiasSlope,
                          ShadowViewTarget{stats, ShadowViewGroup::WorldOnly, cascade, true});
                  }
              });

    // Only the densely-assigned slots render; an unassigned slot's layers are
    // never sampled (no fragment carries its index), so they need no clear. An
    // assigned slot whose caster produced no shadow draw still clears here —
    // correctly reading "no occluder" (depth 1.0) for its forward fragments.
    timeGroup(
        ProfilePass::ShadowSelf, validity.self,
        [&]
        {
            for (int slot = 0;
                 slot < activeSelfShadowCasters && slot < kMaxSkinnedSelfShadowCasters; ++slot)
            {
                const auto viewSlot = static_cast<std::size_t>(slot);
                const ShadowRenderView* view = viewFor(ShadowViewGroup::Self, viewSlot);
                if (view == nullptr)
                {
                    continue;
                }
                ShadowPushConstants pc{};
                pc.matrixIndex = -1;
                pc.selfShadowSlot = slot;
                // From the SET, not scanned out of the draw span: the slot's matrix is a property
                // of the view, and searching the commands for it was a second place the same value
                // lived.
                pc.lightViewProj = view->viewProj();
                const ShadowDrawFilter filter{.selfShadowSlot = slot};
                // No frustum: the slot filter already restricts this layer to its one caster.
                const ShadowLodContext lod = lodContextFor(*view);
                // FIRST layer: the main pipeline pair with an all-faces policy — since SH-05 made
                // cull mode dynamic, "capture whatever faces the light sees first" is recorded
                // state rather than a pipeline that differed from the main one in nothing else.
                layeredIteration(selfShadowFirstMapHandle_, static_cast<uint32_t>(slot),
                                 kSkinnedSelfShadowMapExtent, pc, selfShadowDraws, filter, lod,
                                 shadowPipelines_, ShadowFaceCull::AllFaces, 0.0f, 0.0f,
                                 ShadowViewTarget{stats, ShadowViewGroup::Self, viewSlot, true});
                // Second depth layer: same logical view re-rasterised, so it hits the resolver's
                // frame cache — one decision, two layers — and its cost counts while its selection
                // does not (that would double the histogram for one decision).
                layeredIteration(selfShadowMapHandle_, static_cast<uint32_t>(slot),
                                 kSkinnedSelfShadowMapExtent, pc, selfShadowDraws, filter, lod,
                                 selfShadowSecondPipelines_, ShadowFaceCull::BackFacesOnly, 0.0f,
                                 0.0f,
                                 ShadowViewTarget{stats, ShadowViewGroup::Self, viewSlot, false});
            }
        });

    timeGroup(ProfilePass::ShadowSpot, validity.spot,
              [&]
              {
                  for (int s = 0; s < activeSpotCasters && s < kMaxSpotShadowCasters; ++s)
                  {
                      const auto viewSlot = static_cast<std::size_t>(s);
                      const ShadowRenderView* view = viewFor(ShadowViewGroup::Spot, viewSlot);
                      if (view == nullptr)
                      {
                          continue;
                      }
                      ShadowPushConstants pc{};
                      pc.matrixIndex = kShadowSpotMatrixBase + s;
                      const std::optional<Frustum> frustum = frustumFor(*view, cullingEnabled);
                      recordShadowIteration(
                          resources_->vulkanImage(spotShadowMapHandle_), static_cast<uint32_t>(s),
                          resources_->vulkanShadowMapLayerView(spotShadowMapHandle_,
                                                               static_cast<uint32_t>(s)),
                          kSpotShadowMapExtent, pc, shadowDraws,
                          ShadowDrawFilter{.frustum = frustum ? &*frustum : nullptr},
                          lodContextFor(*view), shadowPipelines_, ShadowFaceCull::PerCaster,
                          kPunctualShadowRasterBiasConstant, kPunctualShadowRasterBiasSlope,
                          ShadowViewTarget{stats, ShadowViewGroup::Spot, viewSlot, true});
                  }
              });

    timeGroup(ProfilePass::ShadowPoint, validity.point,
              [&]
              {
                  for (std::size_t p = 0; p < pointCasters.size() &&
                                          p < static_cast<std::size_t>(kMaxPointShadowCasters);
                       ++p)
                  {
                      for (uint32_t face = 0; face < kCubeFaceCount; ++face)
                      {
                          // ONE derivation feeding the matrix slot, the attachment layer and the
                          // diagnostic row — the point family's matrix index, depth layer and view
                          // slot are all 6*p + face.
                          const std::size_t viewSlot = shadowPointViewSlot(p, face);
                          const ShadowRenderView* view = viewFor(ShadowViewGroup::Point, viewSlot);
                          if (view == nullptr)
                          {
                              continue;
                          }
                          ShadowPushConstants pc{};
                          pc.matrixIndex = kShadowPointMatrixBase + static_cast<int>(viewSlot);
                          pc.lightPosRange[0] = pointCasters[p].worldPosition.x();
                          pc.lightPosRange[1] = pointCasters[p].worldPosition.y();
                          pc.lightPosRange[2] = pointCasters[p].worldPosition.z();
                          pc.lightPosRange[3] = pointCasters[p].range;
                          const std::optional<Frustum> frustum = frustumFor(*view, cullingEnabled);
                          recordShadowIteration(
                              resources_->vulkanImage(pointShadowMapHandle_),
                              static_cast<uint32_t>(viewSlot),
                              resources_->vulkanPointShadowFaceView(pointShadowMapHandle_,
                                                                    static_cast<uint32_t>(p), face),
                              kPointShadowMapExtent, pc, shadowDraws,
                              ShadowDrawFilter{.frustum = frustum ? &*frustum : nullptr},
                              lodContextFor(*view), shadowPipelines_, ShadowFaceCull::PerCaster,
                              kPunctualShadowRasterBiasConstant, kPunctualShadowRasterBiasSlope,
                              ShadowViewTarget{stats, ShadowViewGroup::Point, viewSlot, true});
                      }
                  }
              });
}

} // namespace fire_engine
