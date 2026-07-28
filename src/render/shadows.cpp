#include <fire_engine/render/shadows.hpp>

#include <optional>

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
        return frustum == nullptr || frustum->intersects(dc.shadowBounds);
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

void recordShadowDrawBucket(vk::CommandBuffer cmd, std::span<const DrawCommand> shadowDraws,
                            const Resources& resources, PipelineHandle pipelineHandle,
                            ShadowDrawFilter filter, const ShadowViewTarget& target)
{
    // Before walking the span, so a view that renders and clears with nothing to draw still
    // reports as rasterised — that empty-but-rendered view is itself a finding.
    ShadowViewStats& viewStats = target.view();
    viewStats.beginRasterPass();

    bool pipelineBound = false;
    for (const auto& dc : shadowDraws)
    {
        // One observation per walked command, carrying the filter verdict that is about to decide
        // the draw — so `candidate - drawn` is exactly this view's filter yield and can never be
        // reconstructed differently from what was recorded.
        const bool accepted = filter.accepts(dc);
        viewStats.observe(dc.indexCount / 3, accepted, dc.lodLevel, target.countSelection());
        if (!accepted)
        {
            continue;
        }
        if (!pipelineBound)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             resources.vulkanPipeline(pipelineHandle));
            pipelineBound = true;
        }
        if (dc.vertexBuffer != NullBuffer)
        {
            cmd.bindVertexBuffers(0, resources.vulkanBuffer(dc.vertexBuffer), {vk::DeviceSize{0}});
        }

        vk::IndexType indexType =
            dc.indexType == DrawIndexType::UInt32 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
        cmd.bindIndexBuffer(resources.vulkanBuffer(dc.indexBuffer), 0, indexType);

        // Shadow set 0 is pushed inline (core 1.4 push descriptors) — no allocated
        // per-object descriptor set, mirroring the forward pass.
        pushShadowObjectDescriptors(cmd, resources, resources.vulkanPipelineLayout(pipelineHandle),
                                    dc);
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
}

} // namespace

Shadows::Shadows(const Device& device, Resources& resources)
    : resources_{&resources},
      shadowPipeline_(device, Pipeline::shadowConfig()),
      selfShadowFirstPipeline_(device, Pipeline::selfShadowFirstConfig()),
      selfShadowSecondPipeline_(device, Pipeline::selfShadowSecondConfig())
{
    shadowPipelineHandle_ =
        resources_->registerPipeline(shadowPipeline_.pipeline(), shadowPipeline_.pipelineLayout());
    selfShadowFirstPipelineHandle_ = resources_->registerPipeline(
        selfShadowFirstPipeline_.pipeline(), selfShadowFirstPipeline_.pipelineLayout());
    selfShadowSecondPipelineHandle_ = resources_->registerPipeline(
        selfShadowSecondPipeline_.pipeline(), selfShadowSecondPipeline_.pipelineLayout());

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
                         std::span<const Mat4> shadowViewProjs, bool cullingEnabled,
                         bool renderWorldShadow, ShadowFrameStats& stats,
                         const GpuProfiler& profiler, uint32_t frameIndex) const
{
    // Bottom-to-bottom group timing (the VDPM sub-stage pattern): both boundaries are
    // bottom-of-pipe stamps, so adjacent sub-millisecond groups cannot overlap and inflate each
    // other the way repeated top-to-bottom spans do. A group that records nothing leaves its two
    // stamps unwritten and reports 0.
    // `active` gates the STAMPS, not the body: a family that renders nothing this frame must leave
    // its two timestamps unwritten so the pass reports 0, rather than recording an empty span that
    // reads as a small real cost and inflates the frame total. An active family with zero candidate
    // draws is still timed — its clears and layout barriers are real GPU work.
    const auto timeGroup = [&](ProfilePass pass, bool active, auto&& body)
    {
        if (!active)
        {
            body(); // no-op for an inactive family, but keeps the control flow in one place
            return;
        }
        profiler.stampBottom(cmd, frameIndex, pass, false);
        body();
        profiler.stampBottom(cmd, frameIndex, pass, true);
    };

    const vk::ClearValue depthClear{.depthStencil =
                                        vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0}};

    // Drop casters whose world bounds fall outside the light/cascade frustum at
    // `matrixIndex`. A self-shadow slot (matrixIndex < 0), disabled culling, or
    // invalid matrix index passes everything through.
    const auto shadowFrustumFor = [&](int matrixIndex) -> std::optional<Frustum>
    {
        if (!cullingEnabled || matrixIndex < 0 ||
            matrixIndex >= static_cast<int>(shadowViewProjs.size()))
        {
            return std::nullopt;
        }
        return Frustum::fromViewProj(shadowViewProjs[static_cast<std::size_t>(matrixIndex)]);
    };

    const auto findSelfShadowViewProj = [](std::span<const DrawCommand> draws, int slot) -> Mat4
    {
        for (const DrawCommand& dc : draws)
        {
            if (dc.selfShadowSlot == slot)
            {
                return dc.selfShadowViewProj;
            }
        }
        return Mat4::identity();
    };

    // Renders one shadow layer with depth-only dynamic rendering. depthLayer is
    // the array-layer subresource the barriers target; depthView is the matching
    // single-layer attachment view. Depth rests in DepthStencilReadOnlyOptimal
    // (the forward-sampler layout) between frames, so we cycle it
    // ReadOnly → Attachment → ReadOnly. No colour attachment: current MoltenVK
    // commits depth-only stores under dynamic rendering.
    auto recordShadowIteration =
        [&](vk::Image depthImage, uint32_t depthLayer, vk::ImageView depthView, uint32_t extent,
            const ShadowPushConstants& pc, std::span<const DrawCommand> draws,
            ShadowDrawFilter filter, PipelineHandle pipelineHandle, float depthBiasConstant,
            float depthBiasSlope, const ShadowViewTarget& target)
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
        const vk::PipelineLayout shadowPipelineLayout =
            resources_->vulkanPipelineLayout(pipelineHandle);
        cmd.pushConstants<ShadowPushConstants>(
            shadowPipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        recordShadowDrawBucket(cmd, draws, *resources_, pipelineHandle, filter, target);
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
                                ShadowDrawFilter filter, PipelineHandle pipelineHandle,
                                float depthBiasConstant, float depthBiasSlope,
                                const ShadowViewTarget& target)
    {
        recordShadowIteration(resources_->vulkanImage(depthHandle), layer,
                              resources_->vulkanShadowMapLayerView(depthHandle, layer), extent, pc,
                              draws, filter, pipelineHandle, depthBiasConstant, depthBiasSlope,
                              target);
    };

    // The main CSM and the world-only CSM are recorded as CONTIGUOUS groups rather than interleaved
    // per cascade. Nothing depends on the interleaving (each layer is independently barriered), and
    // grouping them is what lets each family carry one bottom-to-bottom timestamp boundary in the
    // per-group GPU timing — interleaved, the two families' costs could not be separated at all.
    timeGroup(ProfilePass::ShadowCascades, /*active=*/true,
              [&]
              {
                  for (uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
                  {
                      ShadowPushConstants pc{};
                      pc.matrixIndex = kShadowCascadeMatrixBase + static_cast<int>(cascade);
                      const std::optional<Frustum> frustum = shadowFrustumFor(pc.matrixIndex);
                      const ShadowDrawFilter filter{.frustum = frustum ? &*frustum : nullptr};
                      layeredIteration(
                          shadowMapHandle_, cascade, kShadowMapExtent, pc, shadowDraws, filter,
                          shadowPipelineHandle_, kDirectionalShadowRasterBiasConstant,
                          kDirectionalShadowRasterBiasSlope,
                          ShadowViewTarget{stats, ShadowViewGroup::Cascade, cascade, true});
                  }
              });

    // The world-only CSM exists so skinned receivers can sample a cascade without their own
    // geometry; with no skinned draw this frame nothing samples it, so skip the duplicate
    // 4-cascade render entirely (stale content is unreachable — see the recordPass contract in
    // shadows.hpp). Skipping leaves its diagnostic rows untouched, which is the honest report: the
    // views were not rasterised.
    timeGroup(ProfilePass::ShadowWorldOnly, renderWorldShadow,
              [&]
              {
                  if (renderWorldShadow)
                  {
                      for (uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
                      {
                          ShadowPushConstants pc{};
                          pc.matrixIndex = kShadowCascadeMatrixBase + static_cast<int>(cascade);
                          const std::optional<Frustum> frustum = shadowFrustumFor(pc.matrixIndex);
                          const ShadowDrawFilter filter{.frustum = frustum ? &*frustum : nullptr};
                          layeredIteration(
                              worldShadowMapHandle_, cascade, kShadowMapExtent, pc,
                              worldOnlyShadowDraws, filter, shadowPipelineHandle_,
                              kDirectionalShadowRasterBiasConstant,
                              kDirectionalShadowRasterBiasSlope,
                              ShadowViewTarget{stats, ShadowViewGroup::WorldOnly, cascade, true});
                      }
                  }
              });

    // Only the densely-assigned slots render; an unassigned slot's layers are
    // never sampled (no fragment carries its index), so they need no clear. An
    // assigned slot whose caster produced no shadow draw still clears here —
    // correctly reading "no occluder" (depth 1.0) for its forward fragments.
    timeGroup(
        ProfilePass::ShadowSelf, activeSelfShadowCasters > 0,
        [&]
        {
            for (int slot = 0;
                 slot < activeSelfShadowCasters && slot < kMaxSkinnedSelfShadowCasters; ++slot)
            {
                ShadowPushConstants pc{};
                pc.matrixIndex = -1;
                pc.selfShadowSlot = slot;
                pc.lightViewProj = findSelfShadowViewProj(selfShadowDraws, slot);
                const ShadowDrawFilter filter{.selfShadowSlot = slot};
                const auto viewSlot = static_cast<std::size_t>(slot);
                layeredIteration(selfShadowFirstMapHandle_, static_cast<uint32_t>(slot),
                                 kSkinnedSelfShadowMapExtent, pc, selfShadowDraws, filter,
                                 selfShadowFirstPipelineHandle_, 0.0f, 0.0f,
                                 ShadowViewTarget{stats, ShadowViewGroup::Self, viewSlot, true});
                // Second depth layer: same logical view re-rasterised, so cost counts but selection
                // does not (it would double the histogram for one decision).
                layeredIteration(selfShadowMapHandle_, static_cast<uint32_t>(slot),
                                 kSkinnedSelfShadowMapExtent, pc, selfShadowDraws, filter,
                                 selfShadowSecondPipelineHandle_, 0.0f, 0.0f,
                                 ShadowViewTarget{stats, ShadowViewGroup::Self, viewSlot, false});
            }
        });

    timeGroup(ProfilePass::ShadowSpot, activeSpotCasters > 0,
              [&]
              {
                  for (int s = 0; s < activeSpotCasters && s < kMaxSpotShadowCasters; ++s)
                  {
                      ShadowPushConstants pc{};
                      pc.matrixIndex = kShadowSpotMatrixBase + s;
                      const std::optional<Frustum> frustum = shadowFrustumFor(pc.matrixIndex);
                      recordShadowIteration(
                          resources_->vulkanImage(spotShadowMapHandle_), static_cast<uint32_t>(s),
                          resources_->vulkanShadowMapLayerView(spotShadowMapHandle_,
                                                               static_cast<uint32_t>(s)),
                          kSpotShadowMapExtent, pc, shadowDraws,
                          ShadowDrawFilter{.frustum = frustum ? &*frustum : nullptr},
                          shadowPipelineHandle_, kPunctualShadowRasterBiasConstant,
                          kPunctualShadowRasterBiasSlope,
                          ShadowViewTarget{stats, ShadowViewGroup::Spot,
                                           static_cast<std::size_t>(s), true});
                  }
              });

    timeGroup(ProfilePass::ShadowPoint, !pointCasters.empty(),
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
                          ShadowPushConstants pc{};
                          pc.matrixIndex = kShadowPointMatrixBase + static_cast<int>(viewSlot);
                          pc.lightPosRange[0] = pointCasters[p].worldPosition.x();
                          pc.lightPosRange[1] = pointCasters[p].worldPosition.y();
                          pc.lightPosRange[2] = pointCasters[p].worldPosition.z();
                          pc.lightPosRange[3] = pointCasters[p].range;
                          const std::optional<Frustum> frustum = shadowFrustumFor(pc.matrixIndex);
                          recordShadowIteration(
                              resources_->vulkanImage(pointShadowMapHandle_),
                              static_cast<uint32_t>(viewSlot),
                              resources_->vulkanPointShadowFaceView(pointShadowMapHandle_,
                                                                    static_cast<uint32_t>(p), face),
                              kPointShadowMapExtent, pc, shadowDraws,
                              ShadowDrawFilter{.frustum = frustum ? &*frustum : nullptr},
                              shadowPipelineHandle_, kPunctualShadowRasterBiasConstant,
                              kPunctualShadowRasterBiasSlope,
                              ShadowViewTarget{stats, ShadowViewGroup::Point, viewSlot, true});
                      }
                  }
              });
}

} // namespace fire_engine
