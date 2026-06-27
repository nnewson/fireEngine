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

void recordShadowDrawBucket(vk::CommandBuffer cmd, std::span<const DrawCommand> shadowDraws,
                            const Resources& resources, PipelineHandle pipelineHandle,
                            ShadowDrawFilter filter = {})
{
    bool pipelineBound = false;
    for (const auto& dc : shadowDraws)
    {
        if (!filter.accepts(dc))
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

        // Shadow set 0 is pushed inline (VK_KHR_push_descriptor) — no allocated
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

    // Point casters: one cubemap-array depth image, six faces per caster. Layout
    // 6 * cube + face matches Resources::vulkanPointShadowFaceView and the
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
                         std::span<const DrawCommand> selfShadowDraws, int activeSpotCasters,
                         std::span<const PointShadowCaster> pointCasters,
                         std::span<const Mat4> shadowViewProjs, bool cullingEnabled) const
{
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
            float depthBiasSlope)
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
        recordShadowDrawBucket(cmd, draws, *resources_, pipelineHandle, filter);
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
                                float depthBiasConstant, float depthBiasSlope)
    {
        recordShadowIteration(resources_->vulkanImage(depthHandle), layer,
                              resources_->vulkanShadowMapLayerView(depthHandle, layer), extent, pc,
                              draws, filter, pipelineHandle, depthBiasConstant, depthBiasSlope);
    };

    for (uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        ShadowPushConstants pc{};
        pc.matrixIndex = kShadowCascadeMatrixBase + static_cast<int>(cascade);
        const std::optional<Frustum> frustum = shadowFrustumFor(pc.matrixIndex);
        const ShadowDrawFilter filter{.frustum = frustum ? &*frustum : nullptr};
        layeredIteration(shadowMapHandle_, cascade, kShadowMapExtent, pc, shadowDraws, filter,
                         shadowPipelineHandle_, kDirectionalShadowRasterBiasConstant,
                         kDirectionalShadowRasterBiasSlope);
        layeredIteration(worldShadowMapHandle_, cascade, kShadowMapExtent, pc, worldOnlyShadowDraws,
                         filter, shadowPipelineHandle_, kDirectionalShadowRasterBiasConstant,
                         kDirectionalShadowRasterBiasSlope);
    }

    for (int slot = 0; slot < kMaxSkinnedSelfShadowCasters; ++slot)
    {
        ShadowPushConstants pc{};
        pc.matrixIndex = -1;
        pc.selfShadowSlot = slot;
        pc.lightViewProj = findSelfShadowViewProj(selfShadowDraws, slot);
        const ShadowDrawFilter filter{.selfShadowSlot = slot};
        layeredIteration(selfShadowFirstMapHandle_, static_cast<uint32_t>(slot),
                         kSkinnedSelfShadowMapExtent, pc, selfShadowDraws, filter,
                         selfShadowFirstPipelineHandle_, 0.0f, 0.0f);
        layeredIteration(selfShadowMapHandle_, static_cast<uint32_t>(slot),
                         kSkinnedSelfShadowMapExtent, pc, selfShadowDraws, filter,
                         selfShadowSecondPipelineHandle_, 0.0f, 0.0f);
    }

    for (int s = 0; s < activeSpotCasters && s < kMaxSpotShadowCasters; ++s)
    {
        ShadowPushConstants pc{};
        pc.matrixIndex = kShadowSpotMatrixBase + s;
        const std::optional<Frustum> frustum = shadowFrustumFor(pc.matrixIndex);
        recordShadowIteration(
            resources_->vulkanImage(spotShadowMapHandle_), static_cast<uint32_t>(s),
            resources_->vulkanShadowMapLayerView(spotShadowMapHandle_, static_cast<uint32_t>(s)),
            kSpotShadowMapExtent, pc, shadowDraws,
            ShadowDrawFilter{.frustum = frustum ? &*frustum : nullptr}, shadowPipelineHandle_,
            kPunctualShadowRasterBiasConstant, kPunctualShadowRasterBiasSlope);
    }

    for (std::size_t p = 0;
         p < pointCasters.size() && p < static_cast<std::size_t>(kMaxPointShadowCasters); ++p)
    {
        for (uint32_t face = 0; face < 6; ++face)
        {
            ShadowPushConstants pc{};
            pc.matrixIndex =
                kShadowPointMatrixBase + 6 * static_cast<int>(p) + static_cast<int>(face);
            pc.lightPosRange[0] = pointCasters[p].worldPosition.x();
            pc.lightPosRange[1] = pointCasters[p].worldPosition.y();
            pc.lightPosRange[2] = pointCasters[p].worldPosition.z();
            pc.lightPosRange[3] = pointCasters[p].range;
            const std::optional<Frustum> frustum = shadowFrustumFor(pc.matrixIndex);
            recordShadowIteration(
                resources_->vulkanImage(pointShadowMapHandle_), static_cast<uint32_t>(6 * p + face),
                resources_->vulkanPointShadowFaceView(pointShadowMapHandle_,
                                                      static_cast<uint32_t>(p), face),
                kPointShadowMapExtent, pc, shadowDraws,
                ShadowDrawFilter{.frustum = frustum ? &*frustum : nullptr}, shadowPipelineHandle_,
                kPunctualShadowRasterBiasConstant, kPunctualShadowRasterBiasSlope);
        }
    }
}

} // namespace fire_engine
