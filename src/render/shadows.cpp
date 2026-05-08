#include <fire_engine/render/shadows.hpp>

#include <array>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

void recordShadowDrawBucket(vk::CommandBuffer cmd, const std::vector<DrawCommand>& shadowDraws,
                            const Resources& resources, PipelineHandle pipelineHandle)
{
    bool pipelineBound = false;
    for (const auto& dc : shadowDraws)
    {
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

        vk::DescriptorSet ds = resources.vulkanDescriptorSet(dc.descriptorSet);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               resources.vulkanPipelineLayout(pipelineHandle), 0, ds, {});
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
}

void createLayeredShadowFramebuffer(const Device& device, Resources& resources, RenderPass& pass,
                                    TextureHandle depthHandle, TextureHandle colourHandle,
                                    uint32_t extent, uint32_t layerCount)
{
    std::vector<vk::ImageView> depthViews;
    std::vector<vk::ImageView> colourViews;
    depthViews.reserve(layerCount);
    colourViews.reserve(layerCount);
    for (uint32_t layer = 0; layer < layerCount; ++layer)
    {
        depthViews.push_back(resources.vulkanShadowMapLayerView(depthHandle, layer));
        colourViews.push_back(resources.vulkanShadowColourLayerView(colourHandle, layer));
    }
    pass.createShadowFramebuffer(device, colourViews, depthViews, extent);
}

} // namespace

Shadows::Shadows(const Device& device, Resources& resources)
    : device_{&device},
      resources_{&resources},
      shadowPass_(RenderPass::createShadow(device)),
      shadowPipeline_(device, Pipeline::shadowConfig(shadowPass_.renderPass())),
      selfShadowFirstPipeline_(device, Pipeline::selfShadowFirstConfig(shadowPass_.renderPass())),
      selfShadowSecondPipeline_(device, Pipeline::selfShadowSecondConfig(shadowPass_.renderPass())),
      spotShadowPass_(RenderPass::createShadow(device)),
      pointShadowPass_(RenderPass::createShadow(device)),
      worldShadowPass_(RenderPass::createShadow(device)),
      selfShadowFirstPass_(RenderPass::createShadow(device)),
      selfShadowSecondPass_(RenderPass::createShadow(device))
{
    shadowPipelineHandle_ =
        resources_->registerPipeline(shadowPipeline_.pipeline(), shadowPipeline_.pipelineLayout());
    selfShadowFirstPipelineHandle_ = resources_->registerPipeline(
        selfShadowFirstPipeline_.pipeline(), selfShadowFirstPipeline_.pipelineLayout());
    selfShadowSecondPipelineHandle_ = resources_->registerPipeline(
        selfShadowSecondPipeline_.pipeline(), selfShadowSecondPipeline_.pipelineLayout());
    shadowMapHandle_ = resources_->createShadowMap(shadowMapExtent, shadowCascadeCount);
    shadowColourHandle_ =
        resources_->createShadowColourAttachment(shadowMapExtent, shadowCascadeCount, true);
    createLayeredShadowFramebuffer(*device_, *resources_, shadowPass_, shadowMapHandle_,
                                   shadowColourHandle_, shadowMapExtent, shadowCascadeCount);

    worldShadowMapHandle_ = resources_->createShadowMap(shadowMapExtent, shadowCascadeCount);
    worldShadowColourHandle_ =
        resources_->createShadowColourAttachment(shadowMapExtent, shadowCascadeCount, false);
    createLayeredShadowFramebuffer(*device_, *resources_, worldShadowPass_,
                                   worldShadowMapHandle_, worldShadowColourHandle_,
                                   shadowMapExtent, shadowCascadeCount);

    selfShadowFirstMapHandle_ =
        resources_->createShadowMap(skinnedSelfShadowMapExtent, MAX_SKINNED_SELF_SHADOW_CASTERS);
    selfShadowFirstColourHandle_ = resources_->createShadowColourAttachment(
        skinnedSelfShadowMapExtent, MAX_SKINNED_SELF_SHADOW_CASTERS, false);
    createLayeredShadowFramebuffer(*device_, *resources_, selfShadowFirstPass_,
                                   selfShadowFirstMapHandle_, selfShadowFirstColourHandle_,
                                   skinnedSelfShadowMapExtent, MAX_SKINNED_SELF_SHADOW_CASTERS);

    selfShadowMapHandle_ =
        resources_->createShadowMap(skinnedSelfShadowMapExtent, MAX_SKINNED_SELF_SHADOW_CASTERS);
    selfShadowColourHandle_ = resources_->createShadowColourAttachment(
        skinnedSelfShadowMapExtent, MAX_SKINNED_SELF_SHADOW_CASTERS, false);
    createLayeredShadowFramebuffer(*device_, *resources_, selfShadowSecondPass_,
                                   selfShadowMapHandle_, selfShadowColourHandle_,
                                   skinnedSelfShadowMapExtent, MAX_SKINNED_SELF_SHADOW_CASTERS);

    // Spot casters share a 2D-array depth image, one layer per caster.
    spotShadowMapHandle_ =
        resources_->createShadowMap(spotShadowMapExtent, MAX_SPOT_SHADOW_CASTERS);
    spotShadowColourHandle_ = resources_->createShadowColourAttachment(spotShadowMapExtent);
    std::vector<vk::ImageView> spotDepthViews;
    spotDepthViews.reserve(MAX_SPOT_SHADOW_CASTERS);
    for (uint32_t i = 0; i < MAX_SPOT_SHADOW_CASTERS; ++i)
    {
        spotDepthViews.push_back(resources_->vulkanShadowMapLayerView(spotShadowMapHandle_, i));
    }
    std::array<vk::ImageView, 1> spotColourViews = {resources_->vulkanImageView(spotShadowColourHandle_)};
    spotShadowPass_.createShadowFramebuffer(*device_, spotColourViews, spotDepthViews,
                                            spotShadowMapExtent);

    // Point casters: one cubemap-array depth image, six face framebuffers per
    // caster. Layout 6 * cube + face matches Resources::vulkanPointShadowFaceView
    // and the matrixIndex layout in ShadowUBO::lightViewProj.
    pointShadowMapHandle_ =
        resources_->createPointShadowMap(pointShadowMapExtent, MAX_POINT_SHADOW_CASTERS);
    pointShadowColourHandle_ = resources_->createShadowColourAttachment(pointShadowMapExtent);
    std::vector<vk::ImageView> pointFaceViews;
    pointFaceViews.reserve(static_cast<std::size_t>(MAX_POINT_SHADOW_CASTERS) * 6);
    for (uint32_t cube = 0; cube < MAX_POINT_SHADOW_CASTERS; ++cube)
    {
        for (uint32_t face = 0; face < 6; ++face)
        {
            pointFaceViews.push_back(
                resources_->vulkanPointShadowFaceView(pointShadowMapHandle_, cube, face));
        }
    }
    std::array<vk::ImageView, 1> pointColourViews = {resources_->vulkanImageView(pointShadowColourHandle_)};
    pointShadowPass_.createShadowFramebuffer(*device_, pointColourViews, pointFaceViews,
                                             pointShadowMapExtent);

    resources_->descriptors().shadowDescriptorSetLayout(shadowPipeline_.descriptorSetLayout());
    resources_->shadowMap(shadowMapHandle_);
    resources_->worldShadowMap(worldShadowMapHandle_);
    resources_->selfShadowFirstMap(selfShadowFirstMapHandle_);
    resources_->selfShadowMap(selfShadowMapHandle_);
    resources_->spotShadowMap(spotShadowMapHandle_);
    resources_->pointShadowMap(pointShadowMapHandle_);
    resources_->shadowDebugImage(shadowColourHandle_);
}

void Shadows::recordPass(vk::CommandBuffer cmd, const std::vector<DrawCommand>& shadowDraws,
                         const std::vector<DrawCommand>& worldOnlyShadowDraws,
                         const std::vector<DrawCommand>& selfShadowDraws,
                         int activeSpotCasters,
                         std::span<const PointShadowCaster> pointCasters) const
{
    std::array<vk::ClearValue, 2> shadowClears = {
        vk::ClearValue{.color = vk::ClearColorValue{.float32 = {{0.0f, 0.0f, 0.0f, 1.0f}}}},
        vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0}},
    };

    auto recordShadowIteration = [&](vk::RenderPass renderPass, vk::Framebuffer framebuffer,
                                     uint32_t extent, const ShadowPushConstants& pc,
                                     const std::vector<DrawCommand>& draws,
                                     PipelineHandle pipelineHandle,
                                     float depthBiasConstant, float depthBiasSlope)
    {
        vk::Viewport vp{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(extent),
            .height = static_cast<float>(extent),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vk::Rect2D scissor{
            .offset = vk::Offset2D{.x = 0, .y = 0},
            .extent = vk::Extent2D{.width = extent, .height = extent},
        };
        vk::RenderPassBeginInfo begin{
            .renderPass = renderPass,
            .framebuffer = framebuffer,
            .renderArea = scissor,
            .clearValueCount = static_cast<uint32_t>(shadowClears.size()),
            .pClearValues = shadowClears.data(),
        };
        cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
        cmd.setViewport(0, vp);
        cmd.setScissor(0, scissor);
        cmd.setDepthBias(depthBiasConstant, 0.0f, depthBiasSlope);
        const vk::PipelineLayout shadowPipelineLayout =
            resources_->vulkanPipelineLayout(pipelineHandle);
        cmd.pushConstants<ShadowPushConstants>(
            shadowPipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        recordShadowDrawBucket(cmd, draws, *resources_, pipelineHandle);
        cmd.endRenderPass();
    };

    for (uint32_t cascade = 0; cascade < shadowCascadeCount; ++cascade)
    {
        ShadowPushConstants pc{};
        pc.matrixIndex = SHADOW_CASCADE_MATRIX_BASE + static_cast<int>(cascade);
        recordShadowIteration(shadowPass_.renderPass(), shadowPass_.framebuffer(cascade),
                              shadowMapExtent, pc, shadowDraws, shadowPipelineHandle_,
                              directionalShadowRasterBiasConstant,
                              directionalShadowRasterBiasSlope);
        recordShadowIteration(worldShadowPass_.renderPass(), worldShadowPass_.framebuffer(cascade),
                              shadowMapExtent, pc, worldOnlyShadowDraws, shadowPipelineHandle_,
                              directionalShadowRasterBiasConstant,
                              directionalShadowRasterBiasSlope);
    }

    std::array<std::vector<DrawCommand>, MAX_SKINNED_SELF_SHADOW_CASTERS> selfShadowSlotDraws;
    for (const auto& dc : selfShadowDraws)
    {
        if (dc.selfShadowSlot >= 0 && dc.selfShadowSlot < MAX_SKINNED_SELF_SHADOW_CASTERS)
        {
            selfShadowSlotDraws[static_cast<std::size_t>(dc.selfShadowSlot)].push_back(dc);
        }
    }

    for (int slot = 0; slot < MAX_SKINNED_SELF_SHADOW_CASTERS; ++slot)
    {
        const std::vector<DrawCommand>& slotDraws =
            selfShadowSlotDraws[static_cast<std::size_t>(slot)];
        ShadowPushConstants pc{};
        pc.matrixIndex = -1;
        pc.selfShadowSlot = slot;
        if (!slotDraws.empty())
        {
            pc.lightViewProj = slotDraws.front().selfShadowViewProj;
        }
        recordShadowIteration(selfShadowFirstPass_.renderPass(),
                              selfShadowFirstPass_.framebuffer(static_cast<std::size_t>(slot)),
                              skinnedSelfShadowMapExtent, pc, slotDraws,
                              selfShadowFirstPipelineHandle_, 0.0f, 0.0f);
        recordShadowIteration(selfShadowSecondPass_.renderPass(),
                              selfShadowSecondPass_.framebuffer(static_cast<std::size_t>(slot)),
                              skinnedSelfShadowMapExtent, pc, slotDraws,
                              selfShadowSecondPipelineHandle_, 0.0f, 0.0f);
    }

    for (int s = 0; s < activeSpotCasters && s < MAX_SPOT_SHADOW_CASTERS; ++s)
    {
        ShadowPushConstants pc{};
        pc.matrixIndex = SHADOW_SPOT_MATRIX_BASE + s;
        recordShadowIteration(spotShadowPass_.renderPass(),
                              spotShadowPass_.framebuffer(static_cast<std::size_t>(s)),
                              spotShadowMapExtent, pc, shadowDraws, shadowPipelineHandle_,
                              punctualShadowRasterBiasConstant,
                              punctualShadowRasterBiasSlope);
    }

    for (std::size_t p = 0;
         p < pointCasters.size() && p < static_cast<std::size_t>(MAX_POINT_SHADOW_CASTERS); ++p)
    {
        for (uint32_t face = 0; face < 6; ++face)
        {
            ShadowPushConstants pc{};
            pc.matrixIndex =
                SHADOW_POINT_MATRIX_BASE + 6 * static_cast<int>(p) + static_cast<int>(face);
            pc.lightPosRange[0] = pointCasters[p].worldPosition.x();
            pc.lightPosRange[1] = pointCasters[p].worldPosition.y();
            pc.lightPosRange[2] = pointCasters[p].worldPosition.z();
            pc.lightPosRange[3] = pointCasters[p].range;
            recordShadowIteration(pointShadowPass_.renderPass(),
                                  pointShadowPass_.framebuffer(6 * p + face),
                                  pointShadowMapExtent, pc, shadowDraws, shadowPipelineHandle_,
                                  punctualShadowRasterBiasConstant,
                                  punctualShadowRasterBiasSlope);
        }
    }
}

} // namespace fire_engine
