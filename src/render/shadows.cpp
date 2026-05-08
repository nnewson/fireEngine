#include <fire_engine/render/shadows.hpp>

#include <array>
#include <limits>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

void recordShadowDrawBucket(vk::CommandBuffer cmd, const std::vector<DrawCommand>& shadowDraws,
                            const Resources& resources)
{
    auto lastBoundPipeline = PipelineHandle{std::numeric_limits<uint32_t>::max()};
    for (const auto& dc : shadowDraws)
    {
        if (dc.pipeline != lastBoundPipeline)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             resources.vulkanPipeline(dc.pipeline));
            lastBoundPipeline = dc.pipeline;
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
                               resources.vulkanPipelineLayout(dc.pipeline), 0, ds, {});
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
}

} // namespace

Shadows::Shadows(const Device& device, Resources& resources)
    : device_{&device},
      resources_{&resources},
      shadowPass_(RenderPass::createShadow(device)),
      shadowPipeline_(device, Pipeline::shadowConfig(shadowPass_.renderPass())),
      spotShadowPass_(RenderPass::createShadow(device)),
      pointShadowPass_(RenderPass::createShadow(device))
{
    shadowPipelineHandle_ =
        resources_->registerPipeline(shadowPipeline_.pipeline(), shadowPipeline_.pipelineLayout());
    shadowMapHandle_ = resources_->createShadowMap(shadowMapExtent, shadowCascadeCount);
    shadowColourHandle_ =
        resources_->createShadowColourAttachment(shadowMapExtent, shadowCascadeCount, true);

    std::vector<vk::ImageView> cascadeDepthViews;
    std::vector<vk::ImageView> cascadeColourViews;
    cascadeDepthViews.reserve(shadowCascadeCount);
    cascadeColourViews.reserve(shadowCascadeCount);
    for (uint32_t layer = 0; layer < shadowCascadeCount; ++layer)
    {
        cascadeDepthViews.push_back(resources_->vulkanShadowMapLayerView(shadowMapHandle_, layer));
        cascadeColourViews.push_back(resources_->vulkanShadowColourLayerView(shadowColourHandle_, layer));
    }

    shadowPass_.createShadowFramebuffer(*device_, cascadeColourViews, cascadeDepthViews,
                                        shadowMapExtent);

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
    resources_->spotShadowMap(spotShadowMapHandle_);
    resources_->pointShadowMap(pointShadowMapHandle_);
    resources_->shadowDebugImage(shadowColourHandle_);
}

void Shadows::recordPass(vk::CommandBuffer cmd, const std::vector<DrawCommand>& shadowDraws,
                         int activeSpotCasters,
                         std::span<const PointShadowCaster> pointCasters) const
{
    std::array<vk::ClearValue, 2> shadowClears = {
        vk::ClearValue{.color = vk::ClearColorValue{.float32 = {{0.0f, 0.0f, 0.0f, 1.0f}}}},
        vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0}},
    };

    const vk::PipelineLayout shadowPipelineLayout =
        resources_->vulkanPipelineLayout(shadowPipelineHandle_);

    auto recordShadowIteration = [&](vk::RenderPass renderPass, vk::Framebuffer framebuffer,
                                     uint32_t extent, const ShadowPushConstants& pc,
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
        cmd.pushConstants<ShadowPushConstants>(
            shadowPipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        recordShadowDrawBucket(cmd, shadowDraws, *resources_);
        cmd.endRenderPass();
    };

    for (uint32_t cascade = 0; cascade < shadowCascadeCount; ++cascade)
    {
        ShadowPushConstants pc{};
        pc.matrixIndex = SHADOW_CASCADE_MATRIX_BASE + static_cast<int>(cascade);
        recordShadowIteration(shadowPass_.renderPass(), shadowPass_.framebuffer(cascade),
                              shadowMapExtent, pc, directionalShadowRasterBiasConstant,
                              directionalShadowRasterBiasSlope);
    }

    for (int s = 0; s < activeSpotCasters && s < MAX_SPOT_SHADOW_CASTERS; ++s)
    {
        ShadowPushConstants pc{};
        pc.matrixIndex = SHADOW_SPOT_MATRIX_BASE + s;
        recordShadowIteration(spotShadowPass_.renderPass(),
                              spotShadowPass_.framebuffer(static_cast<std::size_t>(s)),
                              spotShadowMapExtent, pc, punctualShadowRasterBiasConstant,
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
                                  pointShadowMapExtent, pc, punctualShadowRasterBiasConstant,
                                  punctualShadowRasterBiasSlope);
        }
    }
}

} // namespace fire_engine
