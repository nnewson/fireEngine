#include <fire_engine/render/post_processing.hpp>

#include <algorithm>

#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

PostProcessing::PostProcessing(const Device& device, const Swapchain& swapchain,
                               Resources& resources)
    : device_{&device},
      swapchain_{&swapchain},
      resources_{&resources},
      postProcessPass_(RenderPass::createPostProcess(device, swapchain)),
      bloomDownPass_(RenderPass::createBloomDown(device, vk::Format::eR16G16B16A16Sfloat)),
      bloomUpPass_(RenderPass::createBloomUp(device, vk::Format::eR16G16B16A16Sfloat)),
      postProcessPipeline_(device, Pipeline::postProcessConfig(postProcessPass_.renderPass())),
      bloomDownsamplePipeline_(device,
                               Pipeline::bloomDownsampleConfig(bloomDownPass_.renderPass())),
      bloomUpsamplePipeline_(device, Pipeline::bloomUpsampleConfig(bloomUpPass_.renderPass()))
{
    offscreenColourHandle_ = resources_->createOffscreenColourTarget(swapchain_->extent());
    postProcessPass_.createPostProcessFramebuffers(*device_, *swapchain_);
    buildBloomResources();
    std::array<uint16_t, 3> postProcessIndices{0, 1, 2};
    postProcessIndexBuffer_ = resources_->createIndexBuffer(postProcessIndices);
    postProcessDescSets_ = resources_->descriptors().createPostProcessDescriptors(
        postProcessPipeline_.descriptorSetLayout(), offscreenColourHandle_, bloomChainHandle_);
}

void PostProcessing::buildBloomResources()
{
    auto extent = swapchain_->extent();
    uint32_t bloomWidth = std::max(1u, extent.width / 2);
    uint32_t bloomHeight = std::max(1u, extent.height / 2);

    bloomChainHandle_ = resources_->createBloomChain(bloomWidth, bloomHeight, kBloomMipCount);

    std::vector<vk::ImageView> downViews;
    std::vector<vk::Extent2D> downExtents;
    downViews.reserve(kBloomMipCount);
    downExtents.reserve(kBloomMipCount);
    for (uint32_t m = 0; m < kBloomMipCount; ++m)
    {
        downViews.push_back(resources_->vulkanBloomMipView(bloomChainHandle_, m));
        downExtents.push_back({std::max(1u, bloomWidth >> m), std::max(1u, bloomHeight >> m)});
    }
    bloomDownPass_.createColourFramebuffersPerMip(*device_, downViews, downExtents);

    std::vector<vk::ImageView> upViews;
    std::vector<vk::Extent2D> upExtents;
    upViews.reserve(kBloomMipCount - 1);
    upExtents.reserve(kBloomMipCount - 1);
    for (uint32_t m = 0; m < kBloomMipCount - 1; ++m)
    {
        upViews.push_back(resources_->vulkanBloomMipView(bloomChainHandle_, m));
        upExtents.push_back({std::max(1u, bloomWidth >> m), std::max(1u, bloomHeight >> m)});
    }
    bloomUpPass_.createColourFramebuffersPerMip(*device_, upViews, upExtents);

    bloomDownDescSets_.clear();
    bloomDownDescSets_.reserve(kBloomMipCount);
    bloomDownDescSets_.push_back(resources_->descriptors().createImageViewDescriptor(
        bloomDownsamplePipeline_.descriptorSetLayout(),
        resources_->vulkanImageView(offscreenColourHandle_),
        resources_->vulkanSampler(offscreenColourHandle_)));
    for (uint32_t m = 0; m < kBloomMipCount - 1; ++m)
    {
        bloomDownDescSets_.push_back(resources_->descriptors().createImageViewDescriptor(
            bloomDownsamplePipeline_.descriptorSetLayout(),
            resources_->vulkanBloomMipView(bloomChainHandle_, m),
            resources_->vulkanSampler(bloomChainHandle_)));
    }

    bloomUpDescSets_.clear();
    bloomUpDescSets_.reserve(kBloomMipCount - 1);
    for (uint32_t m = 0; m < kBloomMipCount - 1; ++m)
    {
        bloomUpDescSets_.push_back(resources_->descriptors().createImageViewDescriptor(
            bloomUpsamplePipeline_.descriptorSetLayout(),
            resources_->vulkanBloomMipView(bloomChainHandle_, m + 1),
            resources_->vulkanSampler(bloomChainHandle_)));
    }
}

void PostProcessing::recordBloomPasses(vk::CommandBuffer cmd) const
{
    auto extent = swapchain_->extent();
    uint32_t bloomWidth = std::max(1u, extent.width / 2);
    uint32_t bloomHeight = std::max(1u, extent.height / 2);

    const vk::PipelineLayout downLayout = bloomDownsamplePipeline_.pipelineLayout();
    const vk::PipelineLayout upLayout = bloomUpsamplePipeline_.pipelineLayout();

    for (uint32_t m = 0; m < kBloomMipCount; ++m)
    {
        uint32_t dstW = std::max(1u, bloomWidth >> m);
        uint32_t dstH = std::max(1u, bloomHeight >> m);
        uint32_t srcW = (m == 0) ? extent.width : std::max(1u, bloomWidth >> (m - 1));
        uint32_t srcH = (m == 0) ? extent.height : std::max(1u, bloomHeight >> (m - 1));

        vk::Rect2D renderArea{
            .offset = vk::Offset2D{.x = 0, .y = 0},
            .extent = vk::Extent2D{.width = dstW, .height = dstH},
        };
        vk::RenderPassBeginInfo begin{
            .renderPass = bloomDownPass_.renderPass(),
            .framebuffer = bloomDownPass_.framebuffer(m),
            .renderArea = renderArea,
        };
        cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
        cmd.setViewport(0, makeFullViewport(static_cast<float>(dstW), static_cast<float>(dstH)));
        cmd.setScissor(0, renderArea);
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, bloomDownsamplePipeline_.pipeline());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, downLayout, 0,
                               resources_->vulkanDescriptorSet(bloomDownDescSets_[m]), {});
        BloomPushConstants pc{};
        pc.invInputResolution[0] = 1.0f / static_cast<float>(srcW);
        pc.invInputResolution[1] = 1.0f / static_cast<float>(srcH);
        pc.isFirstPass = (m == 0) ? 1 : 0;
        cmd.pushConstants<BloomPushConstants>(downLayout, vk::ShaderStageFlagBits::eFragment, 0,
                                              pc);
        cmd.draw(3, 1, 0, 0);
        cmd.endRenderPass();
    }

    for (int m = static_cast<int>(kBloomMipCount) - 2; m >= 0; --m)
    {
        uint32_t dstW = std::max(1u, bloomWidth >> m);
        uint32_t dstH = std::max(1u, bloomHeight >> m);
        uint32_t srcW = std::max(1u, bloomWidth >> (m + 1));
        uint32_t srcH = std::max(1u, bloomHeight >> (m + 1));

        vk::Rect2D renderArea{
            .offset = vk::Offset2D{.x = 0, .y = 0},
            .extent = vk::Extent2D{.width = dstW, .height = dstH},
        };
        vk::RenderPassBeginInfo begin{
            .renderPass = bloomUpPass_.renderPass(),
            .framebuffer = bloomUpPass_.framebuffer(static_cast<uint32_t>(m)),
            .renderArea = renderArea,
        };
        cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
        cmd.setViewport(0, makeFullViewport(static_cast<float>(dstW), static_cast<float>(dstH)));
        cmd.setScissor(0, renderArea);
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, bloomUpsamplePipeline_.pipeline());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, upLayout, 0,
                               resources_->vulkanDescriptorSet(bloomUpDescSets_[m]), {});
        BloomPushConstants pc{};
        pc.invInputResolution[0] = 1.0f / static_cast<float>(srcW);
        pc.invInputResolution[1] = 1.0f / static_cast<float>(srcH);
        pc.isFirstPass = 0;
        cmd.pushConstants<BloomPushConstants>(upLayout, vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.draw(3, 1, 0, 0);
        cmd.endRenderPass();
    }
}

void PostProcessing::transitionOffscreenForSampling(vk::CommandBuffer cmd) const
{
    vk::ImageMemoryBarrier2 offscreenReadyForSampling{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = resources_->vulkanImage(offscreenColourHandle_),
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &offscreenReadyForSampling});
}

void PostProcessing::recordPostProcessPass(vk::CommandBuffer cmd, uint32_t imageIndex,
                                           uint32_t currentFrame) const
{
    auto extent = swapchain_->extent();
    vk::Rect2D renderArea{
        .offset = vk::Offset2D{.x = 0, .y = 0},
        .extent = extent,
    };
    vk::RenderPassBeginInfo ppBegin{
        .renderPass = postProcessPass_.renderPass(),
        .framebuffer = postProcessPass_.framebuffer(imageIndex),
        .renderArea = renderArea,
    };
    cmd.beginRenderPass(ppBegin, vk::SubpassContents::eInline);

    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, postProcessPipeline_.pipeline());

    vk::DescriptorSet ppSet = resources_->vulkanDescriptorSet(postProcessDescSets_[currentFrame]);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, postProcessPipeline_.pipelineLayout(),
                           0, ppSet, {});
    PostProcessPushConstants ppc{};
    ppc.kBloomStrength = kBloomStrength;
    cmd.pushConstants<PostProcessPushConstants>(postProcessPipeline_.pipelineLayout(),
                                                vk::ShaderStageFlagBits::eFragment, 0, ppc);
    cmd.bindIndexBuffer(resources_->vulkanBuffer(postProcessIndexBuffer_), 0,
                        vk::IndexType::eUint16);
    cmd.drawIndexed(3, 1, 0, 0, 0);
    cmd.endRenderPass();
}

void PostProcessing::recreate()
{
    resources_->releaseTexture(offscreenColourHandle_);
    offscreenColourHandle_ = resources_->createOffscreenColourTarget(swapchain_->extent());
    postProcessPass_.createPostProcessFramebuffers(*device_, *swapchain_);

    resources_->releaseTexture(bloomChainHandle_);
    buildBloomResources();
    resources_->descriptors().updatePostProcessDescriptors(
        postProcessDescSets_, offscreenColourHandle_, bloomChainHandle_);
}

} // namespace fire_engine
