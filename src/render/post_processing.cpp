#include <fire_engine/render/post_processing.hpp>

#include <algorithm>

#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

namespace
{

// Single-subresource layout transition through synchronization2, used to cycle
// bloom-chain mips and the swapchain image between colour-attachment and
// shader-read/present states (dynamic rendering does no implicit transitions).
void imageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageAspectFlags aspect,
                  uint32_t baseMip, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
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
                                                      .baseMipLevel = baseMip,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}

} // namespace

PostProcessing::PostProcessing(const Device& device, const Swapchain& swapchain,
                               Resources& resources)
    : swapchain_{&swapchain},
      resources_{&resources},
      postProcessPipeline_(device, Pipeline::postProcessConfig(swapchain.format())),
      bloomDownsamplePipeline_(device,
                               Pipeline::bloomDownsampleConfig(vk::Format::eR16G16B16A16Sfloat)),
      bloomUpsamplePipeline_(device, Pipeline::bloomUpsampleConfig(vk::Format::eR16G16B16A16Sfloat))
{
    offscreenColourHandle_ = resources_->createOffscreenColourTarget(swapchain_->extent());
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
    const vk::Image bloomImage = resources_->vulkanImage(bloomChainHandle_);

    // Downsample: each mip is written once then sampled by the next (coarser)
    // pass, so we transition it Undefined → ColorAttachment (loadOp DontCare
    // discards) for the draw and → ShaderReadOnly afterwards. The source mip's
    // post-draw ShaderReadOnly barrier from the previous iteration provides the
    // read-after-write ordering for this pass's sample.
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

        imageBarrier(cmd, bloomImage, vk::ImageAspectFlagBits::eColor, m,
                     vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                     vk::PipelineStageFlagBits2::eColorAttachmentOutput, {},
                     vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                     vk::AccessFlagBits2::eColorAttachmentWrite);

        vk::RenderingAttachmentInfo colour{
            .imageView = resources_->vulkanBloomMipView(bloomChainHandle_, m),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eDontCare,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
        cmd.beginRendering(makeRenderingInfo(renderArea, {&colour, 1}, nullptr));
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
        cmd.endRendering();

        imageBarrier(cmd, bloomImage, vk::ImageAspectFlagBits::eColor, m,
                     vk::ImageLayout::eColorAttachmentOptimal,
                     vk::ImageLayout::eShaderReadOnlyOptimal,
                     vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                     vk::AccessFlagBits2::eColorAttachmentWrite,
                     vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
    }

    // Upsample: additively blend each coarser mip into the next finer one. The
    // destination mip already holds its downsample result (loadOp Load), so it
    // is transitioned ShaderReadOnly → ColorAttachment for the blend and back to
    // ShaderReadOnly so the next finer pass (and the final tone-map) can sample.
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

        imageBarrier(
            cmd, bloomImage, vk::ImageAspectFlagBits::eColor, static_cast<uint32_t>(m),
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite);

        vk::RenderingAttachmentInfo colour{
            .imageView =
                resources_->vulkanBloomMipView(bloomChainHandle_, static_cast<uint32_t>(m)),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
        cmd.beginRendering(makeRenderingInfo(renderArea, {&colour, 1}, nullptr));
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
        cmd.endRendering();

        imageBarrier(cmd, bloomImage, vk::ImageAspectFlagBits::eColor, static_cast<uint32_t>(m),
                     vk::ImageLayout::eColorAttachmentOptimal,
                     vk::ImageLayout::eShaderReadOnlyOptimal,
                     vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                     vk::AccessFlagBits2::eColorAttachmentWrite,
                     vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
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

    const vk::Image swapImage = swapchain_->images()[imageIndex];

    // Bring the acquired swapchain image into colour-attachment layout (its
    // contents are overwritten, loadOp DontCare). The acquire→write execution
    // dependency is carried by the imageAvailable semaphore wait at submit.
    imageBarrier(cmd, swapImage, vk::ImageAspectFlagBits::eColor, 0, vk::ImageLayout::eUndefined,
                 vk::ImageLayout::eColorAttachmentOptimal,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput, {},
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite);

    vk::RenderingAttachmentInfo colour{
        .imageView = *swapchain_->imageViews()[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmd.beginRendering(makeRenderingInfo(renderArea, {&colour, 1}, nullptr));

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
    cmd.endRendering();

    // Transition to present layout. The render→present dependency is carried by
    // the renderFinished semaphore signalled at submit, so dstStage is bottom.
    imageBarrier(cmd, swapImage, vk::ImageAspectFlagBits::eColor, 0,
                 vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite,
                 vk::PipelineStageFlagBits2::eBottomOfPipe, {});
}

void PostProcessing::recreate()
{
    resources_->releaseTexture(offscreenColourHandle_);
    offscreenColourHandle_ = resources_->createOffscreenColourTarget(swapchain_->extent());

    resources_->releaseTexture(bloomChainHandle_);
    buildBloomResources();
    resources_->descriptors().updatePostProcessDescriptors(
        postProcessDescSets_, offscreenColourHandle_, bloomChainHandle_);
}

} // namespace fire_engine
