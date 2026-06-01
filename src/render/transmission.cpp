#include <fire_engine/render/transmission.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

void recordTransmissionDrawBucket(vk::CommandBuffer cmd, std::span<const DrawCommand> drawCommands,
                                  const Resources& resources, vk::DescriptorSet globalSet)
{
    auto lastBoundPipeline = PipelineHandle{std::numeric_limits<uint32_t>::max()};
    for (const auto& dc : drawCommands)
    {
        if (dc.pipeline != lastBoundPipeline)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             resources.vulkanPipeline(dc.pipeline));
            // Transmission draws use forward pipelines; bind set 1 (globals)
            // whenever the pipeline changes so it survives any prior pipeline
            // switch that might have used a set-1-incompatible layout.
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                   resources.vulkanPipelineLayout(dc.pipeline), 1, globalSet, {});
            lastBoundPipeline = dc.pipeline;
        }
        if (dc.vertexBuffer != NullBuffer)
        {
            cmd.bindVertexBuffers(0, resources.vulkanBuffer(dc.vertexBuffer), {vk::DeviceSize{0}});
        }

        vk::IndexType indexType =
            dc.indexType == DrawIndexType::UInt32 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
        cmd.bindIndexBuffer(resources.vulkanBuffer(dc.indexBuffer), 0, indexType);

        vk::DescriptorSet descriptorSet = resources.vulkanDescriptorSet(dc.descriptorSet);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               resources.vulkanPipelineLayout(dc.pipeline), 0, descriptorSet, {});
        ForwardPushConstants pc{};
        pc.selfShadowSlot = dc.selfShadowSlot;
        cmd.pushConstants<ForwardPushConstants>(resources.vulkanPipelineLayout(dc.pipeline),
                                                vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
}

} // namespace

Transmission::Transmission(const Device& device, const Swapchain& swapchain, Resources& resources,
                           TextureHandle offscreenColourHandle)
    : device_{&device},
      swapchain_{&swapchain},
      resources_{&resources},
      forwardTransmissionPass_(RenderPass::createForwardTransmission(device)),
      offscreenColourHandle_{offscreenColourHandle}
{
    rebuildSceneColorChain();
}

void Transmission::recordPass(vk::CommandBuffer cmd,
                              std::span<const DrawCommand> transmissiveDraws,
                              vk::DescriptorSet globalSet) const
{
    if (transmissiveDraws.empty())
    {
        return;
    }

    recordSceneColorCapture(cmd);
    recordForwardTransmissionPass(cmd, transmissiveDraws, globalSet);
}

void Transmission::recreate(TextureHandle offscreenColourHandle)
{
    offscreenColourHandle_ = offscreenColourHandle;
    buildFramebuffer();
    rebuildSceneColorChain();
}

void Transmission::buildFramebuffer()
{
    forwardTransmissionPass_.createForwardFramebuffer(
        *device_, resources_->vulkanImageView(offscreenColourHandle_), swapchain_->depthView(),
        swapchain_->extent());
}

void Transmission::rebuildSceneColorChain()
{
    if (sceneColorHandle_ != NullTexture)
    {
        resources_->releaseTexture(sceneColorHandle_);
        sceneColorHandle_ = NullTexture;
    }

    const auto extent = swapchain_->extent();
    const uint32_t maxDim = std::max(extent.width, extent.height);
    sceneColorMipLevels_ = static_cast<uint32_t>(std::bit_width(maxDim));

    sceneColorHandle_ =
        resources_->createSceneColorTarget(extent.width, extent.height, sceneColorMipLevels_);
    resources_->sharedTextures().sceneColor = sceneColorHandle_;
}

void Transmission::recordSceneColorCapture(vk::CommandBuffer cmd) const
{
    auto extent = swapchain_->extent();
    auto hdrImage = resources_->vulkanImage(offscreenColourHandle_);
    auto sceneImage = resources_->vulkanImage(sceneColorHandle_);

    auto colourRange = [](uint32_t baseMip, uint32_t levels)
    {
        return vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = baseMip,
            .levelCount = levels,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
    };

    vk::ImageMemoryBarrier hdrToSrc{
        .srcAccessMask = vk::AccessFlagBits::eShaderRead,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = hdrImage,
        .subresourceRange = colourRange(0, 1),
    };
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                        vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, hdrToSrc);

    vk::ImageMemoryBarrier sceneToDst{
        .srcAccessMask = vk::AccessFlagBits::eShaderRead,
        .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = sceneImage,
        .subresourceRange = colourRange(0, sceneColorMipLevels_),
    };
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                        vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, sceneToDst);

    vk::ImageBlit blit0{
        .srcSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                     .mipLevel = 0,
                                                     .baseArrayLayer = 0,
                                                     .layerCount = 1},
        .srcOffsets =
            std::array<vk::Offset3D, 2>{vk::Offset3D{.x = 0, .y = 0, .z = 0},
                                        vk::Offset3D{.x = static_cast<int32_t>(extent.width),
                                                     .y = static_cast<int32_t>(extent.height),
                                                     .z = 1}},
        .dstSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                     .mipLevel = 0,
                                                     .baseArrayLayer = 0,
                                                     .layerCount = 1},
        .dstOffsets =
            std::array<vk::Offset3D, 2>{vk::Offset3D{.x = 0, .y = 0, .z = 0},
                                        vk::Offset3D{.x = static_cast<int32_t>(extent.width),
                                                     .y = static_cast<int32_t>(extent.height),
                                                     .z = 1}},
    };
    cmd.blitImage(hdrImage, vk::ImageLayout::eTransferSrcOptimal, sceneImage,
                  vk::ImageLayout::eTransferDstOptimal, blit0, vk::Filter::eLinear);

    int32_t mipWidth = static_cast<int32_t>(extent.width);
    int32_t mipHeight = static_cast<int32_t>(extent.height);
    for (uint32_t i = 1; i < sceneColorMipLevels_; ++i)
    {
        vk::ImageMemoryBarrier srcReady{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sceneImage,
            .subresourceRange = colourRange(i - 1, 1),
        };
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, srcReady);

        const int32_t nextWidth = std::max(1, mipWidth >> 1);
        const int32_t nextHeight = std::max(1, mipHeight >> 1);
        vk::ImageBlit blit{
            .srcSubresource =
                vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                           .mipLevel = i - 1,
                                           .baseArrayLayer = 0,
                                           .layerCount = 1},
            .srcOffsets =
                std::array<vk::Offset3D, 2>{vk::Offset3D{.x = 0, .y = 0, .z = 0},
                                            vk::Offset3D{.x = mipWidth, .y = mipHeight, .z = 1}},
            .dstSubresource =
                vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                           .mipLevel = i,
                                           .baseArrayLayer = 0,
                                           .layerCount = 1},
            .dstOffsets =
                std::array<vk::Offset3D, 2>{vk::Offset3D{.x = 0, .y = 0, .z = 0},
                                            vk::Offset3D{.x = nextWidth, .y = nextHeight, .z = 1}},
        };
        cmd.blitImage(sceneImage, vk::ImageLayout::eTransferSrcOptimal, sceneImage,
                      vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        vk::ImageMemoryBarrier srcDone{
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sceneImage,
            .subresourceRange = colourRange(i - 1, 1),
        };
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, srcDone);

        mipWidth = nextWidth;
        mipHeight = nextHeight;
    }

    vk::ImageMemoryBarrier lastMipDone{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = sceneImage,
        .subresourceRange = colourRange(sceneColorMipLevels_ - 1, 1),
    };
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, lastMipDone);

    vk::ImageMemoryBarrier hdrBack{
        .srcAccessMask = vk::AccessFlagBits::eTransferRead,
        .dstAccessMask =
            vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = hdrImage,
        .subresourceRange = colourRange(0, 1),
    };
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {}, hdrBack);
}

void Transmission::recordForwardTransmissionPass(
    vk::CommandBuffer cmd, std::span<const DrawCommand> transmissiveDraws,
    vk::DescriptorSet globalSet) const
{
    auto extent = swapchain_->extent();
    vk::Rect2D renderArea{
        .offset = vk::Offset2D{.x = 0, .y = 0},
        .extent = extent,
    };
    vk::RenderPassBeginInfo begin{
        .renderPass = forwardTransmissionPass_.renderPass(),
        .framebuffer = forwardTransmissionPass_.framebuffer(0),
        .renderArea = renderArea,
    };
    cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
    cmd.setViewport(0, vk::Viewport{.x = 0.0f,
                                    .y = 0.0f,
                                    .width = static_cast<float>(extent.width),
                                    .height = static_cast<float>(extent.height),
                                    .minDepth = 0.0f,
                                    .maxDepth = 1.0f});
    cmd.setScissor(0, renderArea);
    recordTransmissionDrawBucket(cmd, transmissiveDraws, *resources_, globalSet);
    cmd.endRenderPass();
}

} // namespace fire_engine
