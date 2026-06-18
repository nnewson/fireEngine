#include <fire_engine/render/transmission.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

#include <fire_engine/render/descriptors.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

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
            // Set 2 — bindless materials (forward shader indexes it for every draw).
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                   resources.vulkanPipelineLayout(dc.pipeline), 2,
                                   resources.bindlessDescriptorSet(), {});
            lastBoundPipeline = dc.pipeline;
        }
        // Transmissive draws always use the merged opaque/double-sided forward
        // pipeline (BLEND draws are bucketed separately), which declares cull
        // mode dynamic — set it per draw from the material's double-sided flag.
        cmd.setCullMode(dc.doubleSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
        if (dc.vertexBuffer != NullBuffer)
        {
            cmd.bindVertexBuffers(0, resources.vulkanBuffer(dc.vertexBuffer), {vk::DeviceSize{0}});
        }

        vk::IndexType indexType =
            dc.indexType == DrawIndexType::UInt32 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
        cmd.bindIndexBuffer(resources.vulkanBuffer(dc.indexBuffer), 0, indexType);

        // Forward set 0 pushed inline (VK_KHR_push_descriptor); transmissive draws
        // are always the merged opaque/double-sided forward pipeline.
        pushForwardObjectDescriptors(cmd, resources, resources.vulkanPipelineLayout(dc.pipeline),
                                     dc);
        ForwardPushConstants pc{};
        pc.selfShadowSlot = dc.selfShadowSlot;
        pc.materialIndex = dc.materialIndex;
        cmd.pushConstants<ForwardPushConstants>(resources.vulkanPipelineLayout(dc.pipeline),
                                                vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
}

} // namespace

Transmission::Transmission(const Swapchain& swapchain, Resources& resources,
                           TextureHandle offscreenColourHandle)
    : swapchain_{&swapchain},
      resources_{&resources},
      offscreenColourHandle_{offscreenColourHandle}
{
    rebuildSceneColorChain();
}

void Transmission::recordPass(vk::CommandBuffer cmd, std::span<const DrawCommand> transmissiveDraws,
                              vk::DescriptorSet globalSet) const
{
    if (transmissiveDraws.empty())
    {
        return;
    }

    recordSceneColorCapture(cmd);
    recordForwardTransmissionPass(cmd, transmissiveDraws, globalSet);
}

void Transmission::recreate(TextureHandle offscreenColourHandle, TextureHandle velocityHandle)
{
    offscreenColourHandle_ = offscreenColourHandle;
    velocityHandle_ = velocityHandle;
    rebuildSceneColorChain();
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

    vk::ImageMemoryBarrier2 hdrToSrc{
        .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = hdrImage,
        .subresourceRange = colourRange(0, 1),
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &hdrToSrc});

    vk::ImageMemoryBarrier2 sceneToDst{
        .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = sceneImage,
        .subresourceRange = colourRange(0, sceneColorMipLevels_),
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &sceneToDst});

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
        vk::ImageMemoryBarrier2 srcReady{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sceneImage,
            .subresourceRange = colourRange(i - 1, 1),
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &srcReady});

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

        vk::ImageMemoryBarrier2 srcDone{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sceneImage,
            .subresourceRange = colourRange(i - 1, 1),
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &srcDone});

        mipWidth = nextWidth;
        mipHeight = nextHeight;
    }

    vk::ImageMemoryBarrier2 lastMipDone{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = sceneImage,
        .subresourceRange = colourRange(sceneColorMipLevels_ - 1, 1),
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &lastMipDone});

    vk::ImageMemoryBarrier2 hdrBack{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask =
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = hdrImage,
        .subresourceRange = colourRange(0, 1),
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &hdrBack});
}

void Transmission::recordForwardTransmissionPass(vk::CommandBuffer cmd,
                                                 std::span<const DrawCommand> transmissiveDraws,
                                                 vk::DescriptorSet globalSet) const
{
    auto extent = swapchain_->extent();
    vk::Rect2D renderArea{
        .offset = vk::Offset2D{.x = 0, .y = 0},
        .extent = extent,
    };

    // The forward pipelines write HDR colour + a velocity attachment (TAA). The
    // HDR target was brought to ColorAttachmentOptimal by recordSceneColorCapture;
    // the velocity target is ShaderReadOnly out of the forward pass — bring it
    // back to a colour attachment so transmissive draws can write motion vectors.
    vk::ImageMemoryBarrier2 velocityToAttach{
        .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = resources_->vulkanImage(velocityHandle_),
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &velocityToAttach});

    // Depth is still DepthStencilAttachmentOptimal from the forward pass; HDR +
    // velocity are loaded (loadOp Load) and rendered on top.
    std::array<vk::RenderingAttachmentInfo, 2> colours{
        vk::RenderingAttachmentInfo{
            .imageView = resources_->vulkanImageView(offscreenColourHandle_),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        },
        vk::RenderingAttachmentInfo{
            .imageView = resources_->vulkanImageView(velocityHandle_),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        },
    };
    vk::RenderingAttachmentInfo depth{
        .imageView = swapchain_->depthView(),
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
    };
    cmd.beginRendering(makeRenderingInfo(renderArea, colours, &depth));
    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);
    recordTransmissionDrawBucket(cmd, transmissiveDraws, *resources_, globalSet);
    cmd.endRendering();

    // Mirror the old pass's finalLayout: leave the HDR target in ShaderReadOnly
    // for bloom / post-process sampling.
    vk::ImageMemoryBarrier2 toShaderRead{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
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
    vk::ImageMemoryBarrier2 velocityToShaderRead{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = resources_->vulkanImage(velocityHandle_),
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    std::array<vk::ImageMemoryBarrier2, 2> barriers{toShaderRead, velocityToShaderRead};
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
                           .pImageMemoryBarriers = barriers.data()});
}

} // namespace fire_engine
