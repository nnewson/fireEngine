#include <fire_engine/render/taa.hpp>

#include <array>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

namespace
{

// History + offscreen targets are RG16B16A16 HDR — must match the offscreen
// colour target the resolve blits into and reuses for history.
constexpr vk::Format kHdrFormat = vk::Format::eR16G16B16A16Sfloat;

// Single-subresource colour layout transition (synchronization2). Dynamic
// rendering performs no implicit transitions, so the resolve cycles its targets
// between colour-attachment / transfer / shader-read explicitly.
void imageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout oldLayout,
                  vk::ImageLayout newLayout, vk::PipelineStageFlags2 srcStage,
                  vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                  vk::AccessFlags2 dstAccess)
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
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}

} // namespace

Taa::Taa(const Device& device, const Swapchain& swapchain, Resources& resources,
         TextureHandle offscreenColour)
    : swapchain_{&swapchain},
      resources_{&resources},
      resolvePipeline_(device, Pipeline::taaResolveConfig(kHdrFormat)),
      offscreenHandle_{offscreenColour}
{
    velocityHandle_ = resources_->createVelocityTarget(swapchain_->extent());
    for (auto& h : historyHandles_)
    {
        h = resources_->createOffscreenColourTarget(swapchain_->extent());
    }
    resolveDescSets_ = resources_->descriptors().createTaaResolveDescriptors(
        resolvePipeline_.descriptorSetLayout(), offscreenHandle_, velocityHandle_, historyHandles_);
}

void Taa::recreate(TextureHandle offscreenColour)
{
    resources_->releaseTexture(velocityHandle_);
    velocityHandle_ = resources_->createVelocityTarget(swapchain_->extent());
    for (auto& h : historyHandles_)
    {
        resources_->releaseTexture(h);
        h = resources_->createOffscreenColourTarget(swapchain_->extent());
    }
    offscreenHandle_ = offscreenColour;
    resources_->descriptors().updateTaaResolveDescriptors(resolveDescSets_, offscreenHandle_,
                                                          velocityHandle_, historyHandles_);
    historyWritten_.fill(false);
}

void Taa::recordResolve(vk::CommandBuffer cmd, uint32_t currentFrame, float historyBlend,
                        float sharpen)
{
    const auto extent = swapchain_->extent();
    const uint32_t cur = currentFrame % kMaxFramesInFlight;
    const uint32_t prev = (kMaxFramesInFlight - 1) - cur;

    const vk::Image offscreenImage = resources_->vulkanImage(offscreenHandle_);
    const vk::Image curHistory = resources_->vulkanImage(historyHandles_[cur]);

    // The previous-history slot must be in a defined layout before the resolve
    // samples it. After its first write it already rests in ShaderReadOnly; the
    // very first time (or post-resize) it is Undefined, so transition it (the
    // contents are unused — historyValid is 0 below).
    if (!historyWritten_[prev])
    {
        imageBarrier(cmd, resources_->vulkanImage(historyHandles_[prev]),
                     vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal,
                     vk::PipelineStageFlagBits2::eFragmentShader, {},
                     vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
    }

    // Destination history slot → colour attachment (loadOp DontCare overwrites
    // every pixel, so its prior contents are discarded via Undefined).
    imageBarrier(cmd, curHistory, vk::ImageLayout::eUndefined,
                 vk::ImageLayout::eColorAttachmentOptimal,
                 vk::PipelineStageFlagBits2::eFragmentShader, {},
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite);

    vk::Rect2D renderArea{.offset = vk::Offset2D{.x = 0, .y = 0}, .extent = extent};
    vk::RenderingAttachmentInfo colour{
        .imageView = resources_->vulkanImageView(historyHandles_[cur]),
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmd.beginRendering(makeRenderingInfo(renderArea, {&colour, 1}, nullptr));
    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolvePipeline_.pipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, resolvePipeline_.pipelineLayout(), 0,
                           resources_->vulkanDescriptorSet(resolveDescSets_[cur]), {});
    TaaResolvePushConstants push{};
    push.texelSize[0] = 1.0f / static_cast<float>(extent.width);
    push.texelSize[1] = 1.0f / static_cast<float>(extent.height);
    push.historyBlend = historyBlend;
    push.sharpen = sharpen;
    push.historyValid = historyWritten_[prev] ? 1 : 0;
    cmd.pushConstants<TaaResolvePushConstants>(resolvePipeline_.pipelineLayout(),
                                               vk::ShaderStageFlagBits::eFragment, 0, push);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();

    // Blit the resolved history into the offscreen HDR target so the downstream
    // particle / bloom / post passes consume the anti-aliased image. The current
    // colour was sampled as a shader read in the resolve, so order that read
    // before the transfer write (WAR).
    imageBarrier(cmd, curHistory, vk::ImageLayout::eColorAttachmentOptimal,
                 vk::ImageLayout::eTransferSrcOptimal,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eTransfer,
                 vk::AccessFlagBits2::eTransferRead);
    imageBarrier(cmd, offscreenImage, vk::ImageLayout::eShaderReadOnlyOptimal,
                 vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eFragmentShader,
                 vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eTransfer,
                 vk::AccessFlagBits2::eTransferWrite);

    vk::ImageBlit blit{
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
    cmd.blitImage(curHistory, vk::ImageLayout::eTransferSrcOptimal, offscreenImage,
                  vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);

    // Offscreen back to ShaderReadOnly (the state the particle pass expects as
    // its precondition; mirrors the forward/transmission handoff). History slot
    // back to ShaderReadOnly so the next frame can sample it as "previous".
    imageBarrier(cmd, offscreenImage, vk::ImageLayout::eTransferDstOptimal,
                 vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
                 vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eFragmentShader,
                 vk::AccessFlagBits2::eShaderRead);
    imageBarrier(cmd, curHistory, vk::ImageLayout::eTransferSrcOptimal,
                 vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
                 vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eFragmentShader,
                 vk::AccessFlagBits2::eShaderRead);

    historyWritten_[cur] = true;
}

} // namespace fire_engine
