#pragma once

#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace fire_engine
{

class Device;
class Swapchain;

// Owns a VkRenderPass and the framebuffers that render into it. A single
// logical pass (forward, shadow, post-process) is represented by one
// RenderPass instance. Framebuffers may be backed by swapchain images
// (forward) or by offscreen attachments (shadow/post-process).
class RenderPass
{
public:
    RenderPass() = default;
    ~RenderPass() = default;

    RenderPass(const RenderPass&) = delete;
    RenderPass& operator=(const RenderPass&) = delete;
    RenderPass(RenderPass&&) noexcept = default;
    RenderPass& operator=(RenderPass&&) noexcept = default;

    // Creates the forward-lit render pass (colour + depth) that writes into
    // the offscreen HDR target. Framebuffers are not yet built — call
    // createForwardFramebuffers once the render targets exist.
    [[nodiscard]] static RenderPass createForward(const Device& device);

    // Builds a single framebuffer using an offscreen HDR colour view and the
    // shared depth view at the given extent. Safe to call repeatedly.
    void createForwardFramebuffer(const Device& device, vk::ImageView colourView,
                                  vk::ImageView depthView, vk::Extent2D extent);

    // Forward-transmission render pass. KHR_materials_transmission F3 splits
    // the forward pass: the first sub-pass renders opaque + blend into the
    // HDR target, the renderer then captures sceneColor via a blit chain, and
    // this pass loads the HDR contents and renders transmissive draws on top.
    // Differs from createForward in loadOp (eLoad) and initialLayout
    // (eColorAttachmentOptimal — the renderer issues an explicit barrier
    // bringing the HDR target back into that layout after the blit).
    [[nodiscard]] static RenderPass createForwardTransmission(const Device& device);

    // Creates a depth-only shadow render pass. A single D32_SFLOAT attachment
    // with eClear loadOp / eStore storeOp and a finalLayout of
    // eShaderReadOnlyOptimal so the implicit transition exposes the depth map
    // to the forward fragment shader without an explicit pipeline barrier.
    [[nodiscard]] static RenderPass createShadow(const Device& device);

    // Builds one framebuffer per supplied depth view (shadow-map layer). The
    // colour attachment span may contain either one shared view or one view per
    // depth layer. Owned framebuffer list is replaced on every call.
    void createShadowFramebuffer(const Device& device, std::span<const vk::ImageView> colourViews,
                                 std::span<const vk::ImageView> depthViews, uint32_t extent);

    // Post-process render pass: a single colour attachment in swapchain
    // format, consumed by the presentation engine. Fragment shader samples
    // the forward HDR target via descriptor binding; the sample itself is
    // gated by a subpass dependency on colour-attachment-output.
    [[nodiscard]] static RenderPass createPostProcess(const Device& device,
                                                      const Swapchain& swapchain);

    // Builds one framebuffer per swapchain image, each wrapping the
    // swapchain's colour view at the swapchain extent.
    void createPostProcessFramebuffers(const Device& device, const Swapchain& swapchain);

    // Creates a colour-only render pass for utility offscreen rendering,
    // ending in shader-read layout.
    [[nodiscard]] static RenderPass createOffscreenColour(const Device& device,
                                                          vk::Format colourFormat);

    // Builds one framebuffer per provided colour view at the given square
    // extent. Used by cubemap face utility passes.
    void createColourFramebuffers(const Device& device, std::span<const vk::ImageView> colourViews,
                                  uint32_t extent);

    // Bloom downsample render pass — single colour attachment, loadOp=DontCare
    // (overwrites destination mip), storeOp=Store, finalLayout=eShaderReadOnly.
    [[nodiscard]] static RenderPass createBloomDown(const Device& device, vk::Format colourFormat);

    // Bloom upsample render pass — loadOp=Load + initialLayout=eShaderReadOnly
    // so the previous downsample's mip content is preserved. Pipeline uses
    // additive blending to accumulate the upsampled contribution.
    [[nodiscard]] static RenderPass createBloomUp(const Device& device, vk::Format colourFormat);

    // Builds one framebuffer per (colour view, extent) pair. Used by bloom
    // chain passes where each mip has a distinct extent.
    void createColourFramebuffersPerMip(const Device& device,
                                        std::span<const vk::ImageView> colourViews,
                                        std::span<const vk::Extent2D> extents);

    [[nodiscard]] vk::RenderPass renderPass() const noexcept
    {
        return *renderPass_;
    }

    [[nodiscard]] vk::Framebuffer framebuffer(std::size_t index) const noexcept
    {
        return *framebuffers_[index];
    }

    [[nodiscard]] std::size_t framebufferCount() const noexcept
    {
        return framebuffers_.size();
    }

private:
    vk::raii::RenderPass renderPass_{nullptr};
    std::vector<vk::raii::Framebuffer> framebuffers_;
};

} // namespace fire_engine
