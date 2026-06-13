#pragma once

#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace fire_engine
{

// Describes the attachment formats a pipeline renders into, for dynamic
// rendering (VK_KHR_dynamic_rendering, core in Vulkan 1.3). Replaces the
// VkRenderPass handle that pipelines previously referenced: a pipeline is
// created with a VkPipelineRenderingCreateInfo built from these formats, and
// record sites assemble a vk::RenderingInfo from image views directly — no
// VkRenderPass and no VkFramebuffer objects are involved.
struct RenderTarget
{
    std::vector<vk::Format> colourFormats;
    vk::Format depthFormat{vk::Format::eUndefined};

    [[nodiscard]] bool hasDepth() const noexcept
    {
        return depthFormat != vk::Format::eUndefined;
    }
};

// Assembles a vk::RenderingInfo over the given colour attachments and optional
// depth attachment (nullptr when none) at the supplied render area, with
// layerCount = 1. The attachment arrays must outlive the begin-rendering call.
[[nodiscard]] vk::RenderingInfo
makeRenderingInfo(vk::Rect2D area, std::span<const vk::RenderingAttachmentInfo> colours,
                  const vk::RenderingAttachmentInfo* depth);

} // namespace fire_engine
