#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace fire_engine
{

// A vk::Viewport spanning the full target with depth in [0, 1]. Five+ call
// sites in render/ were repeating the same six-field designated initialiser.
[[nodiscard]]
inline vk::Viewport makeFullViewport(float width, float height) noexcept
{
    return vk::Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = width,
        .height = height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
}

[[nodiscard]]
inline vk::Viewport makeFullViewport(vk::Extent2D extent) noexcept
{
    return makeFullViewport(static_cast<float>(extent.width), static_cast<float>(extent.height));
}

} // namespace fire_engine
