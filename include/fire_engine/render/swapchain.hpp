#pragma once

#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/platform/window.hpp>
#include <fire_engine/render/device.hpp>

namespace fire_engine
{

class Swapchain
{
public:
    explicit Swapchain(const Device& device, const Window& window);
    ~Swapchain() = default;

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) noexcept = default;
    Swapchain& operator=(Swapchain&&) noexcept = default;

    void createDepthResources(const Device& device);
    void recreate(const Device& device, const Window& window);

    [[nodiscard]] vk::SwapchainKHR swapchain() const noexcept
    {
        return *swapchain_;
    }
    [[nodiscard]] const std::vector<vk::Image>& images() const noexcept
    {
        return images_;
    }
    [[nodiscard]] const std::vector<vk::raii::ImageView>& imageViews() const noexcept
    {
        return views_;
    }
    [[nodiscard]] vk::Format format() const noexcept
    {
        return format_;
    }
    [[nodiscard]] vk::Extent2D extent() const noexcept
    {
        return extent_;
    }
    [[nodiscard]] vk::ImageView depthView() const noexcept
    {
        return *depthView_;
    }

private:
    void createSwapchain(const Device& device, const Window& window);
    void createImageViews();

    [[nodiscard]] static vk::SurfaceFormatKHR chooseSwapFormat(const Device& device);
    [[nodiscard]] static vk::PresentModeKHR chooseSwapPresentMode(const Device& device);
    [[nodiscard]] static vk::Extent2D chooseSwapExtent(const Window& window,
                                                       const vk::SurfaceCapabilitiesKHR& caps);
    [[nodiscard]] vk::raii::ImageView createImageView(vk::Image img, vk::Format fmt,
                                                      vk::ImageAspectFlags aspect);

    const vk::raii::Device* device_{nullptr};
    vk::raii::SwapchainKHR swapchain_{nullptr};
    std::vector<vk::Image> images_;
    std::vector<vk::raii::ImageView> views_;
    vk::Format format_{};
    vk::Extent2D extent_{};

    vk::raii::Image depthImage_{nullptr};
    vk::raii::DeviceMemory depthMem_{nullptr};
    vk::raii::ImageView depthView_{nullptr};
};

} // namespace fire_engine
