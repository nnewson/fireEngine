#include <algorithm>

#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

Swapchain::Swapchain(const Device& device, const Window& window)
    : device_(&device.device())
{
    createSwapchain(device, window);
    createImageViews();
}

void Swapchain::createDepthResources(const Device& device)
{
    vk::Format depthFmt = vk::Format::eD32Sfloat;
    vk::ImageCreateInfo ci{
        .imageType = vk::ImageType::e2D,
        .format = depthFmt,
        .extent = vk::Extent3D{.width = extent_.width, .height = extent_.height, .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    depthImage_ = vk::raii::Image(*device_, ci);

    auto req = depthImage_.getMemoryRequirements();
    vk::MemoryAllocateInfo ai{
        .allocationSize = req.size,
        .memoryTypeIndex =
            device.findMemoryType(req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    };
    depthMem_ = vk::raii::DeviceMemory(*device_, ai);
    depthImage_.bindMemory(*depthMem_, 0);
    depthView_ = createImageView(*depthImage_, depthFmt, vk::ImageAspectFlagBits::eDepth);
}

void Swapchain::recreate(const Device& device, const Window& window)
{
    // Destroy in reverse dependency order. Dynamic rendering uses no
    // framebuffers; the caller recreates dependent attachments (offscreen
    // colour, bloom chain, scene-colour) after this call returns.
    views_.clear();
    depthView_ = nullptr;
    depthImage_ = nullptr;
    depthMem_ = nullptr;
    swapchain_ = nullptr;

    createSwapchain(device, window);
    createImageViews();
    createDepthResources(device);
}

void Swapchain::createSwapchain(const Device& device, const Window& window)
{
    auto caps = device.physicalDevice().getSurfaceCapabilitiesKHR(*device.surface());
    auto fmt = chooseSwapFormat(device);
    auto mode = chooseSwapPresentMode(device);
    auto extent = chooseSwapExtent(window, caps);

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0)
    {
        imgCount = std::min(imgCount, caps.maxImageCount);
    }

    uint32_t families[] = {device.graphicsFamily(), device.presentFamily()};
    bool concurrent = device.graphicsFamily() != device.presentFamily();

    vk::SwapchainCreateInfoKHR ci{
        .surface = *device.surface(),
        .minImageCount = imgCount,
        .imageFormat = fmt.format,
        .imageColorSpace = fmt.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = concurrent ? 2u : 0u,
        .pQueueFamilyIndices = concurrent ? families : nullptr,
        .preTransform = caps.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = mode,
        .clipped = vk::True,
    };

    swapchain_ = vk::raii::SwapchainKHR(*device_, ci);
    images_ = swapchain_.getImages();
    format_ = fmt.format;
    extent_ = extent;
}

void Swapchain::createImageViews()
{
    views_.clear();
    views_.reserve(images_.size());
    for (size_t i = 0; i < images_.size(); ++i)
    {
        views_.push_back(createImageView(images_[i], format_, vk::ImageAspectFlagBits::eColor));
    }
}

vk::SurfaceFormatKHR Swapchain::chooseSwapFormat(const Device& device)
{
    auto fmts = device.physicalDevice().getSurfaceFormatsKHR(*device.surface());
    for (auto& f : fmts)
    {
        if (f.format == vk::Format::eB8G8R8A8Srgb &&
            f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return f;
        }
    }
    return fmts[0];
}

vk::PresentModeKHR Swapchain::chooseSwapPresentMode(const Device& device)
{
    auto modes = device.physicalDevice().getSurfacePresentModesKHR(*device.surface());
    for (auto& m : modes)
    {
        if (m == vk::PresentModeKHR::eMailbox)
        {
            return m;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Swapchain::chooseSwapExtent(const Window& window,
                                         const vk::SurfaceCapabilitiesKHR& caps)
{
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return caps.currentExtent;
    }
    auto [w, h] = window.framebufferSize();
    vk::Extent2D ext{.width = static_cast<uint32_t>(w), .height = static_cast<uint32_t>(h)};
    ext.width = std::clamp(ext.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    ext.height = std::clamp(ext.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return ext;
}

vk::raii::ImageView Swapchain::createImageView(vk::Image img, vk::Format fmt,
                                               vk::ImageAspectFlags aspect)
{
    vk::ImageViewCreateInfo ci{
        .image = img,
        .viewType = vk::ImageViewType::e2D,
        .format = fmt,
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = aspect,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    return vk::raii::ImageView(*device_, ci);
}

} // namespace fire_engine
