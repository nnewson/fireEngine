#include <fire_engine/render/frame.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

Frame::Frame(const Device& device, Swapchain& swapchain)
    : device_(&device.device()),
      swapchainImageCount_(swapchain.images().size()),
      cmdPool_(createCommandPool(device))
{
    createCommandBuffers();
    createSyncObjects();
}

vk::raii::CommandPool Frame::createCommandPool(const Device& device)
{
    vk::CommandPoolCreateInfo ci{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = device.graphicsFamily(),
    };
    return vk::raii::CommandPool(device.device(), ci);
}

void Frame::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo ai{
        .commandPool = *cmdPool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(kMaxFramesInFlight),
    };
    auto bufs = device_->allocateCommandBuffers(ai);
    cmdBufs_.reserve(bufs.size());
    for (auto& b : bufs)
    {
        cmdBufs_.push_back(std::move(b));
    }
}

void Frame::createSyncObjects()
{
    // imageAvail (acquire signal) and renderDone (present wait) stay binary —
    // the WSI acquire/present path doesn't accept timeline semaphores.
    vk::SemaphoreCreateInfo sci{};

    imageAvail_.reserve(kMaxFramesInFlight);
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        imageAvail_.emplace_back(*device_, sci);
    }

    renderDone_.reserve(swapchainImageCount_);
    for (size_t i = 0; i < swapchainImageCount_; ++i)
    {
        renderDone_.emplace_back(*device_, sci);
    }

    // One monotonic timeline semaphore (initial value 0) for frame pacing.
    vk::SemaphoreTypeCreateInfo typeInfo{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0,
    };
    vk::SemaphoreCreateInfo timelineCi{.pNext = &typeInfo};
    timeline_ = vk::raii::Semaphore(*device_, timelineCi);
}

void Frame::destroyRenderFinishedSemaphores()
{
    renderDone_.clear();
}

void Frame::createRenderFinishedSemaphores(size_t count)
{
    vk::SemaphoreCreateInfo sci{};
    renderDone_.clear();
    renderDone_.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        renderDone_.emplace_back(*device_, sci);
    }
}

} // namespace fire_engine
