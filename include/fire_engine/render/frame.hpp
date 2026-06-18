#pragma once

#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

class Frame
{
public:
    Frame(const Device& device, Swapchain& swapchain);
    ~Frame() = default;

    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    Frame(Frame&&) noexcept = default;
    Frame& operator=(Frame&&) noexcept = default;

    [[nodiscard]] vk::CommandBuffer commandBuffer(uint32_t index) const noexcept
    {
        return *cmdBufs_[index];
    }

    [[nodiscard]] vk::Semaphore imageAvailable(uint32_t index) const noexcept
    {
        return *imageAvail_[index];
    }

    [[nodiscard]] vk::Semaphore renderFinished(uint32_t imageIndex) const noexcept
    {
        return *renderDone_[imageIndex];
    }

    // Single monotonic timeline semaphore driving CPU↔GPU frame pacing (replaces
    // the per-frame in-flight fences). Created once; survives swapchain resize.
    [[nodiscard]] vk::Semaphore timeline() const noexcept
    {
        return *timeline_;
    }

    [[nodiscard]] vk::CommandPool commandPool() const noexcept
    {
        return *cmdPool_;
    }

    void destroyRenderFinishedSemaphores();
    void createRenderFinishedSemaphores(size_t count);

private:
    [[nodiscard]] static vk::raii::CommandPool createCommandPool(const Device& device);
    void createCommandBuffers();
    void createSyncObjects();

    const vk::raii::Device* device_{nullptr};
    size_t swapchainImageCount_{0};

    // Declaration order matters: command buffers destroyed before pool
    vk::raii::CommandPool cmdPool_{nullptr};
    std::vector<vk::raii::CommandBuffer> cmdBufs_;

    std::vector<vk::raii::Semaphore> imageAvail_;
    std::vector<vk::raii::Semaphore> renderDone_;
    vk::raii::Semaphore timeline_{nullptr};
};

} // namespace fire_engine
