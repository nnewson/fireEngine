#pragma once

#include <optional>
#include <utility>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/platform/window.hpp>

namespace fire_engine
{

class Device
{
public:
    explicit Device(const Window& window);
    ~Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept = default;
    Device& operator=(Device&&) noexcept = default;

    [[nodiscard]] const vk::raii::Instance& instance() const noexcept
    {
        return instance_;
    }
    [[nodiscard]] const vk::raii::SurfaceKHR& surface() const noexcept
    {
        return surface_;
    }
    [[nodiscard]] const vk::raii::PhysicalDevice& physicalDevice() const noexcept
    {
        return physDevice_;
    }
    [[nodiscard]] const vk::raii::Device& device() const noexcept
    {
        return device_;
    }
    // One shared pipeline cache fed to every graphics/compute pipeline creation
    // (forward, IBL precompute, post/bloom/shadow/particle/cloth). Lets the
    // driver dedupe compilation work and warms pipeline recreation on resize.
    [[nodiscard]] const vk::raii::PipelineCache& pipelineCache() const noexcept
    {
        return pipelineCache_;
    }
    [[nodiscard]] const vk::raii::Queue& graphicsQueue() const noexcept
    {
        return graphicsQueue_;
    }
    [[nodiscard]] const vk::raii::Queue& presentQueue() const noexcept
    {
        return presentQueue_;
    }
    [[nodiscard]] uint32_t graphicsFamily() const noexcept
    {
        return graphicsFamily_;
    }
    [[nodiscard]] uint32_t presentFamily() const noexcept
    {
        return presentFamily_;
    }

    [[nodiscard]] uint32_t findMemoryType(uint32_t filter, vk::MemoryPropertyFlags props) const;

    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
    createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                 vk::MemoryPropertyFlags props) const;

private:
    void createInstance();
    void createSurface(const Window& window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createPipelineCache();

    [[nodiscard]] bool isDeviceSuitable(const vk::raii::PhysicalDevice& d);
    [[nodiscard]] std::pair<std::optional<uint32_t>, std::optional<uint32_t>>
    findQueueFamilies(const vk::raii::PhysicalDevice& d);
    void printValidationInfo() const;

    vk::raii::Context context_;
    vk::raii::Instance instance_{nullptr};
    vk::raii::SurfaceKHR surface_{nullptr};
    vk::raii::PhysicalDevice physDevice_{nullptr};
    vk::raii::Device device_{nullptr};
    vk::raii::PipelineCache pipelineCache_{nullptr};
    vk::raii::Queue graphicsQueue_{nullptr};
    vk::raii::Queue presentQueue_{nullptr};
    uint32_t graphicsFamily_{0};
    uint32_t presentFamily_{0};
};

} // namespace fire_engine
