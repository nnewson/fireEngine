#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/platform/window.hpp>
#include <fire_engine/render/vma.hpp>

namespace fire_engine
{

class Device
{
public:
    explicit Device(const Window& window);
    // Surface-free, compute-only device: no WSI/GLFW instance extensions, no surface, no swapchain
    // — it requires a single graphics+compute queue family (the same-queue path the renderer
    // already dispatches compute on) and keeps the production feature set (BDA, sync2, …). For
    // headless offscreen compute (the VDPM GPU harness). A production mode of Device, not a
    // test-only shim.
    [[nodiscard]] static Device headlessCompute();
    ~Device(); // persists the pipeline cache to disk

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
    // Absent (nullopt) for a headless compute device — it has no surface, so no present family.
    // Windowed callers (the swapchain) deref it directly.
    [[nodiscard]] std::optional<uint32_t> presentFamily() const noexcept
    {
        return presentFamily_;
    }

    // The process-wide VMA arena (Layer 3 of the GPU resource model). Every buffer/image
    // allocation goes through this rather than a per-resource vkAllocateMemory.
    [[nodiscard]] VmaAllocator allocator() const noexcept
    {
        return allocator_.get();
    }

    // Sub-allocates a buffer from the VMA arena. `props` carries the memory requirements the
    // caller needs (host-visible/coherent vs device-local); a host-visible request is created
    // persistently mapped, its pointer reachable via UniqueVmaBuffer::mapped().
    [[nodiscard]] UniqueVmaBuffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                               vk::MemoryPropertyFlags props) const;

private:
    enum class Mode : std::uint8_t
    {
        Windowed,       // surface + swapchain + present queue (the app)
        HeadlessCompute // no surface/swapchain, one graphics+compute queue (offscreen)
    };
    explicit Device(Mode mode); // shared init for the surface-free path

    void createInstance();
    void createSurface(const Window& window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createAllocator();
    void createPipelineCache();
    void savePipelineCache() const noexcept;

    [[nodiscard]] bool headless() const noexcept
    {
        return mode_ == Mode::HeadlessCompute;
    }
    // The device extensions required for the current mode (swapchain only when windowed).
    [[nodiscard]] std::vector<const char*> requiredDeviceExtensions() const;
    [[nodiscard]] bool isDeviceSuitable(const vk::raii::PhysicalDevice& d);
    [[nodiscard]] std::pair<std::optional<uint32_t>, std::optional<uint32_t>>
    findQueueFamilies(const vk::raii::PhysicalDevice& d);
    void printValidationInfo() const;

    Mode mode_{Mode::Windowed};
    vk::raii::Context context_;
    vk::raii::Instance instance_{nullptr};
    vk::raii::SurfaceKHR surface_{nullptr};
    vk::raii::PhysicalDevice physDevice_{nullptr};
    vk::raii::Device device_{nullptr};
    vk::raii::PipelineCache pipelineCache_{nullptr};
    vk::raii::Queue graphicsQueue_{nullptr};
    vk::raii::Queue presentQueue_{nullptr};
    uint32_t graphicsFamily_{0};
    std::optional<uint32_t> presentFamily_; // nullopt when headless (no surface)
    // Declared last so it is destroyed first — before device_/instance_, which it was built
    // from and which must outlive it.
    VmaAllocatorHandle allocator_;
};

} // namespace fire_engine
