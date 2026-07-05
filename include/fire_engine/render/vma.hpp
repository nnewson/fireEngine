#pragma once

#include <cstddef>
#include <span>
#include <utility>

// vk_mem_alloc.h is a large third-party header; pull it in as cleanly as we can and silence
// its own diagnostics so the engine's -Wall -Wextra -Wpedantic stay strict for our code. The
// amalgamated implementation (VMA_IMPLEMENTATION) lives in a single TU: src/render/vma_impl.cpp.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-extension"
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <vk_mem_alloc.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <vulkan/vulkan.hpp>

namespace fire_engine
{

// Layer 3 of the GPU resource model (see CLAUDE.md "GPU resource model"): the process-wide
// VMA arena that sub-allocates all buffers and images. Move-only RAII owner; Device owns one
// and destroys it before the VkDevice/VkInstance it was built from.
class VmaAllocatorHandle
{
public:
    VmaAllocatorHandle() noexcept = default;

    explicit VmaAllocatorHandle(VmaAllocator allocator) noexcept
        : allocator_(allocator)
    {
    }

    ~VmaAllocatorHandle()
    {
        reset();
    }

    VmaAllocatorHandle(const VmaAllocatorHandle&) = delete;
    VmaAllocatorHandle& operator=(const VmaAllocatorHandle&) = delete;

    VmaAllocatorHandle(VmaAllocatorHandle&& other) noexcept
        : allocator_(std::exchange(other.allocator_, nullptr))
    {
    }

    VmaAllocatorHandle& operator=(VmaAllocatorHandle&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            allocator_ = std::exchange(other.allocator_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] VmaAllocator get() const noexcept
    {
        return allocator_;
    }

    explicit operator bool() const noexcept
    {
        return allocator_ != nullptr;
    }

private:
    void reset() noexcept
    {
        if (allocator_ != nullptr)
        {
            vmaDestroyAllocator(allocator_);
            allocator_ = nullptr;
        }
    }

    VmaAllocator allocator_{nullptr};
};

// A VkBuffer plus its VMA sub-allocation, owned as a unit (Approach A — see CLAUDE.md). Move-
// only; frees both via vmaDestroyBuffer. `operator*` returns the `vk::Buffer` handle to mirror
// vk::raii, so call sites read the same. Host-visible allocations created MAPPED expose their
// persistent pointer through mapped().
class UniqueVmaBuffer
{
public:
    UniqueVmaBuffer() noexcept = default;

    UniqueVmaBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, void* mapped,
                    std::size_t size) noexcept
        : allocator_(allocator),
          buffer_(buffer),
          allocation_(allocation),
          mapped_(mapped),
          size_(size)
    {
    }

    ~UniqueVmaBuffer()
    {
        reset();
    }

    UniqueVmaBuffer(const UniqueVmaBuffer&) = delete;
    UniqueVmaBuffer& operator=(const UniqueVmaBuffer&) = delete;

    UniqueVmaBuffer(UniqueVmaBuffer&& other) noexcept
        : allocator_(other.allocator_),
          buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
          allocation_(std::exchange(other.allocation_, nullptr)),
          mapped_(std::exchange(other.mapped_, nullptr)),
          size_(std::exchange(other.size_, 0))
    {
    }

    UniqueVmaBuffer& operator=(UniqueVmaBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            allocator_ = other.allocator_;
            buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
            allocation_ = std::exchange(other.allocation_, nullptr);
            mapped_ = std::exchange(other.mapped_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    [[nodiscard]] vk::Buffer operator*() const noexcept
    {
        return vk::Buffer{buffer_};
    }

    // Persistently-mapped bytes for host-visible buffers, sized to the requested buffer size so
    // writes are bounds-checkable (CR-21). Empty span for device-local / unmapped buffers.
    [[nodiscard]] std::span<std::byte> mapped() const noexcept
    {
        return mapped_ != nullptr ? std::span<std::byte>{static_cast<std::byte*>(mapped_), size_}
                                  : std::span<std::byte>{};
    }

    explicit operator bool() const noexcept
    {
        return buffer_ != VK_NULL_HANDLE;
    }

private:
    void reset() noexcept
    {
        if (buffer_ != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator_, buffer_, allocation_);
            buffer_ = VK_NULL_HANDLE;
            allocation_ = nullptr;
            mapped_ = nullptr;
            size_ = 0;
        }
    }

    VmaAllocator allocator_{nullptr};
    VkBuffer buffer_{VK_NULL_HANDLE};
    VmaAllocation allocation_{nullptr};
    void* mapped_{nullptr};
    std::size_t size_{0};
};

// A VkImage plus its VMA sub-allocation, owned as a unit (Approach A). Move-only; frees both
// via vmaDestroyImage. `operator*` returns the `vk::Image` handle to mirror vk::raii.
class UniqueVmaImage
{
public:
    UniqueVmaImage() noexcept = default;

    UniqueVmaImage(VmaAllocator allocator, VkImage image, VmaAllocation allocation) noexcept
        : allocator_(allocator),
          image_(image),
          allocation_(allocation)
    {
    }

    ~UniqueVmaImage()
    {
        reset();
    }

    UniqueVmaImage(const UniqueVmaImage&) = delete;
    UniqueVmaImage& operator=(const UniqueVmaImage&) = delete;

    UniqueVmaImage(UniqueVmaImage&& other) noexcept
        : allocator_(other.allocator_),
          image_(std::exchange(other.image_, VK_NULL_HANDLE)),
          allocation_(std::exchange(other.allocation_, nullptr))
    {
    }

    UniqueVmaImage& operator=(UniqueVmaImage&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            allocator_ = other.allocator_;
            image_ = std::exchange(other.image_, VK_NULL_HANDLE);
            allocation_ = std::exchange(other.allocation_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] vk::Image operator*() const noexcept
    {
        return vk::Image{image_};
    }

    explicit operator bool() const noexcept
    {
        return image_ != VK_NULL_HANDLE;
    }

private:
    void reset() noexcept
    {
        if (image_ != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator_, image_, allocation_);
            image_ = VK_NULL_HANDLE;
            allocation_ = nullptr;
        }
    }

    VmaAllocator allocator_{nullptr};
    VkImage image_{VK_NULL_HANDLE};
    VmaAllocation allocation_{nullptr};
};

} // namespace fire_engine
