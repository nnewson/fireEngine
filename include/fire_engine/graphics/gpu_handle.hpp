#pragma once

#include <cstdint>
#include <limits>

namespace fire_engine
{

enum class BufferHandle : uint32_t
{
};

enum class TextureHandle : uint32_t
{
};

enum class DescriptorSetHandle : uint32_t
{
};

enum class PipelineHandle : uint32_t
{
};

inline constexpr auto NullBuffer = BufferHandle{std::numeric_limits<uint32_t>::max()};
inline constexpr auto NullTexture = TextureHandle{std::numeric_limits<uint32_t>::max()};
inline constexpr auto NullDescriptorSet = DescriptorSetHandle{std::numeric_limits<uint32_t>::max()};
inline constexpr auto NullPipeline = PipelineHandle{std::numeric_limits<uint32_t>::max()};

// Handles pack a table index in the low bits and a generation counter in the high bits. The
// index is what addresses the resource table / bindless array; the generation is bumped when a
// slot is reused (see Resources' texture free-list) so a stale handle to a recycled slot is
// detectably invalid on lookup. 24-bit index (16M slots) + 8-bit generation.
inline constexpr uint32_t kHandleIndexBits = 24;
inline constexpr uint32_t kHandleIndexMask = (uint32_t{1} << kHandleIndexBits) - 1u;
inline constexpr uint32_t kHandleGenerationMask = 0xFFu;

template <typename Handle>
[[nodiscard]] constexpr uint32_t handleIndex(Handle handle) noexcept
{
    return static_cast<uint32_t>(handle) & kHandleIndexMask;
}

template <typename Handle>
[[nodiscard]] constexpr uint32_t handleGeneration(Handle handle) noexcept
{
    return (static_cast<uint32_t>(handle) >> kHandleIndexBits) & kHandleGenerationMask;
}

template <typename Handle>
[[nodiscard]] constexpr Handle makeHandle(uint32_t index, uint32_t generation) noexcept
{
    return Handle{((generation & kHandleGenerationMask) << kHandleIndexBits) |
                  (index & kHandleIndexMask)};
}

} // namespace fire_engine
