#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

#include <fire_engine/graphics/gpu_handle.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Null handle constants
// ---------------------------------------------------------------------------

TEST_CASE("GpuHandle.NullBufferIsMaxUint32", "[GpuHandle]")
{
    CHECK(static_cast<uint32_t>(NullBuffer) == std::numeric_limits<uint32_t>::max());
}

TEST_CASE("GpuHandle.NullTextureIsMaxUint32", "[GpuHandle]")
{
    CHECK(static_cast<uint32_t>(NullTexture) == std::numeric_limits<uint32_t>::max());
}

TEST_CASE("GpuHandle.NullDescriptorSetIsMaxUint32", "[GpuHandle]")
{
    CHECK(static_cast<uint32_t>(NullDescriptorSet) == std::numeric_limits<uint32_t>::max());
}

TEST_CASE("GpuHandle.NullPipelineIsMaxUint32", "[GpuHandle]")
{
    CHECK(static_cast<uint32_t>(NullPipeline) == std::numeric_limits<uint32_t>::max());
}

// ---------------------------------------------------------------------------
// Constexpr verification
// ---------------------------------------------------------------------------

TEST_CASE("GpuHandle.NullBufferIsConstexpr", "[GpuHandle]")
{
    static_assert(NullBuffer == BufferHandle{std::numeric_limits<uint32_t>::max()});
}

TEST_CASE("GpuHandle.NullTextureIsConstexpr", "[GpuHandle]")
{
    static_assert(NullTexture == TextureHandle{std::numeric_limits<uint32_t>::max()});
}

TEST_CASE("GpuHandle.NullDescriptorSetIsConstexpr", "[GpuHandle]")
{
    static_assert(NullDescriptorSet == DescriptorSetHandle{std::numeric_limits<uint32_t>::max()});
}

TEST_CASE("GpuHandle.NullPipelineIsConstexpr", "[GpuHandle]")
{
    static_assert(NullPipeline == PipelineHandle{std::numeric_limits<uint32_t>::max()});
}

// ---------------------------------------------------------------------------
// Handle equality and comparison
// ---------------------------------------------------------------------------

TEST_CASE("GpuHandle.BufferHandleEqualityWithSameValue", "[GpuHandle]")
{
    auto a = BufferHandle{42};
    auto b = BufferHandle{42};
    CHECK(a == b);
}

TEST_CASE("GpuHandle.BufferHandleInequalityWithDifferentValues", "[GpuHandle]")
{
    auto a = BufferHandle{0};
    auto b = BufferHandle{1};
    CHECK(a != b);
}

TEST_CASE("GpuHandle.TextureHandleEqualityWithSameValue", "[GpuHandle]")
{
    auto a = TextureHandle{7};
    auto b = TextureHandle{7};
    CHECK(a == b);
}

TEST_CASE("GpuHandle.DescriptorSetHandleEqualityWithSameValue", "[GpuHandle]")
{
    auto a = DescriptorSetHandle{3};
    auto b = DescriptorSetHandle{3};
    CHECK(a == b);
}

TEST_CASE("GpuHandle.PipelineHandleEqualityWithSameValue", "[GpuHandle]")
{
    auto a = PipelineHandle{9};
    auto b = PipelineHandle{9};
    CHECK(a == b);
}

TEST_CASE("GpuHandle.PipelineHandleInequalityWithDifferentValues", "[GpuHandle]")
{
    auto a = PipelineHandle{0};
    auto b = PipelineHandle{1};
    CHECK(a != b);
}

// ---------------------------------------------------------------------------
// Null vs valid handles
// ---------------------------------------------------------------------------

TEST_CASE("GpuHandle.ValidBufferHandleNotEqualToNull", "[GpuHandle]")
{
    auto valid = BufferHandle{0};
    CHECK(valid != NullBuffer);
}

TEST_CASE("GpuHandle.ValidTextureHandleNotEqualToNull", "[GpuHandle]")
{
    auto valid = TextureHandle{0};
    CHECK(valid != NullTexture);
}

TEST_CASE("GpuHandle.ValidDescriptorSetHandleNotEqualToNull", "[GpuHandle]")
{
    auto valid = DescriptorSetHandle{0};
    CHECK(valid != NullDescriptorSet);
}

TEST_CASE("GpuHandle.ValidPipelineHandleNotEqualToNull", "[GpuHandle]")
{
    auto valid = PipelineHandle{0};
    CHECK(valid != NullPipeline);
}

// ---------------------------------------------------------------------------
// Type safety — handles are distinct types
// ---------------------------------------------------------------------------

TEST_CASE("GpuHandle.HandleTypesAreDistinct", "[GpuHandle]")
{
    static_assert(!std::is_same_v<BufferHandle, TextureHandle>);
    static_assert(!std::is_same_v<BufferHandle, DescriptorSetHandle>);
    static_assert(!std::is_same_v<TextureHandle, DescriptorSetHandle>);
    static_assert(!std::is_same_v<BufferHandle, PipelineHandle>);
    static_assert(!std::is_same_v<TextureHandle, PipelineHandle>);
    static_assert(!std::is_same_v<DescriptorSetHandle, PipelineHandle>);
}

// ---------------------------------------------------------------------------
// Round-trip cast
// ---------------------------------------------------------------------------

TEST_CASE("GpuHandle.BufferHandleRoundTrip", "[GpuHandle]")
{
    uint32_t id = 123;
    auto handle = BufferHandle{id};
    CHECK(static_cast<uint32_t>(handle) == id);
}

TEST_CASE("GpuHandle.TextureHandleRoundTrip", "[GpuHandle]")
{
    uint32_t id = 456;
    auto handle = TextureHandle{id};
    CHECK(static_cast<uint32_t>(handle) == id);
}

TEST_CASE("GpuHandle.DescriptorSetHandleRoundTrip", "[GpuHandle]")
{
    uint32_t id = 789;
    auto handle = DescriptorSetHandle{id};
    CHECK(static_cast<uint32_t>(handle) == id);
}

TEST_CASE("GpuHandle.PipelineHandleRoundTrip", "[GpuHandle]")
{
    uint32_t id = 1011;
    auto handle = PipelineHandle{id};
    CHECK(static_cast<uint32_t>(handle) == id);
}

// ---------------------------------------------------------------------------
// Index / generation packing (CR-12)
// ---------------------------------------------------------------------------

TEST_CASE("GpuHandle.MakeHandlePacksIndexAndGeneration", "[GpuHandle]")
{
    const auto handle = makeHandle<TextureHandle>(1234u, 7u);
    CHECK(handleIndex(handle) == 1234u);
    CHECK(handleGeneration(handle) == 7u);
}

TEST_CASE("GpuHandle.GenerationZeroLeavesRawValueAsIndex", "[GpuHandle]")
{
    // A generation-0 handle's raw value equals its index, so pre-generation call sites that
    // built handles as raw indices keep round-tripping through handleIndex unchanged.
    const auto handle = makeHandle<BufferHandle>(42u, 0u);
    CHECK(static_cast<uint32_t>(handle) == 42u);
    CHECK(handleIndex(handle) == 42u);
    CHECK(handleGeneration(handle) == 0u);
}

TEST_CASE("GpuHandle.IndexAndGenerationOccupyDisjointBits", "[GpuHandle]")
{
    // Max 24-bit index and max 8-bit generation coexist without corrupting each other.
    const auto handle = makeHandle<TextureHandle>(kHandleIndexMask, kHandleGenerationMask);
    CHECK(handleIndex(handle) == kHandleIndexMask);
    CHECK(handleGeneration(handle) == kHandleGenerationMask);
}

TEST_CASE("GpuHandle.SameSlotDifferentGenerationsAreDistinctHandles", "[GpuHandle]")
{
    // The crux of stale-handle detection: reusing a slot bumps the generation, so the old and
    // new handles for the same index compare unequal.
    const auto oldHandle = makeHandle<TextureHandle>(5u, 3u);
    const auto newHandle = makeHandle<TextureHandle>(5u, 4u);
    CHECK(oldHandle != newHandle);
    CHECK(handleIndex(oldHandle) == handleIndex(newHandle));
    CHECK(handleGeneration(oldHandle) != handleGeneration(newHandle));
}

TEST_CASE("GpuHandle.PackingIsConstexpr", "[GpuHandle]")
{
    static_assert(handleIndex(makeHandle<TextureHandle>(9u, 2u)) == 9u);
    static_assert(handleGeneration(makeHandle<TextureHandle>(9u, 2u)) == 2u);
}

// ---------------------------------------------------------------------------
// MappedMemory alias
// ---------------------------------------------------------------------------

TEST_CASE("GpuHandle.MappedMemoryIsVoidPointer", "[GpuHandle]")
{
    static_assert(std::is_same_v<MappedMemory, void*>);
}

TEST_CASE("GpuHandle.MappedMemoryDefaultInitializesNull", "[GpuHandle]")
{
    MappedMemory ptr{};
    CHECK(ptr == nullptr);
}
