#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

#include <fire_engine/graphics/mapped_buffer.hpp>

using fire_engine::writeMapped;

namespace
{
struct Record
{
    std::uint32_t a;
    std::uint32_t b;
};
static_assert(std::is_trivially_copyable_v<Record>);
} // namespace

TEST_CASE("MappedBuffer.TypedWriteLandsTheBytes", "[MappedBuffer]")
{
    std::array<std::byte, sizeof(Record)> buf{};
    const Record rec{0x11223344U, 0xAABBCCDDU};

    writeMapped(std::span<std::byte>{buf}, rec);

    Record out{};
    std::memcpy(&out, buf.data(), sizeof(out));
    CHECK(out.a == rec.a);
    CHECK(out.b == rec.b);
}

TEST_CASE("MappedBuffer.ByteWriteCopiesExactCountAndLeavesTheRest", "[MappedBuffer]")
{
    std::array<std::byte, 8> buf{};
    const std::array<std::uint8_t, 3> src{1, 2, 3};

    writeMapped(std::span<std::byte>{buf}, src.data(), src.size());

    CHECK(std::to_integer<int>(buf[0]) == 1);
    CHECK(std::to_integer<int>(buf[1]) == 2);
    CHECK(std::to_integer<int>(buf[2]) == 3);
    CHECK(std::to_integer<int>(buf[3]) == 0); // beyond the write, untouched
}

TEST_CASE("MappedBuffer.SubspanOffsetWrite", "[MappedBuffer]")
{
    // Mirrors the material-SSBO write: record N goes into the big span at offset N * sizeof.
    std::array<std::byte, 16> buf{};
    constexpr std::size_t offset = 2 * sizeof(std::uint32_t);
    const std::uint32_t value = 0xDEADBEEFU;

    writeMapped(std::span<std::byte>{buf}.subspan(offset), value);

    std::uint32_t out = 0;
    std::memcpy(&out, buf.data() + offset, sizeof(out));
    CHECK(out == value);

    std::uint32_t first = 0;
    std::memcpy(&first, buf.data(), sizeof(first));
    CHECK(first == 0U); // earlier bytes untouched
}
