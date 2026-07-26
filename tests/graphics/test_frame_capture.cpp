#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/frame_capture.hpp>

using fire_engine::captureByteSize;
using fire_engine::CaptureFormat;
using fire_engine::toRgba8;

// ---------------------------------------------------------------------------
// Swapchain readback → RGBA8.
//
// The two ways this silently produces a plausible-but-wrong reference image are a missed
// channel swap (a blue-looking scene that still "renders") and a missed row pitch (a
// diagonally sheared picture). Both are pinned here over synthetic buffers, because neither
// is visible from a build or a VUID count — only from someone eventually noticing a committed
// baseline looks odd.
// ---------------------------------------------------------------------------

namespace
{
// Two pixels, byte-explicit, in each layout. Alpha is deliberately NOT 255 in the source, to
// prove the converter forces it opaque rather than passing it through.
constexpr std::array<std::uint8_t, 8> kRgbaSource{
    0x10, 0x20, 0x30, 0x40, // pixel 0: R=10 G=20 B=30
    0xA0, 0xB0, 0xC0, 0x00, // pixel 1: R=A0 G=B0 C=C0
};
constexpr std::array<std::uint8_t, 8> kBgraSource{
    0x30, 0x20, 0x10, 0x40, // pixel 0: same colour, B first
    0xC0, 0xB0, 0xA0, 0x00, // pixel 1
};

std::span<const std::byte> bytesOf(const std::array<std::uint8_t, 8>& src)
{
    return {reinterpret_cast<const std::byte*>(src.data()), src.size()};
}
} // namespace

TEST_CASE("FrameCapture.Rgba8PassesChannelsThrough", "[FrameCapture]")
{
    const std::vector<std::uint8_t> out =
        toRgba8(bytesOf(kRgbaSource), 2, 1, 8, CaptureFormat::Rgba8);

    REQUIRE(out.size() == 8);
    CHECK(out[0] == 0x10);
    CHECK(out[1] == 0x20);
    CHECK(out[2] == 0x30);
    CHECK(out[4] == 0xA0);
    CHECK(out[5] == 0xB0);
    CHECK(out[6] == 0xC0);
}

TEST_CASE("FrameCapture.Bgra8SwapsRedAndBlue", "[FrameCapture]")
{
    // The common swapchain layout. Getting this wrong yields an image that still looks like a
    // scene — just with the sky orange — which is exactly why it needs a byte-level test.
    const std::vector<std::uint8_t> out =
        toRgba8(bytesOf(kBgraSource), 2, 1, 8, CaptureFormat::Bgra8);

    REQUIRE(out.size() == 8);
    CHECK(out[0] == 0x10);
    CHECK(out[1] == 0x20);
    CHECK(out[2] == 0x30);
    CHECK(out[4] == 0xA0);
    CHECK(out[5] == 0xB0);
    CHECK(out[6] == 0xC0);
}

TEST_CASE("FrameCapture.BothLayoutsAgreeOnTheSameColour", "[FrameCapture]")
{
    // The two sources describe identical pixels in different orders, so the converted output
    // must be byte-identical — a stronger statement than either test alone.
    CHECK(toRgba8(bytesOf(kRgbaSource), 2, 1, 8, CaptureFormat::Rgba8) ==
          toRgba8(bytesOf(kBgraSource), 2, 1, 8, CaptureFormat::Bgra8));
}

TEST_CASE("FrameCapture.AlphaIsForcedOpaque", "[FrameCapture]")
{
    const std::vector<std::uint8_t> out =
        toRgba8(bytesOf(kRgbaSource), 2, 1, 8, CaptureFormat::Rgba8);

    REQUIRE(out.size() == 8);
    CHECK(out[3] == 0xFF);
    CHECK(out[7] == 0xFF);
}

TEST_CASE("FrameCapture.RowPitchPaddingIsSkipped", "[FrameCapture]")
{
    // A linear Vulkan image may pad each row. Treating the pitch as width * 4 shifts every
    // row by the padding and produces the classic sheared screenshot.
    constexpr std::size_t width = 2;
    constexpr std::size_t height = 2;
    constexpr std::size_t pitch = 12; // 8 bytes of pixels + 4 bytes of padding
    std::array<std::uint8_t, height * pitch> padded{};
    padded[0] = 0x11;
    padded[4] = 0x22;
    padded[8] = 0xEE; // padding — must not appear in the output
    padded[pitch + 0] = 0x33;
    padded[pitch + 4] = 0x44;
    padded[pitch + 8] = 0xEE;

    const std::vector<std::uint8_t> out =
        toRgba8({reinterpret_cast<const std::byte*>(padded.data()), padded.size()}, width, height,
                pitch, CaptureFormat::Rgba8);

    REQUIRE(out.size() == width * height * 4);
    CHECK(out[0] == 0x11);
    CHECK(out[4] == 0x22);
    CHECK(out[8] == 0x33); // second row starts right after the first, padding dropped
    CHECK(out[12] == 0x44);
}

TEST_CASE("FrameCapture.UndersizedMappingYieldsNothing", "[FrameCapture]")
{
    // Rather than read past the staging buffer and encode whatever followed it. The caller
    // reports a failed capture; it does not get a corrupt reference image.
    std::array<std::uint8_t, 4> tooSmall{};
    CHECK(toRgba8({reinterpret_cast<const std::byte*>(tooSmall.data()), tooSmall.size()}, 4, 4, 16,
                  CaptureFormat::Rgba8)
              .empty());
}

TEST_CASE("FrameCapture.ByteSizeMatchesPitchTimesHeight", "[FrameCapture]")
{
    STATIC_REQUIRE(captureByteSize(600, std::size_t{800} * 4) == std::size_t{600} * 800 * 4);
    STATIC_REQUIRE(captureByteSize(2, 12) == 24U); // padded rows count their padding
}
