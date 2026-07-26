#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fire_engine
{

// Frame capture: turning a read-back swapchain image into 8-bit RGBA rows a PNG encoder can
// take. Vulkan-free on purpose — the format handling and the row/channel arithmetic are the
// parts that silently produce a plausible-but-wrong image (blue faces, a shifted picture),
// and they are exactly the parts a headless test can pin.
//
// The render layer maps the swapchain's vk::Format onto this enum and REJECTS anything else
// rather than guessing: a mis-decoded capture that still looks like a picture is worse than
// no capture, because it would be committed as a reference image.
enum class CaptureFormat : std::uint8_t
{
    Rgba8, // R in the first byte
    Bgra8, // B in the first byte — the common swapchain layout
};

// Converts a mapped linear image into tightly-packed RGBA8 rows.
//
// `rowPitch` is the mapped image's bytes per row, which is NOT width * 4 in general: a linear
// Vulkan image may pad each row, and treating the pitch as implicit is what produces the
// classic diagonally-sheared screenshot. Alpha is forced opaque — the swapchain's alpha is
// meaningless once composited, and a PNG viewer showing a semi-transparent reference image
// would be its own source of confusion.
[[nodiscard]] std::vector<std::uint8_t> toRgba8(std::span<const std::byte> mapped,
                                                std::size_t width, std::size_t height,
                                                std::size_t rowPitch, CaptureFormat format);

// Writes tightly-packed RGBA8 rows to `path` as a PNG. Returns false if the encoder or the
// file write fails, so the caller can report a failed capture instead of exiting "successfully"
// with no file. Kept beside the conversion (and out of the render layer) because encoding a
// PNG has nothing to do with Vulkan.
[[nodiscard]] bool writeRgba8Png(const char* path, std::span<const std::uint8_t> rgba,
                                 std::size_t width, std::size_t height);

// Bytes the caller must map for a given geometry — the readback buffer size.
[[nodiscard]] constexpr std::size_t captureByteSize(std::size_t height,
                                                    std::size_t rowPitch) noexcept
{
    return height * rowPitch;
}

} // namespace fire_engine
