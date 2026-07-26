#include <fire_engine/graphics/frame_capture.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace fire_engine
{

std::vector<std::uint8_t> toRgba8(std::span<const std::byte> mapped, std::size_t width,
                                  std::size_t height, std::size_t rowPitch, CaptureFormat format)
{
    // A short mapping would read past the end of the staging buffer. Returning empty is the
    // CONTRACT, not a debug-only safety net: the caller reports a failed capture, which is
    // strictly better than writing a reference image built from whatever followed the buffer.
    // (Deliberately not an assert — that would abort the very case the tests pin.)
    if (rowPitch < width * 4 || mapped.size() < captureByteSize(height, rowPitch))
    {
        return {};
    }

    std::vector<std::uint8_t> rgba(width * height * 4);
    const std::size_t red = format == CaptureFormat::Bgra8 ? 2 : 0;
    const std::size_t blue = format == CaptureFormat::Bgra8 ? 0 : 2;
    for (std::size_t y = 0; y < height; ++y)
    {
        const std::byte* src = mapped.data() + y * rowPitch;
        std::uint8_t* dst = rgba.data() + y * width * 4;
        for (std::size_t x = 0; x < width; ++x)
        {
            const std::byte* pixel = src + x * 4;
            dst[x * 4 + 0] = static_cast<std::uint8_t>(pixel[red]);
            dst[x * 4 + 1] = static_cast<std::uint8_t>(pixel[1]);
            dst[x * 4 + 2] = static_cast<std::uint8_t>(pixel[blue]);
            dst[x * 4 + 3] = 0xFF;
        }
    }
    return rgba;
}

bool writeRgba8Png(const char* path, std::span<const std::uint8_t> rgba, std::size_t width,
                   std::size_t height)
{
    if (path == nullptr || rgba.size() < width * height * 4)
    {
        return false;
    }
    // Row stride is tight: the converter above packs rows without padding.
    return stbi_write_png(path, static_cast<int>(width), static_cast<int>(height), 4, rgba.data(),
                          static_cast<int>(width * 4)) != 0;
}

} // namespace fire_engine
