#include <fire_engine/graphics/frame_capture.hpp>

// stb is compiled INTO this translation unit, and since the vcpkg include tree is now a `-I` path
// (it has to precede /usr/local/include — see the note in CMakeLists.txt) its warnings are no
// longer suppressed by -isystem. stb's `= { 0 }` aggregate initialisers are idiomatic C and not
// ours to fix, so the one warning they raise is silenced here, at the include, rather than by
// weakening the flag for our own code.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
// stb's HDR writer uses sprintf into a fixed buffer. We never call it (only PNG), but the
// implementation still compiles, and macOS marks sprintf deprecated.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <stb_image_write.h>
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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
