#include <fire_engine/graphics/image.hpp>

#include <memory>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace fire_engine
{

namespace
{

constexpr int desiredChannels = 4;

[[nodiscard]] bool isHdrPath(const std::string& path)
{
    return stbi_is_hdr(path.c_str()) != 0;
}

[[nodiscard]] bool isHdrMemory(const uint8_t* data, std::size_t size_bytes)
{
    return stbi_is_hdr_from_memory(data, static_cast<int>(size_bytes)) != 0;
}

[[nodiscard]] std::string stbFailureDetail()
{
    const char* reason = stbi_failure_reason();
    if (reason == nullptr || reason[0] == '\0')
    {
        return "unknown stb_image error";
    }
    return reason;
}

} // namespace

Image Image::load_from_file(const std::string& path)
{
    int width = 0;
    int height = 0;
    int fileChannels = 0;

    Image img;
    img.width_ = width;
    img.height_ = height;
    img.channels_ = desiredChannels;

    if (isHdrPath(path))
    {
        std::unique_ptr<float, decltype(&stbi_image_free)> raw{
            stbi_loadf(path.c_str(), &width, &height, &fileChannels, desiredChannels),
            stbi_image_free};
        if (raw == nullptr)
        {
            throw std::runtime_error("Failed to load image: " + path + " (" + stbFailureDetail() +
                                     ")");
        }

        img.width_ = width;
        img.height_ = height;
        img.channels_ = desiredChannels;
        img.pixelType_ = ImagePixelType::Float32;
        std::size_t pixelCount = static_cast<std::size_t>(width) * height * desiredChannels;
        img.pixels32f_.assign(raw.get(), raw.get() + pixelCount);
        return img;
    }

    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw{
        stbi_load(path.c_str(), &width, &height, &fileChannels, desiredChannels), stbi_image_free};
    if (raw == nullptr)
    {
        throw std::runtime_error("Failed to load image: " + path + " (" + stbFailureDetail() + ")");
    }

    img.width_ = width;
    img.height_ = height;
    img.channels_ = desiredChannels;
    img.pixelType_ = ImagePixelType::Uint8;
    std::size_t byteCount = static_cast<std::size_t>(width) * height * desiredChannels;
    img.pixels8_.assign(raw.get(), raw.get() + byteCount);
    return img;
}

Image Image::load_from_memory(const uint8_t* data, std::size_t size_bytes, const std::string& label)
{
    int width = 0;
    int height = 0;
    int fileChannels = 0;

    Image img;

    if (isHdrMemory(data, size_bytes))
    {
        std::unique_ptr<float, decltype(&stbi_image_free)> raw{
            stbi_loadf_from_memory(data, static_cast<int>(size_bytes), &width, &height,
                                   &fileChannels, desiredChannels),
            stbi_image_free};
        if (raw == nullptr)
        {
            throw std::runtime_error("Failed to decode image: " + label + " (" +
                                     stbFailureDetail() + ")");
        }

        img.width_ = width;
        img.height_ = height;
        img.channels_ = desiredChannels;
        img.pixelType_ = ImagePixelType::Float32;
        std::size_t pixelCount = static_cast<std::size_t>(width) * height * desiredChannels;
        img.pixels32f_.assign(raw.get(), raw.get() + pixelCount);
        return img;
    }

    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw{
        stbi_load_from_memory(data, static_cast<int>(size_bytes), &width, &height, &fileChannels,
                              desiredChannels),
        stbi_image_free};
    if (raw == nullptr)
    {
        throw std::runtime_error("Failed to decode image: " + label + " (" + stbFailureDetail() +
                                 ")");
    }

    img.width_ = width;
    img.height_ = height;
    img.channels_ = desiredChannels;
    img.pixelType_ = ImagePixelType::Uint8;
    std::size_t byteCount = static_cast<std::size_t>(width) * height * desiredChannels;
    img.pixels8_.assign(raw.get(), raw.get() + byteCount);
    return img;
}

Image::Image(Image&& other) noexcept
    : width_(other.width_),
      height_(other.height_),
      channels_(other.channels_),
      pixelType_(other.pixelType_),
      pixels8_(std::move(other.pixels8_)),
      pixels32f_(std::move(other.pixels32f_))
{
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.pixelType_ = ImagePixelType::Uint8;
}

Image& Image::operator=(Image&& other) noexcept
{
    if (this != &other)
    {
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        pixelType_ = other.pixelType_;
        pixels8_ = std::move(other.pixels8_);
        pixels32f_ = std::move(other.pixels32f_);
        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
        other.pixelType_ = ImagePixelType::Uint8;
    }
    return *this;
}

} // namespace fire_engine
