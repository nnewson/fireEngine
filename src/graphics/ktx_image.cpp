#include <fire_engine/graphics/ktx_image.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace fire_engine
{

namespace
{

constexpr ktxTextureCreateFlags createFlags = KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT;

[[nodiscard]] std::string ktxFailureDetail(KTX_error_code error)
{
    const char* reason = ktxErrorString(error);
    if (reason == nullptr || reason[0] == '\0')
    {
        return "unknown libktx error";
    }

    return reason;
}

void throwOnKtxError(KTX_error_code error, const std::string& action, const std::string& label)
{
    if (error != KTX_SUCCESS)
    {
        throw std::runtime_error("Failed to " + action + " KTX image: " + label + " (" +
                                 ktxFailureDetail(error) + ")");
    }
}

} // namespace

KtxImage KtxImage::load_from_file(const std::string& path)
{
    ktxTexture* texture = nullptr;
    throwOnKtxError(ktxTexture_CreateFromNamedFile(path.c_str(), createFlags, &texture), "load",
                    path);
    return KtxImage(texture);
}

KtxImage KtxImage::load_from_memory(const uint8_t* data, std::size_t size_bytes,
                                    const std::string& label)
{
    ktxTexture* texture = nullptr;
    throwOnKtxError(ktxTexture_CreateFromMemory(data, static_cast<ktx_size_t>(size_bytes),
                                                createFlags, &texture),
                    "decode", label);
    return KtxImage(texture);
}

KtxImage::~KtxImage()
{
    reset();
}

KtxImage::KtxImage(KtxImage&& other) noexcept
    : texture_(std::exchange(other.texture_, nullptr))
{
}

KtxImage& KtxImage::operator=(KtxImage&& other) noexcept
{
    if (this != &other)
    {
        reset();
        texture_ = std::exchange(other.texture_, nullptr);
    }

    return *this;
}

bool KtxImage::isKtx2() const noexcept
{
    return texture_ != nullptr && texture_->classId == ktxTexture2_c;
}

bool KtxImage::needsTranscoding() const noexcept
{
    return texture_ != nullptr && ktxTexture_NeedsTranscoding(texture_) != 0;
}

uint32_t KtxImage::vkFormat() const noexcept
{
    if (!isKtx2())
    {
        return 0u;
    }

    return reinterpret_cast<const ktxTexture2*>(texture_)->vkFormat;
}

uint32_t KtxImage::element_size() const noexcept
{
    return texture_ == nullptr ? 0u : ktxTexture_GetElementSize(texture_);
}

std::size_t KtxImage::size_bytes() const noexcept
{
    return texture_ == nullptr ? 0u : static_cast<std::size_t>(ktxTexture_GetDataSize(texture_));
}

const uint8_t* KtxImage::data() const noexcept
{
    return texture_ == nullptr ? nullptr : ktxTexture_GetData(texture_);
}

void KtxImage::reset() noexcept
{
    if (texture_ != nullptr)
    {
        ktxTexture_Destroy(texture_);
        texture_ = nullptr;
    }
}

} // namespace fire_engine
