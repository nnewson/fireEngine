#include <fire_engine/graphics/ktx_image.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

using fire_engine::KtxImage;

namespace
{

class ScopedKtxTexture2
{
public:
    explicit ScopedKtxTexture2(ktxTexture2* texture) noexcept
        : texture_(texture)
    {
    }

    ~ScopedKtxTexture2()
    {
        if (texture_ != nullptr)
        {
            ktxTexture2_Destroy(texture_);
        }
    }

    ScopedKtxTexture2(const ScopedKtxTexture2&) = delete;
    ScopedKtxTexture2& operator=(const ScopedKtxTexture2&) = delete;

    [[nodiscard]]
    ktxTexture2* get() const noexcept
    {
        return texture_;
    }

private:
    ktxTexture2* texture_{nullptr};
};

[[nodiscard]] std::filesystem::path testKtx2Path()
{
    return std::filesystem::temp_directory_path() / "fire_engine_test_2x2.ktx2";
}

void createTestKtx2File(const std::filesystem::path& path)
{
    const ktxTextureCreateInfo createInfo{
        .glInternalformat = 0,
        .vkFormat = VK_FORMAT_R8G8B8A8_UNORM,
        .pDfd = nullptr,
        .baseWidth = 2,
        .baseHeight = 2,
        .baseDepth = 1,
        .numDimensions = 2,
        .numLevels = 1,
        .numLayers = 1,
        .numFaces = 1,
        .isArray = KTX_FALSE,
        .generateMipmaps = KTX_FALSE,
    };

    ktxTexture2* texture = nullptr;
    ASSERT_EQ(ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture),
              KTX_SUCCESS);
    ScopedKtxTexture2 owned(texture);

    const std::array<uint8_t, 16> pixels{
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    };

    ASSERT_EQ(
        ktxTexture_SetImageFromMemory(ktxTexture(texture), 0, 0, 0, pixels.data(), pixels.size()),
        KTX_SUCCESS);
    ASSERT_EQ(ktxTexture_WriteToNamedFile(ktxTexture(texture), path.string().c_str()), KTX_SUCCESS);
}

[[nodiscard]] std::vector<uint8_t> readBytes(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open KTX test file");
    }

    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(size);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!file.good() && !file.eof())
    {
        throw std::runtime_error("Failed to read KTX test file");
    }

    return bytes;
}

} // namespace

TEST(KtxImageConstruction, DefaultIsEmpty)
{
    KtxImage image;
    EXPECT_EQ(image.width(), 0u);
    EXPECT_EQ(image.height(), 0u);
    EXPECT_EQ(image.depth(), 0u);
    EXPECT_EQ(image.dimensions(), 0u);
    EXPECT_EQ(image.levels(), 0u);
    EXPECT_EQ(image.layers(), 0u);
    EXPECT_EQ(image.faces(), 0u);
    EXPECT_EQ(image.element_size(), 0u);
    EXPECT_EQ(image.size_bytes(), 0u);
    EXPECT_EQ(image.data(), nullptr);
    EXPECT_TRUE(image.empty());
    EXPECT_FALSE(image.compressed());
    EXPECT_FALSE(image.needsTranscoding());
    EXPECT_FALSE(image.isKtx2());
    EXPECT_EQ(image.vkFormat(), 0u);
}

TEST(KtxImageTraits, IsNonCopyable)
{
    EXPECT_FALSE(std::is_copy_constructible_v<KtxImage>);
    EXPECT_FALSE(std::is_copy_assignable_v<KtxImage>);
}

TEST(KtxImageTraits, IsNothrowMovable)
{
    EXPECT_TRUE(std::is_nothrow_move_constructible_v<KtxImage>);
    EXPECT_TRUE(std::is_nothrow_move_assignable_v<KtxImage>);
}

TEST(KtxImageLoading, LoadValidKtx2FromFile)
{
    const auto path = testKtx2Path();
    createTestKtx2File(path);

    KtxImage image = KtxImage::load_from_file(path.string());

    ASSERT_FALSE(image.empty());
    EXPECT_EQ(image.width(), 2u);
    EXPECT_EQ(image.height(), 2u);
    EXPECT_EQ(image.depth(), 1u);
    EXPECT_EQ(image.dimensions(), 2u);
    EXPECT_EQ(image.levels(), 1u);
    EXPECT_EQ(image.layers(), 1u);
    EXPECT_EQ(image.faces(), 1u);
    EXPECT_FALSE(image.array());
    EXPECT_FALSE(image.cubemap());
    EXPECT_FALSE(image.compressed());
    EXPECT_FALSE(image.needsTranscoding());
    EXPECT_TRUE(image.isKtx2());
    EXPECT_EQ(image.vkFormat(), static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_UNORM));
    EXPECT_EQ(image.element_size(), 4u);
    EXPECT_EQ(image.size_bytes(), 16u);
    ASSERT_NE(image.data(), nullptr);
    EXPECT_EQ(image.data()[0], 255u);
    EXPECT_EQ(image.data()[1], 0u);
    EXPECT_EQ(image.data()[2], 0u);
    EXPECT_EQ(image.data()[3], 255u);

    std::filesystem::remove(path);
}

TEST(KtxImageLoading, LoadValidKtx2FromMemory)
{
    const auto path = testKtx2Path();
    createTestKtx2File(path);

    const std::vector<uint8_t> bytes = readBytes(path);
    KtxImage image = KtxImage::load_from_memory(bytes.data(), bytes.size(), "test-ktx2");

    ASSERT_FALSE(image.empty());
    EXPECT_EQ(image.width(), 2u);
    EXPECT_EQ(image.height(), 2u);
    EXPECT_TRUE(image.isKtx2());
    EXPECT_EQ(image.vkFormat(), static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_UNORM));
    EXPECT_EQ(image.size_bytes(), 16u);

    std::filesystem::remove(path);
}

TEST(KtxImageLoading, NonExistentFileThrows)
{
    EXPECT_THROW(KtxImage::load_from_file("test_assets/nonexistent.ktx2"), std::runtime_error);
}

TEST(KtxImageLoading, InvalidMemoryThrows)
{
    const std::array<uint8_t, 4> bytes{0, 1, 2, 3};
    EXPECT_THROW(KtxImage::load_from_memory(bytes.data(), bytes.size(), "invalid"),
                 std::runtime_error);
}

TEST(KtxImageMove, MoveConstructionTransfersState)
{
    const auto path = testKtx2Path();
    createTestKtx2File(path);

    KtxImage original = KtxImage::load_from_file(path.string());
    const uint8_t* originalData = original.data();

    KtxImage moved(std::move(original));

    EXPECT_EQ(moved.width(), 2u);
    EXPECT_EQ(moved.data(), originalData);
    EXPECT_TRUE(original.empty());
    EXPECT_EQ(original.data(), nullptr);

    std::filesystem::remove(path);
}

TEST(KtxImageMove, MoveAssignmentTransfersState)
{
    const auto path = testKtx2Path();
    createTestKtx2File(path);

    KtxImage original = KtxImage::load_from_file(path.string());
    KtxImage target;

    target = std::move(original);

    EXPECT_EQ(target.width(), 2u);
    EXPECT_FALSE(target.empty());
    EXPECT_TRUE(original.empty());

    std::filesystem::remove(path);
}
