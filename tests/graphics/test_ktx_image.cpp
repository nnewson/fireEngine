#include <fire_engine/graphics/ktx_image.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
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
    REQUIRE(ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture) ==
            KTX_SUCCESS);
    ScopedKtxTexture2 owned(texture);

    const std::array<uint8_t, 16> pixels{
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    };

    REQUIRE(ktxTexture_SetImageFromMemory(ktxTexture(texture), 0, 0, 0, pixels.data(),
                                          pixels.size()) == KTX_SUCCESS);
    REQUIRE(ktxTexture_WriteToNamedFile(ktxTexture(texture), path.string().c_str()) == KTX_SUCCESS);
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

TEST_CASE("KtxImageConstruction.DefaultIsEmpty", "[KtxImageConstruction]")
{
    KtxImage image;
    CHECK(image.width() == 0u);
    CHECK(image.height() == 0u);
    CHECK(image.depth() == 0u);
    CHECK(image.dimensions() == 0u);
    CHECK(image.levels() == 0u);
    CHECK(image.layers() == 0u);
    CHECK(image.faces() == 0u);
    CHECK(image.element_size() == 0u);
    CHECK(image.size_bytes() == 0u);
    CHECK(image.data() == nullptr);
    CHECK(image.empty());
    CHECK_FALSE(image.compressed());
    CHECK_FALSE(image.needsTranscoding());
    CHECK_FALSE(image.isKtx2());
    CHECK(image.vkFormat() == 0u);
}

TEST_CASE("KtxImageTraits.IsNonCopyable", "[KtxImageTraits]")
{
    static_assert(!std::is_copy_constructible_v<KtxImage>);
    static_assert(!std::is_copy_assignable_v<KtxImage>);
}

TEST_CASE("KtxImageTraits.IsNothrowMovable", "[KtxImageTraits]")
{
    static_assert(std::is_nothrow_move_constructible_v<KtxImage>);
    static_assert(std::is_nothrow_move_assignable_v<KtxImage>);
}

TEST_CASE("KtxImageLoading.LoadValidKtx2FromFile", "[KtxImageLoading]")
{
    const auto path = testKtx2Path();
    createTestKtx2File(path);

    KtxImage image = KtxImage::load_from_file(path.string());

    REQUIRE_FALSE(image.empty());
    CHECK(image.width() == 2u);
    CHECK(image.height() == 2u);
    CHECK(image.depth() == 1u);
    CHECK(image.dimensions() == 2u);
    CHECK(image.levels() == 1u);
    CHECK(image.layers() == 1u);
    CHECK(image.faces() == 1u);
    CHECK_FALSE(image.array());
    CHECK_FALSE(image.cubemap());
    CHECK_FALSE(image.compressed());
    CHECK_FALSE(image.needsTranscoding());
    CHECK(image.isKtx2());
    CHECK(image.vkFormat() == static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_UNORM));
    CHECK(image.element_size() == 4u);
    CHECK(image.size_bytes() == 16u);
    REQUIRE(image.data() != nullptr);
    CHECK(image.data()[0] == 255u);
    CHECK(image.data()[1] == 0u);
    CHECK(image.data()[2] == 0u);
    CHECK(image.data()[3] == 255u);

    std::filesystem::remove(path);
}

TEST_CASE("KtxImageLoading.LoadValidKtx2FromMemory", "[KtxImageLoading]")
{
    const auto path = testKtx2Path();
    createTestKtx2File(path);

    const std::vector<uint8_t> bytes = readBytes(path);
    KtxImage image = KtxImage::load_from_memory(bytes.data(), bytes.size(), "test-ktx2");

    REQUIRE_FALSE(image.empty());
    CHECK(image.width() == 2u);
    CHECK(image.height() == 2u);
    CHECK(image.isKtx2());
    CHECK(image.vkFormat() == static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_UNORM));
    CHECK(image.size_bytes() == 16u);

    std::filesystem::remove(path);
}

TEST_CASE("KtxImageLoading.NonExistentFileThrows", "[KtxImageLoading]")
{
    CHECK_THROWS_AS(KtxImage::load_from_file("test_assets/nonexistent.ktx2"), std::runtime_error);
}

TEST_CASE("KtxImageLoading.InvalidMemoryThrows", "[KtxImageLoading]")
{
    const std::array<uint8_t, 4> bytes{0, 1, 2, 3};
    CHECK_THROWS_AS(KtxImage::load_from_memory(bytes.data(), bytes.size(), "invalid"),
                    std::runtime_error);
}

TEST_CASE("KtxImageMove.MoveConstructionTransfersState", "[KtxImageMove]")
{
    const auto path = testKtx2Path();
    createTestKtx2File(path);

    KtxImage original = KtxImage::load_from_file(path.string());
    const uint8_t* originalData = original.data();

    KtxImage moved(std::move(original));

    CHECK(moved.width() == 2u);
    CHECK(moved.data() == originalData);
    CHECK(original.empty());
    CHECK(original.data() == nullptr);

    std::filesystem::remove(path);
}

TEST_CASE("KtxImageMove.MoveAssignmentTransfersState", "[KtxImageMove]")
{
    const auto path = testKtx2Path();
    createTestKtx2File(path);

    KtxImage original = KtxImage::load_from_file(path.string());
    KtxImage target;

    target = std::move(original);

    CHECK(target.width() == 2u);
    CHECK_FALSE(target.empty());
    CHECK(original.empty());

    std::filesystem::remove(path);
}
