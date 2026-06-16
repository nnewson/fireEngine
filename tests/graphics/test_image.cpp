#include <fire_engine/graphics/image.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Image;

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("ImageConstruction.DefaultIsEmpty", "[ImageConstruction]")
{
    Image img;
    CHECK(img.width() == 0);
    CHECK(img.height() == 0);
    CHECK(img.channels() == 0);
    CHECK(img.size_bytes() == 0u);
    CHECK(img.empty());
    CHECK(img.data() == nullptr);
    CHECK(img.dataf() == nullptr);
    CHECK(img.pixelType() == fire_engine::ImagePixelType::Uint8);
}

// ==========================================================================
// Loading
// ==========================================================================

TEST_CASE("ImageLoading.LoadValidPng", "[ImageLoading]")
{
    Image img = Image::load_from_file("test_assets/test_2x2.png");
    CHECK(img.width() == 2);
    CHECK(img.height() == 2);
    CHECK(img.channels() == 4);
    CHECK_FALSE(img.empty());
    CHECK(img.pixelType() == fire_engine::ImagePixelType::Uint8);
}

TEST_CASE("ImageLoading.SizeBytesMatchesDimensions", "[ImageLoading]")
{
    Image img = Image::load_from_file("test_assets/test_2x2.png");
    std::size_t expected = static_cast<std::size_t>(img.width()) * img.height() * img.channels();
    CHECK(img.size_bytes() == expected);
}

TEST_CASE("ImageLoading.AlwaysLoadsAsRGBA", "[ImageLoading]")
{
    Image img = Image::load_from_file("test_assets/test_2x2.png");
    CHECK(img.channels() == 4);
}

TEST_CASE("ImageLoading.DataIsNotNull", "[ImageLoading]")
{
    Image img = Image::load_from_file("test_assets/test_2x2.png");
    CHECK(img.data() != nullptr);
    CHECK(img.dataf() == nullptr);
}

TEST_CASE("ImageLoading.LoadValidPngFromMemory", "[ImageLoading]")
{
    std::ifstream file("test_assets/test_2x2.png", std::ios::binary | std::ios::ate);
    REQUIRE(file.is_open());

    auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(size);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    REQUIRE((file.good() || file.eof()));

    Image img = Image::load_from_memory(bytes.data(), bytes.size(), "embedded-test");
    CHECK(img.width() == 2);
    CHECK(img.height() == 2);
    CHECK(img.channels() == 4);
    CHECK_FALSE(img.empty());
}

TEST_CASE("ImageLoading.LoadHdrEnvironmentAsFloatImage", "[ImageLoading]")
{
    REQUIRE(std::filesystem::exists("skybox.hdr"));

    Image img = Image::load_from_file("skybox.hdr");
    CHECK(img.width() > 0);
    CHECK(img.height() > 0);
    CHECK(img.channels() == 4);
    CHECK_FALSE(img.empty());
    CHECK(img.pixelType() == fire_engine::ImagePixelType::Float32);
    CHECK(img.data() == nullptr);
    CHECK(img.dataf() != nullptr);
}

TEST_CASE("ImageLoading.PixelDataIsCorrect", "[ImageLoading]")
{
    Image img = Image::load_from_file("test_assets/test_2x2.png");
    const uint8_t* d = img.data();

    // Row 0, Col 0: red (255, 0, 0, 255)
    CHECK(d[0] == 255);
    CHECK(d[1] == 0);
    CHECK(d[2] == 0);
    CHECK(d[3] == 255);

    // Row 0, Col 1: green (0, 255, 0, 255)
    CHECK(d[4] == 0);
    CHECK(d[5] == 255);
    CHECK(d[6] == 0);
    CHECK(d[7] == 255);

    // Row 1, Col 0: blue (0, 0, 255, 255)
    CHECK(d[8] == 0);
    CHECK(d[9] == 0);
    CHECK(d[10] == 255);
    CHECK(d[11] == 255);

    // Row 1, Col 1: white (255, 255, 255, 255)
    CHECK(d[12] == 255);
    CHECK(d[13] == 255);
    CHECK(d[14] == 255);
    CHECK(d[15] == 255);
}

TEST_CASE("ImageLoading.NonExistentFileThrows", "[ImageLoading]")
{
    CHECK_THROWS_AS(Image::load_from_file("test_assets/nonexistent.png"), std::runtime_error);
}

TEST_CASE("ImageLoading.InvalidFileThrows", "[ImageLoading]")
{
    CHECK_THROWS_AS(Image::load_from_file("test_assets/"), std::runtime_error);
}

// ==========================================================================
// Move Semantics
// ==========================================================================

TEST_CASE("ImageMove.MoveConstructTransfersData", "[ImageMove]")
{
    Image a = Image::load_from_file("test_assets/test_2x2.png");
    const uint8_t* originalData = a.data();
    int originalWidth = a.width();

    Image b{std::move(a)};

    CHECK(b.width() == originalWidth);
    CHECK(b.data() == originalData);
    CHECK_FALSE(b.empty());
}

TEST_CASE("ImageMove.MoveConstructZerosSource", "[ImageMove]")
{
    Image a = Image::load_from_file("test_assets/test_2x2.png");
    Image b{std::move(a)};

    CHECK(a.width() == 0);
    CHECK(a.height() == 0);
    CHECK(a.channels() == 0);
    CHECK(a.empty());
}

TEST_CASE("ImageMove.MoveAssignTransfersData", "[ImageMove]")
{
    Image a = Image::load_from_file("test_assets/test_2x2.png");
    int originalWidth = a.width();

    Image b;
    b = std::move(a);

    CHECK(b.width() == originalWidth);
    CHECK_FALSE(b.empty());
    CHECK(a.width() == 0);
    CHECK(a.empty());
}

TEST_CASE("ImageMove.MoveAssignSelfIsNoOp", "[ImageMove]")
{
    Image a = Image::load_from_file("test_assets/test_2x2.png");
    int originalWidth = a.width();

    auto& ref = a;
    a = std::move(ref);

    CHECK(a.width() == originalWidth);
    CHECK_FALSE(a.empty());
}

// ==========================================================================
// Copy Semantics
// ==========================================================================

TEST_CASE("ImageCopy.CopyConstructCreatesIndependentCopy", "[ImageCopy]")
{
    Image a = Image::load_from_file("test_assets/test_2x2.png");
    Image b{a};

    CHECK(b.width() == a.width());
    CHECK(b.height() == a.height());
    CHECK(b.channels() == a.channels());
    CHECK(b.size_bytes() == a.size_bytes());
    CHECK(b.data() != a.data());

    // Pixel data should match
    for (std::size_t i = 0; i < a.size_bytes(); ++i)
    {
        INFO("mismatch at byte " << i);
        CHECK(b.data()[i] == a.data()[i]);
    }
}

TEST_CASE("ImageCopy.CopyAssignCreatesIndependentCopy", "[ImageCopy]")
{
    Image a = Image::load_from_file("test_assets/test_2x2.png");
    Image b;
    b = a;

    CHECK(b.width() == a.width());
    CHECK(b.data() != a.data());
    CHECK_FALSE(a.empty());
}

// ==========================================================================
// Multiple Loads
// ==========================================================================

TEST_CASE("ImageMultipleLoads.TwoLoadsAreIndependent", "[ImageMultipleLoads]")
{
    Image a = Image::load_from_file("test_assets/test_2x2.png");
    Image b = Image::load_from_file("test_assets/test_2x2.png");

    CHECK(a.width() == b.width());
    CHECK(a.size_bytes() == b.size_bytes());
    CHECK(a.data() != b.data());
}
