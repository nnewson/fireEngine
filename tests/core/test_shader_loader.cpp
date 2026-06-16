#include <fire_engine/core/shader_loader.hpp>

#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::ShaderLoader;

// ==========================================================================
// load_from_file
// ==========================================================================

TEST_CASE("ShaderLoaderLoadFromFile.LoadsFileData", "[ShaderLoaderLoadFromFile]")
{
    auto data = ShaderLoader::load_from_file("test_assets/test_data.bin");
    CHECK_FALSE(data.empty());
}

TEST_CASE("ShaderLoaderLoadFromFile.DataMatchesFileContents", "[ShaderLoaderLoadFromFile]")
{
    auto data = ShaderLoader::load_from_file("test_assets/test_data.bin");
    std::string content(data.begin(), data.end());
    CHECK(content.find("HELLO") != std::string::npos);
}

TEST_CASE("ShaderLoaderLoadFromFile.SizeMatchesFileSize", "[ShaderLoaderLoadFromFile]")
{
    auto data = ShaderLoader::load_from_file("test_assets/test_data.bin");
    CHECK(data.size() > 0u);
}

TEST_CASE("ShaderLoaderLoadFromFile.NonExistentFileThrows", "[ShaderLoaderLoadFromFile]")
{
    CHECK_THROWS_AS(static_cast<void>(ShaderLoader::load_from_file("test_assets/nonexistent.spv")),
                    std::runtime_error);
}

TEST_CASE("ShaderLoaderLoadFromFile.TwoLoadsReturnSameData", "[ShaderLoaderLoadFromFile]")
{
    auto a = ShaderLoader::load_from_file("test_assets/test_data.bin");
    auto b = ShaderLoader::load_from_file("test_assets/test_data.bin");
    CHECK(a == b);
}
