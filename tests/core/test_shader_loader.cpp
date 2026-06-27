#include <fire_engine/core/shader_loader.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::ShaderLoader;

namespace
{

[[nodiscard]] std::filesystem::path tempShaderPath(const std::string& name)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("fire_engine_shader_loader_" + std::to_string(stamp) + "_" + name);
}

void writeBytes(const std::filesystem::path& path, const std::vector<char>& bytes)
{
    std::ofstream out(path, std::ios::binary);
    if (!bytes.empty())
    {
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
}

void writeWords(const std::filesystem::path& path, std::span<const std::uint32_t> words)
{
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
}

} // namespace

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

TEST_CASE("ShaderLoaderLoadSpirvFromFile.ValidHeaderLoadsWords", "[ShaderLoaderLoadSpirvFromFile]")
{
    const std::filesystem::path path = tempShaderPath("valid.spv");
    const std::array words{0x07230203u, 0x00010000u, 0u, 0u, 0u};
    writeWords(path, words);

    const std::vector<std::uint32_t> loaded = ShaderLoader::load_spirv_from_file(path.string());

    CHECK(loaded == std::vector<std::uint32_t>{words.begin(), words.end()});
}

TEST_CASE("ShaderLoaderLoadSpirvFromFile.NonExistentFileThrows", "[ShaderLoaderLoadSpirvFromFile]")
{
    CHECK_THROWS_AS(
        static_cast<void>(ShaderLoader::load_spirv_from_file("test_assets/nonexistent.spv")),
        std::runtime_error);
}

TEST_CASE("ShaderLoaderLoadSpirvFromFile.EmptyFileThrows", "[ShaderLoaderLoadSpirvFromFile]")
{
    const std::filesystem::path path = tempShaderPath("empty.spv");
    writeBytes(path, {});

    CHECK_THROWS_AS(static_cast<void>(ShaderLoader::load_spirv_from_file(path.string())),
                    std::runtime_error);
}

TEST_CASE("ShaderLoaderLoadSpirvFromFile.UnalignedSizeThrows", "[ShaderLoaderLoadSpirvFromFile]")
{
    CHECK_THROWS_AS(
        static_cast<void>(ShaderLoader::load_spirv_from_file("test_assets/test_data.bin")),
        std::runtime_error);
}

TEST_CASE("ShaderLoaderLoadSpirvFromFile.ShortHeaderThrows", "[ShaderLoaderLoadSpirvFromFile]")
{
    const std::filesystem::path path = tempShaderPath("short.spv");
    const std::array words{0x07230203u};
    writeWords(path, words);

    CHECK_THROWS_AS(static_cast<void>(ShaderLoader::load_spirv_from_file(path.string())),
                    std::runtime_error);
}

TEST_CASE("ShaderLoaderLoadSpirvFromFile.BadMagicThrows", "[ShaderLoaderLoadSpirvFromFile]")
{
    const std::filesystem::path path = tempShaderPath("bad_magic.spv");
    const std::array words{0u, 0x00010000u, 0u, 0u, 0u};
    writeWords(path, words);

    CHECK_THROWS_AS(static_cast<void>(ShaderLoader::load_spirv_from_file(path.string())),
                    std::runtime_error);
}
