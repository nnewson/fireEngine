#include <fire_engine/core/shader_loader.hpp>

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace fire_engine
{

namespace
{

constexpr std::uint32_t kSpirvMagic = 0x07230203;
constexpr std::size_t kSpirvHeaderWordCount = 5;

[[nodiscard]] std::runtime_error loadError(const std::string& message, const std::string& path)
{
    return std::runtime_error(message + ": " + path);
}

} // namespace

std::vector<char> ShaderLoader::load_from_file(const std::string& path)
{
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open())
    {
        throw loadError("failed to open file", path);
    }

    const std::ifstream::pos_type end = f.tellg();
    if (end == std::ifstream::pos_type{-1})
    {
        throw loadError("failed to determine file size", path);
    }

    const auto size = static_cast<std::streamoff>(end);
    if (size < 0)
    {
        throw loadError("invalid file size", path);
    }
    if (static_cast<std::uintmax_t>(size) >
        static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
    {
        throw loadError("file is too large to read", path);
    }

    std::vector<char> buf(static_cast<std::size_t>(size));
    if (!f.seekg(0, std::ios::beg))
    {
        throw loadError("failed to seek file", path);
    }
    if (!buf.empty() && !f.read(buf.data(), static_cast<std::streamsize>(buf.size())))
    {
        throw loadError("failed to read complete file", path);
    }
    return buf;
}

std::vector<std::uint32_t> ShaderLoader::load_spirv_from_file(const std::string& path)
{
    const std::vector<char> bytes = load_from_file(path);
    if (bytes.empty())
    {
        throw loadError("SPIR-V file is empty", path);
    }
    if (bytes.size() % sizeof(std::uint32_t) != 0)
    {
        throw loadError("SPIR-V file size is not 32-bit aligned", path);
    }

    std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    if (words.size() < kSpirvHeaderWordCount)
    {
        throw loadError("SPIR-V file is smaller than the header", path);
    }
    if (words.front() != kSpirvMagic)
    {
        throw loadError("SPIR-V magic number mismatch", path);
    }
    return words;
}

} // namespace fire_engine
