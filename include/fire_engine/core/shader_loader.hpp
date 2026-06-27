#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fire_engine
{

class ShaderLoader
{
public:
    ShaderLoader() = delete;

    [[nodiscard]]
    static std::vector<char> load_from_file(const std::string& path);

    [[nodiscard]]
    static std::vector<std::uint32_t> load_spirv_from_file(const std::string& path);
};

} // namespace fire_engine
