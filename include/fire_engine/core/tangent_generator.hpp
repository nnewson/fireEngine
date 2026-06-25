#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

struct TangentGenerationResult
{
    bool succeeded{false};
    std::string reason{};
    std::vector<Vec4> tangents{};
};

class TangentGenerator
{
public:
    TangentGenerator() = delete;

    [[nodiscard]]
    static TangentGenerationResult
    generate(std::span<const Vec3> positions, std::span<const Vec3> normals,
             std::span<const Vec2> texcoords, std::span<const uint32_t> indices);
};

} // namespace fire_engine
