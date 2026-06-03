#pragma once

#include <array>
#include <cstddef>

#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Vulkan cubemap face order: +X, -X, +Y, -Y, +Z, -Z.
// Forward and up vectors below match the IBL prefilter convention used in the
// environment precompute pass so cube sampling stays consistent across face
// boundaries — anything generating per-face view matrices for a cubemap render
// should pull from this single source.

inline constexpr std::size_t kCubemapFaceCount = 6;

inline constexpr std::array<Vec3, kCubemapFaceCount> kCubemapFaceForward{
    Vec3{1.0f, 0.0f, 0.0f},  Vec3{-1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
    Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f},  Vec3{0.0f, 0.0f, -1.0f},
};

inline constexpr std::array<Vec3, kCubemapFaceCount> kCubemapFaceUp{
    Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f},
    Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f},
};

} // namespace fire_engine
