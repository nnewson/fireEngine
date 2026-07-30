#pragma once

#include <array>
#include <cstddef>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Vulkan cubemap face order: +X, -X, +Y, -Y, +Z, -Z.
// Forward and up vectors below match the IBL prefilter convention used in the
// environment precompute pass so cube sampling stays consistent across face
// boundaries — anything generating per-face view matrices for a cubemap render
// should pull from this single source.

// The count lives in graphics/gpu_limits.hpp (kCubeFaceCount) — it is shared with shadow key
// validation, matrix indexing and layer indexing, and a second definition here would be free to
// drift while every index stayed plausibly in range. This alias keeps the render-side spelling.
inline constexpr std::size_t kCubemapFaceCount = kCubeFaceCount;

inline constexpr std::array<Vec3, kCubemapFaceCount> kCubemapFaceForward{
    Vec3{1.0f, 0.0f, 0.0f},  Vec3{-1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
    Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f},  Vec3{0.0f, 0.0f, -1.0f},
};

inline constexpr std::array<Vec3, kCubemapFaceCount> kCubemapFaceUp{
    Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f},
    Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f},
};

} // namespace fire_engine
