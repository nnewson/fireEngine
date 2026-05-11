#pragma once

#include <array>
#include <cmath>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

struct ViewBasis
{
    Vec3 forward;
    Vec3 right;
    Vec3 up;
};

[[nodiscard]]
inline Vec3 normaliseOr(Vec3 value, Vec3 fallback) noexcept
{
    if (value.magnitudeSquared() < float_epsilon * float_epsilon)
    {
        return Vec3::normalise(fallback);
    }
    return Vec3::normalise(value);
}

[[nodiscard]]
inline Vec3 stableUpForForward(Vec3 forward, Vec3 preferredUp = {0.0f, 1.0f, 0.0f}) noexcept
{
    const Vec3 unitForward = normaliseOr(forward, {0.0f, 0.0f, -1.0f});
    const std::array<Vec3, 4> candidates{
        normaliseOr(preferredUp, {0.0f, 1.0f, 0.0f}),
        Vec3{0.0f, 1.0f, 0.0f},
        Vec3{0.0f, 0.0f, 1.0f},
        Vec3{1.0f, 0.0f, 0.0f},
    };

    for (const Vec3& candidate : candidates)
    {
        if (std::abs(Vec3::dotProduct(unitForward, candidate)) < 0.99f)
        {
            return candidate;
        }
    }

    return Vec3{0.0f, 1.0f, 0.0f};
}

[[nodiscard]]
inline ViewBasis makeViewBasis(Vec3 eye, Vec3 target, Vec3 preferredUp = {0.0f, 1.0f, 0.0f},
                               Vec3 fallbackForward = {0.0f, 0.0f, -1.0f}) noexcept
{
    const Vec3 forward = normaliseOr(target - eye, fallbackForward);
    const Vec3 upSeed = stableUpForForward(forward, preferredUp);
    const Vec3 right = normaliseOr(Vec3::crossProduct(forward, upSeed), {1.0f, 0.0f, 0.0f});
    const Vec3 up = normaliseOr(Vec3::crossProduct(right, forward), upSeed);
    return {forward, right, up};
}

} // namespace fire_engine
