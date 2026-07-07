#pragma once

#include <algorithm>
#include <array>
#include <limits>

#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

struct Bounds3
{
    Vec3 min{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()};
    Vec3 max{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
             std::numeric_limits<float>::lowest()};
    bool valid{false};

    void expand(Vec3 p) noexcept
    {
        if (!valid)
        {
            min = p;
            max = p;
            valid = true;
            return;
        }

        min = {std::min(min.x(), p.x()), std::min(min.y(), p.y()), std::min(min.z(), p.z())};
        max = {std::max(max.x(), p.x()), std::max(max.y(), p.y()), std::max(max.z(), p.z())};
    }

    [[nodiscard]] Vec3 center() const noexcept
    {
        return (min + max) * 0.5f;
    }

    [[nodiscard]] Vec3 extent() const noexcept
    {
        return max - min;
    }

    // The eight corners, taking min/max per bit of the index (bit 0 -> x, 1 -> y, 2 -> z).
    [[nodiscard]] std::array<Vec3, 8> corners() const noexcept
    {
        return {Vec3{min.x(), min.y(), min.z()}, Vec3{max.x(), min.y(), min.z()},
                Vec3{min.x(), max.y(), min.z()}, Vec3{max.x(), max.y(), min.z()},
                Vec3{min.x(), min.y(), max.z()}, Vec3{max.x(), min.y(), max.z()},
                Vec3{min.x(), max.y(), max.z()}, Vec3{max.x(), max.y(), max.z()}};
    }
};

} // namespace fire_engine
