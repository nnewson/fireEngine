#pragma once

#include <algorithm>
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
};

} // namespace fire_engine
