#pragma once

#include <algorithm>
#include <array>
#include <cmath>
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

    // `expand` that cannot SWALLOW a corrupt point. Returns false when `p` is non-finite, and
    // leaves the box unchanged.
    //
    // Plain `expand` uses std::min / std::max, which return the OTHER operand when one side is NaN
    // — so a NaN vertex silently leaves finite bounds that do not contain the geometry they claim
    // to. Anything that FITS to bounds (SH-06's cascade depth range) has to be able to tell "no
    // vertices" from "a vertex nobody can bound": the first contributes nothing, the second must
    // stop the fit, because a range tightened around geometry it never accounted for clips it.
    [[nodiscard]] bool expandChecked(Vec3 p) noexcept
    {
        if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
        {
            return false;
        }
        expand(p);
        return true;
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
