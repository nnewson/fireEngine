#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

enum class Axis : std::uint8_t
{
    X,
    Y,
    Z,
};

// Axis-aligned bounding box. Pure value type — no incremental-construction
// state; callers that want to grow a box point-by-point should use
// graphics::Bounds3 instead.
struct AABB
{
    Vec3 min;
    Vec3 max;

    [[nodiscard]] constexpr float axisMin(Axis axis) const noexcept
    {
        switch (axis)
        {
        case Axis::X:
            return min.x();
        case Axis::Y:
            return min.y();
        case Axis::Z:
            return min.z();
        }
        return min.x();
    }

    [[nodiscard]] constexpr float axisMax(Axis axis) const noexcept
    {
        switch (axis)
        {
        case Axis::X:
            return max.x();
        case Axis::Y:
            return max.y();
        case Axis::Z:
            return max.z();
        }
        return max.x();
    }

    [[nodiscard]] constexpr Vec3 center() const noexcept
    {
        return (min + max) * 0.5f;
    }

    [[nodiscard]] constexpr Vec3 extent() const noexcept
    {
        return max - min;
    }

    // The eight corners, taking min/max per bit of the index (bit 0 -> x, 1 -> y, 2 -> z).
    [[nodiscard]] constexpr std::array<Vec3, 8> corners() const noexcept
    {
        return {Vec3{min.x(), min.y(), min.z()}, Vec3{max.x(), min.y(), min.z()},
                Vec3{min.x(), max.y(), min.z()}, Vec3{max.x(), max.y(), min.z()},
                Vec3{min.x(), min.y(), max.z()}, Vec3{max.x(), min.y(), max.z()},
                Vec3{min.x(), max.y(), max.z()}, Vec3{max.x(), max.y(), max.z()}};
    }

    // The AABB enclosing this box after transforming its eight corners (as points) by `m`.
    [[nodiscard]] AABB transformedBy(const Mat4& m) const noexcept
    {
        const std::array<Vec3, 8> pts = corners();
        const auto transform = [&m](const Vec3& v)
        {
            const Vec4 h = m * Vec4{v.x(), v.y(), v.z(), 1.0f};
            return Vec3{h.x(), h.y(), h.z()};
        };
        Vec3 lo = transform(pts.front());
        Vec3 hi = lo;
        for (std::size_t i = 1; i < pts.size(); ++i)
        {
            const Vec3 p = transform(pts[i]);
            lo = {std::min(lo.x(), p.x()), std::min(lo.y(), p.y()), std::min(lo.z(), p.z())};
            hi = {std::max(hi.x(), p.x()), std::max(hi.y(), p.y()), std::max(hi.z(), p.z())};
        }
        return {lo, hi};
    }
};

} // namespace fire_engine
