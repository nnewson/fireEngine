#pragma once

#include <cstdint>

#include <fire_engine/math/vec3.hpp>

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
};

} // namespace fire_engine
