#pragma once

#include <cmath>
#include <limits>
#include <variant>

#include <fire_engine/collision/world_shape.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Support functions: the farthest point of a convex shape along a direction `dir`
// (need not be unit). These are the only thing GJK/EPA need from a shape, so any
// shape with a support function — including future ones — drops into the convex
// narrowphase. For rounded shapes the support is the core support plus radius·dir̂.

namespace detail
{

[[nodiscard]] inline Vec3 unitOr(const Vec3& dir, const Vec3& fallback) noexcept
{
    const float lenSq = dir.magnitudeSquared();
    return lenSq > 1e-12f ? dir * (1.0f / std::sqrt(lenSq)) : fallback;
}

} // namespace detail

[[nodiscard]] inline Vec3 supportPoint(const WorldSphere& s, const Vec3& dir) noexcept
{
    return s.center + detail::unitOr(dir, Vec3{0.0f, 1.0f, 0.0f}) * s.radius;
}

[[nodiscard]] inline Vec3 supportPoint(const WorldBox& b, const Vec3& dir) noexcept
{
    const Vec3 ax = b.orientation.rotate(Vec3{1.0f, 0.0f, 0.0f});
    const Vec3 ay = b.orientation.rotate(Vec3{0.0f, 1.0f, 0.0f});
    const Vec3 az = b.orientation.rotate(Vec3{0.0f, 0.0f, 1.0f});
    return b.center +
           ax * (Vec3::dotProduct(dir, ax) >= 0.0f ? b.halfExtents.x() : -b.halfExtents.x()) +
           ay * (Vec3::dotProduct(dir, ay) >= 0.0f ? b.halfExtents.y() : -b.halfExtents.y()) +
           az * (Vec3::dotProduct(dir, az) >= 0.0f ? b.halfExtents.z() : -b.halfExtents.z());
}

[[nodiscard]] inline Vec3 supportPoint(const WorldCapsule& c, const Vec3& dir) noexcept
{
    const Vec3 base = Vec3::dotProduct(dir, c.p1 - c.p0) >= 0.0f ? c.p1 : c.p0;
    return base + detail::unitOr(dir, Vec3{0.0f, 1.0f, 0.0f}) * c.radius;
}

[[nodiscard]] inline Vec3 supportPoint(const WorldConvex& h, const Vec3& dir) noexcept
{
    Vec3 best{};
    float bestDot = -std::numeric_limits<float>::infinity();
    for (const Vec3& v : h.vertices)
    {
        const float d = Vec3::dotProduct(dir, v);
        if (d > bestDot)
        {
            bestDot = d;
            best = v;
        }
    }
    return best;
}

[[nodiscard]] inline Vec3 supportPoint(const WorldShape& shape, const Vec3& dir) noexcept
{
    return std::visit([&dir](const auto& s) { return supportPoint(s, dir); }, shape);
}

} // namespace fire_engine
