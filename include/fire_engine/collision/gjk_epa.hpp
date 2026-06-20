#pragma once

#include <fire_engine/collision/world_shape.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Result of a GJK + EPA convex contact query (world space). `normal` is unit and
// points from B toward A (the direction to push A out), matching the narrowphase
// `b -> a` convention. When `colliding`, `depth` is the EPA penetration depth and
// `pointA`/`pointB` are the contact points on each surface. When separated,
// `colliding` is false, `depth` is the gap distance, and `pointA`/`pointB` are the
// closest points (witnesses) on each shape.
struct ConvexContact
{
    bool colliding{false};
    Vec3 normal{};
    float depth{0.0f};
    Vec3 pointA{};
    Vec3 pointB{};
};

// GJK distance + EPA penetration between two convex world shapes, using only their
// support functions (so it works for any shape, including WorldConvex). Universal,
// deterministic (fixed iteration order, no RNG); the reusable core P6 mesh reuses.
[[nodiscard]] ConvexContact gjkEpaContact(const WorldShape& a, const WorldShape& b) noexcept;

} // namespace fire_engine
