#pragma once

#include <cstdint>

#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/physics_handle.hpp>

namespace fire_engine
{

// Layer/mask filter for spatial queries, mirroring the collider convention: a collider
// is considered when (filter.mask & collider.layer) and (collider.mask & filter.layer)
// are both non-zero. The defaults (layer 1, mask all) hit everything.
struct QueryFilter
{
    std::uint32_t layer{1U};
    std::uint32_t mask{~0U};
};

// A ray/collider hit: which collider+body was struck, the world-space hit point and
// surface normal, and the distance along the ray.
struct RaycastHit
{
    PhysicsColliderHandle collider{};
    PhysicsBodyHandle body{};
    Vec3 point{};
    Vec3 normal{};
    float distance{0.0f};
};

// A shapecast hit: the first collider the swept shape reaches, the contact point on its
// surface, the surface normal (pointing back toward the swept shape), and the
// time-of-impact distance along the sweep.
struct ShapecastHit
{
    PhysicsColliderHandle collider{};
    PhysicsBodyHandle body{};
    Vec3 point{};
    Vec3 normal{};
    float distance{0.0f};
};

// A collider overlapping the query region.
struct OverlapHit
{
    PhysicsColliderHandle collider{};
    PhysicsBodyHandle body{};
};

} // namespace fire_engine
