#pragma once

#include <limits>
#include <optional>

#include <fire_engine/collision/aabb.hpp>
#include <fire_engine/collision/world_shape.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// A half-line for ray queries: `origin` + t·`direction`, t in [0, maxDistance].
// `direction` is expected to be unit length so t reads as a world-space distance.
struct Ray
{
    Vec3 origin{};
    Vec3 direction{0.0f, 0.0f, -1.0f};
    float maxDistance{std::numeric_limits<float>::max()};
};

// A ray/shape intersection: the world-space hit point, the outward surface normal at
// that point, and the distance along the ray (t, since direction is unit).
struct RayHit
{
    float distance{0.0f};
    Vec3 point{};
    Vec3 normal{};
};

// Slab test against an axis-aligned box. Returns true when the ray enters the box
// within [0, maxDistance]; `tNear` is the entry distance (0 when the origin is inside).
// Used as the broadphase reject before the exact shape test.
[[nodiscard]] bool rayIntersectsAabb(const Ray& ray, const AABB& box, float& tNear) noexcept;

// Exact ray/shape intersections (nearest hit in front of the origin). Each returns the
// entry hit; a ray starting inside reports the forward exit/surface as appropriate.
[[nodiscard]] std::optional<RayHit> rayIntersect(const Ray& ray,
                                                 const WorldSphere& sphere) noexcept;
[[nodiscard]] std::optional<RayHit> rayIntersect(const Ray& ray, const WorldBox& box) noexcept;
[[nodiscard]] std::optional<RayHit> rayIntersect(const Ray& ray,
                                                 const WorldCapsule& capsule) noexcept;
[[nodiscard]] std::optional<RayHit> rayIntersect(const Ray& ray,
                                                 const WorldConvex& convex) noexcept;
[[nodiscard]] std::optional<RayHit> rayIntersect(const Ray& ray, const WorldShape& shape) noexcept;

// Möller–Trumbore ray/triangle (two-sided). Used for static mesh colliders, whose
// triangles are tested via their per-collider BVH.
[[nodiscard]] std::optional<RayHit> rayIntersectTriangle(const Ray& ray, const Vec3& v0,
                                                         const Vec3& v1, const Vec3& v2) noexcept;

} // namespace fire_engine
