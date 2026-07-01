#pragma once

#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Minimal rigid (proper) transform — a unit quaternion rotation plus a translation,
// mapping a point p ↦ rotation·p + translation. Unlike scene::Transform it carries no
// scale or cached matrices: articulation forward kinematics composes thousands of these
// per step, so it stays a lightweight value type. Spatial dynamics (Phase B) build on it.
struct RigidTransform
{
    Quaternion rotation{Quaternion::identity()};
    Vec3 translation{};

    [[nodiscard]]
    constexpr Vec3 transformPoint(const Vec3& p) const noexcept
    {
        return rotation.rotate(p) + translation;
    }

    [[nodiscard]]
    constexpr Vec3 transformDirection(const Vec3& d) const noexcept
    {
        return rotation.rotate(d);
    }

    // Compose: (a * b) applies b first, then a — (a*b).transformPoint(p) ==
    // a.transformPoint(b.transformPoint(p)).
    [[nodiscard]]
    constexpr RigidTransform operator*(const RigidTransform& rhs) const noexcept
    {
        return RigidTransform{rotation * rhs.rotation,
                              rotation.rotate(rhs.translation) + translation};
    }

    // Inverse rigid transform: p ↦ rotationᵀ·(p − translation).
    [[nodiscard]]
    RigidTransform inverse() const noexcept
    {
        const Quaternion invRotation{-rotation.x(), -rotation.y(), -rotation.z(), rotation.w()};
        return RigidTransform{invRotation, invRotation.rotate(translation * -1.0f)};
    }
};

// Spatial (6-D) vector in Plücker coordinates: an angular part over a linear part. The
// motion interpretation is (ω, v) — a rigid body's twist; the force interpretation is
// (τ, f) — a wrench. Phase A uses it only as the foundation value type (joint motion
// subspaces, link twists); the spatial-inertia / articulated-body operators that consume
// it arrive with the ABA dynamics in Phase B.
struct SpatialVector
{
    Vec3 angular{};
    Vec3 linear{};

    [[nodiscard]]
    constexpr SpatialVector operator+(const SpatialVector& rhs) const noexcept
    {
        return SpatialVector{angular + rhs.angular, linear + rhs.linear};
    }

    [[nodiscard]]
    constexpr SpatialVector operator-(const SpatialVector& rhs) const noexcept
    {
        return SpatialVector{angular - rhs.angular, linear - rhs.linear};
    }

    [[nodiscard]]
    constexpr SpatialVector operator*(float s) const noexcept
    {
        return SpatialVector{angular * s, linear * s};
    }

    // Spatial dot product pairs a motion vector with a force vector (power): the angular
    // part dots with the force's torque and the linear with its force, summed.
    [[nodiscard]]
    constexpr float dot(const SpatialVector& rhs) const noexcept
    {
        return Vec3::dotProduct(angular, rhs.angular) + Vec3::dotProduct(linear, rhs.linear);
    }
};

[[nodiscard]]
constexpr SpatialVector operator*(float s, const SpatialVector& v) noexcept
{
    return v * s;
}

} // namespace fire_engine
