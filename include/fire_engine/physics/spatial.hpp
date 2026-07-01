#pragma once

#include <fire_engine/math/mat3.hpp>
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

// Spatial motion cross product v ×  u (Featherstone's crm): the acceleration a moving
// frame's velocity induces on another motion vector. (angular; linear) ordering.
[[nodiscard]]
constexpr SpatialVector crossMotion(const SpatialVector& v, const SpatialVector& u) noexcept
{
    return SpatialVector{Vec3::crossProduct(v.angular, u.angular),
                         Vec3::crossProduct(v.angular, u.linear) +
                             Vec3::crossProduct(v.linear, u.angular)};
}

// Spatial force cross product v ×* f (Featherstone's crf, the dual of crossMotion): the
// rate of change a moving frame's velocity induces on a force/momentum vector.
[[nodiscard]]
constexpr SpatialVector crossForce(const SpatialVector& v, const SpatialVector& f) noexcept
{
    return SpatialVector{Vec3::crossProduct(v.angular, f.angular) +
                             Vec3::crossProduct(v.linear, f.linear),
                         Vec3::crossProduct(v.angular, f.linear)};
}

// A 6x6 spatial matrix as four 3x3 blocks [[a, b], [c, d]], acting on a spatial vector
// (angular; linear): out.angular = a·ang + b·lin, out.linear = c·ang + d·lin. Used for
// spatial (articulated-body) inertias and the Plücker coordinate transforms.
struct SpatialMatrix
{
    Mat3 a{};
    Mat3 b{};
    Mat3 c{};
    Mat3 d{};

    [[nodiscard]]
    constexpr SpatialVector operator*(const SpatialVector& v) const noexcept
    {
        return SpatialVector{a * v.angular + b * v.linear, c * v.angular + d * v.linear};
    }

    [[nodiscard]]
    constexpr SpatialMatrix operator*(const SpatialMatrix& m) const noexcept
    {
        return SpatialMatrix{a * m.a + b * m.c, a * m.b + b * m.d, c * m.a + d * m.c,
                             c * m.b + d * m.d};
    }

    [[nodiscard]]
    constexpr SpatialMatrix operator+(const SpatialMatrix& m) const noexcept
    {
        return SpatialMatrix{a + m.a, b + m.b, c + m.c, d + m.d};
    }

    [[nodiscard]]
    constexpr SpatialMatrix operator-(const SpatialMatrix& m) const noexcept
    {
        return SpatialMatrix{a - m.a, b - m.b, c - m.c, d - m.d};
    }
};

// Outer product s·fᵀ of a force-like column `s` with a force-like row `f`, as the 6x6
// spatial matrix whose blocks are the 3x3 outer products. Used for the ABA rank-1
// articulated-inertia update U·D⁻¹·Uᵀ.
[[nodiscard]]
constexpr SpatialMatrix spatialOuter(const SpatialVector& s, const SpatialVector& f) noexcept
{
    const auto outer = [](const Vec3& u, const Vec3& w)
    { return Mat3::fromColumns(w * u.x(), w * u.y(), w * u.z()); };
    return SpatialMatrix{outer(s.angular, f.angular), outer(s.angular, f.linear),
                         outer(s.linear, f.angular), outer(s.linear, f.linear)};
}

// Plücker MOTION transform child→parent for a rigid transform T = (E, r) that maps a
// child-frame point to the parent frame (p_parent = E·p_child + r): a motion vector
// (ω; v) maps to (E·ω; r×(E·ω) + E·v). As a block matrix [[R,0],[skew(r)R, R]].
[[nodiscard]]
inline SpatialMatrix motionTransform(const RigidTransform& t) noexcept
{
    const Mat3 r = Mat3::fromQuaternion(t.rotation);
    return SpatialMatrix{r, Mat3{}, Mat3::skew(t.translation) * r, r};
}

// Plücker FORCE transform child→parent (dual of motionTransform): a force vector (n; f)
// maps to (E·n + r×(E·f); E·f). Block matrix [[R, skew(r)R],[0, R]].
[[nodiscard]]
inline SpatialMatrix forceTransform(const RigidTransform& t) noexcept
{
    const Mat3 r = Mat3::fromQuaternion(t.rotation);
    return SpatialMatrix{r, Mat3::skew(t.translation) * r, Mat3{}, r};
}

// Rigid-body spatial inertia about a link frame's origin, from mass `m`, centre of mass
// `com` (in that frame), and rotational inertia `inertia` about the COM. As a block
// matrix [[Iₒ, skew(h)], [skew(h)ᵀ, m·1]] with h = m·com and Iₒ = I_com − m·skew(com)²
// (parallel-axis to the origin). Symmetric and constant in the link frame.
[[nodiscard]]
inline SpatialMatrix spatialInertia(float m, const Vec3& com, const Mat3& inertia) noexcept
{
    const Vec3 h = com * m;
    const Mat3 sc = Mat3::skew(com);
    const Mat3 io = inertia - (sc * sc) * m; // I_com − m·skew(com)²  (parallel axis to origin)
    const Mat3 sh = Mat3::skew(h);
    return SpatialMatrix{io, sh, sh.transpose(), Mat3::identity() * m};
}

} // namespace fire_engine
