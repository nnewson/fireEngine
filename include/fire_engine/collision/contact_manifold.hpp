#pragma once

#include <array>

#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Maximum contact points in a manifold (a box face clipped against another).
inline constexpr int kMaxManifoldPoints = 4;

struct ManifoldPoint
{
    Vec3 position{};      // world-space contact point
    float penetration{0.0f}; // overlap depth along the manifold normal (>= 0)
};

// Geometry-only contact manifold produced by the narrowphase. Body-agnostic —
// PhysicsWorld pairs it with the colliding bodies. The normal points from the
// second shape toward the first (target → moving), i.e. the direction to push
// the first/moving shape to separate.
struct ContactManifold
{
    Vec3 normal{};
    std::array<ManifoldPoint, kMaxManifoldPoints> points{};
    int count{0};

    // Deepest penetration across the contact points (the response's push-out
    // distance). 0 when there are no points.
    [[nodiscard]] float maxPenetration() const noexcept
    {
        float d = 0.0f;
        for (int i = 0; i < count; ++i)
        {
            d = points[i].penetration > d ? points[i].penetration : d;
        }
        return d;
    }
};

} // namespace fire_engine
