#pragma once

#include <array>

#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Maximum contact points in a manifold (a box face clipped against another).
inline constexpr int kMaxManifoldPoints = 4;

// Manifold-richness offset: a clipped face point is kept as a manifold point when it
// lies within this distance *above* the reference face (not just touching/penetrating),
// with its true signed gap recorded as a negative penetration. This widens a
// near-degenerate support (a tetra landing almost-but-not-quite flat) from a single
// point into a small patch, so friction is shared instead of pumping spin at one vertex.
// The extra points are speculative (the solver brakes their closing but never pushes
// them), so they add support without lifting a resting body.
//
// This is deliberately distinct from the physics layer's kSpeculativeDistance (anti-tunnel
// *pair reach*): this governs how many *points* a confirmed contact keeps, not whether a
// pair collides at all.
inline constexpr float kContactOffset = 0.01f;

struct ManifoldPoint
{
    Vec3 position{}; // world-space contact point
    // Overlap depth along the manifold normal: > 0 penetrating, < 0 a speculative gap
    // (a near-contact point kept within kContactOffset of the surface).
    float penetration{0.0f};
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
