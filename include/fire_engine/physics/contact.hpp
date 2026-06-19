#pragma once

#include <vector>

#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/physics_handle.hpp>

namespace fire_engine
{

struct Contact
{
    PhysicsBodyHandle firstBody;
    PhysicsBodyHandle secondBody;
    PhysicsColliderHandle firstCollider;
    PhysicsColliderHandle secondCollider;
    float toi{0.0f};
    Vec3 normal{};
};

struct ContactManifold
{
    PhysicsBodyHandle firstBody;
    PhysicsBodyHandle secondBody;
    std::vector<Contact> contacts;
};

// Lightweight per-step contact record for debug visualisation only. The current
// swept-AABB narrowphase has no real contact manifold, so `point` is an
// approximation (the moving body's position advanced to the time of impact); it
// sharpens once P1 lands shape-specific manifolds. Vulkan-free.
struct DebugContact
{
    Vec3 point{};
    Vec3 normal{};
};

} // namespace fire_engine
