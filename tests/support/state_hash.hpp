#pragma once

#include <bit>
#include <cstdint>
#include <span>

#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/physics_body.hpp>
#include <fire_engine/physics/physics_handle.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/transform.hpp>

// Deterministic state hashing for the physics determinism harness. Folds the raw
// bit pattern of each float into a 64-bit FNV-1a hash, so two runs that produce
// bit-identical body state produce the same hash (and any drift is caught).

namespace fire_engine::test
{

[[nodiscard]]
inline std::uint64_t hashFloat(std::uint64_t h, float v) noexcept
{
    std::uint32_t bits = std::bit_cast<std::uint32_t>(v);
    for (int i = 0; i < 4; ++i)
    {
        h ^= static_cast<std::uint64_t>(bits & 0xFFu);
        h *= 0x100000001b3ULL; // FNV-1a prime
        bits >>= 8;
    }
    return h;
}

[[nodiscard]]
inline std::uint64_t hashVec3(std::uint64_t h, const Vec3& v) noexcept
{
    h = hashFloat(h, v.x());
    h = hashFloat(h, v.y());
    h = hashFloat(h, v.z());
    return h;
}

// Hashes the full simulated state (transform + velocities) of the given bodies,
// in the caller-supplied handle order — so the test owns a deterministic order
// without PhysicsWorld needing a public iterator.
[[nodiscard]]
inline std::uint64_t hashBodyState(const PhysicsWorld& world,
                                   std::span<const PhysicsBodyHandle> handles)
{
    std::uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a offset basis
    for (const PhysicsBodyHandle handle : handles)
    {
        if (const auto transform = world.bodyTransform(handle); transform.has_value())
        {
            h = hashVec3(h, transform->position());
            const Quaternion r = transform->rotation();
            h = hashFloat(h, r.x());
            h = hashFloat(h, r.y());
            h = hashFloat(h, r.z());
            h = hashFloat(h, r.w());
            h = hashVec3(h, transform->scale());
        }
        if (const PhysicsBody* body = world.body(handle); body != nullptr)
        {
            h = hashVec3(h, body->linearVelocity());
            h = hashVec3(h, body->angularVelocity());
        }
    }
    return h;
}

} // namespace fire_engine::test
