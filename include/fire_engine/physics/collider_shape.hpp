#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include <fire_engine/collision/collider.hpp>
#include <fire_engine/collision/world_shape.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/physics_body.hpp>

namespace fire_engine
{

struct AabbShape
{
    AABB bounds{};
};

struct BoxShape
{
    Vec3 halfExtents{0.5f, 0.5f, 0.5f};
    Vec3 center{};
};

struct SphereShape
{
    float radius{0.5f};
    Vec3 center{};
};

struct CapsuleShape
{
    float radius{0.5f};
    float halfHeight{0.5f};
    Vec3 center{};
};

// An authored convex hull: local-space vertices plus polygon faces (each a CCW
// outward loop of vertex indices + local normal — see `ConvexFace`). Built from a
// mesh by `buildConvexHull`; composed into a `WorldConvex` per step by
// `PhysicsWorld::worldShape` for the GJK/EPA narrowphase.
struct ConvexHullShape
{
    std::vector<Vec3> vertices;
    std::vector<ConvexFace> faces;
};

using ColliderShape = std::variant<AabbShape, BoxShape, SphereShape, CapsuleShape, ConvexHullShape>;

struct ColliderDesc
{
    ColliderShape shape{AabbShape{}};
    std::uint32_t collisionLayer{1U};
    std::uint32_t collisionMask{~0U};
    PhysicsMaterial material{};
};

// One primitive of a compound collider: a (non-compound) shape placed at a local
// offset within the body. A body's compound collider is a list of these passed to
// PhysicsWorld::createCompoundCollider, which creates one child collider per entry
// (each registered with the broadphase) and aggregates their mass properties into the
// body's centre of mass + inertia. Compounds do not nest.
struct CompoundChild
{
    ColliderShape shape{AabbShape{}};
    Vec3 localPosition{};
    Quaternion localRotation{Quaternion::identity()};
    PhysicsMaterial material{};
};

} // namespace fire_engine
