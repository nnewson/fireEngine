#pragma once

#include <variant>

#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// World-space collision primitives, neutral to the physics layer. PhysicsWorld
// composes each authored ColliderShape with its body transform into one of these
// (see PhysicsWorld::worldShape); the narrowphase consumes them. Keeping them in
// collision/ avoids an upward dependency on physics/ColliderShape.
//
// An authored AABB is just a WorldBox with identity orientation.

struct WorldSphere
{
    Vec3 center{};
    float radius{0.5f};
};

struct WorldBox
{
    Vec3 center{};
    Vec3 halfExtents{0.5f, 0.5f, 0.5f};
    Quaternion orientation{Quaternion::identity()};
};

struct WorldCapsule
{
    Vec3 p0{};
    Vec3 p1{};
    float radius{0.5f};
};

using WorldShape = std::variant<WorldSphere, WorldBox, WorldCapsule>;

} // namespace fire_engine
