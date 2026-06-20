#pragma once

#include <span>
#include <variant>
#include <vector>

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

// One polygon face of a convex hull: its outward normal (in whatever frame the
// owning vertices live) and an ordered (CCW-outward) loop of vertex indices.
// Shared by the authored `ConvexHullShape` (local) and `WorldConvex` (the loops
// are frame-independent; world normals are recomputed from the world vertices).
struct ConvexFace
{
    Vec3 normal{};
    std::vector<int> loop;
};

// A convex hull in world space. `vertices` is owned (the body transform applied to
// the hull's local vertices each step); `faces` references the source
// `ConvexHullShape::faces` (stable for the step). Support queries read `vertices`;
// the manifold clip reads `faces` (recomputing world normals from `vertices`).
struct WorldConvex
{
    std::vector<Vec3> vertices;
    std::span<const ConvexFace> faces;
};

using WorldShape = std::variant<WorldSphere, WorldBox, WorldCapsule, WorldConvex>;

} // namespace fire_engine
