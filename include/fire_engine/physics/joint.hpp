#pragma once

#include <cstdint>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/physics_handle.hpp>

namespace fire_engine
{

enum class JointType
{
    Distance,   // hold the two anchors at a fixed separation
    BallSocket, // hold the two anchors coincident (3 DOF rotation)
    Hinge,      // ball-socket + axis alignment (1 DOF rotation)
};

// Optional angular limits (P4 Phase 3). Measured from the relative orientation of B
// in A's frame at joint-creation time (the "rest" relative frame), so a limit reads
// 0 at the pose the joint was built in.
//   - Hinge: `hinge` enables a [lowerAngle, upperAngle] clamp on the rotation about
//     the hinge axis (radians, signed).
//   - BallSocket: `coneTwist` enables a swing cone (half-angle `swingLimit`) about
//     the twist axis (the joint's axisA) plus a ±`twistLimit` clamp on the twist.
// Limits are unilateral: a row is added only while a limit is violated, so a joint
// inside its range behaves exactly as the bilateral joint and adds no energy.
struct JointLimits
{
    bool hinge{false};
    float lowerAngle{0.0f};
    float upperAngle{0.0f};
    bool coneTwist{false};
    float swingLimit{pi};
    float twistLimit{pi};
};

// Authored joint between two bodies. Anchors and axes are in each body's local
// frame (origin at the centre of mass); PhysicsWorld composes them with the live
// body transforms each step. For Hinge, `axisA`/`axisB` are the (local) hinge axes
// kept aligned; for Distance, `restLength` is the target anchor separation.
struct JointDesc
{
    JointType type{JointType::BallSocket};
    PhysicsBodyHandle bodyA;
    PhysicsBodyHandle bodyB;
    Vec3 anchorA{};
    Vec3 anchorB{};
    Vec3 axisA{0.0f, 1.0f, 0.0f};
    Vec3 axisB{0.0f, 1.0f, 0.0f};
    float restLength{0.0f};
    JointLimits limits{};
};

// World-space joint handed to the JointSolver each step (PhysicsWorld composes the
// local anchors/axes with the current body transforms). `bodyA`/`bodyB` index the
// solver-body array; `key` identifies the joint for warm starting.
struct JointInput
{
    int bodyA{-1};
    int bodyB{-1};
    JointType type{JointType::BallSocket};
    Vec3 anchorA{};
    Vec3 anchorB{};
    Vec3 axisA{};
    Vec3 axisB{};
    Vec3 twistAxisLocal{}; // axisA in A's local frame, for swing-twist decomposition
    float restLength{0.0f};
    // qD = relative orientation of B in A's frame, relative to the rest frame at
    // creation (identity when the joint sits at its rest pose). Limits decompose this.
    Quaternion relative{};
    JointLimits limits{};
    std::uint64_t key{0};
};

} // namespace fire_engine
