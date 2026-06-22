#pragma once

#include <cstdint>

#include <fire_engine/physics/physics_handle.hpp>

namespace fire_engine
{

// Lifecycle phase of an overlap between two colliders, diffed step-to-step.
enum class EventPhase : std::uint8_t
{
    Enter, // began overlapping this step
    Stay,  // overlapping this step and the last
    Exit,  // overlapped last step, no longer
};

// One overlap-lifecycle event between two colliders. Used for both trigger events (a
// collider flagged isTrigger generates these instead of a solver response) and collision
// events (two solid colliders actually touching). The collider handles are ordered
// (first < second by value) so a pair has a stable identity across steps.
struct ContactEvent
{
    PhysicsColliderHandle first;
    PhysicsColliderHandle second;
    EventPhase phase{EventPhase::Enter};
};

} // namespace fire_engine
