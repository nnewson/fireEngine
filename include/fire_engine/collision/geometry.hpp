#pragma once

#include <fire_engine/collision/world_shape.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Reusable closest-point primitives for the shape-specific narrowphase. Kept
// separate from narrow_phase.cpp so they can be unit-tested in isolation.

// Closest point on the segment [a, b] to p. Degenerate (a == b) returns a.
[[nodiscard]] Vec3 closestPointOnSegment(const Vec3& p, const Vec3& a, const Vec3& b) noexcept;

// Closest point on or inside an oriented box to p (clamp in box-local space,
// transform back). Returns p's projection onto the box surface when outside, or
// p itself when inside.
[[nodiscard]] Vec3 closestPointOnObb(const Vec3& p, const WorldBox& box) noexcept;

struct SegmentClosest
{
    Vec3 c1{}; // closest point on segment 1
    Vec3 c2{}; // closest point on segment 2
};

// Closest points between segments [p1, q1] and [p2, q2] (Ericson, RTCD §5.1.9).
// Handles parallel and degenerate segments.
[[nodiscard]] SegmentClosest closestPointsBetweenSegments(const Vec3& p1, const Vec3& q1,
                                                          const Vec3& p2, const Vec3& q2) noexcept;

} // namespace fire_engine
