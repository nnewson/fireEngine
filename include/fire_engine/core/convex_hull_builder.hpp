#pragma once

#include <cstdint>
#include <span>

#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/collider_shape.hpp>

namespace fire_engine
{

// Build a `ConvexHullShape` from a triangle mesh: weld coincident vertices, then
// merge coplanar triangles into polygon faces (each an ordered, CCW-outward
// boundary loop + outward normal). Assumes the input mesh is convex (typical for an
// authored convex collider); `isConvex` can validate the result. Returns an empty
// hull for degenerate input.
[[nodiscard]] ConvexHullShape buildConvexHull(std::span<const Vec3> positions,
                                              std::span<const std::uint32_t> indices);

// True if every vertex lies behind (or on) every face's plane — i.e. the hull built
// from a mesh is actually convex.
[[nodiscard]] bool isConvex(const ConvexHullShape& hull) noexcept;

} // namespace fire_engine
