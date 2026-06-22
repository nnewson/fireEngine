#pragma once

#include <optional>

#include <fire_engine/collision/world_shape.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Result of a shape cast: the time-of-impact distance along the sweep direction, the
// contact point on the target surface, and the surface normal (pointing from the target
// back toward the swept shape).
struct ToiHit
{
    float distance{0.0f};
    Vec3 point{};
    Vec3 normal{};
};

// Sweep `moving` along unit `direction` up to `maxDistance` and return the first contact
// with the static `target`, via GJK conservative advancement (reuses `gjkEpaContact`'s
// gap distance + separating normal). An already-overlapping start returns distance 0.
// `nullopt` when the swept shape never reaches the target within `maxDistance`.
[[nodiscard]] std::optional<ToiHit> shapeCast(const WorldShape& moving, const Vec3& direction,
                                              float maxDistance, const WorldShape& target) noexcept;

} // namespace fire_engine
