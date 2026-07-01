#pragma once

#include <cstddef>
#include <span>

#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

class Articulation;

// A contact between a point rigidly fixed on an articulation link and a static half-space
// (the plane {x : dot(normal, x) >= offset}). The minimal contact form for the Phase D gate
// — a fixed-base articulation resting on a floor — exercising the full path: ABA dynamics +
// the ConstraintBody seam (normal + Coulomb friction) over the TGS substep loop.
struct ArticulationPlaneContact
{
    std::size_t link{0};
    Vec3 localPoint{};    // contact point in the link's local frame
    Vec3 normal{0, 1, 0}; // plane outward normal (world, unit)
    float offset{0.0f};   // plane: dot(normal, x) >= offset
    float friction{0.5f};
};

// Advance a fixed-base articulation one fixed step `dt` under `gravity` + passive
// `jointDamping`, resolving each plane contact through the articulated impulse response.
// TGS-style: kSubstepCount substeps of h = dt/N, each integrating the free dynamics then
// solving the contacts (soft non-penetration bias + velocity-only Coulomb friction) via the
// ConstraintBody seam. Fixed-base only for now (floating base lands with the 6-DOF root).
void stepArticulationOnPlanes(Articulation& articulation,
                              std::span<const ArticulationPlaneContact> contacts,
                              const Vec3& gravity, float dt, float jointDamping = 0.0f);

} // namespace fire_engine
