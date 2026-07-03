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

// A self-collision contact between two links of the *same* articulation (e.g. a thigh capsule
// vs the torso). Points are in each link's local frame; the relative separation along `normal`
// is tracked across the substep loop as both links move. `normal` (world, unit) points from B
// toward A — the direction to push A off B. `offset` sets the touching baseline:
// separation = dot(normal, worldA − worldB) − offset. Solved through the pair impulse response.
struct ArticulationLinkContact
{
    std::size_t linkA{0};
    std::size_t linkB{0};
    Vec3 localA{};
    Vec3 localB{};
    Vec3 normal{0, 1, 0};
    float offset{0.0f};
    float friction{0.5f};
};

// Advance an articulation one fixed step `dt` under `gravity` + passive `jointDamping`, resolving
// plane contacts (link vs static) and self-collision link contacts (link vs link, same
// articulation) through the articulated impulse response. TGS-style: kSubstepCount substeps of
// h = dt/N, each integrating the free dynamics then solving the contacts + joint limits (soft
// non-penetration + velocity-only Coulomb friction) via the ConstraintBody / pair-impulse seams.
void stepArticulationOnPlanes(Articulation& articulation,
                              std::span<const ArticulationPlaneContact> contacts,
                              const Vec3& gravity, float dt, float jointDamping = 0.0f,
                              std::span<const ArticulationLinkContact> linkContacts = {});

} // namespace fire_engine
