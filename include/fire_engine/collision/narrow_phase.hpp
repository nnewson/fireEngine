#pragma once

#include <optional>

#include <fire_engine/collision/contact_manifold.hpp>
#include <fire_engine/collision/world_shape.hpp>

namespace fire_engine
{

class NarrowPhase
{
public:
    NarrowPhase() = default;
    ~NarrowPhase() = default;

    NarrowPhase(const NarrowPhase&) = default;
    NarrowPhase& operator=(const NarrowPhase&) = default;
    NarrowPhase(NarrowPhase&&) noexcept = default;
    NarrowPhase& operator=(NarrowPhase&&) noexcept = default;

    // Shape-specific contact manifold for a candidate pair, in world space. The
    // manifold normal points from `b` toward `a` (so `a` is pushed out along it).
    // Dispatches by shape pair: sphere/box/capsule analytic + box-box SAT with face
    // clipping.
    //
    // `speculativeMargin` widens contact generation to *separated* shapes whose gap
    // is within the margin (used for speculative-contact CCD). With margin 0 (the
    // default) only overlapping shapes produce a contact. A `ManifoldPoint`'s
    // `penetration` is signed: > 0 overlap, < 0 a gap (= -separation). Returns
    // nullopt when the shapes are separated by more than the margin.
    [[nodiscard]]
    std::optional<ContactManifold> collide(const WorldShape& a, const WorldShape& b,
                                           float speculativeMargin = 0.0f) const noexcept;
};

} // namespace fire_engine
