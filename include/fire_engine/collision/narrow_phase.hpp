#pragma once

#include <optional>

#include <fire_engine/collision/collider.hpp>
#include <fire_engine/collision/contact_manifold.hpp>
#include <fire_engine/collision/world_shape.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

struct SweptAabbContact
{
    float toi{0.0f};
    Vec3 normal{};
};

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
    // Returns nullopt when the shapes are separated. Dispatches by shape pair:
    // sphere/box/capsule analytic + box-box SAT with face clipping.
    [[nodiscard]]
    std::optional<ContactManifold> collide(const WorldShape& a, const WorldShape& b) const noexcept;

    // Legacy swept-AABB time-of-impact test. Retained as the seed for deferred
    // continuous-collision / speculative contacts; no longer on the step() path.
    [[nodiscard]]
    std::optional<SweptAabbContact> sweptAabb(const Collider& moving,
                                              const Collider& target) const noexcept;
};

} // namespace fire_engine
