#pragma once

#include <cstddef>
#include <vector>

#include <fire_engine/collision/collider.hpp>
#include <fire_engine/collision/collider_id.hpp>

namespace fire_engine
{

// A candidate overlapping pair emitted by a broadphase: the two collider ids
// (ordered first < second) and pointers to the colliders themselves. The narrowphase
// consumes these to produce contacts.
struct CollisionPair
{
    ColliderId firstId;
    ColliderId secondId;
    const Collider* first;
    const Collider* second;
};

// Abstract broadphase: tracks a set of colliders and reports the pairs whose
// (swept) world AABBs overlap and whose layer/mask filters permit a collision.
// PhysicsWorld owns one of these; SweepAndPruneBroadPhase and
// DynamicAabbTreeBroadPhase implement it interchangeably.
class BroadPhase
{
public:
    BroadPhase() = default;
    virtual ~BroadPhase() = default;

    // Register a collider, assigning + returning its ColliderId. The collider's
    // world/swept bounds must be current (call Collider::update first).
    virtual ColliderId addCollider(Collider& collider) = 0;

    [[nodiscard]]
    virtual bool removeCollider(ColliderId colliderId) = 0;

    [[nodiscard]]
    virtual bool removeCollider(Collider& collider) = 0;

    virtual void clear() = 0;

    // Refresh the structure from the colliders' current bounds and recompute the
    // possible-pair list.
    virtual void update() = 0;

    // Rebuild the structure + pair list from scratch (used after the solver moves
    // many bodies at once).
    virtual void rebuild() = 0;

    [[nodiscard]]
    virtual const std::vector<CollisionPair>& possiblePairs() const noexcept = 0;

    // Cross-check the maintained pair set against a brute-force O(n²) overlap test.
    [[nodiscard]]
    virtual bool validate() const = 0;

    [[nodiscard]]
    virtual std::size_t colliderCount() const noexcept = 0;

protected:
    // Protected so a base pointer can't slice, but a concrete broadphase can still be
    // copyable/movable as it declares (SAP is move-only).
    BroadPhase(const BroadPhase&) = default;
    BroadPhase& operator=(const BroadPhase&) = default;
    BroadPhase(BroadPhase&&) noexcept = default;
    BroadPhase& operator=(BroadPhase&&) noexcept = default;
};

} // namespace fire_engine
