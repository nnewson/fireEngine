#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <fire_engine/collision/aabb_bvh.hpp>
#include <fire_engine/collision/broad_phase.hpp>

namespace fire_engine
{

// Dynamic AABB tree broadphase. A thin wrapper over the generic `AabbBvh<Collider*>`:
// it owns collider-id assignment, the swept-bounds → BVH refresh, the layer/mask
// filter, and the sorted possible-pair list. The tree mechanics (insert / remove /
// balance / query) live in the reusable `AabbBvh` core.
//
// Pairs are regenerated each update()/rebuild() by querying every leaf and **sorted by
// collider id**, so the list is deterministic regardless of tree shape.
class DynamicAabbTreeBroadPhase final : public BroadPhase
{
public:
    DynamicAabbTreeBroadPhase() = default;
    ~DynamicAabbTreeBroadPhase() override = default;

    DynamicAabbTreeBroadPhase(const DynamicAabbTreeBroadPhase&) = delete;
    DynamicAabbTreeBroadPhase& operator=(const DynamicAabbTreeBroadPhase&) = delete;
    DynamicAabbTreeBroadPhase(DynamicAabbTreeBroadPhase&&) noexcept = default;
    DynamicAabbTreeBroadPhase& operator=(DynamicAabbTreeBroadPhase&&) noexcept = default;

    ColliderId addCollider(Collider& collider) override;

    [[nodiscard]]
    bool removeCollider(ColliderId colliderId) override;

    [[nodiscard]]
    bool removeCollider(Collider& collider) override;

    void clear() override;
    void update() override;
    void rebuild() override;

    [[nodiscard]]
    const std::vector<CollisionPair>& possiblePairs() const noexcept override
    {
        return possiblePairs_;
    }

    [[nodiscard]]
    bool validate() const override;

    [[nodiscard]]
    std::size_t colliderCount() const noexcept override
    {
        return proxyByColliderId_.size();
    }

private:
    void regeneratePairs();

    AabbBvh<Collider*> bvh_;
    ColliderId nextColliderId_{ColliderId{1U}};
    // Collider id value → BVH proxy handle. Only ever looked up by key (never iterated
    // to produce the pair list, which is sorted), so its ordering can't leak out.
    std::unordered_map<std::uint32_t, int> proxyByColliderId_;
    std::vector<CollisionPair> possiblePairs_;
};

} // namespace fire_engine
