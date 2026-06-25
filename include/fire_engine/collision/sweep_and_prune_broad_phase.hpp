#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

#include <fire_engine/collision/broad_phase.hpp>
#include <fire_engine/collision/collider.hpp>

namespace fire_engine
{

class SweepAndPruneBroadPhase final : public BroadPhase
{
public:
    SweepAndPruneBroadPhase() = default;
    ~SweepAndPruneBroadPhase() override = default;

    SweepAndPruneBroadPhase(const SweepAndPruneBroadPhase&) = delete;
    SweepAndPruneBroadPhase& operator=(const SweepAndPruneBroadPhase&) = delete;
    SweepAndPruneBroadPhase(SweepAndPruneBroadPhase&&) noexcept = default;
    SweepAndPruneBroadPhase& operator=(SweepAndPruneBroadPhase&&) noexcept = default;

    // Ensure Collider::update() is called before adding and after moving colliders,
    // so endpoint values are up to date before incremental SAP sorting.
    ColliderId addCollider(Collider& collider) override;

    [[nodiscard]]
    bool removeCollider(ColliderId colliderId) override;

    [[nodiscard]]
    bool removeCollider(Collider& collider) override;

    void clear() override;
    void update() override;
    void updateCollider(Collider& collider);
    void updateEndPoint(EndPoint& endPoint);
    void rebuild() override;

    [[nodiscard]]
    bool validate() const override;

    [[nodiscard]]
    const std::vector<CollisionPair>& possiblePairs() const noexcept override
    {
        return possiblePairs_;
    }

    [[nodiscard]]
    std::size_t colliderCount() const noexcept override
    {
        return colliders_.size();
    }

private:
    struct ColliderEntry
    {
        ColliderId id;
        Collider* collider;
    };

    struct PairKey
    {
        ColliderId first;
        ColliderId second;

        [[nodiscard]]
        friend bool operator==(const PairKey&, const PairKey&) noexcept = default;
    };

    struct PairKeyHash
    {
        [[nodiscard]]
        std::size_t operator()(PairKey key) const noexcept;
    };

    struct PairState
    {
        const Collider* first{nullptr};
        const Collider* second{nullptr};
        // Index into possiblePairs_ when this pair is currently in the
        // possible-list; npos otherwise. Lets removePossiblePair do an
        // O(1) swap-and-pop instead of a linear find_if.
        std::size_t possibleIndex{std::numeric_limits<std::size_t>::max()};
        unsigned char axisMask{0};
        bool possible{false};
    };

    ColliderId nextColliderId_{ColliderId{1U}};
    std::vector<ColliderEntry> colliders_;
    std::vector<EndPoint*> xEndPoints_;
    std::vector<EndPoint*> yEndPoints_;
    std::vector<EndPoint*> zEndPoints_;
    std::unordered_map<PairKey, PairState, PairKeyHash> pairStates_;
    std::vector<CollisionPair> possiblePairs_;

    // Bulk-add the new collider's six endpoints to the axis vectors and
    // sort them in place via binary insertion. Used by addCollider so a new
    // collider does not trigger a full re-sort of every axis.
    void addEndPointsSorted(Collider& collider);
    void insertEndPointSorted(EndPoint* endPoint, std::vector<EndPoint*>& endPoints);
    void removeEndPoints(Collider& collider);
    void sortAndIndexEndPoints(std::vector<EndPoint*>& endPoints);
    void rebuildPairStates();
    // Full-sweep every pair on this axis. Used only by rebuild() now.
    void sweepAxis(std::span<EndPoint* const> endPoints);
    // Sweep only forming overlap pairs that involve `target`. Used by
    // addCollider to incrementally pair the new collider against the
    // already-tracked ones without rebuilding all pair states.
    void sweepAxisForCollider(std::span<EndPoint* const> endPoints, const Collider* target);
    void updateEndPoint(EndPoint& endPoint, bool refreshPairs);
    void updatePairAxis(const Collider* lhs, const Collider* rhs, CollisionAxis axis,
                        bool overlaps);
    void addPossiblePair(PairKey key, PairState& state);
    void removePossiblePair(PairState& state);
    void removePairStatesFor(ColliderId colliderId);

    [[nodiscard]]
    std::vector<EndPoint*>& endPointsForAxis(CollisionAxis axis) noexcept;

    [[nodiscard]]
    const std::vector<EndPoint*>& endPointsForAxis(CollisionAxis axis) const noexcept;

    [[nodiscard]]
    ColliderId colliderId(const Collider& collider) const noexcept;

    [[nodiscard]]
    static PairKey orderedPair(ColliderId lhs, ColliderId rhs) noexcept;

    [[nodiscard]]
    static PairKey orderedPair(const Collider* lhs, const Collider* rhs) noexcept;

    [[nodiscard]]
    static bool lessThan(const EndPoint& lhs, const EndPoint& rhs) noexcept;

    [[nodiscard]]
    static bool overlapsOnAxis(const Collider& lhs, const Collider& rhs,
                               CollisionAxis axis) noexcept;

    [[nodiscard]]
    static bool canCollide(const Collider& lhs, const Collider& rhs) noexcept;

    [[nodiscard]]
    static unsigned char axisBit(CollisionAxis axis) noexcept;

    static void updateIndices(std::vector<EndPoint*>& endPoints) noexcept;
    static void resetIndices(std::vector<EndPoint*>& endPoints) noexcept;
};

} // namespace fire_engine
