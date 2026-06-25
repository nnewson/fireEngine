#include <fire_engine/collision/sweep_and_prune_broad_phase.hpp>

#include <algorithm>
#include <limits>

namespace fire_engine
{

namespace
{

constexpr std::size_t kInvalidPairIndex = std::numeric_limits<std::size_t>::max();

} // namespace

ColliderId SweepAndPruneBroadPhase::addCollider(Collider& collider)
{
    const ColliderId existingId = colliderId(collider);
    if (existingId.valid())
    {
        return existingId;
    }

    const ColliderId id = nextColliderId_;
    nextColliderId_ = ColliderId{nextColliderId_.value() + 1U};
    collider.colliderId(id);

    colliders_.push_back({id, &collider});

    // Incremental: binary-insert the new collider's endpoints into each
    // already-sorted axis, then sweep each axis once forming overlap pairs
    // ONLY between the new collider and existing colliders. Existing pair
    // states are preserved — no global rebuild.
    addEndPointsSorted(collider);
    sweepAxisForCollider(xEndPoints_, &collider);
    sweepAxisForCollider(yEndPoints_, &collider);
    sweepAxisForCollider(zEndPoints_, &collider);
    return id;
}

bool SweepAndPruneBroadPhase::removeCollider(ColliderId colliderId)
{
    const auto found =
        std::find_if(colliders_.begin(), colliders_.end(),
                     [colliderId](const ColliderEntry& entry) { return entry.id == colliderId; });
    if (found == colliders_.end())
    {
        return false;
    }

    Collider* removed = found->collider;
    removeEndPoints(*removed);
    colliders_.erase(found);
    removePairStatesFor(colliderId);
    removed->colliderId(ColliderId{});
    return true;
}

bool SweepAndPruneBroadPhase::removeCollider(Collider& collider)
{
    const ColliderId id = colliderId(collider);
    if (!id.valid())
    {
        return false;
    }

    return removeCollider(id);
}

void SweepAndPruneBroadPhase::clear()
{
    resetIndices(xEndPoints_);
    resetIndices(yEndPoints_);
    resetIndices(zEndPoints_);
    for (ColliderEntry& entry : colliders_)
    {
        entry.collider->colliderId(ColliderId{});
    }

    colliders_.clear();
    xEndPoints_.clear();
    yEndPoints_.clear();
    zEndPoints_.clear();
    pairStates_.clear();
    possiblePairs_.clear();
}

void SweepAndPruneBroadPhase::update()
{
    for (ColliderEntry& entry : colliders_)
    {
        for (EndPoint* endPoint : entry.collider->endPoints())
        {
            updateEndPoint(*endPoint, false);
        }
    }
}

void SweepAndPruneBroadPhase::updateCollider(Collider& collider)
{
    if (!colliderId(collider).valid())
    {
        return;
    }

    for (EndPoint* endPoint : collider.endPoints())
    {
        updateEndPoint(*endPoint, false);
    }
}

void SweepAndPruneBroadPhase::updateEndPoint(EndPoint& endPoint)
{
    updateEndPoint(endPoint, true);
}

void SweepAndPruneBroadPhase::rebuild()
{
    sortAndIndexEndPoints(xEndPoints_);
    sortAndIndexEndPoints(yEndPoints_);
    sortAndIndexEndPoints(zEndPoints_);
    rebuildPairStates();
}

bool SweepAndPruneBroadPhase::validate() const
{
    const auto validateAxis = [this](CollisionAxis axis)
    {
        const std::vector<EndPoint*>& endPoints = endPointsForAxis(axis);
        for (std::size_t i = 0; i < endPoints.size(); ++i)
        {
            const EndPoint* endPoint = endPoints[i];
            if (endPoint == nullptr || endPoint->axis() != axis || endPoint->index() != i ||
                !endPoint->colliderId().valid())
            {
                return false;
            }

            if (i > 0 && lessThan(*endPoint, *endPoints[i - 1]))
            {
                return false;
            }
        }

        return true;
    };

    if (!validateAxis(CollisionAxis::X) || !validateAxis(CollisionAxis::Y) ||
        !validateAxis(CollisionAxis::Z))
    {
        return false;
    }

    for (std::size_t first = 0; first < colliders_.size(); ++first)
    {
        for (std::size_t second = first + 1; second < colliders_.size(); ++second)
        {
            const Collider& lhs = *colliders_[first].collider;
            const Collider& rhs = *colliders_[second].collider;
            unsigned char expectedMask = 0;
            if (canCollide(lhs, rhs) && overlapsOnAxis(lhs, rhs, CollisionAxis::X))
            {
                expectedMask |= axisBit(CollisionAxis::X);
            }
            if (canCollide(lhs, rhs) && overlapsOnAxis(lhs, rhs, CollisionAxis::Y))
            {
                expectedMask |= axisBit(CollisionAxis::Y);
            }
            if (canCollide(lhs, rhs) && overlapsOnAxis(lhs, rhs, CollisionAxis::Z))
            {
                expectedMask |= axisBit(CollisionAxis::Z);
            }

            const PairKey key = orderedPair(colliders_[first].id, colliders_[second].id);
            const auto found = pairStates_.find(key);
            if (expectedMask == 0)
            {
                if (found != pairStates_.end())
                {
                    return false;
                }
            }
            else if (found == pairStates_.end() || found->second.axisMask != expectedMask)
            {
                return false;
            }
        }
    }

    constexpr unsigned char allAxesMask = 1U | 2U | 4U;
    std::size_t expectedPossibleCount = 0;
    for (const auto& [key, pairState] : pairStates_)
    {
        const bool expectedPossible = pairState.axisMask == allAxesMask;
        if (pairState.possible != expectedPossible)
        {
            return false;
        }

        if (expectedPossible)
        {
            ++expectedPossibleCount;
            const auto found = std::find_if(
                possiblePairs_.begin(), possiblePairs_.end(), [key](const CollisionPair& pair)
                { return pair.firstId == key.first && pair.secondId == key.second; });
            if (found == possiblePairs_.end())
            {
                return false;
            }
        }
    }

    return possiblePairs_.size() == expectedPossibleCount;
}

void SweepAndPruneBroadPhase::addEndPointsSorted(Collider& collider)
{
    insertEndPointSorted(&collider.minEndPoint(CollisionAxis::X), xEndPoints_);
    insertEndPointSorted(&collider.maxEndPoint(CollisionAxis::X), xEndPoints_);
    insertEndPointSorted(&collider.minEndPoint(CollisionAxis::Y), yEndPoints_);
    insertEndPointSorted(&collider.maxEndPoint(CollisionAxis::Y), yEndPoints_);
    insertEndPointSorted(&collider.minEndPoint(CollisionAxis::Z), zEndPoints_);
    insertEndPointSorted(&collider.maxEndPoint(CollisionAxis::Z), zEndPoints_);
}

void SweepAndPruneBroadPhase::insertEndPointSorted(EndPoint* endPoint,
                                                   std::vector<EndPoint*>& endPoints)
{
    const auto pos = std::lower_bound(endPoints.begin(), endPoints.end(), endPoint,
                                      [](const EndPoint* lhs, const EndPoint* rhs)
                                      { return lessThan(*lhs, *rhs); });
    const auto idx = static_cast<std::size_t>(pos - endPoints.begin());
    endPoints.insert(pos, endPoint);
    // Re-index everything from the insertion point onward.
    for (std::size_t i = idx; i < endPoints.size(); ++i)
    {
        endPoints[i]->index(i);
    }
}

void SweepAndPruneBroadPhase::sweepAxisForCollider(std::span<EndPoint* const> endPoints,
                                                   const Collider* target)
{
    std::vector<const Collider*> active;
    active.reserve(colliders_.size());
    bool targetActive = false;

    for (const EndPoint* endPoint : endPoints)
    {
        const Collider* current = endPoint->collider();
        if (endPoint->isMin())
        {
            if (current == target)
            {
                // Target's interval opens — every collider in `active`
                // already overlaps it on this axis.
                for (const Collider* a : active)
                {
                    updatePairAxis(a, target, endPoint->axis(), true);
                }
                targetActive = true;
            }
            else if (targetActive)
            {
                // A different collider's interval opens while target is open
                // — they overlap on this axis.
                updatePairAxis(current, target, endPoint->axis(), true);
            }
            active.push_back(current);
        }
        else
        {
            const auto found = std::find(active.begin(), active.end(), current);
            if (found != active.end())
            {
                active.erase(found);
            }
            if (current == target)
            {
                // Once the target's interval closes nothing further on this
                // axis can overlap it.
                break;
            }
        }
    }
}

void SweepAndPruneBroadPhase::removeEndPoints(Collider& collider)
{
    auto removeFromAxis = [](std::vector<EndPoint*>& endPoints, EndPoint& endPoint)
    {
        const auto found = std::find(endPoints.begin(), endPoints.end(), &endPoint);
        if (found != endPoints.end())
        {
            endPoints.erase(found);
            endPoint.index(EndPoint::invalidIndex);
            updateIndices(endPoints);
        }
    };

    removeFromAxis(xEndPoints_, collider.minEndPoint(CollisionAxis::X));
    removeFromAxis(xEndPoints_, collider.maxEndPoint(CollisionAxis::X));
    removeFromAxis(yEndPoints_, collider.minEndPoint(CollisionAxis::Y));
    removeFromAxis(yEndPoints_, collider.maxEndPoint(CollisionAxis::Y));
    removeFromAxis(zEndPoints_, collider.minEndPoint(CollisionAxis::Z));
    removeFromAxis(zEndPoints_, collider.maxEndPoint(CollisionAxis::Z));
}

void SweepAndPruneBroadPhase::sortAndIndexEndPoints(std::vector<EndPoint*>& endPoints)
{
    std::sort(endPoints.begin(), endPoints.end(),
              [](const EndPoint* lhs, const EndPoint* rhs) { return lessThan(*lhs, *rhs); });
    updateIndices(endPoints);
}

void SweepAndPruneBroadPhase::rebuildPairStates()
{
    pairStates_.clear();
    possiblePairs_.clear();
    sweepAxis(xEndPoints_);
    sweepAxis(yEndPoints_);
    sweepAxis(zEndPoints_);
}

void SweepAndPruneBroadPhase::sweepAxis(std::span<EndPoint* const> endPoints)
{
    std::vector<const Collider*> active;
    active.reserve(colliders_.size());

    for (const EndPoint* endPoint : endPoints)
    {
        const Collider* current = endPoint->collider();
        if (endPoint->isMin())
        {
            for (const Collider* activeCollider : active)
            {
                updatePairAxis(activeCollider, current, endPoint->axis(), true);
            }

            active.push_back(current);
            continue;
        }

        const auto found = std::find(active.begin(), active.end(), current);
        if (found != active.end())
        {
            active.erase(found);
        }
    }
}

void SweepAndPruneBroadPhase::updateEndPoint(EndPoint& endPoint, bool /*refreshPairs*/)
{
    std::vector<EndPoint*>& endPoints = endPointsForAxis(endPoint.axis());
    std::size_t index = endPoint.index();
    if (index >= endPoints.size() || endPoints[index] != &endPoint)
    {
        const auto found = std::find(endPoints.begin(), endPoints.end(), &endPoint);
        if (found == endPoints.end())
        {
            return;
        }

        index = static_cast<std::size_t>(std::distance(endPoints.begin(), found));
        endPoint.index(index);
    }

    while (index > 0 && lessThan(*endPoints[index], *endPoints[index - 1]))
    {
        EndPoint* moving = endPoints[index];
        EndPoint* crossed = endPoints[index - 1];
        if (moving->isMin() != crossed->isMin())
        {
            updatePairAxis(moving->collider(), crossed->collider(), moving->axis(),
                           moving->isMin() && !crossed->isMin());
        }

        std::swap(endPoints[index], endPoints[index - 1]);
        endPoints[index]->index(index);
        endPoints[index - 1]->index(index - 1);
        --index;
    }

    while (index + 1 < endPoints.size() && lessThan(*endPoints[index + 1], *endPoints[index]))
    {
        EndPoint* moving = endPoints[index];
        EndPoint* crossed = endPoints[index + 1];
        if (moving->isMin() != crossed->isMin())
        {
            updatePairAxis(moving->collider(), crossed->collider(), moving->axis(),
                           !moving->isMin() && crossed->isMin());
        }

        std::swap(endPoints[index], endPoints[index + 1]);
        endPoints[index]->index(index);
        endPoints[index + 1]->index(index + 1);
        ++index;
    }
}

void SweepAndPruneBroadPhase::updatePairAxis(const Collider* lhs, const Collider* rhs,
                                             CollisionAxis axis, bool overlaps)
{
    if (lhs == rhs || lhs == nullptr || rhs == nullptr || !canCollide(*lhs, *rhs))
    {
        return;
    }

    const PairKey key = orderedPair(lhs, rhs);
    auto found = pairStates_.find(key);
    const unsigned char bit = axisBit(axis);

    if (found == pairStates_.end())
    {
        if (!overlaps)
        {
            return;
        }

        const Collider* first = lhs->colliderId() == key.first ? lhs : rhs;
        const Collider* second = lhs->colliderId() == key.first ? rhs : lhs;
        PairState newState{};
        newState.first = first;
        newState.second = second;
        newState.axisMask = bit;
        const auto [inserted, _] = pairStates_.emplace(key, newState);
        found = inserted;
    }
    else if (overlaps)
    {
        found->second.axisMask |= bit;
    }
    else
    {
        found->second.axisMask &= static_cast<unsigned char>(~bit);
    }

    if (found->second.axisMask == 0)
    {
        // No axis still claims this pair — drop the possible-list entry first
        // (using the iterator we still hold) and then erase the map entry.
        if (found->second.possible)
        {
            found->second.possible = false;
            removePossiblePair(found->second);
        }
        pairStates_.erase(found);
        return;
    }

    constexpr unsigned char allAxesMask = 1U | 2U | 4U;
    const bool nowPossible = found->second.axisMask == allAxesMask;
    if (nowPossible && !found->second.possible)
    {
        found->second.possible = true;
        addPossiblePair(key, found->second);
    }
    else if (!nowPossible && found->second.possible)
    {
        found->second.possible = false;
        removePossiblePair(found->second);
    }
}

void SweepAndPruneBroadPhase::addPossiblePair(PairKey key, PairState& state)
{
    if (state.possibleIndex != kInvalidPairIndex)
    {
        return;
    }
    state.possibleIndex = possiblePairs_.size();
    possiblePairs_.push_back({key.first, key.second, state.first, state.second});
}

void SweepAndPruneBroadPhase::removePossiblePair(PairState& state)
{
    if (state.possibleIndex == kInvalidPairIndex)
    {
        return;
    }

    const std::size_t idx = state.possibleIndex;
    const std::size_t last = possiblePairs_.size() - 1;

    if (idx != last)
    {
        // Swap-and-pop — fix the swapped pair's stored index in its PairState.
        possiblePairs_[idx] = possiblePairs_[last];
        const PairKey swappedKey =
            orderedPair(possiblePairs_[idx].firstId, possiblePairs_[idx].secondId);
        const auto found = pairStates_.find(swappedKey);
        if (found != pairStates_.end())
        {
            found->second.possibleIndex = idx;
        }
    }

    possiblePairs_.pop_back();
    state.possibleIndex = kInvalidPairIndex;
}

void SweepAndPruneBroadPhase::removePairStatesFor(ColliderId colliderId)
{
    for (auto it = pairStates_.begin(); it != pairStates_.end();)
    {
        if (it->first.first == colliderId || it->first.second == colliderId)
        {
            if (it->second.possible)
            {
                removePossiblePair(it->second);
            }
            it = pairStates_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

std::vector<EndPoint*>& SweepAndPruneBroadPhase::endPointsForAxis(CollisionAxis axis) noexcept
{
    switch (axis)
    {
    case CollisionAxis::X:
        return xEndPoints_;
    case CollisionAxis::Y:
        return yEndPoints_;
    case CollisionAxis::Z:
        return zEndPoints_;
    }

    return xEndPoints_;
}

const std::vector<EndPoint*>&
SweepAndPruneBroadPhase::endPointsForAxis(CollisionAxis axis) const noexcept
{
    switch (axis)
    {
    case CollisionAxis::X:
        return xEndPoints_;
    case CollisionAxis::Y:
        return yEndPoints_;
    case CollisionAxis::Z:
        return zEndPoints_;
    }

    return xEndPoints_;
}

ColliderId SweepAndPruneBroadPhase::colliderId(const Collider& collider) const noexcept
{
    for (const ColliderEntry& entry : colliders_)
    {
        if (entry.collider == &collider)
        {
            return entry.id;
        }
    }

    return {};
}

SweepAndPruneBroadPhase::PairKey SweepAndPruneBroadPhase::orderedPair(ColliderId lhs,
                                                                      ColliderId rhs) noexcept
{
    if (lhs < rhs)
    {
        return {lhs, rhs};
    }

    return {rhs, lhs};
}

SweepAndPruneBroadPhase::PairKey SweepAndPruneBroadPhase::orderedPair(const Collider* lhs,
                                                                      const Collider* rhs) noexcept
{
    return orderedPair(lhs->colliderId(), rhs->colliderId());
}

bool SweepAndPruneBroadPhase::lessThan(const EndPoint& lhs, const EndPoint& rhs) noexcept
{
    if (lhs.value() != rhs.value())
    {
        return lhs.value() < rhs.value();
    }

    if (lhs.isMin() != rhs.isMin())
    {
        return lhs.isMin();
    }

    if (lhs.colliderId() != rhs.colliderId())
    {
        return lhs.colliderId() < rhs.colliderId();
    }

    return false;
}

bool SweepAndPruneBroadPhase::overlapsOnAxis(const Collider& lhs, const Collider& rhs,
                                             CollisionAxis axis) noexcept
{
    const AABB lhsBounds = lhs.sweptWorldBounds();
    const AABB rhsBounds = rhs.sweptWorldBounds();
    return lhsBounds.axisMin(axis) <= rhsBounds.axisMax(axis) &&
           lhsBounds.axisMax(axis) >= rhsBounds.axisMin(axis);
}

bool SweepAndPruneBroadPhase::canCollide(const Collider& lhs, const Collider& rhs) noexcept
{
    return (lhs.collisionMask() & rhs.collisionLayer()) != 0U &&
           (rhs.collisionMask() & lhs.collisionLayer()) != 0U;
}

unsigned char SweepAndPruneBroadPhase::axisBit(CollisionAxis axis) noexcept
{
    switch (axis)
    {
    case CollisionAxis::X:
        return 1U;
    case CollisionAxis::Y:
        return 2U;
    case CollisionAxis::Z:
        return 4U;
    }

    return 1U;
}

void SweepAndPruneBroadPhase::updateIndices(std::vector<EndPoint*>& endPoints) noexcept
{
    for (std::size_t i = 0; i < endPoints.size(); ++i)
    {
        endPoints[i]->index(i);
    }
}

void SweepAndPruneBroadPhase::resetIndices(std::vector<EndPoint*>& endPoints) noexcept
{
    for (EndPoint* endPoint : endPoints)
    {
        endPoint->index(EndPoint::invalidIndex);
    }
}

std::size_t SweepAndPruneBroadPhase::PairKeyHash::operator()(PairKey key) const noexcept
{
    constexpr std::size_t multiplier = 0x9E3779B97F4A7C15ULL;
    return static_cast<std::size_t>(key.first.value()) * multiplier ^
           static_cast<std::size_t>(key.second.value());
}

} // namespace fire_engine
