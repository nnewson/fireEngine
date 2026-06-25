#include <fire_engine/collision/dynamic_aabb_tree_broad_phase.hpp>

#include <algorithm>
#include <vector>

namespace fire_engine
{

namespace
{

[[nodiscard]] bool canCollide(const Collider& a, const Collider& b) noexcept
{
    return (a.collisionMask() & b.collisionLayer()) != 0U &&
           (b.collisionMask() & a.collisionLayer()) != 0U;
}

} // namespace

ColliderId DynamicAabbTreeBroadPhase::addCollider(Collider& collider)
{
    const ColliderId id = nextColliderId_;
    nextColliderId_ = ColliderId{nextColliderId_.value() + 1U};
    collider.colliderId(id);

    const int proxy = bvh_.createProxy(collider.sweptWorldBounds(), &collider);
    proxyByColliderId_.emplace(id.value(), proxy);
    return id;
}

bool DynamicAabbTreeBroadPhase::removeCollider(ColliderId colliderId)
{
    const auto it = proxyByColliderId_.find(colliderId.value());
    if (it == proxyByColliderId_.end())
    {
        return false;
    }
    bvh_.destroyProxy(it->second);
    proxyByColliderId_.erase(it);
    return true;
}

bool DynamicAabbTreeBroadPhase::removeCollider(Collider& collider)
{
    return removeCollider(collider.colliderId());
}

void DynamicAabbTreeBroadPhase::clear()
{
    bvh_.clear();
    nextColliderId_ = ColliderId{1U};
    proxyByColliderId_.clear();
    possiblePairs_.clear();
}

void DynamicAabbTreeBroadPhase::update()
{
    // Refresh any proxy whose collider left its fat box, then recompute the pair list.
    for (const auto& [idValue, proxy] : proxyByColliderId_)
    {
        bvh_.moveProxy(proxy, bvh_.payload(proxy)->sweptWorldBounds());
    }
    regeneratePairs();
}

void DynamicAabbTreeBroadPhase::rebuild()
{
    // Full rebuild: re-create every proxy from current bounds, then the pair list.
    std::vector<Collider*> colliders;
    colliders.reserve(proxyByColliderId_.size());
    for (const auto& [idValue, proxy] : proxyByColliderId_)
    {
        colliders.push_back(bvh_.payload(proxy));
    }

    bvh_.clear();
    proxyByColliderId_.clear();
    for (Collider* collider : colliders)
    {
        const int proxy = bvh_.createProxy(collider->sweptWorldBounds(), collider);
        proxyByColliderId_.emplace(collider->colliderId().value(), proxy);
    }
    regeneratePairs();
}

void DynamicAabbTreeBroadPhase::regeneratePairs()
{
    possiblePairs_.clear();
    for (const auto& [idValue, proxy] : proxyByColliderId_)
    {
        Collider* a = bvh_.payload(proxy);
        const ColliderId aid = a->colliderId();
        bvh_.query(bvh_.fatBounds(proxy),
                   [&](int other)
                   {
                       Collider* b = bvh_.payload(other);
                       const ColliderId bid = b->colliderId();
                       // Emit each unordered pair once (id strictly greater), only for a
                       // real swept-bounds overlap that passes the layer/mask filter.
                       if (aid < bid &&
                           aabbOverlaps(a->sweptWorldBounds(), b->sweptWorldBounds()) &&
                           canCollide(*a, *b))
                       {
                           possiblePairs_.push_back(CollisionPair{aid, bid, a, b});
                       }
                   });
    }
    // Sort by (firstId, secondId) so the list is independent of tree shape / map order.
    std::ranges::sort(possiblePairs_,
                      [](const CollisionPair& lhs, const CollisionPair& rhs)
                      {
                          if (lhs.firstId != rhs.firstId)
                          {
                              return lhs.firstId < rhs.firstId;
                          }
                          return lhs.secondId < rhs.secondId;
                      });
}

bool DynamicAabbTreeBroadPhase::validate() const
{
    // Brute-force O(n²): every swept-bounds-overlapping, collidable pair must appear
    // exactly once in the maintained list.
    std::vector<Collider*> colliders;
    colliders.reserve(proxyByColliderId_.size());
    for (const auto& [idValue, proxy] : proxyByColliderId_)
    {
        colliders.push_back(bvh_.payload(proxy));
    }

    std::size_t expected = 0;
    for (std::size_t i = 0; i < colliders.size(); ++i)
    {
        for (std::size_t j = i + 1; j < colliders.size(); ++j)
        {
            if (aabbOverlaps(colliders[i]->sweptWorldBounds(), colliders[j]->sweptWorldBounds()) &&
                canCollide(*colliders[i], *colliders[j]))
            {
                ++expected;
            }
        }
    }
    return expected == possiblePairs_.size();
}

} // namespace fire_engine
