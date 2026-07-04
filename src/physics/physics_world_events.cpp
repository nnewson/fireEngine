#include <fire_engine/physics/physics_world.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/physics/collision_event.hpp>

namespace fire_engine
{

std::vector<ClothCollider> PhysicsWorld::gatherColliders() const
{
    std::vector<ClothCollider> out;
    out.reserve(colliderCount());

    for (const ColliderEntry& entry : colliders_)
    {
        // Mesh colliders have no cloth-collider encoding (the AabbShape proxy is only
        // a broadphase bound); the cloth solver skips them.
        if (!entry.active || entry.mesh != nullptr || findBody(entry.body) == nullptr)
        {
            continue;
        }
        // Reuse the shared world-space composition, then encode as a ClothCollider.
        std::visit(
            [&out](const auto& shape)
            {
                using T = std::decay_t<decltype(shape)>;
                if constexpr (std::is_same_v<T, WorldSphere>)
                {
                    out.push_back(makeSphereCollider(shape.center, shape.radius));
                }
                else if constexpr (std::is_same_v<T, WorldBox>)
                {
                    out.push_back(
                        makeBoxCollider(shape.center, shape.halfExtents, shape.orientation));
                }
                else if constexpr (std::is_same_v<T, WorldCapsule>)
                {
                    out.push_back(makeCapsuleCollider(shape.p0, shape.p1, shape.radius));
                }
                // WorldConvex has no ClothCollider encoding — the cloth solver only
                // supports plane/sphere/box/capsule, so convex hulls are skipped here.
            },
            worldShape(entry));
    }

    return out;
}

void PhysicsWorld::recordOverlap(PhysicsColliderHandle first, PhysicsColliderHandle second,
                                 bool trigger)
{
    const std::uint32_t lo = std::min(first.value(), second.value());
    const std::uint32_t hi = std::max(first.value(), second.value());
    const std::uint64_t key = (static_cast<std::uint64_t>(lo) << 32) | hi;
    (trigger ? triggerOverlaps_ : collisionOverlaps_).insert(key);
}

void PhysicsWorld::removeOverlapPairsForCollider(PhysicsColliderHandle collider)
{
    const auto containsCollider = [value = collider.value()](std::uint64_t key)
    {
        return static_cast<std::uint32_t>(key >> 32) == value ||
               static_cast<std::uint32_t>(key & 0xFFFFFFFFULL) == value;
    };
    const auto eraseMatching = [&containsCollider](std::unordered_set<std::uint64_t>& set)
    {
        for (auto it = set.begin(); it != set.end();)
        {
            if (containsCollider(*it))
            {
                it = set.erase(it);
            }
            else
            {
                ++it;
            }
        }
    };

    eraseMatching(triggerOverlaps_);
    eraseMatching(previousTriggerOverlaps_);
    eraseMatching(collisionOverlaps_);
    eraseMatching(previousCollisionOverlaps_);
}

void PhysicsWorld::updateOverlapEvents()
{
    const auto build = [](const std::unordered_set<std::uint64_t>& previous,
                          const std::unordered_set<std::uint64_t>& current,
                          std::vector<ContactEvent>& out)
    {
        out.clear();
        const auto event = [](std::uint64_t key, EventPhase phase)
        {
            return ContactEvent{
                PhysicsColliderHandle{static_cast<std::uint32_t>(key >> 32)},
                PhysicsColliderHandle{static_cast<std::uint32_t>(key & 0xFFFFFFFFULL)}, phase};
        };
        for (const std::uint64_t key : current)
        {
            out.push_back(
                event(key, previous.contains(key) ? EventPhase::Stay : EventPhase::Enter));
        }
        for (const std::uint64_t key : previous)
        {
            if (!current.contains(key))
            {
                out.push_back(event(key, EventPhase::Exit));
            }
        }
        // Deterministic order regardless of hash-set iteration (events don't affect the
        // simulation, but a stable order keeps consumers/tests reproducible).
        std::sort(out.begin(), out.end(),
                  [](const ContactEvent& a, const ContactEvent& b)
                  {
                      if (a.first.value() != b.first.value())
                      {
                          return a.first.value() < b.first.value();
                      }
                      return a.second.value() < b.second.value();
                  });
    };

    build(previousTriggerOverlaps_, triggerOverlaps_, triggerEvents_);
    build(previousCollisionOverlaps_, collisionOverlaps_, collisionEvents_);
    previousTriggerOverlaps_ = triggerOverlaps_;
    previousCollisionOverlaps_ = collisionOverlaps_;
}

} // namespace fire_engine
