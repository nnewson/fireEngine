#include <fire_engine/physics/physics_world.hpp>

#include <cstdint>
#include <type_traits>
#include <vector>

#include <fire_engine/physics/contact.hpp>

namespace fire_engine
{

void PhysicsWorld::captureDebugContacts(std::span<const SolverContact> contacts)
{
    debugContacts_.clear();
    for (const SolverContact& contact : contacts)
    {
        // Real manifold points + normal (target -> moving) from the narrowphase.
        // Speculative gap points (penetration < 0) are predictions, not actual
        // contacts, so they are filtered out of the debug view.
        for (int i = 0; i < contact.manifold.count; ++i)
        {
            if (contact.manifold.points[i].penetration < 0.0f)
            {
                continue;
            }
            debugContacts_.push_back(
                DebugContact{contact.manifold.points[i].position, contact.manifold.normal});
        }
    }
}

std::vector<AABB> PhysicsWorld::debugColliderBounds() const
{
    std::vector<AABB> bounds;
    bounds.reserve(colliderCount());
    for (const ColliderEntry& entry : colliders_)
    {
        if (entry.active)
        {
            bounds.push_back(entry.collider.worldBounds());
        }
    }
    return bounds;
}

std::vector<DebugJointAnchor> PhysicsWorld::debugJointAnchors() const
{
    std::vector<DebugJointAnchor> anchors;
    anchors.reserve(jointCount());
    for (const JointEntry& entry : joints_)
    {
        if (!entry.active)
        {
            continue;
        }

        const BodyEntry* a = findBody(entry.desc.bodyA);
        const BodyEntry* b = findBody(entry.desc.bodyB);
        if (a == nullptr || b == nullptr || !a->active || !b->active)
        {
            continue;
        }

        const Mat3 ra = Mat3::fromQuaternion(a->transform.rotation());
        const Mat3 rb = Mat3::fromQuaternion(b->transform.rotation());
        anchors.push_back(DebugJointAnchor{a->transform.position(), b->transform.position(),
                                           a->transform.position() + ra * entry.desc.anchorA,
                                           b->transform.position() + rb * entry.desc.anchorB});
    }
    return anchors;
}

std::vector<std::uint8_t> PhysicsWorld::debugColliderSleeping() const
{
    // Mirror gatherColliders()'s iteration + skip conditions exactly so the flags
    // line up one-to-one with its emitted shapes. Keep the two in sync.
    std::vector<std::uint8_t> out;
    out.reserve(colliderCount());
    for (const ColliderEntry& entry : colliders_)
    {
        if (!entry.active || entry.mesh != nullptr || findBody(entry.body) == nullptr)
        {
            continue;
        }
        // Convex hulls have no ClothCollider encoding (gatherColliders skips them).
        const bool encodable =
            std::visit([](const auto& shape)
                       { return !std::is_same_v<std::decay_t<decltype(shape)>, WorldConvex>; },
                       worldShape(entry));
        if (!encodable)
        {
            continue;
        }
        out.push_back(sleeping(entry.body) ? std::uint8_t{1} : std::uint8_t{0});
    }
    return out;
}

} // namespace fire_engine
