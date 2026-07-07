#include <fire_engine/physics/physics_world.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <fire_engine/collision/broad_phase.hpp>
#include <fire_engine/collision/narrow_phase.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec4.hpp>
#include <fire_engine/physics/articulation_contact.hpp>
#include <fire_engine/physics/physics_constants.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] float maxJointRate(const Articulation& art) noexcept
{
    float maxRate = 0.0f;
    for (const float qDot : art.qDot())
    {
        maxRate = std::max(maxRate, std::abs(qDot));
    }
    return maxRate;
}

[[nodiscard]] bool belowArticulationSleepThreshold(const Articulation& art,
                                                   float angularThreshold) noexcept
{
    const SpatialVector baseVel = art.baseVelocity();
    return baseVel.linear.magnitudeSquared() < kLinearSleepThreshold * kLinearSleepThreshold &&
           baseVel.angular.magnitudeSquared() < angularThreshold * angularThreshold &&
           maxJointRate(art) < angularThreshold;
}

void zeroArticulationVelocity(Articulation& art) noexcept
{
    art.baseVelocity(SpatialVector{});
    for (int dof = 0; dof < art.dofCount(); ++dof)
    {
        art.qDot(dof, 0.0f);
    }
}

} // namespace

PhysicsArticulationHandle PhysicsWorld::createArticulation()
{
    const GenerationalSlotPool::Slot slot = articulationSlots_.acquire();
    const PhysicsArticulationHandle handle =
        PhysicsArticulationHandle::make(slot.index, slot.generation);
    // The sleep arrays are 1:1 with articulations_ by slot. Grow all three in lockstep, or reset
    // the recycled slot's payload (a fresh 0-link articulation) + sleep state.
    if (slot.index >= articulations_.size())
    {
        articulations_.emplace_back();
        articulationSleepTimers_.push_back(0.0f);
        articulationSleeping_.push_back(0U);
    }
    else
    {
        articulations_[slot.index] = Articulation{};
        articulationSleepTimers_[slot.index] = 0.0f;
        articulationSleeping_[slot.index] = 0U;
    }
    return handle;
}

bool PhysicsWorld::destroyArticulation(PhysicsArticulationHandle handle)
{
    if (findArticulation(handle) == nullptr)
    {
        return false;
    }
    // Destroy every collider owned by this articulation's links (removes them from the broadphase
    // and recycles their collider slots). Marking active=false leaves the deque entry in place.
    for (ColliderEntry& entry : colliders_)
    {
        if (entry.active && entry.articulation == handle)
        {
            deactivateCollider(entry);
        }
    }
    const std::uint32_t index = handle.index();
    articulations_[index] = Articulation{}; // release links; a 0-link articulation is inert
    articulationSleepTimers_[index] = 0.0f;
    articulationSleeping_[index] = 0U;
    articulationSlots_.release(index); // bumps generation → stale handles detectably invalid
    return true;
}

Articulation* PhysicsWorld::findArticulation(PhysicsArticulationHandle handle) noexcept
{
    const std::uint32_t index = handle.index();
    return articulationSlots_.valid(index, handle.generation()) ? &articulations_[index] : nullptr;
}

const Articulation* PhysicsWorld::findArticulation(PhysicsArticulationHandle handle) const noexcept
{
    const std::uint32_t index = handle.index();
    return articulationSlots_.valid(index, handle.generation()) ? &articulations_[index] : nullptr;
}

Articulation* PhysicsWorld::articulation(PhysicsArticulationHandle handle) noexcept
{
    return findArticulation(handle);
}

const Articulation* PhysicsWorld::articulation(PhysicsArticulationHandle handle) const noexcept
{
    return findArticulation(handle);
}

std::size_t PhysicsWorld::articulationCount() const noexcept
{
    return articulationSlots_.liveCount();
}

PhysicsColliderHandle PhysicsWorld::attachLinkCollider(PhysicsArticulationHandle handle, int link,
                                                       const ColliderDesc& desc)
{
    const Articulation* art = findArticulation(handle);
    if (art == nullptr || link < 0 || static_cast<std::size_t>(link) >= art->linkCount())
    {
        return PhysicsColliderHandle{};
    }
    return addLinkColliderEntry(handle, link, desc.shape, desc.material, desc.collisionLayer,
                                desc.collisionMask, Vec3{}, desc.localRotation, desc.isTrigger);
}

PhysicsColliderHandle PhysicsWorld::addLinkColliderEntry(
    PhysicsArticulationHandle articulationHandle, int link, const ColliderShape& shape,
    const PhysicsMaterial& material, std::uint32_t collisionLayer, std::uint32_t collisionMask,
    const Vec3& localPosition, const Quaternion& localRotation, bool isTrigger)
{
    const GenerationalSlotPool::Slot slot = colliderSlots_.acquire();
    const PhysicsColliderHandle handle = PhysicsColliderHandle::make(slot.index, slot.generation);

    Collider collider;
    const Mat4 childMat = Mat4::translate(localPosition) * localRotation.toMat4();
    collider.localBounds(localBounds(shape).transformedBy(childMat));
    collider.collisionLayer(collisionLayer);
    collider.collisionMask(collisionMask);
    collider.isTrigger(isTrigger);

    ColliderEntry newEntry{handle,
                           PhysicsBodyHandle{},
                           articulationHandle,
                           link,
                           std::move(collider),
                           shape,
                           material,
                           localPosition,
                           localRotation,
                           true,
                           nullptr};
    if (slot.index >= colliders_.size())
    {
        colliders_.push_back(std::move(newEntry));
    }
    else
    {
        colliders_[slot.index] = std::move(newEntry);
    }
    ColliderEntry& entry = colliders_[slot.index];
    colliderIndexByPointer_.emplace(&entry.collider, slot.index);
    // Seed the swept bound from the link's current forward-kinematics pose.
    const OwnerPose owner = colliderOwnerPose(entry);
    entry.collider.resetFrame(owner.world);
    broadPhase_->addCollider(entry.collider);
    return handle;
}

void PhysicsWorld::stepArticulations(float dt)
{
    if (articulationSleepTimers_.size() != articulations_.size())
    {
        articulationSleepTimers_.resize(articulations_.size(), 0.0f);
    }
    if (articulationSleeping_.size() != articulations_.size())
    {
        articulationSleeping_.resize(articulations_.size(), 0U);
    }

    // Gather each articulation's contacts from this step's broadphase pairs. A link collider
    // paired with a *static* rigid collider yields one plane contact per manifold point,
    // stored in link-local space so the substep loop tracks it as the link moves. Link-vs-
    // dynamic-rigid and link-vs-link (self-collision) are deferred to the full pipeline.
    std::vector<std::vector<ArticulationPlaneContact>> perArticulation(articulations_.size());
    std::vector<std::vector<ArticulationLinkContact>> perArticulationLink(articulations_.size());

    for (const CollisionPair& pair : broadPhase_->possiblePairs())
    {
        ColliderEntry* first = findCollider(pair.first);
        ColliderEntry* second = findCollider(pair.second);
        if (first == nullptr || second == nullptr)
        {
            continue;
        }

        // Self-collision: both colliders are links of the *same* articulation. Skip adjacent
        // (parent-child) bones — they share a joint and always overlap there — and solve the rest
        // as link-vs-link contacts so limbs stack into a plausible pose instead of folding through
        // each other.
        if (first->isLinkCollider() && second->isLinkCollider())
        {
            if (first->articulation.value() != second->articulation.value())
            {
                continue; // different articulations (cross-ragdoll collision) — deferred
            }
            Articulation* artPtr = findArticulation(first->articulation);
            if (artPtr == nullptr)
            {
                continue;
            }
            Articulation& art = *artPtr;
            const auto a = static_cast<std::size_t>(first->link);
            const auto b = static_cast<std::size_t>(second->link);
            if (a >= art.linkCount() || b >= art.linkCount())
            {
                continue;
            }
            if (art.parent(a) == static_cast<int>(b) || art.parent(b) == static_cast<int>(a) ||
                art.selfCollisionExcluded(a, b))
            {
                continue; // adjacent or rest-pose-overlapping bones — not a real self-contact
            }
            const auto manifold = narrowPhase_->collide(worldShape(*first), worldShape(*second),
                                                        kSpeculativeDistance);
            if (!manifold.has_value() || manifold->count == 0)
            {
                continue;
            }
            const RigidTransform invA = art.linkWorld(a).inverse();
            const RigidTransform invB = art.linkWorld(b).inverse();
            const float friction = std::sqrt(std::max(first->material.friction, 0.0f) *
                                             std::max(second->material.friction, 0.0f));
            for (int p = 0; p < manifold->count; ++p)
            {
                const Vec3 wp = manifold->points[static_cast<std::size_t>(p)].position;
                const float pen = manifold->points[static_cast<std::size_t>(p)].penetration;
                // normal points B(second) -> A(first); offset = pen so prepare-time sep = -pen.
                perArticulationLink[first->articulation.index()].push_back(
                    ArticulationLinkContact{a, b, invA.transformPoint(wp), invB.transformPoint(wp),
                                            manifold->normal, pen, friction});
            }
            continue;
        }

        ColliderEntry* link = nullptr;
        ColliderEntry* other = nullptr;
        if (first->isLinkCollider() && !second->isLinkCollider())
        {
            link = first;
            other = second;
        }
        else if (second->isLinkCollider() && !first->isLinkCollider())
        {
            link = second;
            other = first;
        }
        else
        {
            continue; // neither is a link — not this pass
        }

        const BodyEntry* otherBody = findBody(other->body);
        if (otherBody == nullptr || otherBody->body.type() != PhysicsBodyType::Static)
        {
            continue; // link-vs-static only for now
        }

        Articulation* artPtr = findArticulation(link->articulation);
        if (artPtr == nullptr)
        {
            continue;
        }
        Articulation& art = *artPtr;
        const auto linkIndex = static_cast<std::size_t>(link->link);
        if (link->link < 0 || linkIndex >= art.linkCount())
        {
            continue;
        }

        // Manifold with the normal pointing other -> link (the direction to push the link
        // out of the static surface).
        const auto manifold =
            narrowPhase_->collide(worldShape(*link), worldShape(*other), kSpeculativeDistance);
        if (!manifold.has_value() || manifold->count == 0)
        {
            continue;
        }

        const RigidTransform linkInv = art.linkWorld(linkIndex).inverse();
        const float friction = std::sqrt(std::max(link->material.friction, 0.0f) *
                                         std::max(other->material.friction, 0.0f));
        for (int p = 0; p < manifold->count; ++p)
        {
            const Vec3 worldPoint = manifold->points[static_cast<std::size_t>(p)].position;
            const float penetration = manifold->points[static_cast<std::size_t>(p)].penetration;
            // Plane through the contact point with the manifold normal; offset chosen so the
            // prepare-time separation is −penetration (dot(n, wp) − offset = −penetration).
            perArticulation[link->articulation.index()].push_back(ArticulationPlaneContact{
                linkIndex, linkInv.transformPoint(worldPoint), manifold->normal,
                Vec3::dotProduct(manifold->normal, worldPoint) + penetration, friction});
        }
    }

    // Step every articulation (contacts or not — they all fall under gravity).
    for (std::size_t i = 0; i < articulations_.size(); ++i)
    {
        Articulation& art = articulations_[i];
        if (!sleepingEnabled_)
        {
            articulationSleepTimers_[i] = 0.0f;
            articulationSleeping_[i] = 0U;
            stepArticulationOnPlanes(art, perArticulation[i], gravity_, dt, kArticulationDamping,
                                     perArticulationLink[i]);
            continue;
        }

        if (articulationSleeping_[i] != 0U)
        {
            if (belowArticulationSleepThreshold(art, kArticulationAngularSleepThreshold))
            {
                zeroArticulationVelocity(art);
                continue;
            }
            articulationSleeping_[i] = 0U;
            articulationSleepTimers_[i] = 0.0f;
        }

        stepArticulationOnPlanes(art, perArticulation[i], gravity_, dt, kArticulationDamping,
                                 perArticulationLink[i]);

        if (belowArticulationSleepThreshold(art, kArticulationAngularSleepThreshold))
        {
            articulationSleepTimers_[i] += dt;
        }
        else
        {
            articulationSleepTimers_[i] = 0.0f;
        }

        if (articulationSleepTimers_[i] >= kSleepTime)
        {
            zeroArticulationVelocity(art);
            articulationSleeping_[i] = 1U;
        }
    }
}

} // namespace fire_engine
