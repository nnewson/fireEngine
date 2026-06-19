#include <fire_engine/physics/physics_world.hpp>

#include <algorithm>
#include <type_traits>
#include <variant>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

PhysicsBodyHandle PhysicsWorld::createBody(const PhysicsBodyDesc& desc)
{
    const PhysicsBodyHandle handle = nextBodyHandle_;
    nextBodyHandle_ = PhysicsBodyHandle{nextBodyHandle_.value() + 1U};

    PhysicsBody body;
    body.type(desc.type);
    body.mass(desc.mass);
    body.linearVelocity(desc.linearVelocity);
    body.angularVelocity(desc.angularVelocity);
    body.gravityScale(desc.gravityScale);
    body.material(desc.material);

    Transform transform;
    transform.position(desc.position);
    transform.rotation(desc.rotation);
    transform.scale(desc.scale);
    transform.update(Mat4::identity());

    bodyIndexByHandle_.emplace(handle.value(), bodies_.size());
    bodies_.push_back({handle, body, transform, desc.position, true, {}});
    return handle;
}

PhysicsColliderHandle PhysicsWorld::createCollider(PhysicsBodyHandle bodyHandle,
                                                   const ColliderDesc& desc)
{
    BodyEntry* owner = findBody(bodyHandle);
    if (owner == nullptr)
    {
        return {};
    }

    const PhysicsColliderHandle handle = nextColliderHandle_;
    nextColliderHandle_ = PhysicsColliderHandle{nextColliderHandle_.value() + 1U};

    Collider collider;
    collider.localBounds(localBounds(desc.shape));
    collider.collisionLayer(desc.collisionLayer);
    collider.collisionMask(desc.collisionMask);
    collider.resetFrame(owner->transform.world());

    colliderIndexByHandle_.emplace(handle.value(), colliders_.size());
    colliders_.push_back(
        {handle, bodyHandle, std::move(collider), desc.shape, desc.material, true});
    colliderIndexByPointer_.emplace(&colliders_.back().collider, colliders_.size() - 1);
    owner->colliders.push_back(handle);
    broadPhase_.addCollider(colliders_.back().collider);
    return handle;
}

bool PhysicsWorld::destroyBody(PhysicsBodyHandle handle)
{
    BodyEntry* bodyEntry = findBody(handle);
    if (bodyEntry == nullptr)
    {
        return false;
    }

    for (PhysicsColliderHandle colliderHandle : bodyEntry->colliders)
    {
        ColliderEntry* colliderEntry = findCollider(colliderHandle);
        if (colliderEntry != nullptr && colliderEntry->active)
        {
            [[maybe_unused]] const bool removed =
                broadPhase_.removeCollider(colliderEntry->collider);
            colliderEntry->active = false;
        }
    }

    bodyEntry->active = false;
    return true;
}

void PhysicsWorld::clear()
{
    broadPhase_.clear();
    bodies_.clear();
    colliders_.clear();
    bodyIndexByHandle_.clear();
    colliderIndexByHandle_.clear();
    colliderIndexByPointer_.clear();
    nextBodyHandle_ = PhysicsBodyHandle{1U};
    nextColliderHandle_ = PhysicsColliderHandle{1U};
}

void PhysicsWorld::step(float fixedDt)
{
    if (fixedDt <= 0.0f)
    {
        return;
    }

    for (BodyEntry& entry : bodies_)
    {
        if (!entry.active || entry.body.type() != PhysicsBodyType::Dynamic)
        {
            continue;
        }

        Vec3 velocity = entry.body.linearVelocity();
        velocity += gravity_ * entry.body.gravityScale() * fixedDt;
        entry.body.linearVelocity(velocity);
        entry.transform.position(entry.transform.position() + velocity * fixedDt);
        entry.transform.update(Mat4::identity());
    }

    updateColliders();
    broadPhase_.update();

    auto frameContacts = contacts();
    // Snapshot for debug draw before applyResponses advances bodies to the TOI.
    captureDebugContacts(frameContacts);
    if (applyResponses(frameContacts))
    {
        resetResolvedColliders();
        broadPhase_.rebuild();
    }

    capturePreviousPositions();
}

void PhysicsWorld::captureDebugContacts(const std::vector<SolverContact>& contacts)
{
    debugContacts_.clear();
    for (const SolverContact& contact : contacts)
    {
        // Real manifold points + normal (target -> moving) from the narrowphase.
        for (int i = 0; i < contact.manifold.count; ++i)
        {
            debugContacts_.push_back(
                DebugContact{contact.manifold.points[i].position, contact.manifold.normal});
        }
    }
}

std::vector<AABB> PhysicsWorld::debugColliderBounds() const
{
    std::vector<AABB> bounds;
    bounds.reserve(colliders_.size());
    for (const ColliderEntry& entry : colliders_)
    {
        if (entry.active)
        {
            bounds.push_back(entry.collider.worldBounds());
        }
    }
    return bounds;
}

bool PhysicsWorld::valid(PhysicsBodyHandle handle) const noexcept
{
    return findBody(handle) != nullptr;
}

bool PhysicsWorld::valid(PhysicsColliderHandle handle) const noexcept
{
    return findCollider(handle) != nullptr;
}

std::size_t PhysicsWorld::bodyCount() const noexcept
{
    return static_cast<std::size_t>(
        std::ranges::count_if(bodies_, [](const BodyEntry& entry) { return entry.active; }));
}

std::size_t PhysicsWorld::colliderCount() const noexcept
{
    return static_cast<std::size_t>(
        std::ranges::count_if(colliders_, [](const ColliderEntry& entry) { return entry.active; }));
}

namespace
{

[[nodiscard]] Vec3 transformPoint(const Mat4& m, Vec3 p)
{
    const Vec4 r = m * Vec4{p.x(), p.y(), p.z(), 1.0f};
    return {r.x(), r.y(), r.z()};
}

// Per-axis scale from the world matrix columns.
[[nodiscard]] Vec3 matrixScale(const Mat4& m)
{
    return {Vec3{m[0, 0], m[1, 0], m[2, 0]}.magnitude(),
            Vec3{m[0, 1], m[1, 1], m[2, 1]}.magnitude(),
            Vec3{m[0, 2], m[1, 2], m[2, 2]}.magnitude()};
}

} // namespace

WorldShape PhysicsWorld::worldShape(const ColliderEntry& entry) const
{
    // Identity when the owner is missing (shouldn't happen for active colliders).
    const BodyEntry* owner = findBody(entry.body);
    const Mat4 world = owner != nullptr ? owner->transform.world() : Mat4::identity();
    const Quaternion rot = owner != nullptr ? owner->transform.rotation() : Quaternion::identity();
    const Vec3 s = matrixScale(world);
    const float uniform = std::max({s.x(), s.y(), s.z()});

    if (const auto* sphere = std::get_if<SphereShape>(&entry.shape))
    {
        return WorldSphere{transformPoint(world, sphere->center), sphere->radius * uniform};
    }
    if (const auto* box = std::get_if<BoxShape>(&entry.shape))
    {
        return WorldBox{transformPoint(world, box->center),
                        Vec3{box->halfExtents.x() * s.x(), box->halfExtents.y() * s.y(),
                             box->halfExtents.z() * s.z()},
                        rot};
    }
    if (const auto* capsule = std::get_if<CapsuleShape>(&entry.shape))
    {
        const Vec3 c = capsule->center;
        const Vec3 p0 = transformPoint(world, Vec3{c.x(), c.y() - capsule->halfHeight, c.z()});
        const Vec3 p1 = transformPoint(world, Vec3{c.x(), c.y() + capsule->halfHeight, c.z()});
        return WorldCapsule{p0, p1, capsule->radius * uniform};
    }
    const auto& aabb = std::get<AabbShape>(entry.shape);
    const Vec3 he = aabb.bounds.extent() * 0.5f;
    return WorldBox{transformPoint(world, aabb.bounds.center()),
                    Vec3{he.x() * s.x(), he.y() * s.y(), he.z() * s.z()}, rot};
}

std::vector<ClothCollider> PhysicsWorld::gatherColliders() const
{
    std::vector<ClothCollider> out;
    out.reserve(colliders_.size());

    for (const ColliderEntry& entry : colliders_)
    {
        if (!entry.active || findBody(entry.body) == nullptr)
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
                    out.push_back(makeBoxCollider(shape.center, shape.halfExtents, shape.orientation));
                }
                else // WorldCapsule
                {
                    out.push_back(makeCapsuleCollider(shape.p0, shape.p1, shape.radius));
                }
            },
            worldShape(entry));
    }

    return out;
}

const PhysicsBody* PhysicsWorld::body(PhysicsBodyHandle handle) const noexcept
{
    const BodyEntry* entry = findBody(handle);
    return entry == nullptr ? nullptr : &entry->body;
}

PhysicsBody* PhysicsWorld::body(PhysicsBodyHandle handle) noexcept
{
    BodyEntry* entry = findBody(handle);
    return entry == nullptr ? nullptr : &entry->body;
}

std::optional<Transform> PhysicsWorld::bodyTransform(PhysicsBodyHandle handle) const noexcept
{
    const BodyEntry* entry = findBody(handle);
    if (entry == nullptr)
    {
        return std::nullopt;
    }

    return entry->transform;
}

void PhysicsWorld::setBodyTransform(PhysicsBodyHandle handle, const Transform& transform) noexcept
{
    BodyEntry* entry = findBody(handle);
    if (entry == nullptr)
    {
        return;
    }

    entry->transform = transform;
    entry->transform.update(Mat4::identity());
}

void PhysicsWorld::setBodyVelocity(PhysicsBodyHandle handle, Vec3 velocity) noexcept
{
    BodyEntry* entry = findBody(handle);
    if (entry != nullptr)
    {
        entry->body.linearVelocity(velocity);
    }
}

PhysicsWorld::BodyEntry* PhysicsWorld::findBody(PhysicsBodyHandle handle) noexcept
{
    const auto it = bodyIndexByHandle_.find(handle.value());
    if (it == bodyIndexByHandle_.end())
    {
        return nullptr;
    }
    BodyEntry& entry = bodies_[it->second];
    return entry.active ? &entry : nullptr;
}

const PhysicsWorld::BodyEntry* PhysicsWorld::findBody(PhysicsBodyHandle handle) const noexcept
{
    const auto it = bodyIndexByHandle_.find(handle.value());
    if (it == bodyIndexByHandle_.end())
    {
        return nullptr;
    }
    const BodyEntry& entry = bodies_[it->second];
    return entry.active ? &entry : nullptr;
}

PhysicsWorld::ColliderEntry* PhysicsWorld::findCollider(PhysicsColliderHandle handle) noexcept
{
    const auto it = colliderIndexByHandle_.find(handle.value());
    if (it == colliderIndexByHandle_.end())
    {
        return nullptr;
    }
    ColliderEntry& entry = colliders_[it->second];
    return entry.active ? &entry : nullptr;
}

const PhysicsWorld::ColliderEntry*
PhysicsWorld::findCollider(PhysicsColliderHandle handle) const noexcept
{
    const auto it = colliderIndexByHandle_.find(handle.value());
    if (it == colliderIndexByHandle_.end())
    {
        return nullptr;
    }
    const ColliderEntry& entry = colliders_[it->second];
    return entry.active ? &entry : nullptr;
}

PhysicsWorld::ColliderEntry* PhysicsWorld::findCollider(const Collider* collider) noexcept
{
    const auto it = colliderIndexByPointer_.find(collider);
    if (it == colliderIndexByPointer_.end())
    {
        return nullptr;
    }
    ColliderEntry& entry = colliders_[it->second];
    return entry.active ? &entry : nullptr;
}

AABB PhysicsWorld::localBounds(const ColliderShape& shape) const noexcept
{
    return std::visit(
        [](const auto& value) -> AABB
        {
            using Shape = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Shape, AabbShape>)
            {
                return value.bounds;
            }
            else if constexpr (std::is_same_v<Shape, BoxShape>)
            {
                return {value.center - value.halfExtents, value.center + value.halfExtents};
            }
            else if constexpr (std::is_same_v<Shape, SphereShape>)
            {
                const Vec3 extents{value.radius, value.radius, value.radius};
                return {value.center - extents, value.center + extents};
            }
            else
            {
                const Vec3 extents{value.radius, value.radius + value.halfHeight, value.radius};
                return {value.center - extents, value.center + extents};
            }
        },
        shape);
}

void PhysicsWorld::updateCollider(ColliderEntry& collider)
{
    const BodyEntry* owner = findBody(collider.body);
    if (owner == nullptr)
    {
        return;
    }

    collider.collider.localBounds(localBounds(collider.shape));
    collider.collider.update(owner->transform.world());
}

void PhysicsWorld::resetCollider(ColliderEntry& collider)
{
    const BodyEntry* owner = findBody(collider.body);
    if (owner == nullptr)
    {
        return;
    }

    collider.collider.localBounds(localBounds(collider.shape));
    collider.collider.resetFrame(owner->transform.world());
}

void PhysicsWorld::updateColliders()
{
    for (ColliderEntry& collider : colliders_)
    {
        if (collider.active)
        {
            updateCollider(collider);
        }
    }
}

void PhysicsWorld::resetResolvedColliders()
{
    for (ColliderEntry& collider : colliders_)
    {
        if (collider.active)
        {
            resetCollider(collider);
        }
    }
}

void PhysicsWorld::capturePreviousPositions() noexcept
{
    for (BodyEntry& body : bodies_)
    {
        if (body.active)
        {
            body.previousPosition = body.transform.position();
        }
    }
}

std::vector<PhysicsWorld::SolverContact> PhysicsWorld::contacts()
{
    std::vector<SolverContact> result;
    result.reserve(broadPhase_.possiblePairs().size());
    for (const CollisionPair& pair : broadPhase_.possiblePairs())
    {
        auto contact = contactForPair(pair);
        if (contact.has_value())
        {
            result.push_back(contact.value());
        }
    }
    return result;
}

std::optional<PhysicsWorld::ContactCandidate>
PhysicsWorld::contactCandidateForPair(const CollisionPair& pair)
{
    ColliderEntry* firstCollider = findCollider(pair.first);
    ColliderEntry* secondCollider = findCollider(pair.second);
    if (firstCollider == nullptr || secondCollider == nullptr)
    {
        return std::nullopt;
    }

    BodyEntry* firstBody = findBody(firstCollider->body);
    BodyEntry* secondBody = findBody(secondCollider->body);
    if (firstBody == nullptr || secondBody == nullptr)
    {
        return std::nullopt;
    }

    BodyEntry* moving = nullptr;
    BodyEntry* target = nullptr;
    ColliderEntry* movingCollider = nullptr;
    ColliderEntry* targetCollider = nullptr;

    auto select = [&](BodyEntry* selectedMoving, BodyEntry* selectedTarget,
                      ColliderEntry* selectedMovingCollider, ColliderEntry* selectedTargetCollider)
    {
        moving = selectedMoving;
        target = selectedTarget;
        movingCollider = selectedMovingCollider;
        targetCollider = selectedTargetCollider;
    };

    if (firstBody->body.type() == PhysicsBodyType::Dynamic)
    {
        select(firstBody, secondBody, firstCollider, secondCollider);
    }
    else if (secondBody->body.type() == PhysicsBodyType::Dynamic)
    {
        select(secondBody, firstBody, secondCollider, firstCollider);
    }
    else if (firstBody->body.type() == PhysicsBodyType::Kinematic)
    {
        select(firstBody, secondBody, firstCollider, secondCollider);
    }
    else if (secondBody->body.type() == PhysicsBodyType::Kinematic)
    {
        select(secondBody, firstBody, secondCollider, firstCollider);
    }

    if (moving == nullptr || target == nullptr || !movable(*moving))
    {
        return std::nullopt;
    }

    return ContactCandidate{moving, target, movingCollider, targetCollider};
}

std::optional<PhysicsWorld::SolverContact> PhysicsWorld::contactForPair(const CollisionPair& pair)
{
    auto candidate = contactCandidateForPair(pair);
    if (!candidate.has_value())
    {
        return std::nullopt;
    }

    // Shape-specific manifold in world space. The normal points target -> moving,
    // i.e. the direction to push the moving body out of penetration.
    const WorldShape movingShape = worldShape(*candidate->movingCollider);
    const WorldShape targetShape = worldShape(*candidate->targetCollider);
    auto manifold = narrowPhase_.collide(movingShape, targetShape);
    if (!manifold.has_value() || manifold->count == 0)
    {
        return std::nullopt;
    }

    return SolverContact{*manifold,         candidate->moving,         candidate->target,
                         candidate->movingCollider, candidate->targetCollider};
}

float PhysicsWorld::pushWeight(const BodyEntry* body) noexcept
{
    if (body == nullptr)
    {
        return 0.0f;
    }
    switch (body->body.type())
    {
    case PhysicsBodyType::Dynamic:
        return body->body.inverseMass();
    case PhysicsBodyType::Kinematic:
        return 1.0f; // scene-driven, but still slides out of penetration
    case PhysicsBodyType::Static:
    default:
        return 0.0f;
    }
}

bool PhysicsWorld::applyResponses(std::vector<SolverContact>& contacts)
{
    // Discrete, penetration-based response (P1): push the bodies apart along the
    // manifold normal split by push weight, then reflect each dynamic body's
    // velocity. No iterations / friction / restitution-by-impulse — that's the
    // P2 sequential-impulse solver. Pairs are processed in the deterministic
    // broadphase order (no sort), so the determinism harness still holds.
    bool resolved = false;
    for (const SolverContact& contact : contacts)
    {
        const Vec3 n = contact.manifold.normal; // target -> moving
        const float pen = contact.manifold.maxPenetration();
        if (pen <= 0.0f)
        {
            continue;
        }

        const float wMoving = pushWeight(contact.moving);
        const float wTarget = pushWeight(contact.target);
        const float wSum = wMoving + wTarget;
        if (wSum <= 0.0f)
        {
            continue; // both immovable
        }

        const float correction = pen / wSum;
        if (wMoving > 0.0f && contact.moving != nullptr)
        {
            contact.moving->transform.position(contact.moving->transform.position() +
                                               n * (correction * wMoving));
            contact.moving->transform.update(Mat4::identity());
        }
        if (wTarget > 0.0f && contact.target != nullptr)
        {
            contact.target->transform.position(contact.target->transform.position() -
                                               n * (correction * wTarget));
            contact.target->transform.update(Mat4::identity());
        }

        // Reflect the approaching velocity of each dynamic body about the normal.
        if (contact.moving != nullptr && contact.moving->body.type() == PhysicsBodyType::Dynamic)
        {
            contact.moving->body.reflectLinearVelocity(n);
        }
        if (contact.target != nullptr && contact.target->body.type() == PhysicsBodyType::Dynamic)
        {
            contact.target->body.reflectLinearVelocity(n * -1.0f);
        }
        resolved = true;
    }

    return resolved;
}

bool PhysicsWorld::movable(const BodyEntry& body) noexcept
{
    return body.body.type() == PhysicsBodyType::Dynamic ||
           body.body.type() == PhysicsBodyType::Kinematic;
}

} // namespace fire_engine
