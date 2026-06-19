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
    debugContacts_.reserve(contacts.size());
    for (const SolverContact& contact : contacts)
    {
        // Approximate the contact point as the moving body advanced to the time
        // of impact (no real manifold yet — see DebugContact). Fall back to the
        // target body's position when only the target moved.
        Vec3 point{};
        if (contact.moving != nullptr)
        {
            point = contact.moving->previousPosition + frameDelta(*contact.moving) * contact.toi;
        }
        else if (contact.target != nullptr)
        {
            point = contact.target->transform.position();
        }
        debugContacts_.push_back(DebugContact{point, contact.normal});
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

std::vector<ClothCollider> PhysicsWorld::gatherColliders() const
{
    std::vector<ClothCollider> out;
    out.reserve(colliders_.size());

    for (const ColliderEntry& entry : colliders_)
    {
        if (!entry.active)
        {
            continue;
        }
        const BodyEntry* owner = findBody(entry.body);
        if (owner == nullptr)
        {
            continue;
        }

        const Mat4 world = owner->transform.world();
        const Quaternion rot = owner->transform.rotation();
        const Vec3 s = matrixScale(world);
        const float uniform = std::max({s.x(), s.y(), s.z()});

        if (const auto* sphere = std::get_if<SphereShape>(&entry.shape))
        {
            out.push_back(makeSphereCollider(transformPoint(world, sphere->center),
                                             sphere->radius * uniform));
        }
        else if (const auto* box = std::get_if<BoxShape>(&entry.shape))
        {
            out.push_back(
                makeBoxCollider(transformPoint(world, box->center),
                                Vec3{box->halfExtents.x() * s.x(), box->halfExtents.y() * s.y(),
                                     box->halfExtents.z() * s.z()},
                                rot));
        }
        else if (const auto* capsule = std::get_if<CapsuleShape>(&entry.shape))
        {
            const Vec3 c = capsule->center;
            const Vec3 p0 = transformPoint(world, Vec3{c.x(), c.y() - capsule->halfHeight, c.z()});
            const Vec3 p1 = transformPoint(world, Vec3{c.x(), c.y() + capsule->halfHeight, c.z()});
            out.push_back(makeCapsuleCollider(p0, p1, capsule->radius * uniform));
        }
        else if (const auto* aabb = std::get_if<AabbShape>(&entry.shape))
        {
            const Vec3 he = aabb->bounds.extent() * 0.5f;
            out.push_back(makeBoxCollider(transformPoint(world, aabb->bounds.center()),
                                          Vec3{he.x() * s.x(), he.y() * s.y(), he.z() * s.z()},
                                          rot));
        }
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

    auto contact = narrowPhase_.sweptAabb(candidate->movingCollider->collider,
                                          candidate->targetCollider->collider);
    if (!contact.has_value())
    {
        return std::nullopt;
    }

    const Vec3 relativeDelta = frameDelta(*candidate->moving) - frameDelta(*candidate->target);
    if (Vec3::dotProduct(relativeDelta, contact->normal) >= 0.0f)
    {
        return std::nullopt;
    }

    return SolverContact{contact->toi,      contact->normal,           candidate->moving,
                         candidate->target, candidate->movingCollider, candidate->targetCollider};
}

bool PhysicsWorld::applyResponses(std::vector<SolverContact>& contacts)
{
    std::ranges::sort(contacts, [](const SolverContact& lhs, const SolverContact& rhs)
                      { return lhs.toi < rhs.toi; });

    bool resolved = false;
    for (const SolverContact& contact : contacts)
    {
        const bool movingHadFrameDelta =
            contact.moving != nullptr && movedThisFrame(*contact.moving);
        if (movingHadFrameDelta)
        {
            if (contact.moving->body.type() == PhysicsBodyType::Kinematic)
            {
                slideFrameMovement(*contact.moving, contact.toi, contact.normal);
            }
            else
            {
                moveToFrameTime(*contact.moving, contact.toi);
            }
            resolved = true;
        }

        if (contact.target != nullptr &&
            contact.target->body.type() == PhysicsBodyType::Kinematic &&
            movedThisFrame(*contact.target))
        {
            slideFrameMovement(*contact.target, contact.toi, contact.normal * -1.0f);
            resolved = true;
        }

        if (movingHadFrameDelta && contact.moving->body.type() == PhysicsBodyType::Dynamic)
        {
            contact.moving->body.reflectLinearVelocity(contact.normal);
        }
    }

    return resolved;
}

bool PhysicsWorld::movable(const BodyEntry& body) noexcept
{
    return body.body.type() == PhysicsBodyType::Dynamic ||
           body.body.type() == PhysicsBodyType::Kinematic;
}

bool PhysicsWorld::movedThisFrame(const BodyEntry& body) noexcept
{
    return frameDelta(body).magnitudeSquared() > float_epsilon * float_epsilon;
}

Vec3 PhysicsWorld::frameDelta(const BodyEntry& body) noexcept
{
    return body.transform.position() - body.previousPosition;
}

void PhysicsWorld::moveToFrameTime(BodyEntry& body, float toi) noexcept
{
    body.transform.position(body.previousPosition + frameDelta(body) * toi);
    body.transform.update(Mat4::identity());
}

void PhysicsWorld::slideFrameMovement(BodyEntry& body, float toi, Vec3 normal) noexcept
{
    const Vec3 delta = frameDelta(body);
    const float blocked = Vec3::dotProduct(delta, normal);
    if (blocked >= 0.0f)
    {
        return;
    }

    const Vec3 tangentDelta = delta - normal * blocked;
    const Vec3 resolvedDelta = delta * toi + tangentDelta * (1.0f - toi);
    body.transform.position(body.previousPosition + resolvedDelta);
    body.transform.update(Mat4::identity());
}

} // namespace fire_engine
