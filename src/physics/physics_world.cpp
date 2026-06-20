#include <fire_engine/physics/physics_world.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec4.hpp>
#include <fire_engine/physics/physics_constants.hpp>

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

    // Integrate velocity only (gravity); positions advance after the velocity
    // solve so contact impulses are applied before the bodies move.
    for (BodyEntry& entry : bodies_)
    {
        if (!entry.active || entry.body.type() != PhysicsBodyType::Dynamic)
        {
            continue;
        }

        Vec3 velocity = entry.body.linearVelocity();
        velocity += gravity_ * entry.body.gravityScale() * fixedDt;
        entry.body.linearVelocity(velocity);
    }

    updateColliders(fixedDt);
    broadPhase_.update();

    auto frameContacts = contacts(fixedDt);
    // Snapshot for debug draw before the solver advances bodies.
    captureDebugContacts(frameContacts);

    if (solveAndIntegrate(frameContacts, fixedDt))
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
                    out.push_back(
                        makeBoxCollider(shape.center, shape.halfExtents, shape.orientation));
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

void PhysicsWorld::updateCollider(ColliderEntry& collider, float dt)
{
    const BodyEntry* owner = findBody(collider.body);
    if (owner == nullptr)
    {
        return;
    }

    // Predicted displacement this step (dynamic bodies only — kinematic/static were
    // already moved into place by the scene before step()). Threaded into the swept
    // bound so the broadphase pairs fast movers with what they are about to reach.
    const Vec3 motion =
        owner->body.type() == PhysicsBodyType::Dynamic ? owner->body.linearVelocity() * dt : Vec3{};

    collider.collider.localBounds(localBounds(collider.shape));
    collider.collider.update(owner->transform.world(), motion);
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

void PhysicsWorld::updateColliders(float dt)
{
    for (ColliderEntry& collider : colliders_)
    {
        if (collider.active)
        {
            updateCollider(collider, dt);
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

std::vector<PhysicsWorld::SolverContact> PhysicsWorld::contacts(float dt)
{
    std::vector<SolverContact> result;
    result.reserve(broadPhase_.possiblePairs().size());
    for (const CollisionPair& pair : broadPhase_.possiblePairs())
    {
        auto contact = contactForPair(pair, dt);
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

std::optional<PhysicsWorld::SolverContact> PhysicsWorld::contactForPair(const CollisionPair& pair,
                                                                        float dt)
{
    auto candidate = contactCandidateForPair(pair);
    if (!candidate.has_value())
    {
        return std::nullopt;
    }

    // Speculative-contact margin (CCD): how far the pair could close this step.
    // Using the pre-solve velocities is conservative (the solver only brakes), so a
    // fast mover within this margin generates a negative-penetration gap contact the
    // solver clamps against — no tunnelling. Slow pairs get ~kSpeculativeDistance.
    const float closingReach = (candidate->moving->body.linearVelocity().magnitude() +
                                candidate->target->body.linearVelocity().magnitude()) *
                                   dt +
                               kSpeculativeDistance;

    // Shape-specific manifold in world space. The normal points target -> moving,
    // i.e. the direction to push the moving body out of penetration.
    const WorldShape movingShape = worldShape(*candidate->movingCollider);
    const WorldShape targetShape = worldShape(*candidate->targetCollider);
    auto manifold = narrowPhase_.collide(movingShape, targetShape, closingReach);
    if (!manifold.has_value() || manifold->count == 0)
    {
        return std::nullopt;
    }

    return SolverContact{*manifold, candidate->moving, candidate->target, candidate->movingCollider,
                         candidate->targetCollider};
}

float PhysicsWorld::velocityInvMass(const BodyEntry& body) noexcept
{
    return body.body.type() == PhysicsBodyType::Dynamic ? body.body.inverseMass() : 0.0f;
}

float PhysicsWorld::positionWeight(const BodyEntry& body) noexcept
{
    switch (body.body.type())
    {
    case PhysicsBodyType::Dynamic:
        return body.body.inverseMass();
    case PhysicsBodyType::Kinematic:
        return 1.0f; // scene-driven, but still slides out of penetration
    case PhysicsBodyType::Static:
    default:
        return 0.0f;
    }
}

namespace
{

[[nodiscard]] std::uint64_t pairKey(PhysicsColliderHandle a, PhysicsColliderHandle b) noexcept
{
    std::uint32_t lo = a.value();
    std::uint32_t hi = b.value();
    if (lo > hi)
    {
        std::swap(lo, hi);
    }
    return (static_cast<std::uint64_t>(hi) << 32) | static_cast<std::uint64_t>(lo);
}

} // namespace

bool PhysicsWorld::solveAndIntegrate(const std::vector<SolverContact>& contacts, float dt)
{
    // Flat solver-body view, indexed 1:1 with bodies_ (insertion order → stable
    // and deterministic). Static/Kinematic carry invMass 0 so contact impulses
    // never shove a scene-driven body; positionWeight still lets Kinematic slide
    // out of penetration in the split-impulse pass.
    std::vector<SolverBody> solverBodies(bodies_.size());
    for (std::size_t i = 0; i < bodies_.size(); ++i)
    {
        const BodyEntry& entry = bodies_[i];
        solverBodies[i].velocity = entry.body.linearVelocity();
        solverBodies[i].position = entry.transform.position();
        solverBodies[i].invMass = entry.active ? velocityInvMass(entry) : 0.0f;
        solverBodies[i].positionWeight = entry.active ? positionWeight(entry) : 0.0f;
    }

    const BodyEntry* base = bodies_.data();
    std::vector<SolverContactInput> inputs;
    inputs.reserve(contacts.size());
    for (const SolverContact& contact : contacts)
    {
        SolverContactInput in;
        in.bodyA = static_cast<int>(contact.moving - base);
        in.bodyB = static_cast<int>(contact.target - base);
        in.normal = contact.manifold.normal;
        in.pointCount = contact.manifold.count;
        for (int p = 0; p < contact.manifold.count; ++p)
        {
            const auto pi = static_cast<std::size_t>(p);
            in.points[pi] = contact.manifold.points[pi].position;
            in.penetration[pi] = contact.manifold.points[pi].penetration;
        }
        const PhysicsMaterial ma = contact.moving->body.material();
        const PhysicsMaterial mb = contact.target->body.material();
        in.restitution = std::max(ma.restitution, mb.restitution);
        in.friction = std::sqrt(std::max(ma.friction, 0.0f) * std::max(mb.friction, 0.0f));
        in.key = pairKey(contact.movingCollider->handle, contact.targetCollider->handle);
        inputs.push_back(in);
    }

    // Velocity (impulse) solve.
    if (!inputs.empty())
    {
        solver_.prepare(solverBodies, inputs, dt);
        solver_.warmStart(solverBodies);
        for (int i = 0; i < kVelocityIterations; ++i)
        {
            solver_.solveVelocity(solverBodies);
        }
    }

    // Write solved velocities back and integrate positions with them (Dynamic).
    for (std::size_t i = 0; i < bodies_.size(); ++i)
    {
        BodyEntry& entry = bodies_[i];
        if (!entry.active || entry.body.type() != PhysicsBodyType::Dynamic)
        {
            continue;
        }
        entry.body.linearVelocity(solverBodies[i].velocity);
        solverBodies[i].position += solverBodies[i].velocity * dt;
    }

    // Split-impulse positional correction (may also nudge Kinematic bodies).
    if (!inputs.empty())
    {
        solver_.solvePosition(solverBodies);
        solver_.store();
    }

    // Write final positions back (everything but Static).
    bool moved = false;
    for (std::size_t i = 0; i < bodies_.size(); ++i)
    {
        BodyEntry& entry = bodies_[i];
        if (!entry.active || entry.body.type() == PhysicsBodyType::Static)
        {
            continue;
        }
        if (entry.transform.position() != solverBodies[i].position)
        {
            entry.transform.position(solverBodies[i].position);
            entry.transform.update(Mat4::identity());
            moved = true;
        }
    }

    return moved;
}

bool PhysicsWorld::movable(const BodyEntry& body) noexcept
{
    return body.body.type() == PhysicsBodyType::Dynamic ||
           body.body.type() == PhysicsBodyType::Kinematic;
}

} // namespace fire_engine
