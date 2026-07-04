#include <fire_engine/physics/physics_world.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

#include <fire_engine/collision/broad_phase.hpp>
#include <fire_engine/collision/dynamic_aabb_tree_broad_phase.hpp>
#include <fire_engine/collision/gjk_epa.hpp>
#include <fire_engine/collision/narrow_phase.hpp>
#include <fire_engine/core/log.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec4.hpp>
#include <fire_engine/physics/articulation_contact.hpp>
#include <fire_engine/physics/contact_solver.hpp>
#include <fire_engine/physics/island.hpp>
#include <fire_engine/physics/joint_solver.hpp>
#include <fire_engine/physics/physics_constants.hpp>

namespace fire_engine
{

namespace
{

// Diagonal inverse inertia in the body's local (principal) frame for a shape of
// the given mass, with the body's per-axis scale folded in. Zero for non-positive
// mass or a degenerate axis (→ infinite inertia about that axis). All current
// shapes have a diagonal inertia tensor in their local frame.
// Diagonal inertia (about the shape's centre of mass) in the body's local frame.
[[nodiscard]] Vec3 localInertia(const ColliderShape& shape, float mass, Vec3 scale)
{
    if (mass <= 0.0f)
    {
        return {};
    }
    const float uniform = std::max({scale.x(), scale.y(), scale.z()});

    auto boxInertia = [mass](Vec3 h) -> Vec3
    {
        return {mass / 3.0f * (h.y() * h.y() + h.z() * h.z()),
                mass / 3.0f * (h.x() * h.x() + h.z() * h.z()),
                mass / 3.0f * (h.x() * h.x() + h.y() * h.y())};
    };

    const Vec3 inertia = std::visit(
        [&](const auto& s) -> Vec3
        {
            using S = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<S, BoxShape>)
            {
                return boxInertia({s.halfExtents.x() * scale.x(), s.halfExtents.y() * scale.y(),
                                   s.halfExtents.z() * scale.z()});
            }
            else if constexpr (std::is_same_v<S, SphereShape>)
            {
                const float r = s.radius * uniform;
                const float i = 0.4f * mass * r * r;
                return {i, i, i};
            }
            else if constexpr (std::is_same_v<S, CapsuleShape>)
            {
                // Solid capsule about its local-Y axis: a cylinder plus two
                // hemispheres (= one sphere), mass split by volume.
                const float r = s.radius * uniform;
                const float h = 2.0f * s.halfHeight * scale.y(); // cylinder length
                const float vc = pi * r * r * h;
                const float vs = 4.0f / 3.0f * pi * r * r * r;
                const float total = vc + vs;
                const float mc = total > 0.0f ? mass * vc / total : 0.0f;
                const float ms = total > 0.0f ? mass * vs / total : 0.0f;
                const float iAxis = mc * 0.5f * r * r + ms * 0.4f * r * r;
                const float iPerp = mc * (0.25f * r * r + h * h / 12.0f) +
                                    ms * (0.4f * r * r + 0.375f * r * h + 0.25f * h * h);
                return {iPerp, iAxis, iPerp};
            }
            else if constexpr (std::is_same_v<S, AabbShape>)
            {
                const Vec3 e = s.bounds.extent();
                return boxInertia(
                    {e.x() * 0.5f * scale.x(), e.y() * 0.5f * scale.y(), e.z() * 0.5f * scale.z()});
            }
            else // ConvexHullShape — approximate with the hull's AABB box inertia
            {
                Vec3 lo = s.vertices.empty() ? Vec3{} : s.vertices.front();
                Vec3 hi = lo;
                for (const Vec3& v : s.vertices)
                {
                    lo = {std::min(lo.x(), v.x()), std::min(lo.y(), v.y()),
                          std::min(lo.z(), v.z())};
                    hi = {std::max(hi.x(), v.x()), std::max(hi.y(), v.y()),
                          std::max(hi.z(), v.z())};
                }
                const Vec3 he = (hi - lo) * 0.5f;
                return boxInertia({he.x() * scale.x(), he.y() * scale.y(), he.z() * scale.z()});
            }
        },
        shape);

    return inertia;
}

// Invert a diagonal inertia component-wise (0 ⇒ infinite inertia about that axis).
[[nodiscard]] Vec3 invertInertia(Vec3 inertia) noexcept
{
    return {inertia.x() > 0.0f ? 1.0f / inertia.x() : 0.0f,
            inertia.y() > 0.0f ? 1.0f / inertia.y() : 0.0f,
            inertia.z() > 0.0f ? 1.0f / inertia.z() : 0.0f};
}

[[nodiscard]] Vec3 localInverseInertia(const ColliderShape& shape, float mass, Vec3 scale)
{
    return invertInertia(localInertia(shape, mass, scale));
}

// Approximate volume of a shape (with the body scale folded in) — used to split a
// compound body's mass across its children by volume (equal density).
[[nodiscard]] float shapeVolume(const ColliderShape& shape, Vec3 scale)
{
    const float uniform = std::max({scale.x(), scale.y(), scale.z()});
    return std::visit(
        [&](const auto& s) -> float
        {
            using S = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<S, BoxShape>)
            {
                return 8.0f * s.halfExtents.x() * scale.x() * s.halfExtents.y() * scale.y() *
                       s.halfExtents.z() * scale.z();
            }
            else if constexpr (std::is_same_v<S, SphereShape>)
            {
                const float r = s.radius * uniform;
                return 4.0f / 3.0f * pi * r * r * r;
            }
            else if constexpr (std::is_same_v<S, CapsuleShape>)
            {
                const float r = s.radius * uniform;
                const float h = 2.0f * s.halfHeight * scale.y();
                return pi * r * r * h + 4.0f / 3.0f * pi * r * r * r;
            }
            else if constexpr (std::is_same_v<S, AabbShape>)
            {
                const Vec3 e = s.bounds.extent();
                return e.x() * scale.x() * e.y() * scale.y() * e.z() * scale.z();
            }
            else // ConvexHullShape — AABB volume of the vertices
            {
                if (s.vertices.empty())
                {
                    return 0.0f;
                }
                Vec3 lo = s.vertices.front();
                Vec3 hi = lo;
                for (const Vec3& v : s.vertices)
                {
                    lo = {std::min(lo.x(), v.x()), std::min(lo.y(), v.y()),
                          std::min(lo.z(), v.z())};
                    hi = {std::max(hi.x(), v.x()), std::max(hi.y(), v.y()),
                          std::max(hi.z(), v.z())};
                }
                const Vec3 e = hi - lo;
                return e.x() * scale.x() * e.y() * scale.y() * e.z() * scale.z();
            }
        },
        shape);
}

// The shape's centre of mass in its (unscaled) local frame — the centre offset for the
// analytic primitives, the AABB centre of the vertices for a hull (matching the AABB
// inertia approximation). Zero for a centred shape.
[[nodiscard]] Vec3 shapeCenter(const ColliderShape& shape)
{
    return std::visit(
        [](const auto& s) -> Vec3
        {
            using S = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<S, AabbShape>)
            {
                return s.bounds.center();
            }
            else if constexpr (std::is_same_v<S, ConvexHullShape>)
            {
                if (s.vertices.empty())
                {
                    return Vec3{};
                }
                Vec3 lo = s.vertices.front();
                Vec3 hi = lo;
                for (const Vec3& v : s.vertices)
                {
                    lo = {std::min(lo.x(), v.x()), std::min(lo.y(), v.y()),
                          std::min(lo.z(), v.z())};
                    hi = {std::max(hi.x(), v.x()), std::max(hi.y(), v.y()),
                          std::max(hi.z(), v.z())};
                }
                return (lo + hi) * 0.5f;
            }
            else // BoxShape / SphereShape / CapsuleShape
            {
                return s.center;
            }
        },
        shape);
}

[[nodiscard]] Vec3 transformPoint(const Mat4& m, Vec3 p)
{
    const Vec4 r = m * Vec4{p.x(), p.y(), p.z(), 1.0f};
    return {r.x(), r.y(), r.z()};
}

// AABB enclosing `a` after transforming its 8 corners by `m` (used to place a
// compound child's local bounds within the body frame).
[[nodiscard]] AABB transformAabb(const Mat4& m, const AABB& a)
{
    Vec3 lo;
    Vec3 hi;
    for (int i = 0; i < 8; ++i)
    {
        const Vec3 corner{(i & 1) ? a.max.x() : a.min.x(), (i & 2) ? a.max.y() : a.min.y(),
                          (i & 4) ? a.max.z() : a.min.z()};
        const Vec3 p = transformPoint(m, corner);
        if (i == 0)
        {
            lo = p;
            hi = p;
        }
        else
        {
            lo = {std::min(lo.x(), p.x()), std::min(lo.y(), p.y()), std::min(lo.z(), p.z())};
            hi = {std::max(hi.x(), p.x()), std::max(hi.y(), p.y()), std::max(hi.z(), p.z())};
        }
    }
    return {lo, hi};
}

struct CompoundMassProperties
{
    Vec3 com;
    Vec3 inverseInertia;
};

// Aggregate a compound's children into a body centre of mass + diagonal inverse
// inertia. Mass is split across children by volume (equal density). The inertia is
// the parallel-axis sum of each child's (rotated) diagonal inertia about the body
// COM; only the diagonal is kept (off-diagonal products of inertia are dropped —
// exact for compounds symmetric about the body axes, approximate otherwise).
[[nodiscard]] CompoundMassProperties compoundMassProperties(std::span<const CompoundChild> children,
                                                            float totalMass, Vec3 scale)
{
    const std::size_t n = children.size();
    std::vector<float> mass(n, 0.0f);
    std::vector<Vec3> childCom(n); // child COM in the body frame

    float totalVolume = 0.0f;
    std::vector<float> volumes(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i)
    {
        volumes[i] = shapeVolume(children[i].shape, scale);
        totalVolume += volumes[i];
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        mass[i] = totalVolume > 0.0f ? totalMass * volumes[i] / totalVolume
                                     : totalMass / static_cast<float>(n);
        const Vec3 sc = shapeCenter(children[i].shape);
        const Vec3 scaled{sc.x() * scale.x(), sc.y() * scale.y(), sc.z() * scale.z()};
        childCom[i] = children[i].localPosition + children[i].localRotation.rotate(scaled);
    }

    Vec3 com{};
    for (std::size_t i = 0; i < n; ++i)
    {
        com += childCom[i] * mass[i];
    }
    com = totalMass > 0.0f ? com / totalMass : Vec3{};

    Vec3 diag{};
    for (std::size_t i = 0; i < n; ++i)
    {
        const Vec3 childI = localInertia(children[i].shape, mass[i], scale);
        const Mat3 r = Mat3::fromQuaternion(children[i].localRotation);
        const Vec3 c0 = r * Vec3{1.0f, 0.0f, 0.0f};
        const Vec3 c1 = r * Vec3{0.0f, 1.0f, 0.0f};
        const Vec3 c2 = r * Vec3{0.0f, 0.0f, 1.0f};
        // Diagonal of R·diag(childI)·Rᵀ.
        const Vec3 rotDiag{c0.x() * c0.x() * childI.x() + c1.x() * c1.x() * childI.y() +
                               c2.x() * c2.x() * childI.z(),
                           c0.y() * c0.y() * childI.x() + c1.y() * c1.y() * childI.y() +
                               c2.y() * c2.y() * childI.z(),
                           c0.z() * c0.z() * childI.x() + c1.z() * c1.z() * childI.y() +
                               c2.z() * c2.z() * childI.z()};
        // Parallel-axis diagonal: mass·(|d|² − d_a²) about each axis a.
        const Vec3 d = childCom[i] - com;
        const float d2 = Vec3::dotProduct(d, d);
        const Vec3 par{mass[i] * (d2 - d.x() * d.x()), mass[i] * (d2 - d.y() * d.y()),
                       mass[i] * (d2 - d.z() * d.z())};
        diag += rotDiag + par;
    }
    return {com, invertInertia(diag)};
}

} // namespace

PhysicsWorld::PhysicsWorld()
    : PhysicsWorld{std::make_unique<DynamicAabbTreeBroadPhase>()}
{
}

PhysicsWorld::PhysicsWorld(std::unique_ptr<BroadPhase> broadPhase)
    : broadPhase_{std::move(broadPhase)},
      narrowPhase_{std::make_unique<NarrowPhase>()},
      solver_{std::make_unique<ContactSolver>()},
      jointSolver_{std::make_unique<JointSolver>()}
{
}

PhysicsWorld::~PhysicsWorld() = default;

PhysicsWorld::PhysicsWorld(PhysicsWorld&&) noexcept = default;

PhysicsWorld& PhysicsWorld::operator=(PhysicsWorld&&) noexcept = default;

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
    body.allowSleeping(desc.allowSleeping);

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

    const PhysicsColliderHandle handle =
        addColliderEntry(*owner, desc.shape, desc.material, desc.collisionLayer, desc.collisionMask,
                         Vec3{}, Quaternion::identity(), desc.isTrigger);

    // Mass properties from the (now-known) shape + mass; dynamic bodies only, others
    // keep infinite inertia (zero). Single collider; the centre of mass is the shape's
    // (scaled) centre — zero for a centred shape, so a centred body is unchanged. The
    // inertia is taken about that COM.
    if (owner->body.type() == PhysicsBodyType::Dynamic)
    {
        const Vec3 scale = owner->transform.scale();
        owner->body.inverseInertiaLocal(localInverseInertia(desc.shape, owner->body.mass(), scale));
        const Vec3 center = shapeCenter(desc.shape);
        owner->body.centerOfMassLocal(
            {center.x() * scale.x(), center.y() * scale.y(), center.z() * scale.z()});
    }
    return handle;
}

PhysicsColliderHandle PhysicsWorld::createCompoundCollider(PhysicsBodyHandle bodyHandle,
                                                           std::span<const CompoundChild> children,
                                                           std::uint32_t collisionLayer,
                                                           std::uint32_t collisionMask)
{
    BodyEntry* owner = findBody(bodyHandle);
    if (owner == nullptr || children.empty())
    {
        return {};
    }

    PhysicsColliderHandle first;
    for (const CompoundChild& child : children)
    {
        const PhysicsColliderHandle h =
            addColliderEntry(*owner, child.shape, child.material, collisionLayer, collisionMask,
                             child.localPosition, child.localRotation);
        if (!first.valid())
        {
            first = h;
        }
    }

    // Aggregate the children's mass properties into the body (Dynamic only).
    if (owner->body.type() == PhysicsBodyType::Dynamic)
    {
        const CompoundMassProperties mp =
            compoundMassProperties(children, owner->body.mass(), owner->transform.scale());
        owner->body.centerOfMassLocal(mp.com);
        owner->body.inverseInertiaLocal(mp.inverseInertia);
    }
    return first;
}

PhysicsColliderHandle PhysicsWorld::createMeshCollider(PhysicsBodyHandle bodyHandle,
                                                       const StaticMeshShape& mesh,
                                                       const PhysicsMaterial& material,
                                                       std::uint32_t collisionLayer,
                                                       std::uint32_t collisionMask)
{
    BodyEntry* owner = findBody(bodyHandle);
    if (owner == nullptr || mesh.vertices.empty() || mesh.indices.size() < 3)
    {
        return {};
    }
    if (owner->body.type() != PhysicsBodyType::Static)
    {
        log::warn(log::category::physics, "mesh colliders are static-only; ignoring.");
        return {};
    }

    // Whole-mesh local AABB as the broadphase proxy shape.
    AABB bounds{mesh.vertices.front(), mesh.vertices.front()};
    for (const Vec3& v : mesh.vertices)
    {
        bounds.min = {std::min(bounds.min.x(), v.x()), std::min(bounds.min.y(), v.y()),
                      std::min(bounds.min.z(), v.z())};
        bounds.max = {std::max(bounds.max.x(), v.x()), std::max(bounds.max.y(), v.y()),
                      std::max(bounds.max.z(), v.z())};
    }
    const PhysicsColliderHandle handle =
        addColliderEntry(*owner, AabbShape{bounds}, material, collisionLayer, collisionMask, Vec3{},
                         Quaternion::identity());

    // Build the world-space triangles + BVH once (the static body keeps a fixed
    // transform). Stored on the collider entry; contacts() resolves against them.
    auto data = std::make_shared<MeshCollisionData>();
    data->indices = mesh.indices;
    const Mat4 world = owner->transform.world();
    data->worldVertices.reserve(mesh.vertices.size());
    for (const Vec3& v : mesh.vertices)
    {
        data->worldVertices.push_back(transformPoint(world, v));
    }
    const std::size_t triCount = data->indices.size() / 3;
    for (std::size_t t = 0; t < triCount; ++t)
    {
        const Vec3& v0 = data->worldVertices[data->indices[3 * t + 0]];
        const Vec3& v1 = data->worldVertices[data->indices[3 * t + 1]];
        const Vec3& v2 = data->worldVertices[data->indices[3 * t + 2]];
        const AABB tri{{std::min({v0.x(), v1.x(), v2.x()}), std::min({v0.y(), v1.y(), v2.y()}),
                        std::min({v0.z(), v1.z(), v2.z()})},
                       {std::max({v0.x(), v1.x(), v2.x()}), std::max({v0.y(), v1.y(), v2.y()}),
                        std::max({v0.z(), v1.z(), v2.z()})}};
        (void)data->bvh.createProxy(tri, static_cast<int>(t));
    }

    ColliderEntry* entry = findCollider(handle);
    if (entry != nullptr)
    {
        entry->mesh = std::move(data);
    }
    return handle;
}

PhysicsColliderHandle
PhysicsWorld::addColliderEntry(BodyEntry& owner, const ColliderShape& shape,
                               const PhysicsMaterial& material, std::uint32_t collisionLayer,
                               std::uint32_t collisionMask, const Vec3& localPosition,
                               const Quaternion& localRotation, bool isTrigger)
{
    const PhysicsColliderHandle handle = nextColliderHandle_;
    nextColliderHandle_ = PhysicsColliderHandle{nextColliderHandle_.value() + 1U};

    Collider collider;
    // Local bounds include the child's offset within the body (identity for a single
    // collider), so the broadphase covers the right region.
    const Mat4 childMat = Mat4::translate(localPosition) * localRotation.toMat4();
    collider.localBounds(transformAabb(childMat, localBounds(shape)));
    collider.collisionLayer(collisionLayer);
    collider.collisionMask(collisionMask);
    collider.isTrigger(isTrigger);
    collider.resetFrame(owner.transform.world());

    colliderIndexByHandle_.emplace(handle.value(), colliders_.size());
    colliders_.push_back({handle, owner.handle, PhysicsArticulationHandle{}, -1,
                          std::move(collider), shape, material, localPosition, localRotation, true,
                          nullptr});
    colliderIndexByPointer_.emplace(&colliders_.back().collider, colliders_.size() - 1);
    owner.colliders.push_back(handle);
    broadPhase_->addCollider(colliders_.back().collider);
    return handle;
}

void PhysicsWorld::deactivateCollider(ColliderEntry& collider)
{
    if (!collider.active)
    {
        return;
    }

    [[maybe_unused]] const bool removed = broadPhase_->removeCollider(collider.collider);
    colliderIndexByHandle_.erase(collider.handle.value());
    colliderIndexByPointer_.erase(&collider.collider);
    removeOverlapPairsForCollider(collider.handle);
    collider.active = false;
}

void PhysicsWorld::deactivateJoint(std::size_t jointIndex)
{
    JointEntry& joint = joints_[jointIndex];
    if (!joint.active)
    {
        return;
    }

    jointIndexByHandle_.erase(joint.handle.value());
    joint.active = false;
}

void PhysicsWorld::deactivateJointsForBody(PhysicsBodyHandle body)
{
    for (std::size_t i = 0; i < joints_.size(); ++i)
    {
        const JointEntry& joint = joints_[i];
        if (joint.active && (joint.desc.bodyA == body || joint.desc.bodyB == body))
        {
            deactivateJoint(i);
        }
    }
}

bool PhysicsWorld::destroyBody(PhysicsBodyHandle handle)
{
    BodyEntry* bodyEntry = findBody(handle);
    if (bodyEntry == nullptr)
    {
        return false;
    }

    deactivateJointsForBody(handle);
    for (PhysicsColliderHandle colliderHandle : bodyEntry->colliders)
    {
        ColliderEntry* colliderEntry = findCollider(colliderHandle);
        if (colliderEntry != nullptr)
        {
            deactivateCollider(*colliderEntry);
        }
    }

    bodyIndexByHandle_.erase(handle.value());
    bodyEntry->active = false;
    return true;
}

PhysicsConstraintHandle PhysicsWorld::createJoint(const JointDesc& desc)
{
    const BodyEntry* a = findBody(desc.bodyA);
    const BodyEntry* b = findBody(desc.bodyB);
    if (a == nullptr || b == nullptr)
    {
        return {};
    }

    const PhysicsConstraintHandle handle = nextJointHandle_;
    nextJointHandle_ = PhysicsConstraintHandle{nextJointHandle_.value() + 1U};

    // Capture the rest relative orientation (B in A's frame) so limits read 0 here.
    const Quaternion restRelative = a->transform.rotation().conjugate() * b->transform.rotation();

    jointIndexByHandle_.emplace(handle.value(), joints_.size());
    joints_.push_back({handle, desc, restRelative, true});
    return handle;
}

bool PhysicsWorld::destroyJoint(PhysicsConstraintHandle handle)
{
    const auto it = jointIndexByHandle_.find(handle.value());
    if (it == jointIndexByHandle_.end() || !joints_[it->second].active)
    {
        return false;
    }
    deactivateJoint(it->second);
    return true;
}

void PhysicsWorld::clear()
{
    broadPhase_->clear();
    bodies_.clear();
    colliders_.clear();
    joints_.clear();
    articulations_.clear();
    articulationSleepTimers_.clear();
    articulationSleeping_.clear();
    bodyIndexByHandle_.clear();
    colliderIndexByHandle_.clear();
    jointIndexByHandle_.clear();
    colliderIndexByPointer_.clear();
    articulationIndexByHandle_.clear();
    nextBodyHandle_ = PhysicsBodyHandle{1U};
    nextColliderHandle_ = PhysicsColliderHandle{1U};
    nextJointHandle_ = PhysicsConstraintHandle{1U};
    nextArticulationHandle_ = PhysicsArticulationHandle{1U};
    triggerOverlaps_.clear();
    previousTriggerOverlaps_.clear();
    collisionOverlaps_.clear();
    previousCollisionOverlaps_.clear();
    triggerEvents_.clear();
    collisionEvents_.clear();
}

void PhysicsWorld::step(float fixedDt)
{
    if (fixedDt <= 0.0f)
    {
        return;
    }

    // Refresh articulation link world transforms from their generalized coordinates so the
    // link colliders' swept bounds (updateColliders, below) track the current pose. Phase A
    // articulations are kinematic (q is set externally), so this is pure forward kinematics;
    // the ABA dynamics that advance q live inside the solve in a later phase.
    for (Articulation& art : articulations_)
    {
        art.forwardKinematics();
    }

    // Gravity is integrated per substep inside the TGS solve (solveIsland), not here:
    // collision detection runs once on this step's start-of-frame velocities, then the
    // substep loop applies gravity + solves + integrates positions kSubstepCount times.
    updateColliders(fixedDt);
    broadPhase_->update();

    auto frameContacts = contacts(fixedDt);
    // Snapshot for debug draw before the solver advances bodies.
    captureDebugContacts(frameContacts);

    if (solveAndIntegrate(frameContacts, fixedDt))
    {
        resetResolvedColliders();
        broadPhase_->rebuild();
    }

    // Reduced-coordinate articulations advance in their own solve pass (link-vs-static
    // contacts through the ConstraintBody seam). Uses the broadphase pairs from this step's
    // start pose; the substep loop tracks each contact in link-local space as the link moves.
    if (!articulations_.empty())
    {
        stepArticulations(fixedDt);
    }

    // Diff this step's overlaps against the previous step into enter/stay/exit events.
    updateOverlapEvents();

    capturePreviousPositions();
}

bool PhysicsWorld::valid(PhysicsBodyHandle handle) const noexcept
{
    return findBody(handle) != nullptr;
}

bool PhysicsWorld::valid(PhysicsColliderHandle handle) const noexcept
{
    return findCollider(handle) != nullptr;
}

bool PhysicsWorld::valid(PhysicsConstraintHandle handle) const noexcept
{
    const auto it = jointIndexByHandle_.find(handle.value());
    return it != jointIndexByHandle_.end() && joints_[it->second].active;
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

std::size_t PhysicsWorld::jointCount() const noexcept
{
    return static_cast<std::size_t>(
        std::ranges::count_if(joints_, [](const JointEntry& entry) { return entry.active; }));
}

const std::vector<CollisionPair>& PhysicsWorld::possiblePairs() const noexcept
{
    return broadPhase_->possiblePairs();
}

bool PhysicsWorld::validateBroadPhase() const
{
    return broadPhase_->validate();
}

namespace
{

// A body is sleep-eligible this step when both its linear and angular speeds are
// below the sleep thresholds (squared comparison).
[[nodiscard]] bool belowSleepThreshold(const PhysicsBody& body, float angularThreshold) noexcept
{
    return body.linearVelocity().magnitudeSquared() <
               kLinearSleepThreshold * kLinearSleepThreshold &&
           body.angularVelocity().magnitudeSquared() < angularThreshold * angularThreshold;
}

} // namespace

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
    // An externally repositioned body must re-simulate.
    entry->sleeping = false;
    entry->sleepTimer = 0.0f;
}

void PhysicsWorld::setBodyVelocity(PhysicsBodyHandle handle, Vec3 velocity) noexcept
{
    BodyEntry* entry = findBody(handle);
    if (entry != nullptr)
    {
        entry->body.linearVelocity(velocity);
        // An externally driven velocity must re-simulate.
        entry->sleeping = false;
        entry->sleepTimer = 0.0f;
    }
}

void PhysicsWorld::wake(PhysicsBodyHandle handle) noexcept
{
    BodyEntry* entry = findBody(handle);
    if (entry != nullptr)
    {
        entry->sleeping = false;
        entry->sleepTimer = 0.0f;
    }
}

bool PhysicsWorld::sleeping(PhysicsBodyHandle handle) const noexcept
{
    const BodyEntry* entry = findBody(handle);
    return entry != nullptr && entry->sleeping;
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
    // Fresh overlap sets for this step; appendContactsForPair records into them and
    // step() diffs them against the previous step into enter/stay/exit events.
    triggerOverlaps_.clear();
    collisionOverlaps_.clear();

    // Solve in a canonical pair order — sorted by (firstId, secondId) — so the
    // (order-dependent) sequential-impulse result is independent of which broadphase
    // produced the pairs and of that broadphase's internal ordering. The dynamic AABB
    // tree already emits id-sorted pairs, so this is a no-op for the default path;
    // sweep-and-prune emits update-ordered pairs, which this canonicalises.
    std::vector<CollisionPair> pairs = broadPhase_->possiblePairs();
    std::sort(pairs.begin(), pairs.end(),
              [](const CollisionPair& lhs, const CollisionPair& rhs)
              {
                  if (lhs.firstId != rhs.firstId)
                  {
                      return lhs.firstId < rhs.firstId;
                  }
                  return lhs.secondId < rhs.secondId;
              });

    std::vector<SolverContact> result;
    result.reserve(pairs.size());
    for (const CollisionPair& pair : pairs)
    {
        appendContactsForPair(pair, dt, result);
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

void PhysicsWorld::appendContactsForPair(const CollisionPair& pair, float dt,
                                         std::vector<SolverContact>& out)
{
    const auto candidate = contactCandidateForPair(pair);
    if (!candidate.has_value())
    {
        return;
    }
    // A trigger collider generates overlap events but no solver response — detect the
    // overlap, then return without pushing contacts.
    const bool triggerPair = candidate->movingCollider->collider.isTrigger() ||
                             candidate->targetCollider->collider.isTrigger();

    // A static mesh is always the target (it never moves). Resolve the moving body
    // against its triangles; otherwise the ordinary single-manifold path.
    if (candidate->targetCollider->mesh != nullptr)
    {
        const bool overlapped = appendMeshContacts(*candidate, dt, triggerPair ? nullptr : &out);
        if (overlapped)
        {
            recordOverlap(candidate->movingCollider->handle, candidate->targetCollider->handle,
                          triggerPair);
        }
        return;
    }
    if (candidate->movingCollider->mesh != nullptr)
    {
        return; // a mesh is static and never the moving body
    }
    if (const auto contact = singleContact(*candidate, dt))
    {
        // Deepest signed penetration across the manifold (a speculative gap is negative).
        // A trigger fires only on real overlap (> 0); a collision event fires when the
        // pair is touching — within the small speculative contact band — since the solver
        // brakes resting contacts at ~0 penetration so they rarely read strictly positive.
        float signedPenetration = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < contact->manifold.count; ++i)
        {
            signedPenetration =
                std::max(signedPenetration,
                         contact->manifold.points[static_cast<std::size_t>(i)].penetration);
        }
        if (triggerPair)
        {
            if (signedPenetration > 0.0f)
            {
                recordOverlap(candidate->movingCollider->handle, candidate->targetCollider->handle,
                              true);
            }
            return; // no solver response for a trigger
        }
        out.push_back(*contact);
        if (signedPenetration > -kSpeculativeDistance)
        {
            recordOverlap(candidate->movingCollider->handle, candidate->targetCollider->handle,
                          false);
        }
    }
}

std::optional<PhysicsWorld::SolverContact>
PhysicsWorld::singleContact(const ContactCandidate& candidate, float dt)
{
    // Speculative-contact margin (CCD): how far the pair could close this step.
    // Using the pre-solve velocities is conservative (the solver only brakes), so a
    // fast mover within this margin generates a negative-penetration gap contact the
    // solver clamps against — no tunnelling. Slow pairs get ~kSpeculativeDistance.
    const float closingReach = (candidate.moving->body.linearVelocity().magnitude() +
                                candidate.target->body.linearVelocity().magnitude()) *
                                   dt +
                               kSpeculativeDistance;

    // Shape-specific manifold in world space. The normal points target -> moving,
    // i.e. the direction to push the moving body out of penetration.
    const WorldShape movingShape = worldShape(*candidate.movingCollider);
    const WorldShape targetShape = worldShape(*candidate.targetCollider);
    auto manifold = narrowPhase_->collide(movingShape, targetShape, closingReach);
    if (!manifold.has_value() || manifold->count == 0)
    {
        return std::nullopt;
    }

    return SolverContact{*manifold,
                         candidate.moving,
                         candidate.target,
                         candidate.movingCollider,
                         candidate.targetCollider,
                         0};
}

bool PhysicsWorld::appendMeshContacts(const ContactCandidate& candidate, float dt,
                                      std::vector<SolverContact>* out)
{
    const WorldShape movingShape = worldShape(*candidate.movingCollider);
    const MeshCollisionData& mesh = *candidate.targetCollider->mesh;
    const float closingReach =
        candidate.moving->body.linearVelocity().magnitude() * dt + kSpeculativeDistance;

    bool overlapped = false;

    // Query the triangle BVH with the moving collider's swept world bounds; collide the
    // moving shape against each candidate triangle (a flat WorldConvex) individually.
    const AABB query = candidate.movingCollider->collider.sweptWorldBounds();
    mesh.bvh.query(
        query,
        [&](int proxy)
        {
            const int t = mesh.bvh.payload(proxy);
            const auto base = static_cast<std::size_t>(t) * 3;
            const Vec3 v0 = mesh.worldVertices[mesh.indices[base + 0]];
            const Vec3 v1 = mesh.worldVertices[mesh.indices[base + 1]];
            const Vec3 v2 = mesh.worldVertices[mesh.indices[base + 2]];
            const std::array<ConvexFace, 1> faces{ConvexFace{Vec3{}, std::vector<int>{0, 1, 2}}};
            WorldConvex triangle;
            triangle.vertices = {v0, v1, v2};
            triangle.faces = faces;

            auto manifold = narrowPhase_->collide(movingShape, WorldShape{triangle}, closingReach);
            if (!manifold.has_value() || manifold->count == 0)
            {
                return;
            }
            const Vec3 n = manifold->normal;
            if (!std::isfinite(n.x()) || !std::isfinite(n.y()) || !std::isfinite(n.z()) ||
                n.magnitudeSquared() <= 1.0e-8f)
            {
                return; // degenerate EPA result
            }

            // Reconstruct the contact against the triangle's (CCW-outward) face plane:
            // force the normal to the surface normal and re-measure each point's
            // penetration as its signed depth below that plane. EPA on a flat triangle
            // gives unreliable normals/depths (especially where the body overhangs a
            // shared edge, where it returns a sideways *edge* normal that would kick the
            // body into a growing rock); using only its contact *points* + the true plane
            // is stable. A back-facing triangle (body behind it) is skipped — mesh
            // collision is one-sided (front face only).
            const Vec3 faceNormal = Vec3::normalise(Vec3::crossProduct(v1 - v0, v2 - v0));
            if (Vec3::dotProduct(n, faceNormal) <= 0.0f)
            {
                return;
            }
            ContactManifold planar;
            planar.normal = faceNormal;
            for (int i = 0; i < manifold->count; ++i)
            {
                const Vec3 pos = manifold->points[static_cast<std::size_t>(i)].position;
                if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z()))
                {
                    continue; // EPA can return a garbage witness point for a flat triangle
                }
                const float penetration = Vec3::dotProduct(faceNormal, v0 - pos);
                // Keep only points within the speculative band of the surface; an
                // implausibly deep one is a degenerate EPA result, not a real contact.
                if (penetration < -closingReach || penetration > kMaxMeshPenetration)
                {
                    continue;
                }
                planar.points[static_cast<std::size_t>(planar.count)] = {pos, penetration};
                ++planar.count;
                overlapped = overlapped || penetration > 0.0f;
            }
            if (planar.count == 0)
            {
                return;
            }

            if (out != nullptr)
            {
                out->push_back(SolverContact{planar, candidate.moving, candidate.target,
                                             candidate.movingCollider, candidate.targetCollider,
                                             static_cast<std::uint32_t>(t) + 1});
            }
        });
    return overlapped;
}

float PhysicsWorld::velocityInvMass(const BodyEntry& body) noexcept
{
    return body.body.type() == PhysicsBodyType::Dynamic ? body.body.inverseMass() : 0.0f;
}

float PhysicsWorld::positionWeight(const BodyEntry& body) noexcept
{
    switch (body.body.type())
    {
    case PhysicsBodyType::Kinematic:
        return 1.0f; // scene-driven: not velocity-solved, but slides out of penetration
    case PhysicsBodyType::Dynamic: // resolves penetration through the soft bias (TGS), not here
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

void projectSleepingJointAnchors(const Island& island, std::span<const JointInput> jointInputs,
                                 std::vector<SolverBody>& solverBodies) noexcept
{
    struct AnchorProjection
    {
        int bodyA{-1};
        int bodyB{-1};
        Vec3 rA{};
        Vec3 rB{};
    };

    std::vector<AnchorProjection> projections;
    projections.reserve(island.joints.size());
    for (const int ji : island.joints)
    {
        const JointInput& joint = jointInputs[static_cast<std::size_t>(ji)];
        if (joint.type != JointType::BallSocket && joint.type != JointType::Hinge)
        {
            continue;
        }

        const SolverBody& a = solverBodies[static_cast<std::size_t>(joint.bodyA)];
        const SolverBody& b = solverBodies[static_cast<std::size_t>(joint.bodyB)];
        projections.push_back(AnchorProjection{joint.bodyA, joint.bodyB, joint.anchorA - a.position,
                                               joint.anchorB - b.position});
    }

    constexpr int kSleepProjectionIterations = 12;
    for (int iter = 0; iter < kSleepProjectionIterations; ++iter)
    {
        for (const AnchorProjection& projection : projections)
        {
            SolverBody& a = solverBodies[static_cast<std::size_t>(projection.bodyA)];
            SolverBody& b = solverBodies[static_cast<std::size_t>(projection.bodyB)];
            const float invMassSum = a.invMass + b.invMass;
            if (invMassSum <= 0.0f)
            {
                continue;
            }

            const Vec3 separation = (a.position + projection.rA) - (b.position + projection.rB);
            a.position -= separation * (a.invMass / invMassSum);
            b.position += separation * (b.invMass / invMassSum);
        }
    }
}

} // namespace

std::vector<JointInput> PhysicsWorld::buildJointInputs() const
{
    // Compose each joint's local anchors/axes with the live body transforms. Anchors
    // are relative to the body centre of mass (assumed unit scale, as for ragdoll
    // bodies); the indices match the solver-body array (1:1 with bodies_).
    std::vector<JointInput> inputs;
    inputs.reserve(jointCount());
    for (const JointEntry& entry : joints_)
    {
        if (!entry.active)
        {
            continue;
        }
        const auto ia = bodyIndexByHandle_.find(entry.desc.bodyA.value());
        const auto ib = bodyIndexByHandle_.find(entry.desc.bodyB.value());
        if (ia == bodyIndexByHandle_.end() || ib == bodyIndexByHandle_.end())
        {
            continue;
        }
        const BodyEntry& a = bodies_[ia->second];
        const BodyEntry& b = bodies_[ib->second];
        if (!a.active || !b.active)
        {
            continue;
        }

        const Mat3 ra = Mat3::fromQuaternion(a.transform.rotation());
        const Mat3 rb = Mat3::fromQuaternion(b.transform.rotation());

        JointInput in;
        in.bodyA = static_cast<int>(ia->second);
        in.bodyB = static_cast<int>(ib->second);
        in.type = entry.desc.type;
        in.anchorA = a.transform.position() + ra * entry.desc.anchorA;
        in.anchorB = b.transform.position() + rb * entry.desc.anchorB;
        in.axisA = ra * entry.desc.axisA;
        in.axisB = rb * entry.desc.axisB;
        in.twistAxisLocal = entry.desc.axisA;
        in.restLength = entry.desc.restLength;
        // Relative orientation of B in A's frame, taken relative to the rest frame:
        // identity at the creation pose, so the limit decomposition reads 0 there.
        in.relative = entry.restRelative.conjugate() *
                      (a.transform.rotation().conjugate() * b.transform.rotation());
        in.limits = entry.desc.limits;
        in.key = entry.handle.value();
        inputs.push_back(in);
    }
    return inputs;
}

bool PhysicsWorld::refreshIslandContacts(const Island& island,
                                         std::span<const SolverBody> solverBodies,
                                         std::span<const SolverContact> solverContacts,
                                         std::vector<SolverContactInput>& islandContacts,
                                         std::vector<std::uint8_t>& flags, float dt,
                                         float remaining)
{
    flags.assign(islandContacts.size(), 0);

    // A Dynamic body sweeping more than kSubstepRefreshRotation this step (at its current
    // solver angular velocity — the impact itself generates the spin) has a stale manifold.
    // Squared compare, no sqrt.
    const auto fastRotating = [&](const BodyEntry* entry)
    {
        if (entry->body.type() != PhysicsBodyType::Dynamic)
        {
            return false;
        }
        const auto b = static_cast<std::size_t>(entry - bodies_.data());
        return solverBodies[b].angularVelocity.magnitudeSquared() * dt * dt >
               kSubstepRefreshRotation * kSubstepRefreshRotation;
    };

    // The hypothetical owner pose of a body at its in-flight SolverBody state. The solver
    // integrates about the world centre of mass; convert back to the transform origin
    // exactly as the end-of-step writeback does (origin = COM − R·comLocal), then compose
    // the same TRS Transform::update builds. Static/Kinematic solver poses never advance
    // mid-step and were seeded from the same transform, so this is exact for them too.
    const auto ownerPoseAt = [&](const BodyEntry* entry)
    {
        const auto b = static_cast<std::size_t>(entry - bodies_.data());
        const SolverBody& sb = solverBodies[b];
        const Vec3 origin = sb.position - sb.orientation.rotate(entry->body.centerOfMassLocal());
        const Vec3 scale = entry->transform.scale();
        const Mat4 world = Mat4::translate(origin) * sb.orientation.toMat4() * Mat4::scale(scale);
        return OwnerPose{world, sb.orientation, scale, true};
    };

    bool any = false;
    for (std::size_t k = 0; k < island.contacts.size(); ++k)
    {
        const SolverContact& source = solverContacts[static_cast<std::size_t>(island.contacts[k])];
        if (source.subKey != 0)
        {
            continue; // mesh-triangle contacts are not refreshed (P9.6 stage 1)
        }
        if (!fastRotating(source.moving) && !fastRotating(source.target))
        {
            continue;
        }

        // Re-collide at the current poses, with the speculative band sized for the
        // remaining travel this step (mirrors the step-start closingReach).
        const auto mi = static_cast<std::size_t>(source.moving - bodies_.data());
        const auto ti = static_cast<std::size_t>(source.target - bodies_.data());
        const float closingReach =
            (solverBodies[mi].velocity.magnitude() + solverBodies[ti].velocity.magnitude()) *
                remaining +
            kSpeculativeDistance;
        const auto manifold = narrowPhase_->collide(
            worldShapeAt(*source.movingCollider, ownerPoseAt(source.moving)),
            worldShapeAt(*source.targetCollider, ownerPoseAt(source.target)), closingReach);

        SolverContactInput& in = islandContacts[k];
        if (!manifold.has_value() || manifold->count == 0)
        {
            in.pointCount = 0; // mid-step separation: the rows drop
        }
        else
        {
            in.normal = manifold->normal;
            in.pointCount = manifold->count;
            for (int p = 0; p < manifold->count; ++p)
            {
                const auto pi = static_cast<std::size_t>(p);
                in.points[pi] = manifold->points[pi].position;
                in.penetration[pi] = manifold->points[pi].penetration;
            }
        }
        flags[k] = 1;
        any = true;
    }
    return any;
}

void PhysicsWorld::solveIsland(const Island& island, std::vector<SolverBody>& solverBodies,
                               std::span<const SolverContactInput> contactInputs,
                               std::span<const SolverContact> solverContacts,
                               std::span<const JointInput> jointInputs, float dt)
{
    // Narrow the global constraint inputs to this island's subset (the solvers index
    // the shared global SolverBody array, so only which constraints they build changes).
    std::vector<SolverContactInput> islandContacts;
    islandContacts.reserve(island.contacts.size());
    for (const int c : island.contacts)
    {
        islandContacts.push_back(contactInputs[static_cast<std::size_t>(c)]);
    }
    std::vector<JointInput> islandJoints;
    islandJoints.reserve(island.joints.size());
    for (const int j : island.joints)
    {
        islandJoints.push_back(jointInputs[static_cast<std::size_t>(j)]);
    }

    const bool haveContacts = !islandContacts.empty();
    const bool haveJoints = !islandJoints.empty();

    // TGS soft-step (P9.2). Collision detection already ran once (in step()); here the
    // fixed step is split into kSubstepCount sub-steps of h. Constraints are prepared
    // ONCE at h (soft coefficients + anchors), then each substep: integrate gravity,
    // warm-start, bias-solve (joints then contacts, recomputing separation/error from
    // the moving pose), integrate positions, then a no-bias *relax* solve that removes
    // the bias velocity (the dissipation that stops the correction pumping energy).
    const float h = dt / static_cast<float>(kSubstepCount);

    if (haveContacts)
    {
        solver_->prepare(solverBodies, islandContacts, h);
    }
    if (haveJoints)
    {
        jointSolver_->prepare(solverBodies, islandJoints, h);
    }

    std::vector<std::uint8_t> refreshFlags;

    for (int s = 0; s < kSubstepCount; ++s)
    {
        // Mid-step manifold refresh (P9.6): halfway through the substeps, re-collide the
        // contacts of fast-rotating bodies at their current in-flight poses and re-prepare
        // those solver rows — the step-start manifold is stale by up to half the body's
        // per-step sweep by now. Gated, so slow bodies never pay for it.
        if (s == kSubstepCount / 2 && haveContacts &&
            refreshIslandContacts(island, solverBodies, solverContacts, islandContacts,
                                  refreshFlags, dt, dt - h * static_cast<float>(s)))
        {
            solver_->refresh(solverBodies, islandContacts, refreshFlags);
        }

        // Integrate velocities: gravity on this island's dynamic bodies (Kinematic
        // members are scene-driven nodes; Static are boundaries).
        for (const int bi : island.bodies)
        {
            const auto b = static_cast<std::size_t>(bi);
            BodyEntry& entry = bodies_[b];
            if (entry.body.type() != PhysicsBodyType::Dynamic)
            {
                continue;
            }
            solverBodies[b].velocity += gravity_ * entry.body.gravityScale() * h;
        }

        if (haveJoints)
        {
            jointSolver_->warmStart(solverBodies);
        }
        if (haveContacts)
        {
            solver_->warmStart(solverBodies);
        }

        for (int i = 0; i < kVelocityIterations; ++i)
        {
            if (haveJoints)
            {
                jointSolver_->solveVelocity(solverBodies, /*useBias=*/true);
            }
            if (haveContacts)
            {
                solver_->solveVelocity(solverBodies, /*useBias=*/true);
            }
        }

        // Integrate positions over h from the biased velocities.
        for (const int bi : island.bodies)
        {
            const auto b = static_cast<std::size_t>(bi);
            BodyEntry& entry = bodies_[b];
            if (entry.body.type() != PhysicsBodyType::Dynamic)
            {
                continue;
            }
            solverBodies[b].position += solverBodies[b].velocity * h;
            if (solverBodies[b].angularVelocity.magnitudeSquared() > 0.0f)
            {
                solverBodies[b].orientation =
                    solverBodies[b].orientation.integrate(solverBodies[b].angularVelocity, h);
            }
        }

        // Relax: same solve with no bias, removing the soft bias velocity so the
        // constraint correction does not feed energy back into the system.
        for (int i = 0; i < kVelocityIterations; ++i)
        {
            if (haveJoints)
            {
                jointSolver_->solveVelocity(solverBodies, /*useBias=*/false);
            }
            if (haveContacts)
            {
                solver_->solveVelocity(solverBodies, /*useBias=*/false);
            }
        }
    }

    // Single end-of-step restitution pass at the true impact velocity.
    if (haveContacts)
    {
        solver_->applyRestitution(solverBodies);
        // Push scene-driven Kinematic bodies out of penetration (Dynamic bodies already
        // resolved via the soft bias; positionWeight is 0 for them, so this only moves
        // Kinematic members).
        solver_->solvePosition(solverBodies);
    }

    // Write the solved velocities back to the dynamic bodies (positions/orientations
    // were advanced into solverBodies above and are written back by the caller), then
    // append this island's warm-start impulses.
    for (const int bi : island.bodies)
    {
        const auto b = static_cast<std::size_t>(bi);
        BodyEntry& entry = bodies_[b];
        if (entry.body.type() != PhysicsBodyType::Dynamic)
        {
            continue;
        }
        entry.body.linearVelocity(solverBodies[b].velocity);
        entry.body.angularVelocity(solverBodies[b].angularVelocity);
    }

    if (haveContacts)
    {
        solver_->store();
    }
    if (haveJoints)
    {
        jointSolver_->store();
    }
}

bool PhysicsWorld::islandShouldSleep(const Island& island) const
{
    if (!sleepingEnabled_)
    {
        return false;
    }

    const float angularThreshold = island.joints.empty() ? kAngularSleepThreshold : 0.15f;
    bool anyDynamic = false;
    for (const int bi : island.bodies)
    {
        const BodyEntry& entry = bodies_[static_cast<std::size_t>(bi)];
        if (entry.body.type() == PhysicsBodyType::Kinematic)
        {
            // A kinematic that moved this step keeps the island (its riders) awake.
            if (entry.transform.position() != entry.previousPosition)
            {
                return false;
            }
            continue;
        }
        anyDynamic = true;
        if (!entry.body.allowSleeping() || !belowSleepThreshold(entry.body, angularThreshold) ||
            entry.sleepTimer < kSleepTime)
        {
            return false;
        }
    }
    return anyDynamic;
}

bool PhysicsWorld::solveAndIntegrate(std::span<const SolverContact> contacts, float dt)
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
        solverBodies[i].angularVelocity = entry.body.angularVelocity();
        // The solver integrates about the world centre of mass, not the transform
        // origin (they coincide when centerOfMassLocal is zero — the common case).
        solverBodies[i].position = entry.transform.position() + entry.transform.rotation().rotate(
                                                                    entry.body.centerOfMassLocal());
        solverBodies[i].orientation = entry.transform.rotation();
        solverBodies[i].invMass = entry.active ? velocityInvMass(entry) : 0.0f;
        solverBodies[i].inverseInertiaLocal =
            entry.active ? entry.body.inverseInertiaLocal() : Vec3{};
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
        // Mesh triangles share a collider pair; mix the sub-key in so each warm-starts
        // independently. subKey 0 (ordinary pairs) leaves the key unchanged.
        if (contact.subKey != 0)
        {
            in.key ^= static_cast<std::uint64_t>(contact.subKey) * 0x9E3779B97F4A7C15ULL;
        }
        inputs.push_back(in);
    }

    // World-space joints for this step, composed from the live body transforms.
    const std::vector<JointInput> jointInputs = buildJointInputs();

    // Partition the movable bodies into islands (connected components linked by
    // movable-movable contacts/joints) and solve each independently. Islands don't
    // share solver bodies or constraints, so solving them separately is equivalent to
    // one global solve while letting P5.2 sleep whole islands. Kinematic bodies are
    // nodes (their contacts must be solved for the position pass) but never
    // velocity-integrated; Static bodies are pure boundaries.
    std::vector<bool> movableNode(bodies_.size(), false);
    for (std::size_t i = 0; i < bodies_.size(); ++i)
    {
        movableNode[i] = bodies_[i].active && movable(bodies_[i]);
    }
    std::vector<IslandEdge> contactEdges;
    contactEdges.reserve(inputs.size());
    for (const SolverContactInput& in : inputs)
    {
        contactEdges.push_back(IslandEdge{in.bodyA, in.bodyB});
    }
    std::vector<IslandEdge> jointEdges;
    jointEdges.reserve(jointInputs.size());
    for (const JointInput& in : jointInputs)
    {
        jointEdges.push_back(IslandEdge{in.bodyA, in.bodyB});
    }
    const std::vector<Island> islands =
        buildIslands(bodies_.size(), movableNode, contactEdges, jointEdges);

    // One solver instance services every island in turn; the warm-start cache is
    // cleared once, appended per island, and committed once. Fully-settled islands
    // sleep — skipped entirely until disturbed.
    solver_->beginStore();
    jointSolver_->beginStore();
    for (const Island& island : islands)
    {
        if (islandShouldSleep(island))
        {
            // Put (or keep) the island asleep: zero the dynamic members' velocities,
            // flag them sleeping, and skip the solve + integration.
            projectSleepingJointAnchors(island, jointInputs, solverBodies);
            for (const int bi : island.bodies)
            {
                BodyEntry& entry = bodies_[static_cast<std::size_t>(bi)];
                if (entry.body.type() != PhysicsBodyType::Dynamic)
                {
                    continue;
                }
                entry.body.linearVelocity(Vec3{});
                entry.body.angularVelocity(Vec3{});
                entry.sleeping = true;
            }
            continue;
        }

        // Awake: wake any sleeping members, resetting their timer on the transition.
        for (const int bi : island.bodies)
        {
            BodyEntry& entry = bodies_[static_cast<std::size_t>(bi)];
            if (entry.sleeping)
            {
                entry.sleeping = false;
                entry.sleepTimer = 0.0f;
            }
        }

        solveIsland(island, solverBodies, inputs, contacts, jointInputs, dt);

        // Accumulate each dynamic member's sleep timer from its post-solve velocity.
        for (const int bi : island.bodies)
        {
            BodyEntry& entry = bodies_[static_cast<std::size_t>(bi)];
            if (entry.body.type() != PhysicsBodyType::Dynamic)
            {
                continue;
            }
            const float angularThreshold = island.joints.empty() ? kAngularSleepThreshold : 0.15f;
            if (belowSleepThreshold(entry.body, angularThreshold))
            {
                entry.sleepTimer += dt;
            }
            else
            {
                entry.sleepTimer = 0.0f;
            }
        }
    }
    solver_->commitStore();
    jointSolver_->commitStore();

    // Write final positions + orientations back (everything but Static).
    bool moved = false;
    for (std::size_t i = 0; i < bodies_.size(); ++i)
    {
        BodyEntry& entry = bodies_[i];
        if (!entry.active || entry.body.type() == PhysicsBodyType::Static)
        {
            continue;
        }
        bool changed = false;
        // solverBodies[i].position is the world centre of mass; convert back to the
        // transform origin (origin = COM − R·comLocal). Identity when comLocal is zero.
        const Vec3 origin = solverBodies[i].position -
                            solverBodies[i].orientation.rotate(entry.body.centerOfMassLocal());
        if (entry.transform.position() != origin)
        {
            entry.transform.position(origin);
            changed = true;
        }
        // Orientation was integrated into the solver body (from the solved angular
        // velocity, plus any pseudo-angular position correction) — Dynamic only.
        if (entry.body.type() == PhysicsBodyType::Dynamic &&
            !(entry.transform.rotation() == solverBodies[i].orientation))
        {
            entry.transform.rotation(solverBodies[i].orientation);
            changed = true;
        }
        if (changed)
        {
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
