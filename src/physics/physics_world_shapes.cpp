#include <fire_engine/physics/physics_world.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <variant>

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

namespace
{

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

[[nodiscard]] Vec3 matrixScale(const Mat4& m)
{
    return {Vec3{m[0, 0], m[1, 0], m[2, 0]}.magnitude(),
            Vec3{m[0, 1], m[1, 1], m[2, 1]}.magnitude(),
            Vec3{m[0, 2], m[1, 2], m[2, 2]}.magnitude()};
}

} // namespace

WorldShape PhysicsWorld::composeWorldShape(const ColliderShape& shape, const Mat4& world,
                                           const Quaternion& rot, const Vec3& scale)
{
    const float uniform = std::max({scale.x(), scale.y(), scale.z()});

    if (const auto* sphere = std::get_if<SphereShape>(&shape))
    {
        return WorldSphere{transformPoint(world, sphere->center), sphere->radius * uniform};
    }
    if (const auto* box = std::get_if<BoxShape>(&shape))
    {
        return WorldBox{transformPoint(world, box->center),
                        Vec3{box->halfExtents.x() * scale.x(), box->halfExtents.y() * scale.y(),
                             box->halfExtents.z() * scale.z()},
                        rot};
    }
    if (const auto* capsule = std::get_if<CapsuleShape>(&shape))
    {
        const Vec3 c = capsule->center;
        const Vec3 p0 = transformPoint(world, Vec3{c.x(), c.y() - capsule->halfHeight, c.z()});
        const Vec3 p1 = transformPoint(world, Vec3{c.x(), c.y() + capsule->halfHeight, c.z()});
        return WorldCapsule{p0, p1, capsule->radius * uniform};
    }
    if (const auto* hull = std::get_if<ConvexHullShape>(&shape))
    {
        WorldConvex convex;
        convex.vertices.reserve(hull->vertices.size());
        for (const Vec3& v : hull->vertices)
        {
            convex.vertices.push_back(transformPoint(world, v));
        }
        convex.faces = hull->faces;
        return convex;
    }
    const auto& aabb = std::get<AabbShape>(shape);
    const Vec3 he = aabb.bounds.extent() * 0.5f;
    return WorldBox{transformPoint(world, aabb.bounds.center()),
                    Vec3{he.x() * scale.x(), he.y() * scale.y(), he.z() * scale.z()}, rot};
}

AABB PhysicsWorld::aabbOfWorldShape(const WorldShape& shape) noexcept
{
    return std::visit(
        [](const auto& s) -> AABB
        {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, WorldSphere>)
            {
                const Vec3 r{s.radius, s.radius, s.radius};
                return AABB{s.center - r, s.center + r};
            }
            else if constexpr (std::is_same_v<T, WorldBox>)
            {
                const Vec3 ax = s.orientation.rotate(Vec3{1.0f, 0.0f, 0.0f});
                const Vec3 ay = s.orientation.rotate(Vec3{0.0f, 1.0f, 0.0f});
                const Vec3 az = s.orientation.rotate(Vec3{0.0f, 0.0f, 1.0f});
                const Vec3 e{
                    std::abs(ax.x()) * s.halfExtents.x() + std::abs(ay.x()) * s.halfExtents.y() +
                        std::abs(az.x()) * s.halfExtents.z(),
                    std::abs(ax.y()) * s.halfExtents.x() + std::abs(ay.y()) * s.halfExtents.y() +
                        std::abs(az.y()) * s.halfExtents.z(),
                    std::abs(ax.z()) * s.halfExtents.x() + std::abs(ay.z()) * s.halfExtents.y() +
                        std::abs(az.z()) * s.halfExtents.z()};
                return AABB{s.center - e, s.center + e};
            }
            else if constexpr (std::is_same_v<T, WorldCapsule>)
            {
                const Vec3 lo{std::min(s.p0.x(), s.p1.x()), std::min(s.p0.y(), s.p1.y()),
                              std::min(s.p0.z(), s.p1.z())};
                const Vec3 hi{std::max(s.p0.x(), s.p1.x()), std::max(s.p0.y(), s.p1.y()),
                              std::max(s.p0.z(), s.p1.z())};
                const Vec3 r{s.radius, s.radius, s.radius};
                return AABB{lo - r, hi + r};
            }
            else
            {
                Vec3 lo{s.vertices[0]};
                Vec3 hi{s.vertices[0]};
                for (const Vec3& v : s.vertices)
                {
                    lo = {std::min(lo.x(), v.x()), std::min(lo.y(), v.y()),
                          std::min(lo.z(), v.z())};
                    hi = {std::max(hi.x(), v.x()), std::max(hi.y(), v.y()),
                          std::max(hi.z(), v.z())};
                }
                return AABB{lo, hi};
            }
        },
        shape);
}

PhysicsWorld::OwnerPose PhysicsWorld::colliderOwnerPose(const ColliderEntry& entry) const
{
    if (entry.isLinkCollider())
    {
        const Articulation* art = findArticulation(entry.articulation);
        if (art == nullptr || entry.link < 0 ||
            static_cast<std::size_t>(entry.link) >= art->linkCount())
        {
            return OwnerPose{};
        }
        const RigidTransform lw = art->linkWorld(static_cast<std::size_t>(entry.link));
        const Mat4 world = Mat4::translate(lw.translation) * lw.rotation.toMat4();
        return OwnerPose{world, lw.rotation, Vec3{1.0f, 1.0f, 1.0f}, true};
    }

    const BodyEntry* owner = findBody(entry.body);
    if (owner == nullptr)
    {
        return OwnerPose{};
    }
    const Mat4 world = owner->transform.world();
    return OwnerPose{world, owner->transform.rotation(), matrixScale(world), true};
}

WorldShape PhysicsWorld::worldShapeAt(const ColliderEntry& entry, const OwnerPose& owner) const
{
    // Compose the owner world with the collider's local offset (identity for a plain
    // single collider; a compound child's placement otherwise). Shape dimensions take
    // the owner scale; orientation is owner × child rotation. Split from worldShape so a
    // caller can compose against a *hypothetical* owner pose (the mid-step manifold
    // refresh builds one from the in-flight SolverBody state).
    const Mat4 world =
        owner.world * (Mat4::translate(entry.localPosition) * entry.localRotation.toMat4());
    const Quaternion rot = owner.rotation * entry.localRotation;
    return composeWorldShape(entry.shape, world, rot, owner.scale);
}

WorldShape PhysicsWorld::worldShape(const ColliderEntry& entry) const
{
    // Owner world pose — a rigid body's transform or an articulation link's
    // forward-kinematics pose (identity when the owner is missing, which shouldn't happen
    // for an active collider).
    return worldShapeAt(entry, colliderOwnerPose(entry));
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
            else if constexpr (std::is_same_v<Shape, CapsuleShape>)
            {
                const Vec3 extents{value.radius, value.radius + value.halfHeight, value.radius};
                return {value.center - extents, value.center + extents};
            }
            else // ConvexHullShape — AABB of the hull vertices
            {
                AABB bounds{value.vertices.empty() ? Vec3{} : value.vertices.front(),
                            value.vertices.empty() ? Vec3{} : value.vertices.front()};
                for (const Vec3& v : value.vertices)
                {
                    bounds.min = {std::min(bounds.min.x(), v.x()), std::min(bounds.min.y(), v.y()),
                                  std::min(bounds.min.z(), v.z())};
                    bounds.max = {std::max(bounds.max.x(), v.x()), std::max(bounds.max.y(), v.y()),
                                  std::max(bounds.max.z(), v.z())};
                }
                return bounds;
            }
        },
        shape);
}

void PhysicsWorld::updateCollider(ColliderEntry& collider, float dt)
{
    const OwnerPose owner = colliderOwnerPose(collider);
    if (!owner.valid)
    {
        return;
    }

    // Predicted displacement this step (rigid Dynamic bodies only — kinematic/static were
    // already moved into place by the scene before step(); a link collider's motion comes
    // from forward kinematics, so it carries no separate linear-velocity term in Phase A).
    // Threaded into the swept bound so the broadphase pairs fast movers with what they
    // are about to reach.
    Vec3 motion{};
    if (!collider.isLinkCollider())
    {
        const BodyEntry* body = findBody(collider.body);
        if (body != nullptr && body->body.type() == PhysicsBodyType::Dynamic)
        {
            motion = body->body.linearVelocity() * dt;
        }
    }

    const Mat4 childMat = Mat4::translate(collider.localPosition) * collider.localRotation.toMat4();
    collider.collider.localBounds(transformAabb(childMat, localBounds(collider.shape)));
    collider.collider.update(owner.world, motion);
}

void PhysicsWorld::resetCollider(ColliderEntry& collider)
{
    const OwnerPose owner = colliderOwnerPose(collider);
    if (!owner.valid)
    {
        return;
    }

    const Mat4 childMat = Mat4::translate(collider.localPosition) * collider.localRotation.toMat4();
    collider.collider.localBounds(transformAabb(childMat, localBounds(collider.shape)));
    collider.collider.resetFrame(owner.world);
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

} // namespace fire_engine
