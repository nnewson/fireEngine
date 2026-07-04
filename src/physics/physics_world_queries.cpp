#include <fire_engine/physics/physics_world.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include <fire_engine/collision/gjk_epa.hpp>
#include <fire_engine/collision/ray.hpp>
#include <fire_engine/collision/shape_cast.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

namespace
{

// Per-axis scale from the world matrix columns.
[[nodiscard]] Vec3 matrixScale(const Mat4& m)
{
    return {Vec3{m[0, 0], m[1, 0], m[2, 0]}.magnitude(),
            Vec3{m[0, 1], m[1, 1], m[2, 1]}.magnitude(),
            Vec3{m[0, 2], m[1, 2], m[2, 2]}.magnitude()};
}

[[nodiscard]] bool passesFilter(const Collider& collider, QueryFilter filter) noexcept
{
    return (filter.mask & collider.collisionLayer()) != 0U &&
           (collider.collisionMask() & filter.layer) != 0U;
}

} // namespace

std::optional<RaycastHit> PhysicsWorld::raycast(const Ray& ray, QueryFilter filter) const
{
    std::optional<RaycastHit> best;
    const auto consider = [&](const RayHit& hit, const ColliderEntry& e)
    {
        if (!best.has_value() || hit.distance < best->distance)
        {
            best = RaycastHit{e.handle, e.body, hit.point, hit.normal, hit.distance};
        }
    };

    for (const ColliderEntry& e : colliders_)
    {
        if (!e.active || !passesFilter(e.collider, filter))
        {
            continue;
        }
        float tNear = 0.0f;
        if (!rayIntersectsAabb(ray, e.collider.worldBounds(), tNear))
        {
            continue;
        }
        if (best.has_value() && tNear >= best->distance)
        {
            continue; // the whole AABB is farther than the closest hit so far
        }

        if (e.mesh)
        {
            const MeshCollisionData& mesh = *e.mesh;
            mesh.bvh.traverse(
                [&](const AABB& box)
                {
                    float t = 0.0f;
                    return rayIntersectsAabb(ray, box, t);
                },
                [&](int proxy)
                {
                    const auto base = static_cast<std::size_t>(mesh.bvh.payload(proxy)) * 3;
                    const Vec3& v0 = mesh.worldVertices[mesh.indices[base + 0]];
                    const Vec3& v1 = mesh.worldVertices[mesh.indices[base + 1]];
                    const Vec3& v2 = mesh.worldVertices[mesh.indices[base + 2]];
                    if (auto hit = rayIntersectTriangle(ray, v0, v1, v2))
                    {
                        consider(*hit, e);
                    }
                });
        }
        else if (auto hit = rayIntersect(ray, worldShape(e)))
        {
            consider(*hit, e);
        }
    }
    return best;
}

std::vector<RaycastHit> PhysicsWorld::raycastAll(const Ray& ray, QueryFilter filter) const
{
    std::vector<RaycastHit> hits;
    for (const ColliderEntry& e : colliders_)
    {
        if (!e.active || !passesFilter(e.collider, filter))
        {
            continue;
        }
        float tNear = 0.0f;
        if (!rayIntersectsAabb(ray, e.collider.worldBounds(), tNear))
        {
            continue;
        }

        if (e.mesh)
        {
            const MeshCollisionData& mesh = *e.mesh;
            std::optional<RayHit> nearest;
            mesh.bvh.traverse(
                [&](const AABB& box)
                {
                    float t = 0.0f;
                    return rayIntersectsAabb(ray, box, t);
                },
                [&](int proxy)
                {
                    const auto base = static_cast<std::size_t>(mesh.bvh.payload(proxy)) * 3;
                    const Vec3& v0 = mesh.worldVertices[mesh.indices[base + 0]];
                    const Vec3& v1 = mesh.worldVertices[mesh.indices[base + 1]];
                    const Vec3& v2 = mesh.worldVertices[mesh.indices[base + 2]];
                    if (auto hit = rayIntersectTriangle(ray, v0, v1, v2))
                    {
                        if (!nearest.has_value() || hit->distance < nearest->distance)
                        {
                            nearest = hit;
                        }
                    }
                });
            if (nearest.has_value())
            {
                hits.push_back(RaycastHit{e.handle, e.body, nearest->point, nearest->normal,
                                          nearest->distance});
            }
        }
        else if (auto hit = rayIntersect(ray, worldShape(e)))
        {
            hits.push_back(RaycastHit{e.handle, e.body, hit->point, hit->normal, hit->distance});
        }
    }
    return hits;
}

std::optional<ShapecastHit> PhysicsWorld::shapecast(const ColliderShape& shape,
                                                    const Transform& pose, Vec3 direction,
                                                    float maxDistance, QueryFilter filter) const
{
    const Vec3 dir = Vec3::normalise(direction);
    const WorldShape moving =
        composeWorldShape(shape, pose.world(), pose.rotation(), matrixScale(pose.world()));
    // Sweep AABB: the moving shape's bounds extended along the sweep.
    AABB sweptBounds = aabbOfWorldShape(moving);
    const Vec3 sweep = dir * maxDistance;
    sweptBounds = AABB{{std::min(sweptBounds.min.x(), sweptBounds.min.x() + sweep.x()),
                        std::min(sweptBounds.min.y(), sweptBounds.min.y() + sweep.y()),
                        std::min(sweptBounds.min.z(), sweptBounds.min.z() + sweep.z())},
                       {std::max(sweptBounds.max.x(), sweptBounds.max.x() + sweep.x()),
                        std::max(sweptBounds.max.y(), sweptBounds.max.y() + sweep.y()),
                        std::max(sweptBounds.max.z(), sweptBounds.max.z() + sweep.z())}};

    std::optional<ShapecastHit> best;
    const auto consider = [&](const ToiHit& hit, const ColliderEntry& e)
    {
        if (!best.has_value() || hit.distance < best->distance)
        {
            best = ShapecastHit{e.handle, e.body, hit.point, hit.normal, hit.distance};
        }
    };

    for (const ColliderEntry& e : colliders_)
    {
        if (!e.active || !passesFilter(e.collider, filter))
        {
            continue;
        }
        if (!aabbOverlaps(e.collider.worldBounds(), sweptBounds))
        {
            continue;
        }

        if (e.mesh)
        {
            const MeshCollisionData& mesh = *e.mesh;
            mesh.bvh.query(
                sweptBounds,
                [&](int proxy)
                {
                    const auto base = static_cast<std::size_t>(mesh.bvh.payload(proxy)) * 3;
                    const std::array<ConvexFace, 1> faces{
                        ConvexFace{Vec3{}, std::vector<int>{0, 1, 2}}};
                    WorldConvex triangle;
                    triangle.vertices = {mesh.worldVertices[mesh.indices[base + 0]],
                                         mesh.worldVertices[mesh.indices[base + 1]],
                                         mesh.worldVertices[mesh.indices[base + 2]]};
                    triangle.faces = faces;
                    if (auto hit = shapeCast(moving, dir, maxDistance, WorldShape{triangle}))
                    {
                        consider(*hit, e);
                    }
                });
        }
        else if (auto hit = shapeCast(moving, dir, maxDistance, worldShape(e)))
        {
            consider(*hit, e);
        }
    }
    return best;
}

std::vector<OverlapHit> PhysicsWorld::overlapWorldShape(const WorldShape& query,
                                                        const AABB& queryAabb,
                                                        QueryFilter filter) const
{
    std::vector<OverlapHit> hits;
    for (const ColliderEntry& e : colliders_)
    {
        if (!e.active || !passesFilter(e.collider, filter))
        {
            continue;
        }
        if (!aabbOverlaps(e.collider.worldBounds(), queryAabb))
        {
            continue;
        }

        if (e.mesh)
        {
            const MeshCollisionData& mesh = *e.mesh;
            bool overlapped = false;
            mesh.bvh.query(queryAabb,
                           [&](int proxy)
                           {
                               if (overlapped)
                               {
                                   return;
                               }
                               const auto base =
                                   static_cast<std::size_t>(mesh.bvh.payload(proxy)) * 3;
                               const std::array<ConvexFace, 1> faces{
                                   ConvexFace{Vec3{}, std::vector<int>{0, 1, 2}}};
                               WorldConvex triangle;
                               triangle.vertices = {mesh.worldVertices[mesh.indices[base + 0]],
                                                    mesh.worldVertices[mesh.indices[base + 1]],
                                                    mesh.worldVertices[mesh.indices[base + 2]]};
                               triangle.faces = faces;
                               if (gjkEpaContact(query, WorldShape{triangle}).colliding)
                               {
                                   overlapped = true;
                               }
                           });
            if (overlapped)
            {
                hits.push_back(OverlapHit{e.handle, e.body});
            }
        }
        else if (gjkEpaContact(query, worldShape(e)).colliding)
        {
            hits.push_back(OverlapHit{e.handle, e.body});
        }
    }
    return hits;
}

std::vector<OverlapHit> PhysicsWorld::overlapShape(const ColliderShape& shape,
                                                   const Transform& pose, QueryFilter filter) const
{
    const WorldShape query =
        composeWorldShape(shape, pose.world(), pose.rotation(), matrixScale(pose.world()));
    return overlapWorldShape(query, aabbOfWorldShape(query), filter);
}

std::vector<OverlapHit> PhysicsWorld::overlapSphere(Vec3 center, float radius,
                                                    QueryFilter filter) const
{
    const WorldShape query = WorldSphere{center, radius};
    return overlapWorldShape(query, aabbOfWorldShape(query), filter);
}

} // namespace fire_engine
