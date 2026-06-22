#include <fire_engine/physics/physics_world.hpp>

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/core/convex_hull_builder.hpp>

using Catch::Approx;
using fire_engine::BoxShape;
using fire_engine::buildConvexHull;
using fire_engine::ColliderDesc;
using fire_engine::ColliderShape;
using fire_engine::ConvexHullShape;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsColliderHandle;
using fire_engine::PhysicsWorld;
using fire_engine::QueryFilter;
using fire_engine::Ray;
using fire_engine::SphereShape;
using fire_engine::StaticMeshShape;
using fire_engine::Transform;
using fire_engine::Vec3;

namespace
{

PhysicsColliderHandle addStatic(PhysicsWorld& world, Vec3 position, const ColliderShape& shape,
                                std::uint32_t layer = 1U, std::uint32_t mask = ~0U)
{
    PhysicsBodyDesc body;
    body.type = PhysicsBodyType::Static;
    body.position = position;
    const PhysicsBodyHandle handle = world.createBody(body);
    ColliderDesc collider;
    collider.shape = shape;
    collider.collisionLayer = layer;
    collider.collisionMask = mask;
    return world.createCollider(handle, collider);
}

ConvexHullShape boxHull(float h)
{
    const std::vector<Vec3> verts{
        {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
        {-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h},
    };
    const std::vector<std::uint32_t> idx{
        1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3, 3, 7, 6, 3, 6, 2,
        0, 1, 5, 0, 5, 4, 4, 5, 6, 4, 6, 7, 0, 3, 2, 0, 2, 1,
    };
    return buildConvexHull(verts, idx);
}

} // namespace

TEST_CASE("PhysicsQuery.RaycastHitsSphere", "[PhysicsQuery]")
{
    PhysicsWorld world;
    const auto collider = addStatic(world, {5.0f, 0.0f, 0.0f}, SphereShape{1.0f, Vec3{}});

    const auto hit = world.raycast(Ray{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f});
    REQUIRE(hit.has_value());
    CHECK(hit->collider == collider);
    CHECK(hit->distance == Approx(4.0f).margin(1e-3f));
    CHECK(hit->normal.x() == Approx(-1.0f).margin(1e-3f));
}

TEST_CASE("PhysicsQuery.RaycastMissesOffAxis", "[PhysicsQuery]")
{
    PhysicsWorld world;
    addStatic(world, {5.0f, 0.0f, 0.0f}, SphereShape{1.0f, Vec3{}});
    CHECK_FALSE(world.raycast(Ray{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 100.0f}).has_value());
}

TEST_CASE("PhysicsQuery.RaycastReturnsNearest", "[PhysicsQuery]")
{
    PhysicsWorld world;
    const auto near = addStatic(world, {3.0f, 0.0f, 0.0f}, SphereShape{0.5f, Vec3{}});
    addStatic(world, {6.0f, 0.0f, 0.0f}, SphereShape{0.5f, Vec3{}});

    const auto hit = world.raycast(Ray{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f});
    REQUIRE(hit.has_value());
    CHECK(hit->collider == near);
    CHECK(hit->distance == Approx(2.5f).margin(1e-3f));
}

TEST_CASE("PhysicsQuery.RaycastRespectsLayerFilter", "[PhysicsQuery]")
{
    PhysicsWorld world;
    addStatic(world, {5.0f, 0.0f, 0.0f}, SphereShape{1.0f, Vec3{}}, /*layer=*/0b10U);

    // Filter mask admits only layer bit 0b01 → the layer-0b10 collider is excluded.
    const QueryFilter filter{.layer = 1U, .mask = 0b01U};
    CHECK_FALSE(
        world.raycast(Ray{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f}, filter).has_value());
    // No filter (defaults) hits it.
    CHECK(world.raycast(Ray{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f}).has_value());
}

TEST_CASE("PhysicsQuery.RaycastHitsConvexHull", "[PhysicsQuery]")
{
    PhysicsWorld world;
    addStatic(world, {5.0f, 0.0f, 0.0f}, boxHull(1.0f));
    const auto hit = world.raycast(Ray{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f});
    REQUIRE(hit.has_value());
    CHECK(hit->distance == Approx(4.0f).margin(1e-2f)); // front face at x = 4
}

TEST_CASE("PhysicsQuery.RaycastHitsStaticMesh", "[PhysicsQuery]")
{
    PhysicsWorld world;
    PhysicsBodyDesc floorDesc;
    floorDesc.type = PhysicsBodyType::Static;
    const PhysicsBodyHandle floor = world.createBody(floorDesc);
    StaticMeshShape mesh;
    mesh.vertices = {
        {-10.0f, 0.0f, -10.0f}, {10.0f, 0.0f, -10.0f}, {10.0f, 0.0f, 10.0f}, {-10.0f, 0.0f, 10.0f}};
    mesh.indices = {0, 1, 2, 0, 2, 3};
    (void)world.createMeshCollider(floor, mesh);

    const auto hit = world.raycast(Ray{{0.0f, 5.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, 100.0f});
    REQUIRE(hit.has_value());
    CHECK(hit->distance == Approx(5.0f).margin(1e-3f));
    CHECK(hit->normal.y() == Approx(1.0f).margin(1e-3f));
}

TEST_CASE("PhysicsQuery.OverlapSphereFindsAndExcludes", "[PhysicsQuery]")
{
    PhysicsWorld world;
    const auto box = addStatic(world, {0.0f, 0.0f, 0.0f}, BoxShape{Vec3{0.5f, 0.5f, 0.5f}, Vec3{}});
    addStatic(world, {10.0f, 0.0f, 0.0f}, BoxShape{Vec3{0.5f, 0.5f, 0.5f}, Vec3{}});

    const auto hits = world.overlapSphere(Vec3{0.0f, 0.0f, 0.0f}, 1.0f);
    REQUIRE(hits.size() == 1);
    CHECK(hits.front().collider == box);
}

TEST_CASE("PhysicsQuery.OverlapShapeBox", "[PhysicsQuery]")
{
    PhysicsWorld world;
    const auto box = addStatic(world, {0.0f, 0.0f, 0.0f}, BoxShape{Vec3{0.5f, 0.5f, 0.5f}, Vec3{}});

    Transform pose;
    pose.position(Vec3{0.4f, 0.0f, 0.0f});
    pose.update(fire_engine::Mat4::identity());
    const auto hits = world.overlapShape(BoxShape{Vec3{0.5f, 0.5f, 0.5f}, Vec3{}}, pose);
    REQUIRE(hits.size() == 1);
    CHECK(hits.front().collider == box);
}

TEST_CASE("PhysicsQuery.ShapecastSphereHitsCollider", "[PhysicsQuery]")
{
    PhysicsWorld world;
    const auto target = addStatic(world, {5.0f, 0.0f, 0.0f}, SphereShape{0.5f, Vec3{}});

    Transform pose;
    pose.position(Vec3{0.0f, 0.0f, 0.0f});
    pose.update(fire_engine::Mat4::identity());
    const auto hit =
        world.shapecast(SphereShape{0.5f, Vec3{}}, pose, Vec3{1.0f, 0.0f, 0.0f}, 100.0f);
    REQUIRE(hit.has_value());
    CHECK(hit->collider == target);
    CHECK(hit->distance == Approx(4.0f).margin(1e-2f));
}
