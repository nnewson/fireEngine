#include <fire_engine/physics/physics_world.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::BoxShape;
using fire_engine::ColliderDesc;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsMaterial;
using fire_engine::PhysicsWorld;
using fire_engine::SphereShape;
using fire_engine::StaticMeshShape;
using fire_engine::Vec3;

namespace
{

// A flat 20×20 floor mesh at y = 0 (two triangles, wound CCW from above → +y normal).
StaticMeshShape floorMesh()
{
    StaticMeshShape m;
    m.vertices = {
        {-10.0f, 0.0f, -10.0f}, {10.0f, 0.0f, -10.0f}, {10.0f, 0.0f, 10.0f}, {-10.0f, 0.0f, 10.0f}};
    m.indices = {0, 2, 1, 0, 3, 2};
    return m;
}

// A thin 10×10 wall mesh in the x-y plane at z = 0 (two triangles, +z normal).
StaticMeshShape wallMesh()
{
    StaticMeshShape m;
    m.vertices = {
        {-5.0f, -5.0f, 0.0f}, {5.0f, -5.0f, 0.0f}, {5.0f, 5.0f, 0.0f}, {-5.0f, 5.0f, 0.0f}};
    m.indices = {0, 2, 1, 0, 3, 2}; // front face toward -z
    return m;
}

PhysicsBodyHandle createMeshFloor(PhysicsWorld& physics, const StaticMeshShape& mesh)
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    const auto body = physics.createBody(desc);
    PhysicsMaterial mat;
    mat.restitution = 0.0f;
    mat.friction = 0.6f;
    (void)physics.createMeshCollider(body, mesh, mat);
    return body;
}

void stepMany(PhysicsWorld& physics, int steps)
{
    for (int i = 0; i < steps; ++i)
    {
        physics.step(1.0f / 60.0f);
    }
}

} // namespace

TEST_CASE("StaticMesh.RejectsNonStaticBody", "[StaticMesh]")
{
    PhysicsWorld physics;
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    const auto body = physics.createBody(desc);
    CHECK_FALSE(physics.createMeshCollider(body, floorMesh()).valid());
}

TEST_CASE("StaticMesh.BoxSettlesOnMeshFloor", "[StaticMesh]")
{
    // A box dropped onto a triangulated floor settles on its surface — proving contacts
    // are generated against the mesh's triangles (multiple per pair), not its AABB.
    PhysicsWorld physics;
    createMeshFloor(physics, floorMesh());

    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = {0.0f, 3.0f, 0.0f};
    desc.gravityScale = 1.0f;
    desc.material.restitution = 0.0f;
    const auto box = physics.createBody(desc);
    ColliderDesc collider;
    collider.shape = BoxShape{Vec3{0.5f, 0.5f, 0.5f}, Vec3{}};
    (void)physics.createCollider(box, collider);

    stepMany(physics, 240);

    const auto p = physics.bodyTransform(box)->position();
    CHECK(p.y() == Catch::Approx(0.5f).margin(0.02f)); // half-extent above the floor
    CHECK(physics.sleeping(box));
}

TEST_CASE("StaticMesh.SphereSettlesOnMeshFloor", "[StaticMesh]")
{
    PhysicsWorld physics;
    createMeshFloor(physics, floorMesh());

    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = {0.0f, 3.0f, 0.0f};
    desc.gravityScale = 1.0f;
    desc.material.restitution = 0.0f;
    const auto ball = physics.createBody(desc);
    ColliderDesc collider;
    collider.shape = SphereShape{0.5f, Vec3{}};
    (void)physics.createCollider(ball, collider);

    stepMany(physics, 240);

    CHECK(physics.bodyTransform(ball)->position().y() == Catch::Approx(0.5f).margin(0.02f));
}

TEST_CASE("StaticMesh.FrictionDeceleratesSlideOnMesh", "[StaticMesh]")
{
    // A box sliding across the mesh floor under friction loses its horizontal speed,
    // while on a frictionless mesh it keeps sliding.
    auto finalSpeed = [](float friction)
    {
        PhysicsWorld physics;
        StaticMeshShape mesh = floorMesh();
        PhysicsBodyDesc floorDesc;
        floorDesc.type = PhysicsBodyType::Static;
        floorDesc.material.friction = friction; // friction combines from the body materials
        const auto floor = physics.createBody(floorDesc);
        PhysicsMaterial mat;
        mat.restitution = 0.0f;
        mat.friction = friction;
        (void)physics.createMeshCollider(floor, mesh, mat);

        PhysicsBodyDesc desc;
        desc.type = PhysicsBodyType::Dynamic;
        desc.position = {0.0f, 0.5f, 0.0f};
        desc.linearVelocity = {3.0f, 0.0f, 0.0f};
        desc.gravityScale = 1.0f;
        desc.material.restitution = 0.0f;
        desc.material.friction = friction;
        const auto box = physics.createBody(desc);
        ColliderDesc collider;
        collider.shape = BoxShape{Vec3{0.5f, 0.5f, 0.5f}, Vec3{}};
        (void)physics.createCollider(box, collider);

        stepMany(physics, 180);
        return physics.body(box)->linearVelocity().x();
    };

    CHECK(finalSpeed(0.8f) < 1.0f);                              // high friction → decelerated
    CHECK(finalSpeed(0.0f) == Catch::Approx(3.0f).margin(0.1f)); // frictionless → keeps sliding
}

TEST_CASE("StaticMesh.FastBodyDoesNotTunnelThroughThinMesh", "[StaticMesh]")
{
    // A fast sphere fired at a thin mesh wall must be stopped by the speculative
    // swept-AABB triangle query, not pass through it.
    PhysicsWorld physics;
    PhysicsBodyDesc wallDesc;
    wallDesc.type = PhysicsBodyType::Static;
    const auto wall = physics.createBody(wallDesc);
    PhysicsMaterial mat;
    mat.restitution = 0.0f;
    (void)physics.createMeshCollider(wall, wallMesh(), mat);

    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = {0.0f, 0.0f, -5.0f};
    desc.linearVelocity = {0.0f, 0.0f, 120.0f}; // 2 m/step at 1/60 — would tunnel a thin wall
    desc.gravityScale = 0.0f;
    desc.material.restitution = 0.0f;
    const auto ball = physics.createBody(desc);
    ColliderDesc collider;
    collider.shape = SphereShape{0.5f, Vec3{}};
    (void)physics.createCollider(ball, collider);

    stepMany(physics, 60);

    // Stopped on the near (-z) side of the wall (sphere radius 0.5 → centre z ≈ -0.5),
    // not tunnelled to large +z.
    CHECK(physics.bodyTransform(ball)->position().z() < 0.0f);
}
