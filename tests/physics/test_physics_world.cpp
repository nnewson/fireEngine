#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/scene_graph.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <utility>

#include <fire_engine/core/convex_hull_builder.hpp>

using fire_engine::AabbShape;
using fire_engine::BoxShape;
using fire_engine::buildConvexHull;
using fire_engine::ClothColliderType;
using fire_engine::ColliderDesc;
using fire_engine::ConvexHullShape;
using fire_engine::InputState;
using fire_engine::Node;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsColliderHandle;
using fire_engine::PhysicsMaterial;
using fire_engine::PhysicsWorld;
using fire_engine::pi;
using fire_engine::Quaternion;
using fire_engine::SceneGraph;
using fire_engine::SphereShape;
using fire_engine::Vec3;

namespace
{

ColliderDesc unitCollider()
{
    ColliderDesc desc;
    desc.shape = AabbShape{{Vec3{0.0f, 0.0f, 0.0f}, Vec3{1.0f, 1.0f, 1.0f}}};
    return desc;
}

// A cube convex hull of half-extent `h`, built from a triangle mesh (exercises the
// full authoring path: weld + coplanar-merge into 6 quad faces).
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

PhysicsBodyHandle createBox(PhysicsWorld& physics, PhysicsBodyType type, Vec3 position,
                            Vec3 velocity = {})
{
    PhysicsBodyDesc bodyDesc;
    bodyDesc.type = type;
    bodyDesc.position = position;
    bodyDesc.linearVelocity = velocity;
    bodyDesc.gravityScale = 0.0f;
    auto handle = physics.createBody(bodyDesc);
    [[maybe_unused]] const auto collider = physics.createCollider(handle, unitCollider());
    return handle;
}

// A wide static (or kinematic) floor whose top face sits at y = 0. Restitution 0 so
// bodies settle (and can sleep) rather than bouncing on the default elastic material.
PhysicsBodyHandle createFloor(PhysicsWorld& physics, PhysicsBodyType type = PhysicsBodyType::Static)
{
    PhysicsBodyDesc bodyDesc;
    bodyDesc.type = type;
    bodyDesc.material.restitution = 0.0f;
    auto handle = physics.createBody(bodyDesc);
    ColliderDesc desc;
    desc.shape = AabbShape{{Vec3{-10.0f, -1.0f, -10.0f}, Vec3{10.0f, 0.0f, 10.0f}}};
    [[maybe_unused]] const auto collider = physics.createCollider(handle, desc);
    return handle;
}

// A gravity-driven dynamic box (unit collider, origin at its min corner).
PhysicsBodyHandle createFallingBox(PhysicsWorld& physics, Vec3 position)
{
    PhysicsBodyDesc bodyDesc;
    bodyDesc.type = PhysicsBodyType::Dynamic;
    bodyDesc.position = position;
    bodyDesc.gravityScale = 1.0f;
    bodyDesc.material.restitution = 0.0f;
    auto handle = physics.createBody(bodyDesc);
    [[maybe_unused]] const auto collider = physics.createCollider(handle, unitCollider());
    return handle;
}

void stepMany(PhysicsWorld& physics, int steps)
{
    for (int i = 0; i < steps; ++i)
    {
        physics.step(1.0f / 60.0f);
    }
}

} // namespace

TEST_CASE("PhysicsWorld.CreatesBodiesAndCollidersWithStableHandles", "[PhysicsWorld]")
{
    PhysicsWorld physics;
    auto body = createBox(physics, PhysicsBodyType::Static, {});

    CHECK(body.valid());
    CHECK(physics.valid(body));
    CHECK(physics.bodyCount() == 1U);
    CHECK(physics.colliderCount() == 1U);
    CHECK(physics.validateBroadPhase());
}

TEST_CASE("PhysicsWorld.DestroyBodyInvalidatesBodyAndItsColliders", "[PhysicsWorld]")
{
    PhysicsWorld physics;

    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    const auto first = physics.createBody(desc);
    const auto firstCollider = physics.createCollider(first, unitCollider());

    desc.position = Vec3{5.0f, 0.0f, 0.0f};
    const auto second = physics.createBody(desc);
    const auto secondCollider = physics.createCollider(second, unitCollider());

    REQUIRE(physics.valid(first));
    REQUIRE(physics.valid(firstCollider));
    CHECK(physics.bodyCount() == 2U);
    CHECK(physics.colliderCount() == 2U);

    CHECK(physics.destroyBody(first));

    // The first body and its collider are gone; the second entry is untouched
    // even though backing storage remains stable for live solver indices.
    CHECK_FALSE(physics.valid(first));
    CHECK_FALSE(physics.valid(firstCollider));
    CHECK(physics.valid(second));
    CHECK(physics.valid(secondCollider));
    CHECK(physics.bodyCount() == 1U);
    CHECK(physics.colliderCount() == 1U);
    CHECK(physics.validateBroadPhase());

    // A fresh body created after a destroy still resolves through the
    // side-tables and gets a distinct handle.
    const auto third = physics.createBody(desc);
    CHECK(physics.valid(third));
    CHECK(third != first);
    CHECK(physics.bodyCount() == 2U);
}

TEST_CASE("PhysicsWorld.DestroyBodyRejectsUnknownAndAlreadyDestroyedHandles", "[PhysicsWorld]")
{
    PhysicsWorld physics;
    CHECK_FALSE(physics.destroyBody(PhysicsBodyHandle{}));

    const auto body = createBox(physics, PhysicsBodyType::Static, {});
    CHECK(physics.destroyBody(body));
    CHECK_FALSE(physics.destroyBody(body)); // already inactive
}

TEST_CASE("PhysicsWorld.DynamicBodyIntegratesVelocityOnStep", "[PhysicsWorld]")
{
    PhysicsWorld physics;
    auto body = createBox(physics, PhysicsBodyType::Dynamic, {}, {2.0f, 0.0f, -1.0f});

    physics.step(0.5f);

    auto transform = physics.bodyTransform(body);
    REQUIRE(transform.has_value());
    CHECK(transform->position() == Vec3(1.0f, 0.0f, -0.5f));
}

TEST_CASE("PhysicsWorld.DynamicBodySettlesAgainstStaticContact", "[PhysicsWorld]")
{
    // P2 sequential-impulse solver: a slow dynamic body driven into a static box has
    // its approaching velocity removed by the normal impulse (no bounce — the 1 m/s
    // approach is below the restitution threshold) and is pushed out of penetration by
    // the split-impulse position pass over several steps. A sphere (radius 0.5) is used
    // for the moving body so the head-on contact has no rotational degree of freedom to
    // drift along (a frictionless box-vs-box wall contact with no gravity can skid/spin);
    // it settles just touching the static box (centre x≈2.0) at rest, not tunnelling.
    PhysicsWorld physics;
    PhysicsBodyDesc dd;
    dd.type = PhysicsBodyType::Dynamic;
    dd.position = {1.8f, 0.0f, 0.0f};
    dd.linearVelocity = {1.0f, 0.0f, 0.0f};
    dd.gravityScale = 0.0f;
    auto dynamic = physics.createBody(dd);
    ColliderDesc sphere;
    sphere.shape = SphereShape{0.5f, Vec3{}};
    (void)physics.createCollider(dynamic, sphere);
    PhysicsBodyDesc sd;
    sd.type = PhysicsBodyType::Static;
    sd.position = {3.0f, 0.0f, 0.0f};
    auto staticBody = physics.createBody(sd);
    ColliderDesc staticColl;
    staticColl.shape = BoxShape{Vec3{0.5f, 0.5f, 0.5f}, Vec3{}};
    (void)physics.createCollider(staticBody, staticColl);

    for (int i = 0; i < 120; ++i)
    {
        physics.step(1.0f / 60.0f);
    }

    auto transform = physics.bodyTransform(dynamic);
    REQUIRE(transform.has_value());
    REQUIRE(physics.body(dynamic) != nullptr);
    CHECK(transform->position().approxEqual(Vec3(2.0f, 0.0f, 0.0f), 0.02f));
    CHECK(physics.body(dynamic)->linearVelocity().approxEqual(Vec3(0.0f, 0.0f, 0.0f), 0.05f));
}

TEST_CASE("PhysicsWorld.OffCenterColliderRotatesAboutCenterOfMass", "[PhysicsWorld]")
{
    // True COM offset (P6): a body whose collider is offset from the transform origin
    // spins about its centre of mass, not the origin. A free body (no gravity, no
    // contacts) given an angular velocity about a principal axis keeps its COM fixed
    // while the origin orbits around it.
    PhysicsWorld physics;
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.gravityScale = 0.0f;
    desc.angularVelocity = {0.0f, pi, 0.0f}; // π rad/s about y → 180° over one second
    const auto body = physics.createBody(desc);
    ColliderDesc collider;
    collider.shape = BoxShape{Vec3{0.5f, 0.5f, 0.5f}, Vec3{1.0f, 0.0f, 0.0f}}; // COM at +x
    (void)physics.createCollider(body, collider);

    auto worldCom = [&]
    {
        const auto t = physics.bodyTransform(body).value();
        return t.position() + t.rotation().rotate(physics.body(body)->centerOfMassLocal());
    };

    // The COM starts at world (1,0,0); the origin at (0,0,0).
    REQUIRE(worldCom().approxEqual(Vec3(1.0f, 0.0f, 0.0f), 1e-4f));
    REQUIRE(physics.bodyTransform(body)->position().approxEqual(Vec3(0.0f, 0.0f, 0.0f), 1e-4f));

    for (int i = 0; i < 120; ++i)
    {
        physics.step(1.0f / 120.0f);
    }

    // After ~180° about y: COM unmoved, the origin has swung to the far side (~(2,0,0)).
    CHECK(worldCom().approxEqual(Vec3(1.0f, 0.0f, 0.0f), 0.02f));
    CHECK(physics.bodyTransform(body)->position().approxEqual(Vec3(2.0f, 0.0f, 0.0f), 0.05f));
}

TEST_CASE("PhysicsWorld.KinematicBodyPushesOutOfStaticContact", "[PhysicsWorld]")
{
    // A kinematic body does not integrate velocity, but the split-impulse position
    // pass still resolves it out of penetration (positionWeight 1). Placed at x=2.7
    // it overlaps the static box (x=3) by 0.8 in x (the minimum-overlap axis), so it
    // is pushed back along -x to just touching (body x≈2.0) over several steps; the
    // perpendicular z=0.2 offset is preserved — the analogue of the old "slide".
    PhysicsWorld physics;
    auto kinematic = createBox(physics, PhysicsBodyType::Kinematic, {});
    createBox(physics, PhysicsBodyType::Static, {3.0f, 0.0f, 0.0f});

    auto transform = physics.bodyTransform(kinematic);
    REQUIRE(transform.has_value());
    transform->position({2.7f, 0.0f, 0.2f});
    physics.setBodyTransform(kinematic, transform.value());

    for (int i = 0; i < 120; ++i)
    {
        physics.step(1.0f / 60.0f);
    }

    auto resolved = physics.bodyTransform(kinematic);
    REQUIRE(resolved.has_value());
    CHECK(resolved->position().approxEqual(Vec3(2.0f, 0.0f, 0.2f), 0.02f));
}

TEST_CASE("PhysicsWorld.FrictionDeceleratesSlidingDynamicBox", "[PhysicsWorld]")
{
    // Coulomb friction goes live end-to-end (material → solver tangent rows): a
    // dynamic box sliding along a high-friction static floor under gravity loses
    // its horizontal velocity, while the same box on a frictionless floor keeps
    // sliding at its initial speed.
    auto finalSpeedX = [](float friction)
    {
        PhysicsWorld physics;

        PhysicsBodyDesc floor;
        floor.type = PhysicsBodyType::Static;
        floor.material = PhysicsMaterial{.restitution = 0.0f, .friction = friction};
        const auto floorBody = physics.createBody(floor);
        [[maybe_unused]] const auto floorCollider = physics.createCollider(
            floorBody, ColliderDesc{.shape = BoxShape{.halfExtents = {10.0f, 0.5f, 10.0f}}});

        PhysicsBodyDesc box;
        box.type = PhysicsBodyType::Dynamic;
        box.position = {0.0f, 1.0f, 0.0f}; // box bottom (0.5) rests on the floor top (0.5)
        box.linearVelocity = {3.0f, 0.0f, 0.0f};
        box.material = PhysicsMaterial{.restitution = 0.0f, .friction = friction};
        const auto boxBody = physics.createBody(box);
        [[maybe_unused]] const auto boxCollider = physics.createCollider(
            boxBody, ColliderDesc{.shape = BoxShape{.halfExtents = {0.5f, 0.5f, 0.5f}}});

        for (int i = 0; i < 120; ++i)
        {
            physics.step(1.0f / 60.0f);
        }
        return physics.body(boxBody)->linearVelocity().x();
    };

    CHECK(finalSpeedX(1.0f) == Catch::Approx(0.0f).margin(0.1f)); // friction brings it to rest
    CHECK(finalSpeedX(0.0f) > 2.5f);                              // frictionless keeps sliding
}

TEST_CASE("PhysicsWorld.StackOfBoxesSettlesAndStaysStill", "[PhysicsWorld]")
{
    // The headline win of the warm-started impulse solver: a stack of dynamic
    // boxes dropped onto a static floor settles into a resting stack and then
    // stays still (no jitter, no slow sink/drift) rather than buzzing apart.
    PhysicsWorld physics;

    PhysicsBodyDesc floor;
    floor.type = PhysicsBodyType::Static;
    floor.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
    const auto floorBody = physics.createBody(floor);
    [[maybe_unused]] const auto floorCollider = physics.createCollider(
        floorBody, ColliderDesc{.shape = BoxShape{.halfExtents = {10.0f, 0.5f, 10.0f}}});

    std::array<PhysicsBodyHandle, 3> boxes{};
    for (int i = 0; i < 3; ++i)
    {
        PhysicsBodyDesc box;
        box.type = PhysicsBodyType::Dynamic;
        box.gravityScale = 1.0f;
        box.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
        // Small gaps above the floor top (0.5): centres start near 1.05, 2.10, 3.15
        // so they fall a little and settle into contact.
        box.position = {0.0f, 1.05f + static_cast<float>(i) * 1.05f, 0.0f};
        boxes[static_cast<std::size_t>(i)] = physics.createBody(box);
        [[maybe_unused]] const auto collider = physics.createCollider(
            boxes[static_cast<std::size_t>(i)],
            ColliderDesc{.shape = BoxShape{.halfExtents = {0.5f, 0.5f, 0.5f}}});
    }

    const auto runSteps = [&](int n)
    {
        for (int i = 0; i < n; ++i)
        {
            physics.step(1.0f / 60.0f);
        }
    };

    runSteps(210);
    std::array<Vec3, 3> settled{};
    for (std::size_t k = 0; k < boxes.size(); ++k)
    {
        settled[k] = physics.bodyTransform(boxes[k])->position();
    }

    runSteps(30);
    for (std::size_t k = 0; k < boxes.size(); ++k)
    {
        const Vec3 now = physics.bodyTransform(boxes[k])->position();
        CHECK(now.approxEqual(settled[k], 0.01f)); // stable across the last 30 steps
        CHECK(physics.body(boxes[k])->linearVelocity().approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 0.05f));
    }

    // Resting stack: bottom box just above the floor (~1.0), each box above it.
    const float y0 = physics.bodyTransform(boxes[0])->position().y();
    const float y1 = physics.bodyTransform(boxes[1])->position().y();
    const float y2 = physics.bodyTransform(boxes[2])->position().y();
    CHECK(y0 == Catch::Approx(1.0f).margin(0.1f));
    CHECK(y1 > y0);
    CHECK(y2 > y1);
}

TEST_CASE("PhysicsWorld.BoxDroppedFlatRestsFlatAndStill", "[PhysicsWorld]")
{
    // A box dropped flat onto the floor has symmetric contacts → no net torque, so
    // it settles flat (orientation ≈ identity) and comes fully to rest. This guards
    // resting stability with the angular solver active.
    PhysicsWorld physics;

    PhysicsBodyDesc floor;
    floor.type = PhysicsBodyType::Static;
    floor.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
    const auto floorBody = physics.createBody(floor);
    [[maybe_unused]] const auto floorCollider = physics.createCollider(
        floorBody, ColliderDesc{.shape = BoxShape{.halfExtents = {10.0f, 0.5f, 10.0f}}});

    PhysicsBodyDesc box;
    box.type = PhysicsBodyType::Dynamic;
    box.gravityScale = 1.0f;
    box.position = {0.0f, 2.0f, 0.0f}; // bottom at 1.5, floor top 0.5 → falls 1.0
    box.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
    const auto boxBody = physics.createBody(box);
    [[maybe_unused]] const auto boxCollider = physics.createCollider(
        boxBody, ColliderDesc{.shape = BoxShape{.halfExtents = {0.5f, 0.5f, 0.5f}}});

    for (int i = 0; i < 240; ++i)
    {
        physics.step(1.0f / 60.0f);
    }

    const auto t = physics.bodyTransform(boxBody);
    REQUIRE(t.has_value());
    CHECK(t->position().y() == Catch::Approx(1.0f).margin(0.05f)); // resting on the floor
    CHECK(t->rotation().rotate(Vec3{0.0f, 1.0f, 0.0f}).approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 0.02f));
    CHECK(physics.body(boxBody)->angularVelocity().approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 0.05f));
    CHECK(physics.body(boxBody)->linearVelocity().approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 0.05f));
}

TEST_CASE("PhysicsWorld.ConvexHullCubeRestsFlatLikeABox", "[PhysicsWorld]")
{
    // A dynamic ConvexHullShape cube (GJK/EPA + face-clip narrowphase) dropped flat
    // onto the floor settles flat and still — the same outcome as the primitive box,
    // proving the convex path produces a stable multi-point resting manifold.
    PhysicsWorld physics;

    PhysicsBodyDesc floor;
    floor.type = PhysicsBodyType::Static;
    floor.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
    const auto floorBody = physics.createBody(floor);
    [[maybe_unused]] const auto floorCollider = physics.createCollider(
        floorBody, ColliderDesc{.shape = BoxShape{.halfExtents = {10.0f, 0.5f, 10.0f}}});

    PhysicsBodyDesc box;
    box.type = PhysicsBodyType::Dynamic;
    box.gravityScale = 1.0f;
    box.position = {0.0f, 2.0f, 0.0f};
    box.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
    const auto boxBody = physics.createBody(box);
    [[maybe_unused]] const auto boxCollider =
        physics.createCollider(boxBody, ColliderDesc{.shape = boxHull(0.5f)});

    for (int i = 0; i < 240; ++i)
    {
        physics.step(1.0f / 60.0f);
    }

    const auto t = physics.bodyTransform(boxBody);
    REQUIRE(t.has_value());
    CHECK(t->position().y() == Catch::Approx(1.0f).margin(0.05f));
    CHECK(t->rotation().rotate(Vec3{0.0f, 1.0f, 0.0f}).approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 0.03f));
    CHECK(physics.body(boxBody)->angularVelocity().approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 0.05f));
    CHECK(physics.body(boxBody)->linearVelocity().approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 0.05f));
}

TEST_CASE("PhysicsWorld.TallTiltedBoxTopplesOntoItsSide", "[PhysicsWorld]")
{
    // A tall box tilted past its balance angle topples onto its long side: its local
    // +y axis ends up horizontal, and it comes to rest. The headline P3 behaviour.
    PhysicsWorld physics;

    PhysicsBodyDesc floor;
    floor.type = PhysicsBodyType::Static;
    floor.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.6f};
    const auto floorBody = physics.createBody(floor);
    [[maybe_unused]] const auto floorCollider = physics.createCollider(
        floorBody, ColliderDesc{.shape = BoxShape{.halfExtents = {10.0f, 0.5f, 10.0f}}});

    // Tall box (half-extents 0.3 × 1.0 × 0.3): topple angle ≈ atan(0.3/1.0) ≈ 16.7°;
    // a 30° tilt is safely past it.
    const float angle = 30.0f * pi / 180.0f;
    PhysicsBodyDesc box;
    box.type = PhysicsBodyType::Dynamic;
    box.gravityScale = 1.0f;
    box.position = {0.0f, 2.0f, 0.0f};
    box.rotation = Quaternion{0.0f, 0.0f, std::sin(angle * 0.5f), std::cos(angle * 0.5f)};
    box.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.6f};
    const auto boxBody = physics.createBody(box);
    [[maybe_unused]] const auto boxCollider = physics.createCollider(
        boxBody, ColliderDesc{.shape = BoxShape{.halfExtents = {0.3f, 1.0f, 0.3f}}});

    for (int i = 0; i < 360; ++i)
    {
        physics.step(1.0f / 60.0f);
    }

    const auto t = physics.bodyTransform(boxBody);
    REQUIRE(t.has_value());
    const Vec3 up = t->rotation().rotate(Vec3{0.0f, 1.0f, 0.0f});
    CHECK(std::abs(up.y()) < 0.2f); // local +y is now ~horizontal → toppled onto its side
    CHECK(physics.body(boxBody)->angularVelocity().approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 0.1f));
}

TEST_CASE("PhysicsWorld.FreeBodyConservesAngularVelocityAndIntegratesOrientation", "[PhysicsWorld]")
{
    // A dynamic body with no contacts and a seeded angular velocity keeps that
    // angular velocity (no torque) and its orientation advances accordingly —
    // π/2 rad/s about Y for 1 s is a 90° turn, which maps +x → -z.
    PhysicsWorld physics;
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.gravityScale = 0.0f;
    desc.angularVelocity = {0.0f, pi * 0.5f, 0.0f};
    const auto body = physics.createBody(desc);

    for (int i = 0; i < 60; ++i)
    {
        physics.step(1.0f / 60.0f);
    }

    REQUIRE(physics.body(body) != nullptr);
    CHECK(physics.body(body)->angularVelocity().approxEqual(Vec3{0.0f, pi * 0.5f, 0.0f}, 1e-5f));
    const Quaternion r = physics.bodyTransform(body)->rotation();
    CHECK(r.rotate(Vec3{1.0f, 0.0f, 0.0f}).approxEqual(Vec3{0.0f, 0.0f, -1.0f}, 1e-3f));
}

TEST_CASE("PhysicsWorld.FastBulletDoesNotTunnelThroughThinWall", "[PhysicsWorld]")
{
    // Speculative-margin CCD: a fast 'bullet' aimed at a thin static wall would
    // tunnel under discrete-only collision — at 300 m/s it moves 5 m/step, so its
    // end-of-step pose is well past the wall. The motion-scaled speculative margin
    // generates a gap contact and the solver brakes the closing velocity to exactly
    // reach the wall, so the bullet ends resting against the near face, not beyond.
    PhysicsWorld physics;

    PhysicsBodyDesc wall;
    wall.type = PhysicsBodyType::Static;
    wall.position = {2.5f, 0.0f, 0.0f};
    wall.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.0f};
    const auto wallBody = physics.createBody(wall);
    [[maybe_unused]] const auto wallCollider = physics.createCollider(
        wallBody, ColliderDesc{.shape = BoxShape{.halfExtents = {0.1f, 2.0f, 2.0f}}});

    PhysicsBodyDesc bullet;
    bullet.type = PhysicsBodyType::Dynamic;
    bullet.gravityScale = 0.0f;
    bullet.linearVelocity = {300.0f, 0.0f, 0.0f};
    bullet.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.0f};
    const auto bulletBody = physics.createBody(bullet);
    [[maybe_unused]] const auto bulletCollider = physics.createCollider(
        bulletBody, ColliderDesc{.shape = BoxShape{.halfExtents = {0.1f, 0.1f, 0.1f}}});

    for (int i = 0; i < 30; ++i)
    {
        physics.step(1.0f / 60.0f);
    }

    const Vec3 p = physics.bodyTransform(bulletBody)->position();
    // Wall near face at x=2.4, bullet half-width 0.1 → rests at body x≈2.3.
    CHECK(p.x() < 2.4f);                              // never crossed to the far side
    CHECK(p.x() == Catch::Approx(2.3f).margin(0.1f)); // settled against the near face
    CHECK(physics.body(bulletBody)->linearVelocity().approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 0.1f));
}

TEST_CASE("SceneGraphPhysicsSync.KinematicNodesPushAndDynamicNodesPullTransforms",
          "[SceneGraphPhysicsSync]")
{
    PhysicsWorld physics;
    SceneGraph scene;

    auto dynamicNode = std::make_unique<Node>("Dynamic");
    auto dynamicHandle =
        createBox(physics, PhysicsBodyType::Dynamic, {10.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f});
    dynamicNode->physicsBodyHandle(dynamicHandle);
    Node* dynamic = &scene.addNode(std::move(dynamicNode));

    auto kinematicNode = std::make_unique<Node>("Kinematic");
    auto kinematicHandle = createBox(physics, PhysicsBodyType::Kinematic, {});
    kinematicNode->physicsBodyHandle(kinematicHandle);
    kinematicNode->transform().position({5.0f, 0.0f, 0.0f});
    Node* kinematic = &scene.addNode(std::move(kinematicNode));

    REQUIRE(dynamic->hasPhysicsBodyHandle());
    REQUIRE(kinematic->hasPhysicsBodyHandle());
    REQUIRE(physics.body(kinematicHandle) != nullptr);

    scene.update(InputState{});
    REQUIRE(kinematic->transform().position() == Vec3(5.0f, 0.0f, 0.0f));
    scene.submitPhysics(physics);
    auto syncedKinematicTransform = physics.bodyTransform(kinematicHandle);
    REQUIRE(syncedKinematicTransform.has_value());
    REQUIRE(syncedKinematicTransform->position() == Vec3(5.0f, 0.0f, 0.0f));
    physics.step(0.5f);
    scene.applyPhysics(physics);

    auto kinematicTransform = physics.bodyTransform(kinematicHandle);
    REQUIRE(kinematicTransform.has_value());
    CHECK(kinematicTransform->position() == Vec3(5.0f, 0.0f, 0.0f));
    CHECK(dynamic->transform().position() == Vec3(11.0f, 0.0f, 0.0f));
}

TEST_CASE("SceneGraphPhysicsSync.KinematicResolutionPullsBackToNode", "[SceneGraphPhysicsSync]")
{
    PhysicsWorld physics;
    SceneGraph scene;

    auto paddleNode = std::make_unique<Node>("Paddle");
    auto paddleHandle = createBox(physics, PhysicsBodyType::Kinematic, {});
    paddleNode->physicsBodyHandle(paddleHandle);
    Node* paddle = &scene.addNode(std::move(paddleNode));

    createBox(physics, PhysicsBodyType::Static, {3.0f, 0.0f, 0.0f});

    paddle->transform().position({2.7f, 0.0f, 0.2f});

    // Drive the full scene sync each step: submitPhysics pushes the node target in,
    // the solver corrects penetration, applyPhysics pulls the corrected transform
    // back onto the node. Over several frames the paddle settles out of the static
    // box at x≈2.0 with its perpendicular z=0.2 preserved.
    for (int i = 0; i < 120; ++i)
    {
        scene.update(InputState{});
        scene.submitPhysics(physics);
        physics.step(1.0f / 60.0f);
        scene.applyPhysics(physics);
    }

    CHECK(paddle->transform().position().approxEqual(Vec3(2.0f, 0.0f, 0.2f), 0.02f));
}

TEST_CASE("PhysicsWorld.GatherCollidersComposesWorldSphere", "[PhysicsWorld]")
{
    PhysicsWorld physics;
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    desc.position = Vec3(2.0f, 3.0f, 4.0f);
    desc.scale = Vec3(2.0f, 2.0f, 2.0f);
    const auto body = physics.createBody(desc);
    [[maybe_unused]] const auto collider =
        physics.createCollider(body, ColliderDesc{.shape = SphereShape{.radius = 0.5f}});

    const auto colliders = physics.gatherColliders();
    REQUIRE(colliders.size() == 1u);
    const auto& c = colliders.front();
    CHECK(c.type == std::to_underlying(ClothColliderType::Sphere));
    // a = (world center.xyz, world radius); body scale 2 → radius 0.5 * 2 = 1.0.
    CHECK(c.a[0] == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(c.a[1] == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(c.a[2] == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(c.a[3] == Catch::Approx(1.0f).margin(1e-5f));
}

// ==========================================================================
// Sleeping (P5)
// ==========================================================================

TEST_CASE("PhysicsWorld.StationaryDynamicBodySleepsAfterTimeout", "[PhysicsWorld][Sleep]")
{
    // A lone dynamic body at rest (no gravity, no contacts) is below the sleep
    // thresholds every step, so its singleton island sleeps after kSleepTime (0.5s).
    PhysicsWorld physics;
    const auto body = createBox(physics, PhysicsBodyType::Dynamic, {});

    physics.step(1.0f / 60.0f);
    CHECK_FALSE(physics.sleeping(body)); // not yet — timer still below kSleepTime

    stepMany(physics, 40); // > 0.5s total
    CHECK(physics.sleeping(body));
}

TEST_CASE("PhysicsWorld.WakeAndSetVelocityResumeSimulation", "[PhysicsWorld][Sleep]")
{
    PhysicsWorld physics;
    const auto body = createBox(physics, PhysicsBodyType::Dynamic, {});
    stepMany(physics, 40);
    REQUIRE(physics.sleeping(body));

    physics.wake(body);
    CHECK_FALSE(physics.sleeping(body));

    // Re-sleeps, then an externally set velocity wakes it and it moves.
    stepMany(physics, 40);
    REQUIRE(physics.sleeping(body));
    physics.setBodyVelocity(body, {1.0f, 0.0f, 0.0f});
    CHECK_FALSE(physics.sleeping(body));

    const float beforeX = physics.bodyTransform(body)->position().x();
    physics.step(1.0f / 60.0f);
    CHECK(physics.bodyTransform(body)->position().x() > beforeX);
}

TEST_CASE("PhysicsWorld.AllowSleepingFalseNeverSleeps", "[PhysicsWorld][Sleep]")
{
    PhysicsWorld physics;
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.gravityScale = 0.0f;
    desc.allowSleeping = false;
    const auto body = physics.createBody(desc);
    [[maybe_unused]] const auto collider = physics.createCollider(body, unitCollider());

    stepMany(physics, 120);
    CHECK_FALSE(physics.sleeping(body));
}

TEST_CASE("PhysicsWorld.SleepingDisabledKeepsBodiesAwake", "[PhysicsWorld][Sleep]")
{
    PhysicsWorld physics;
    physics.sleepingEnabled(false);
    const auto body = createBox(physics, PhysicsBodyType::Dynamic, {});

    stepMany(physics, 120);
    CHECK_FALSE(physics.sleeping(body));
}

TEST_CASE("PhysicsWorld.BoxSettlesOnFloorAndSleeps", "[PhysicsWorld][Sleep]")
{
    // A gravity box dropped onto a static floor settles to rest and then sleeps.
    PhysicsWorld physics;
    createFloor(physics);
    const auto box = createFallingBox(physics, {0.0f, 2.0f, 0.0f});

    stepMany(physics, 240); // fall + settle + kSleepTime
    CHECK(physics.sleeping(box));
    // Resting on the floor top (y = 0); unit collider origin is its min corner.
    CHECK(physics.bodyTransform(box)->position().y() == Catch::Approx(0.0f).margin(0.02f));
}

TEST_CASE("PhysicsWorld.SleepingBodyIsWokenByAnIncomingBody", "[PhysicsWorld][Sleep]")
{
    // One body sleeps; another slides into it. The new contact merges them into one
    // island whose moving member keeps it awake, so the sleeper wakes and is pushed.
    PhysicsWorld physics;
    const auto sleeper = createBox(physics, PhysicsBodyType::Dynamic, {0.0f, 0.0f, 0.0f});
    const auto mover =
        createBox(physics, PhysicsBodyType::Dynamic, {-4.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f});

    // Let the sleeper settle while the mover is still far away.
    stepMany(physics, 40);
    REQUIRE(physics.sleeping(sleeper));
    REQUIRE_FALSE(physics.sleeping(mover));

    // Run until the mover reaches and pushes the sleeper.
    stepMany(physics, 120);
    CHECK_FALSE(physics.sleeping(sleeper));
    CHECK(physics.bodyTransform(sleeper)->position().x() > 0.01f);
}

TEST_CASE("PhysicsWorld.MovingKinematicFloorKeepsRiderAwake", "[PhysicsWorld][Sleep]")
{
    // A box resting on a kinematic floor that keeps moving must not sleep, while the
    // same box on a parked kinematic floor does sleep.
    auto riderSleeps = [](bool moveFloor)
    {
        PhysicsWorld physics;
        const auto floor = createFloor(physics, PhysicsBodyType::Kinematic);
        const auto box = createFallingBox(physics, {0.0f, 1.0f, 0.0f});

        for (int i = 0; i < 300; ++i)
        {
            if (moveFloor)
            {
                // Nudge the kinematic floor sideways each step (scene-driven motion).
                auto t = physics.bodyTransform(floor).value();
                t.position(t.position() + Vec3{0.001f, 0.0f, 0.0f});
                physics.setBodyTransform(floor, t);
            }
            physics.step(1.0f / 60.0f);
        }
        return physics.sleeping(box);
    };

    CHECK_FALSE(riderSleeps(true)); // moving floor → rider stays awake
    CHECK(riderSleeps(false));      // parked floor → rider sleeps
}
