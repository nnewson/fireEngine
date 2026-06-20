#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/scene_graph.hpp>

#include <array>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::AabbShape;
using fire_engine::BoxShape;
using fire_engine::ClothColliderType;
using fire_engine::ColliderDesc;
using fire_engine::InputState;
using fire_engine::Node;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsColliderHandle;
using fire_engine::PhysicsMaterial;
using fire_engine::PhysicsWorld;
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

TEST_CASE("PhysicsWorld.DestroyBodyTombstonesBodyAndItsColliders", "[PhysicsWorld]")
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
    // even though both share the same backing containers and side-tables.
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
    CHECK_FALSE(physics.destroyBody(body)); // already tombstoned
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
    // P2 sequential-impulse solver: a slow dynamic box driven into a static box has
    // its approaching velocity removed by the normal impulse (no bounce — the 1 m/s
    // approach is below the restitution threshold) and is pushed out of penetration
    // by the split-impulse position pass over several steps. Unit colliders span
    // the body origin + (0.5,0.5,0.5), so it settles just touching the static box
    // (body x≈2.0) at rest, rather than tunnelling, jittering, or bouncing.
    PhysicsWorld physics;
    auto dynamic =
        createBox(physics, PhysicsBodyType::Dynamic, {1.8f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
    createBox(physics, PhysicsBodyType::Static, {3.0f, 0.0f, 0.0f});

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
    CHECK(c.type == static_cast<int>(ClothColliderType::Sphere));
    // a = (world center.xyz, world radius); body scale 2 → radius 0.5 * 2 = 1.0.
    CHECK(c.a[0] == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(c.a[1] == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(c.a[2] == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(c.a[3] == Catch::Approx(1.0f).margin(1e-5f));
}
