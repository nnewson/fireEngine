#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/scene_graph.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::AabbShape;
using fire_engine::ClothColliderType;
using fire_engine::ColliderDesc;
using fire_engine::InputState;
using fire_engine::Node;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsColliderHandle;
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

TEST_CASE("PhysicsWorld.DynamicBodyReflectsAndPushesOutOfStaticContact", "[PhysicsWorld]")
{
    // P1 discrete response: the body integrates, the shape-specific narrowphase
    // reports the penetration, then it is pushed out along the contact normal and
    // its velocity is reflected (default restitution 1). Unit colliders span the
    // body origin + (0.5,0.5,0.5); the dynamic box at x=1.8 moving +x integrates to
    // 2.8 and overlaps the static box at x=3 by 0.8, so it is pushed back to exactly
    // touching (body x=2.0) and the +x velocity flips to -x. (Fast movers that would
    // tunnel in one step are a deferred CCD/speculative-contact concern.)
    PhysicsWorld physics;
    auto dynamic =
        createBox(physics, PhysicsBodyType::Dynamic, {1.8f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
    createBox(physics, PhysicsBodyType::Static, {3.0f, 0.0f, 0.0f});

    physics.step(1.0f);

    auto transform = physics.bodyTransform(dynamic);
    REQUIRE(transform.has_value());
    REQUIRE(physics.body(dynamic) != nullptr);
    CHECK(transform->position().approxEqual(Vec3(2.0f, 0.0f, 0.0f), 1e-5f));
    CHECK(physics.body(dynamic)->linearVelocity().approxEqual(Vec3(-1.0f, 0.0f, 0.0f), 1e-5f));
}

TEST_CASE("PhysicsWorld.KinematicBodyPushesOutOfStaticContact", "[PhysicsWorld]")
{
    // A kinematic body does not integrate velocity, but it still resolves out of
    // any penetration along the minimum-overlap axis (pushWeight 1). Placed at
    // x=2.7 it overlaps the static box (x=3) by 0.8 in x while only touching in the
    // perpendicular axes, so it is pushed back to body x=2.0; the perpendicular
    // z=0.2 offset is preserved — the discrete analogue of the old "slide".
    PhysicsWorld physics;
    auto kinematic = createBox(physics, PhysicsBodyType::Kinematic, {});
    createBox(physics, PhysicsBodyType::Static, {3.0f, 0.0f, 0.0f});

    auto transform = physics.bodyTransform(kinematic);
    REQUIRE(transform.has_value());
    transform->position({2.7f, 0.0f, 0.2f});
    physics.setBodyTransform(kinematic, transform.value());

    physics.step(1.0f);

    auto resolved = physics.bodyTransform(kinematic);
    REQUIRE(resolved.has_value());
    CHECK(resolved->position().approxEqual(Vec3(2.0f, 0.0f, 0.2f), 1e-5f));
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
    scene.update(InputState{});
    scene.submitPhysics(physics);
    physics.step(1.0f);
    scene.applyPhysics(physics);

    // Kinematic penetration push-out pulls back onto the node (see
    // KinematicBodyPushesOutOfStaticContact): x corrected to 2.0, z preserved.
    CHECK(paddle->transform().position().approxEqual(Vec3(2.0f, 0.0f, 0.2f), 1e-5f));
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
