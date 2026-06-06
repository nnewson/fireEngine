#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/scene_graph.hpp>

#include <gtest/gtest.h>

using fire_engine::AabbShape;
using fire_engine::ColliderDesc;
using fire_engine::InputState;
using fire_engine::Node;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsColliderHandle;
using fire_engine::PhysicsWorld;
using fire_engine::SceneGraph;
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

TEST(PhysicsWorld, CreatesBodiesAndCollidersWithStableHandles)
{
    PhysicsWorld physics;
    auto body = createBox(physics, PhysicsBodyType::Static, {});

    EXPECT_TRUE(body.valid());
    EXPECT_TRUE(physics.valid(body));
    EXPECT_EQ(physics.bodyCount(), 1U);
    EXPECT_EQ(physics.colliderCount(), 1U);
    EXPECT_TRUE(physics.validateBroadPhase());
}

TEST(PhysicsWorld, DestroyBodyTombstonesBodyAndItsColliders)
{
    PhysicsWorld physics;

    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    const auto first = physics.createBody(desc);
    const auto firstCollider = physics.createCollider(first, unitCollider());

    desc.position = Vec3{5.0f, 0.0f, 0.0f};
    const auto second = physics.createBody(desc);
    const auto secondCollider = physics.createCollider(second, unitCollider());

    ASSERT_TRUE(physics.valid(first));
    ASSERT_TRUE(physics.valid(firstCollider));
    EXPECT_EQ(physics.bodyCount(), 2U);
    EXPECT_EQ(physics.colliderCount(), 2U);

    EXPECT_TRUE(physics.destroyBody(first));

    // The first body and its collider are gone; the second entry is untouched
    // even though both share the same backing containers and side-tables.
    EXPECT_FALSE(physics.valid(first));
    EXPECT_FALSE(physics.valid(firstCollider));
    EXPECT_TRUE(physics.valid(second));
    EXPECT_TRUE(physics.valid(secondCollider));
    EXPECT_EQ(physics.bodyCount(), 1U);
    EXPECT_EQ(physics.colliderCount(), 1U);
    EXPECT_TRUE(physics.validateBroadPhase());

    // A fresh body created after a destroy still resolves through the
    // side-tables and gets a distinct handle.
    const auto third = physics.createBody(desc);
    EXPECT_TRUE(physics.valid(third));
    EXPECT_NE(third, first);
    EXPECT_EQ(physics.bodyCount(), 2U);
}

TEST(PhysicsWorld, DestroyBodyRejectsUnknownAndAlreadyDestroyedHandles)
{
    PhysicsWorld physics;
    EXPECT_FALSE(physics.destroyBody(PhysicsBodyHandle{}));

    const auto body = createBox(physics, PhysicsBodyType::Static, {});
    EXPECT_TRUE(physics.destroyBody(body));
    EXPECT_FALSE(physics.destroyBody(body)); // already tombstoned
}

TEST(PhysicsWorld, DynamicBodyIntegratesVelocityOnStep)
{
    PhysicsWorld physics;
    auto body = createBox(physics, PhysicsBodyType::Dynamic, {}, {2.0f, 0.0f, -1.0f});

    physics.step(0.5f);

    auto transform = physics.bodyTransform(body);
    ASSERT_TRUE(transform.has_value());
    EXPECT_EQ(transform->position(), Vec3(1.0f, 0.0f, -0.5f));
}

TEST(PhysicsWorld, DynamicBodyReflectsAndStopsAtStaticContact)
{
    PhysicsWorld physics;
    auto dynamic = createBox(physics, PhysicsBodyType::Dynamic, {}, {4.0f, 0.0f, 0.0f});
    createBox(physics, PhysicsBodyType::Static, {3.0f, 0.0f, 0.0f});

    physics.step(1.0f);

    auto transform = physics.bodyTransform(dynamic);
    ASSERT_TRUE(transform.has_value());
    ASSERT_NE(physics.body(dynamic), nullptr);
    EXPECT_EQ(transform->position(), Vec3(2.0f, 0.0f, 0.0f));
    EXPECT_EQ(physics.body(dynamic)->linearVelocity(), Vec3(-4.0f, 0.0f, 0.0f));
}

TEST(PhysicsWorld, KinematicBodySlidesAtStaticContact)
{
    PhysicsWorld physics;
    auto kinematic = createBox(physics, PhysicsBodyType::Kinematic, {});
    createBox(physics, PhysicsBodyType::Static, {3.0f, 0.0f, 0.0f});

    auto transform = physics.bodyTransform(kinematic);
    ASSERT_TRUE(transform.has_value());
    transform->position({4.0f, 1.0f, 0.0f});
    physics.setBodyTransform(kinematic, transform.value());

    physics.step(1.0f);

    auto resolved = physics.bodyTransform(kinematic);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->position(), Vec3(2.0f, 1.0f, 0.0f));
}

TEST(SceneGraphPhysicsSync, KinematicNodesPushAndDynamicNodesPullTransforms)
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

    ASSERT_TRUE(dynamic->hasPhysicsBodyHandle());
    ASSERT_TRUE(kinematic->hasPhysicsBodyHandle());
    ASSERT_NE(physics.body(kinematicHandle), nullptr);

    scene.update(InputState{});
    ASSERT_EQ(kinematic->transform().position(), Vec3(5.0f, 0.0f, 0.0f));
    scene.submitPhysics(physics);
    auto syncedKinematicTransform = physics.bodyTransform(kinematicHandle);
    ASSERT_TRUE(syncedKinematicTransform.has_value());
    ASSERT_EQ(syncedKinematicTransform->position(), Vec3(5.0f, 0.0f, 0.0f));
    physics.step(0.5f);
    scene.applyPhysics(physics);

    auto kinematicTransform = physics.bodyTransform(kinematicHandle);
    ASSERT_TRUE(kinematicTransform.has_value());
    EXPECT_EQ(kinematicTransform->position(), Vec3(5.0f, 0.0f, 0.0f));
    EXPECT_EQ(dynamic->transform().position(), Vec3(11.0f, 0.0f, 0.0f));
}

TEST(SceneGraphPhysicsSync, KinematicResolutionPullsBackToNode)
{
    PhysicsWorld physics;
    SceneGraph scene;

    auto paddleNode = std::make_unique<Node>("Paddle");
    auto paddleHandle = createBox(physics, PhysicsBodyType::Kinematic, {});
    paddleNode->physicsBodyHandle(paddleHandle);
    Node* paddle = &scene.addNode(std::move(paddleNode));

    createBox(physics, PhysicsBodyType::Static, {3.0f, 0.0f, 0.0f});

    paddle->transform().position({4.0f, 1.0f, 0.0f});
    scene.update(InputState{});
    scene.submitPhysics(physics);
    physics.step(1.0f);
    scene.applyPhysics(physics);

    EXPECT_EQ(paddle->transform().position(), Vec3(2.0f, 1.0f, 0.0f));
}
