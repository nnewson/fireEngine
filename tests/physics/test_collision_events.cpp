#include <fire_engine/physics/physics_world.hpp>

#include <catch2/catch_test_macros.hpp>

using fire_engine::BoxShape;
using fire_engine::ColliderDesc;
using fire_engine::ContactEvent;
using fire_engine::EventPhase;
using fire_engine::Mat4;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsColliderHandle;
using fire_engine::PhysicsMaterial;
using fire_engine::PhysicsWorld;
using fire_engine::SphereShape;
using fire_engine::Transform;
using fire_engine::Vec3;

namespace
{

constexpr float kDt = 1.0f / 60.0f;

bool hasEvent(const std::vector<ContactEvent>& events, PhysicsColliderHandle a,
              PhysicsColliderHandle b, EventPhase phase)
{
    for (const ContactEvent& e : events)
    {
        const bool pair = (e.first == a && e.second == b) || (e.first == b && e.second == a);
        if (pair && e.phase == phase)
        {
            return true;
        }
    }
    return false;
}

void moveTo(PhysicsWorld& world, PhysicsBodyHandle body, Vec3 position)
{
    Transform t;
    t.position(position);
    t.update(Mat4::identity());
    world.setBodyTransform(body, t);
}

} // namespace

TEST_CASE("CollisionEvents.TriggerEnterStayExit", "[CollisionEvents]")
{
    PhysicsWorld world;

    PhysicsBodyDesc zoneDesc;
    zoneDesc.type = PhysicsBodyType::Static;
    const PhysicsBodyHandle zoneBody = world.createBody(zoneDesc);
    ColliderDesc zoneCol;
    zoneCol.shape = BoxShape{Vec3{1.0f, 1.0f, 1.0f}, Vec3{}};
    zoneCol.isTrigger = true;
    const PhysicsColliderHandle trigger = world.createCollider(zoneBody, zoneCol);

    PhysicsBodyDesc moverDesc;
    moverDesc.type = PhysicsBodyType::Kinematic;
    moverDesc.position = {5.0f, 0.0f, 0.0f};
    const PhysicsBodyHandle moverBody = world.createBody(moverDesc);
    const PhysicsColliderHandle mover =
        world.createCollider(moverBody, ColliderDesc{.shape = SphereShape{0.3f, Vec3{}}});

    // Outside → no events.
    world.step(kDt);
    CHECK(world.triggerEvents().empty());

    // Move inside → Enter.
    moveTo(world, moverBody, {0.0f, 0.0f, 0.0f});
    world.step(kDt);
    CHECK(hasEvent(world.triggerEvents(), trigger, mover, EventPhase::Enter));

    // Still inside → Stay.
    world.step(kDt);
    CHECK(hasEvent(world.triggerEvents(), trigger, mover, EventPhase::Stay));

    // Move out → Exit.
    moveTo(world, moverBody, {5.0f, 0.0f, 0.0f});
    world.step(kDt);
    CHECK(hasEvent(world.triggerEvents(), trigger, mover, EventPhase::Exit));

    // Gone → no further events.
    world.step(kDt);
    CHECK(world.triggerEvents().empty());
}

TEST_CASE("CollisionEvents.DestroyedColliderDoesNotEmitStaleTriggerExit", "[CollisionEvents]")
{
    PhysicsWorld world;

    PhysicsBodyDesc zoneDesc;
    zoneDesc.type = PhysicsBodyType::Static;
    const PhysicsBodyHandle zoneBody = world.createBody(zoneDesc);
    ColliderDesc zoneCol;
    zoneCol.shape = BoxShape{Vec3{1.0f, 1.0f, 1.0f}, Vec3{}};
    zoneCol.isTrigger = true;
    const PhysicsColliderHandle trigger = world.createCollider(zoneBody, zoneCol);

    PhysicsBodyDesc moverDesc;
    moverDesc.type = PhysicsBodyType::Kinematic;
    const PhysicsBodyHandle moverBody = world.createBody(moverDesc);
    const PhysicsColliderHandle mover =
        world.createCollider(moverBody, ColliderDesc{.shape = SphereShape{0.3f, Vec3{}}});

    world.step(kDt);
    REQUIRE(hasEvent(world.triggerEvents(), trigger, mover, EventPhase::Enter));

    CHECK(world.destroyBody(moverBody));
    CHECK_FALSE(world.valid(mover));

    world.step(kDt);
    CHECK_FALSE(hasEvent(world.triggerEvents(), trigger, mover, EventPhase::Exit));
}

TEST_CASE("CollisionEvents.TriggerHasNoSolverResponse", "[CollisionEvents]")
{
    PhysicsWorld world;

    // A trigger volume at the origin (half-extent 1) — a dynamic body must fall through.
    PhysicsBodyDesc zoneDesc;
    zoneDesc.type = PhysicsBodyType::Static;
    const PhysicsBodyHandle zoneBody = world.createBody(zoneDesc);
    (void)world.createCollider(
        zoneBody,
        ColliderDesc{.shape = BoxShape{Vec3{1.0f, 1.0f, 1.0f}, Vec3{}}, .isTrigger = true});

    PhysicsBodyDesc ballDesc;
    ballDesc.type = PhysicsBodyType::Dynamic;
    ballDesc.position = {0.0f, 3.0f, 0.0f};
    ballDesc.gravityScale = 1.0f;
    ballDesc.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.0f};
    const PhysicsBodyHandle ball = world.createBody(ballDesc);
    (void)world.createCollider(ball, ColliderDesc{.shape = SphereShape{0.3f, Vec3{}}});

    bool sawEnter = false;
    for (int i = 0; i < 120; ++i)
    {
        world.step(kDt);
        if (!world.triggerEvents().empty())
        {
            sawEnter = sawEnter || world.triggerEvents().front().phase == EventPhase::Enter;
        }
    }

    // It passed clean through (well below the zone), and the trigger fired on the way.
    CHECK(world.bodyTransform(ball)->position().y() < -2.0f);
    CHECK(sawEnter);
}

TEST_CASE("CollisionEvents.SolidContactEmitsCollisionEvent", "[CollisionEvents]")
{
    PhysicsWorld world;

    PhysicsBodyDesc floorDesc;
    floorDesc.type = PhysicsBodyType::Static;
    const PhysicsBodyHandle floorBody = world.createBody(floorDesc);
    const PhysicsColliderHandle floor = world.createCollider(
        floorBody, ColliderDesc{.shape = BoxShape{Vec3{10.0f, 0.5f, 10.0f}, Vec3{}}});

    PhysicsBodyDesc ballDesc;
    ballDesc.type = PhysicsBodyType::Dynamic;
    ballDesc.position = {0.0f, 2.0f, 0.0f};
    ballDesc.gravityScale = 1.0f;
    ballDesc.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.0f};
    const PhysicsBodyHandle ball = world.createBody(ballDesc);
    const PhysicsColliderHandle ballCol =
        world.createCollider(ball, ColliderDesc{.shape = SphereShape{0.5f, Vec3{}}});

    bool sawContact = false;
    for (int i = 0; i < 120; ++i)
    {
        world.step(kDt);
        if (hasEvent(world.collisionEvents(), floor, ballCol, EventPhase::Enter) ||
            hasEvent(world.collisionEvents(), floor, ballCol, EventPhase::Stay))
        {
            sawContact = true;
        }
    }
    CHECK(sawContact);
    // The ball rests on the floor surface (no fall-through), confirming a solid response.
    CHECK(world.bodyTransform(ball)->position().y() > 0.4f);
}
