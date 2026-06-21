#include <fire_engine/physics/physics_world.hpp>

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::BoxShape;
using fire_engine::ColliderDesc;
using fire_engine::CompoundChild;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsWorld;
using fire_engine::Quaternion;
using fire_engine::SphereShape;
using fire_engine::Vec3;

namespace
{

CompoundChild boxChild(Vec3 halfExtents, Vec3 position)
{
    CompoundChild child;
    child.shape = BoxShape{halfExtents, Vec3{}};
    child.localPosition = position;
    return child;
}

// A wide static floor whose top sits at y = 0 (restitution 0 so bodies settle).
PhysicsBodyHandle createFloor(PhysicsWorld& physics)
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    desc.material.restitution = 0.0f;
    const auto handle = physics.createBody(desc);
    ColliderDesc collider;
    collider.shape =
        fire_engine::AabbShape{{Vec3{-10.0f, -1.0f, -10.0f}, Vec3{10.0f, 0.0f, 10.0f}}};
    (void)physics.createCollider(handle, collider);
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

TEST_CASE("Compound.AggregatesMassProperties", "[Compound]")
{
    // Two unit cubes (half-extent 0.5) at ±x: symmetric, so the COM is the origin and
    // the inertia tensor is diagonal (exact). With total mass 6 the analytic principal
    // moments are Ixx = 1, Iyy = Izz = 7 (own m/6 each + parallel-axis m·d² off-axis).
    PhysicsWorld physics;
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.mass = 6.0f;
    const auto body = physics.createBody(desc);

    const std::vector<CompoundChild> children{
        boxChild({0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}),
        boxChild({0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}),
    };
    const auto handle = physics.createCompoundCollider(body, children);

    CHECK(handle.valid());
    CHECK(physics.colliderCount() == 2U);

    const auto* b = physics.body(body);
    REQUIRE(b != nullptr);
    CHECK(b->centerOfMassLocal().approxEqual(Vec3{}, 1e-4f));
    const Vec3 invI = b->inverseInertiaLocal();
    CHECK(invI.x() == Catch::Approx(1.0f).margin(0.01f));         // 1 / Ixx
    CHECK(invI.y() == Catch::Approx(1.0f / 7.0f).margin(0.005f)); // 1 / Iyy
    CHECK(invI.z() == Catch::Approx(1.0f / 7.0f).margin(0.005f)); // 1 / Izz
}

TEST_CASE("Compound.OffsetComShiftsCenterOfMass", "[Compound]")
{
    // Two cubes of different size: the COM shifts toward the larger one (more volume →
    // more mass). Big cube (half-extent 1) at -x, small (half-extent 0.5) at +x.
    PhysicsWorld physics;
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    const auto body = physics.createBody(desc);
    const std::vector<CompoundChild> children{
        boxChild({1.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}),
        boxChild({0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}),
    };
    (void)physics.createCompoundCollider(body, children);

    // Volumes 8 and 1 → masses 8/9 and 1/9 → COM x = (8/9·-1 + 1/9·1) = -7/9 ≈ -0.778.
    CHECK(physics.body(body)->centerOfMassLocal().x() == Catch::Approx(-7.0f / 9.0f).margin(0.01f));
}

TEST_CASE("Compound.RestsStablyOnFloor", "[Compound]")
{
    // A wide "dumbbell" compound dropped flat settles upright (COM centred over the
    // base) and sleeps — the two children both contact the floor.
    PhysicsWorld physics;
    createFloor(physics);

    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = {0.0f, 2.0f, 0.0f};
    desc.gravityScale = 1.0f;
    desc.mass = 2.0f;
    desc.material.restitution = 0.0f;
    const auto body = physics.createBody(desc);
    const std::vector<CompoundChild> children{
        boxChild({0.5f, 0.5f, 0.5f}, {-0.6f, 0.0f, 0.0f}),
        boxChild({0.5f, 0.5f, 0.5f}, {0.6f, 0.0f, 0.0f}),
    };
    (void)physics.createCompoundCollider(body, children);

    stepMany(physics, 240);

    const auto t = physics.bodyTransform(body).value();
    // Boxes (half-height 0.5) rest on the floor top (y=0) → body origin y ≈ 0.5.
    CHECK(t.position().y() == Catch::Approx(0.5f).margin(0.03f));
    // Did not tip over: still roughly upright.
    CHECK(t.rotation().approxEqual(Quaternion::identity(), 0.05f));
    CHECK(physics.sleeping(body));
}

TEST_CASE("Compound.OffsetChildBlocksAnIncomingBody", "[Compound]")
{
    // A static compound with a child box well off the origin: a body fired at that
    // child is blocked by it (proving the offset child collides, not just a child at
    // the body origin).
    PhysicsWorld physics;
    PhysicsBodyDesc wallDesc;
    wallDesc.type = PhysicsBodyType::Static;
    const auto wall = physics.createBody(wallDesc);
    const std::vector<CompoundChild> children{
        boxChild({0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f}),
        boxChild({0.5f, 0.5f, 0.5f}, {4.0f, 0.0f, 0.0f}),
    };
    (void)physics.createCompoundCollider(wall, children);

    // Sphere fired in +z at the offset child (x=4).
    PhysicsBodyDesc ballDesc;
    ballDesc.type = PhysicsBodyType::Dynamic;
    ballDesc.position = {4.0f, 0.0f, -3.0f};
    ballDesc.linearVelocity = {0.0f, 0.0f, 3.0f};
    ballDesc.gravityScale = 0.0f;
    ballDesc.material.restitution = 0.0f;
    const auto ball = physics.createBody(ballDesc);
    ColliderDesc sphere;
    sphere.shape = SphereShape{0.5f, Vec3{}};
    (void)physics.createCollider(ball, sphere);

    stepMany(physics, 120);

    // Blocked at the box's -z face: sphere (r 0.5) + box (half 0.5) → centre z ≈ -1.
    const auto p = physics.bodyTransform(ball)->position();
    CHECK(p.z() == Catch::Approx(-1.0f).margin(0.05f));
    CHECK(p.x() == Catch::Approx(4.0f).margin(0.05f)); // stayed in front of the offset child
}
