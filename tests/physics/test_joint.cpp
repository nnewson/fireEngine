#include <fire_engine/physics/physics_world.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::ColliderDesc;
using fire_engine::JointDesc;
using fire_engine::JointType;
using fire_engine::Mat3;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsConstraintHandle;
using fire_engine::PhysicsWorld;
using fire_engine::Quaternion;
using fire_engine::SphereShape;
using fire_engine::Vec3;

namespace
{

constexpr float kDt = 1.0f / 120.0f;

// A body with a sphere collider, so the collider path assigns a well-defined
// inverse inertia (joints exercise the angular response). `withCollider` is false
// for a bare anchor body — jointed bodies typically overlap, and without a second
// collider the narrowphase generates no contact to fight the joint.
PhysicsBodyHandle makeBall(PhysicsWorld& physics, PhysicsBodyType type, Vec3 position,
                           Vec3 angularVelocity = {}, bool withCollider = true)
{
    PhysicsBodyDesc desc;
    desc.type = type;
    desc.position = position;
    desc.angularVelocity = angularVelocity;
    desc.gravityScale = 1.0f;
    desc.mass = 1.0f;
    const PhysicsBodyHandle handle = physics.createBody(desc);

    if (withCollider)
    {
        ColliderDesc collider;
        collider.shape = SphereShape{0.5f, Vec3{}};
        [[maybe_unused]] const auto c = physics.createCollider(handle, collider);
    }
    return handle;
}

void simulate(PhysicsWorld& physics, int steps)
{
    for (int i = 0; i < steps; ++i)
    {
        physics.step(kDt);
    }
}

Vec3 position(const PhysicsWorld& physics, PhysicsBodyHandle handle)
{
    return physics.bodyTransform(handle)->position();
}

// Signed rotation angle of `q` about unit `axis` (mirror of the solver's twist
// extraction; valid for a rotation purely about `axis`).
float angleAbout(const Quaternion& q0, const Vec3& axis)
{
    Quaternion q = q0;
    if (q.w() < 0.0f)
    {
        q = -q;
    }
    const float d = q.x() * axis.x() + q.y() * axis.y() + q.z() * axis.z();
    return 2.0f * std::atan2(d, q.w());
}

} // namespace

TEST_CASE("Joint.CreateAndDestroyTracksCount", "[Joint]")
{
    PhysicsWorld physics;
    const PhysicsBodyHandle a = makeBall(physics, PhysicsBodyType::Static, {});
    const PhysicsBodyHandle b = makeBall(physics, PhysicsBodyType::Dynamic, {0.0f, -2.0f, 0.0f});

    JointDesc desc;
    desc.type = JointType::Distance;
    desc.bodyA = a;
    desc.bodyB = b;
    desc.restLength = 2.0f;

    const PhysicsConstraintHandle joint = physics.createJoint(desc);
    CHECK(joint.valid());
    CHECK(physics.valid(joint));
    CHECK(physics.jointCount() == 1U);

    CHECK(physics.destroyJoint(joint));
    CHECK_FALSE(physics.valid(joint));
    CHECK(physics.jointCount() == 0U);
    CHECK_FALSE(physics.destroyJoint(joint));
}

TEST_CASE("Joint.CreateRejectsMissingBody", "[Joint]")
{
    PhysicsWorld physics;
    const PhysicsBodyHandle a = makeBall(physics, PhysicsBodyType::Static, {});

    JointDesc desc;
    desc.bodyA = a;
    desc.bodyB = PhysicsBodyHandle{}; // null
    const PhysicsConstraintHandle joint = physics.createJoint(desc);
    CHECK_FALSE(joint.valid());
    CHECK(physics.jointCount() == 0U);
}

TEST_CASE("Joint.DistanceHoldsBodiesAtRestLength", "[Joint]")
{
    PhysicsWorld physics;
    const PhysicsBodyHandle anchor = makeBall(physics, PhysicsBodyType::Static, {0.0f, 0.0f, 0.0f});
    const PhysicsBodyHandle ball = makeBall(physics, PhysicsBodyType::Dynamic, {0.0f, -2.0f, 0.0f});

    JointDesc desc;
    desc.type = JointType::Distance;
    desc.bodyA = anchor;
    desc.bodyB = ball;
    desc.restLength = 2.0f;
    [[maybe_unused]] const auto joint = physics.createJoint(desc);

    // Let it settle hanging under gravity.
    simulate(physics, 600);

    const Vec3 delta = position(physics, ball) - position(physics, anchor);
    // The bilateral distance constraint holds the separation at the rest length.
    CHECK(delta.magnitude() == Catch::Approx(2.0f).margin(0.01f));
    // Hangs directly below the anchor (gravity points -y).
    CHECK(position(physics, ball).y() == Catch::Approx(-2.0f).margin(0.02f));
}

TEST_CASE("Joint.BallSocketPinsAnchorToWorldPoint", "[Joint]")
{
    PhysicsWorld physics;
    const PhysicsBodyHandle anchor =
        makeBall(physics, PhysicsBodyType::Static, {1.0f, 5.0f, 0.0f}, Vec3{}, false);
    const PhysicsBodyHandle ball = makeBall(physics, PhysicsBodyType::Dynamic, {1.0f, 5.0f, 0.0f});

    JointDesc desc;
    desc.type = JointType::BallSocket;
    desc.bodyA = anchor;
    desc.bodyB = ball;
    desc.anchorA = Vec3{}; // both anchors at each body's centre → coincident pin
    desc.anchorB = Vec3{};
    [[maybe_unused]] const auto joint = physics.createJoint(desc);

    simulate(physics, 600);

    // The ball-socket holds the two anchors coincident, so the dynamic body's centre
    // stays at the pin despite gravity.
    const Vec3 p = position(physics, ball);
    CHECK(p.x() == Catch::Approx(1.0f).margin(0.01f));
    CHECK(p.y() == Catch::Approx(5.0f).margin(0.01f));
    CHECK(p.z() == Catch::Approx(0.0f).margin(0.01f));
}

TEST_CASE("Joint.HingeConfinesRotationToItsAxis", "[Joint]")
{
    PhysicsWorld physics;
    const PhysicsBodyHandle anchor =
        makeBall(physics, PhysicsBodyType::Static, {0.0f, 0.0f, 0.0f}, Vec3{}, false);
    // Spin the dynamic body about x; the hinge axis is z, so the off-axis spin must
    // be cancelled and the body's local z stays aligned to world z.
    const PhysicsBodyHandle ball =
        makeBall(physics, PhysicsBodyType::Dynamic, {0.0f, 0.0f, 0.0f}, Vec3{6.0f, 0.0f, 0.0f});

    JointDesc desc;
    desc.type = JointType::Hinge;
    desc.bodyA = anchor;
    desc.bodyB = ball;
    desc.anchorA = Vec3{};
    desc.anchorB = Vec3{};
    desc.axisA = Vec3{0.0f, 0.0f, 1.0f};
    desc.axisB = Vec3{0.0f, 0.0f, 1.0f};
    [[maybe_unused]] const auto joint = physics.createJoint(desc);

    simulate(physics, 300);

    // The body's local z axis, in world space, must remain (anti)parallel to the
    // hinge axis — the perpendicular rotational DOF are removed.
    const Mat3 r = Mat3::fromQuaternion(physics.bodyTransform(ball)->rotation());
    const Vec3 axisWorld = r * Vec3{0.0f, 0.0f, 1.0f};
    CHECK(std::abs(Vec3::dotProduct(axisWorld, Vec3{0.0f, 0.0f, 1.0f})) ==
          Catch::Approx(1.0f).margin(0.02f));
}

TEST_CASE("Joint.HingeAngleLimitStopsAtBound", "[Joint]")
{
    PhysicsWorld physics;
    const PhysicsBodyHandle anchor =
        makeBall(physics, PhysicsBodyType::Static, {0.0f, 0.0f, 0.0f}, Vec3{}, false);
    // Spin about the hinge axis (z); the [−0.5, +0.5] rad limit must arrest it.
    const PhysicsBodyHandle ball =
        makeBall(physics, PhysicsBodyType::Dynamic, {0.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 2.0f});

    JointDesc desc;
    desc.type = JointType::Hinge;
    desc.bodyA = anchor;
    desc.bodyB = ball;
    desc.axisA = Vec3{0.0f, 0.0f, 1.0f};
    desc.axisB = Vec3{0.0f, 0.0f, 1.0f};
    desc.limits.hinge = true;
    desc.limits.lowerAngle = -0.5f;
    desc.limits.upperAngle = 0.5f;
    [[maybe_unused]] const auto joint = physics.createJoint(desc);

    simulate(physics, 300);

    const float angle = angleAbout(physics.bodyTransform(ball)->rotation(), Vec3{0.0f, 0.0f, 1.0f});
    // Came to rest at the upper limit (small Baumgarte overshoot tolerated), and did
    // not blow through it.
    CHECK(angle == Catch::Approx(0.5f).margin(0.05f));
    CHECK(angle < 0.6f);
}

TEST_CASE("Joint.ConeTwistSwingLimitCapsSwing", "[Joint]")
{
    PhysicsWorld physics;
    const PhysicsBodyHandle anchor =
        makeBall(physics, PhysicsBodyType::Static, {0.0f, 0.0f, 0.0f}, Vec3{}, false);
    // Twist axis is y; spin about x is pure swing away from that axis.
    const PhysicsBodyHandle ball =
        makeBall(physics, PhysicsBodyType::Dynamic, {0.0f, 0.0f, 0.0f}, Vec3{2.0f, 0.0f, 0.0f});

    JointDesc desc;
    desc.type = JointType::BallSocket;
    desc.bodyA = anchor;
    desc.bodyB = ball;
    desc.axisA = Vec3{0.0f, 1.0f, 0.0f}; // twist axis
    desc.limits.coneTwist = true;
    desc.limits.swingLimit = 0.4f;
    desc.limits.twistLimit = 3.0f; // loose, not under test here
    [[maybe_unused]] const auto joint = physics.createJoint(desc);

    simulate(physics, 300);

    // Pure x-rotation → swing angle equals the rotation about x; the cone caps it.
    const float swing =
        std::abs(angleAbout(physics.bodyTransform(ball)->rotation(), Vec3{1.0f, 0.0f, 0.0f}));
    CHECK(swing == Catch::Approx(0.4f).margin(0.05f));
    CHECK(swing < 0.5f);
}

TEST_CASE("Joint.TwoBodiesJoinedRestOnFloorStable", "[Joint]")
{
    // A distance-linked pair resting on the floor: joints and contacts interleave
    // without fighting (the pair settles to a stable, bounded state).
    PhysicsWorld physics;

    PhysicsBodyDesc floorDesc;
    floorDesc.type = PhysicsBodyType::Static;
    floorDesc.position = Vec3{0.0f, -1.0f, 0.0f};
    const PhysicsBodyHandle floor = physics.createBody(floorDesc);
    ColliderDesc floorCollider;
    floorCollider.shape = fire_engine::BoxShape{Vec3{10.0f, 1.0f, 10.0f}, Vec3{}};
    [[maybe_unused]] const auto fc = physics.createCollider(floor, floorCollider);

    const PhysicsBodyHandle a = makeBall(physics, PhysicsBodyType::Dynamic, {-0.6f, 1.0f, 0.0f});
    const PhysicsBodyHandle b = makeBall(physics, PhysicsBodyType::Dynamic, {0.6f, 1.0f, 0.0f});

    JointDesc desc;
    desc.type = JointType::Distance;
    desc.bodyA = a;
    desc.bodyB = b;
    desc.restLength = 1.2f;
    [[maybe_unused]] const auto joint = physics.createJoint(desc);

    simulate(physics, 600);

    // Both balls come to rest on the floor (sphere radius 0.5, floor top at y=0).
    CHECK(position(physics, a).y() == Catch::Approx(0.5f).margin(0.05f));
    CHECK(position(physics, b).y() == Catch::Approx(0.5f).margin(0.05f));
    // The link still holds them roughly at the rest length.
    CHECK((position(physics, a) - position(physics, b)).magnitude() ==
          Catch::Approx(1.2f).margin(0.1f));
}
