#include <fire_engine/physics/character_controller.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/physics/physics_world.hpp>

using Catch::Approx;
using fire_engine::CharacterController;
using fire_engine::CharacterControllerConfig;
using fire_engine::CharacterMoveResult;
using fire_engine::ColliderDesc;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsWorld;
using fire_engine::Quaternion;
using fire_engine::Vec3;
using BoxShape = fire_engine::BoxShape;

namespace
{

// A static box. Rotation is set before the collider is created so its world bounds /
// world shape are composed from the rotated transform.
void addStaticBox(PhysicsWorld& world, Vec3 position, Vec3 halfExtents,
                  Quaternion rotation = Quaternion::identity())
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    desc.position = position;
    desc.rotation = rotation;
    const PhysicsBodyHandle body = world.createBody(desc);
    (void)world.createCollider(body, ColliderDesc{.shape = BoxShape{halfExtents, Vec3{}}});
}

// Capsule-centre rest height on a floor whose top is at `floorTop`, for the default config
// (radius 0.35, height 1.8 → half-segment 0.55; bottom = centre − 0.9).
constexpr float kDefaultBottomOffset = 0.9f;

} // namespace

TEST_CASE("CharacterController.GroundedOnFloorSnaps", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f}); // top at y = 0.5

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.7f, 0.0f}};
    const CharacterMoveResult r = cc.move(world, {0.0f, -0.1f, 0.0f});

    CHECK(r.grounded);
    CHECK(r.groundNormal.y() == Approx(1.0f).margin(1e-2f));
    // Snapped to rest: bottom on the floor top (0.5) → centre 1.4.
    CHECK(r.position.y() == Approx(0.5f + kDefaultBottomOffset).margin(0.05f));
}

TEST_CASE("CharacterController.AirborneIsNotGrounded", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f});

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 10.0f, 0.0f}};
    const CharacterMoveResult r = cc.move(world, {0.0f, -0.1f, 0.0f});
    CHECK_FALSE(r.grounded);
}

TEST_CASE("CharacterController.BlockedByWall", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {2.0f, 0.0f, 0.0f}, {0.5f, 5.0f, 5.0f}); // near face at x = 1.5

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.0f, 0.0f}};
    (void)cc.move(world, {3.0f, 0.0f, 0.0f}); // walk hard into the wall

    // Stops with its radius (+ skin) short of the wall face, nowhere near x = 3.
    CHECK(cc.position().x() < 1.2f);
    CHECK(cc.position().x() > 1.0f);
}

TEST_CASE("CharacterController.SlidesAlongWall", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {2.0f, 0.0f, 0.0f}, {0.5f, 5.0f, 5.0f}); // wall facing -x at x = 1.5

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.0f, 0.0f}};
    (void)cc.move(world, {3.0f, 0.0f, 1.0f}); // into the wall, angled along +z

    CHECK(cc.position().x() < 1.2f); // blocked on x
    CHECK(cc.position().z() > 0.8f); // slid along z
}

TEST_CASE("CharacterController.StepsUpLowLedge", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f}); // floor, top 0.5
    addStaticBox(world, {2.0f, 0.5f, 0.0f}, {1.0f, 0.15f, 2.0f});  // ledge top at 0.5+0.15+0.5

    // Start grounded on the floor in front of the ledge; walk onto it (a handful of
    // moves — too many would walk off the far edge of the ledge).
    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.4f, 0.0f}};
    CharacterMoveResult r;
    for (int i = 0; i < 6; ++i)
    {
        r = cc.move(world, Vec3{0.3f, 0.0f, 0.0f} + Vec3{0.0f, -0.05f, 0.0f});
    }
    // Mounted the ledge: on top of it (x within [1,3]) and risen ~0.15 above the floor
    // rest height (1.4 → ~1.55).
    CHECK(r.grounded);
    CHECK(cc.position().x() > 1.4f);
    CHECK(cc.position().y() > 1.5f);
}

TEST_CASE("CharacterController.WalksUpGentleRamp", "[CharacterController]")
{
    PhysicsWorld world;
    // ~30° ramp: tilt the box up axis toward (-sin, cos, 0) so +x is uphill.
    const float a = 0.5236f; // 30°
    const Quaternion tilt =
        Quaternion::fromVectors(Vec3{0.0f, 1.0f, 0.0f}, Vec3{-std::sin(a), std::cos(a), 0.0f});
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {6.0f, 0.25f, 6.0f}, tilt);

    // Start just above the ramp surface near the centre and settle briefly (gravity
    // slides a capsule down a tilted ramp, so a long settle would walk it off the edge).
    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.4f, 0.0f}};
    CharacterMoveResult r;
    for (int i = 0; i < 15; ++i)
    {
        r = cc.move(world, {0.0f, -0.1f, 0.0f});
    }
    REQUIRE(r.grounded);

    // Uphill = opposite the ground normal's horizontal lean.
    Vec3 uphill = Vec3{-r.groundNormal.x(), 0.0f, -r.groundNormal.z()};
    uphill = Vec3::normalise(uphill);
    const float startY = cc.position().y();
    for (int i = 0; i < 40; ++i)
    {
        (void)cc.move(world, uphill * 0.05f + Vec3{0.0f, -0.02f, 0.0f});
    }
    CHECK(cc.position().y() > startY + 0.1f); // climbed
}

TEST_CASE("CharacterController.SteepSlopeIsNotClimbed", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f}); // floor
    // ~70° face: too steep to walk up (normal.y ≈ 0.34 < maxSlopeCosine 0.64).
    const float a = 1.2217f; // 70°
    const Quaternion tilt =
        Quaternion::fromVectors(Vec3{0.0f, 1.0f, 0.0f}, Vec3{-std::sin(a), std::cos(a), 0.0f});
    addStaticBox(world, {3.0f, 2.0f, 0.0f}, {1.5f, 0.25f, 4.0f}, tilt);

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.4f, 0.0f}};
    const float startY = cc.position().y();
    for (int i = 0; i < 40; ++i)
    {
        (void)cc.move(world, Vec3{0.2f, 0.0f, 0.0f} + Vec3{0.0f, -0.05f, 0.0f});
    }
    // It walked into the slope but did not climb it.
    CHECK(cc.position().y() < startY + 0.2f);
}
