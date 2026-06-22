#include <fire_engine/collision/shape_cast.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using fire_engine::shapeCast;
using fire_engine::Vec3;
using fire_engine::WorldBox;
using fire_engine::WorldCapsule;
using fire_engine::WorldShape;
using fire_engine::WorldSphere;

TEST_CASE("ShapeCast.SphereHitsSphereAtGapDistance", "[ShapeCast]")
{
    // Unit-radius spheres: centres 5 apart, radii sum 1 → contact when the moving
    // centre reaches x = 4.
    const WorldShape moving = WorldSphere{Vec3{0.0f, 0.0f, 0.0f}, 0.5f};
    const WorldShape target = WorldSphere{Vec3{5.0f, 0.0f, 0.0f}, 0.5f};
    const auto hit = shapeCast(moving, Vec3{1.0f, 0.0f, 0.0f}, 10.0f, target);
    REQUIRE(hit.has_value());
    CHECK(hit->distance == Approx(4.0f).margin(1e-3f));
    CHECK(hit->normal.x() == Approx(-1.0f).margin(1e-3f)); // points back toward the sweep
}

TEST_CASE("ShapeCast.BoxHitsBox", "[ShapeCast]")
{
    const WorldShape moving = WorldBox{Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.5f, 0.5f, 0.5f}, {}};
    const WorldShape target = WorldBox{Vec3{4.0f, 0.0f, 0.0f}, Vec3{0.5f, 0.5f, 0.5f}, {}};
    const auto hit = shapeCast(moving, Vec3{1.0f, 0.0f, 0.0f}, 10.0f, target);
    REQUIRE(hit.has_value());
    CHECK(hit->distance == Approx(3.0f).margin(1e-3f)); // faces meet at x-gap 3
}

TEST_CASE("ShapeCast.AlreadyOverlappingReturnsZero", "[ShapeCast]")
{
    const WorldShape moving = WorldSphere{Vec3{0.0f, 0.0f, 0.0f}, 1.0f};
    const WorldShape target = WorldSphere{Vec3{0.5f, 0.0f, 0.0f}, 1.0f};
    const auto hit = shapeCast(moving, Vec3{1.0f, 0.0f, 0.0f}, 10.0f, target);
    REQUIRE(hit.has_value());
    CHECK(hit->distance == Approx(0.0f).margin(1e-4f));
}

TEST_CASE("ShapeCast.MissWhenSweepFallsShort", "[ShapeCast]")
{
    const WorldShape moving = WorldSphere{Vec3{0.0f, 0.0f, 0.0f}, 0.5f};
    const WorldShape target = WorldSphere{Vec3{5.0f, 0.0f, 0.0f}, 0.5f};
    CHECK_FALSE(shapeCast(moving, Vec3{1.0f, 0.0f, 0.0f}, 2.0f, target).has_value());
}

TEST_CASE("ShapeCast.MissWhenSweepingAway", "[ShapeCast]")
{
    const WorldShape moving = WorldSphere{Vec3{0.0f, 0.0f, 0.0f}, 0.5f};
    const WorldShape target = WorldSphere{Vec3{5.0f, 0.0f, 0.0f}, 0.5f};
    CHECK_FALSE(shapeCast(moving, Vec3{-1.0f, 0.0f, 0.0f}, 100.0f, target).has_value());
}

TEST_CASE("ShapeCast.CapsuleHitsSphere", "[ShapeCast]")
{
    const WorldShape moving = WorldCapsule{Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}, 0.5f};
    const WorldShape target = WorldSphere{Vec3{4.0f, 0.0f, 0.0f}, 0.5f};
    const auto hit = shapeCast(moving, Vec3{1.0f, 0.0f, 0.0f}, 10.0f, target);
    REQUIRE(hit.has_value());
    CHECK(hit->distance == Approx(3.0f).margin(1e-3f)); // capsule radius 0.5 + sphere 0.5
}
