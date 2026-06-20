#include <fire_engine/collision/gjk_epa.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/collision/world_shape.hpp>

using fire_engine::ConvexContact;
using fire_engine::gjkEpaContact;
using fire_engine::Quaternion;
using fire_engine::Vec3;
using fire_engine::WorldBox;
using fire_engine::WorldShape;
using fire_engine::WorldSphere;

namespace
{

WorldBox unitBox(Vec3 center)
{
    return WorldBox{center, {1.0f, 1.0f, 1.0f}, Quaternion::identity()};
}

} // namespace

TEST_CASE("GjkEpa.SeparatedBoxesReportGapAndWitnesses", "[GjkEpa]")
{
    // Two unit boxes 1.0 apart along x (faces at x=1 and x=2).
    const WorldShape a{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape b{unitBox({3.0f, 0.0f, 0.0f})};

    const ConvexContact c = gjkEpaContact(a, b);
    CHECK_FALSE(c.colliding);
    CHECK(c.depth == Catch::Approx(1.0f).margin(1e-4f)); // gap distance
    // Witnesses on the facing faces; normal points B -> A = -x.
    CHECK(c.pointA.x() == Catch::Approx(1.0f).margin(1e-3f));
    CHECK(c.pointB.x() == Catch::Approx(2.0f).margin(1e-3f));
    CHECK(c.normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-3f));
}

TEST_CASE("GjkEpa.SeparatedSphereBoxDistance", "[GjkEpa]")
{
    // Sphere radius 0.5 at x=2.5; box spans [-1,1]. Closest points: box face x=1,
    // sphere surface x=2.0 → gap 1.0.
    const WorldShape box{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape sphere{WorldSphere{{2.5f, 0.0f, 0.0f}, 0.5f}};

    const ConvexContact c = gjkEpaContact(box, sphere);
    CHECK_FALSE(c.colliding);
    CHECK(c.depth == Catch::Approx(1.0f).margin(1e-3f));
}

TEST_CASE("GjkEpa.OverlappingBoxesReportEpaDepthAndNormal", "[GjkEpa]")
{
    // Overlapping unit boxes (centres 1.0 apart along x, each half-extent 1.0):
    // x overlap is 1.0 (the minimum axis), y/z overlap 2.0. EPA → depth 1, normal
    // along x. Normal points b -> a: a is at -x of b, so -x.
    const WorldShape a{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape b{unitBox({1.0f, 0.0f, 0.0f})};

    const ConvexContact c = gjkEpaContact(a, b);
    REQUIRE(c.colliding);
    CHECK(c.depth == Catch::Approx(1.0f).margin(1e-3f));
    CHECK(c.normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-3f));
}

TEST_CASE("GjkEpa.OverlappingBoxesDeepYAxis", "[GjkEpa]")
{
    // Boxes overlapping most along x but the minimum-penetration axis is y here:
    // a at origin, b shifted (0.2, 1.6, 0) → y overlap 0.4 (min), normal +y? b is
    // above a, so b -> a normal is -y.
    const WorldShape a{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape b{unitBox({0.2f, 1.6f, 0.0f})};

    const ConvexContact c = gjkEpaContact(a, b);
    REQUIRE(c.colliding);
    CHECK(c.depth == Catch::Approx(0.4f).margin(1e-2f));
    CHECK(c.normal.approxEqual(Vec3{0.0f, -1.0f, 0.0f}, 1e-2f));
}

TEST_CASE("GjkEpa.TouchingReportsColliding", "[GjkEpa]")
{
    // Boxes exactly touching (faces coincident at x=1).
    const WorldShape a{unitBox({0.0f, 0.0f, 0.0f})};
    const WorldShape b{unitBox({2.0f, 0.0f, 0.0f})};

    const ConvexContact c = gjkEpaContact(a, b);
    // Touching is the boundary; either a ~0 gap or colliding is acceptable.
    if (!c.colliding)
    {
        CHECK(c.depth == Catch::Approx(0.0f).margin(1e-3f));
    }
}
