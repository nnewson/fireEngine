#include <fire_engine/collision/narrow_phase.hpp>
#include <fire_engine/collision/world_shape.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::NarrowPhase;
using fire_engine::Quaternion;
using fire_engine::Vec3;
using fire_engine::WorldBox;
using fire_engine::WorldCapsule;
using fire_engine::WorldShape;
using fire_engine::WorldSphere;

// --- Shape-specific contact manifolds (collide). Normal points b -> a. -------

TEST_CASE("Collide.SphereSphere", "[Collide]")
{
    NarrowPhase np;
    const WorldShape a{WorldSphere{{0.0f, 0.0f, 0.0f}, 1.0f}};
    const WorldShape b{WorldSphere{{1.5f, 0.0f, 0.0f}, 1.0f}};
    auto m = np.collide(a, b);
    REQUIRE(m.has_value());
    CHECK(m->count == 1);
    CHECK(m->normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-5f)); // b -> a
    CHECK(m->maxPenetration() == Catch::Approx(0.5f).margin(1e-5f));

    // Separated → no manifold.
    const WorldShape far{WorldSphere{{5.0f, 0.0f, 0.0f}, 1.0f}};
    CHECK_FALSE(np.collide(a, far).has_value());
}

TEST_CASE("Collide.SphereBox", "[Collide]")
{
    NarrowPhase np;
    const WorldShape s{WorldSphere{{1.4f, 0.0f, 0.0f}, 0.5f}};
    const WorldShape box{WorldBox{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()}};
    auto m = np.collide(s, box);
    REQUIRE(m.has_value());
    CHECK(m->normal.approxEqual(Vec3{1.0f, 0.0f, 0.0f}, 1e-5f)); // box(b) -> sphere(a)
    CHECK(m->maxPenetration() == Catch::Approx(0.1f).margin(1e-5f));
}

TEST_CASE("Collide.SphereCapsule", "[Collide]")
{
    NarrowPhase np;
    const WorldShape s{WorldSphere{{0.8f, 0.0f, 0.0f}, 0.5f}};
    const WorldShape cap{WorldCapsule{{0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.5f}};
    auto m = np.collide(s, cap);
    REQUIRE(m.has_value());
    CHECK(m->normal.approxEqual(Vec3{1.0f, 0.0f, 0.0f}, 1e-5f));
    CHECK(m->maxPenetration() == Catch::Approx(0.2f).margin(1e-5f));
}

TEST_CASE("Collide.CapsuleCapsule", "[Collide]")
{
    NarrowPhase np;
    const WorldShape a{WorldCapsule{{-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 0.5f}};
    const WorldShape b{WorldCapsule{{0.0f, -1.0f, 0.6f}, {0.0f, 1.0f, 0.6f}, 0.5f}};
    auto m = np.collide(a, b);
    REQUIRE(m.has_value());
    // Closest approach along z; a is at z=0, b at z=0.6, radii sum 1.0 → pen 0.4.
    CHECK(m->normal.approxEqual(Vec3{0.0f, 0.0f, -1.0f}, 1e-4f));
    CHECK(m->maxPenetration() == Catch::Approx(0.4f).margin(1e-4f));
}

TEST_CASE("Collide.BoxBoxFaceManifold", "[Collide]")
{
    NarrowPhase np;
    const WorldShape a{WorldBox{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()}};
    const WorldShape b{WorldBox{{1.5f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()}};
    auto m = np.collide(a, b);
    REQUIRE(m.has_value());
    CHECK(m->normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-5f)); // b -> a
    CHECK(m->maxPenetration() == Catch::Approx(0.5f).margin(1e-4f));
    CHECK(m->count == 4); // overlapping faces clip to a 4-point manifold
}

TEST_CASE("Collide.BoxBoxSeparated", "[Collide]")
{
    NarrowPhase np;
    const WorldShape a{WorldBox{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()}};
    const WorldShape b{WorldBox{{3.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()}};
    CHECK_FALSE(np.collide(a, b).has_value());
}

TEST_CASE("Collide.BoxCapsule", "[Collide]")
{
    NarrowPhase np;
    const WorldShape box{WorldBox{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()}};
    const WorldShape cap{WorldCapsule{{1.4f, -2.0f, 0.0f}, {1.4f, 2.0f, 0.0f}, 0.5f}};
    auto m = np.collide(box, cap);
    REQUIRE(m.has_value());
    // box(a) <- cap(b): normal points cap -> box = -x.
    CHECK(m->normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-4f));
    CHECK(m->maxPenetration() == Catch::Approx(0.1f).margin(1e-4f));
}

// --- Speculative margin: separated-but-within-margin → negative-penetration gap --

TEST_CASE("Collide.SpeculativeSphereGapWithinMargin", "[Collide]")
{
    NarrowPhase np;
    const WorldShape a{WorldSphere{{0.0f, 0.0f, 0.0f}, 1.0f}};
    const WorldShape b{WorldSphere{{2.5f, 0.0f, 0.0f}, 1.0f}}; // surfaces 0.5 apart

    CHECK_FALSE(np.collide(a, b).has_value()); // margin 0 → separated, no contact

    auto m = np.collide(a, b, 1.0f); // margin covers the 0.5 gap
    REQUIRE(m.has_value());
    CHECK(m->count == 1);
    CHECK(m->normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-5f));          // b -> a
    CHECK(m->points[0].penetration == Catch::Approx(-0.5f).margin(1e-5f)); // negative = gap

    CHECK_FALSE(np.collide(a, b, 0.3f).has_value()); // margin < gap → still none
}

TEST_CASE("Collide.SpeculativeBoxGapWithinMargin", "[Collide]")
{
    NarrowPhase np;
    const WorldShape a{WorldBox{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()}};
    const WorldShape b{WorldBox{{3.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()}};
    // Faces 1.0 apart along x (the BoxBoxSeparated case).
    CHECK_FALSE(np.collide(a, b).has_value()); // margin 0 → none

    auto m = np.collide(a, b, 1.5f);
    REQUIRE(m.has_value());
    REQUIRE(m->count >= 1);
    CHECK(m->normal.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-5f));          // b -> a
    CHECK(m->points[0].penetration == Catch::Approx(-1.0f).margin(1e-4f)); // gap
    CHECK(m->maxPenetration() == Catch::Approx(0.0f).margin(1e-6f));       // gaps don't count
}
