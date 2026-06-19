#include <fire_engine/collision/narrow_phase.hpp>
#include <fire_engine/collision/world_shape.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Collider;
using fire_engine::ContactManifold;
using fire_engine::Mat4;
using fire_engine::NarrowPhase;
using fire_engine::Quaternion;
using fire_engine::Vec3;
using fire_engine::WorldBox;
using fire_engine::WorldCapsule;
using fire_engine::WorldShape;
using fire_engine::WorldSphere;

namespace
{

Collider makeCollider(Vec3 min, Vec3 max, Mat4 world = Mat4::identity())
{
    Collider collider;
    collider.localBounds({min, max});
    collider.update(world);
    return collider;
}

} // namespace

TEST_CASE("SweptAabb.MovingBoxReportsTimeOfImpactAndNormal", "[SweptAabb]")
{
    NarrowPhase narrowPhase;
    Collider moving = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider target = makeCollider({3.0f, 0.0f, 0.0f}, {4.0f, 1.0f, 1.0f});

    moving.update(Mat4::translate({4.0f, 0.0f, 0.0f}));

    auto contact = narrowPhase.sweptAabb(moving, target);
    REQUIRE(contact.has_value());
    CHECK(contact->toi == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(contact->normal == Vec3(-1.0f, 0.0f, 0.0f));
}

TEST_CASE("SweptAabb.MissReturnsNoContact", "[SweptAabb]")
{
    NarrowPhase narrowPhase;
    Collider moving = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider target = makeCollider({3.0f, 3.0f, 0.0f}, {4.0f, 4.0f, 1.0f});

    moving.update(Mat4::translate({4.0f, 0.0f, 0.0f}));

    CHECK_FALSE(narrowPhase.sweptAabb(moving, target).has_value());
}

TEST_CASE("SweptAabb.StartingOverlapReturnsImmediateContact", "[SweptAabb]")
{
    NarrowPhase narrowPhase;
    Collider moving = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider target = makeCollider({1.0f, 0.0f, 0.0f}, {3.0f, 2.0f, 2.0f});

    moving.update(Mat4::translate({1.0f, 0.0f, 0.0f}));

    auto contact = narrowPhase.sweptAabb(moving, target);
    REQUIRE(contact.has_value());
    CHECK(contact->toi == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(contact->normal == Vec3(-1.0f, 0.0f, 0.0f));
}

TEST_CASE("SweptAabb.StartingTouchUsesGeometryNormalSoMovingAwayCanBeIgnored", "[SweptAabb]")
{
    NarrowPhase narrowPhase;
    Collider moving = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider target = makeCollider({1.0f, 0.0f, 0.0f}, {2.0f, 1.0f, 1.0f});

    moving.update(Mat4::translate({-1.0f, 0.0f, 0.0f}));

    auto contact = narrowPhase.sweptAabb(moving, target);
    REQUIRE(contact.has_value());
    CHECK(contact->toi == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(contact->normal == Vec3(-1.0f, 0.0f, 0.0f));
}

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
