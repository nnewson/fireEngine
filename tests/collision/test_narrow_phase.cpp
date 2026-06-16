#include <fire_engine/collision/narrow_phase.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Collider;
using fire_engine::Mat4;
using fire_engine::NarrowPhase;
using fire_engine::Vec3;

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
