#include <fire_engine/collision/sweep_and_prune_broad_phase.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::AABB;
using fire_engine::Collider;
using fire_engine::ColliderId;
using fire_engine::EndPoint;
using fire_engine::Mat4;
using fire_engine::SweepAndPruneBroadPhase;
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

bool containsPair(const std::vector<fire_engine::CollisionPair>& pairs, const Collider& first,
                  const Collider& second)
{
    for (const fire_engine::CollisionPair& pair : pairs)
    {
        if ((pair.first == &first && pair.second == &second) ||
            (pair.first == &second && pair.second == &first))
        {
            return true;
        }
    }

    return false;
}

} // namespace

TEST_CASE("ColliderUpdate.TranslatesLocalBoundsAsPositions", "[ColliderUpdate]")
{
    Collider collider = makeCollider({0.0f, 1.0f, 2.0f}, {1.0f, 2.0f, 3.0f},
                                     Mat4::translate({10.0f, 20.0f, 30.0f}));

    AABB bounds = collider.worldBounds();
    CHECK(bounds.min.x() == Catch::Approx(10.0f).margin(1e-5f));
    CHECK(bounds.min.y() == Catch::Approx(21.0f).margin(1e-5f));
    CHECK(bounds.min.z() == Catch::Approx(32.0f).margin(1e-5f));
    CHECK(bounds.max.x() == Catch::Approx(11.0f).margin(1e-5f));
    CHECK(bounds.max.y() == Catch::Approx(22.0f).margin(1e-5f));
    CHECK(bounds.max.z() == Catch::Approx(33.0f).margin(1e-5f));
}

TEST_CASE("ColliderUpdate.RebuildsConservativeBoundsForNegativeScale", "[ColliderUpdate]")
{
    Collider collider =
        makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}, Mat4::scale({-2.0f, 3.0f, -4.0f}));

    AABB bounds = collider.worldBounds();
    CHECK(bounds.min.x() == Catch::Approx(-2.0f).margin(1e-5f));
    CHECK(bounds.min.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(bounds.min.z() == Catch::Approx(-12.0f).margin(1e-5f));
    CHECK(bounds.max.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(bounds.max.y() == Catch::Approx(6.0f).margin(1e-5f));
    CHECK(bounds.max.z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("ColliderUpdate.FirstUpdateSeedsPreviousAndSweptBoundsToCurrent", "[ColliderUpdate]")
{
    Collider collider =
        makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Mat4::translate({2.0f, 0.0f, 0.0f}));

    CHECK(collider.previousWorldBounds().min == Vec3(2.0f, 0.0f, 0.0f));
    CHECK(collider.previousWorldBounds().max == Vec3(3.0f, 1.0f, 1.0f));
    CHECK(collider.sweptWorldBounds().min == Vec3(2.0f, 0.0f, 0.0f));
    CHECK(collider.sweptWorldBounds().max == Vec3(3.0f, 1.0f, 1.0f));
}

TEST_CASE("ColliderUpdate.SweptBoundsMergePreviousAndCurrentBounds", "[ColliderUpdate]")
{
    Collider collider = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});

    collider.update(Mat4::translate({3.0f, 0.0f, 0.0f}));

    CHECK(collider.previousWorldBounds().min == Vec3(0.0f, 0.0f, 0.0f));
    CHECK(collider.previousWorldBounds().max == Vec3(1.0f, 1.0f, 1.0f));
    CHECK(collider.worldBounds().min == Vec3(3.0f, 0.0f, 0.0f));
    CHECK(collider.worldBounds().max == Vec3(4.0f, 1.0f, 1.0f));
    CHECK(collider.sweptWorldBounds().min == Vec3(0.0f, 0.0f, 0.0f));
    CHECK(collider.sweptWorldBounds().max == Vec3(4.0f, 1.0f, 1.0f));
}

TEST_CASE("SweepAndPruneBroadPhase.EmptyAndSingleColliderProduceNoPairs",
          "[SweepAndPruneBroadPhase]")
{
    SweepAndPruneBroadPhase broadPhase;
    broadPhase.update();
    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());

    Collider collider = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    ColliderId id = broadPhase.addCollider(collider);
    broadPhase.update();

    CHECK(id.valid());
    CHECK(collider.colliderId() == id);
    CHECK(broadPhase.colliderCount() == 1U);
    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.OverlapOnAllAxesProducesPossiblePair",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});

    SweepAndPruneBroadPhase broadPhase;
    ColliderId firstId = broadPhase.addCollider(first);
    ColliderId secondId = broadPhase.addCollider(second);
    broadPhase.update();

    REQUIRE(broadPhase.possiblePairs().size() == 1U);
    CHECK(containsPair(broadPhase.possiblePairs(), first, second));
    CHECK(broadPhase.possiblePairs()[0].firstId == firstId);
    CHECK(broadPhase.possiblePairs()[0].secondId == secondId);
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.AddColliderReturnsStableIdsAndAssignsEndPointIndices",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider second = makeCollider({2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f});

    SweepAndPruneBroadPhase broadPhase;
    ColliderId firstId = broadPhase.addCollider(first);
    ColliderId secondId = broadPhase.addCollider(second);

    CHECK(firstId.valid());
    CHECK(secondId.valid());
    CHECK(firstId != secondId);
    CHECK(first.colliderId() == firstId);
    CHECK(second.colliderId() == secondId);

    for (EndPoint* endPoint : first.endPoints())
    {
        CHECK(endPoint->index() != EndPoint::invalidIndex);
        CHECK(endPoint->colliderId() == firstId);
    }

    for (EndPoint* endPoint : second.endPoints())
    {
        CHECK(endPoint->index() != EndPoint::invalidIndex);
        CHECK(endPoint->colliderId() == secondId);
    }

    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.OverlapOnOnlyTwoAxesIsPruned", "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 3.0f}, {3.0f, 3.0f, 4.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    broadPhase.update();

    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.TouchingBoundsProducePossiblePair", "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider second = makeCollider({1.0f, 0.0f, 0.0f}, {2.0f, 1.0f, 1.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    broadPhase.update();

    REQUIRE(broadPhase.possiblePairs().size() == 1U);
    CHECK(containsPair(broadPhase.possiblePairs(), first, second));
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.UpdateColliderIncrementallyAddsPossiblePair",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider second = makeCollider({3.0f, 0.0f, 0.0f}, {4.0f, 1.0f, 1.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    REQUIRE(broadPhase.possiblePairs().empty());

    second.update(Mat4::translate({-2.5f, 0.0f, 0.0f}));
    broadPhase.updateCollider(second);

    REQUIRE(broadPhase.possiblePairs().size() == 1U);
    CHECK(containsPair(broadPhase.possiblePairs(), first, second));
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.UpdateColliderIncrementallyRemovesPossiblePair",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    REQUIRE(broadPhase.possiblePairs().size() == 1U);

    second.update(Mat4::translate({5.0f, 0.0f, 0.0f}));
    broadPhase.updateCollider(second);

    REQUIRE(broadPhase.possiblePairs().size() == 1U);
    CHECK(containsPair(broadPhase.possiblePairs(), first, second));
    CHECK(broadPhase.validate());

    second.resetFrame(Mat4::translate({5.0f, 0.0f, 0.0f}));
    broadPhase.updateCollider(second);

    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.UpdateIncrementallyProcessesAllRegisteredColliders",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider second = makeCollider({3.0f, 0.0f, 0.0f}, {4.0f, 1.0f, 1.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    REQUIRE(broadPhase.possiblePairs().empty());

    second.update(Mat4::translate({-2.5f, 0.0f, 0.0f}));
    broadPhase.update();

    REQUIRE(broadPhase.possiblePairs().size() == 1U);
    CHECK(containsPair(broadPhase.possiblePairs(), first, second));
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.SweptBoundsProducePossiblePairForFastPassThrough",
          "[SweepAndPruneBroadPhase]")
{
    Collider moving = makeCollider({-3.0f, 0.0f, 0.0f}, {-2.0f, 1.0f, 1.0f});
    Collider target = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(moving);
    broadPhase.addCollider(target);
    REQUIRE(broadPhase.possiblePairs().empty());

    moving.update(Mat4::translate({5.0f, 0.0f, 0.0f}));
    broadPhase.updateCollider(moving);

    REQUIRE(broadPhase.possiblePairs().size() == 1U);
    CHECK(containsPair(broadPhase.possiblePairs(), moving, target));
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.RebuildSynchronisesStateAfterFilterChange",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    REQUIRE(broadPhase.possiblePairs().size() == 1U);

    first.collisionMask(0U);
    broadPhase.rebuild();

    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.CollisionMasksFilterPairs", "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});
    first.collisionLayer(1U << 0U);
    first.collisionMask(1U << 1U);
    second.collisionLayer(1U << 2U);
    second.collisionMask(1U << 0U);

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);

    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());

    second.collisionLayer(1U << 1U);
    broadPhase.rebuild();

    REQUIRE(broadPhase.possiblePairs().size() == 1U);
    CHECK(containsPair(broadPhase.possiblePairs(), first, second));
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.MultipleCollidersOnlyReturnMultiaxisOverlaps",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});
    Collider third = makeCollider({4.0f, 4.0f, 4.0f}, {5.0f, 5.0f, 5.0f});
    Collider fourth = makeCollider({1.0f, 1.0f, 3.1f}, {3.0f, 3.0f, 4.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    broadPhase.addCollider(third);
    broadPhase.addCollider(fourth);
    broadPhase.update();

    REQUIRE(broadPhase.possiblePairs().size() == 1U);
    CHECK(containsPair(broadPhase.possiblePairs(), first, second));
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.DuplicateColliderIsIgnored", "[SweepAndPruneBroadPhase]")
{
    Collider collider = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});

    SweepAndPruneBroadPhase broadPhase;
    ColliderId firstAdd = broadPhase.addCollider(collider);
    ColliderId secondAdd = broadPhase.addCollider(collider);
    broadPhase.update();

    CHECK(firstAdd == secondAdd);
    CHECK(broadPhase.colliderCount() == 1U);
    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.RemoveColliderByIdClearsPairsAndEndpointIndices",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});

    SweepAndPruneBroadPhase broadPhase;
    ColliderId firstId = broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    REQUIRE(broadPhase.possiblePairs().size() == 1U);

    CHECK(broadPhase.removeCollider(firstId));

    CHECK(broadPhase.colliderCount() == 1U);
    CHECK_FALSE(first.colliderId().valid());
    CHECK(broadPhase.possiblePairs().empty());
    for (EndPoint* endPoint : first.endPoints())
    {
        CHECK(endPoint->index() == EndPoint::invalidIndex);
        CHECK_FALSE(endPoint->colliderId().valid());
    }
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.RemoveColliderByReferenceClearsPairs",
          "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    REQUIRE(broadPhase.possiblePairs().size() == 1U);

    CHECK(broadPhase.removeCollider(second));

    CHECK(broadPhase.colliderCount() == 1U);
    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.RemoveUnregisteredColliderReturnsFalse",
          "[SweepAndPruneBroadPhase]")
{
    Collider collider = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    SweepAndPruneBroadPhase broadPhase;

    CHECK_FALSE(broadPhase.removeCollider(collider));
    CHECK_FALSE(broadPhase.removeCollider(ColliderId{123U}));
    CHECK(broadPhase.validate());
}

TEST_CASE("SweepAndPruneBroadPhase.ClearRemovesCollidersAndPairs", "[SweepAndPruneBroadPhase]")
{
    Collider first = makeCollider({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
    Collider second = makeCollider({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});

    SweepAndPruneBroadPhase broadPhase;
    broadPhase.addCollider(first);
    broadPhase.addCollider(second);
    broadPhase.update();
    REQUIRE(broadPhase.possiblePairs().size() == 1U);

    broadPhase.clear();

    CHECK(broadPhase.colliderCount() == 0U);
    CHECK_FALSE(first.colliderId().valid());
    CHECK_FALSE(second.colliderId().valid());
    CHECK(broadPhase.possiblePairs().empty());
    CHECK(broadPhase.validate());

    CHECK_FALSE(broadPhase.removeCollider(first));
    CHECK_FALSE(broadPhase.removeCollider(second));
}
