#include <fire_engine/collision/dynamic_aabb_tree_broad_phase.hpp>

#include <deque>
#include <map>
#include <set>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include <fire_engine/collision/sweep_and_prune_broad_phase.hpp>

using fire_engine::BroadPhase;
using fire_engine::Collider;
using fire_engine::CollisionPair;
using fire_engine::DynamicAabbTreeBroadPhase;
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

bool containsPair(const std::vector<CollisionPair>& pairs, const Collider& a, const Collider& b)
{
    for (const CollisionPair& p : pairs)
    {
        if ((p.first == &a && p.second == &b) || (p.first == &b && p.second == &a))
        {
            return true;
        }
    }
    return false;
}

// The broadphase's possible pairs as a set of {lowIndex, highIndex} collider indices,
// so two implementations can be compared independently of id assignment / ordering.
std::set<std::pair<int, int>> indexPairs(BroadPhase& bp, const std::deque<Collider>& colliders)
{
    std::map<const Collider*, int> index;
    int i = 0;
    for (const Collider& c : colliders)
    {
        index.emplace(&c, i++);
    }
    bp.update();
    std::set<std::pair<int, int>> out;
    for (const CollisionPair& p : bp.possiblePairs())
    {
        const int a = index.at(p.first);
        const int b = index.at(p.second);
        out.emplace(std::min(a, b), std::max(a, b));
    }
    return out;
}

} // namespace

TEST_CASE("DynamicAabbTree.OverlapProducesAPair", "[DynamicAabbTree]")
{
    DynamicAabbTreeBroadPhase tree;
    Collider a = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider b = makeCollider({0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f});
    (void)tree.addCollider(a);
    (void)tree.addCollider(b);
    tree.update();

    CHECK(tree.colliderCount() == 2U);
    CHECK(tree.possiblePairs().size() == 1U);
    CHECK(containsPair(tree.possiblePairs(), a, b));
    CHECK(tree.validate());
}

TEST_CASE("DynamicAabbTree.SeparatedCollidersProduceNoPair", "[DynamicAabbTree]")
{
    DynamicAabbTreeBroadPhase tree;
    Collider a = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider b = makeCollider({5.0f, 5.0f, 5.0f}, {6.0f, 6.0f, 6.0f});
    (void)tree.addCollider(a);
    (void)tree.addCollider(b);
    tree.update();

    CHECK(tree.possiblePairs().empty());
    CHECK(tree.validate());
}

TEST_CASE("DynamicAabbTree.RemovingAColliderDropsItsPairs", "[DynamicAabbTree]")
{
    DynamicAabbTreeBroadPhase tree;
    Collider a = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider b = makeCollider({0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f});
    (void)tree.addCollider(a);
    (void)tree.addCollider(b);
    tree.update();
    REQUIRE(tree.possiblePairs().size() == 1U);

    CHECK(tree.removeCollider(b));
    tree.update();
    CHECK(tree.possiblePairs().empty());
    CHECK(tree.colliderCount() == 1U);
    CHECK(tree.validate());
    CHECK_FALSE(tree.removeCollider(b)); // already gone
}

TEST_CASE("DynamicAabbTree.MovingAColliderFormsAndBreaksPairs", "[DynamicAabbTree]")
{
    DynamicAabbTreeBroadPhase tree;
    Collider a = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider b =
        makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Mat4::translate({5.0f, 0.0f, 0.0f}));
    (void)tree.addCollider(a);
    (void)tree.addCollider(b);
    tree.update();
    REQUIRE(tree.possiblePairs().empty());

    // Move b onto a.
    b.update(Mat4::translate({0.5f, 0.0f, 0.0f}));
    tree.update();
    CHECK(tree.possiblePairs().size() == 1U);
    CHECK(tree.validate());

    // Move b away again. The swept bounds cover the motion path for one step (CCD),
    // so the pair only clears once b has settled at the new spot (second update).
    b.update(Mat4::translate({5.0f, 0.0f, 0.0f}));
    tree.update();
    b.update(Mat4::translate({5.0f, 0.0f, 0.0f}));
    tree.update();
    CHECK(tree.possiblePairs().empty());
    CHECK(tree.validate());
}

TEST_CASE("DynamicAabbTree.LayerMaskFilteringExcludesPairs", "[DynamicAabbTree]")
{
    DynamicAabbTreeBroadPhase tree;
    Collider a = makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Collider b = makeCollider({0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f});
    // a only collides with layer 2; b is on layer 4 → filtered out.
    a.collisionMask(2U);
    b.collisionLayer(4U);
    (void)tree.addCollider(a);
    (void)tree.addCollider(b);
    tree.update();

    CHECK(tree.possiblePairs().empty());
    CHECK(tree.validate());
}

TEST_CASE("DynamicAabbTree.PairSetMatchesSweepAndPrune", "[DynamicAabbTree]")
{
    // A scene with a mix of overlaps, touches, separations and a stacked cluster: the
    // tree's pair set must equal sweep-and-prune's on the same colliders.
    std::deque<Collider> colliders;
    colliders.push_back(makeCollider({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}));
    colliders.push_back(makeCollider({0.5f, 0.0f, 0.0f}, {1.5f, 1.0f, 1.0f}));       // overlaps 0
    colliders.push_back(makeCollider({1.0f, 0.0f, 0.0f}, {2.0f, 1.0f, 1.0f}));       // touches 1
    colliders.push_back(makeCollider({10.0f, 10.0f, 10.0f}, {11.0f, 11.0f, 11.0f})); // alone
    colliders.push_back(makeCollider({0.2f, 0.2f, 0.2f}, {0.8f, 0.8f, 0.8f}));       // inside 0
    colliders.push_back(makeCollider({-0.5f, -0.5f, -0.5f}, {0.1f, 0.1f, 0.1f}));    // clips 0

    SweepAndPruneBroadPhase sap;
    for (Collider& c : colliders)
    {
        (void)sap.addCollider(c);
    }
    const auto sapPairs = indexPairs(sap, colliders);

    DynamicAabbTreeBroadPhase tree;
    for (Collider& c : colliders)
    {
        (void)tree.addCollider(c);
    }
    const auto treePairs = indexPairs(tree, colliders);

    CHECK(treePairs == sapPairs);
    CHECK(tree.validate());
    CHECK(sap.validate());
}

TEST_CASE("DynamicAabbTree.ManyCollidersStayValidAfterChurn", "[DynamicAabbTree]")
{
    // Insert a grid, remove every other one, move the rest — validate() (vs brute
    // force) must hold throughout, exercising insert/remove/refit/balance.
    std::deque<Collider> colliders;
    DynamicAabbTreeBroadPhase tree;
    for (int i = 0; i < 20; ++i)
    {
        const float x = static_cast<float>(i) * 0.75f; // adjacent boxes overlap
        colliders.push_back(makeCollider({x, 0.0f, 0.0f}, {x + 1.0f, 1.0f, 1.0f}));
    }
    for (Collider& c : colliders)
    {
        (void)tree.addCollider(c);
    }
    tree.update();
    CHECK(tree.validate());

    for (std::size_t i = 0; i < colliders.size(); i += 2)
    {
        CHECK(tree.removeCollider(colliders[i]));
    }
    tree.update();
    CHECK(tree.validate());
    CHECK(tree.colliderCount() == 10U);

    for (std::size_t i = 1; i < colliders.size(); i += 2)
    {
        colliders[i].update(Mat4::translate({0.0f, 5.0f, 0.0f}));
    }
    tree.update();
    CHECK(tree.validate());
}
