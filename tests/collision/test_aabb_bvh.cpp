#include <fire_engine/collision/aabb_bvh.hpp>

#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using fire_engine::AABB;
using fire_engine::AabbBvh;
using fire_engine::aabbOverlaps;

namespace
{

AABB box(float x, float y, float z, float size = 1.0f)
{
    return AABB{{x, y, z}, {x + size, y + size, z + size}};
}

// Payloads of every leaf whose (fat) box overlaps `q`.
std::set<int> queryPayloads(const AabbBvh<int>& bvh, const AABB& q)
{
    std::set<int> out;
    bvh.query(q, [&](int proxy) { out.insert(bvh.payload(proxy)); });
    return out;
}

} // namespace

TEST_CASE("AabbBvh.QueryReturnsOverlappingPayloads", "[AabbBvh]")
{
    // margin 0 → stored boxes are tight, so a query matches a brute-force overlap test.
    AabbBvh<int> bvh{0.0f};
    std::vector<AABB> boxes;
    for (int i = 0; i < 8; ++i)
    {
        boxes.push_back(box(static_cast<float>(i) * 2.0f, 0.0f, 0.0f));
        (void)bvh.createProxy(boxes.back(), i);
    }
    CHECK(bvh.size() == 8U);

    const AABB q = box(3.0f, 0.0f, 0.0f, 4.0f); // spans x∈[3,7]
    std::set<int> expected;
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i)
    {
        if (aabbOverlaps(boxes[static_cast<std::size_t>(i)], q))
        {
            expected.insert(i);
        }
    }
    CHECK(queryPayloads(bvh, q) == expected);
}

TEST_CASE("AabbBvh.EmptyTreeQueriesNothing", "[AabbBvh]")
{
    const AabbBvh<int> bvh;
    CHECK(bvh.empty());
    CHECK(queryPayloads(bvh, box(0.0f, 0.0f, 0.0f, 100.0f)).empty());
}

TEST_CASE("AabbBvh.MoveProxyNoOpWhileInsideFatBox", "[AabbBvh]")
{
    AabbBvh<int> bvh{0.5f}; // fat margin 0.5
    const int proxy = bvh.createProxy(box(0.0f, 0.0f, 0.0f), 0);

    // A small move stays inside the fat box → no re-insert.
    CHECK_FALSE(bvh.moveProxy(proxy, box(0.2f, 0.0f, 0.0f)));
    // A large move leaves the fat box → re-insert.
    CHECK(bvh.moveProxy(proxy, box(5.0f, 0.0f, 0.0f)));
    // After the move, the proxy is found at its new location, not the old one.
    CHECK(queryPayloads(bvh, box(5.0f, 0.0f, 0.0f)) == std::set<int>{0});
    CHECK(queryPayloads(bvh, box(0.0f, 0.0f, 0.0f)).empty());
}

TEST_CASE("AabbBvh.DestroyRemovesFromQueries", "[AabbBvh]")
{
    AabbBvh<int> bvh{0.0f};
    const int a = bvh.createProxy(box(0.0f, 0.0f, 0.0f), 1);
    (void)bvh.createProxy(box(0.5f, 0.0f, 0.0f), 2);
    REQUIRE(queryPayloads(bvh, box(0.0f, 0.0f, 0.0f, 2.0f)) == std::set<int>{1, 2});

    bvh.destroyProxy(a);
    CHECK(bvh.size() == 1U);
    CHECK(queryPayloads(bvh, box(0.0f, 0.0f, 0.0f, 2.0f)) == std::set<int>{2});
}

TEST_CASE("AabbBvh.StaysCorrectAfterChurn", "[AabbBvh]")
{
    // Insert a grid, destroy a third, move a third — a query must still match brute force.
    AabbBvh<int> bvh{0.0f};
    std::vector<AABB> boxes;
    std::vector<int> proxies;
    std::vector<bool> alive;
    for (int i = 0; i < 30; ++i)
    {
        const int colIndex = i % 6;
        const int rowIndex = i / 6;
        boxes.push_back(
            box(static_cast<float>(colIndex) * 1.5f, static_cast<float>(rowIndex) * 1.5f, 0.0f));
        proxies.push_back(bvh.createProxy(boxes.back(), i));
        alive.push_back(true);
    }
    for (int i = 0; i < 30; i += 3)
    {
        bvh.destroyProxy(proxies[static_cast<std::size_t>(i)]);
        alive[static_cast<std::size_t>(i)] = false;
    }
    for (int i = 1; i < 30; i += 3)
    {
        boxes[static_cast<std::size_t>(i)] = box(static_cast<float>(i), 20.0f, 0.0f);
        bvh.moveProxy(proxies[static_cast<std::size_t>(i)], boxes[static_cast<std::size_t>(i)]);
    }

    const AABB q = box(-1.0f, 18.0f, -1.0f, 40.0f);
    std::set<int> expected;
    for (int i = 0; i < 30; ++i)
    {
        if (alive[static_cast<std::size_t>(i)] &&
            aabbOverlaps(boxes[static_cast<std::size_t>(i)], q))
        {
            expected.insert(i);
        }
    }
    CHECK(queryPayloads(bvh, q) == expected);
}
