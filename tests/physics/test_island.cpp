#include <fire_engine/physics/island.hpp>

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

using fire_engine::buildIslands;
using fire_engine::Island;
using fire_engine::IslandEdge;

namespace
{

// Index of the island containing dynamic body `b`, or -1.
int islandOf(const std::vector<Island>& islands, int b)
{
    for (int i = 0; i < static_cast<int>(islands.size()); ++i)
    {
        if (std::ranges::find(islands[static_cast<std::size_t>(i)].bodies, b) !=
            islands[static_cast<std::size_t>(i)].bodies.end())
        {
            return i;
        }
    }
    return -1;
}

} // namespace

TEST_CASE("Island.IsolatedDynamicBodiesEachFormASingleton", "[Island]")
{
    // Three dynamic bodies, no edges → three singleton islands.
    const std::vector<bool> dynamic{true, true, true};
    const auto islands = buildIslands(3, dynamic, {}, {});

    REQUIRE(islands.size() == 3U);
    for (const Island& island : islands)
    {
        CHECK(island.bodies.size() == 1U);
        CHECK(island.contacts.empty());
        CHECK(island.joints.empty());
    }
}

TEST_CASE("Island.ContactMergesTwoDynamicBodies", "[Island]")
{
    const std::vector<bool> dynamic{true, true, true};
    // Bodies 0 and 1 share a contact; body 2 is alone.
    const std::vector<IslandEdge> contacts{{0, 1}};
    const auto islands = buildIslands(3, dynamic, contacts, {});

    REQUIRE(islands.size() == 2U);
    CHECK(islandOf(islands, 0) == islandOf(islands, 1));
    CHECK(islandOf(islands, 2) != islandOf(islands, 0));

    // The contact is attributed to the merged island.
    const int merged = islandOf(islands, 0);
    REQUIRE(merged >= 0);
    CHECK(islands[static_cast<std::size_t>(merged)].contacts == std::vector<int>{0});
}

TEST_CASE("Island.JointMergesTwoDynamicBodies", "[Island]")
{
    const std::vector<bool> dynamic{true, true};
    const std::vector<IslandEdge> joints{{0, 1}};
    const auto islands = buildIslands(2, dynamic, {}, joints);

    REQUIRE(islands.size() == 1U);
    CHECK(islands[0].bodies == std::vector<int>{0, 1});
    CHECK(islands[0].joints == std::vector<int>{0});
}

TEST_CASE("Island.StaticBoundaryDoesNotMergeIslands", "[Island]")
{
    // Body 1 is static (the floor). Two dynamic boxes (0 and 2) each rest on it but
    // must stay in separate islands — the floor is a boundary, not a connector.
    const std::vector<bool> dynamic{true, false, true};
    const std::vector<IslandEdge> contacts{{0, 1}, {2, 1}};
    const auto islands = buildIslands(3, dynamic, contacts, {});

    REQUIRE(islands.size() == 2U);
    CHECK(islandOf(islands, 0) != islandOf(islands, 2));
    // Each contact is attributed to its dynamic endpoint's island.
    CHECK(islands[static_cast<std::size_t>(islandOf(islands, 0))].contacts == std::vector<int>{0});
    CHECK(islands[static_cast<std::size_t>(islandOf(islands, 2))].contacts == std::vector<int>{1});
    // The static body is never an island member.
    CHECK(islandOf(islands, 1) == -1);
}

TEST_CASE("Island.ChainOfContactsFormsOneIsland", "[Island]")
{
    // A stack: 0-1, 1-2, 2-3 all dynamic → one island of four.
    const std::vector<bool> dynamic{true, true, true, true};
    const std::vector<IslandEdge> contacts{{0, 1}, {1, 2}, {2, 3}};
    const auto islands = buildIslands(4, dynamic, contacts, {});

    REQUIRE(islands.size() == 1U);
    CHECK(islands[0].bodies == std::vector<int>{0, 1, 2, 3});
    CHECK(islands[0].contacts == std::vector<int>{0, 1, 2});
}

TEST_CASE("Island.OrderingIsDeterministic", "[Island]")
{
    // Islands ordered by lowest member index; members ascending. Edges given out of
    // order must not change the partition or its ordering.
    const std::vector<bool> dynamic{true, true, true, true};
    const std::vector<IslandEdge> contacts{{3, 1}, {2, 0}};
    const auto a = buildIslands(4, dynamic, contacts, {});
    const auto b = buildIslands(4, dynamic, {{2, 0}, {3, 1}}, {});

    REQUIRE(a.size() == 2U);
    // First island owns body 0 (lowest index), second owns body 1.
    CHECK(a[0].bodies == std::vector<int>{0, 2});
    CHECK(a[1].bodies == std::vector<int>{1, 3});
    // Same partition regardless of edge order.
    CHECK(b[0].bodies == a[0].bodies);
    CHECK(b[1].bodies == a[1].bodies);
}
