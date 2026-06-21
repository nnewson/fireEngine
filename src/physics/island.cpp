#include <fire_engine/physics/island.hpp>

#include <numeric>
#include <utility>

namespace fire_engine
{

namespace
{

// Union-find with path halving + union by rank. Indices are body indices.
class DisjointSet
{
public:
    explicit DisjointSet(std::size_t n)
        : parent_(n),
          rank_(n, 0)
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    [[nodiscard]] int find(int x) noexcept
    {
        while (parent_[static_cast<std::size_t>(x)] != x)
        {
            parent_[static_cast<std::size_t>(x)] =
                parent_[static_cast<std::size_t>(parent_[static_cast<std::size_t>(x)])];
            x = parent_[static_cast<std::size_t>(x)];
        }
        return x;
    }

    void unite(int a, int b) noexcept
    {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb)
        {
            return;
        }
        if (rank_[static_cast<std::size_t>(ra)] < rank_[static_cast<std::size_t>(rb)])
        {
            std::swap(ra, rb);
        }
        parent_[static_cast<std::size_t>(rb)] = ra;
        if (rank_[static_cast<std::size_t>(ra)] == rank_[static_cast<std::size_t>(rb)])
        {
            ++rank_[static_cast<std::size_t>(ra)];
        }
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

} // namespace

std::vector<Island> buildIslands(std::size_t bodyCount, const std::vector<bool>& movable,
                                 const std::vector<IslandEdge>& contactEdges,
                                 const std::vector<IslandEdge>& jointEdges)
{
    const auto bothMovable = [&](const IslandEdge& e)
    {
        return e.bodyA >= 0 && e.bodyB >= 0 && movable[static_cast<std::size_t>(e.bodyA)] &&
               movable[static_cast<std::size_t>(e.bodyB)];
    };

    DisjointSet sets{bodyCount};
    for (const IslandEdge& edge : contactEdges)
    {
        if (bothMovable(edge))
        {
            sets.unite(edge.bodyA, edge.bodyB);
        }
    }
    for (const IslandEdge& edge : jointEdges)
    {
        if (bothMovable(edge))
        {
            sets.unite(edge.bodyA, edge.bodyB);
        }
    }

    // Group movable bodies by root. Iterating body indices ascending makes the island
    // order follow each island's lowest member index (deterministic).
    std::vector<int> islandOfRoot(bodyCount, -1);
    std::vector<Island> islands;
    for (int i = 0; i < static_cast<int>(bodyCount); ++i)
    {
        if (!movable[static_cast<std::size_t>(i)])
        {
            continue;
        }
        const int root = sets.find(i);
        int& slot = islandOfRoot[static_cast<std::size_t>(root)];
        if (slot == -1)
        {
            slot = static_cast<int>(islands.size());
            islands.emplace_back();
        }
        islands[static_cast<std::size_t>(slot)].bodies.push_back(i);
    }

    // Attribute each contact/joint to the island of its movable endpoint (both
    // endpoints share an island when both are movable).
    const auto movableEndpoint = [&](const IslandEdge& e)
    {
        if (e.bodyA >= 0 && movable[static_cast<std::size_t>(e.bodyA)])
        {
            return e.bodyA;
        }
        if (e.bodyB >= 0 && movable[static_cast<std::size_t>(e.bodyB)])
        {
            return e.bodyB;
        }
        return -1;
    };

    for (int c = 0; c < static_cast<int>(contactEdges.size()); ++c)
    {
        const int node = movableEndpoint(contactEdges[static_cast<std::size_t>(c)]);
        if (node >= 0)
        {
            const int isl = islandOfRoot[static_cast<std::size_t>(sets.find(node))];
            islands[static_cast<std::size_t>(isl)].contacts.push_back(c);
        }
    }
    for (int j = 0; j < static_cast<int>(jointEdges.size()); ++j)
    {
        const int node = movableEndpoint(jointEdges[static_cast<std::size_t>(j)]);
        if (node >= 0)
        {
            const int isl = islandOfRoot[static_cast<std::size_t>(sets.find(node))];
            islands[static_cast<std::size_t>(isl)].joints.push_back(j);
        }
    }

    return islands;
}

} // namespace fire_engine
