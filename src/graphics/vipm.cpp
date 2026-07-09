#include <fire_engine/graphics/vipm.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace fire_engine
{

namespace
{

struct PosKey
{
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    bool operator==(const PosKey&) const noexcept = default;
};

struct PosKeyHash
{
    std::size_t operator()(const PosKey& k) const noexcept
    {
        std::size_t h = k.x;
        h = h * 1000003u ^ k.y;
        h = h * 1000003u ^ k.z;
        return h;
    }
};

[[nodiscard]] PosKey posKey(const Vec3& p) noexcept
{
    return PosKey{std::bit_cast<std::uint32_t>(p.x()), std::bit_cast<std::uint32_t>(p.y()),
                  std::bit_cast<std::uint32_t>(p.z())};
}

[[nodiscard]] float wedgeDistance(const Vertex& a, const Vertex& b) noexcept
{
    const Vec2 uv0 = a.texCoord() - b.texCoord();
    const Vec2 uv1 = a.texCoord1() - b.texCoord1();
    const Vec3 normal = a.normal() - b.normal();
    const Vec4 tangent = a.tangent() - b.tangent();
    return Vec2::dotProduct(uv0, uv0) + Vec2::dotProduct(uv1, uv1) +
           0.25f * Vec3::dotProduct(normal, normal) + 0.25f * Vec4::dotProduct(tangent, tangent);
}

[[nodiscard]] std::uint32_t nearestWedge(std::span<const Vertex> vertices,
                                         std::span<const std::uint32_t> wedges,
                                         const Vertex& source) noexcept
{
    std::uint32_t best = wedges.empty() ? 0u : wedges.front();
    float bestDist = std::numeric_limits<float>::max();
    for (const std::uint32_t w : wedges)
    {
        const float dist = wedgeDistance(vertices[w], source);
        if (dist < bestDist)
        {
            bestDist = dist;
            best = w;
        }
    }
    return best;
}

} // namespace

std::vector<MorphVertex> buildVipmMorphData(std::span<const Vertex> vertices,
                                            std::span<const MeshCollapse> collapses,
                                            std::span<const ProgressiveLod> lods)
{
    const std::size_t n = vertices.size();

    // A vertex geomorphs toward the full render identity of the resolved survivor wedge, so the
    // topology swap does not change position, normal, tangent, or either UV set.
    auto morphToward = [&vertices](uint32_t target, float level)
    {
        const Vertex& t = vertices[target];
        return MorphVertex{t.position(), level,        t.normal(),   0.0f,
                           t.tangent(),  t.texCoord(), t.texCoord1()};
    };

    // Default: never collapses — target is the vertex itself, sentinel level (never morphs).
    std::vector<MorphVertex> morph(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        morph[i] = morphToward(static_cast<uint32_t>(i), kVipmNeverCollapses);
    }

    std::vector<uint32_t> weld(n);
    std::vector<std::vector<uint32_t>> canonicalWedges(n);
    std::unordered_map<PosKey, uint32_t, PosKeyHash> weldMap;
    weldMap.reserve(n);
    for (std::uint32_t v = 0; v < n; ++v)
    {
        weld[v] = weldMap.try_emplace(posKey(vertices[v].position()), v).first->second;
        canonicalWedges[weld[v]].push_back(v);
    }

    // Union-find over the position-welded vertices. Subset placement means a collapse merges
    // `removed` into `kept` and `kept` keeps its original position, so after replaying each exact
    // LOD cut, find(v) is v's surviving representative — a vertex still present at that cut — and
    // its position is simply that vertex's (unchanged) original position.
    // Index-fill by hand (not std::iota): std::ranges::iota isn't portable across the Apple-Clang
    // libc++ we build against, and plain std::iota trips modernize-use-ranges under the
    // now-blocking clang-tidy gate.
    std::vector<uint32_t> parent(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        parent[i] = static_cast<uint32_t>(i);
    }
    auto find = [&parent](uint32_t v)
    {
        while (parent[v] != v)
        {
            parent[v] = parent[parent[v]]; // path halving
            v = parent[v];
        }
        return v;
    };

    std::size_t collapseIdx = 0;
    for (std::size_t level = 1; level < lods.size(); ++level)
    {
        const std::size_t targetCollapseCount =
            std::min(lods[level].collapseCount, collapses.size());
        std::vector<uint32_t> rootBefore(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            rootBefore[i] = find(weld[i]);
        }

        while (collapseIdx < targetCollapseCount)
        {
            const MeshCollapse& c = collapses[collapseIdx];
            parent[find(c.removed)] = find(c.kept);
            ++collapseIdx;
        }

        for (std::uint32_t v = 0; v < n; ++v)
        {
            if (morph[v].collapseLevel != kVipmNeverCollapses)
            {
                continue;
            }
            const uint32_t rootAfter = find(weld[v]);
            if (rootBefore[v] == rootAfter)
            {
                continue;
            }
            const uint32_t targetWedge =
                nearestWedge(vertices, canonicalWedges[rootAfter], vertices[v]);
            morph[v] = morphToward(targetWedge, static_cast<float>(level));
        }
    }

    return morph;
}

} // namespace fire_engine
