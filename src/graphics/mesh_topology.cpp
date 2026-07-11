#include <fire_engine/graphics/mesh_topology.hpp>

#include <bit>
#include <cstddef>
#include <limits>
#include <unordered_map>

#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine::mesh_topology
{

namespace
{

// Exact-position key: the raw float bits of (x, y, z), so two vertices weld iff their positions are
// bit-for-bit equal (the way glTF authors a seam duplicate — same position, different attributes).
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

} // namespace

std::vector<std::uint32_t> weldByPosition(std::span<const Vertex> vertices)
{
    const auto n = static_cast<std::uint32_t>(vertices.size());
    std::vector<std::uint32_t> weld(n);
    std::unordered_map<PosKey, std::uint32_t, PosKeyHash> weldMap;
    weldMap.reserve(n);
    for (std::uint32_t v = 0; v < n; ++v)
    {
        weld[v] = weldMap.try_emplace(posKey(vertices[v].position()), v).first->second;
    }
    return weld;
}

float wedgeDistance(const Vertex& a, const Vertex& b) noexcept
{
    const Vec2 uv0 = a.texCoord() - b.texCoord();
    const Vec2 uv1 = a.texCoord1() - b.texCoord1();
    const Vec3 nrm = a.normal() - b.normal();
    const Vec4 tan = a.tangent() - b.tangent();
    return Vec2::dotProduct(uv0, uv0) + Vec2::dotProduct(uv1, uv1) +
           0.25f * Vec3::dotProduct(nrm, nrm) + 0.25f * Vec4::dotProduct(tan, tan);
}

std::uint32_t nearestWedge(std::span<const Vertex> vertices, std::span<const std::uint32_t> wedges,
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

std::vector<std::vector<std::uint32_t>> canonicalWedges(std::span<const std::uint32_t> weld)
{
    std::vector<std::vector<std::uint32_t>> wedges(weld.size());
    for (std::uint32_t v = 0; v < weld.size(); ++v)
    {
        wedges[weld[v]].push_back(v);
    }
    return wedges;
}

} // namespace fire_engine::mesh_topology
