#include <fire_engine/graphics/mesh_topology.hpp>

#include <array>
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

CanonicalWedgesCsr canonicalWedgesCsr(std::span<const std::uint32_t> weld)
{
    const auto n = static_cast<std::uint32_t>(weld.size());
    CanonicalWedgesCsr csr;
    // Counting sort by canonical id, in ascending original-vertex order — so each bucket ends up in
    // the SAME order as canonicalWedges' push_back, and a nearestWedge tie picks the identical
    // lowest-index wedge. Every original vertex belongs to exactly one canonical bucket, so the
    // flat array is exactly `n` long.
    csr.offsets.assign(n + 1, 0);
    for (std::uint32_t v = 0; v < n; ++v)
    {
        ++csr.offsets[weld[v] + 1]; // count into the NEXT slot, so the prefix sum yields starts
    }
    for (std::uint32_t c = 0; c < n; ++c)
    {
        csr.offsets[c + 1] += csr.offsets[c];
    }
    csr.wedges.resize(n);
    std::vector<std::uint32_t> cursor(csr.offsets.begin(), csr.offsets.end() - 1);
    for (std::uint32_t v = 0; v < n; ++v)
    {
        csr.wedges[cursor[weld[v]]++] = v;
    }
    return csr;
}

std::vector<std::array<std::uint32_t, 3>> canonicalFaces(std::span<const std::uint32_t> weld,
                                                         std::span<const std::uint32_t> indices)
{
    std::vector<std::array<std::uint32_t, 3>> faces;
    faces.reserve(indices.size() / 3);
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const std::array<std::uint32_t, 3> f{weld[indices[i]], weld[indices[i + 1]],
                                             weld[indices[i + 2]]};
        if (f[0] == f[1] || f[1] == f[2] || f[0] == f[2])
        {
            continue;
        }
        faces.push_back(f);
    }
    return faces;
}

} // namespace fire_engine::mesh_topology
