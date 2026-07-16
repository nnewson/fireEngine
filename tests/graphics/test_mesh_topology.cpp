#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_topology.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>

using namespace fire_engine;

namespace
{

Vertex at(const Vec3& pos, float u = 0.0f)
{
    return Vertex{pos, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{u, 0.0f}};
}

// `positions` canonical positions, each duplicated into `wedgesPer` render wedges at the SAME
// (bit-identical) position but distinct UVs — so weldByPosition welds each group and every
// canonical carries `wedgesPer` wedges (the multi-wedge case CSR bucketing must get right, at
// scale). The wedges of a position are interleaved across the array (stride = positions) so the
// buckets are NOT contiguous in original-index space — a real ordering test.
std::vector<Vertex> seamedMesh(int positions, int wedgesPer)
{
    std::vector<Vertex> verts;
    for (int w = 0; w < wedgesPer; ++w)
    {
        for (int p = 0; p < positions; ++p)
        {
            verts.push_back(at(Vec3{static_cast<float>(p), 0.0f, 0.0f}, static_cast<float>(w)));
        }
    }
    return verts;
}

} // namespace

TEST_CASE("canonicalWedgesCsr buckets match canonicalWedges byte-for-byte", "[topology]")
{
    // The CSR layout is the GPU-shaped equivalent of the vector-of-vectors the oracle keeps; the
    // two must produce IDENTICAL buckets (contents AND order) or the seam-preserving emit would
    // diverge. Checked over a hand-built seam mesh (duplicate positions ⇒ multi-wedge canonicals)
    // and a real sphere. This is what lets the parallel front carry only the CSR without touching
    // the oracle.
    SECTION("hand-built duplicate-position mesh")
    {
        const Vec3 a{0.0f, 0.0f, 0.0f};
        const Vec3 b{1.0f, 0.0f, 0.0f};
        const Vec3 c{2.0f, 0.0f, 0.0f};
        // weld: 0→0, 1→1, 2→0, 3→3, 4→0, 5→1 — canonical 0 has three wedges, 1 has two, 3 has one.
        const std::vector<Vertex> verts{at(a), at(b), at(a), at(c), at(a), at(b)};
        const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(verts);

        const auto vov = mesh_topology::canonicalWedges(weld);
        const auto csr = mesh_topology::canonicalWedgesCsr(weld);

        REQUIRE(csr.offsets.size() == weld.size() + 1);
        REQUIRE(csr.offsets.front() == 0u);
        REQUIRE(csr.offsets.back() == static_cast<std::uint32_t>(weld.size()));
        for (std::uint32_t v = 0; v < weld.size(); ++v)
        {
            const std::span<const std::uint32_t> bucket = csr.forCanonical(v);
            const std::vector<std::uint32_t> asVec(bucket.begin(), bucket.end());
            CHECK(asVec == vov[v]); // contents AND order, including the empty non-canonical slots
        }
    }
    SECTION("multi-wedge mesh at scale (interleaved wedges)")
    {
        const std::vector<Vertex> verts = seamedMesh(50, 3); // 50 canonicals, 3 wedges each
        const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(verts);
        const auto vov = mesh_topology::canonicalWedges(weld);
        const auto csr = mesh_topology::canonicalWedgesCsr(weld);

        REQUIRE(csr.offsets.size() == weld.size() + 1);
        std::size_t multiWedge = 0;
        for (std::uint32_t v = 0; v < weld.size(); ++v)
        {
            const std::span<const std::uint32_t> bucket = csr.forCanonical(v);
            const std::vector<std::uint32_t> asVec(bucket.begin(), bucket.end());
            CHECK(asVec == vov[v]);
            multiWedge += bucket.size() > 1 ? 1 : 0;
        }
        CHECK(multiWedge == 50); // every canonical genuinely carries several wedges
    }
    SECTION("empty weld")
    {
        const auto csr = mesh_topology::canonicalWedgesCsr({});
        CHECK(csr.wedges.empty());
        CHECK(csr.offsets.size() == 1u); // vertexCount + 1
        CHECK(csr.offsets.front() == 0u);
    }
}

TEST_CASE("nearestWedge breaks an exact distance tie by first (lowest-index) wedge", "[topology]")
{
    // The seam-restore tie-break: when two wedges are EXACTLY equidistant from the source, the emit
    // must be deterministic. nearestWedge keeps the first minimum (strict <), and
    // CSR/canonicalWedges store buckets ascending, so first == lowest original index. Uses
    // exactly-representable symmetric UVs (±0.5 ⇒ both squared distances 0.25f, bit-identical), so
    // the tie is exact, not approximate.
    const Vec3 pos{0.0f, 0.0f, 0.0f};
    const Vertex w0{pos, Colour3{}, Vec3{0, 0, 1}, Vec2{0.5f, 0.0f}};  // index 0
    const Vertex w1{pos, Colour3{}, Vec3{0, 0, 1}, Vec2{-0.5f, 0.0f}}; // index 1
    const Vertex source{pos, Colour3{}, Vec3{0, 0, 1}, Vec2{0.0f, 0.0f}};
    const std::vector<Vertex> verts{w0, w1, source};

    // The distances are provably identical (exact tie), so the result is decided purely by order.
    REQUIRE(mesh_topology::wedgeDistance(w0, source) == mesh_topology::wedgeDistance(w1, source));

    const std::vector<std::uint32_t> ascending{0, 1};
    CHECK(mesh_topology::nearestWedge(verts, ascending, source) == 0u); // first wins ⇒ lowest index
    const std::vector<std::uint32_t> reversed{1, 0};
    CHECK(mesh_topology::nearestWedge(verts, reversed, source) ==
          1u); // confirms it's first, not min
}
