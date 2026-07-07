#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vertex.hpp>

using namespace fire_engine;

namespace
{

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

[[nodiscard]] Vertex vtx(Vec3 p)
{
    return Vertex{p, Colour3{}, Vec3{}, Vec2{}};
}

// A flat (n×n cell) grid in the z=0 plane, welded. Perfectly coplanar, so QEM should collapse the
// whole interior + straight boundaries to the four corners at ~zero error.
[[nodiscard]] Mesh makeFlatGrid(int n)
{
    Mesh m;
    for (int i = 0; i <= n; ++i)
    {
        for (int j = 0; j <= n; ++j)
        {
            m.vertices.push_back(vtx(Vec3{static_cast<float>(j), static_cast<float>(i), 0.0f}));
        }
    }
    auto at = [n](int i, int j) { return static_cast<uint32_t>(i * (n + 1) + j); };
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            m.indices.insert(m.indices.end(), {at(i, j), at(i, j + 1), at(i + 1, j + 1)});
            m.indices.insert(m.indices.end(), {at(i, j), at(i + 1, j + 1), at(i + 1, j)});
        }
    }
    return m;
}

// A welded UV sphere (single-vertex poles, seam wrapped). Closed, curved — collapses with bounded
// error.
[[nodiscard]] Mesh makeUvSphere(int stacks, int slices)
{
    Mesh m;
    const float pi = std::numbers::pi_v<float>;
    const uint32_t north = 0;
    m.vertices.push_back(vtx(Vec3{0.0f, 1.0f, 0.0f}));
    for (int i = 1; i < stacks; ++i)
    {
        const float theta = pi * static_cast<float>(i) / static_cast<float>(stacks);
        for (int j = 0; j < slices; ++j)
        {
            const float phi = 2.0f * pi * static_cast<float>(j) / static_cast<float>(slices);
            m.vertices.push_back(vtx(Vec3{std::sin(theta) * std::cos(phi), std::cos(theta),
                                          std::sin(theta) * std::sin(phi)}));
        }
    }
    const auto south = static_cast<uint32_t>(m.vertices.size());
    m.vertices.push_back(vtx(Vec3{0.0f, -1.0f, 0.0f}));

    auto ring = [slices](int i, int j)
    { return static_cast<uint32_t>(1 + (i - 1) * slices + (j % slices)); };
    for (int j = 0; j < slices; ++j)
    {
        m.indices.insert(m.indices.end(), {north, ring(1, j), ring(1, j + 1)});
    }
    for (int i = 1; i < stacks - 1; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            m.indices.insert(m.indices.end(), {ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)});
            m.indices.insert(m.indices.end(), {ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)});
        }
    }
    for (int j = 0; j < slices; ++j)
    {
        m.indices.insert(m.indices.end(), {south, ring(stacks - 1, j + 1), ring(stacks - 1, j)});
    }
    return m;
}

// A flat grid where every triangle carries its own three vertices (positions duplicated at every
// shared edge) — the way glTF splits vertices at UV/normal seams. Left un-welded this is all
// boundary and cannot collapse; the simplifier must weld coincident positions to simplify it.
[[nodiscard]] Mesh makeShatteredGrid(int n)
{
    const Mesh welded = makeFlatGrid(n);
    Mesh m;
    for (std::size_t i = 0; i + 3 <= welded.indices.size(); i += 3)
    {
        const auto base = static_cast<uint32_t>(m.vertices.size());
        for (int k = 0; k < 3; ++k)
        {
            m.vertices.push_back(welded.vertices[welded.indices[i + k]]);
        }
        m.indices.insert(m.indices.end(), {base, base + 1, base + 2});
    }
    return m;
}

[[nodiscard]] std::size_t triCount(const std::vector<uint32_t>& indices)
{
    return indices.size() / 3;
}

// Replay a full collapse sequence via union-find and emit the surviving triangles, normalised for
// comparison (each triangle's vertices sorted, then the triangle list sorted).
[[nodiscard]] std::vector<std::array<uint32_t, 3>> replay(std::size_t vertexCount,
                                                          const std::vector<uint32_t>& indices,
                                                          const std::vector<MeshCollapse>& seq)
{
    std::vector<uint32_t> remap(vertexCount);
    // std::iota (C++11), not std::ranges::iota — the ranges variant (C++23) is missing from some
    // Apple Clang libc++ versions on GitHub's macos-latest runners, so it fails intermittently
    // there.
    std::iota(remap.begin(), remap.end(), 0u);
    auto resolve = [&](uint32_t v)
    {
        while (remap[v] != v)
        {
            v = remap[v];
        }
        return v;
    };
    for (const auto& c : seq)
    {
        remap[resolve(c.removed)] = resolve(c.kept);
    }
    std::vector<std::array<uint32_t, 3>> out;
    for (std::size_t i = 0; i + 3 <= indices.size(); i += 3)
    {
        std::array<uint32_t, 3> t{resolve(indices[i]), resolve(indices[i + 1]),
                                  resolve(indices[i + 2])};
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2])
        {
            continue;
        }
        std::ranges::sort(t);
        out.push_back(t);
    }
    std::ranges::sort(out);
    return out;
}

[[nodiscard]] bool noDegenerateOrOutOfRange(const std::vector<uint32_t>& indices,
                                            std::size_t vCount)
{
    for (std::size_t i = 0; i + 3 <= indices.size(); i += 3)
    {
        const uint32_t a = indices[i];
        const uint32_t b = indices[i + 1];
        const uint32_t c = indices[i + 2];
        if (a >= vCount || b >= vCount || c >= vCount)
        {
            return false;
        }
        if (a == b || b == c || a == c)
        {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("MeshSimplifier.FlatGridCollapsesToPlaneAtZeroError", "[MeshSimplifier]")
{
    const Mesh grid = makeFlatGrid(8); // 128 triangles
    const QuadricSimplifier simp;

    // Ratio below 2/128 forces maximal collapse; a flat quad bottoms out at two triangles.
    const SimplifiedMesh out = simp.simplify(grid.vertices, grid.indices, 0.01f);

    // Coplanar interior + straight borders collapse to the four corners → two triangles, ~no error.
    CHECK(triCount(out.indices) == 2);
    CHECK(out.error < 1e-3f);
    CHECK(noDegenerateOrOutOfRange(out.indices, grid.vertices.size()));
}

TEST_CASE("MeshSimplifier.UvSphereSimplifiesWithBoundedError", "[MeshSimplifier]")
{
    const Mesh sphere = makeUvSphere(16, 24);
    const std::size_t original = triCount(sphere.indices);
    const QuadricSimplifier simp;

    const SimplifiedMesh out = simp.simplify(sphere.vertices, sphere.indices, 0.25f);

    CHECK(triCount(out.indices) < original);
    CHECK(triCount(out.indices) >= original / 5); // roughly hit the target, not over-collapsed
    CHECK(out.error < 0.1f);                      // unit sphere: squared deviation stays small
    CHECK(noDegenerateOrOutOfRange(out.indices, sphere.vertices.size()));

    // Output indices are a strict subset of the originals (subset placement — no moved vertices).
    std::vector<uint8_t> used(sphere.vertices.size(), 0);
    for (const std::size_t i : sphere.indices)
    {
        used[i] = 1;
    }
    for (const uint32_t idx : out.indices)
    {
        CHECK(used[idx] == 1);
    }
}

TEST_CASE("MeshSimplifier.WeldsCoincidentSeamVertices", "[MeshSimplifier]")
{
    const Mesh shattered = makeShatteredGrid(8); // 128 tris, every vertex duplicated
    const std::size_t original = triCount(shattered.indices);
    const QuadricSimplifier simp;

    const SimplifiedMesh out = simp.simplify(shattered.vertices, shattered.indices, 0.05f);

    // Without position welding this all-boundary mesh couldn't collapse at all; with it, the flat
    // grid coarsens just like its welded form.
    CHECK(original == 128);
    CHECK(triCount(out.indices) < 20);
    CHECK(out.error < 1e-3f);
    CHECK(noDegenerateOrOutOfRange(out.indices, shattered.vertices.size()));
}

TEST_CASE("MeshSimplifier.PreservesPerCornerUvAcrossSeam", "[MeshSimplifier]")
{
    // Two triangles meet at the position edge p0-p1 but carry different UVs on each side — a UV
    // seam. Position welding collapses p0's two wedges to one canonical for connectivity, but the
    // emit must give each triangle back its own UV rather than smearing one across the seam.
    auto v = [](Vec3 p, Vec2 uv) { return Vertex{p, Colour3{}, Vec3{}, uv}; };
    Mesh m;
    m.vertices = {
        v({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}),  // 0: p0, triangle A
        v({1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}),  // 1: p1, triangle A
        v({0.5f, 1.0f, 0.0f}, {0.5f, 1.0f}),  // 2
        v({0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}),  // 3: p0, triangle B (same position as 0, other UV)
        v({1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}),  // 4: p1, triangle B
        v({0.5f, -1.0f, 0.0f}, {0.5f, 0.0f}), // 5
    };
    m.indices = {0, 1, 2, 3, 4, 5};
    const QuadricSimplifier simp;

    const SimplifiedMesh out =
        simp.simplify(m.vertices, m.indices, 1.0f); // no collapse; test the emit
    REQUIRE(out.indices.size() == 6);

    // Triangle A's p0 corner keeps UV.t = 0; triangle B's keeps UV.t = 1 — the seam is not merged.
    CHECK(m.vertices[out.indices[0]].texCoord().t() == 0.0f);
    CHECK(m.vertices[out.indices[3]].texCoord().t() == 1.0f);
}

TEST_CASE("MeshSimplifier.IsDeterministic", "[MeshSimplifier]")
{
    const Mesh sphere = makeUvSphere(12, 16);
    const QuadricSimplifier simp;

    const SimplifiedMesh a = simp.simplify(sphere.vertices, sphere.indices, 0.3f);
    const SimplifiedMesh b = simp.simplify(sphere.vertices, sphere.indices, 0.3f);

    CHECK(a.indices == b.indices);
}

TEST_CASE("MeshSimplifier.CollapseSequenceReplayMatchesSimplify", "[MeshSimplifier]")
{
    const Mesh sphere = makeUvSphere(10, 12);
    const QuadricSimplifier simp;

    const std::vector<MeshCollapse> seq = simp.collapseSequence(sphere.vertices, sphere.indices);
    const SimplifiedMesh full = simp.simplify(sphere.vertices, sphere.indices, 0.0f);

    // Replaying the recorded stream reproduces the fully-collapsed mesh — the sequence is faithful.
    const auto replayed = replay(sphere.vertices.size(), sphere.indices, seq);

    std::vector<std::array<uint32_t, 3>> direct;
    for (std::size_t i = 0; i + 3 <= full.indices.size(); i += 3)
    {
        std::array<uint32_t, 3> t{full.indices[i], full.indices[i + 1], full.indices[i + 2]};
        std::ranges::sort(t);
        direct.push_back(t);
    }
    std::ranges::sort(direct);

    CHECK(replayed == direct);
    CHECK_FALSE(seq.empty());
}
