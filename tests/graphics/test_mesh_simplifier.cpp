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

// A cube with per-face vertices (24 verts, 12 tris). Adjacent faces share no vertex index, so every
// face is an isolated boundary-locked quad — it should not simplify at all.
[[nodiscard]] Mesh makeSeamCube()
{
    Mesh m;
    const std::array<Vec3, 8> c{
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};
    const std::array<std::array<int, 4>, 6> faces{
        {{0, 1, 2, 3}, {5, 4, 7, 6}, {4, 5, 1, 0}, {1, 5, 6, 2}, {3, 2, 6, 7}, {4, 0, 3, 7}}};
    for (const auto& f : faces)
    {
        const auto base = static_cast<uint32_t>(m.vertices.size());
        for (const int idx : f)
        {
            m.vertices.push_back(vtx(c[static_cast<std::size_t>(idx)]));
        }
        m.indices.insert(m.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
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
    std::ranges::iota(remap, 0u);
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

TEST_CASE("MeshSimplifier.SeamCubeIsBoundaryLocked", "[MeshSimplifier]")
{
    const Mesh cube = makeSeamCube(); // 12 triangles, per-face verts
    const QuadricSimplifier simp;

    const SimplifiedMesh out = simp.simplify(cube.vertices, cube.indices, 0.1f);

    // Every edge is a boundary of an isolated quad, so nothing can collapse without folding.
    CHECK(triCount(out.indices) == 12);
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
