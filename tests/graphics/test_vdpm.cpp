#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm.hpp>

using namespace fire_engine;

namespace
{

struct Mesh
{
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
};

// An n x n vertex grid (2 triangles per quad). No attribute seams, so canonical == original and the
// finest emitted faces equal the input triangles.
Mesh makeGrid(int n)
{
    Mesh m;
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const float u = static_cast<float>(x) / static_cast<float>(n - 1);
            const float v = static_cast<float>(y) / static_cast<float>(n - 1);
            m.verts.push_back(Vertex{Vec3{static_cast<float>(x), static_cast<float>(y), 0.0f},
                                     Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{u, v}});
        }
    }
    for (int y = 0; y < n - 1; ++y)
    {
        for (int x = 0; x < n - 1; ++x)
        {
            const auto a = static_cast<uint32_t>(y * n + x);
            const auto b = static_cast<uint32_t>(y * n + x + 1);
            const auto c = static_cast<uint32_t>((y + 1) * n + x);
            const auto d = static_cast<uint32_t>((y + 1) * n + x + 1);
            m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
        }
    }
    return m;
}

using FaceSet = std::vector<std::array<uint32_t, 3>>;

FaceSet normalize(FaceSet f)
{
    for (std::array<uint32_t, 3>& t : f)
    {
        std::sort(t.begin(), t.end());
    }
    std::sort(f.begin(), f.end());
    return f;
}

FaceSet facesOf(const std::vector<uint32_t>& idx)
{
    FaceSet f;
    for (std::size_t i = 0; i + 2 < idx.size(); i += 3)
    {
        f.push_back({idx[i], idx[i + 1], idx[i + 2]});
    }
    return f;
}

// A grid displaced into a bumpy surface, so collapses carry real geometric error (a flat grid is
// coplanar -> ~0 error -> nothing distance-dependent to refine).
Mesh makeBumpyGrid(int n)
{
    Mesh m = makeGrid(n);
    for (Vertex& v : m.verts)
    {
        const Vec3 p = v.position();
        const float z = 0.5f * std::sin(p.x() * 0.9f) * std::sin(p.y() * 0.9f);
        v.position(Vec3{p.x(), p.y(), z});
    }
    return m;
}

std::vector<MeshCollapse> collapsesOf(const Mesh& m)
{
    const QuadricSimplifier simp;
    return simp.collapseSequence(m.verts, m.indices);
}

} // namespace

TEST_CASE("ActiveFront: fully refined reproduces the finest index buffer", "[vdpm]")
{
    const Mesh m = makeGrid(9);
    const auto collapses = collapsesOf(m);
    REQUIRE(!collapses.empty());

    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapses);
    front.refineAll();
    // Grid has no seams, so canonical faces == the input triangles.
    CHECK(normalize(front.emitActiveCanonical()) == normalize(facesOf(m.indices)));
}

TEST_CASE("ActiveFront: fully refined emitActiveIndices restores the original render wedges",
          "[vdpm]")
{
    const Mesh grid = makeGrid(9);
    ActiveFront f1 = ActiveFront::build(grid.verts, grid.indices, collapsesOf(grid));
    f1.refineAll();
    // No-seam grid: fully-refined emit is the identity index buffer.
    CHECK(normalize(facesOf(f1.emitActiveIndices(grid.verts, grid.indices))) ==
          normalize(facesOf(grid.indices)));
}

TEST_CASE("ActiveFront: emitActiveIndices restores per-corner wedges across a UV seam", "[vdpm]")
{
    // A quad split into two triangles sharing an edge that is a UV seam: the two shared-position
    // vertices are duplicated with *different* UVs. Emit must return each corner to its own UV
    // wedge (nearest by attribute), not snap both sides to a single canonical vertex.
    Mesh m;
    m.verts = {
        Vertex{Vec3{0, 0, 0}, Colour3{}, Vec3{0, 0, 1}, Vec2{0.0f, 0.0f}},
        Vertex{Vec3{1, 0, 0}, Colour3{}, Vec3{0, 0, 1}, Vec2{1.0f, 0.0f}}, // pos(1,0) uvA
        Vertex{Vec3{0, 1, 0}, Colour3{}, Vec3{0, 0, 1}, Vec2{0.0f, 1.0f}}, // pos(0,1) uvA
        Vertex{Vec3{1, 0, 0}, Colour3{}, Vec3{0, 0, 1}, Vec2{0.0f, 0.5f}}, // pos(1,0) uvB (seam)
        Vertex{Vec3{0, 1, 0}, Colour3{}, Vec3{0, 0, 1}, Vec2{0.5f, 0.0f}}, // pos(0,1) uvB (seam)
        Vertex{Vec3{1, 1, 0}, Colour3{}, Vec3{0, 0, 1}, Vec2{1.0f, 1.0f}},
    };
    m.indices = {0, 1, 2, 3, 5, 4};

    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapsesOf(m));
    // Each corner's own wedge is nearest to itself, so emit reproduces the input exactly.
    CHECK(normalize(facesOf(front.emitActiveIndices(m.verts, m.indices))) ==
          normalize(facesOf(m.indices)));
}

TEST_CASE("ActiveFront: refineAll then coarsenAll round-trips to the coarsest front", "[vdpm]")
{
    const Mesh m = makeGrid(9);
    const auto collapses = collapsesOf(m);

    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapses);
    const FaceSet coarsest = normalize(front.emitActiveCanonical());
    REQUIRE(!coarsest.empty());

    front.refineAll();
    CHECK(normalize(front.emitActiveCanonical()).size() == normalize(facesOf(m.indices)).size());
    front.coarsenAll();
    CHECK(normalize(front.emitActiveCanonical()) == coarsest);
}

TEST_CASE("ActiveFront: every emitted face is a valid, all-active triangle", "[vdpm]")
{
    const Mesh m = makeGrid(9);
    const auto collapses = collapsesOf(m);

    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapses);
    // Partially refine (every other split we can) to reach a mixed front.
    for (uint32_t i = 0; i < front.forest().splits.size(); i += 2)
    {
        front.refine(i);
    }
    for (const std::array<uint32_t, 3>& t : front.emitActiveCanonical())
    {
        CHECK(t[0] != t[1]);
        CHECK(t[1] != t[2]);
        CHECK(t[0] != t[2]);
        CHECK(front.active(t[0]));
        CHECK(front.active(t[1]));
        CHECK(front.active(t[2]));
    }
}

TEST_CASE("ActiveFront: a single legal refine/coarsen round-trips", "[vdpm]")
{
    const Mesh m = makeGrid(9);
    const auto collapses = collapsesOf(m);

    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapses);
    const FaceSet before = normalize(front.emitActiveCanonical());

    // In the coarsest front at least one split (the coarsest collapse) is legally refinable.
    uint32_t legal = UINT32_MAX;
    for (uint32_t i = 0; i < front.forest().splits.size(); ++i)
    {
        if (front.refine(i))
        {
            legal = i;
            break;
        }
    }
    REQUIRE(legal != UINT32_MAX);
    CHECK(front.refined(legal));
    CHECK(front.coarsen(legal));
    CHECK(normalize(front.emitActiveCanonical()) == before);
}

TEST_CASE("ActiveFront: illegal ops are rejected without mutating the front", "[vdpm]")
{
    const Mesh m = makeGrid(9);
    const auto collapses = collapsesOf(m);

    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapses);
    // Nothing refined yet: any coarsen is illegal.
    CHECK_FALSE(front.coarsen(0));
    front.refineAll();
    // Everything refined: any refine is illegal (already refined).
    CHECK_FALSE(front.refine(0));
    // The fixpoint coarsenAll must still fully drain despite the leaf-only coarsen constraint.
    front.coarsenAll();
    for (uint32_t i = 0; i < front.forest().splits.size(); ++i)
    {
        CHECK_FALSE(front.refined(i));
    }
    // An out-of-range split index is rejected, not a crash.
    CHECK_FALSE(front.refine(static_cast<uint32_t>(front.forest().splits.size())));
    CHECK_FALSE(front.coarsen(static_cast<uint32_t>(front.forest().splits.size())));
}

TEST_CASE("ActiveFront: refineForView refines nearer views more than distant ones", "[vdpm]")
{
    const Mesh m = makeBumpyGrid(17);
    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapsesOf(m));
    const Mat4 world = Mat4::identity();
    const float projScaleY = 1.0f;
    const float viewportHeight = 1000.0f;
    const float budget = 2.0f;
    const float noSilhouette = 0.0f;

    front.refineForView(m.verts, world, Vec3{8.0f, 8.0f, 6.0f}, projScaleY, viewportHeight, budget,
                        noSilhouette);
    const std::size_t nearCount = front.emitActiveCanonical().size();

    front.refineForView(m.verts, world, Vec3{8.0f, 8.0f, 400.0f}, projScaleY, viewportHeight,
                        budget, noSilhouette);
    const std::size_t farCount = front.emitActiveCanonical().size();

    CHECK(nearCount > farCount); // closer view resolves more triangles
    CHECK(farCount >= 2);        // never below the coarsest
}

TEST_CASE("ActiveFront: boundary splits are recorded with an invalid vr", "[vdpm]")
{
    const Mesh m = makeGrid(9);
    const auto collapses = collapsesOf(m);

    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapses);
    // A grid's border edges are boundary (one adjacent face), so some split carries vr == invalid.
    bool hasBoundary = false;
    for (const VertexSplit& s : front.forest().splits)
    {
        if (s.vr == kInvalidVertex)
        {
            hasBoundary = true;
            break;
        }
    }
    CHECK(hasBoundary);
}

TEST_CASE("buildVertexForest: welds coincident duplicated vertices back into one topology",
          "[vdpm]")
{
    // Per-corner-duplicated grid: every triangle gets its own three vertices at shared positions
    // (the pathological seam case). Welding must recover the shared connectivity so the forest
    // simplifies comparably to the connected grid; left split, each triangle is a boundary-locked
    // island.
    const Mesh shared = makeGrid(9);
    Mesh dup;
    for (const uint32_t idx : shared.indices)
    {
        dup.indices.push_back(static_cast<uint32_t>(dup.verts.size()));
        dup.verts.push_back(shared.verts[idx]);
    }

    const VertexForest sharedForest =
        buildVertexForest(shared.verts, shared.indices, collapsesOf(shared));
    const VertexForest dupForest = buildVertexForest(dup.verts, dup.indices, collapsesOf(dup));

    CHECK(!sharedForest.splits.empty());
    CHECK(!dupForest.splits.empty());
    // Welding recovered connectivity, so the duplicated mesh collapses in the same ballpark.
    CHECK(dupForest.splits.size() >= sharedForest.splits.size() / 2);
}
