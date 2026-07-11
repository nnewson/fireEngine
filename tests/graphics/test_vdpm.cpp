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

// A UV sphere with radial normals — clear front/back hemispheres for the back-face-gate test.
Mesh makeUvSphere(int rings, int segments)
{
    Mesh m;
    constexpr float pi = 3.14159265f;
    for (int r = 0; r <= rings; ++r)
    {
        const float lat = pi * (static_cast<float>(r) / static_cast<float>(rings) - 0.5f);
        for (int s = 0; s <= segments; ++s)
        {
            const float lon = 2.0f * pi * static_cast<float>(s) / static_cast<float>(segments);
            const Vec3 nrm{std::cos(lat) * std::cos(lon), std::sin(lat),
                           std::cos(lat) * std::sin(lon)};
            m.verts.push_back(Vertex{nrm, Colour3{}, nrm,
                                     Vec2{static_cast<float>(s) / static_cast<float>(segments),
                                          static_cast<float>(r) / static_cast<float>(rings)}});
        }
    }
    const int stride = segments + 1;
    for (int r = 0; r < rings; ++r)
    {
        for (int s = 0; s < segments; ++s)
        {
            const auto a = static_cast<uint32_t>(r * stride + s);
            const auto b = static_cast<uint32_t>(r * stride + s + 1);
            const auto c = static_cast<uint32_t>((r + 1) * stride + s);
            const auto d = static_cast<uint32_t>((r + 1) * stride + s + 1);
            m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
        }
    }
    return m;
}

// A flat grid with a deliberately nonlinear (quadratic) UV map: geometry stays coplanar (geometric
// δ ~0) but the parameterisation stretches, so the UV channel must accumulate.
Mesh makeFlatNonlinearUvGrid(int n)
{
    Mesh m = makeGrid(n);
    for (Vertex& v : m.verts)
    {
        const Vec2 uv = v.texCoord();
        v.texCoord(Vec2{uv.s() * uv.s(), uv.t() * uv.t()});
    }
    return m;
}

// A flat grid whose per-vertex normals fan smoothly across it, with geometry left coplanar and UV
// left affine. Only the shading channel can see it: the geometry (point-to-plane) and UV
// (affine-exact) channels both read ~0. Models a smooth-shaded curve authored flat — exactly the
// lighting error a coarse collapse flattens without moving a vertex off-plane. (Vertex normals
// don't steer the simplifier — the R⁵ quadric is position+UV and the flip veto uses face normals —
// so the collapse stream matches makeGrid; only the measured deviation differs.)
Mesh makeFlatFannedNormalGrid(int n)
{
    Mesh m = makeGrid(n);
    for (Vertex& v : m.verts)
    {
        const float ang = 0.08f * v.position().x(); // 0 .. ~1.3 rad across the grid
        v.normal(Vec3{std::sin(ang), 0.0f, std::cos(ang)});
    }
    return m;
}

// A flat grid with a constant normal but a tangent that fans across it (w = +1 handedness). Only
// the tangent channel sees it: geometry is coplanar, UV affine, and the shading normal constant.
// The normal-map-frame-drift analogue of makeFlatFannedNormalGrid.
Mesh makeFlatFannedTangentGrid(int n)
{
    Mesh m = makeGrid(n);
    for (Vertex& v : m.verts)
    {
        const float ang = 0.08f * v.position().x(); // tangent swings in the xy-plane
        v.tangent(Vec4{std::cos(ang), std::sin(ang), 0.0f, 1.0f});
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
                        noSilhouette, 2.0f, 0.0f, 0.0f, 0.0f);
    const std::size_t nearCount = front.emitActiveCanonical().size();

    front.refineForView(m.verts, world, Vec3{8.0f, 8.0f, 400.0f}, projScaleY, viewportHeight,
                        budget, noSilhouette, 2.0f, 0.0f, 0.0f, 0.0f);
    const std::size_t farCount = front.emitActiveCanonical().size();

    CHECK(nearCount > farCount); // closer view resolves more triangles
    CHECK(farCount >= 2);        // never below the coarsest
}

TEST_CASE("refineForView: back-face suppression coarsens the hidden hemisphere", "[vdpm]")
{
    const Mesh m = makeUvSphere(24, 32);
    ActiveFront front = ActiveFront::build(m.verts, m.indices, collapsesOf(m));
    const Mat4 world = Mat4::identity();
    const float projScaleY = 1.0f;
    const float viewportHeight = 1000.0f;
    const float budget = 2.0f;
    const float noSilhouette = 0.0f;
    const Vec3 cam{0.0f, 0.0f, 4.0f};

    // Threshold 2.0 never trips (facing >= -1); 0.5 suppresses the clearly back-facing hemisphere.
    front.refineForView(m.verts, world, cam, projScaleY, viewportHeight, budget, noSilhouette, 2.0f,
                        0.0f, 0.0f, 0.0f);
    const std::size_t off = front.emitActiveCanonical().size();
    front.refineForView(m.verts, world, cam, projScaleY, viewportHeight, budget, noSilhouette, 0.5f,
                        0.0f, 0.0f, 0.0f);
    const std::size_t on = front.emitActiveCanonical().size();
    CHECK(on < off); // the hidden hemisphere's discretionary detail is dropped
    CHECK(on >= 2);

    // Suppression must not break the distance criterion: a near view still resolves more than a
    // far.
    front.refineForView(m.verts, world, Vec3{0.0f, 0.0f, 2.5f}, projScaleY, viewportHeight, budget,
                        noSilhouette, 0.5f, 0.0f, 0.0f, 0.0f);
    const std::size_t nearCount = front.emitActiveCanonical().size();
    front.refineForView(m.verts, world, Vec3{0.0f, 0.0f, 60.0f}, projScaleY, viewportHeight, budget,
                        noSilhouette, 0.5f, 0.0f, 0.0f, 0.0f);
    const std::size_t farCount = front.emitActiveCanonical().size();
    CHECK(nearCount > farCount);
}

TEST_CASE("Deviation radius is ~0 on a flat mesh and accumulates on a curved one", "[vdpm]")
{
    // Flat grid: every collapse is coplanar, so the point-to-plane dev is ~0 and nothing
    // accumulates.
    float flatMax = 0.0f;
    for (const MeshCollapse& c : collapsesOf(makeGrid(17)))
    {
        flatMax = std::max(flatMax, c.deviationRadius);
    }
    CHECK(flatMax < 1e-3f);

    // Bumpy grid (amplitude 0.5): real curvature, so the coarse approximation deviates and the
    // radius accumulates to a meaningful, bounded fraction of the mesh.
    float bumpMax = 0.0f;
    for (const MeshCollapse& c : collapsesOf(makeBumpyGrid(17)))
    {
        bumpMax = std::max(bumpMax, c.deviationRadius);
    }
    CHECK(bumpMax > 0.1f);
    CHECK(bumpMax < 50.0f);
}

TEST_CASE("UV deviation channel sees texture stretch geometry cannot", "[vdpm]")
{
    // Flat + affine UV: both channels stay ~0.
    float affineUv = 0.0f;
    float affineGeom = 0.0f;
    for (const MeshCollapse& c : collapsesOf(makeGrid(17)))
    {
        affineUv = std::max(affineUv, c.uvDeviationRadius);
        affineGeom = std::max(affineGeom, c.deviationRadius);
    }
    CHECK(affineUv < 1e-3f);
    CHECK(affineGeom < 1e-3f);

    // Flat + nonlinear UV: geometry is still flat (geometric δ is blind), but the UV channel
    // accumulates real stretch — proving the channel adds information geometry cannot see.
    float skewUv = 0.0f;
    float skewGeom = 0.0f;
    for (const MeshCollapse& c : collapsesOf(makeFlatNonlinearUvGrid(17)))
    {
        skewUv = std::max(skewUv, c.uvDeviationRadius);
        skewGeom = std::max(skewGeom, c.deviationRadius);
    }
    CHECK(skewGeom < 1e-3f);
    CHECK(skewUv > 0.01f);
}

TEST_CASE("Shading-normal channel sees lighting error geometry and UV cannot", "[vdpm]")
{
    // Flat grid, affine UV, constant normals: all three channels stay ~0.
    float flatNormal = 0.0f;
    for (const MeshCollapse& c : collapsesOf(makeGrid(17)))
    {
        flatNormal = std::max(flatNormal, c.normalDeviationRadius);
    }
    CHECK(flatNormal < 1e-3f);

    // Flat grid, affine UV, but the normals fan across it: geometry (coplanar) and UV (affine) are
    // both blind, yet the shading channel accumulates the fan angle — the information neither other
    // channel carries.
    float fanNormal = 0.0f;
    float fanGeom = 0.0f;
    float fanUv = 0.0f;
    for (const MeshCollapse& c : collapsesOf(makeFlatFannedNormalGrid(17)))
    {
        fanNormal = std::max(fanNormal, c.normalDeviationRadius);
        fanGeom = std::max(fanGeom, c.deviationRadius);
        fanUv = std::max(fanUv, c.uvDeviationRadius);
    }
    CHECK(fanGeom < 1e-3f);
    CHECK(fanUv < 1e-3f);
    CHECK(fanNormal > 0.01f);
}

TEST_CASE("Tangent channel sees normal-map frame drift independently of the normal", "[vdpm]")
{
    // makeGrid has no tangents (Vec4{}) — the zero-length guard must read the tangent channel as 0,
    // so a mesh without tangents costs nothing.
    float noTangent = 0.0f;
    for (const MeshCollapse& c : collapsesOf(makeGrid(17)))
    {
        noTangent = std::max(noTangent, c.tangentDeviationRadius);
    }
    CHECK(noTangent < 1e-3f);

    // Flat grid, affine UV, constant normal, but a fanning tangent: geometry, UV, and the shading
    // normal are all blind; only the tangent channel accumulates the frame drift.
    float fanTangent = 0.0f;
    float fanNormal = 0.0f;
    float fanGeom = 0.0f;
    float fanUv = 0.0f;
    for (const MeshCollapse& c : collapsesOf(makeFlatFannedTangentGrid(17)))
    {
        fanTangent = std::max(fanTangent, c.tangentDeviationRadius);
        fanNormal = std::max(fanNormal, c.normalDeviationRadius);
        fanGeom = std::max(fanGeom, c.deviationRadius);
        fanUv = std::max(fanUv, c.uvDeviationRadius);
    }
    CHECK(fanGeom < 1e-3f);
    CHECK(fanUv < 1e-3f);
    CHECK(fanNormal < 1e-3f);
    CHECK(fanTangent > 0.01f);
}

TEST_CASE("Forest deviation is monotone along parent ancestry; vl/vr probe", "[vdpm]")
{
    const Mesh m = makeBumpyGrid(17);
    const VertexForest f = buildVertexForest(m.verts, m.indices, collapsesOf(m));

    int parentViolations = 0;
    int vlvrViolations = 0;
    auto probe = [&](uint32_t dep, float dependentError, int& counter)
    {
        if (dep == kInvalidVertex)
        {
            return;
        }
        const uint32_t depSplit = f.removingSplit[dep];
        if (depSplit == kNoSplit)
        {
            return; // a root dependency is always active
        }
        if (f.splits[depSplit].error < dependentError - 1e-4f)
        {
            ++counter;
        }
    };
    for (const VertexSplit& s : f.splits)
    {
        probe(s.parent, s.error, parentViolations);
        probe(s.vl, s.error, vlvrViolations);
        probe(s.vr, s.error, vlvrViolations);
    }

    // Accumulation up the kept-chain guarantees the parent dependency carries >= the dependent's
    // radius.
    CHECK(parentViolations == 0);
    // vl/vr are neighbours, not folded descendants, so their radius is NOT monotone — this
    // documents that the non-monotonicity is real, which is exactly why refineForView force-refines
    // its dependencies to stay legal (a simple gate-then-refine loop would leave splits stuck).
    CHECK(vlvrViolations > 0);
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
