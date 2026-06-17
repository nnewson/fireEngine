#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include <fire_engine/graphics/cloth.hpp>

using namespace fire_engine;

namespace
{

// Within each colour span, no particle index may appear twice — that is the
// race-free property the GPU Gauss-Seidel solver relies on.
void expectColoursRaceFree(const ClothMesh& mesh)
{
    REQUIRE(mesh.colourRanges.size() >= 1u);
    CHECK(mesh.colourRanges.front() == 0u);
    CHECK(mesh.colourRanges.back() == mesh.constraints.size());

    for (uint32_t c = 0; c < mesh.numColours(); ++c)
    {
        std::unordered_set<uint32_t> seen;
        for (uint32_t e = mesh.colourRanges[c]; e < mesh.colourRanges[c + 1]; ++e)
        {
            const ClothConstraint& con = mesh.constraints[e];
            INFO("particle " << con.a << " repeats in colour " << c);
            CHECK(seen.insert(con.a).second);
            INFO("particle " << con.b << " repeats in colour " << c);
            CHECK(seen.insert(con.b).second);
        }
    }
}

} // namespace

TEST_CASE("Cloth.GridParticleAndIndexCounts", "[Cloth]")
{
    ClothGridParams params;
    params.resX = 8;
    params.resZ = 5;
    const ClothMesh mesh = makeGridCloth(params);

    CHECK(mesh.particles.size() == 8u * 5u);
    CHECK(mesh.vertices.size() == 8u * 5u);
    CHECK(mesh.indices.size() == (8u - 1u) * (5u - 1u) * 6u);
}

TEST_CASE("Cloth.PinsTopCorners", "[Cloth]")
{
    ClothGridParams params;
    params.resX = 6;
    params.resZ = 6;
    params.pinTopCorners = true;
    const ClothMesh mesh = makeGridCloth(params);

    int pinned = 0;
    for (const ClothParticle& p : mesh.particles)
    {
        if (p.invMass == 0.0f)
        {
            ++pinned;
        }
    }
    CHECK(pinned == 2);
    // The two far-Z corners (j == resZ - 1).
    CHECK(mesh.particles[(6u - 1u) * 6u + 0u].invMass == 0.0f);
    CHECK(mesh.particles[(6u - 1u) * 6u + 5u].invMass == 0.0f);
}

TEST_CASE("Cloth.ColouringIsRaceFree", "[Cloth]")
{
    ClothGridParams params;
    params.resX = 16;
    params.resZ = 12;
    const ClothMesh mesh = makeGridCloth(params);

    CHECK(mesh.constraints.size() > 0u);
    CHECK(mesh.numColours() > 0u);
    expectColoursRaceFree(mesh);
}

TEST_CASE("Cloth.ColouringPreservesConstraintCount", "[Cloth]")
{
    std::vector<ClothConstraint> constraints{
        {0, 1, 1.0f, 0.0f}, {1, 2, 1.0f, 0.0f}, {2, 3, 1.0f, 0.0f}, {0, 2, 1.0f, 0.0f}};
    const std::size_t before = constraints.size();
    std::vector<uint32_t> ranges;
    colourConstraints(constraints, ranges, 4);

    CHECK(constraints.size() == before);
    CHECK(ranges.back() == before);
    // 0-1, 1-2, 2-3 form a path sharing endpoints; 0-2 shares with several. Greedy
    // needs at least 2 colours here.
    CHECK(ranges.size() - 1 >= 2u);
}

TEST_CASE("Cloth.RestLengthsMatchSpacing", "[Cloth]")
{
    ClothGridParams params;
    params.resX = 4;
    params.resZ = 4;
    params.spacing = 0.1f;
    const ClothMesh mesh = makeGridCloth(params);

    // Every structural (axis-aligned) constraint should rest at exactly spacing;
    // shear at sqrt(2)*spacing; bend at 2*spacing. Just assert all are positive
    // and the minimum equals the spacing.
    float minRest = 1e9f;
    for (const ClothConstraint& c : mesh.constraints)
    {
        CHECK(c.restLength > 0.0f);
        minRest = std::min(minRest, c.restLength);
    }
    CHECK(minRest == Catch::Approx(params.spacing).margin(1e-5f));
}

TEST_CASE("Cloth.PerTypeComplianceIsTagged", "[Cloth]")
{
    ClothGridParams params;
    params.resX = 5;
    params.resZ = 5;
    params.spacing = 0.1f;
    params.structuralCompliance = 0.0f;
    params.shearCompliance = 1.0e-4f;
    params.bendCompliance = 5.0e-4f;
    const ClothMesh mesh = makeGridCloth(params);

    // Classify each constraint by rest length: ~spacing = structural,
    // ~sqrt(2)*spacing = shear, ~2*spacing = bend. Each class must carry its
    // authored compliance.
    const float s = params.spacing;
    for (const ClothConstraint& c : mesh.constraints)
    {
        if (c.restLength == Catch::Approx(s).margin(1e-4f))
        {
            CHECK(c.compliance == Catch::Approx(params.structuralCompliance).margin(1e-9f));
        }
        else if (c.restLength == Catch::Approx(s * 1.41421356f).margin(1e-4f))
        {
            CHECK(c.compliance == Catch::Approx(params.shearCompliance).margin(1e-9f));
        }
        else if (c.restLength == Catch::Approx(2.0f * s).margin(1e-4f))
        {
            CHECK(c.compliance == Catch::Approx(params.bendCompliance).margin(1e-9f));
        }
    }
}

TEST_CASE("Cloth.NormalAdjacencyIsCsrAndComplete", "[Cloth]")
{
    // Two triangles sharing edge (1,2): {0,1,2} and {2,1,3}.
    const std::vector<uint32_t> indices{0, 1, 2, 2, 1, 3};
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> tris;
    buildNormalAdjacency(indices, 4, offsets, tris);

    REQUIRE(offsets.size() == 5u); // particleCount + 1
    CHECK(offsets.front() == 0u);
    CHECK(offsets.back() == tris.size()); // total incidences
    CHECK(tris.size() == 6u);             // 2 triangles * 3 vertices

    auto incidentTris = [&](uint32_t v)
    {
        std::unordered_set<uint32_t> result;
        for (uint32_t k = offsets[v]; k < offsets[v + 1]; ++k)
        {
            result.insert(tris[k]);
        }
        return result;
    };
    // Vertices 0 and 3 touch one triangle each; 1 and 2 (the shared edge) touch both.
    CHECK(incidentTris(0) == std::unordered_set<uint32_t>{0});
    CHECK(incidentTris(3) == std::unordered_set<uint32_t>{1});
    CHECK(incidentTris(1) == (std::unordered_set<uint32_t>{0, 1}));
    CHECK(incidentTris(2) == (std::unordered_set<uint32_t>{0, 1}));
}

TEST_CASE("Cloth.GridAdjacencyMatchesParticleCount", "[Cloth]")
{
    ClothGridParams params;
    params.resX = 6;
    params.resZ = 4;
    const ClothMesh mesh = makeGridCloth(params);

    REQUIRE(mesh.normalAdjOffsets.size() == mesh.particles.size() + 1u);
    CHECK(mesh.normalAdjOffsets.back() == mesh.normalAdjTris.size());
    // Each triangle contributes 3 incidences, so the flat list equals the index count.
    CHECK(mesh.normalAdjTris.size() == mesh.indices.size());
    for (std::size_t v = 0; v + 1 < mesh.normalAdjOffsets.size(); ++v)
    {
        CHECK(mesh.normalAdjOffsets[v + 1] >= mesh.normalAdjOffsets[v]);
    }
}

namespace
{

// A two-quad mesh (3x2 vertices) in the XZ plane at Y=1. Layout (X across, Z down):
//   z=0:  0(0,0) 1(1,0) 2(2,0)
//   z=1:  3(0,1) 4(1,1) 5(2,1)
// When `duplicateSeam` is set, a 7th vertex coincident with vertex 1 is added and
// referenced, so welding must collapse it back to 6 particles.
ClothMesh buildTwoQuadCloth(ClothMeshParams params, bool duplicateSeam)
{
    auto vert = [](float x, float z)
    { return Vertex(Vec3{x, 1.0f, z}, Colour3{1, 1, 1}, Vec3{0, 1, 0}, Vec2{x, z}); };

    std::vector<Vertex> verts{vert(0, 0), vert(1, 0), vert(2, 0),
                              vert(0, 1), vert(1, 1), vert(2, 1)};
    std::vector<uint32_t> indices{0, 3, 1, 1, 3, 4, 1, 4, 2, 2, 4, 5};

    if (duplicateSeam)
    {
        const auto dup = static_cast<uint32_t>(verts.size());
        verts.push_back(vert(1, 0)); // coincident with vertex 1
        indices[6] = dup;            // a right-quad triangle now references the duplicate
    }
    return makeClothFromMesh(verts, indices, params);
}

} // namespace

TEST_CASE("Cloth.FromMeshWeldsCoincidentPositions", "[Cloth]")
{
    const ClothMesh welded = buildTwoQuadCloth({}, /*duplicateSeam=*/true);
    // Seven input vertices, but the duplicate at (1,0) welds back to six particles.
    CHECK(welded.particles.size() == 6u);
    // Render vertices keep the original (un-welded) count for the draw.
    CHECK(welded.vertices.size() == 7u);
}

TEST_CASE("Cloth.FromMeshBuildsSharedEdgeConstraintsOnce", "[Cloth]")
{
    const ClothMesh mesh = buildTwoQuadCloth({}, /*duplicateSeam=*/false);

    // Structural edges are deduped, so the interior edge shared by two triangles
    // appears once. Verify no duplicate (a,b) constraint and all rest lengths > 0.
    std::unordered_set<uint64_t> seen;
    for (const ClothConstraint& c : mesh.constraints)
    {
        CHECK(c.restLength > 0.0f);
        const uint64_t key = (static_cast<uint64_t>(std::min(c.a, c.b)) << 32) | std::max(c.a, c.b);
        INFO("duplicate constraint between " << c.a << " and " << c.b);
        CHECK(seen.insert(key).second);
    }
    // Colouring + adjacency are filled for the arbitrary-mesh path too.
    CHECK(mesh.numColours() > 0u);
    CHECK(mesh.normalAdjOffsets.size() == mesh.particles.size() + 1u);
}

TEST_CASE("Cloth.FromMeshPinsTopEdge", "[Cloth]")
{
    ClothMeshParams params;
    params.pin = ClothMeshParams::Pin::TopEdge;
    const ClothMesh mesh = buildTwoQuadCloth(params, /*duplicateSeam=*/false);

    // The max-Z edge (z == 1) is three particles; all pinned, the rest free.
    int pinned = 0;
    for (const ClothParticle& p : mesh.particles)
    {
        if (p.invMass == 0.0f)
        {
            ++pinned;
            CHECK(p.position.z() == Catch::Approx(1.0f).margin(1e-4f));
        }
    }
    CHECK(pinned == 3);
}

TEST_CASE("Cloth.FromMeshPinTopCornersPinsTwo", "[Cloth]")
{
    ClothMeshParams params;
    params.pin = ClothMeshParams::Pin::TopCorners;
    const ClothMesh mesh = buildTwoQuadCloth(params, /*duplicateSeam=*/false);

    int pinned = 0;
    for (const ClothParticle& p : mesh.particles)
    {
        if (p.invMass == 0.0f)
        {
            ++pinned;
        }
    }
    CHECK(pinned == 2); // the two extreme-X corners of the max-Z edge
}
