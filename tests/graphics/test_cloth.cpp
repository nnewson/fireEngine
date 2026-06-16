#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <unordered_set>

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
