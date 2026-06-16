#include <gtest/gtest.h>

#include <unordered_set>

#include <fire_engine/graphics/cloth.hpp>

using namespace fire_engine;

namespace
{

// Within each colour span, no particle index may appear twice — that is the
// race-free property the GPU Gauss-Seidel solver relies on.
void expectColoursRaceFree(const ClothMesh& mesh)
{
    ASSERT_GE(mesh.colourRanges.size(), 1u);
    EXPECT_EQ(mesh.colourRanges.front(), 0u);
    EXPECT_EQ(mesh.colourRanges.back(), mesh.constraints.size());

    for (uint32_t c = 0; c < mesh.numColours(); ++c)
    {
        std::unordered_set<uint32_t> seen;
        for (uint32_t e = mesh.colourRanges[c]; e < mesh.colourRanges[c + 1]; ++e)
        {
            const ClothConstraint& con = mesh.constraints[e];
            EXPECT_TRUE(seen.insert(con.a).second)
                << "particle " << con.a << " repeats in colour " << c;
            EXPECT_TRUE(seen.insert(con.b).second)
                << "particle " << con.b << " repeats in colour " << c;
        }
    }
}

} // namespace

TEST(Cloth, GridParticleAndIndexCounts)
{
    ClothGridParams params;
    params.resX = 8;
    params.resZ = 5;
    const ClothMesh mesh = makeGridCloth(params);

    EXPECT_EQ(mesh.particles.size(), 8u * 5u);
    EXPECT_EQ(mesh.vertices.size(), 8u * 5u);
    EXPECT_EQ(mesh.indices.size(), (8u - 1u) * (5u - 1u) * 6u);
}

TEST(Cloth, PinsTopCorners)
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
    EXPECT_EQ(pinned, 2);
    // The two far-Z corners (j == resZ - 1).
    EXPECT_EQ(mesh.particles[(6u - 1u) * 6u + 0u].invMass, 0.0f);
    EXPECT_EQ(mesh.particles[(6u - 1u) * 6u + 5u].invMass, 0.0f);
}

TEST(Cloth, ColouringIsRaceFree)
{
    ClothGridParams params;
    params.resX = 16;
    params.resZ = 12;
    const ClothMesh mesh = makeGridCloth(params);

    EXPECT_GT(mesh.constraints.size(), 0u);
    EXPECT_GT(mesh.numColours(), 0u);
    expectColoursRaceFree(mesh);
}

TEST(Cloth, ColouringPreservesConstraintCount)
{
    std::vector<ClothConstraint> constraints{
        {0, 1, 1.0f, 0.0f}, {1, 2, 1.0f, 0.0f}, {2, 3, 1.0f, 0.0f}, {0, 2, 1.0f, 0.0f}};
    const std::size_t before = constraints.size();
    std::vector<uint32_t> ranges;
    colourConstraints(constraints, ranges, 4);

    EXPECT_EQ(constraints.size(), before);
    EXPECT_EQ(ranges.back(), before);
    // 0-1, 1-2, 2-3 form a path sharing endpoints; 0-2 shares with several. Greedy
    // needs at least 2 colours here.
    EXPECT_GE(ranges.size() - 1, 2u);
}

TEST(Cloth, RestLengthsMatchSpacing)
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
        EXPECT_GT(c.restLength, 0.0f);
        minRest = std::min(minRest, c.restLength);
    }
    EXPECT_NEAR(minRest, params.spacing, 1e-5f);
}
