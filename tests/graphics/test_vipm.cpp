#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

#include <fire_engine/graphics/vipm.hpp>

using namespace fire_engine;

namespace
{

Vertex at(float x)
{
    return Vertex{Vec3{x, 0.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{}};
}

MeshCollapse collapse(uint32_t kept, uint32_t removed, float error)
{
    return MeshCollapse{kept, removed, Vec3{}, error};
}

} // namespace

TEST_CASE("buildVipmMorphData: non-collapsing vertices keep own position + never morph", "[vipm]")
{
    const std::array<Vertex, 3> verts{at(0.0f), at(1.0f), at(2.0f)};
    const std::vector<MeshCollapse> collapses{}; // nothing collapses
    const std::array<float, 2> levels{0.0f, 1.0f};

    const auto morph = buildVipmMorphData(verts, collapses, levels);

    REQUIRE(morph.size() == 3);
    for (std::size_t i = 0; i < morph.size(); ++i)
    {
        CHECK(morph[i].targetPosition.x() == verts[i].position().x());
        CHECK(morph[i].collapseError == kVipmNeverCollapses);
    }
}

TEST_CASE("buildVipmMorphData: a collapse chain in one band resolves to the final survivor",
          "[vipm]")
{
    // 0 -> 1 (err 0.1), then 1 -> 2 (err 0.2); a single band [0, 0.5] contains both.
    const std::array<Vertex, 4> verts{at(0.0f), at(1.0f), at(2.0f), at(3.0f)};
    const std::vector<MeshCollapse> collapses{collapse(1, 0, 0.1f), collapse(2, 1, 0.2f)};
    const std::array<float, 2> levels{0.0f, 0.5f};

    const auto morph = buildVipmMorphData(verts, collapses, levels);

    // Both 0 and 1 must land on vertex 2 (the survivor) — chain A->B->C followed through, so at
    // morphFactor 1 they are coincident with a vertex present after the swap.
    CHECK(morph[0].targetPosition.x() == 2.0f);
    CHECK(morph[0].collapseError == 0.1f);
    CHECK(morph[1].targetPosition.x() == 2.0f);
    CHECK(morph[1].collapseError == 0.2f);
    // 2 and 3 survive.
    CHECK(morph[2].collapseError == kVipmNeverCollapses);
    CHECK(morph[3].collapseError == kVipmNeverCollapses);
}

TEST_CASE("buildVipmMorphData: banding resolves each vertex to its own level's survivor", "[vipm]")
{
    // Same 0->1 (0.1), 1->2 (0.2), but the level boundary at 0.15 splits them into two bands.
    const std::array<Vertex, 4> verts{at(0.0f), at(1.0f), at(2.0f), at(3.0f)};
    const std::vector<MeshCollapse> collapses{collapse(1, 0, 0.1f), collapse(2, 1, 0.2f)};
    const std::array<float, 3> levels{0.0f, 0.15f, 0.5f};

    const auto morph = buildVipmMorphData(verts, collapses, levels);

    // 0 collapses in band 1 (<=0.15): its survivor *at level 1* is vertex 1 (1->2 hasn't happened).
    CHECK(morph[0].targetPosition.x() == 1.0f);
    CHECK(morph[0].collapseError == 0.1f);
    // 1 collapses in band 2: survivor is vertex 2.
    CHECK(morph[1].targetPosition.x() == 2.0f);
    CHECK(morph[1].collapseError == 0.2f);
}

TEST_CASE("selectVipm: topology level matches discrete selectLod", "[vipm]")
{
    const std::array<GeometryLod, 3> lods{GeometryLod{NullBuffer, 0, 0.0f},
                                          GeometryLod{NullBuffer, 0, 1.0f},
                                          GeometryLod{NullBuffer, 0, 4.0f}};
    const float projScaleY = 1.0f;
    const float viewportHeight = 1000.0f;
    const float pixelError = 2.0f;

    for (const float d : {50.0f, 250.0f, 600.0f, 1500.0f})
    {
        const auto sel = selectVipm(lods, d, projScaleY, viewportHeight, pixelError);
        const auto discrete = selectLod(lods, d, projScaleY, viewportHeight, pixelError);
        CHECK(sel.level == discrete);
    }
}

TEST_CASE("selectVipm: morphFactor ramps 0->1 across a band, 0 at the coarsest level", "[vipm]")
{
    const std::array<GeometryLod, 3> lods{GeometryLod{NullBuffer, 0, 0.0f},
                                          GeometryLod{NullBuffer, 0, 1.0f},
                                          GeometryLod{NullBuffer, 0, 4.0f}};
    const float projScaleY = 1.0f;
    const float viewportHeight = 1000.0f;
    const float pixelError = 2.0f;
    // tolerated world error E = pixelError * 2 * d / (projScaleY * viewportHeight) = 0.004 * d.

    // d = 250 -> E = 1.0 (exactly level 1's error): still level 1, morph just starting.
    const auto low = selectVipm(lods, 250.0f, projScaleY, viewportHeight, pixelError);
    CHECK(low.level == 1);
    CHECK(low.nextLevelError == 4.0f);
    CHECK(low.morphFactor == 0.0f);

    // d = 625 -> E = 2.5, halfway from 1.0 to 4.0 -> morphFactor 0.5.
    const auto mid = selectVipm(lods, 625.0f, projScaleY, viewportHeight, pixelError);
    CHECK(mid.level == 1);
    CHECK(mid.morphFactor > 0.49f);
    CHECK(mid.morphFactor < 0.51f);

    // Far enough that the coarsest level is selected -> no morph target, factor 0.
    const auto coarse = selectVipm(lods, 5000.0f, projScaleY, viewportHeight, pixelError);
    CHECK(coarse.level == 2);
    CHECK(coarse.morphFactor == 0.0f);
    CHECK(coarse.nextLevelError == 0.0f);
}
