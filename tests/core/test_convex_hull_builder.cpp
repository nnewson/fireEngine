#include <fire_engine/core/convex_hull_builder.hpp>

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using fire_engine::buildConvexHull;
using fire_engine::ConvexHullShape;
using fire_engine::isConvex;
using fire_engine::Vec3;

namespace
{

// Unit cube: 8 shared vertices, 12 CCW-outward triangles (2 per face).
ConvexHullShape cubeHull()
{
    const std::vector<Vec3> verts{
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1},
    };
    const std::vector<std::uint32_t> idx{
        1, 2, 6, 1, 6, 5, // +x
        0, 4, 7, 0, 7, 3, // -x
        3, 7, 6, 3, 6, 2, // +y
        0, 1, 5, 0, 5, 4, // -y
        4, 5, 6, 4, 6, 7, // +z
        0, 3, 2, 0, 2, 1, // -z
    };
    return buildConvexHull(verts, idx);
}

} // namespace

TEST_CASE("ConvexHullBuilder.CubeMeshMergesToSixQuadFaces", "[ConvexHull]")
{
    const ConvexHullShape hull = cubeHull();
    CHECK(hull.vertices.size() == 8u);
    REQUIRE(hull.faces.size() == 6u);
    for (const auto& face : hull.faces)
    {
        CHECK(face.loop.size() == 4u); // coplanar triangle pairs merged to a quad
    }
    CHECK(isConvex(hull));
}

TEST_CASE("ConvexHullBuilder.TetrahedronMeshGivesFourTriangleFaces", "[ConvexHull]")
{
    const std::vector<Vec3> verts{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
    const std::vector<std::uint32_t> idx{
        0, 2, 1, // outward faces of the tetra
        0, 3, 2, 0, 1, 3, 1, 2, 3,
    };
    const ConvexHullShape hull = buildConvexHull(verts, idx);
    CHECK(hull.vertices.size() == 4u);
    REQUIRE(hull.faces.size() == 4u);
    for (const auto& face : hull.faces)
    {
        CHECK(face.loop.size() == 3u);
    }
}

TEST_CASE("ConvexHullBuilder.RejectsNonConvex", "[ConvexHull]")
{
    ConvexHullShape hull = cubeHull();
    REQUIRE(isConvex(hull));
    // A vertex pushed far in front of the +x face makes the hull non-convex.
    hull.vertices.push_back({5.0f, 0.0f, 0.0f});
    CHECK_FALSE(isConvex(hull));
}
