#include <fire_engine/core/tangent_generator.hpp>

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::TangentGenerator;
using fire_engine::Vec2;
using fire_engine::Vec3;
using fire_engine::Vec4;

TEST_CASE("TangentGenerator.GeneratesTangentsForPlanarQuad", "[TangentGenerator]")
{
    std::vector<Vec3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    };
    std::vector<Vec3> normals(4, {0.0f, 0.0f, 1.0f});
    std::vector<Vec2> texcoords = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
    };
    std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};

    auto result = TangentGenerator::generate(positions, normals, texcoords, indices);

    REQUIRE(result.succeeded);
    REQUIRE(result.tangents.size() == positions.size());
    for (const auto& tangent : result.tangents)
    {
        CHECK(tangent.x() == Catch::Approx(1.0f).margin(1e-4f));
        CHECK(tangent.y() == Catch::Approx(0.0f).margin(1e-4f));
        CHECK(tangent.z() == Catch::Approx(0.0f).margin(1e-4f));
        CHECK(std::abs(tangent.w()) == 1.0f);
    }
}

TEST_CASE("TangentGenerator.GeneratedTangentsAreOrthogonalToNormals", "[TangentGenerator]")
{
    std::vector<Vec3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
    };
    std::vector<Vec3> normals(3, {0.0f, 0.0f, 1.0f});
    std::vector<Vec2> texcoords = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
    };
    std::vector<uint32_t> indices = {0, 1, 2};

    auto result = TangentGenerator::generate(positions, normals, texcoords, indices);

    REQUIRE(result.succeeded);
    for (std::size_t i = 0; i < normals.size(); ++i)
    {
        Vec3 tangent{result.tangents[i].x(), result.tangents[i].y(), result.tangents[i].z()};
        CHECK(Vec3::dotProduct(normals[i], tangent) == Catch::Approx(0.0f).margin(1e-4f));
    }
}

TEST_CASE("TangentGenerator.GeneratesNegativeHandednessForMirroredUvs", "[TangentGenerator]")
{
    std::vector<Vec3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    };
    std::vector<Vec3> normals(4, {0.0f, 0.0f, 1.0f});
    std::vector<Vec2> texcoords = {
        {1.0f, 0.0f},
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
    };
    std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};

    auto result = TangentGenerator::generate(positions, normals, texcoords, indices);

    REQUIRE(result.succeeded);
    for (const auto& tangent : result.tangents)
    {
        CHECK(tangent.w() == Catch::Approx(-1.0f).margin(1e-5f));
    }
}

TEST_CASE("TangentGenerator.FailsWhenUvsAreMissing", "[TangentGenerator]")
{
    std::vector<Vec3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
    };
    std::vector<Vec3> normals(3, {0.0f, 0.0f, 1.0f});
    std::vector<Vec2> texcoords;
    std::vector<uint32_t> indices = {0, 1, 2};

    auto result = TangentGenerator::generate(positions, normals, texcoords, indices);

    CHECK_FALSE(result.succeeded);
    CHECK(result.reason == "missing UVs");
}

TEST_CASE("TangentGenerator.FailsOnDegenerateUvMapping", "[TangentGenerator]")
{
    std::vector<Vec3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
    };
    std::vector<Vec3> normals(3, {0.0f, 0.0f, 1.0f});
    std::vector<Vec2> texcoords = {
        {0.0f, 0.0f},
        {0.0f, 0.0f},
        {0.0f, 0.0f},
    };
    std::vector<uint32_t> indices = {0, 1, 2};

    auto result = TangentGenerator::generate(positions, normals, texcoords, indices);

    CHECK_FALSE(result.succeeded);
    CHECK(result.reason == "degenerate UV mapping");
}

TEST_CASE("TangentGenerator.FallsBackForVerticesOnlyInDegenerateUvTriangles", "[TangentGenerator]")
{
    std::vector<Vec3> positions = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {2.0f, 1.0f, 0.0f},
    };
    std::vector<Vec3> normals(6, {0.0f, 0.0f, 1.0f});
    std::vector<Vec2> texcoords = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}, {2.0f, 0.0f}, {2.0f, 0.0f},
    };
    std::vector<uint32_t> indices = {
        0, 1, 2, 0, 2, 3, 1, 4, 5,
    };

    auto result = TangentGenerator::generate(positions, normals, texcoords, indices);

    REQUIRE(result.succeeded);
    REQUIRE(result.tangents.size() == positions.size());
    for (const auto& tangent : result.tangents)
    {
        Vec3 tangentDir{tangent.x(), tangent.y(), tangent.z()};
        CHECK(tangentDir.magnitudeSquared() > 0.0f);
        CHECK(Vec3::dotProduct(Vec3{0.0f, 0.0f, 1.0f}, tangentDir) ==
              Catch::Approx(0.0f).margin(1e-4f));
        CHECK(std::abs(tangent.w()) == 1.0f);
    }
}
