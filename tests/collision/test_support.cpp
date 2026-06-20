#include <fire_engine/collision/support.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Quaternion;
using fire_engine::supportPoint;
using fire_engine::Vec3;
using fire_engine::WorldBox;
using fire_engine::WorldCapsule;
using fire_engine::WorldConvex;
using fire_engine::WorldShape;
using fire_engine::WorldSphere;

TEST_CASE("Support.Sphere", "[Support]")
{
    const WorldSphere s{{1.0f, 2.0f, 3.0f}, 0.5f};
    CHECK(supportPoint(s, Vec3{1.0f, 0.0f, 0.0f}).approxEqual(Vec3{1.5f, 2.0f, 3.0f}, 1e-5f));
    CHECK(supportPoint(s, Vec3{0.0f, -2.0f, 0.0f}).approxEqual(Vec3{1.0f, 1.5f, 3.0f}, 1e-5f));
}

TEST_CASE("Support.Box", "[Support]")
{
    const WorldBox b{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, Quaternion::identity()};
    // Diagonal direction picks the +++ corner.
    CHECK(supportPoint(b, Vec3{1.0f, 1.0f, 1.0f}).approxEqual(Vec3{1.0f, 1.0f, 1.0f}, 1e-5f));
    CHECK(supportPoint(b, Vec3{-1.0f, 1.0f, -1.0f}).approxEqual(Vec3{-1.0f, 1.0f, -1.0f}, 1e-5f));

    // 90° about Z maps the local +x face toward world +y.
    const float h = std::sqrt(0.5f);
    const WorldBox r{{0.0f, 0.0f, 0.0f}, {2.0f, 1.0f, 1.0f}, Quaternion{0.0f, 0.0f, h, h}};
    // Local +x half-extent 2 now points +y → support along +y is at y≈2.
    CHECK(supportPoint(r, Vec3{0.0f, 1.0f, 0.0f}).y() == Catch::Approx(2.0f).margin(1e-4f));
}

TEST_CASE("Support.Capsule", "[Support]")
{
    const WorldCapsule c{{0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.5f};
    CHECK(supportPoint(c, Vec3{0.0f, 1.0f, 0.0f}).approxEqual(Vec3{0.0f, 1.5f, 0.0f}, 1e-5f));
    CHECK(supportPoint(c, Vec3{0.0f, -1.0f, 0.0f}).approxEqual(Vec3{0.0f, -1.5f, 0.0f}, 1e-5f));
    CHECK(supportPoint(c, Vec3{1.0f, 0.0f, 0.0f}).x() == Catch::Approx(0.5f).margin(1e-5f));
}

TEST_CASE("Support.ConvexAndVariantDispatch", "[Support]")
{
    // Unit cube vertices.
    WorldConvex cube;
    for (float x : {-1.0f, 1.0f})
    {
        for (float y : {-1.0f, 1.0f})
        {
            for (float z : {-1.0f, 1.0f})
            {
                cube.vertices.push_back({x, y, z});
            }
        }
    }
    CHECK(supportPoint(cube, Vec3{1.0f, 1.0f, 1.0f}).approxEqual(Vec3{1.0f, 1.0f, 1.0f}, 1e-5f));
    CHECK(
        supportPoint(cube, Vec3{-1.0f, -1.0f, 1.0f}).approxEqual(Vec3{-1.0f, -1.0f, 1.0f}, 1e-5f));

    // Same through the WorldShape variant dispatch.
    const WorldShape shape{cube};
    CHECK(supportPoint(shape, Vec3{1.0f, 0.0f, 0.0f}).x() == Catch::Approx(1.0f).margin(1e-5f));
}
