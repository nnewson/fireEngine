#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/collision/aabb.hpp>

using fire_engine::AABB;
using fire_engine::Axis;
using fire_engine::Vec3;

TEST_CASE("AABB.DefaultIsZero", "[AABB]")
{
    AABB box{};
    CHECK(box.min == Vec3{});
    CHECK(box.max == Vec3{});
}

TEST_CASE("AABB.AxisMinReturnsMinComponent", "[AABB]")
{
    AABB box{Vec3{1.0f, 2.0f, 3.0f}, Vec3{4.0f, 5.0f, 6.0f}};
    CHECK(box.axisMin(Axis::X) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(box.axisMin(Axis::Y) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(box.axisMin(Axis::Z) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("AABB.AxisMaxReturnsMaxComponent", "[AABB]")
{
    AABB box{Vec3{1.0f, 2.0f, 3.0f}, Vec3{4.0f, 5.0f, 6.0f}};
    CHECK(box.axisMax(Axis::X) == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(box.axisMax(Axis::Y) == Catch::Approx(5.0f).margin(1e-5f));
    CHECK(box.axisMax(Axis::Z) == Catch::Approx(6.0f).margin(1e-5f));
}

TEST_CASE("AABB.CenterIsMidpoint", "[AABB]")
{
    AABB box{Vec3{-2.0f, 0.0f, 4.0f}, Vec3{2.0f, 4.0f, 8.0f}};
    CHECK(box.center() == Vec3(0.0f, 2.0f, 6.0f));
}

TEST_CASE("AABB.ExtentIsMaxMinusMin", "[AABB]")
{
    AABB box{Vec3{1.0f, 2.0f, 3.0f}, Vec3{5.0f, 8.0f, 11.0f}};
    CHECK(box.extent() == Vec3(4.0f, 6.0f, 8.0f));
}

TEST_CASE("AABB.AxisHelpersAreConstexpr", "[AABB]")
{
    constexpr AABB box{Vec3{1.0f, 2.0f, 3.0f}, Vec3{4.0f, 5.0f, 6.0f}};
    static_assert(box.axisMin(Axis::X) == 1.0f);
    static_assert(box.axisMax(Axis::Z) == 6.0f);
}
