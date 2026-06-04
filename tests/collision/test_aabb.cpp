#include <gtest/gtest.h>

#include <fire_engine/collision/aabb.hpp>

using fire_engine::AABB;
using fire_engine::Axis;
using fire_engine::Vec3;

TEST(AABB, DefaultIsZero)
{
    AABB box{};
    EXPECT_EQ(box.min, Vec3{});
    EXPECT_EQ(box.max, Vec3{});
}

TEST(AABB, AxisMinReturnsMinComponent)
{
    AABB box{Vec3{1.0f, 2.0f, 3.0f}, Vec3{4.0f, 5.0f, 6.0f}};
    EXPECT_FLOAT_EQ(box.axisMin(Axis::X), 1.0f);
    EXPECT_FLOAT_EQ(box.axisMin(Axis::Y), 2.0f);
    EXPECT_FLOAT_EQ(box.axisMin(Axis::Z), 3.0f);
}

TEST(AABB, AxisMaxReturnsMaxComponent)
{
    AABB box{Vec3{1.0f, 2.0f, 3.0f}, Vec3{4.0f, 5.0f, 6.0f}};
    EXPECT_FLOAT_EQ(box.axisMax(Axis::X), 4.0f);
    EXPECT_FLOAT_EQ(box.axisMax(Axis::Y), 5.0f);
    EXPECT_FLOAT_EQ(box.axisMax(Axis::Z), 6.0f);
}

TEST(AABB, CenterIsMidpoint)
{
    AABB box{Vec3{-2.0f, 0.0f, 4.0f}, Vec3{2.0f, 4.0f, 8.0f}};
    EXPECT_EQ(box.center(), Vec3(0.0f, 2.0f, 6.0f));
}

TEST(AABB, ExtentIsMaxMinusMin)
{
    AABB box{Vec3{1.0f, 2.0f, 3.0f}, Vec3{5.0f, 8.0f, 11.0f}};
    EXPECT_EQ(box.extent(), Vec3(4.0f, 6.0f, 8.0f));
}

TEST(AABB, AxisHelpersAreConstexpr)
{
    constexpr AABB box{Vec3{1.0f, 2.0f, 3.0f}, Vec3{4.0f, 5.0f, 6.0f}};
    static_assert(box.axisMin(Axis::X) == 1.0f);
    static_assert(box.axisMax(Axis::Z) == 6.0f);
}
