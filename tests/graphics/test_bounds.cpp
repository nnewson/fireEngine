#include <gtest/gtest.h>

#include <fire_engine/graphics/bounds.hpp>

using fire_engine::Bounds3;
using fire_engine::Vec3;

TEST(Bounds3, DefaultIsInvalid)
{
    Bounds3 bounds{};
    EXPECT_FALSE(bounds.valid);
}

TEST(Bounds3, FirstExpansionSeedsMinAndMax)
{
    Bounds3 bounds{};
    bounds.expand({1.0f, 2.0f, 3.0f});

    EXPECT_TRUE(bounds.valid);
    EXPECT_EQ(bounds.min, Vec3(1.0f, 2.0f, 3.0f));
    EXPECT_EQ(bounds.max, Vec3(1.0f, 2.0f, 3.0f));
}

TEST(Bounds3, ExpansionTracksComponentWiseExtremes)
{
    Bounds3 bounds{};
    bounds.expand({1.0f, 4.0f, -2.0f});
    bounds.expand({-3.0f, 2.0f, 5.0f});

    EXPECT_EQ(bounds.min, Vec3(-3.0f, 2.0f, -2.0f));
    EXPECT_EQ(bounds.max, Vec3(1.0f, 4.0f, 5.0f));
}

TEST(Bounds3, CenterAndExtent)
{
    Bounds3 bounds{};
    bounds.expand({-1.0f, 2.0f, 3.0f});
    bounds.expand({3.0f, 6.0f, 11.0f});

    EXPECT_EQ(bounds.center(), Vec3(1.0f, 4.0f, 7.0f));
    EXPECT_EQ(bounds.extent(), Vec3(4.0f, 4.0f, 8.0f));
}
