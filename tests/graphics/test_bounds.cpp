#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/bounds.hpp>

using fire_engine::Bounds3;
using fire_engine::Vec3;

TEST_CASE("Bounds3.DefaultIsInvalid", "[Bounds3]")
{
    Bounds3 bounds{};
    CHECK_FALSE(bounds.valid);
}

TEST_CASE("Bounds3.FirstExpansionSeedsMinAndMax", "[Bounds3]")
{
    Bounds3 bounds{};
    bounds.expand({1.0f, 2.0f, 3.0f});

    CHECK(bounds.valid);
    CHECK(bounds.min == Vec3(1.0f, 2.0f, 3.0f));
    CHECK(bounds.max == Vec3(1.0f, 2.0f, 3.0f));
}

TEST_CASE("Bounds3.ExpansionTracksComponentWiseExtremes", "[Bounds3]")
{
    Bounds3 bounds{};
    bounds.expand({1.0f, 4.0f, -2.0f});
    bounds.expand({-3.0f, 2.0f, 5.0f});

    CHECK(bounds.min == Vec3(-3.0f, 2.0f, -2.0f));
    CHECK(bounds.max == Vec3(1.0f, 4.0f, 5.0f));
}

TEST_CASE("Bounds3.CenterAndExtent", "[Bounds3]")
{
    Bounds3 bounds{};
    bounds.expand({-1.0f, 2.0f, 3.0f});
    bounds.expand({3.0f, 6.0f, 11.0f});

    CHECK(bounds.center() == Vec3(1.0f, 4.0f, 7.0f));
    CHECK(bounds.extent() == Vec3(4.0f, 4.0f, 8.0f));
}
