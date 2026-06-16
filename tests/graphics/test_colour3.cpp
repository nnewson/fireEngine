#include <fire_engine/graphics/colour3.hpp>

#include <limits>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Colour3;

// ---- Helpers ----

static constexpr float kEps = 1e-6f;

static void expectNear(const Colour3& c, float er, float eg, float eb, float eps = kEps)
{
    CHECK(c.r() == Catch::Approx(er).margin(eps));
    CHECK(c.g() == Catch::Approx(eg).margin(eps));
    CHECK(c.b() == Catch::Approx(eb).margin(eps));
}

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("Colour3Construction.DefaultIsZero", "[Colour3Construction]")
{
    Colour3 c{};
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Colour3Construction.ExplicitValues", "[Colour3Construction]")
{
    Colour3 c{0.1f, 0.5f, 0.9f};
    expectNear(c, 0.1f, 0.5f, 0.9f);
}

TEST_CASE("Colour3Construction.PartialArgs", "[Colour3Construction]")
{
    Colour3 cr{0.5f};
    expectNear(cr, 0.5f, 0.0f, 0.0f);

    Colour3 crg{0.5f, 0.6f};
    expectNear(crg, 0.5f, 0.6f, 0.0f);
}

TEST_CASE("Colour3Construction.White", "[Colour3Construction]")
{
    Colour3 c{1.0f, 1.0f, 1.0f};
    expectNear(c, 1.0f, 1.0f, 1.0f);
}

TEST_CASE("Colour3Construction.ValuesOutsideZeroOne", "[Colour3Construction]")
{
    Colour3 c{-0.5f, 2.0f, 100.0f};
    expectNear(c, -0.5f, 2.0f, 100.0f);
}

TEST_CASE("Colour3Construction.CopyConstruct", "[Colour3Construction]")
{
    Colour3 a{0.1f, 0.2f, 0.3f};
    Colour3 b{a};
    expectNear(b, 0.1f, 0.2f, 0.3f);
}

TEST_CASE("Colour3Construction.ImplicitConversion", "[Colour3Construction]")
{
    // Non-explicit constructor allows implicit brace init
    auto fn = [](const Colour3& c) { return c.r(); };
    CHECK(fn({0.5f, 0.0f, 0.0f}) == Catch::Approx(0.5f).margin(1e-5f));
}

// ==========================================================================
// Getters / Setters
// ==========================================================================

TEST_CASE("Colour3Accessors.GettersReturnCorrectValues", "[Colour3Accessors]")
{
    Colour3 c{0.1f, 0.2f, 0.3f};
    CHECK(c.r() == Catch::Approx(0.1f).margin(1e-5f));
    CHECK(c.g() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(c.b() == Catch::Approx(0.3f).margin(1e-5f));
}

TEST_CASE("Colour3Accessors.SettersModifyValues", "[Colour3Accessors]")
{
    Colour3 c{};
    c.r(0.4f);
    c.g(0.5f);
    c.b(0.6f);
    expectNear(c, 0.4f, 0.5f, 0.6f);
}

TEST_CASE("Colour3Accessors.SettersOverwritePreviousValues", "[Colour3Accessors]")
{
    Colour3 c{0.1f, 0.2f, 0.3f};
    c.r(0.9f);
    expectNear(c, 0.9f, 0.2f, 0.3f);
}

// ==========================================================================
// Equality
// ==========================================================================

TEST_CASE("Colour3Equality.EqualColours", "[Colour3Equality]")
{
    Colour3 a{0.5f, 0.6f, 0.7f};
    Colour3 b{0.5f, 0.6f, 0.7f};
    CHECK(a == b);
}

TEST_CASE("Colour3Equality.DifferentR", "[Colour3Equality]")
{
    Colour3 a{0.1f, 0.5f, 0.5f};
    Colour3 b{0.9f, 0.5f, 0.5f};
    CHECK_FALSE(a == b);
}

TEST_CASE("Colour3Equality.DifferentG", "[Colour3Equality]")
{
    Colour3 a{0.5f, 0.1f, 0.5f};
    Colour3 b{0.5f, 0.9f, 0.5f};
    CHECK_FALSE(a == b);
}

TEST_CASE("Colour3Equality.DifferentB", "[Colour3Equality]")
{
    Colour3 a{0.5f, 0.5f, 0.1f};
    Colour3 b{0.5f, 0.5f, 0.9f};
    CHECK_FALSE(a == b);
}

TEST_CASE("Colour3Equality.ZeroColours", "[Colour3Equality]")
{
    Colour3 a{};
    Colour3 b{};
    CHECK(a == b);
}

TEST_CASE("Colour3Equality.NegativeZeroEqualsPositiveZero", "[Colour3Equality]")
{
    Colour3 a{0.0f, 0.0f, 0.0f};
    Colour3 b{-0.0f, -0.0f, -0.0f};
    CHECK(a == b);
}

TEST_CASE("Colour3Equality.NaNNotEqual", "[Colour3Equality]")
{
    float nan = std::numeric_limits<float>::quiet_NaN();
    Colour3 a{nan, 0.0f, 0.0f};
    Colour3 b{nan, 0.0f, 0.0f};
    CHECK_FALSE(a == b);
}

// ==========================================================================
// Constexpr
// ==========================================================================

TEST_CASE("Colour3Constexpr.ConstructionAndAccessAtCompileTime", "[Colour3Constexpr]")
{
    constexpr Colour3 c{0.1f, 0.2f, 0.3f};
    static_assert(c.r() == 0.1f);
    static_assert(c.g() == 0.2f);
    static_assert(c.b() == 0.3f);
}

TEST_CASE("Colour3Constexpr.DefaultConstructionAtCompileTime", "[Colour3Constexpr]")
{
    constexpr Colour3 c{};
    static_assert(c.r() == 0.0f);
    static_assert(c.g() == 0.0f);
    static_assert(c.b() == 0.0f);
}

TEST_CASE("Colour3Constexpr.EqualityAtCompileTime", "[Colour3Constexpr]")
{
    constexpr Colour3 a{0.5f, 0.5f, 0.5f};
    constexpr Colour3 b{0.5f, 0.5f, 0.5f};
    constexpr Colour3 c{1.0f, 0.0f, 0.0f};
    static_assert(a == b);
    static_assert(!(a == c));
}

// ==========================================================================
// Noexcept guarantees
// ==========================================================================

TEST_CASE("Colour3Noexcept.AllOperationsAreNoexcept", "[Colour3Noexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<Colour3>);
    static_assert(test_traits::nothrow_constructible_from_v<Colour3, float, float, float>);
    static_assert(test_traits::has_nothrow_colour3_accessors<Colour3>);
    static_assert(test_traits::has_nothrow_equality<Colour3>);
}
