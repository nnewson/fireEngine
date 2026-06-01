#include <fire_engine/graphics/colour3.hpp>

#include <limits>

#include <gtest/gtest.h>

using fire_engine::Colour3;

// ---- Helpers ----

static constexpr float kEps = 1e-6f;

static void expectNear(const Colour3& c, float er, float eg, float eb, float eps = kEps)
{
    EXPECT_NEAR(c.r(), er, eps);
    EXPECT_NEAR(c.g(), eg, eps);
    EXPECT_NEAR(c.b(), eb, eps);
}

// ==========================================================================
// Construction
// ==========================================================================

TEST(Colour3Construction, DefaultIsZero)
{
    Colour3 c{};
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST(Colour3Construction, ExplicitValues)
{
    Colour3 c{0.1f, 0.5f, 0.9f};
    expectNear(c, 0.1f, 0.5f, 0.9f);
}

TEST(Colour3Construction, PartialArgs)
{
    Colour3 cr{0.5f};
    expectNear(cr, 0.5f, 0.0f, 0.0f);

    Colour3 crg{0.5f, 0.6f};
    expectNear(crg, 0.5f, 0.6f, 0.0f);
}

TEST(Colour3Construction, White)
{
    Colour3 c{1.0f, 1.0f, 1.0f};
    expectNear(c, 1.0f, 1.0f, 1.0f);
}

TEST(Colour3Construction, ValuesOutsideZeroOne)
{
    Colour3 c{-0.5f, 2.0f, 100.0f};
    expectNear(c, -0.5f, 2.0f, 100.0f);
}

TEST(Colour3Construction, CopyConstruct)
{
    Colour3 a{0.1f, 0.2f, 0.3f};
    Colour3 b{a};
    expectNear(b, 0.1f, 0.2f, 0.3f);
}

TEST(Colour3Construction, ImplicitConversion)
{
    // Non-explicit constructor allows implicit brace init
    auto fn = [](const Colour3& c) { return c.r(); };
    EXPECT_FLOAT_EQ(fn({0.5f, 0.0f, 0.0f}), 0.5f);
}

// ==========================================================================
// Getters / Setters
// ==========================================================================

TEST(Colour3Accessors, GettersReturnCorrectValues)
{
    Colour3 c{0.1f, 0.2f, 0.3f};
    EXPECT_FLOAT_EQ(c.r(), 0.1f);
    EXPECT_FLOAT_EQ(c.g(), 0.2f);
    EXPECT_FLOAT_EQ(c.b(), 0.3f);
}

TEST(Colour3Accessors, SettersModifyValues)
{
    Colour3 c{};
    c.r(0.4f);
    c.g(0.5f);
    c.b(0.6f);
    expectNear(c, 0.4f, 0.5f, 0.6f);
}

TEST(Colour3Accessors, SettersOverwritePreviousValues)
{
    Colour3 c{0.1f, 0.2f, 0.3f};
    c.r(0.9f);
    expectNear(c, 0.9f, 0.2f, 0.3f);
}

// ==========================================================================
// Equality
// ==========================================================================

TEST(Colour3Equality, EqualColours)
{
    Colour3 a{0.5f, 0.6f, 0.7f};
    Colour3 b{0.5f, 0.6f, 0.7f};
    EXPECT_TRUE(a == b);
}

TEST(Colour3Equality, DifferentR)
{
    Colour3 a{0.1f, 0.5f, 0.5f};
    Colour3 b{0.9f, 0.5f, 0.5f};
    EXPECT_FALSE(a == b);
}

TEST(Colour3Equality, DifferentG)
{
    Colour3 a{0.5f, 0.1f, 0.5f};
    Colour3 b{0.5f, 0.9f, 0.5f};
    EXPECT_FALSE(a == b);
}

TEST(Colour3Equality, DifferentB)
{
    Colour3 a{0.5f, 0.5f, 0.1f};
    Colour3 b{0.5f, 0.5f, 0.9f};
    EXPECT_FALSE(a == b);
}

TEST(Colour3Equality, ZeroColours)
{
    Colour3 a{};
    Colour3 b{};
    EXPECT_TRUE(a == b);
}

TEST(Colour3Equality, NegativeZeroEqualsPositiveZero)
{
    Colour3 a{0.0f, 0.0f, 0.0f};
    Colour3 b{-0.0f, -0.0f, -0.0f};
    EXPECT_TRUE(a == b);
}

TEST(Colour3Equality, NaNNotEqual)
{
    float nan = std::numeric_limits<float>::quiet_NaN();
    Colour3 a{nan, 0.0f, 0.0f};
    Colour3 b{nan, 0.0f, 0.0f};
    EXPECT_FALSE(a == b);
}

// ==========================================================================
// Constexpr
// ==========================================================================

TEST(Colour3Constexpr, ConstructionAndAccessAtCompileTime)
{
    constexpr Colour3 c{0.1f, 0.2f, 0.3f};
    static_assert(c.r() == 0.1f);
    static_assert(c.g() == 0.2f);
    static_assert(c.b() == 0.3f);
}

TEST(Colour3Constexpr, DefaultConstructionAtCompileTime)
{
    constexpr Colour3 c{};
    static_assert(c.r() == 0.0f);
    static_assert(c.g() == 0.0f);
    static_assert(c.b() == 0.0f);
}

TEST(Colour3Constexpr, EqualityAtCompileTime)
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

TEST(Colour3Noexcept, AllOperationsAreNoexcept)
{
    Colour3 a{0.1f, 0.2f, 0.3f};

    static_assert(noexcept(Colour3{}));
    static_assert(noexcept(Colour3{0.1f, 0.2f, 0.3f}));
    static_assert(noexcept(a.r()));
    static_assert(noexcept(a.g()));
    static_assert(noexcept(a.b()));
    static_assert(noexcept(a.r(0.1f)));
    static_assert(noexcept(a.g(0.1f)));
    static_assert(noexcept(a.b(0.1f)));
    static_assert(noexcept(a == a));
}
