#include <fire_engine/math/vec2.hpp>

#include <cmath>

#include <gtest/gtest.h>

using fire_engine::Vec2;

// ---- Helpers ----

static constexpr float kEps = 1e-6f;

static void expectNear(const Vec2& v, float ex, float ey, float eps = kEps)
{
    EXPECT_NEAR(v.s(), ex, eps);
    EXPECT_NEAR(v.t(), ey, eps);
}

// ==========================================================================
// Construction
// ==========================================================================

TEST(Vec2Construction, DefaultIsZero)
{
    Vec2 v{};
    expectNear(v, 0.0f, 0.0f);
}

TEST(Vec2Construction, ExplicitValues)
{
    Vec2 v{1.0f, 2.0f};
    expectNear(v, 1.0f, 2.0f);
}

TEST(Vec2Construction, PartialArgs)
{
    Vec2 vx{5.0f};
    expectNear(vx, 5.0f, 0.0f);
}

TEST(Vec2Construction, NegativeValues)
{
    Vec2 v{-1.0f, -2.0f};
    expectNear(v, -1.0f, -2.0f);
}

TEST(Vec2Construction, LargeValues)
{
    Vec2 v{1e10f, -1e10f};
    expectNear(v, 1e10f, -1e10f);
}

TEST(Vec2Construction, Constexpr)
{
    constexpr Vec2 v{3.0f, 4.0f};
    static_assert(v.s() == 3.0f);
    static_assert(v.t() == 4.0f);
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST(Vec2Accessors, GettersReturnComponents)
{
    Vec2 v{7.0f, 8.0f};
    EXPECT_FLOAT_EQ(v.s(), 7.0f);
    EXPECT_FLOAT_EQ(v.t(), 8.0f);
}

TEST(Vec2Accessors, SettersModifyComponents)
{
    Vec2 v{};
    v.s(10.0f);
    v.t(20.0f);
    expectNear(v, 10.0f, 20.0f);
}

TEST(Vec2Accessors, SetterGetterRoundTrip)
{
    Vec2 v{};
    v.s(42.0f);
    EXPECT_FLOAT_EQ(v.s(), 42.0f);
    v.t(-1.0f);
    EXPECT_FLOAT_EQ(v.t(), -1.0f);
}

// ==========================================================================
// Equality
// ==========================================================================

TEST(Vec2Equality, EqualVectors)
{
    EXPECT_EQ((Vec2{1.0f, 2.0f}), (Vec2{1.0f, 2.0f}));
}

TEST(Vec2Equality, DifferentVectors)
{
    EXPECT_NE((Vec2{1.0f, 2.0f}), (Vec2{1.0f, 3.0f}));
    EXPECT_NE((Vec2{1.0f, 2.0f}), (Vec2{0.0f, 2.0f}));
}

TEST(Vec2Equality, DefaultEqualDefault)
{
    EXPECT_EQ(Vec2{}, Vec2{});
}

TEST(Vec2Equality, Constexpr)
{
    constexpr Vec2 a{1.0f, 2.0f};
    constexpr Vec2 b{1.0f, 2.0f};
    static_assert(a == b);
}

// ==========================================================================
// Arithmetic — Addition
// ==========================================================================

TEST(Vec2Addition, BasicAdd)
{
    Vec2 a{1.0f, 2.0f};
    Vec2 b{3.0f, 4.0f};
    expectNear(a + b, 4.0f, 6.0f);
}

TEST(Vec2Addition, CompoundAdd)
{
    Vec2 a{1.0f, 2.0f};
    a += Vec2{10.0f, 20.0f};
    expectNear(a, 11.0f, 22.0f);
}

TEST(Vec2Addition, AddZero)
{
    Vec2 a{5.0f, 6.0f};
    expectNear(a + Vec2{}, 5.0f, 6.0f);
}

TEST(Vec2Addition, Constexpr)
{
    constexpr Vec2 r = Vec2{1.0f, 2.0f} + Vec2{3.0f, 4.0f};
    static_assert(r.s() == 4.0f);
    static_assert(r.t() == 6.0f);
}

// ==========================================================================
// Arithmetic — Subtraction
// ==========================================================================

TEST(Vec2Subtraction, BasicSub)
{
    Vec2 a{5.0f, 6.0f};
    Vec2 b{1.0f, 2.0f};
    expectNear(a - b, 4.0f, 4.0f);
}

TEST(Vec2Subtraction, CompoundSub)
{
    Vec2 v{10.0f, 20.0f};
    v -= Vec2{3.0f, 4.0f};
    expectNear(v, 7.0f, 16.0f);
}

TEST(Vec2Subtraction, SubSelf)
{
    Vec2 v{7.0f, 8.0f};
    expectNear(v - v, 0.0f, 0.0f);
}

// ==========================================================================
// Arithmetic — Scalar Multiply
// ==========================================================================

TEST(Vec2ScalarMultiply, BasicMul)
{
    Vec2 v{1.0f, 2.0f};
    expectNear(v * 3.0f, 3.0f, 6.0f);
}

TEST(Vec2ScalarMultiply, CompoundMul)
{
    Vec2 v{2.0f, 3.0f};
    v *= 4.0f;
    expectNear(v, 8.0f, 12.0f);
}

TEST(Vec2ScalarMultiply, MulZero)
{
    Vec2 v{5.0f, 6.0f};
    expectNear(v * 0.0f, 0.0f, 0.0f);
}

TEST(Vec2ScalarMultiply, MulNegative)
{
    Vec2 v{1.0f, -2.0f};
    expectNear(v * -1.0f, -1.0f, 2.0f);
}

// ==========================================================================
// Arithmetic — Scalar Divide
// ==========================================================================

TEST(Vec2ScalarDivide, BasicDiv)
{
    Vec2 v{6.0f, 8.0f};
    expectNear(v / 2.0f, 3.0f, 4.0f);
}

TEST(Vec2ScalarDivide, CompoundDiv)
{
    Vec2 v{10.0f, 20.0f};
    v /= 5.0f;
    expectNear(v, 2.0f, 4.0f);
}

TEST(Vec2ScalarDivide, DivOne)
{
    Vec2 v{3.0f, 4.0f};
    expectNear(v / 1.0f, 3.0f, 4.0f);
}

// ==========================================================================
// Dot Product
// ==========================================================================

TEST(Vec2DotProduct, StaticMethod)
{
    Vec2 a{1.0f, 0.0f};
    Vec2 b{0.0f, 1.0f};
    EXPECT_FLOAT_EQ(Vec2::dotProduct(a, b), 0.0f);
}

TEST(Vec2DotProduct, MemberMethod)
{
    Vec2 a{3.0f, 4.0f};
    Vec2 b{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(a.dotProduct(b), 25.0f);
}

TEST(Vec2DotProduct, Perpendicular)
{
    Vec2 a{1.0f, 0.0f};
    Vec2 b{0.0f, 1.0f};
    EXPECT_FLOAT_EQ(Vec2::dotProduct(a, b), 0.0f);
}

TEST(Vec2DotProduct, Parallel)
{
    Vec2 a{2.0f, 0.0f};
    Vec2 b{3.0f, 0.0f};
    EXPECT_FLOAT_EQ(Vec2::dotProduct(a, b), 6.0f);
}

TEST(Vec2DotProduct, Constexpr)
{
    constexpr Vec2 a{1.0f, 2.0f};
    constexpr Vec2 b{3.0f, 4.0f};
    constexpr float d = Vec2::dotProduct(a, b);
    static_assert(d == 11.0f);
}

// ==========================================================================
// Magnitude
// ==========================================================================

TEST(Vec2Magnitude, UnitX)
{
    Vec2 v{1.0f, 0.0f};
    EXPECT_FLOAT_EQ(v.magnitude(), 1.0f);
}

TEST(Vec2Magnitude, UnitY)
{
    Vec2 v{0.0f, 1.0f};
    EXPECT_FLOAT_EQ(v.magnitude(), 1.0f);
}

TEST(Vec2Magnitude, ThreeFour)
{
    Vec2 v{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(v.magnitude(), 5.0f);
}

TEST(Vec2Magnitude, Zero)
{
    Vec2 v{};
    EXPECT_FLOAT_EQ(v.magnitude(), 0.0f);
}

TEST(Vec2Magnitude, Squared)
{
    Vec2 v{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(v.magnitudeSquared(), 25.0f);
}

// ==========================================================================
// Normalise
// ==========================================================================

TEST(Vec2Normalise, StaticMethod)
{
    Vec2 v{3.0f, 4.0f};
    Vec2 n = Vec2::normalise(v);
    EXPECT_NEAR(n.magnitude(), 1.0f, kEps);
    expectNear(n, 3.0f / 5.0f, 4.0f / 5.0f);
}

TEST(Vec2Normalise, MemberMethod)
{
    Vec2 v{0.0f, 5.0f};
    v.normalise();
    expectNear(v, 0.0f, 1.0f);
}

TEST(Vec2Normalise, ZeroVectorStaysZero)
{
    Vec2 n = Vec2::normalise(Vec2{});
    expectNear(n, 0.0f, 0.0f);
}

TEST(Vec2Normalise, AlreadyUnit)
{
    Vec2 v{1.0f, 0.0f};
    Vec2 n = Vec2::normalise(v);
    expectNear(n, 1.0f, 0.0f);
}

TEST(Vec2Normalise, NegativeComponents)
{
    Vec2 v{-3.0f, -4.0f};
    Vec2 n = Vec2::normalise(v);
    EXPECT_NEAR(n.magnitude(), 1.0f, kEps);
}

// ==========================================================================
// Copy and Move
// ==========================================================================

TEST(Vec2Copy, CopyConstruct)
{
    Vec2 a{1.0f, 2.0f};
    Vec2 b{a};
    expectNear(b, 1.0f, 2.0f);
}

TEST(Vec2Copy, CopyAssign)
{
    Vec2 a{1.0f, 2.0f};
    Vec2 b{};
    b = a;
    expectNear(b, 1.0f, 2.0f);
}

TEST(Vec2Move, MoveConstruct)
{
    Vec2 a{3.0f, 4.0f};
    Vec2 b{std::move(a)};
    expectNear(b, 3.0f, 4.0f);
}

TEST(Vec2Move, MoveAssign)
{
    Vec2 a{5.0f, 6.0f};
    Vec2 b{};
    b = std::move(a);
    expectNear(b, 5.0f, 6.0f);
}

// ==========================================================================
// Noexcept
// ==========================================================================

TEST(Vec2Noexcept, Construction)
{
    static_assert(noexcept(Vec2{}));
    static_assert(noexcept(Vec2{1.0f, 2.0f}));
}

TEST(Vec2Noexcept, Accessors)
{
    Vec2 v{};
    static_assert(noexcept(v.s()));
    static_assert(noexcept(v.t()));
    static_assert(noexcept(v.s(1.0f)));
    static_assert(noexcept(v.t(1.0f)));
}

TEST(Vec2Noexcept, Arithmetic)
{
    Vec2 a{}, b{};
    static_assert(noexcept(a + b));
    static_assert(noexcept(a - b));
    static_assert(noexcept(a * 1.0f));
    static_assert(noexcept(a / 1.0f));
    static_assert(noexcept(a += b));
    static_assert(noexcept(a -= b));
    static_assert(noexcept(a *= 1.0f));
    static_assert(noexcept(a /= 1.0f));
}

TEST(Vec2Noexcept, DotMagnitudeNormalise)
{
    Vec2 v{1.0f, 0.0f};
    static_assert(noexcept(Vec2::dotProduct(v, v)));
    static_assert(noexcept(v.dotProduct(v)));
    static_assert(noexcept(v.magnitude()));
    static_assert(noexcept(v.magnitudeSquared()));
    static_assert(noexcept(Vec2::normalise(v)));
    static_assert(noexcept(v.normalise()));
}

TEST(Vec2Noexcept, Equality)
{
    Vec2 a{}, b{};
    static_assert(noexcept(a == b));
}

// ==========================================================================
// Edge Cases
// ==========================================================================

TEST(Vec2EdgeCases, VerySmallValues)
{
    Vec2 v{1e-20f, 1e-20f};
    EXPECT_NEAR(v.magnitude(), std::sqrt(2e-40f), 1e-25f);
}

TEST(Vec2EdgeCases, MixedSignDot)
{
    Vec2 a{1.0f, -1.0f};
    Vec2 b{-1.0f, 1.0f};
    EXPECT_FLOAT_EQ(Vec2::dotProduct(a, b), -2.0f);
}
