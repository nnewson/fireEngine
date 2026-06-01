#include <fire_engine/math/vec4.hpp>

#include <cmath>

#include <gtest/gtest.h>

using fire_engine::Vec4;

// ---- Helpers ----

static constexpr float kEps = 1e-6f;

static void expectNear(const Vec4& v, float ex, float ey, float ez, float ew, float eps = kEps)
{
    EXPECT_NEAR(v.x(), ex, eps);
    EXPECT_NEAR(v.y(), ey, eps);
    EXPECT_NEAR(v.z(), ez, eps);
    EXPECT_NEAR(v.w(), ew, eps);
}

// ==========================================================================
// Construction
// ==========================================================================

TEST(Vec4Construction, DefaultIsZero)
{
    Vec4 v{};
    expectNear(v, 0.0f, 0.0f, 0.0f, 0.0f);
}

TEST(Vec4Construction, ExplicitValues)
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(v, 1.0f, 2.0f, 3.0f, 4.0f);
}

TEST(Vec4Construction, PartialArgs)
{
    Vec4 vx{5.0f};
    expectNear(vx, 5.0f, 0.0f, 0.0f, 0.0f);

    Vec4 vxy{5.0f, 6.0f};
    expectNear(vxy, 5.0f, 6.0f, 0.0f, 0.0f);

    Vec4 vxyz{5.0f, 6.0f, 7.0f};
    expectNear(vxyz, 5.0f, 6.0f, 7.0f, 0.0f);
}

TEST(Vec4Construction, NegativeValues)
{
    Vec4 v{-1.0f, -2.0f, -3.0f, -4.0f};
    expectNear(v, -1.0f, -2.0f, -3.0f, -4.0f);
}

TEST(Vec4Construction, LargeValues)
{
    Vec4 v{1e10f, -1e10f, 1e10f, -1e10f};
    expectNear(v, 1e10f, -1e10f, 1e10f, -1e10f);
}

TEST(Vec4Construction, Constexpr)
{
    constexpr Vec4 v{3.0f, 4.0f, 5.0f, 6.0f};
    static_assert(v.x() == 3.0f);
    static_assert(v.y() == 4.0f);
    static_assert(v.z() == 5.0f);
    static_assert(v.w() == 6.0f);
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST(Vec4Accessors, GettersReturnComponents)
{
    Vec4 v{7.0f, 8.0f, 9.0f, 10.0f};
    EXPECT_FLOAT_EQ(v.x(), 7.0f);
    EXPECT_FLOAT_EQ(v.y(), 8.0f);
    EXPECT_FLOAT_EQ(v.z(), 9.0f);
    EXPECT_FLOAT_EQ(v.w(), 10.0f);
}

TEST(Vec4Accessors, SettersModifyComponents)
{
    Vec4 v{};
    v.x(10.0f);
    v.y(20.0f);
    v.z(30.0f);
    v.w(40.0f);
    expectNear(v, 10.0f, 20.0f, 30.0f, 40.0f);
}

TEST(Vec4Accessors, SetterGetterRoundTrip)
{
    Vec4 v{};
    v.x(42.0f);
    EXPECT_FLOAT_EQ(v.x(), 42.0f);
    v.w(-1.0f);
    EXPECT_FLOAT_EQ(v.w(), -1.0f);
}

// ==========================================================================
// Equality
// ==========================================================================

TEST(Vec4Equality, EqualVectors)
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(a, b);
}

TEST(Vec4Equality, DifferentVectors)
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{1.0f, 2.0f, 3.0f, 5.0f};
    Vec4 c{0.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
}

TEST(Vec4Equality, DefaultEqualDefault)
{
    EXPECT_EQ(Vec4{}, Vec4{});
}

TEST(Vec4Equality, Constexpr)
{
    constexpr Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr Vec4 b{1.0f, 2.0f, 3.0f, 4.0f};
    static_assert(a == b);
}

// ==========================================================================
// Arithmetic — Addition
// ==========================================================================

TEST(Vec4Addition, BasicAdd)
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{5.0f, 6.0f, 7.0f, 8.0f};
    expectNear(a + b, 6.0f, 8.0f, 10.0f, 12.0f);
}

TEST(Vec4Addition, CompoundAdd)
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    a += Vec4{10.0f, 20.0f, 30.0f, 40.0f};
    expectNear(a, 11.0f, 22.0f, 33.0f, 44.0f);
}

TEST(Vec4Addition, AddZero)
{
    Vec4 a{5.0f, 6.0f, 7.0f, 8.0f};
    expectNear(a + Vec4{}, 5.0f, 6.0f, 7.0f, 8.0f);
}

TEST(Vec4Addition, Constexpr)
{
    constexpr Vec4 r = Vec4{1.0f, 2.0f, 3.0f, 4.0f} + Vec4{5.0f, 6.0f, 7.0f, 8.0f};
    static_assert(r.x() == 6.0f);
    static_assert(r.w() == 12.0f);
}

// ==========================================================================
// Arithmetic — Subtraction
// ==========================================================================

TEST(Vec4Subtraction, BasicSub)
{
    Vec4 a{5.0f, 6.0f, 7.0f, 8.0f};
    Vec4 b{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(a - b, 4.0f, 4.0f, 4.0f, 4.0f);
}

TEST(Vec4Subtraction, CompoundSub)
{
    Vec4 v{10.0f, 20.0f, 30.0f, 40.0f};
    v -= Vec4{3.0f, 4.0f, 5.0f, 6.0f};
    expectNear(v, 7.0f, 16.0f, 25.0f, 34.0f);
}

TEST(Vec4Subtraction, SubSelf)
{
    Vec4 v{7.0f, 8.0f, 9.0f, 10.0f};
    expectNear(v - v, 0.0f, 0.0f, 0.0f, 0.0f);
}

// ==========================================================================
// Arithmetic — Scalar Multiply
// ==========================================================================

TEST(Vec4ScalarMultiply, BasicMul)
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(v * 3.0f, 3.0f, 6.0f, 9.0f, 12.0f);
}

TEST(Vec4ScalarMultiply, CompoundMul)
{
    Vec4 v{2.0f, 3.0f, 4.0f, 5.0f};
    v *= 4.0f;
    expectNear(v, 8.0f, 12.0f, 16.0f, 20.0f);
}

TEST(Vec4ScalarMultiply, MulZero)
{
    Vec4 v{5.0f, 6.0f, 7.0f, 8.0f};
    expectNear(v * 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

TEST(Vec4ScalarMultiply, MulNegative)
{
    Vec4 v{1.0f, -2.0f, 3.0f, -4.0f};
    expectNear(v * -1.0f, -1.0f, 2.0f, -3.0f, 4.0f);
}

// ==========================================================================
// Arithmetic — Scalar Divide
// ==========================================================================

TEST(Vec4ScalarDivide, BasicDiv)
{
    Vec4 v{6.0f, 8.0f, 10.0f, 12.0f};
    expectNear(v / 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
}

TEST(Vec4ScalarDivide, CompoundDiv)
{
    Vec4 v{10.0f, 20.0f, 30.0f, 40.0f};
    v /= 5.0f;
    expectNear(v, 2.0f, 4.0f, 6.0f, 8.0f);
}

TEST(Vec4ScalarDivide, DivOne)
{
    Vec4 v{3.0f, 4.0f, 5.0f, 6.0f};
    expectNear(v / 1.0f, 3.0f, 4.0f, 5.0f, 6.0f);
}

// ==========================================================================
// Dot Product
// ==========================================================================

TEST(Vec4DotProduct, StaticMethod)
{
    Vec4 a{1.0f, 0.0f, 0.0f, 0.0f};
    Vec4 b{0.0f, 1.0f, 0.0f, 0.0f};
    EXPECT_FLOAT_EQ(Vec4::dotProduct(a, b), 0.0f);
}

TEST(Vec4DotProduct, MemberMethod)
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_FLOAT_EQ(a.dotProduct(b), 30.0f);
}

TEST(Vec4DotProduct, Perpendicular)
{
    Vec4 a{1.0f, 0.0f, 0.0f, 0.0f};
    Vec4 b{0.0f, 0.0f, 1.0f, 0.0f};
    EXPECT_FLOAT_EQ(Vec4::dotProduct(a, b), 0.0f);
}

TEST(Vec4DotProduct, Constexpr)
{
    constexpr Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr Vec4 b{5.0f, 6.0f, 7.0f, 8.0f};
    constexpr float d = Vec4::dotProduct(a, b);
    static_assert(d == 70.0f);
}

// ==========================================================================
// Magnitude
// ==========================================================================

TEST(Vec4Magnitude, UnitX)
{
    Vec4 v{1.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_FLOAT_EQ(v.magnitude(), 1.0f);
}

TEST(Vec4Magnitude, UnitW)
{
    Vec4 v{0.0f, 0.0f, 0.0f, 1.0f};
    EXPECT_FLOAT_EQ(v.magnitude(), 1.0f);
}

TEST(Vec4Magnitude, KnownValue)
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_NEAR(v.magnitude(), std::sqrt(30.0f), kEps);
}

TEST(Vec4Magnitude, Zero)
{
    Vec4 v{};
    EXPECT_FLOAT_EQ(v.magnitude(), 0.0f);
}

TEST(Vec4Magnitude, Squared)
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_FLOAT_EQ(v.magnitudeSquared(), 30.0f);
}

// ==========================================================================
// Normalise
// ==========================================================================

TEST(Vec4Normalise, StaticMethod)
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 n = Vec4::normalise(v);
    EXPECT_NEAR(n.magnitude(), 1.0f, kEps);
}

TEST(Vec4Normalise, MemberMethod)
{
    Vec4 v{0.0f, 0.0f, 0.0f, 5.0f};
    v.normalise();
    expectNear(v, 0.0f, 0.0f, 0.0f, 1.0f);
}

TEST(Vec4Normalise, ZeroVectorStaysZero)
{
    Vec4 n = Vec4::normalise(Vec4{});
    expectNear(n, 0.0f, 0.0f, 0.0f, 0.0f);
}

TEST(Vec4Normalise, AlreadyUnit)
{
    Vec4 v{1.0f, 0.0f, 0.0f, 0.0f};
    Vec4 n = Vec4::normalise(v);
    expectNear(n, 1.0f, 0.0f, 0.0f, 0.0f);
}

TEST(Vec4Normalise, NegativeComponents)
{
    Vec4 v{-1.0f, -2.0f, -3.0f, -4.0f};
    Vec4 n = Vec4::normalise(v);
    EXPECT_NEAR(n.magnitude(), 1.0f, kEps);
}

// ==========================================================================
// Copy and Move
// ==========================================================================

TEST(Vec4Copy, CopyConstruct)
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{a};
    expectNear(b, 1.0f, 2.0f, 3.0f, 4.0f);
}

TEST(Vec4Copy, CopyAssign)
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{};
    b = a;
    expectNear(b, 1.0f, 2.0f, 3.0f, 4.0f);
}

TEST(Vec4Move, MoveConstruct)
{
    Vec4 a{3.0f, 4.0f, 5.0f, 6.0f};
    Vec4 b{std::move(a)};
    expectNear(b, 3.0f, 4.0f, 5.0f, 6.0f);
}

TEST(Vec4Move, MoveAssign)
{
    Vec4 a{5.0f, 6.0f, 7.0f, 8.0f};
    Vec4 b{};
    b = std::move(a);
    expectNear(b, 5.0f, 6.0f, 7.0f, 8.0f);
}

// ==========================================================================
// Noexcept
// ==========================================================================

TEST(Vec4Noexcept, Construction)
{
    static_assert(noexcept(Vec4{}));
    static_assert(noexcept(Vec4{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(Vec4Noexcept, Accessors)
{
    Vec4 v{};
    static_assert(noexcept(v.x()));
    static_assert(noexcept(v.y()));
    static_assert(noexcept(v.z()));
    static_assert(noexcept(v.w()));
    static_assert(noexcept(v.x(1.0f)));
    static_assert(noexcept(v.y(1.0f)));
    static_assert(noexcept(v.z(1.0f)));
    static_assert(noexcept(v.w(1.0f)));
}

TEST(Vec4Noexcept, Arithmetic)
{
    Vec4 a{}, b{};
    static_assert(noexcept(a + b));
    static_assert(noexcept(a - b));
    static_assert(noexcept(a * 1.0f));
    static_assert(noexcept(a / 1.0f));
    static_assert(noexcept(a += b));
    static_assert(noexcept(a -= b));
    static_assert(noexcept(a *= 1.0f));
    static_assert(noexcept(a /= 1.0f));
}

TEST(Vec4Noexcept, DotMagnitudeNormalise)
{
    Vec4 v{1.0f, 0.0f, 0.0f, 0.0f};
    static_assert(noexcept(Vec4::dotProduct(v, v)));
    static_assert(noexcept(v.dotProduct(v)));
    static_assert(noexcept(v.magnitude()));
    static_assert(noexcept(v.magnitudeSquared()));
    static_assert(noexcept(Vec4::normalise(v)));
    static_assert(noexcept(v.normalise()));
}

TEST(Vec4Noexcept, Equality)
{
    Vec4 a{}, b{};
    static_assert(noexcept(a == b));
}

// ==========================================================================
// Edge Cases
// ==========================================================================

TEST(Vec4EdgeCases, VerySmallValues)
{
    Vec4 v{1e-20f, 1e-20f, 1e-20f, 1e-20f};
    EXPECT_NEAR(v.magnitude(), 2e-20f, 1e-25f);
}

TEST(Vec4EdgeCases, MixedSignDot)
{
    Vec4 a{1.0f, -1.0f, 1.0f, -1.0f};
    Vec4 b{-1.0f, 1.0f, -1.0f, 1.0f};
    EXPECT_FLOAT_EQ(Vec4::dotProduct(a, b), -4.0f);
}
