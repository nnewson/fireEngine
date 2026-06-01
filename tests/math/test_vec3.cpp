#include <fire_engine/math/vec3.hpp>

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

using fire_engine::Vec3;

// ---- Helpers ----

static constexpr float kEps = 1e-6f;

static void expectNear(const Vec3& v, float ex, float ey, float ez, float eps = kEps)
{
    EXPECT_NEAR(v.x(), ex, eps);
    EXPECT_NEAR(v.y(), ey, eps);
    EXPECT_NEAR(v.z(), ez, eps);
}

// ==========================================================================
// Construction
// ==========================================================================

TEST(Vec3Construction, DefaultIsZero)
{
    Vec3 v{};
    expectNear(v, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3Construction, ExplicitValues)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    expectNear(v, 1.0f, 2.0f, 3.0f);
}

TEST(Vec3Construction, PartialArgs)
{
    Vec3 vx{5.0f};
    expectNear(vx, 5.0f, 0.0f, 0.0f);

    Vec3 vxy{5.0f, 6.0f};
    expectNear(vxy, 5.0f, 6.0f, 0.0f);
}

TEST(Vec3Construction, NegativeValues)
{
    Vec3 v{-1.0f, -2.0f, -3.0f};
    expectNear(v, -1.0f, -2.0f, -3.0f);
}

TEST(Vec3Construction, CopyConstruct)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{a};
    expectNear(b, 1.0f, 2.0f, 3.0f);
}

TEST(Vec3Construction, MoveConstruct)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{std::move(a)};
    expectNear(b, 1.0f, 2.0f, 3.0f);
}

// ==========================================================================
// Getters / Setters
// ==========================================================================

TEST(Vec3Accessors, GettersReturnCorrectValues)
{
    Vec3 v{4.0f, 5.0f, 6.0f};
    EXPECT_FLOAT_EQ(v.x(), 4.0f);
    EXPECT_FLOAT_EQ(v.y(), 5.0f);
    EXPECT_FLOAT_EQ(v.z(), 6.0f);
}

TEST(Vec3Accessors, SettersModifyValues)
{
    Vec3 v{};
    v.x(10.0f);
    v.y(20.0f);
    v.z(30.0f);
    expectNear(v, 10.0f, 20.0f, 30.0f);
}

TEST(Vec3Accessors, SettersOverwritePreviousValues)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    v.x(-1.0f);
    expectNear(v, -1.0f, 2.0f, 3.0f);
}

// ==========================================================================
// Equality
// ==========================================================================

TEST(Vec3Equality, EqualVectors)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    EXPECT_TRUE(a == b);
}

TEST(Vec3Equality, DifferentX)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{9.0f, 2.0f, 3.0f};
    EXPECT_FALSE(a == b);
}

TEST(Vec3Equality, DifferentY)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 9.0f, 3.0f};
    EXPECT_FALSE(a == b);
}

TEST(Vec3Equality, DifferentZ)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 2.0f, 9.0f};
    EXPECT_FALSE(a == b);
}

TEST(Vec3Equality, ZeroVectors)
{
    Vec3 a{};
    Vec3 b{};
    EXPECT_TRUE(a == b);
}

TEST(Vec3Equality, NegativeZeroEqualsPositiveZero)
{
    Vec3 a{0.0f, 0.0f, 0.0f};
    Vec3 b{-0.0f, -0.0f, -0.0f};
    EXPECT_TRUE(a == b);
}

// ==========================================================================
// Addition
// ==========================================================================

TEST(Vec3Addition, BasicAdd)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    Vec3 c = a + b;
    expectNear(c, 5.0f, 7.0f, 9.0f);
}

TEST(Vec3Addition, AddZero)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 zero{};
    Vec3 c = a + zero;
    expectNear(c, 1.0f, 2.0f, 3.0f);
}

TEST(Vec3Addition, AddNegative)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{-1.0f, -2.0f, -3.0f};
    Vec3 c = a + b;
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3Addition, DoesNotMutateOperands)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    [[maybe_unused]] Vec3 c = a + b;
    expectNear(a, 1.0f, 2.0f, 3.0f);
    expectNear(b, 4.0f, 5.0f, 6.0f);
}

TEST(Vec3Addition, PlusEqualsBasic)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    a += {10.0f, 20.0f, 30.0f};
    expectNear(a, 11.0f, 22.0f, 33.0f);
}

TEST(Vec3Addition, PlusEqualsReturnsSelf)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3& ref = (a += {1.0f, 1.0f, 1.0f});
    EXPECT_EQ(&ref, &a);
}

TEST(Vec3Addition, PlusEqualsSelfAdd)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    a += a;
    expectNear(a, 2.0f, 4.0f, 6.0f);
}

// ==========================================================================
// Subtraction
// ==========================================================================

TEST(Vec3Subtraction, BasicSubtract)
{
    Vec3 a{5.0f, 7.0f, 9.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    Vec3 c = a - b;
    expectNear(c, 4.0f, 5.0f, 6.0f);
}

TEST(Vec3Subtraction, SubtractZero)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 c = a - Vec3{0.0f, 0.0f, 0.0f};
    expectNear(c, 1.0f, 2.0f, 3.0f);
}

TEST(Vec3Subtraction, SubtractSelfIsZero)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 c = a - a;
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3Subtraction, DoesNotMutateOperands)
{
    Vec3 a{5.0f, 6.0f, 7.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    [[maybe_unused]] Vec3 c = a - b;
    expectNear(a, 5.0f, 6.0f, 7.0f);
    expectNear(b, 1.0f, 2.0f, 3.0f);
}

TEST(Vec3Subtraction, MinusEqualsBasic)
{
    Vec3 a{10.0f, 20.0f, 30.0f};
    a -= {1.0f, 2.0f, 3.0f};
    expectNear(a, 9.0f, 18.0f, 27.0f);
}

TEST(Vec3Subtraction, MinusEqualsReturnsSelf)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3& ref = (a -= {1.0f, 1.0f, 1.0f});
    EXPECT_EQ(&ref, &a);
}

TEST(Vec3Subtraction, MinusEqualsSelfSubtract)
{
    Vec3 a{5.0f, 6.0f, 7.0f};
    a -= a;
    expectNear(a, 0.0f, 0.0f, 0.0f);
}

// ==========================================================================
// Scalar Multiply
// ==========================================================================

TEST(Vec3ScalarMultiply, BasicMultiply)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v * 2.0f;
    expectNear(r, 2.0f, 4.0f, 6.0f);
}

TEST(Vec3ScalarMultiply, MultiplyByZero)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v * 0.0f;
    expectNear(r, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3ScalarMultiply, MultiplyByOne)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v * 1.0f;
    expectNear(r, 1.0f, 2.0f, 3.0f);
}

TEST(Vec3ScalarMultiply, MultiplyByNegative)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v * -1.0f;
    expectNear(r, -1.0f, -2.0f, -3.0f);
}

TEST(Vec3ScalarMultiply, MultiplyByFraction)
{
    Vec3 v{4.0f, 6.0f, 8.0f};
    Vec3 r = v * 0.5f;
    expectNear(r, 2.0f, 3.0f, 4.0f);
}

TEST(Vec3ScalarMultiply, DoesNotMutateOperand)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    [[maybe_unused]] Vec3 r = v * 5.0f;
    expectNear(v, 1.0f, 2.0f, 3.0f);
}

TEST(Vec3ScalarMultiply, MultiplyEqualsBasic)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    v *= 3.0f;
    expectNear(v, 3.0f, 6.0f, 9.0f);
}

TEST(Vec3ScalarMultiply, MultiplyEqualsReturnsSelf)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3& ref = (v *= 2.0f);
    EXPECT_EQ(&ref, &v);
}

TEST(Vec3ScalarMultiply, ZeroVector)
{
    Vec3 v{};
    Vec3 r = v * 100.0f;
    expectNear(r, 0.0f, 0.0f, 0.0f);
}

// ==========================================================================
// Scalar Divide
// ==========================================================================

TEST(Vec3ScalarDivide, BasicDivide)
{
    Vec3 v{4.0f, 6.0f, 8.0f};
    Vec3 r = v / 2.0f;
    expectNear(r, 2.0f, 3.0f, 4.0f);
}

TEST(Vec3ScalarDivide, DivideByOne)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v / 1.0f;
    expectNear(r, 1.0f, 2.0f, 3.0f);
}

TEST(Vec3ScalarDivide, DivideByNegative)
{
    Vec3 v{2.0f, 4.0f, 6.0f};
    Vec3 r = v / -2.0f;
    expectNear(r, -1.0f, -2.0f, -3.0f);
}

TEST(Vec3ScalarDivide, DivideByFraction)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v / 0.5f;
    expectNear(r, 2.0f, 4.0f, 6.0f);
}

TEST(Vec3ScalarDivide, DoesNotMutateOperand)
{
    Vec3 v{4.0f, 6.0f, 8.0f};
    [[maybe_unused]] Vec3 r = v / 2.0f;
    expectNear(v, 4.0f, 6.0f, 8.0f);
}

TEST(Vec3ScalarDivide, DivideEqualsBasic)
{
    Vec3 v{9.0f, 6.0f, 3.0f};
    v /= 3.0f;
    expectNear(v, 3.0f, 2.0f, 1.0f);
}

TEST(Vec3ScalarDivide, DivideEqualsReturnsSelf)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3& ref = (v /= 2.0f);
    EXPECT_EQ(&ref, &v);
}

TEST(Vec3ScalarDivide, DivideByZeroProducesInfinity)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v / 0.0f;
    EXPECT_TRUE(std::isinf(r.x()));
    EXPECT_TRUE(std::isinf(r.y()));
    EXPECT_TRUE(std::isinf(r.z()));
}

TEST(Vec3ScalarDivide, ZeroVectorDivide)
{
    Vec3 v{};
    Vec3 r = v / 5.0f;
    expectNear(r, 0.0f, 0.0f, 0.0f);
}

// ==========================================================================
// Dot Product
// ==========================================================================

TEST(Vec3DotProduct, OrthogonalVectorsAreZero)
{
    Vec3 x{1.0f, 0.0f, 0.0f};
    Vec3 y{0.0f, 1.0f, 0.0f};
    EXPECT_NEAR(Vec3::dotProduct(x, y), 0.0f, kEps);
}

TEST(Vec3DotProduct, ParallelUnitVectors)
{
    Vec3 a{1.0f, 0.0f, 0.0f};
    EXPECT_NEAR(Vec3::dotProduct(a, a), 1.0f, kEps);
}

TEST(Vec3DotProduct, AntiParallelUnitVectors)
{
    Vec3 a{1.0f, 0.0f, 0.0f};
    Vec3 b{-1.0f, 0.0f, 0.0f};
    EXPECT_NEAR(Vec3::dotProduct(a, b), -1.0f, kEps);
}

TEST(Vec3DotProduct, GeneralCase)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    // 1*4 + 2*5 + 3*6 = 32
    EXPECT_NEAR(Vec3::dotProduct(a, b), 32.0f, kEps);
}

TEST(Vec3DotProduct, Commutative)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    EXPECT_FLOAT_EQ(Vec3::dotProduct(a, b), Vec3::dotProduct(b, a));
}

TEST(Vec3DotProduct, ZeroVector)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 zero{};
    EXPECT_NEAR(Vec3::dotProduct(a, zero), 0.0f, kEps);
}

TEST(Vec3DotProduct, InstanceMethodMatchesStatic)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    EXPECT_FLOAT_EQ(Vec3::dotProduct(a, b), a.dotProduct(b));
}

TEST(Vec3DotProduct, SelfDotIsMagnitudeSquared)
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    EXPECT_NEAR(v.dotProduct(v), v.magnitudeSquared(), kEps);
}

// ==========================================================================
// Magnitude / MagnitudeSquared
// ==========================================================================

TEST(Vec3Magnitude, UnitX)
{
    Vec3 v{1.0f, 0.0f, 0.0f};
    EXPECT_NEAR(v.magnitude(), 1.0f, kEps);
}

TEST(Vec3Magnitude, ThreeFourFive)
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    EXPECT_NEAR(v.magnitude(), 5.0f, kEps);
}

TEST(Vec3Magnitude, ZeroVector)
{
    Vec3 v{};
    EXPECT_NEAR(v.magnitude(), 0.0f, kEps);
}

TEST(Vec3Magnitude, NegativeComponents)
{
    Vec3 v{-3.0f, -4.0f, 0.0f};
    EXPECT_NEAR(v.magnitude(), 5.0f, kEps);
}

TEST(Vec3Magnitude, AllComponentsEqual)
{
    Vec3 v{1.0f, 1.0f, 1.0f};
    EXPECT_NEAR(v.magnitude(), std::sqrt(3.0f), kEps);
}

TEST(Vec3Magnitude, DoesNotMutate)
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    [[maybe_unused]] float m = v.magnitude();
    expectNear(v, 3.0f, 4.0f, 0.0f);
}

TEST(Vec3MagnitudeSquared, BasicCase)
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    EXPECT_NEAR(v.magnitudeSquared(), 25.0f, kEps);
}

TEST(Vec3MagnitudeSquared, ZeroVector)
{
    Vec3 v{};
    EXPECT_NEAR(v.magnitudeSquared(), 0.0f, kEps);
}

TEST(Vec3MagnitudeSquared, UnitVector)
{
    Vec3 v{0.0f, 1.0f, 0.0f};
    EXPECT_NEAR(v.magnitudeSquared(), 1.0f, kEps);
}

TEST(Vec3MagnitudeSquared, ConsistentWithMagnitude)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    float mag = v.magnitude();
    EXPECT_NEAR(v.magnitudeSquared(), mag * mag, kEps);
}

TEST(Vec3MagnitudeSquared, NegativeComponents)
{
    Vec3 v{-2.0f, -3.0f, -4.0f};
    // 4 + 9 + 16 = 29
    EXPECT_NEAR(v.magnitudeSquared(), 29.0f, kEps);
}

// ==========================================================================
// Cross Product
// ==========================================================================

TEST(Vec3CrossProduct, OrthogonalAxes_XcrossY)
{
    Vec3 x{1.0f, 0.0f, 0.0f};
    Vec3 y{0.0f, 1.0f, 0.0f};
    Vec3 c = Vec3::crossProduct(x, y);
    expectNear(c, 0.0f, 0.0f, 1.0f);
}

TEST(Vec3CrossProduct, OrthogonalAxes_YcrossZ)
{
    Vec3 y{0.0f, 1.0f, 0.0f};
    Vec3 z{0.0f, 0.0f, 1.0f};
    Vec3 c = Vec3::crossProduct(y, z);
    expectNear(c, 1.0f, 0.0f, 0.0f);
}

TEST(Vec3CrossProduct, OrthogonalAxes_ZcrossX)
{
    Vec3 z{0.0f, 0.0f, 1.0f};
    Vec3 x{1.0f, 0.0f, 0.0f};
    Vec3 c = Vec3::crossProduct(z, x);
    expectNear(c, 0.0f, 1.0f, 0.0f);
}

TEST(Vec3CrossProduct, AntiCommutative)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    Vec3 ab = Vec3::crossProduct(a, b);
    Vec3 ba = Vec3::crossProduct(b, a);
    expectNear(ab, -ba.x(), -ba.y(), -ba.z());
}

TEST(Vec3CrossProduct, ParallelVectorsGiveZero)
{
    Vec3 a{2.0f, 4.0f, 6.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    Vec3 c = Vec3::crossProduct(a, b);
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3CrossProduct, SelfCrossIsZero)
{
    Vec3 a{3.0f, 7.0f, 11.0f};
    Vec3 c = Vec3::crossProduct(a, a);
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3CrossProduct, ZeroVectorCross)
{
    Vec3 zero{};
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 c = Vec3::crossProduct(zero, a);
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3CrossProduct, InstanceMethodMatchesStatic)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    Vec3 stat = Vec3::crossProduct(a, b);
    Vec3 inst = a.crossProduct(b);
    EXPECT_TRUE(stat == inst);
}

TEST(Vec3CrossProduct, ResultIsPerpendicularToInputs)
{
    Vec3 a{1.0f, 0.0f, 0.0f};
    Vec3 b{1.0f, 1.0f, 0.0f};
    Vec3 c = Vec3::crossProduct(a, b);
    // dot(a,c) should be 0
    float dotAC = a.x() * c.x() + a.y() * c.y() + a.z() * c.z();
    float dotBC = b.x() * c.x() + b.y() * c.y() + b.z() * c.z();
    EXPECT_NEAR(dotAC, 0.0f, kEps);
    EXPECT_NEAR(dotBC, 0.0f, kEps);
}

// ==========================================================================
// Normalise
// ==========================================================================

TEST(Vec3Normalise, UnitXUnchanged)
{
    Vec3 v{1.0f, 0.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 1.0f, 0.0f, 0.0f);
}

TEST(Vec3Normalise, UnitYUnchanged)
{
    Vec3 v{0.0f, 1.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.0f, 1.0f, 0.0f);
}

TEST(Vec3Normalise, ArbitraryVector)
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.6f, 0.8f, 0.0f);
}

TEST(Vec3Normalise, ResultHasUnitLength)
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 n = Vec3::normalise(v);
    float len = std::sqrt(n.x() * n.x() + n.y() * n.y() + n.z() * n.z());
    EXPECT_NEAR(len, 1.0f, kEps);
}

TEST(Vec3Normalise, NegativeVector)
{
    Vec3 v{-3.0f, -4.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, -0.6f, -0.8f, 0.0f);
}

TEST(Vec3Normalise, ZeroVectorReturnsZero)
{
    Vec3 v{};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3Normalise, NearZeroVectorReturnsZero)
{
    Vec3 v{1e-9f, 1e-9f, 1e-9f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.0f, 0.0f, 0.0f);
}

TEST(Vec3Normalise, VeryLargeVector)
{
    Vec3 v{1e18f, 0.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 1.0f, 0.0f, 0.0f);
}

TEST(Vec3Normalise, StaticDoesNotMutateInput)
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    [[maybe_unused]] Vec3 n = Vec3::normalise(v);
    expectNear(v, 3.0f, 4.0f, 0.0f);
}

TEST(Vec3Normalise, InstanceMethodMutatesSelf)
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    v.normalise();
    expectNear(v, 0.6f, 0.8f, 0.0f);
}

TEST(Vec3Normalise, InstanceMethodReturnsNormalisedValue)
{
    Vec3 v{0.0f, 0.0f, 5.0f};
    Vec3 result = v.normalise();
    expectNear(result, 0.0f, 0.0f, 1.0f);
}

TEST(Vec3Normalise, NormaliseAlreadyUnitVector)
{
    Vec3 v{0.0f, 0.0f, 1.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.0f, 0.0f, 1.0f);
}

TEST(Vec3Normalise, Diagonal)
{
    Vec3 v{1.0f, 1.0f, 1.0f};
    Vec3 n = Vec3::normalise(v);
    float expected = 1.0f / std::sqrt(3.0f);
    expectNear(n, expected, expected, expected);
}

// ==========================================================================
// Noexcept guarantees
// ==========================================================================

TEST(Vec3Noexcept, AllOperationsAreNoexcept)
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};

    static_assert(noexcept(Vec3{}));
    static_assert(noexcept(Vec3{1.0f, 2.0f, 3.0f}));
    static_assert(noexcept(a.x()));
    static_assert(noexcept(a.y()));
    static_assert(noexcept(a.z()));
    static_assert(noexcept(a.x(1.0f)));
    static_assert(noexcept(a.y(1.0f)));
    static_assert(noexcept(a.z(1.0f)));
    static_assert(noexcept(a + b));
    static_assert(noexcept(a - b));
    static_assert(noexcept(a += b));
    static_assert(noexcept(a -= b));
    static_assert(noexcept(a * 2.0f));
    static_assert(noexcept(a *= 2.0f));
    static_assert(noexcept(a / 2.0f));
    static_assert(noexcept(a /= 2.0f));
    static_assert(noexcept(a == b));
    static_assert(noexcept(Vec3::dotProduct(a, b)));
    static_assert(noexcept(a.dotProduct(b)));
    static_assert(noexcept(Vec3::crossProduct(a, b)));
    static_assert(noexcept(a.crossProduct(b)));
    static_assert(noexcept(a.magnitude()));
    static_assert(noexcept(a.magnitudeSquared()));
    static_assert(noexcept(Vec3::normalise(a)));
    static_assert(noexcept(a.normalise()));
}

// ==========================================================================
// Edge cases with special float values
// ==========================================================================

TEST(Vec3EdgeCases, InfinityComponents)
{
    float inf = std::numeric_limits<float>::infinity();
    Vec3 v{inf, -inf, 0.0f};
    EXPECT_EQ(v.x(), inf);
    EXPECT_EQ(v.y(), -inf);
    EXPECT_EQ(v.z(), 0.0f);
}

TEST(Vec3EdgeCases, NaNEquality)
{
    float nan = std::numeric_limits<float>::quiet_NaN();
    Vec3 a{nan, 0.0f, 0.0f};
    Vec3 b{nan, 0.0f, 0.0f};
    // NaN != NaN per IEEE 754
    EXPECT_FALSE(a == b);
}

TEST(Vec3EdgeCases, NormaliseInfinityVector)
{
    float inf = std::numeric_limits<float>::infinity();
    Vec3 v{inf, 0.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    // Result is implementation-defined but should not crash
    (void)n;
}

TEST(Vec3EdgeCases, CrossProductLargeValues)
{
    Vec3 a{1e10f, 0.0f, 0.0f};
    Vec3 b{0.0f, 1e10f, 0.0f};
    Vec3 c = Vec3::crossProduct(a, b);
    // Should point in z direction
    EXPECT_GT(c.z(), 0.0f);
    EXPECT_NEAR(c.x(), 0.0f, kEps);
    EXPECT_NEAR(c.y(), 0.0f, kEps);
}

// ==========================================================================
// Compound expression correctness
// ==========================================================================

TEST(Vec3Compound, ChainedAddition)
{
    Vec3 a{1.0f, 0.0f, 0.0f};
    Vec3 b{0.0f, 1.0f, 0.0f};
    Vec3 c{0.0f, 0.0f, 1.0f};
    Vec3 result = a + b + c;
    expectNear(result, 1.0f, 1.0f, 1.0f);
}

TEST(Vec3Compound, SubtractThenAdd)
{
    Vec3 a{5.0f, 5.0f, 5.0f};
    Vec3 b{3.0f, 2.0f, 1.0f};
    Vec3 c{1.0f, 1.0f, 1.0f};
    Vec3 result = a - b + c;
    expectNear(result, 3.0f, 4.0f, 5.0f);
}

TEST(Vec3Compound, PlusEqualsChained)
{
    Vec3 a{1.0f, 1.0f, 1.0f};
    (a += {1.0f, 0.0f, 0.0f}) += {0.0f, 1.0f, 0.0f};
    expectNear(a, 2.0f, 2.0f, 1.0f);
}
