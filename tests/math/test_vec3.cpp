#include <fire_engine/math/vec3.hpp>

#include <cmath>
#include <limits>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Vec3;

// ---- Helpers ----

static constexpr float kEps = 1e-6f;

static void expectNear(const Vec3& v, float ex, float ey, float ez, float eps = kEps)
{
    CHECK(v.x() == Catch::Approx(ex).margin(eps));
    CHECK(v.y() == Catch::Approx(ey).margin(eps));
    CHECK(v.z() == Catch::Approx(ez).margin(eps));
}

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("Vec3Construction.DefaultIsZero", "[Vec3Construction]")
{
    Vec3 v{};
    expectNear(v, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3Construction.ExplicitValues", "[Vec3Construction]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    expectNear(v, 1.0f, 2.0f, 3.0f);
}

TEST_CASE("Vec3Construction.PartialArgs", "[Vec3Construction]")
{
    Vec3 vx{5.0f};
    expectNear(vx, 5.0f, 0.0f, 0.0f);

    Vec3 vxy{5.0f, 6.0f};
    expectNear(vxy, 5.0f, 6.0f, 0.0f);
}

TEST_CASE("Vec3Construction.NegativeValues", "[Vec3Construction]")
{
    Vec3 v{-1.0f, -2.0f, -3.0f};
    expectNear(v, -1.0f, -2.0f, -3.0f);
}

TEST_CASE("Vec3Construction.CopyConstruct", "[Vec3Construction]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{a};
    expectNear(b, 1.0f, 2.0f, 3.0f);
}

TEST_CASE("Vec3Construction.MoveConstruct", "[Vec3Construction]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{std::move(a)};
    expectNear(b, 1.0f, 2.0f, 3.0f);
}

// ==========================================================================
// Getters / Setters
// ==========================================================================

TEST_CASE("Vec3Accessors.GettersReturnCorrectValues", "[Vec3Accessors]")
{
    Vec3 v{4.0f, 5.0f, 6.0f};
    CHECK(v.x() == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(v.y() == Catch::Approx(5.0f).margin(1e-5f));
    CHECK(v.z() == Catch::Approx(6.0f).margin(1e-5f));
}

TEST_CASE("Vec3Accessors.SettersModifyValues", "[Vec3Accessors]")
{
    Vec3 v{};
    v.x(10.0f);
    v.y(20.0f);
    v.z(30.0f);
    expectNear(v, 10.0f, 20.0f, 30.0f);
}

TEST_CASE("Vec3Accessors.SettersOverwritePreviousValues", "[Vec3Accessors]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    v.x(-1.0f);
    expectNear(v, -1.0f, 2.0f, 3.0f);
}

// ==========================================================================
// Equality
// ==========================================================================

TEST_CASE("Vec3Equality.EqualVectors", "[Vec3Equality]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    CHECK(a == b);
}

TEST_CASE("Vec3Equality.DifferentX", "[Vec3Equality]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{9.0f, 2.0f, 3.0f};
    CHECK_FALSE(a == b);
}

TEST_CASE("Vec3Equality.DifferentY", "[Vec3Equality]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 9.0f, 3.0f};
    CHECK_FALSE(a == b);
}

TEST_CASE("Vec3Equality.DifferentZ", "[Vec3Equality]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 2.0f, 9.0f};
    CHECK_FALSE(a == b);
}

TEST_CASE("Vec3Equality.ZeroVectors", "[Vec3Equality]")
{
    Vec3 a{};
    Vec3 b{};
    CHECK(a == b);
}

TEST_CASE("Vec3Equality.NegativeZeroEqualsPositiveZero", "[Vec3Equality]")
{
    Vec3 a{0.0f, 0.0f, 0.0f};
    Vec3 b{-0.0f, -0.0f, -0.0f};
    CHECK(a == b);
}

TEST_CASE("Vec3Equality.BitwiseEqualMatchesOperator", "[Vec3Equality]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    Vec3 c{1.0f, 2.0f, 3.5f};
    CHECK(a.bitwiseEqual(b));
    CHECK_FALSE(a.bitwiseEqual(c));
}

TEST_CASE("Vec3Equality.ApproxEqualWithinTolerance", "[Vec3Equality]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f + 1e-7f, 2.0f - 1e-7f, 3.0f};
    CHECK_FALSE(a == b);
    CHECK(a.approxEqual(b, 1e-6f));
    CHECK_FALSE(a.approxEqual(b, 1e-9f));
}

TEST_CASE("Vec3Equality.ApproxEqualDefaultEpsilon", "[Vec3Equality]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    CHECK(a.approxEqual(b));
}

// ==========================================================================
// Addition
// ==========================================================================

TEST_CASE("Vec3Addition.BasicAdd", "[Vec3Addition]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    Vec3 c = a + b;
    expectNear(c, 5.0f, 7.0f, 9.0f);
}

TEST_CASE("Vec3Addition.AddZero", "[Vec3Addition]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 zero{};
    Vec3 c = a + zero;
    expectNear(c, 1.0f, 2.0f, 3.0f);
}

TEST_CASE("Vec3Addition.AddNegative", "[Vec3Addition]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{-1.0f, -2.0f, -3.0f};
    Vec3 c = a + b;
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3Addition.DoesNotMutateOperands", "[Vec3Addition]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    [[maybe_unused]] Vec3 c = a + b;
    expectNear(a, 1.0f, 2.0f, 3.0f);
    expectNear(b, 4.0f, 5.0f, 6.0f);
}

TEST_CASE("Vec3Addition.PlusEqualsBasic", "[Vec3Addition]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    a += {10.0f, 20.0f, 30.0f};
    expectNear(a, 11.0f, 22.0f, 33.0f);
}

TEST_CASE("Vec3Addition.PlusEqualsReturnsSelf", "[Vec3Addition]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3& ref = (a += {1.0f, 1.0f, 1.0f});
    CHECK(&ref == &a);
}

TEST_CASE("Vec3Addition.PlusEqualsSelfAdd", "[Vec3Addition]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    a += a;
    expectNear(a, 2.0f, 4.0f, 6.0f);
}

// ==========================================================================
// Subtraction
// ==========================================================================

TEST_CASE("Vec3Subtraction.BasicSubtract", "[Vec3Subtraction]")
{
    Vec3 a{5.0f, 7.0f, 9.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    Vec3 c = a - b;
    expectNear(c, 4.0f, 5.0f, 6.0f);
}

TEST_CASE("Vec3Subtraction.SubtractZero", "[Vec3Subtraction]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 c = a - Vec3{0.0f, 0.0f, 0.0f};
    expectNear(c, 1.0f, 2.0f, 3.0f);
}

TEST_CASE("Vec3Subtraction.SubtractSelfIsZero", "[Vec3Subtraction]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 c = a - a;
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3Subtraction.DoesNotMutateOperands", "[Vec3Subtraction]")
{
    Vec3 a{5.0f, 6.0f, 7.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    [[maybe_unused]] Vec3 c = a - b;
    expectNear(a, 5.0f, 6.0f, 7.0f);
    expectNear(b, 1.0f, 2.0f, 3.0f);
}

TEST_CASE("Vec3Subtraction.MinusEqualsBasic", "[Vec3Subtraction]")
{
    Vec3 a{10.0f, 20.0f, 30.0f};
    a -= {1.0f, 2.0f, 3.0f};
    expectNear(a, 9.0f, 18.0f, 27.0f);
}

TEST_CASE("Vec3Subtraction.MinusEqualsReturnsSelf", "[Vec3Subtraction]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3& ref = (a -= {1.0f, 1.0f, 1.0f});
    CHECK(&ref == &a);
}

TEST_CASE("Vec3Subtraction.MinusEqualsSelfSubtract", "[Vec3Subtraction]")
{
    Vec3 a{5.0f, 6.0f, 7.0f};
    a -= a;
    expectNear(a, 0.0f, 0.0f, 0.0f);
}

// ==========================================================================
// Scalar Multiply
// ==========================================================================

TEST_CASE("Vec3ScalarMultiply.BasicMultiply", "[Vec3ScalarMultiply]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v * 2.0f;
    expectNear(r, 2.0f, 4.0f, 6.0f);
}

TEST_CASE("Vec3ScalarMultiply.MultiplyByZero", "[Vec3ScalarMultiply]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v * 0.0f;
    expectNear(r, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3ScalarMultiply.MultiplyByOne", "[Vec3ScalarMultiply]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v * 1.0f;
    expectNear(r, 1.0f, 2.0f, 3.0f);
}

TEST_CASE("Vec3ScalarMultiply.MultiplyByNegative", "[Vec3ScalarMultiply]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v * -1.0f;
    expectNear(r, -1.0f, -2.0f, -3.0f);
}

TEST_CASE("Vec3ScalarMultiply.MultiplyByFraction", "[Vec3ScalarMultiply]")
{
    Vec3 v{4.0f, 6.0f, 8.0f};
    Vec3 r = v * 0.5f;
    expectNear(r, 2.0f, 3.0f, 4.0f);
}

TEST_CASE("Vec3ScalarMultiply.DoesNotMutateOperand", "[Vec3ScalarMultiply]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    [[maybe_unused]] Vec3 r = v * 5.0f;
    expectNear(v, 1.0f, 2.0f, 3.0f);
}

TEST_CASE("Vec3ScalarMultiply.MultiplyEqualsBasic", "[Vec3ScalarMultiply]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    v *= 3.0f;
    expectNear(v, 3.0f, 6.0f, 9.0f);
}

TEST_CASE("Vec3ScalarMultiply.MultiplyEqualsReturnsSelf", "[Vec3ScalarMultiply]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3& ref = (v *= 2.0f);
    CHECK(&ref == &v);
}

TEST_CASE("Vec3ScalarMultiply.ZeroVector", "[Vec3ScalarMultiply]")
{
    Vec3 v{};
    Vec3 r = v * 100.0f;
    expectNear(r, 0.0f, 0.0f, 0.0f);
}

// ==========================================================================
// Scalar Divide
// ==========================================================================

TEST_CASE("Vec3ScalarDivide.BasicDivide", "[Vec3ScalarDivide]")
{
    Vec3 v{4.0f, 6.0f, 8.0f};
    Vec3 r = v / 2.0f;
    expectNear(r, 2.0f, 3.0f, 4.0f);
}

TEST_CASE("Vec3ScalarDivide.DivideByOne", "[Vec3ScalarDivide]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v / 1.0f;
    expectNear(r, 1.0f, 2.0f, 3.0f);
}

TEST_CASE("Vec3ScalarDivide.DivideByNegative", "[Vec3ScalarDivide]")
{
    Vec3 v{2.0f, 4.0f, 6.0f};
    Vec3 r = v / -2.0f;
    expectNear(r, -1.0f, -2.0f, -3.0f);
}

TEST_CASE("Vec3ScalarDivide.DivideByFraction", "[Vec3ScalarDivide]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v / 0.5f;
    expectNear(r, 2.0f, 4.0f, 6.0f);
}

TEST_CASE("Vec3ScalarDivide.DoesNotMutateOperand", "[Vec3ScalarDivide]")
{
    Vec3 v{4.0f, 6.0f, 8.0f};
    [[maybe_unused]] Vec3 r = v / 2.0f;
    expectNear(v, 4.0f, 6.0f, 8.0f);
}

TEST_CASE("Vec3ScalarDivide.DivideEqualsBasic", "[Vec3ScalarDivide]")
{
    Vec3 v{9.0f, 6.0f, 3.0f};
    v /= 3.0f;
    expectNear(v, 3.0f, 2.0f, 1.0f);
}

TEST_CASE("Vec3ScalarDivide.DivideEqualsReturnsSelf", "[Vec3ScalarDivide]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3& ref = (v /= 2.0f);
    CHECK(&ref == &v);
}

TEST_CASE("Vec3ScalarDivide.DivideByZeroProducesInfinity", "[Vec3ScalarDivide]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 r = v / 0.0f;
    CHECK(std::isinf(r.x()));
    CHECK(std::isinf(r.y()));
    CHECK(std::isinf(r.z()));
}

TEST_CASE("Vec3ScalarDivide.ZeroVectorDivide", "[Vec3ScalarDivide]")
{
    Vec3 v{};
    Vec3 r = v / 5.0f;
    expectNear(r, 0.0f, 0.0f, 0.0f);
}

// ==========================================================================
// Dot Product
// ==========================================================================

TEST_CASE("Vec3DotProduct.OrthogonalVectorsAreZero", "[Vec3DotProduct]")
{
    Vec3 x{1.0f, 0.0f, 0.0f};
    Vec3 y{0.0f, 1.0f, 0.0f};
    CHECK(Vec3::dotProduct(x, y) == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("Vec3DotProduct.ParallelUnitVectors", "[Vec3DotProduct]")
{
    Vec3 a{1.0f, 0.0f, 0.0f};
    CHECK(Vec3::dotProduct(a, a) == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Vec3DotProduct.AntiParallelUnitVectors", "[Vec3DotProduct]")
{
    Vec3 a{1.0f, 0.0f, 0.0f};
    Vec3 b{-1.0f, 0.0f, 0.0f};
    CHECK(Vec3::dotProduct(a, b) == Catch::Approx(-1.0f).margin(kEps));
}

TEST_CASE("Vec3DotProduct.GeneralCase", "[Vec3DotProduct]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    // 1*4 + 2*5 + 3*6 = 32
    CHECK(Vec3::dotProduct(a, b) == Catch::Approx(32.0f).margin(kEps));
}

TEST_CASE("Vec3DotProduct.Commutative", "[Vec3DotProduct]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    CHECK(Vec3::dotProduct(a, b) == Catch::Approx(Vec3::dotProduct(b, a)).margin(1e-5f));
}

TEST_CASE("Vec3DotProduct.ZeroVector", "[Vec3DotProduct]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 zero{};
    CHECK(Vec3::dotProduct(a, zero) == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("Vec3DotProduct.InstanceMethodMatchesStatic", "[Vec3DotProduct]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    CHECK(Vec3::dotProduct(a, b) == Catch::Approx(a.dotProduct(b)).margin(1e-5f));
}

TEST_CASE("Vec3DotProduct.SelfDotIsMagnitudeSquared", "[Vec3DotProduct]")
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    CHECK(v.dotProduct(v) == Catch::Approx(v.magnitudeSquared()).margin(kEps));
}

// ==========================================================================
// Magnitude / MagnitudeSquared
// ==========================================================================

TEST_CASE("Vec3Magnitude.UnitX", "[Vec3Magnitude]")
{
    Vec3 v{1.0f, 0.0f, 0.0f};
    CHECK(v.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Vec3Magnitude.ThreeFourFive", "[Vec3Magnitude]")
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    CHECK(v.magnitude() == Catch::Approx(5.0f).margin(kEps));
}

TEST_CASE("Vec3Magnitude.ZeroVector", "[Vec3Magnitude]")
{
    Vec3 v{};
    CHECK(v.magnitude() == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("Vec3Magnitude.NegativeComponents", "[Vec3Magnitude]")
{
    Vec3 v{-3.0f, -4.0f, 0.0f};
    CHECK(v.magnitude() == Catch::Approx(5.0f).margin(kEps));
}

TEST_CASE("Vec3Magnitude.AllComponentsEqual", "[Vec3Magnitude]")
{
    Vec3 v{1.0f, 1.0f, 1.0f};
    CHECK(v.magnitude() == Catch::Approx(std::sqrt(3.0f)).margin(kEps));
}

TEST_CASE("Vec3Magnitude.DoesNotMutate", "[Vec3Magnitude]")
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    [[maybe_unused]] float m = v.magnitude();
    expectNear(v, 3.0f, 4.0f, 0.0f);
}

TEST_CASE("Vec3MagnitudeSquared.BasicCase", "[Vec3MagnitudeSquared]")
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    CHECK(v.magnitudeSquared() == Catch::Approx(25.0f).margin(kEps));
}

TEST_CASE("Vec3MagnitudeSquared.ZeroVector", "[Vec3MagnitudeSquared]")
{
    Vec3 v{};
    CHECK(v.magnitudeSquared() == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("Vec3MagnitudeSquared.UnitVector", "[Vec3MagnitudeSquared]")
{
    Vec3 v{0.0f, 1.0f, 0.0f};
    CHECK(v.magnitudeSquared() == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Vec3MagnitudeSquared.ConsistentWithMagnitude", "[Vec3MagnitudeSquared]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    float mag = v.magnitude();
    CHECK(v.magnitudeSquared() == Catch::Approx(mag * mag).margin(kEps));
}

TEST_CASE("Vec3MagnitudeSquared.NegativeComponents", "[Vec3MagnitudeSquared]")
{
    Vec3 v{-2.0f, -3.0f, -4.0f};
    // 4 + 9 + 16 = 29
    CHECK(v.magnitudeSquared() == Catch::Approx(29.0f).margin(kEps));
}

// ==========================================================================
// Cross Product
// ==========================================================================

TEST_CASE("Vec3CrossProduct.OrthogonalAxes_XcrossY", "[Vec3CrossProduct]")
{
    Vec3 x{1.0f, 0.0f, 0.0f};
    Vec3 y{0.0f, 1.0f, 0.0f};
    Vec3 c = Vec3::crossProduct(x, y);
    expectNear(c, 0.0f, 0.0f, 1.0f);
}

TEST_CASE("Vec3CrossProduct.OrthogonalAxes_YcrossZ", "[Vec3CrossProduct]")
{
    Vec3 y{0.0f, 1.0f, 0.0f};
    Vec3 z{0.0f, 0.0f, 1.0f};
    Vec3 c = Vec3::crossProduct(y, z);
    expectNear(c, 1.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3CrossProduct.OrthogonalAxes_ZcrossX", "[Vec3CrossProduct]")
{
    Vec3 z{0.0f, 0.0f, 1.0f};
    Vec3 x{1.0f, 0.0f, 0.0f};
    Vec3 c = Vec3::crossProduct(z, x);
    expectNear(c, 0.0f, 1.0f, 0.0f);
}

TEST_CASE("Vec3CrossProduct.AntiCommutative", "[Vec3CrossProduct]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    Vec3 ab = Vec3::crossProduct(a, b);
    Vec3 ba = Vec3::crossProduct(b, a);
    expectNear(ab, -ba.x(), -ba.y(), -ba.z());
}

TEST_CASE("Vec3CrossProduct.ParallelVectorsGiveZero", "[Vec3CrossProduct]")
{
    Vec3 a{2.0f, 4.0f, 6.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    Vec3 c = Vec3::crossProduct(a, b);
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3CrossProduct.SelfCrossIsZero", "[Vec3CrossProduct]")
{
    Vec3 a{3.0f, 7.0f, 11.0f};
    Vec3 c = Vec3::crossProduct(a, a);
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3CrossProduct.ZeroVectorCross", "[Vec3CrossProduct]")
{
    Vec3 zero{};
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 c = Vec3::crossProduct(zero, a);
    expectNear(c, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3CrossProduct.InstanceMethodMatchesStatic", "[Vec3CrossProduct]")
{
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    Vec3 stat = Vec3::crossProduct(a, b);
    Vec3 inst = a.crossProduct(b);
    CHECK(stat == inst);
}

TEST_CASE("Vec3CrossProduct.ResultIsPerpendicularToInputs", "[Vec3CrossProduct]")
{
    Vec3 a{1.0f, 0.0f, 0.0f};
    Vec3 b{1.0f, 1.0f, 0.0f};
    Vec3 c = Vec3::crossProduct(a, b);
    // dot(a,c) should be 0
    float dotAC = a.x() * c.x() + a.y() * c.y() + a.z() * c.z();
    float dotBC = b.x() * c.x() + b.y() * c.y() + b.z() * c.z();
    CHECK(dotAC == Catch::Approx(0.0f).margin(kEps));
    CHECK(dotBC == Catch::Approx(0.0f).margin(kEps));
}

// ==========================================================================
// Normalise
// ==========================================================================

TEST_CASE("Vec3Normalise.UnitXUnchanged", "[Vec3Normalise]")
{
    Vec3 v{1.0f, 0.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 1.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3Normalise.UnitYUnchanged", "[Vec3Normalise]")
{
    Vec3 v{0.0f, 1.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.0f, 1.0f, 0.0f);
}

TEST_CASE("Vec3Normalise.ArbitraryVector", "[Vec3Normalise]")
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.6f, 0.8f, 0.0f);
}

TEST_CASE("Vec3Normalise.ResultHasUnitLength", "[Vec3Normalise]")
{
    Vec3 v{1.0f, 2.0f, 3.0f};
    Vec3 n = Vec3::normalise(v);
    float len = std::sqrt(n.x() * n.x() + n.y() * n.y() + n.z() * n.z());
    CHECK(len == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Vec3Normalise.NegativeVector", "[Vec3Normalise]")
{
    Vec3 v{-3.0f, -4.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, -0.6f, -0.8f, 0.0f);
}

TEST_CASE("Vec3Normalise.ZeroVectorReturnsZero", "[Vec3Normalise]")
{
    Vec3 v{};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3Normalise.NearZeroVectorReturnsZero", "[Vec3Normalise]")
{
    Vec3 v{1e-9f, 1e-9f, 1e-9f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3Normalise.VeryLargeVector", "[Vec3Normalise]")
{
    Vec3 v{1e18f, 0.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 1.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec3Normalise.StaticDoesNotMutateInput", "[Vec3Normalise]")
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    [[maybe_unused]] Vec3 n = Vec3::normalise(v);
    expectNear(v, 3.0f, 4.0f, 0.0f);
}

TEST_CASE("Vec3Normalise.InstanceMethodMutatesSelf", "[Vec3Normalise]")
{
    Vec3 v{3.0f, 4.0f, 0.0f};
    v.normalise();
    expectNear(v, 0.6f, 0.8f, 0.0f);
}

TEST_CASE("Vec3Normalise.InstanceMethodReturnsNormalisedValue", "[Vec3Normalise]")
{
    Vec3 v{0.0f, 0.0f, 5.0f};
    Vec3 result = v.normalise();
    expectNear(result, 0.0f, 0.0f, 1.0f);
}

TEST_CASE("Vec3Normalise.NormaliseAlreadyUnitVector", "[Vec3Normalise]")
{
    Vec3 v{0.0f, 0.0f, 1.0f};
    Vec3 n = Vec3::normalise(v);
    expectNear(n, 0.0f, 0.0f, 1.0f);
}

TEST_CASE("Vec3Normalise.Diagonal", "[Vec3Normalise]")
{
    Vec3 v{1.0f, 1.0f, 1.0f};
    Vec3 n = Vec3::normalise(v);
    float expected = 1.0f / std::sqrt(3.0f);
    expectNear(n, expected, expected, expected);
}

// ==========================================================================
// Noexcept guarantees
// ==========================================================================

TEST_CASE("Vec3Noexcept.AllOperationsAreNoexcept", "[Vec3Noexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<Vec3>);
    static_assert(test_traits::nothrow_constructible_from_v<Vec3, float, float, float>);
    static_assert(test_traits::has_nothrow_vec3_accessors<Vec3>);
    static_assert(test_traits::has_nothrow_vec_arithmetic<Vec3>);
    static_assert(test_traits::has_nothrow_vec_common_math<Vec3>);
    static_assert(test_traits::has_nothrow_vec3_cross_product<Vec3>);
    static_assert(test_traits::has_nothrow_equality<Vec3>);
}

// ==========================================================================
// Edge cases with special float values
// ==========================================================================

TEST_CASE("Vec3EdgeCases.InfinityComponents", "[Vec3EdgeCases]")
{
    float inf = std::numeric_limits<float>::infinity();
    Vec3 v{inf, -inf, 0.0f};
    CHECK(v.x() == inf);
    CHECK(v.y() == -inf);
    CHECK(v.z() == 0.0f);
}

TEST_CASE("Vec3EdgeCases.NaNEquality", "[Vec3EdgeCases]")
{
    float nan = std::numeric_limits<float>::quiet_NaN();
    Vec3 a{nan, 0.0f, 0.0f};
    Vec3 b{nan, 0.0f, 0.0f};
    // NaN != NaN per IEEE 754
    CHECK_FALSE(a == b);
}

TEST_CASE("Vec3EdgeCases.NormaliseInfinityVector", "[Vec3EdgeCases]")
{
    float inf = std::numeric_limits<float>::infinity();
    Vec3 v{inf, 0.0f, 0.0f};
    Vec3 n = Vec3::normalise(v);
    // Result is implementation-defined but should not crash
    (void)n;
}

TEST_CASE("Vec3EdgeCases.CrossProductLargeValues", "[Vec3EdgeCases]")
{
    Vec3 a{1e10f, 0.0f, 0.0f};
    Vec3 b{0.0f, 1e10f, 0.0f};
    Vec3 c = Vec3::crossProduct(a, b);
    // Should point in z direction
    CHECK(c.z() > 0.0f);
    CHECK(c.x() == Catch::Approx(0.0f).margin(kEps));
    CHECK(c.y() == Catch::Approx(0.0f).margin(kEps));
}

// ==========================================================================
// Compound expression correctness
// ==========================================================================

TEST_CASE("Vec3Compound.ChainedAddition", "[Vec3Compound]")
{
    Vec3 a{1.0f, 0.0f, 0.0f};
    Vec3 b{0.0f, 1.0f, 0.0f};
    Vec3 c{0.0f, 0.0f, 1.0f};
    Vec3 result = a + b + c;
    expectNear(result, 1.0f, 1.0f, 1.0f);
}

TEST_CASE("Vec3Compound.SubtractThenAdd", "[Vec3Compound]")
{
    Vec3 a{5.0f, 5.0f, 5.0f};
    Vec3 b{3.0f, 2.0f, 1.0f};
    Vec3 c{1.0f, 1.0f, 1.0f};
    Vec3 result = a - b + c;
    expectNear(result, 3.0f, 4.0f, 5.0f);
}

TEST_CASE("Vec3Compound.PlusEqualsChained", "[Vec3Compound]")
{
    Vec3 a{1.0f, 1.0f, 1.0f};
    (a += {1.0f, 0.0f, 0.0f}) += {0.0f, 1.0f, 0.0f};
    expectNear(a, 2.0f, 2.0f, 1.0f);
}
