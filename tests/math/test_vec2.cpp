#include <fire_engine/math/vec2.hpp>

#include <cmath>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Vec2;

// ---- Helpers ----

static constexpr float kEps = 1e-6f;

static void expectNear(const Vec2& v, float ex, float ey, float eps = kEps)
{
    CHECK(v.s() == Catch::Approx(ex).margin(eps));
    CHECK(v.t() == Catch::Approx(ey).margin(eps));
}

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("Vec2Construction.DefaultIsZero", "[Vec2Construction]")
{
    Vec2 v{};
    expectNear(v, 0.0f, 0.0f);
}

TEST_CASE("Vec2Construction.ExplicitValues", "[Vec2Construction]")
{
    Vec2 v{1.0f, 2.0f};
    expectNear(v, 1.0f, 2.0f);
}

TEST_CASE("Vec2Construction.PartialArgs", "[Vec2Construction]")
{
    Vec2 vx{5.0f};
    expectNear(vx, 5.0f, 0.0f);
}

TEST_CASE("Vec2Construction.NegativeValues", "[Vec2Construction]")
{
    Vec2 v{-1.0f, -2.0f};
    expectNear(v, -1.0f, -2.0f);
}

TEST_CASE("Vec2Construction.LargeValues", "[Vec2Construction]")
{
    Vec2 v{1e10f, -1e10f};
    expectNear(v, 1e10f, -1e10f);
}

TEST_CASE("Vec2Construction.Constexpr", "[Vec2Construction]")
{
    constexpr Vec2 v{3.0f, 4.0f};
    static_assert(v.s() == 3.0f);
    static_assert(v.t() == 4.0f);
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("Vec2Accessors.GettersReturnComponents", "[Vec2Accessors]")
{
    Vec2 v{7.0f, 8.0f};
    CHECK(v.s() == Catch::Approx(7.0f).margin(1e-5f));
    CHECK(v.t() == Catch::Approx(8.0f).margin(1e-5f));
}

TEST_CASE("Vec2Accessors.SettersModifyComponents", "[Vec2Accessors]")
{
    Vec2 v{};
    v.s(10.0f);
    v.t(20.0f);
    expectNear(v, 10.0f, 20.0f);
}

TEST_CASE("Vec2Accessors.SetterGetterRoundTrip", "[Vec2Accessors]")
{
    Vec2 v{};
    v.s(42.0f);
    CHECK(v.s() == Catch::Approx(42.0f).margin(1e-5f));
    v.t(-1.0f);
    CHECK(v.t() == Catch::Approx(-1.0f).margin(1e-5f));
}

// ==========================================================================
// Equality
// ==========================================================================

TEST_CASE("Vec2Equality.EqualVectors", "[Vec2Equality]")
{
    CHECK((Vec2{1.0f, 2.0f}) == (Vec2{1.0f, 2.0f}));
}

TEST_CASE("Vec2Equality.DifferentVectors", "[Vec2Equality]")
{
    CHECK((Vec2{1.0f, 2.0f}) != (Vec2{1.0f, 3.0f}));
    CHECK((Vec2{1.0f, 2.0f}) != (Vec2{0.0f, 2.0f}));
}

TEST_CASE("Vec2Equality.DefaultEqualDefault", "[Vec2Equality]")
{
    CHECK(Vec2{} == Vec2{});
}

TEST_CASE("Vec2Equality.Constexpr", "[Vec2Equality]")
{
    constexpr Vec2 a{1.0f, 2.0f};
    constexpr Vec2 b{1.0f, 2.0f};
    static_assert(a == b);
}

// ==========================================================================
// Arithmetic — Addition
// ==========================================================================

TEST_CASE("Vec2Addition.BasicAdd", "[Vec2Addition]")
{
    Vec2 a{1.0f, 2.0f};
    Vec2 b{3.0f, 4.0f};
    expectNear(a + b, 4.0f, 6.0f);
}

TEST_CASE("Vec2Addition.CompoundAdd", "[Vec2Addition]")
{
    Vec2 a{1.0f, 2.0f};
    a += Vec2{10.0f, 20.0f};
    expectNear(a, 11.0f, 22.0f);
}

TEST_CASE("Vec2Addition.AddZero", "[Vec2Addition]")
{
    Vec2 a{5.0f, 6.0f};
    expectNear(a + Vec2{}, 5.0f, 6.0f);
}

TEST_CASE("Vec2Addition.Constexpr", "[Vec2Addition]")
{
    constexpr Vec2 r = Vec2{1.0f, 2.0f} + Vec2{3.0f, 4.0f};
    static_assert(r.s() == 4.0f);
    static_assert(r.t() == 6.0f);
}

// ==========================================================================
// Arithmetic — Subtraction
// ==========================================================================

TEST_CASE("Vec2Subtraction.BasicSub", "[Vec2Subtraction]")
{
    Vec2 a{5.0f, 6.0f};
    Vec2 b{1.0f, 2.0f};
    expectNear(a - b, 4.0f, 4.0f);
}

TEST_CASE("Vec2Subtraction.CompoundSub", "[Vec2Subtraction]")
{
    Vec2 v{10.0f, 20.0f};
    v -= Vec2{3.0f, 4.0f};
    expectNear(v, 7.0f, 16.0f);
}

TEST_CASE("Vec2Subtraction.SubSelf", "[Vec2Subtraction]")
{
    Vec2 v{7.0f, 8.0f};
    expectNear(v - v, 0.0f, 0.0f);
}

// ==========================================================================
// Arithmetic — Scalar Multiply
// ==========================================================================

TEST_CASE("Vec2ScalarMultiply.BasicMul", "[Vec2ScalarMultiply]")
{
    Vec2 v{1.0f, 2.0f};
    expectNear(v * 3.0f, 3.0f, 6.0f);
}

TEST_CASE("Vec2ScalarMultiply.CompoundMul", "[Vec2ScalarMultiply]")
{
    Vec2 v{2.0f, 3.0f};
    v *= 4.0f;
    expectNear(v, 8.0f, 12.0f);
}

TEST_CASE("Vec2ScalarMultiply.MulZero", "[Vec2ScalarMultiply]")
{
    Vec2 v{5.0f, 6.0f};
    expectNear(v * 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec2ScalarMultiply.MulNegative", "[Vec2ScalarMultiply]")
{
    Vec2 v{1.0f, -2.0f};
    expectNear(v * -1.0f, -1.0f, 2.0f);
}

// ==========================================================================
// Arithmetic — Scalar Divide
// ==========================================================================

TEST_CASE("Vec2ScalarDivide.BasicDiv", "[Vec2ScalarDivide]")
{
    Vec2 v{6.0f, 8.0f};
    expectNear(v / 2.0f, 3.0f, 4.0f);
}

TEST_CASE("Vec2ScalarDivide.CompoundDiv", "[Vec2ScalarDivide]")
{
    Vec2 v{10.0f, 20.0f};
    v /= 5.0f;
    expectNear(v, 2.0f, 4.0f);
}

TEST_CASE("Vec2ScalarDivide.DivOne", "[Vec2ScalarDivide]")
{
    Vec2 v{3.0f, 4.0f};
    expectNear(v / 1.0f, 3.0f, 4.0f);
}

// ==========================================================================
// Dot Product
// ==========================================================================

TEST_CASE("Vec2DotProduct.StaticMethod", "[Vec2DotProduct]")
{
    Vec2 a{1.0f, 0.0f};
    Vec2 b{0.0f, 1.0f};
    CHECK(Vec2::dotProduct(a, b) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Vec2DotProduct.MemberMethod", "[Vec2DotProduct]")
{
    Vec2 a{3.0f, 4.0f};
    Vec2 b{3.0f, 4.0f};
    CHECK(a.dotProduct(b) == Catch::Approx(25.0f).margin(1e-5f));
}

TEST_CASE("Vec2DotProduct.Perpendicular", "[Vec2DotProduct]")
{
    Vec2 a{1.0f, 0.0f};
    Vec2 b{0.0f, 1.0f};
    CHECK(Vec2::dotProduct(a, b) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Vec2DotProduct.Parallel", "[Vec2DotProduct]")
{
    Vec2 a{2.0f, 0.0f};
    Vec2 b{3.0f, 0.0f};
    CHECK(Vec2::dotProduct(a, b) == Catch::Approx(6.0f).margin(1e-5f));
}

TEST_CASE("Vec2DotProduct.Constexpr", "[Vec2DotProduct]")
{
    constexpr Vec2 a{1.0f, 2.0f};
    constexpr Vec2 b{3.0f, 4.0f};
    constexpr float d = Vec2::dotProduct(a, b);
    static_assert(d == 11.0f);
}

// ==========================================================================
// Magnitude
// ==========================================================================

TEST_CASE("Vec2Magnitude.UnitX", "[Vec2Magnitude]")
{
    Vec2 v{1.0f, 0.0f};
    CHECK(v.magnitude() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Vec2Magnitude.UnitY", "[Vec2Magnitude]")
{
    Vec2 v{0.0f, 1.0f};
    CHECK(v.magnitude() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Vec2Magnitude.ThreeFour", "[Vec2Magnitude]")
{
    Vec2 v{3.0f, 4.0f};
    CHECK(v.magnitude() == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("Vec2Magnitude.Zero", "[Vec2Magnitude]")
{
    Vec2 v{};
    CHECK(v.magnitude() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Vec2Magnitude.Squared", "[Vec2Magnitude]")
{
    Vec2 v{3.0f, 4.0f};
    CHECK(v.magnitudeSquared() == Catch::Approx(25.0f).margin(1e-5f));
}

// ==========================================================================
// Normalise
// ==========================================================================

TEST_CASE("Vec2Normalise.StaticMethod", "[Vec2Normalise]")
{
    Vec2 v{3.0f, 4.0f};
    Vec2 n = Vec2::normalise(v);
    CHECK(n.magnitude() == Catch::Approx(1.0f).margin(kEps));
    expectNear(n, 3.0f / 5.0f, 4.0f / 5.0f);
}

TEST_CASE("Vec2Normalise.MemberMethod", "[Vec2Normalise]")
{
    Vec2 v{0.0f, 5.0f};
    v.normalise();
    expectNear(v, 0.0f, 1.0f);
}

TEST_CASE("Vec2Normalise.ZeroVectorStaysZero", "[Vec2Normalise]")
{
    Vec2 n = Vec2::normalise(Vec2{});
    expectNear(n, 0.0f, 0.0f);
}

TEST_CASE("Vec2Normalise.AlreadyUnit", "[Vec2Normalise]")
{
    Vec2 v{1.0f, 0.0f};
    Vec2 n = Vec2::normalise(v);
    expectNear(n, 1.0f, 0.0f);
}

TEST_CASE("Vec2Normalise.NegativeComponents", "[Vec2Normalise]")
{
    Vec2 v{-3.0f, -4.0f};
    Vec2 n = Vec2::normalise(v);
    CHECK(n.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

// ==========================================================================
// Copy and Move
// ==========================================================================

TEST_CASE("Vec2Copy.CopyConstruct", "[Vec2Copy]")
{
    Vec2 a{1.0f, 2.0f};
    Vec2 b{a};
    expectNear(b, 1.0f, 2.0f);
}

TEST_CASE("Vec2Copy.CopyAssign", "[Vec2Copy]")
{
    Vec2 a{1.0f, 2.0f};
    Vec2 b{};
    b = a;
    expectNear(b, 1.0f, 2.0f);
}

TEST_CASE("Vec2Move.MoveConstruct", "[Vec2Move]")
{
    Vec2 a{3.0f, 4.0f};
    Vec2 b{std::move(a)};
    expectNear(b, 3.0f, 4.0f);
}

TEST_CASE("Vec2Move.MoveAssign", "[Vec2Move]")
{
    Vec2 a{5.0f, 6.0f};
    Vec2 b{};
    b = std::move(a);
    expectNear(b, 5.0f, 6.0f);
}

// ==========================================================================
// Noexcept
// ==========================================================================

TEST_CASE("Vec2Noexcept.Construction", "[Vec2Noexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<Vec2>);
    static_assert(test_traits::nothrow_constructible_from_v<Vec2, float, float>);
}

TEST_CASE("Vec2Noexcept.Accessors", "[Vec2Noexcept]")
{
    static_assert(test_traits::has_nothrow_vec2_accessors<Vec2>);
}

TEST_CASE("Vec2Noexcept.Arithmetic", "[Vec2Noexcept]")
{
    static_assert(test_traits::has_nothrow_vec_arithmetic<Vec2>);
}

TEST_CASE("Vec2Noexcept.DotMagnitudeNormalise", "[Vec2Noexcept]")
{
    static_assert(test_traits::has_nothrow_vec_common_math<Vec2>);
}

TEST_CASE("Vec2Noexcept.Equality", "[Vec2Noexcept]")
{
    static_assert(test_traits::has_nothrow_equality<Vec2>);
}

// ==========================================================================
// Edge Cases
// ==========================================================================

TEST_CASE("Vec2EdgeCases.VerySmallValues", "[Vec2EdgeCases]")
{
    Vec2 v{1e-20f, 1e-20f};
    CHECK(v.magnitude() == Catch::Approx(std::sqrt(2e-40f)).margin(1e-25f));
}

TEST_CASE("Vec2EdgeCases.MixedSignDot", "[Vec2EdgeCases]")
{
    Vec2 a{1.0f, -1.0f};
    Vec2 b{-1.0f, 1.0f};
    CHECK(Vec2::dotProduct(a, b) == Catch::Approx(-2.0f).margin(1e-5f));
}
