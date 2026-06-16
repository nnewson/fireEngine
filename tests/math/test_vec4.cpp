#include <fire_engine/math/vec4.hpp>

#include <cmath>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Vec4;

// ---- Helpers ----

static constexpr float kEps = 1e-6f;

static void expectNear(const Vec4& v, float ex, float ey, float ez, float ew, float eps = kEps)
{
    CHECK(v.x() == Catch::Approx(ex).margin(eps));
    CHECK(v.y() == Catch::Approx(ey).margin(eps));
    CHECK(v.z() == Catch::Approx(ez).margin(eps));
    CHECK(v.w() == Catch::Approx(ew).margin(eps));
}

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("Vec4Construction.DefaultIsZero", "[Vec4Construction]")
{
    Vec4 v{};
    expectNear(v, 0.0f, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec4Construction.ExplicitValues", "[Vec4Construction]")
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(v, 1.0f, 2.0f, 3.0f, 4.0f);
}

TEST_CASE("Vec4Construction.PartialArgs", "[Vec4Construction]")
{
    Vec4 vx{5.0f};
    expectNear(vx, 5.0f, 0.0f, 0.0f, 0.0f);

    Vec4 vxy{5.0f, 6.0f};
    expectNear(vxy, 5.0f, 6.0f, 0.0f, 0.0f);

    Vec4 vxyz{5.0f, 6.0f, 7.0f};
    expectNear(vxyz, 5.0f, 6.0f, 7.0f, 0.0f);
}

TEST_CASE("Vec4Construction.NegativeValues", "[Vec4Construction]")
{
    Vec4 v{-1.0f, -2.0f, -3.0f, -4.0f};
    expectNear(v, -1.0f, -2.0f, -3.0f, -4.0f);
}

TEST_CASE("Vec4Construction.LargeValues", "[Vec4Construction]")
{
    Vec4 v{1e10f, -1e10f, 1e10f, -1e10f};
    expectNear(v, 1e10f, -1e10f, 1e10f, -1e10f);
}

TEST_CASE("Vec4Construction.Constexpr", "[Vec4Construction]")
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

TEST_CASE("Vec4Accessors.GettersReturnComponents", "[Vec4Accessors]")
{
    Vec4 v{7.0f, 8.0f, 9.0f, 10.0f};
    CHECK(v.x() == Catch::Approx(7.0f).margin(1e-5f));
    CHECK(v.y() == Catch::Approx(8.0f).margin(1e-5f));
    CHECK(v.z() == Catch::Approx(9.0f).margin(1e-5f));
    CHECK(v.w() == Catch::Approx(10.0f).margin(1e-5f));
}

TEST_CASE("Vec4Accessors.SettersModifyComponents", "[Vec4Accessors]")
{
    Vec4 v{};
    v.x(10.0f);
    v.y(20.0f);
    v.z(30.0f);
    v.w(40.0f);
    expectNear(v, 10.0f, 20.0f, 30.0f, 40.0f);
}

TEST_CASE("Vec4Accessors.SetterGetterRoundTrip", "[Vec4Accessors]")
{
    Vec4 v{};
    v.x(42.0f);
    CHECK(v.x() == Catch::Approx(42.0f).margin(1e-5f));
    v.w(-1.0f);
    CHECK(v.w() == Catch::Approx(-1.0f).margin(1e-5f));
}

// ==========================================================================
// Equality
// ==========================================================================

TEST_CASE("Vec4Equality.EqualVectors", "[Vec4Equality]")
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(a == b);
}

TEST_CASE("Vec4Equality.DifferentVectors", "[Vec4Equality]")
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{1.0f, 2.0f, 3.0f, 5.0f};
    Vec4 c{0.0f, 2.0f, 3.0f, 4.0f};
    CHECK(a != b);
    CHECK(a != c);
}

TEST_CASE("Vec4Equality.DefaultEqualDefault", "[Vec4Equality]")
{
    CHECK(Vec4{} == Vec4{});
}

TEST_CASE("Vec4Equality.Constexpr", "[Vec4Equality]")
{
    constexpr Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr Vec4 b{1.0f, 2.0f, 3.0f, 4.0f};
    static_assert(a == b);
}

// ==========================================================================
// Arithmetic — Addition
// ==========================================================================

TEST_CASE("Vec4Addition.BasicAdd", "[Vec4Addition]")
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{5.0f, 6.0f, 7.0f, 8.0f};
    expectNear(a + b, 6.0f, 8.0f, 10.0f, 12.0f);
}

TEST_CASE("Vec4Addition.CompoundAdd", "[Vec4Addition]")
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    a += Vec4{10.0f, 20.0f, 30.0f, 40.0f};
    expectNear(a, 11.0f, 22.0f, 33.0f, 44.0f);
}

TEST_CASE("Vec4Addition.AddZero", "[Vec4Addition]")
{
    Vec4 a{5.0f, 6.0f, 7.0f, 8.0f};
    expectNear(a + Vec4{}, 5.0f, 6.0f, 7.0f, 8.0f);
}

TEST_CASE("Vec4Addition.Constexpr", "[Vec4Addition]")
{
    constexpr Vec4 r = Vec4{1.0f, 2.0f, 3.0f, 4.0f} + Vec4{5.0f, 6.0f, 7.0f, 8.0f};
    static_assert(r.x() == 6.0f);
    static_assert(r.w() == 12.0f);
}

// ==========================================================================
// Arithmetic — Subtraction
// ==========================================================================

TEST_CASE("Vec4Subtraction.BasicSub", "[Vec4Subtraction]")
{
    Vec4 a{5.0f, 6.0f, 7.0f, 8.0f};
    Vec4 b{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(a - b, 4.0f, 4.0f, 4.0f, 4.0f);
}

TEST_CASE("Vec4Subtraction.CompoundSub", "[Vec4Subtraction]")
{
    Vec4 v{10.0f, 20.0f, 30.0f, 40.0f};
    v -= Vec4{3.0f, 4.0f, 5.0f, 6.0f};
    expectNear(v, 7.0f, 16.0f, 25.0f, 34.0f);
}

TEST_CASE("Vec4Subtraction.SubSelf", "[Vec4Subtraction]")
{
    Vec4 v{7.0f, 8.0f, 9.0f, 10.0f};
    expectNear(v - v, 0.0f, 0.0f, 0.0f, 0.0f);
}

// ==========================================================================
// Arithmetic — Scalar Multiply
// ==========================================================================

TEST_CASE("Vec4ScalarMultiply.BasicMul", "[Vec4ScalarMultiply]")
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(v * 3.0f, 3.0f, 6.0f, 9.0f, 12.0f);
}

TEST_CASE("Vec4ScalarMultiply.CompoundMul", "[Vec4ScalarMultiply]")
{
    Vec4 v{2.0f, 3.0f, 4.0f, 5.0f};
    v *= 4.0f;
    expectNear(v, 8.0f, 12.0f, 16.0f, 20.0f);
}

TEST_CASE("Vec4ScalarMultiply.MulZero", "[Vec4ScalarMultiply]")
{
    Vec4 v{5.0f, 6.0f, 7.0f, 8.0f};
    expectNear(v * 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec4ScalarMultiply.MulNegative", "[Vec4ScalarMultiply]")
{
    Vec4 v{1.0f, -2.0f, 3.0f, -4.0f};
    expectNear(v * -1.0f, -1.0f, 2.0f, -3.0f, 4.0f);
}

// ==========================================================================
// Arithmetic — Scalar Divide
// ==========================================================================

TEST_CASE("Vec4ScalarDivide.BasicDiv", "[Vec4ScalarDivide]")
{
    Vec4 v{6.0f, 8.0f, 10.0f, 12.0f};
    expectNear(v / 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
}

TEST_CASE("Vec4ScalarDivide.CompoundDiv", "[Vec4ScalarDivide]")
{
    Vec4 v{10.0f, 20.0f, 30.0f, 40.0f};
    v /= 5.0f;
    expectNear(v, 2.0f, 4.0f, 6.0f, 8.0f);
}

TEST_CASE("Vec4ScalarDivide.DivOne", "[Vec4ScalarDivide]")
{
    Vec4 v{3.0f, 4.0f, 5.0f, 6.0f};
    expectNear(v / 1.0f, 3.0f, 4.0f, 5.0f, 6.0f);
}

// ==========================================================================
// Dot Product
// ==========================================================================

TEST_CASE("Vec4DotProduct.StaticMethod", "[Vec4DotProduct]")
{
    Vec4 a{1.0f, 0.0f, 0.0f, 0.0f};
    Vec4 b{0.0f, 1.0f, 0.0f, 0.0f};
    CHECK(Vec4::dotProduct(a, b) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Vec4DotProduct.MemberMethod", "[Vec4DotProduct]")
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(a.dotProduct(b) == Catch::Approx(30.0f).margin(1e-5f));
}

TEST_CASE("Vec4DotProduct.Perpendicular", "[Vec4DotProduct]")
{
    Vec4 a{1.0f, 0.0f, 0.0f, 0.0f};
    Vec4 b{0.0f, 0.0f, 1.0f, 0.0f};
    CHECK(Vec4::dotProduct(a, b) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Vec4DotProduct.Constexpr", "[Vec4DotProduct]")
{
    constexpr Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr Vec4 b{5.0f, 6.0f, 7.0f, 8.0f};
    constexpr float d = Vec4::dotProduct(a, b);
    static_assert(d == 70.0f);
}

// ==========================================================================
// Magnitude
// ==========================================================================

TEST_CASE("Vec4Magnitude.UnitX", "[Vec4Magnitude]")
{
    Vec4 v{1.0f, 0.0f, 0.0f, 0.0f};
    CHECK(v.magnitude() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Vec4Magnitude.UnitW", "[Vec4Magnitude]")
{
    Vec4 v{0.0f, 0.0f, 0.0f, 1.0f};
    CHECK(v.magnitude() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Vec4Magnitude.KnownValue", "[Vec4Magnitude]")
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(v.magnitude() == Catch::Approx(std::sqrt(30.0f)).margin(kEps));
}

TEST_CASE("Vec4Magnitude.Zero", "[Vec4Magnitude]")
{
    Vec4 v{};
    CHECK(v.magnitude() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Vec4Magnitude.Squared", "[Vec4Magnitude]")
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(v.magnitudeSquared() == Catch::Approx(30.0f).margin(1e-5f));
}

// ==========================================================================
// Normalise
// ==========================================================================

TEST_CASE("Vec4Normalise.StaticMethod", "[Vec4Normalise]")
{
    Vec4 v{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 n = Vec4::normalise(v);
    CHECK(n.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Vec4Normalise.MemberMethod", "[Vec4Normalise]")
{
    Vec4 v{0.0f, 0.0f, 0.0f, 5.0f};
    v.normalise();
    expectNear(v, 0.0f, 0.0f, 0.0f, 1.0f);
}

TEST_CASE("Vec4Normalise.ZeroVectorStaysZero", "[Vec4Normalise]")
{
    Vec4 n = Vec4::normalise(Vec4{});
    expectNear(n, 0.0f, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec4Normalise.AlreadyUnit", "[Vec4Normalise]")
{
    Vec4 v{1.0f, 0.0f, 0.0f, 0.0f};
    Vec4 n = Vec4::normalise(v);
    expectNear(n, 1.0f, 0.0f, 0.0f, 0.0f);
}

TEST_CASE("Vec4Normalise.NegativeComponents", "[Vec4Normalise]")
{
    Vec4 v{-1.0f, -2.0f, -3.0f, -4.0f};
    Vec4 n = Vec4::normalise(v);
    CHECK(n.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

// ==========================================================================
// Copy and Move
// ==========================================================================

TEST_CASE("Vec4Copy.CopyConstruct", "[Vec4Copy]")
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{a};
    expectNear(b, 1.0f, 2.0f, 3.0f, 4.0f);
}

TEST_CASE("Vec4Copy.CopyAssign", "[Vec4Copy]")
{
    Vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    Vec4 b{};
    b = a;
    expectNear(b, 1.0f, 2.0f, 3.0f, 4.0f);
}

TEST_CASE("Vec4Move.MoveConstruct", "[Vec4Move]")
{
    Vec4 a{3.0f, 4.0f, 5.0f, 6.0f};
    Vec4 b{std::move(a)};
    expectNear(b, 3.0f, 4.0f, 5.0f, 6.0f);
}

TEST_CASE("Vec4Move.MoveAssign", "[Vec4Move]")
{
    Vec4 a{5.0f, 6.0f, 7.0f, 8.0f};
    Vec4 b{};
    b = std::move(a);
    expectNear(b, 5.0f, 6.0f, 7.0f, 8.0f);
}

// ==========================================================================
// Noexcept
// ==========================================================================

TEST_CASE("Vec4Noexcept.Construction", "[Vec4Noexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<Vec4>);
    static_assert(test_traits::nothrow_constructible_from_v<Vec4, float, float, float, float>);
}

TEST_CASE("Vec4Noexcept.Accessors", "[Vec4Noexcept]")
{
    static_assert(test_traits::has_nothrow_vec4_accessors<Vec4>);
}

TEST_CASE("Vec4Noexcept.Arithmetic", "[Vec4Noexcept]")
{
    static_assert(test_traits::has_nothrow_vec_arithmetic<Vec4>);
}

TEST_CASE("Vec4Noexcept.DotMagnitudeNormalise", "[Vec4Noexcept]")
{
    static_assert(test_traits::has_nothrow_vec_common_math<Vec4>);
}

TEST_CASE("Vec4Noexcept.Equality", "[Vec4Noexcept]")
{
    static_assert(test_traits::has_nothrow_equality<Vec4>);
}

// ==========================================================================
// Edge Cases
// ==========================================================================

TEST_CASE("Vec4EdgeCases.VerySmallValues", "[Vec4EdgeCases]")
{
    Vec4 v{1e-20f, 1e-20f, 1e-20f, 1e-20f};
    CHECK(v.magnitude() == Catch::Approx(2e-20f).margin(1e-25f));
}

TEST_CASE("Vec4EdgeCases.MixedSignDot", "[Vec4EdgeCases]")
{
    Vec4 a{1.0f, -1.0f, 1.0f, -1.0f};
    Vec4 b{-1.0f, 1.0f, -1.0f, 1.0f};
    CHECK(Vec4::dotProduct(a, b) == Catch::Approx(-4.0f).margin(1e-5f));
}
