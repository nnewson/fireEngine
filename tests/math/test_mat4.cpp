#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/view_basis.hpp>

#include <cmath>
#include <numbers>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Mat4;
using fire_engine::Vec3;
using fire_engine::Vec4;
using fire_engine::ViewBasis;

// ---- Helpers ----

static constexpr float kEps = 1e-5f;

static void expectIdentity(const Mat4& m)
{
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            if (row == col)
            {
                INFO("row=" << row << " col=" << col);
                CHECK((m[row, col]) == Catch::Approx(1.0f).margin(1e-5f));
            }
            else
            {
                INFO("row=" << row << " col=" << col);
                CHECK((m[row, col]) == Catch::Approx(0.0f).margin(1e-5f));
            }
        }
    }
}

static void expectZero(const Mat4& m)
{
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            INFO("row=" << row << " col=" << col);
            CHECK((m[row, col]) == Catch::Approx(0.0f).margin(1e-5f));
        }
    }
}

static void expectNear(const Mat4& a, const Mat4& b, float eps = kEps)
{
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            INFO("row=" << row << " col=" << col);
            CHECK((a[row, col]) == Catch::Approx((b[row, col])).margin(eps));
        }
    }
}

static void expectNear(const Vec4& v, float x, float y, float z, float w, float eps = kEps)
{
    CHECK(v.x() == Catch::Approx(x).margin(eps));
    CHECK(v.y() == Catch::Approx(y).margin(eps));
    CHECK(v.z() == Catch::Approx(z).margin(eps));
    CHECK(v.w() == Catch::Approx(w).margin(eps));
}

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("Mat4Construction.DefaultIsZero", "[Mat4Construction]")
{
    Mat4 m;
    expectZero(m);
}

TEST_CASE("Mat4Construction.CopyConstruct", "[Mat4Construction]")
{
    Mat4 a = Mat4::identity();
    Mat4 b{a};
    expectIdentity(b);
}

TEST_CASE("Mat4Construction.CopyAssign", "[Mat4Construction]")
{
    Mat4 a = Mat4::identity();
    Mat4 b;
    b = a;
    expectIdentity(b);
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("Mat4Accessors.SetAndGet", "[Mat4Accessors]")
{
    Mat4 m;
    m[2, 3] = 42.0f;
    CHECK((m[2, 3]) == Catch::Approx(42.0f).margin(1e-5f));
}

TEST_CASE("Mat4Accessors.SetDoesNotAffectOtherElements", "[Mat4Accessors]")
{
    Mat4 m;
    m[1, 2] = 7.0f;
    CHECK((m[0, 0]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((m[1, 2]) == Catch::Approx(7.0f).margin(1e-5f));
    CHECK((m[2, 1]) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Mat4Accessors.DataReturnsColumnMajorPointer", "[Mat4Accessors]")
{
    Mat4 m = Mat4::identity();
    const float* d = m.data();
    // Column-major: m[0]=m(0,0), m[1]=m(1,0), ..., m[4]=m(0,1)
    CHECK(d[0] == Catch::Approx(1.0f).margin(1e-5f));  // (0,0)
    CHECK(d[1] == Catch::Approx(0.0f).margin(1e-5f));  // (1,0)
    CHECK(d[4] == Catch::Approx(0.0f).margin(1e-5f));  // (0,1)
    CHECK(d[5] == Catch::Approx(1.0f).margin(1e-5f));  // (1,1)
    CHECK(d[15] == Catch::Approx(1.0f).margin(1e-5f)); // (3,3)
}

// ==========================================================================
// Identity
// ==========================================================================

TEST_CASE("Mat4Identity.DiagonalOnes", "[Mat4Identity]")
{
    Mat4 m = Mat4::identity();
    CHECK((m[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[1, 1]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[2, 2]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[3, 3]) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Mat4Identity.OffDiagonalZeros", "[Mat4Identity]")
{
    Mat4 m = Mat4::identity();
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            if (row != col)
            {
                INFO("row=" << row << " col=" << col);
                CHECK((m[row, col]) == Catch::Approx(0.0f).margin(1e-5f));
            }
        }
    }
}

// ==========================================================================
// Equality
// ==========================================================================

TEST_CASE("Mat4Equality.IdenticalMatrices", "[Mat4Equality]")
{
    Mat4 a = Mat4::identity();
    Mat4 b = Mat4::identity();
    CHECK(a == b);
}

TEST_CASE("Mat4Equality.DifferentMatrices", "[Mat4Equality]")
{
    Mat4 a = Mat4::identity();
    Mat4 b;
    CHECK_FALSE(a == b);
}

TEST_CASE("Mat4Equality.SingleElementDifference", "[Mat4Equality]")
{
    Mat4 a = Mat4::identity();
    Mat4 b = Mat4::identity();
    b[2, 3] = 0.001f;
    CHECK_FALSE(a == b);
}

TEST_CASE("Mat4Equality.ZeroMatrices", "[Mat4Equality]")
{
    Mat4 a;
    Mat4 b;
    CHECK(a == b);
}

TEST_CASE("Mat4Equality.BitwiseEqualMatchesOperator", "[Mat4Equality]")
{
    Mat4 a = Mat4::identity();
    Mat4 b = Mat4::identity();
    Mat4 c = Mat4::scale(Vec3{2.0f, 1.0f, 1.0f});
    CHECK(a.bitwiseEqual(b));
    CHECK_FALSE(a.bitwiseEqual(c));
}

TEST_CASE("Mat4Equality.ApproxEqualWithinTolerance", "[Mat4Equality]")
{
    Mat4 a = Mat4::identity();
    Mat4 b = Mat4::identity();
    b[0, 0] = 1.0f + 1e-7f;
    CHECK_FALSE(a == b);
    CHECK(a.approxEqual(b, 1e-6f));
    CHECK_FALSE(a.approxEqual(b, 1e-9f));
}

// ==========================================================================
// Multiplication
// ==========================================================================

TEST_CASE("Mat4Multiply.IdentityTimesIdentity", "[Mat4Multiply]")
{
    Mat4 I = Mat4::identity();
    Mat4 r = I * I;
    expectIdentity(r);
}

TEST_CASE("Mat4Multiply.IdentityTimesMatrix", "[Mat4Multiply]")
{
    Mat4 I = Mat4::identity();
    Mat4 a = Mat4::identity();
    a[0, 3] = 5.0f;
    a[1, 3] = 10.0f;
    a[2, 3] = 15.0f;

    Mat4 r = I * a;
    CHECK((r[0, 3]) == Catch::Approx(5.0f).margin(1e-5f));
    CHECK((r[1, 3]) == Catch::Approx(10.0f).margin(1e-5f));
    CHECK((r[2, 3]) == Catch::Approx(15.0f).margin(1e-5f));
}

TEST_CASE("Mat4Multiply.MatrixTimesIdentity", "[Mat4Multiply]")
{
    Mat4 I = Mat4::identity();
    Mat4 a = Mat4::identity();
    a[0, 3] = 5.0f;

    Mat4 r = a * I;
    CHECK((r[0, 3]) == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("Mat4Multiply.ZeroMatrixTimesAnything", "[Mat4Multiply]")
{
    Mat4 zero;
    Mat4 a = Mat4::identity();
    Mat4 r = zero * a;
    expectZero(r);
}

TEST_CASE("Mat4Multiply.GeneralCase", "[Mat4Multiply]")
{
    // Multiply two known matrices and verify a few elements
    Mat4 a = Mat4::identity();
    a[0, 0] = 2.0f;
    a[1, 1] = 3.0f;
    // a is diag(2, 3, 1, 1)

    Mat4 b = Mat4::identity();
    b[0, 3] = 4.0f; // translation x=4
    b[1, 3] = 5.0f; // translation y=5

    // a * b: scales then translates
    // result should have (0,3) = 2*4 = 8, (1,3) = 3*5 = 15
    Mat4 r = a * b;
    CHECK((r[0, 0]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((r[1, 1]) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK((r[0, 3]) == Catch::Approx(8.0f).margin(1e-5f));
    CHECK((r[1, 3]) == Catch::Approx(15.0f).margin(1e-5f));
}

TEST_CASE("Mat4Multiply.NotCommutative", "[Mat4Multiply]")
{
    Mat4 a = Mat4::identity();
    a[0, 1] = 1.0f; // shear

    Mat4 b = Mat4::identity();
    b[1, 0] = 1.0f; // different shear

    Mat4 ab = a * b;
    Mat4 ba = b * a;
    CHECK_FALSE(ab == ba);
}

TEST_CASE("Mat4Multiply.MultiplyEqualsMatchesMultiply", "[Mat4Multiply]")
{
    Mat4 a = Mat4::identity();
    a[0, 0] = 2.0f;
    Mat4 b = Mat4::identity();
    b[1, 1] = 3.0f;

    Mat4 expected = a * b;
    a *= b;
    CHECK(a == expected);
}

TEST_CASE("Mat4Multiply.Associative", "[Mat4Multiply]")
{
    Mat4 a = Mat4::rotateX(0.5f);
    Mat4 b = Mat4::rotateY(0.7f);
    Mat4 c = Mat4::identity();
    c[0, 3] = 1.0f;
    c[1, 3] = 2.0f;
    c[2, 3] = 3.0f;

    Mat4 ab_c = (a * b) * c;
    Mat4 a_bc = a * (b * c);
    expectNear(ab_c, a_bc);
}

TEST_CASE("Mat4MultiplyVec4.IdentityPreservesVector", "[Mat4MultiplyVec4]")
{
    Vec4 r = Mat4::identity() * Vec4{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(r, 1.0f, 2.0f, 3.0f, 4.0f);
}

TEST_CASE("Mat4MultiplyVec4.TranslationTransformsPosition", "[Mat4MultiplyVec4]")
{
    Vec4 r = Mat4::translate({5.0f, 6.0f, 7.0f}) * Vec4{1.0f, 2.0f, 3.0f, 1.0f};
    expectNear(r, 6.0f, 8.0f, 10.0f, 1.0f);
}

TEST_CASE("Mat4MultiplyVec4.TranslationDoesNotTransformDirection", "[Mat4MultiplyVec4]")
{
    Vec4 r = Mat4::translate({5.0f, 6.0f, 7.0f}) * Vec4{1.0f, 2.0f, 3.0f, 0.0f};
    expectNear(r, 1.0f, 2.0f, 3.0f, 0.0f);
}

TEST_CASE("Mat4MultiplyVec4.ScaleTransformsVector", "[Mat4MultiplyVec4]")
{
    Vec4 r = Mat4::scale({2.0f, 3.0f, 4.0f}) * Vec4{1.0f, 2.0f, 3.0f, 5.0f};
    expectNear(r, 2.0f, 6.0f, 12.0f, 5.0f);
}

TEST_CASE("Mat4MultiplyVec4.GeneralCaseUsesRowByColumnIndexing", "[Mat4MultiplyVec4]")
{
    Mat4 m;
    m[0, 0] = 1.0f;
    m[0, 1] = 2.0f;
    m[0, 2] = 3.0f;
    m[0, 3] = 4.0f;
    m[1, 0] = 5.0f;
    m[1, 1] = 6.0f;
    m[1, 2] = 7.0f;
    m[1, 3] = 8.0f;
    m[2, 0] = 9.0f;
    m[2, 1] = 10.0f;
    m[2, 2] = 11.0f;
    m[2, 3] = 12.0f;
    m[3, 0] = 13.0f;
    m[3, 1] = 14.0f;
    m[3, 2] = 15.0f;
    m[3, 3] = 16.0f;

    Vec4 r = m * Vec4{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(r, 30.0f, 70.0f, 110.0f, 150.0f);
}

// ==========================================================================
// RotateX
// ==========================================================================

TEST_CASE("Mat4RotateX.ZeroAngleIsIdentity", "[Mat4RotateX]")
{
    Mat4 r = Mat4::rotateX(0.0f);
    expectNear(r, Mat4::identity());
}

TEST_CASE("Mat4RotateX.NinetyDegrees", "[Mat4RotateX]")
{
    float angle = std::numbers::pi_v<float> / 2.0f;
    Mat4 r = Mat4::rotateX(angle);

    // X axis unchanged
    CHECK((r[0, 0]) == Catch::Approx(1.0f).margin(kEps));
    // cos(90) ~ 0
    CHECK((r[1, 1]) == Catch::Approx(0.0f).margin(kEps));
    // sin(90) ~ 1
    CHECK((r[2, 1]) == Catch::Approx(1.0f).margin(kEps));
    CHECK((r[1, 2]) == Catch::Approx(-1.0f).margin(kEps));
    CHECK((r[2, 2]) == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("Mat4RotateX.FullRotationIsIdentity", "[Mat4RotateX]")
{
    float angle = 2.0f * std::numbers::pi_v<float>;
    Mat4 r = Mat4::rotateX(angle);
    expectNear(r, Mat4::identity());
}

TEST_CASE("Mat4RotateX.NegativeAngle", "[Mat4RotateX]")
{
    float angle = 0.5f;
    Mat4 pos = Mat4::rotateX(angle);
    Mat4 neg = Mat4::rotateX(-angle);
    // Product should be identity
    Mat4 product = pos * neg;
    expectNear(product, Mat4::identity());
}

// ==========================================================================
// RotateY
// ==========================================================================

TEST_CASE("Mat4RotateY.ZeroAngleIsIdentity", "[Mat4RotateY]")
{
    Mat4 r = Mat4::rotateY(0.0f);
    expectNear(r, Mat4::identity());
}

TEST_CASE("Mat4RotateY.NinetyDegrees", "[Mat4RotateY]")
{
    float angle = std::numbers::pi_v<float> / 2.0f;
    Mat4 r = Mat4::rotateY(angle);

    // Y axis unchanged
    CHECK((r[1, 1]) == Catch::Approx(1.0f).margin(kEps));
    // cos(90) ~ 0
    CHECK((r[0, 0]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((r[2, 2]) == Catch::Approx(0.0f).margin(kEps));
    // sin(90) ~ 1
    CHECK((r[2, 0]) == Catch::Approx(-1.0f).margin(kEps));
    CHECK((r[0, 2]) == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Mat4RotateY.FullRotationIsIdentity", "[Mat4RotateY]")
{
    float angle = 2.0f * std::numbers::pi_v<float>;
    Mat4 r = Mat4::rotateY(angle);
    expectNear(r, Mat4::identity());
}

TEST_CASE("Mat4RotateY.NegativeAngle", "[Mat4RotateY]")
{
    float angle = 0.8f;
    Mat4 pos = Mat4::rotateY(angle);
    Mat4 neg = Mat4::rotateY(-angle);
    Mat4 product = pos * neg;
    expectNear(product, Mat4::identity());
}

// ==========================================================================
// LookAt
// ==========================================================================

TEST_CASE("Mat4LookAt.LookingDownNegativeZ", "[Mat4LookAt]")
{
    // Standard OpenGL-style: eye at origin, looking down -Z, up is +Y
    Mat4 v = Mat4::lookAt({0, 0, 0}, {0, 0, -1}, {0, 1, 0});

    // Should be identity-like (camera at origin looking down -Z)
    // The view matrix maps -Z forward to +Z in view space
    CHECK((v[0, 0]) == Catch::Approx(1.0f).margin(kEps));
    CHECK((v[1, 1]) == Catch::Approx(1.0f).margin(kEps));
    CHECK((v[2, 2]) == Catch::Approx(1.0f).margin(kEps));
    CHECK((v[3, 3]) == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Mat4LookAt.TranslationComponent", "[Mat4LookAt]")
{
    // Eye at (0, 0, 5), looking at origin
    Mat4 v = Mat4::lookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0});

    // Translation in view space: eye is at +5 on Z, so view should translate by -5 on Z
    CHECK((v[2, 3]) == Catch::Approx(-5.0f).margin(kEps));
}

TEST_CASE("Mat4LookAt.LookingAlongPositiveX", "[Mat4LookAt]")
{
    Mat4 v = Mat4::lookAt({0, 0, 0}, {1, 0, 0}, {0, 1, 0});

    // Forward is +X, which maps to -Z in view space
    // So the view matrix's third row should relate to the X world axis
    CHECK((v[2, 0]) == Catch::Approx(-1.0f).margin(kEps));
    // Right is +Z (cross of +X forward and +Y up)
    CHECK((v[0, 2]) == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Mat4LookAt.OffsetEye", "[Mat4LookAt]")
{
    // Just verify it doesn't crash / produces finite values
    Mat4 v = Mat4::lookAt({3, 4, 5}, {0, 0, 0}, {0, 1, 0});
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            INFO("row=" << row << " col=" << col);
            CHECK(std::isfinite(v[row, col]));
        }
    }
}

TEST_CASE("Mat4LookAt.PreservesOrthonormality", "[Mat4LookAt]")
{
    Mat4 v = Mat4::lookAt({1, 2, 3}, {4, 5, 6}, {0, 1, 0});

    // The upper-left 3x3 should be orthonormal (rows are unit vectors, mutually orthogonal)
    for (int r = 0; r < 3; ++r)
    {
        // Row length should be ~1
        float len2 = 0.0f;
        for (int c = 0; c < 3; ++c)
        {
            len2 += v[r, c] * v[r, c];
        }
        INFO("row " << r << " not unit length");
        CHECK(len2 == Catch::Approx(1.0f).margin(kEps));
    }

    // Dot product of row 0 and row 1 should be ~0
    float dot01 = 0.0f;
    for (int c = 0; c < 3; ++c)
    {
        dot01 += v[0, c] * v[1, c];
    }
    CHECK(dot01 == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("Mat4LookAt.SameEyeAndTargetProducesFiniteFallbackView", "[Mat4LookAt]")
{
    Mat4 v = Mat4::lookAt({1, 2, 3}, {1, 2, 3}, {0, 1, 0});

    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            INFO("row=" << row << " col=" << col);
            CHECK(std::isfinite(v[row, col]));
        }
    }
    CHECK((v[2, 3]) == Catch::Approx(-3.0f).margin(kEps));
}

TEST_CASE("Mat4LookAt.ForwardParallelToPreferredUpProducesFiniteOrthonormalView", "[Mat4LookAt]")
{
    Mat4 v = Mat4::lookAt({0, 0, 0}, {0, 1, 0}, {0, 1, 0});

    for (int r = 0; r < 3; ++r)
    {
        float len2 = 0.0f;
        for (int c = 0; c < 3; ++c)
        {
            INFO("row=" << r << " col=" << c);
            CHECK(std::isfinite(v[r, c]));
            len2 += v[r, c] * v[r, c];
        }
        CHECK(len2 == Catch::Approx(1.0f).margin(kEps));
    }
}

TEST_CASE("ViewBasis.HandlesVerticalCameraTargets", "[ViewBasis]")
{
    const ViewBasis upBasis = fire_engine::makeViewBasis({0, 0, 0}, {0, 1, 0});
    const ViewBasis downBasis = fire_engine::makeViewBasis({0, 0, 0}, {0, -1, 0});

    CHECK(upBasis.right.magnitudeSquared() > 0.99f);
    CHECK(upBasis.up.magnitudeSquared() > 0.99f);
    CHECK(downBasis.right.magnitudeSquared() > 0.99f);
    CHECK(downBasis.up.magnitudeSquared() > 0.99f);
    CHECK(Vec3::dotProduct(upBasis.forward, upBasis.up) == Catch::Approx(0.0f).margin(kEps));
    CHECK(Vec3::dotProduct(downBasis.forward, downBasis.up) == Catch::Approx(0.0f).margin(kEps));
}

// ==========================================================================
// Perspective
// ==========================================================================

TEST_CASE("Mat4Perspective.BasicStructure", "[Mat4Perspective]")
{
    float fov = std::numbers::pi_v<float> / 4.0f; // 45 degrees
    Mat4 p = Mat4::perspective(fov, 1.0f, 0.1f, 100.0f);

    // (3,2) should be -1 for Vulkan-style perspective
    CHECK((p[3, 2]) == Catch::Approx(-1.0f).margin(1e-5f));
    // (3,3) should be 0
    CHECK((p[3, 3]) == Catch::Approx(0.0f).margin(1e-5f));
    // Off-diagonal in first two rows/cols should be 0
    CHECK((p[0, 1]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((p[1, 0]) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Mat4Perspective.AspectRatioAffectsX", "[Mat4Perspective]")
{
    float fov = std::numbers::pi_v<float> / 4.0f;
    Mat4 narrow = Mat4::perspective(fov, 0.5f, 0.1f, 100.0f);
    Mat4 wide = Mat4::perspective(fov, 2.0f, 0.1f, 100.0f);

    // Wider aspect => smaller (0,0) value
    CHECK((narrow[0, 0]) > (wide[0, 0]));
    // Y component should be the same regardless of aspect
    CHECK((narrow[1, 1]) == Catch::Approx((wide[1, 1])).margin(1e-5f));
}

TEST_CASE("Mat4Perspective.VulkanYFlip", "[Mat4Perspective]")
{
    float fov = std::numbers::pi_v<float> / 4.0f;
    Mat4 p = Mat4::perspective(fov, 1.0f, 0.1f, 100.0f);

    // Vulkan flips Y: (1,1) should be negative
    CHECK((p[1, 1]) < 0.0f);
}

TEST_CASE("Mat4Perspective.NearFarMapping", "[Mat4Perspective]")
{
    float fov = std::numbers::pi_v<float> / 4.0f;
    Mat4 p = Mat4::perspective(fov, 1.0f, 0.1f, 100.0f);

    // (2,2) and (2,3) define the depth mapping
    // For Vulkan: z_ndc = (far / (near - far)) * z_eye + (near * far) / (near - far)
    float expectedM22 = 100.0f / (0.1f - 100.0f);
    float expectedM23 = (0.1f * 100.0f) / (0.1f - 100.0f);
    CHECK((p[2, 2]) == Catch::Approx(expectedM22).margin(kEps));
    CHECK((p[2, 3]) == Catch::Approx(expectedM23).margin(kEps));
}

TEST_CASE("Mat4Perspective.AllElementsFinite", "[Mat4Perspective]")
{
    float fov = std::numbers::pi_v<float> / 3.0f;
    Mat4 p = Mat4::perspective(fov, 16.0f / 9.0f, 0.01f, 1000.0f);
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            INFO("row=" << row << " col=" << col);
            CHECK(std::isfinite(p[row, col]));
        }
    }
}

// ==========================================================================
// Ortho
// ==========================================================================

TEST_CASE("Mat4Ortho.MapsForwardNegativeZIntoVulkanDepthRange", "[Mat4Ortho]")
{
    Mat4 o = Mat4::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 10.0f);

    Vec4 nearPoint = o * Vec4{0.0f, 0.0f, 0.0f, 1.0f};
    Vec4 farPoint = o * Vec4{0.0f, 0.0f, -10.0f, 1.0f};
    Vec4 midPoint = o * Vec4{0.0f, 0.0f, -5.0f, 1.0f};

    CHECK(nearPoint.z() == Catch::Approx(0.0f).margin(kEps));
    CHECK(farPoint.z() == Catch::Approx(1.0f).margin(kEps));
    CHECK(midPoint.z() == Catch::Approx(0.5f).margin(kEps));
}

TEST_CASE("Mat4Ortho.FlipsYForVulkanClipSpace", "[Mat4Ortho]")
{
    Mat4 o = Mat4::ortho(-2.0f, 2.0f, -2.0f, 2.0f, 0.0f, 10.0f);

    Vec4 top = o * Vec4{0.0f, 2.0f, 0.0f, 1.0f};
    Vec4 bottom = o * Vec4{0.0f, -2.0f, 0.0f, 1.0f};

    CHECK(top.y() == Catch::Approx(-1.0f).margin(kEps));
    CHECK(bottom.y() == Catch::Approx(1.0f).margin(kEps));
}

// ==========================================================================
// Constexpr
// ==========================================================================

TEST_CASE("Mat4Constexpr.DefaultConstruction", "[Mat4Constexpr]")
{
    constexpr Mat4 m;
    static_assert((m[0, 0]) == 0.0f);
    static_assert((m[3, 3]) == 0.0f);
}

TEST_CASE("Mat4Constexpr.Identity", "[Mat4Constexpr]")
{
    constexpr Mat4 I = Mat4::identity();
    static_assert(I[0, 0] == 1.0f);
    static_assert(I[1, 1] == 1.0f);
    static_assert(I[2, 2] == 1.0f);
    static_assert(I[3, 3] == 1.0f);
    static_assert(I[0, 1] == 0.0f);
}

TEST_CASE("Mat4Constexpr.Multiply", "[Mat4Constexpr]")
{
    constexpr Mat4 I = Mat4::identity();
    constexpr Mat4 r = I * I;
    static_assert((r[0, 0]) == 1.0f);
    static_assert((r[1, 1]) == 1.0f);
    static_assert((r[0, 1]) == 0.0f);
}

TEST_CASE("Mat4Constexpr.MultiplyVec4", "[Mat4Constexpr]")
{
    constexpr Vec4 r = Mat4::identity() * Vec4{1.0f, 2.0f, 3.0f, 4.0f};
    static_assert(r.x() == 1.0f);
    static_assert(r.y() == 2.0f);
    static_assert(r.z() == 3.0f);
    static_assert(r.w() == 4.0f);
}

TEST_CASE("Mat4Constexpr.Equality", "[Mat4Constexpr]")
{
    constexpr Mat4 a = Mat4::identity();
    constexpr Mat4 b = Mat4::identity();
    constexpr Mat4 c;
    static_assert(a == b);
    static_assert(!(a == c));
}

// ==========================================================================
// Noexcept
// ==========================================================================

TEST_CASE("Mat4Noexcept.AllOperationsAreNoexcept", "[Mat4Noexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<Mat4>);
    static_assert(test_traits::has_nothrow_mat4_factories<Mat4, Vec3>);
    static_assert(test_traits::has_nothrow_mat4_access<Mat4>);
    static_assert(test_traits::has_nothrow_mat4_arithmetic<Mat4, Vec4>);
    static_assert(test_traits::has_nothrow_equality<Mat4>);
}

// ==========================================================================
// Edge Cases
// ==========================================================================

// ==========================================================================
// RotateZ
// ==========================================================================

TEST_CASE("Mat4RotateZ.ZeroAngleIsIdentity", "[Mat4RotateZ]")
{
    Mat4 r = Mat4::rotateZ(0.0f);
    expectNear(r, Mat4::identity());
}

TEST_CASE("Mat4RotateZ.NinetyDegrees", "[Mat4RotateZ]")
{
    float angle = std::numbers::pi_v<float> / 2.0f;
    Mat4 r = Mat4::rotateZ(angle);

    // Z axis unchanged
    CHECK((r[2, 2]) == Catch::Approx(1.0f).margin(kEps));
    // cos(90) ~ 0
    CHECK((r[0, 0]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((r[1, 1]) == Catch::Approx(0.0f).margin(kEps));
    // sin(90) ~ 1
    CHECK((r[1, 0]) == Catch::Approx(1.0f).margin(kEps));
    CHECK((r[0, 1]) == Catch::Approx(-1.0f).margin(kEps));
}

TEST_CASE("Mat4RotateZ.FullRotationIsIdentity", "[Mat4RotateZ]")
{
    float angle = 2.0f * std::numbers::pi_v<float>;
    Mat4 r = Mat4::rotateZ(angle);
    expectNear(r, Mat4::identity());
}

TEST_CASE("Mat4RotateZ.NegativeAngle", "[Mat4RotateZ]")
{
    float angle = 0.6f;
    Mat4 pos = Mat4::rotateZ(angle);
    Mat4 neg = Mat4::rotateZ(-angle);
    Mat4 product = pos * neg;
    expectNear(product, Mat4::identity());
}

// ==========================================================================
// Translate
// ==========================================================================

TEST_CASE("Mat4Translate.ZeroTranslationIsIdentity", "[Mat4Translate]")
{
    Mat4 t = Mat4::translate({0.0f, 0.0f, 0.0f});
    CHECK(t == Mat4::identity());
}

TEST_CASE("Mat4Translate.SetsColumn3", "[Mat4Translate]")
{
    Mat4 t = Mat4::translate({5.0f, 10.0f, 15.0f});
    CHECK((t[0, 3]) == Catch::Approx(5.0f).margin(1e-5f));
    CHECK((t[1, 3]) == Catch::Approx(10.0f).margin(1e-5f));
    CHECK((t[2, 3]) == Catch::Approx(15.0f).margin(1e-5f));
    CHECK((t[3, 3]) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Mat4Translate.DiagonalRemainsIdentity", "[Mat4Translate]")
{
    Mat4 t = Mat4::translate({1.0f, 2.0f, 3.0f});
    CHECK((t[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((t[1, 1]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((t[2, 2]) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Mat4Translate.NegativeValues", "[Mat4Translate]")
{
    Mat4 t = Mat4::translate({-1.0f, -2.0f, -3.0f});
    CHECK((t[0, 3]) == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK((t[1, 3]) == Catch::Approx(-2.0f).margin(1e-5f));
    CHECK((t[2, 3]) == Catch::Approx(-3.0f).margin(1e-5f));
}

TEST_CASE("Mat4Translate.CompositionAddsTranslations", "[Mat4Translate]")
{
    Mat4 a = Mat4::translate({1.0f, 0.0f, 0.0f});
    Mat4 b = Mat4::translate({0.0f, 2.0f, 3.0f});
    Mat4 ab = a * b;
    CHECK((ab[0, 3]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((ab[1, 3]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((ab[2, 3]) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("Mat4Translate.IsConstexpr", "[Mat4Translate]")
{
    constexpr Mat4 t = Mat4::translate({1.0f, 2.0f, 3.0f});
    static_assert(t[0, 3] == 1.0f);
    static_assert(t[1, 3] == 2.0f);
    static_assert(t[2, 3] == 3.0f);
}

// ==========================================================================
// Scale
// ==========================================================================

TEST_CASE("Mat4Scale.UniformScaleOne", "[Mat4Scale]")
{
    Mat4 s = Mat4::scale({1.0f, 1.0f, 1.0f});
    CHECK(s == Mat4::identity());
}

TEST_CASE("Mat4Scale.SetsDiagonal", "[Mat4Scale]")
{
    Mat4 s = Mat4::scale({2.0f, 3.0f, 4.0f});
    CHECK((s[0, 0]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((s[1, 1]) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK((s[2, 2]) == Catch::Approx(4.0f).margin(1e-5f));
    CHECK((s[3, 3]) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Mat4Scale.OffDiagonalZeros", "[Mat4Scale]")
{
    Mat4 s = Mat4::scale({2.0f, 3.0f, 4.0f});
    CHECK((s[0, 1]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((s[0, 2]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((s[1, 0]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((s[1, 2]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((s[2, 0]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((s[2, 1]) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Mat4Scale.CompositionMultipliesScales", "[Mat4Scale]")
{
    Mat4 a = Mat4::scale({2.0f, 2.0f, 2.0f});
    Mat4 b = Mat4::scale({3.0f, 3.0f, 3.0f});
    Mat4 ab = a * b;
    CHECK((ab[0, 0]) == Catch::Approx(6.0f).margin(1e-5f));
    CHECK((ab[1, 1]) == Catch::Approx(6.0f).margin(1e-5f));
    CHECK((ab[2, 2]) == Catch::Approx(6.0f).margin(1e-5f));
}

TEST_CASE("Mat4Scale.ScaleThenTranslate", "[Mat4Scale]")
{
    // Scale by 2 then translate by (1,0,0): translation is NOT scaled
    Mat4 s = Mat4::scale({2.0f, 2.0f, 2.0f});
    Mat4 t = Mat4::translate({1.0f, 0.0f, 0.0f});
    Mat4 st = s * t;
    CHECK((st[0, 0]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((st[0, 3]) == Catch::Approx(2.0f).margin(1e-5f)); // translation is scaled by parent
}

TEST_CASE("Mat4Scale.TranslateThenScale", "[Mat4Scale]")
{
    // Translate then scale: translation is NOT affected
    Mat4 t = Mat4::translate({1.0f, 0.0f, 0.0f});
    Mat4 s = Mat4::scale({2.0f, 2.0f, 2.0f});
    Mat4 ts = t * s;
    CHECK((ts[0, 0]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((ts[0, 3]) == Catch::Approx(1.0f).margin(1e-5f)); // translation unaffected by scale
}

TEST_CASE("Mat4Scale.IsConstexpr", "[Mat4Scale]")
{
    constexpr Mat4 s = Mat4::scale({2.0f, 3.0f, 4.0f});
    static_assert(s[0, 0] == 2.0f);
    static_assert(s[1, 1] == 3.0f);
    static_assert(s[2, 2] == 4.0f);
}

// ==========================================================================
// Edge Cases
// ==========================================================================

TEST_CASE("Mat4EdgeCases.RotateXThenRotateYNotCommutative", "[Mat4EdgeCases]")
{
    Mat4 rx = Mat4::rotateX(0.3f);
    Mat4 ry = Mat4::rotateY(0.5f);
    Mat4 xy = rx * ry;
    Mat4 yx = ry * rx;
    // Rotations around different axes do not commute
    bool equal = true;
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            if (std::abs(xy[row, col] - yx[row, col]) > kEps)
            {
                equal = false;
            }
        }
    }
    CHECK_FALSE(equal);
}

TEST_CASE("Mat4EdgeCases.InverseRotation", "[Mat4EdgeCases]")
{
    // Rotating by +angle then -angle should give identity
    float angle = 1.23f;
    Mat4 product = Mat4::rotateX(angle) * Mat4::rotateX(-angle);
    expectNear(product, Mat4::identity());

    product = Mat4::rotateY(angle) * Mat4::rotateY(-angle);
    expectNear(product, Mat4::identity());

    product = Mat4::rotateZ(angle) * Mat4::rotateZ(-angle);
    expectNear(product, Mat4::identity());
}

TEST_CASE("Mat4EdgeCases.MultiplyChain", "[Mat4EdgeCases]")
{
    // Chain of rotations should produce finite values
    Mat4 r = Mat4::identity();
    for (int i = 0; i < 100; ++i)
    {
        r = r * Mat4::rotateY(0.01f);
    }
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            INFO("row=" << row << " col=" << col);
            CHECK(std::isfinite(r[row, col]));
        }
    }
}
