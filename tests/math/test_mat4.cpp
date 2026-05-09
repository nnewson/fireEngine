#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/view_basis.hpp>

#include <cmath>
#include <limits>
#include <numbers>

#include <gtest/gtest.h>

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
                EXPECT_FLOAT_EQ((m[row, col]), 1.0f) << "row=" << row << " col=" << col;
            }
            else
            {
                EXPECT_FLOAT_EQ((m[row, col]), 0.0f) << "row=" << row << " col=" << col;
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
            EXPECT_FLOAT_EQ((m[row, col]), 0.0f) << "row=" << row << " col=" << col;
        }
    }
}

static void expectNear(const Mat4& a, const Mat4& b, float eps = kEps)
{
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            EXPECT_NEAR((a[row, col]), (b[row, col]), eps) << "row=" << row << " col=" << col;
        }
    }
}

static void expectNear(const Vec4& v, float x, float y, float z, float w, float eps = kEps)
{
    EXPECT_NEAR(v.x(), x, eps);
    EXPECT_NEAR(v.y(), y, eps);
    EXPECT_NEAR(v.z(), z, eps);
    EXPECT_NEAR(v.w(), w, eps);
}

// ==========================================================================
// Construction
// ==========================================================================

TEST(Mat4Construction, DefaultIsZero)
{
    Mat4 m;
    expectZero(m);
}

TEST(Mat4Construction, CopyConstruct)
{
    Mat4 a = Mat4::identity();
    Mat4 b{a};
    expectIdentity(b);
}

TEST(Mat4Construction, CopyAssign)
{
    Mat4 a = Mat4::identity();
    Mat4 b;
    b = a;
    expectIdentity(b);
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST(Mat4Accessors, SetAndGet)
{
    Mat4 m;
    m[2, 3] = 42.0f;
    EXPECT_FLOAT_EQ((m[2, 3]), 42.0f);
}

TEST(Mat4Accessors, SetDoesNotAffectOtherElements)
{
    Mat4 m;
    m[1, 2] = 7.0f;
    EXPECT_FLOAT_EQ((m[0, 0]), 0.0f);
    EXPECT_FLOAT_EQ((m[1, 2]), 7.0f);
    EXPECT_FLOAT_EQ((m[2, 1]), 0.0f);
}

TEST(Mat4Accessors, DataReturnsColumnMajorPointer)
{
    Mat4 m = Mat4::identity();
    const float* d = m.data();
    // Column-major: m[0]=m(0,0), m[1]=m(1,0), ..., m[4]=m(0,1)
    EXPECT_FLOAT_EQ(d[0], 1.0f);  // (0,0)
    EXPECT_FLOAT_EQ(d[1], 0.0f);  // (1,0)
    EXPECT_FLOAT_EQ(d[4], 0.0f);  // (0,1)
    EXPECT_FLOAT_EQ(d[5], 1.0f);  // (1,1)
    EXPECT_FLOAT_EQ(d[15], 1.0f); // (3,3)
}

// ==========================================================================
// Identity
// ==========================================================================

TEST(Mat4Identity, DiagonalOnes)
{
    Mat4 m = Mat4::identity();
    EXPECT_FLOAT_EQ((m[0, 0]), 1.0f);
    EXPECT_FLOAT_EQ((m[1, 1]), 1.0f);
    EXPECT_FLOAT_EQ((m[2, 2]), 1.0f);
    EXPECT_FLOAT_EQ((m[3, 3]), 1.0f);
}

TEST(Mat4Identity, OffDiagonalZeros)
{
    Mat4 m = Mat4::identity();
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            if (row != col)
            {
                EXPECT_FLOAT_EQ((m[row, col]), 0.0f) << "row=" << row << " col=" << col;
            }
        }
    }
}

// ==========================================================================
// Equality
// ==========================================================================

TEST(Mat4Equality, IdenticalMatrices)
{
    Mat4 a = Mat4::identity();
    Mat4 b = Mat4::identity();
    EXPECT_TRUE(a == b);
}

TEST(Mat4Equality, DifferentMatrices)
{
    Mat4 a = Mat4::identity();
    Mat4 b;
    EXPECT_FALSE(a == b);
}

TEST(Mat4Equality, SingleElementDifference)
{
    Mat4 a = Mat4::identity();
    Mat4 b = Mat4::identity();
    b[2, 3] = 0.001f;
    EXPECT_FALSE(a == b);
}

TEST(Mat4Equality, ZeroMatrices)
{
    Mat4 a;
    Mat4 b;
    EXPECT_TRUE(a == b);
}

// ==========================================================================
// Multiplication
// ==========================================================================

TEST(Mat4Multiply, IdentityTimesIdentity)
{
    Mat4 I = Mat4::identity();
    Mat4 r = I * I;
    expectIdentity(r);
}

TEST(Mat4Multiply, IdentityTimesMatrix)
{
    Mat4 I = Mat4::identity();
    Mat4 a = Mat4::identity();
    a[0, 3] = 5.0f;
    a[1, 3] = 10.0f;
    a[2, 3] = 15.0f;

    Mat4 r = I * a;
    EXPECT_FLOAT_EQ((r[0, 3]), 5.0f);
    EXPECT_FLOAT_EQ((r[1, 3]), 10.0f);
    EXPECT_FLOAT_EQ((r[2, 3]), 15.0f);
}

TEST(Mat4Multiply, MatrixTimesIdentity)
{
    Mat4 I = Mat4::identity();
    Mat4 a = Mat4::identity();
    a[0, 3] = 5.0f;

    Mat4 r = a * I;
    EXPECT_FLOAT_EQ((r[0, 3]), 5.0f);
}

TEST(Mat4Multiply, ZeroMatrixTimesAnything)
{
    Mat4 zero;
    Mat4 a = Mat4::identity();
    Mat4 r = zero * a;
    expectZero(r);
}

TEST(Mat4Multiply, GeneralCase)
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
    EXPECT_FLOAT_EQ((r[0, 0]), 2.0f);
    EXPECT_FLOAT_EQ((r[1, 1]), 3.0f);
    EXPECT_FLOAT_EQ((r[0, 3]), 8.0f);
    EXPECT_FLOAT_EQ((r[1, 3]), 15.0f);
}

TEST(Mat4Multiply, NotCommutative)
{
    Mat4 a = Mat4::identity();
    a[0, 1] = 1.0f; // shear

    Mat4 b = Mat4::identity();
    b[1, 0] = 1.0f; // different shear

    Mat4 ab = a * b;
    Mat4 ba = b * a;
    EXPECT_FALSE(ab == ba);
}

TEST(Mat4Multiply, MultiplyEqualsMatchesMultiply)
{
    Mat4 a = Mat4::identity();
    a[0, 0] = 2.0f;
    Mat4 b = Mat4::identity();
    b[1, 1] = 3.0f;

    Mat4 expected = a * b;
    a *= b;
    EXPECT_TRUE(a == expected);
}

TEST(Mat4Multiply, Associative)
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

TEST(Mat4MultiplyVec4, IdentityPreservesVector)
{
    Vec4 r = Mat4::identity() * Vec4{1.0f, 2.0f, 3.0f, 4.0f};
    expectNear(r, 1.0f, 2.0f, 3.0f, 4.0f);
}

TEST(Mat4MultiplyVec4, TranslationTransformsPosition)
{
    Vec4 r = Mat4::translate({5.0f, 6.0f, 7.0f}) * Vec4{1.0f, 2.0f, 3.0f, 1.0f};
    expectNear(r, 6.0f, 8.0f, 10.0f, 1.0f);
}

TEST(Mat4MultiplyVec4, TranslationDoesNotTransformDirection)
{
    Vec4 r = Mat4::translate({5.0f, 6.0f, 7.0f}) * Vec4{1.0f, 2.0f, 3.0f, 0.0f};
    expectNear(r, 1.0f, 2.0f, 3.0f, 0.0f);
}

TEST(Mat4MultiplyVec4, ScaleTransformsVector)
{
    Vec4 r = Mat4::scale({2.0f, 3.0f, 4.0f}) * Vec4{1.0f, 2.0f, 3.0f, 5.0f};
    expectNear(r, 2.0f, 6.0f, 12.0f, 5.0f);
}

TEST(Mat4MultiplyVec4, GeneralCaseUsesRowByColumnIndexing)
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

TEST(Mat4RotateX, ZeroAngleIsIdentity)
{
    Mat4 r = Mat4::rotateX(0.0f);
    expectNear(r, Mat4::identity());
}

TEST(Mat4RotateX, NinetyDegrees)
{
    float angle = std::numbers::pi_v<float> / 2.0f;
    Mat4 r = Mat4::rotateX(angle);

    // X axis unchanged
    EXPECT_NEAR((r[0, 0]), 1.0f, kEps);
    // cos(90) ~ 0
    EXPECT_NEAR((r[1, 1]), 0.0f, kEps);
    // sin(90) ~ 1
    EXPECT_NEAR((r[2, 1]), 1.0f, kEps);
    EXPECT_NEAR((r[1, 2]), -1.0f, kEps);
    EXPECT_NEAR((r[2, 2]), 0.0f, kEps);
}

TEST(Mat4RotateX, FullRotationIsIdentity)
{
    float angle = 2.0f * std::numbers::pi_v<float>;
    Mat4 r = Mat4::rotateX(angle);
    expectNear(r, Mat4::identity());
}

TEST(Mat4RotateX, NegativeAngle)
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

TEST(Mat4RotateY, ZeroAngleIsIdentity)
{
    Mat4 r = Mat4::rotateY(0.0f);
    expectNear(r, Mat4::identity());
}

TEST(Mat4RotateY, NinetyDegrees)
{
    float angle = std::numbers::pi_v<float> / 2.0f;
    Mat4 r = Mat4::rotateY(angle);

    // Y axis unchanged
    EXPECT_NEAR((r[1, 1]), 1.0f, kEps);
    // cos(90) ~ 0
    EXPECT_NEAR((r[0, 0]), 0.0f, kEps);
    EXPECT_NEAR((r[2, 2]), 0.0f, kEps);
    // sin(90) ~ 1
    EXPECT_NEAR((r[2, 0]), -1.0f, kEps);
    EXPECT_NEAR((r[0, 2]), 1.0f, kEps);
}

TEST(Mat4RotateY, FullRotationIsIdentity)
{
    float angle = 2.0f * std::numbers::pi_v<float>;
    Mat4 r = Mat4::rotateY(angle);
    expectNear(r, Mat4::identity());
}

TEST(Mat4RotateY, NegativeAngle)
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

TEST(Mat4LookAt, LookingDownNegativeZ)
{
    // Standard OpenGL-style: eye at origin, looking down -Z, up is +Y
    Mat4 v = Mat4::lookAt({0, 0, 0}, {0, 0, -1}, {0, 1, 0});

    // Should be identity-like (camera at origin looking down -Z)
    // The view matrix maps -Z forward to +Z in view space
    EXPECT_NEAR((v[0, 0]), 1.0f, kEps);
    EXPECT_NEAR((v[1, 1]), 1.0f, kEps);
    EXPECT_NEAR((v[2, 2]), 1.0f, kEps);
    EXPECT_NEAR((v[3, 3]), 1.0f, kEps);
}

TEST(Mat4LookAt, TranslationComponent)
{
    // Eye at (0, 0, 5), looking at origin
    Mat4 v = Mat4::lookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0});

    // Translation in view space: eye is at +5 on Z, so view should translate by -5 on Z
    EXPECT_NEAR((v[2, 3]), -5.0f, kEps);
}

TEST(Mat4LookAt, LookingAlongPositiveX)
{
    Mat4 v = Mat4::lookAt({0, 0, 0}, {1, 0, 0}, {0, 1, 0});

    // Forward is +X, which maps to -Z in view space
    // So the view matrix's third row should relate to the X world axis
    EXPECT_NEAR((v[2, 0]), -1.0f, kEps);
    // Right is +Z (cross of +X forward and +Y up)
    EXPECT_NEAR((v[0, 2]), 1.0f, kEps);
}

TEST(Mat4LookAt, OffsetEye)
{
    // Just verify it doesn't crash / produces finite values
    Mat4 v = Mat4::lookAt({3, 4, 5}, {0, 0, 0}, {0, 1, 0});
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            EXPECT_TRUE(std::isfinite(v[row, col])) << "row=" << row << " col=" << col;
        }
    }
}

TEST(Mat4LookAt, PreservesOrthonormality)
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
        EXPECT_NEAR(len2, 1.0f, kEps) << "row " << r << " not unit length";
    }

    // Dot product of row 0 and row 1 should be ~0
    float dot01 = 0.0f;
    for (int c = 0; c < 3; ++c)
    {
        dot01 += v[0, c] * v[1, c];
    }
    EXPECT_NEAR(dot01, 0.0f, kEps);
}

TEST(Mat4LookAt, SameEyeAndTargetProducesFiniteFallbackView)
{
    Mat4 v = Mat4::lookAt({1, 2, 3}, {1, 2, 3}, {0, 1, 0});

    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            EXPECT_TRUE(std::isfinite(v[row, col])) << "row=" << row << " col=" << col;
        }
    }
    EXPECT_NEAR((v[2, 3]), -3.0f, kEps);
}

TEST(Mat4LookAt, ForwardParallelToPreferredUpProducesFiniteOrthonormalView)
{
    Mat4 v = Mat4::lookAt({0, 0, 0}, {0, 1, 0}, {0, 1, 0});

    for (int r = 0; r < 3; ++r)
    {
        float len2 = 0.0f;
        for (int c = 0; c < 3; ++c)
        {
            EXPECT_TRUE(std::isfinite(v[r, c])) << "row=" << r << " col=" << c;
            len2 += v[r, c] * v[r, c];
        }
        EXPECT_NEAR(len2, 1.0f, kEps);
    }
}

TEST(ViewBasis, HandlesVerticalCameraTargets)
{
    const ViewBasis upBasis = fire_engine::makeViewBasis({0, 0, 0}, {0, 1, 0});
    const ViewBasis downBasis = fire_engine::makeViewBasis({0, 0, 0}, {0, -1, 0});

    EXPECT_GT(upBasis.right.magnitudeSquared(), 0.99f);
    EXPECT_GT(upBasis.up.magnitudeSquared(), 0.99f);
    EXPECT_GT(downBasis.right.magnitudeSquared(), 0.99f);
    EXPECT_GT(downBasis.up.magnitudeSquared(), 0.99f);
    EXPECT_NEAR(Vec3::dotProduct(upBasis.forward, upBasis.up), 0.0f, kEps);
    EXPECT_NEAR(Vec3::dotProduct(downBasis.forward, downBasis.up), 0.0f, kEps);
}

// ==========================================================================
// Perspective
// ==========================================================================

TEST(Mat4Perspective, BasicStructure)
{
    float fov = std::numbers::pi_v<float> / 4.0f; // 45 degrees
    Mat4 p = Mat4::perspective(fov, 1.0f, 0.1f, 100.0f);

    // (3,2) should be -1 for Vulkan-style perspective
    EXPECT_FLOAT_EQ((p[3, 2]), -1.0f);
    // (3,3) should be 0
    EXPECT_FLOAT_EQ((p[3, 3]), 0.0f);
    // Off-diagonal in first two rows/cols should be 0
    EXPECT_FLOAT_EQ((p[0, 1]), 0.0f);
    EXPECT_FLOAT_EQ((p[1, 0]), 0.0f);
}

TEST(Mat4Perspective, AspectRatioAffectsX)
{
    float fov = std::numbers::pi_v<float> / 4.0f;
    Mat4 narrow = Mat4::perspective(fov, 0.5f, 0.1f, 100.0f);
    Mat4 wide = Mat4::perspective(fov, 2.0f, 0.1f, 100.0f);

    // Wider aspect => smaller (0,0) value
    EXPECT_GT((narrow[0, 0]), (wide[0, 0]));
    // Y component should be the same regardless of aspect
    EXPECT_FLOAT_EQ((narrow[1, 1]), (wide[1, 1]));
}

TEST(Mat4Perspective, VulkanYFlip)
{
    float fov = std::numbers::pi_v<float> / 4.0f;
    Mat4 p = Mat4::perspective(fov, 1.0f, 0.1f, 100.0f);

    // Vulkan flips Y: (1,1) should be negative
    EXPECT_LT((p[1, 1]), 0.0f);
}

TEST(Mat4Perspective, NearFarMapping)
{
    float fov = std::numbers::pi_v<float> / 4.0f;
    Mat4 p = Mat4::perspective(fov, 1.0f, 0.1f, 100.0f);

    // (2,2) and (2,3) define the depth mapping
    // For Vulkan: z_ndc = (far / (near - far)) * z_eye + (near * far) / (near - far)
    float expectedM22 = 100.0f / (0.1f - 100.0f);
    float expectedM23 = (0.1f * 100.0f) / (0.1f - 100.0f);
    EXPECT_NEAR((p[2, 2]), expectedM22, kEps);
    EXPECT_NEAR((p[2, 3]), expectedM23, kEps);
}

TEST(Mat4Perspective, AllElementsFinite)
{
    float fov = std::numbers::pi_v<float> / 3.0f;
    Mat4 p = Mat4::perspective(fov, 16.0f / 9.0f, 0.01f, 1000.0f);
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            EXPECT_TRUE(std::isfinite(p[row, col])) << "row=" << row << " col=" << col;
        }
    }
}

// ==========================================================================
// Ortho
// ==========================================================================

TEST(Mat4Ortho, MapsForwardNegativeZIntoVulkanDepthRange)
{
    Mat4 o = Mat4::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 10.0f);

    Vec4 nearPoint = o * Vec4{0.0f, 0.0f, 0.0f, 1.0f};
    Vec4 farPoint = o * Vec4{0.0f, 0.0f, -10.0f, 1.0f};
    Vec4 midPoint = o * Vec4{0.0f, 0.0f, -5.0f, 1.0f};

    EXPECT_NEAR(nearPoint.z(), 0.0f, kEps);
    EXPECT_NEAR(farPoint.z(), 1.0f, kEps);
    EXPECT_NEAR(midPoint.z(), 0.5f, kEps);
}

TEST(Mat4Ortho, FlipsYForVulkanClipSpace)
{
    Mat4 o = Mat4::ortho(-2.0f, 2.0f, -2.0f, 2.0f, 0.0f, 10.0f);

    Vec4 top = o * Vec4{0.0f, 2.0f, 0.0f, 1.0f};
    Vec4 bottom = o * Vec4{0.0f, -2.0f, 0.0f, 1.0f};

    EXPECT_NEAR(top.y(), -1.0f, kEps);
    EXPECT_NEAR(bottom.y(), 1.0f, kEps);
}

// ==========================================================================
// Constexpr
// ==========================================================================

TEST(Mat4Constexpr, DefaultConstruction)
{
    constexpr Mat4 m;
    static_assert((m[0, 0]) == 0.0f);
    static_assert((m[3, 3]) == 0.0f);
}

TEST(Mat4Constexpr, Identity)
{
    constexpr Mat4 I = Mat4::identity();
    static_assert(I[0, 0] == 1.0f);
    static_assert(I[1, 1] == 1.0f);
    static_assert(I[2, 2] == 1.0f);
    static_assert(I[3, 3] == 1.0f);
    static_assert(I[0, 1] == 0.0f);
}

TEST(Mat4Constexpr, Multiply)
{
    constexpr Mat4 I = Mat4::identity();
    constexpr Mat4 r = I * I;
    static_assert((r[0, 0]) == 1.0f);
    static_assert((r[1, 1]) == 1.0f);
    static_assert((r[0, 1]) == 0.0f);
}

TEST(Mat4Constexpr, MultiplyVec4)
{
    constexpr Vec4 r = Mat4::identity() * Vec4{1.0f, 2.0f, 3.0f, 4.0f};
    static_assert(r.x() == 1.0f);
    static_assert(r.y() == 2.0f);
    static_assert(r.z() == 3.0f);
    static_assert(r.w() == 4.0f);
}

TEST(Mat4Constexpr, Equality)
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

TEST(Mat4Noexcept, AllOperationsAreNoexcept)
{
    Mat4 a = Mat4::identity();
    Mat4 b = Mat4::identity();

    static_assert(noexcept(Mat4{}));
    static_assert(noexcept(Mat4::identity()));
    static_assert(noexcept(a[0, 0]));
    static_assert(noexcept(a[0, 0] = 1.0f));
    static_assert(noexcept(a.data()));
    static_assert(noexcept(a * b));
    static_assert(noexcept(a *= b));
    static_assert(noexcept(a * Vec4{}));
    static_assert(noexcept(a == b));
    static_assert(noexcept(Mat4::rotateX(0.0f)));
    static_assert(noexcept(Mat4::rotateY(0.0f)));
    static_assert(noexcept(Mat4::lookAt({}, {0, 0, -1}, {0, 1, 0})));
    static_assert(noexcept(Mat4::perspective(1.0f, 1.0f, 0.1f, 100.0f)));
}

// ==========================================================================
// Edge Cases
// ==========================================================================

// ==========================================================================
// RotateZ
// ==========================================================================

TEST(Mat4RotateZ, ZeroAngleIsIdentity)
{
    Mat4 r = Mat4::rotateZ(0.0f);
    expectNear(r, Mat4::identity());
}

TEST(Mat4RotateZ, NinetyDegrees)
{
    float angle = std::numbers::pi_v<float> / 2.0f;
    Mat4 r = Mat4::rotateZ(angle);

    // Z axis unchanged
    EXPECT_NEAR((r[2, 2]), 1.0f, kEps);
    // cos(90) ~ 0
    EXPECT_NEAR((r[0, 0]), 0.0f, kEps);
    EXPECT_NEAR((r[1, 1]), 0.0f, kEps);
    // sin(90) ~ 1
    EXPECT_NEAR((r[1, 0]), 1.0f, kEps);
    EXPECT_NEAR((r[0, 1]), -1.0f, kEps);
}

TEST(Mat4RotateZ, FullRotationIsIdentity)
{
    float angle = 2.0f * std::numbers::pi_v<float>;
    Mat4 r = Mat4::rotateZ(angle);
    expectNear(r, Mat4::identity());
}

TEST(Mat4RotateZ, NegativeAngle)
{
    float angle = 0.6f;
    Mat4 pos = Mat4::rotateZ(angle);
    Mat4 neg = Mat4::rotateZ(-angle);
    Mat4 product = pos * neg;
    expectNear(product, Mat4::identity());
}

TEST(Mat4RotateZ, IsNoexcept)
{
    static_assert(noexcept(Mat4::rotateZ(0.0f)));
}

// ==========================================================================
// Translate
// ==========================================================================

TEST(Mat4Translate, ZeroTranslationIsIdentity)
{
    Mat4 t = Mat4::translate({0.0f, 0.0f, 0.0f});
    EXPECT_EQ(t, Mat4::identity());
}

TEST(Mat4Translate, SetsColumn3)
{
    Mat4 t = Mat4::translate({5.0f, 10.0f, 15.0f});
    EXPECT_FLOAT_EQ((t[0, 3]), 5.0f);
    EXPECT_FLOAT_EQ((t[1, 3]), 10.0f);
    EXPECT_FLOAT_EQ((t[2, 3]), 15.0f);
    EXPECT_FLOAT_EQ((t[3, 3]), 1.0f);
}

TEST(Mat4Translate, DiagonalRemainsIdentity)
{
    Mat4 t = Mat4::translate({1.0f, 2.0f, 3.0f});
    EXPECT_FLOAT_EQ((t[0, 0]), 1.0f);
    EXPECT_FLOAT_EQ((t[1, 1]), 1.0f);
    EXPECT_FLOAT_EQ((t[2, 2]), 1.0f);
}

TEST(Mat4Translate, NegativeValues)
{
    Mat4 t = Mat4::translate({-1.0f, -2.0f, -3.0f});
    EXPECT_FLOAT_EQ((t[0, 3]), -1.0f);
    EXPECT_FLOAT_EQ((t[1, 3]), -2.0f);
    EXPECT_FLOAT_EQ((t[2, 3]), -3.0f);
}

TEST(Mat4Translate, CompositionAddsTranslations)
{
    Mat4 a = Mat4::translate({1.0f, 0.0f, 0.0f});
    Mat4 b = Mat4::translate({0.0f, 2.0f, 3.0f});
    Mat4 ab = a * b;
    EXPECT_FLOAT_EQ((ab[0, 3]), 1.0f);
    EXPECT_FLOAT_EQ((ab[1, 3]), 2.0f);
    EXPECT_FLOAT_EQ((ab[2, 3]), 3.0f);
}

TEST(Mat4Translate, IsConstexpr)
{
    constexpr Mat4 t = Mat4::translate({1.0f, 2.0f, 3.0f});
    static_assert(t[0, 3] == 1.0f);
    static_assert(t[1, 3] == 2.0f);
    static_assert(t[2, 3] == 3.0f);
}

// ==========================================================================
// Scale
// ==========================================================================

TEST(Mat4Scale, UniformScaleOne)
{
    Mat4 s = Mat4::scale({1.0f, 1.0f, 1.0f});
    EXPECT_EQ(s, Mat4::identity());
}

TEST(Mat4Scale, SetsDiagonal)
{
    Mat4 s = Mat4::scale({2.0f, 3.0f, 4.0f});
    EXPECT_FLOAT_EQ((s[0, 0]), 2.0f);
    EXPECT_FLOAT_EQ((s[1, 1]), 3.0f);
    EXPECT_FLOAT_EQ((s[2, 2]), 4.0f);
    EXPECT_FLOAT_EQ((s[3, 3]), 1.0f);
}

TEST(Mat4Scale, OffDiagonalZeros)
{
    Mat4 s = Mat4::scale({2.0f, 3.0f, 4.0f});
    EXPECT_FLOAT_EQ((s[0, 1]), 0.0f);
    EXPECT_FLOAT_EQ((s[0, 2]), 0.0f);
    EXPECT_FLOAT_EQ((s[1, 0]), 0.0f);
    EXPECT_FLOAT_EQ((s[1, 2]), 0.0f);
    EXPECT_FLOAT_EQ((s[2, 0]), 0.0f);
    EXPECT_FLOAT_EQ((s[2, 1]), 0.0f);
}

TEST(Mat4Scale, CompositionMultipliesScales)
{
    Mat4 a = Mat4::scale({2.0f, 2.0f, 2.0f});
    Mat4 b = Mat4::scale({3.0f, 3.0f, 3.0f});
    Mat4 ab = a * b;
    EXPECT_FLOAT_EQ((ab[0, 0]), 6.0f);
    EXPECT_FLOAT_EQ((ab[1, 1]), 6.0f);
    EXPECT_FLOAT_EQ((ab[2, 2]), 6.0f);
}

TEST(Mat4Scale, ScaleThenTranslate)
{
    // Scale by 2 then translate by (1,0,0): translation is NOT scaled
    Mat4 s = Mat4::scale({2.0f, 2.0f, 2.0f});
    Mat4 t = Mat4::translate({1.0f, 0.0f, 0.0f});
    Mat4 st = s * t;
    EXPECT_FLOAT_EQ((st[0, 0]), 2.0f);
    EXPECT_FLOAT_EQ((st[0, 3]), 2.0f); // translation is scaled by parent
}

TEST(Mat4Scale, TranslateThenScale)
{
    // Translate then scale: translation is NOT affected
    Mat4 t = Mat4::translate({1.0f, 0.0f, 0.0f});
    Mat4 s = Mat4::scale({2.0f, 2.0f, 2.0f});
    Mat4 ts = t * s;
    EXPECT_FLOAT_EQ((ts[0, 0]), 2.0f);
    EXPECT_FLOAT_EQ((ts[0, 3]), 1.0f); // translation unaffected by scale
}

TEST(Mat4Scale, IsConstexpr)
{
    constexpr Mat4 s = Mat4::scale({2.0f, 3.0f, 4.0f});
    static_assert(s[0, 0] == 2.0f);
    static_assert(s[1, 1] == 3.0f);
    static_assert(s[2, 2] == 4.0f);
}

// ==========================================================================
// Edge Cases
// ==========================================================================

TEST(Mat4EdgeCases, RotateXThenRotateYNotCommutative)
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
    EXPECT_FALSE(equal);
}

TEST(Mat4EdgeCases, InverseRotation)
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

TEST(Mat4EdgeCases, MultiplyChain)
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
            EXPECT_TRUE(std::isfinite(r[row, col])) << "row=" << row << " col=" << col;
        }
    }
}
