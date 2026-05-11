#include <fire_engine/math/quaternion.hpp>

#include <cmath>

#include <gtest/gtest.h>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>

using fire_engine::Mat4;
using fire_engine::Quaternion;
using fire_engine::Vec3;
using fire_engine::Vec4;

// ---- Helpers ----

static constexpr float kEps = 1e-5f;

static void expectNear(const Quaternion& q, float ex, float ey, float ez, float ew,
                       float eps = kEps)
{
    EXPECT_NEAR(q.x(), ex, eps);
    EXPECT_NEAR(q.y(), ey, eps);
    EXPECT_NEAR(q.z(), ez, eps);
    EXPECT_NEAR(q.w(), ew, eps);
}

// Build a quaternion representing a rotation of `angle` radians about an axis.
// Assumes the axis is a unit vector.
static Quaternion axisAngle(float ax, float ay, float az, float angle)
{
    float s = std::sin(angle * 0.5f);
    return {ax * s, ay * s, az * s, std::cos(angle * 0.5f)};
}

// ==========================================================================
// Construction / accessors
// ==========================================================================

TEST(QuaternionConstruction, DefaultIsIdentity)
{
    Quaternion q{};
    expectNear(q, 0.0f, 0.0f, 0.0f, 1.0f);
}

TEST(QuaternionConstruction, ExplicitValues)
{
    Quaternion q{0.1f, 0.2f, 0.3f, 0.4f};
    expectNear(q, 0.1f, 0.2f, 0.3f, 0.4f);
}

TEST(QuaternionConstruction, IdentityFactory)
{
    Quaternion q = Quaternion::identity();
    expectNear(q, 0.0f, 0.0f, 0.0f, 1.0f);
}

TEST(QuaternionAccessors, Setters)
{
    Quaternion q;
    q.x(1.5f);
    q.y(2.5f);
    q.z(3.5f);
    q.w(4.5f);
    expectNear(q, 1.5f, 2.5f, 3.5f, 4.5f);
}

// ==========================================================================
// Equality / unary minus
// ==========================================================================

TEST(QuaternionEquality, SameValuesAreEqual)
{
    Quaternion a{1.0f, 2.0f, 3.0f, 4.0f};
    Quaternion b{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_TRUE(a == b);
}

TEST(QuaternionEquality, DifferentValuesNotEqual)
{
    Quaternion a{1.0f, 2.0f, 3.0f, 4.0f};
    Quaternion b{1.0f, 2.0f, 3.0f, 4.1f};
    EXPECT_FALSE(a == b);
}

TEST(QuaternionUnaryMinus, NegatesAllComponents)
{
    Quaternion q{0.1f, -0.2f, 0.3f, -0.4f};
    Quaternion n = -q;
    expectNear(n, -0.1f, 0.2f, -0.3f, 0.4f);
}

// ==========================================================================
// Dot product
// ==========================================================================

TEST(QuaternionDotProduct, IdentityWithItselfIsOne)
{
    Quaternion id = Quaternion::identity();
    EXPECT_NEAR(Quaternion::dotProduct(id, id), 1.0f, kEps);
}

TEST(QuaternionDotProduct, HandComputed)
{
    Quaternion a{1.0f, 2.0f, 3.0f, 4.0f};
    Quaternion b{5.0f, 6.0f, 7.0f, 8.0f};
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_NEAR(Quaternion::dotProduct(a, b), 70.0f, kEps);
    EXPECT_NEAR(a.dotProduct(b), 70.0f, kEps);
}

// ==========================================================================
// Magnitude / normalise
// ==========================================================================

TEST(QuaternionMagnitude, UnitQuaternionIsOne)
{
    Quaternion id = Quaternion::identity();
    EXPECT_NEAR(id.magnitudeSquared(), 1.0f, kEps);
    EXPECT_NEAR(id.magnitude(), 1.0f, kEps);
}

TEST(QuaternionMagnitude, NonUnit)
{
    Quaternion q{2.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_NEAR(q.magnitudeSquared(), 4.0f, kEps);
    EXPECT_NEAR(q.magnitude(), 2.0f, kEps);
}

TEST(QuaternionNormalise, StaticProducesUnitLength)
{
    Quaternion q{1.0f, 2.0f, 3.0f, 4.0f};
    Quaternion n = Quaternion::normalise(q);
    EXPECT_NEAR(n.magnitude(), 1.0f, kEps);
}

TEST(QuaternionNormalise, InstanceProducesUnitLength)
{
    Quaternion q{1.0f, 2.0f, 3.0f, 4.0f};
    q.normalise();
    EXPECT_NEAR(q.magnitude(), 1.0f, kEps);
}

// ==========================================================================
// SLERP
// ==========================================================================

TEST(QuaternionSlerp, IdenticalInputs)
{
    Quaternion q = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.25f);
    Quaternion result = Quaternion::slerp(q, q, 0.5f);
    expectNear(result, q.x(), q.y(), q.z(), q.w());
}

TEST(QuaternionSlerp, EndpointsAtT0AndT1)
{
    Quaternion a = Quaternion::identity();
    Quaternion b = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.5f);

    Quaternion r0 = Quaternion::slerp(a, b, 0.0f);
    expectNear(r0, a.x(), a.y(), a.z(), a.w());

    Quaternion r1 = Quaternion::slerp(a, b, 1.0f);
    expectNear(r1, b.x(), b.y(), b.z(), b.w());
}

TEST(QuaternionSlerp, MidpointOf90DegYIs45DegY)
{
    Quaternion a = Quaternion::identity();
    Quaternion b = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.5f);
    Quaternion mid = Quaternion::slerp(a, b, 0.5f);

    Quaternion expected = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.25f);
    expectNear(mid, expected.x(), expected.y(), expected.z(), expected.w());
}

TEST(QuaternionSlerp, ShortPathCorrection)
{
    // slerp(q, -q, 0.5) should produce something parallel to q (up to sign)
    Quaternion q = axisAngle(0.0f, 0.0f, 1.0f, fire_engine::pi * 0.3f);
    Quaternion result = Quaternion::slerp(q, -q, 0.5f);
    // Since -q represents the same rotation, slerp should short-path and give q back.
    expectNear(result, q.x(), q.y(), q.z(), q.w());
}

TEST(QuaternionSlerp, NlerpFallbackUnitMagnitude)
{
    // Two nearly-identical quaternions (dot > 0.9995)
    Quaternion a = Quaternion::identity();
    Quaternion b{0.001f, 0.0f, 0.0f, 0.9999995f};
    Quaternion result = Quaternion::slerp(a, b, 0.5f);
    EXPECT_NEAR(result.magnitude(), 1.0f, kEps);
}

// ==========================================================================
// toMat4
// ==========================================================================

TEST(QuaternionToMat4, IdentityIsIdentityMatrix)
{
    Mat4 m = Quaternion::identity().toMat4();
    Mat4 id = Mat4::identity();
    for (int c = 0; c < 4; ++c)
    {
        for (int r = 0; r < 4; ++r)
        {
            EXPECT_NEAR((m[r, c]), (id[r, c]), kEps);
        }
    }
}

TEST(QuaternionToMat4, NinetyDegreeYRotation)
{
    Quaternion q = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.5f);
    Mat4 m = q.toMat4();

    // Expected column-major rotation matrix for 90° about Y:
    //  [  0  0  1  0 ]
    //  [  0  1  0  0 ]
    //  [ -1  0  0  0 ]
    //  [  0  0  0  1 ]
    EXPECT_NEAR((m[0, 0]), 0.0f, kEps);
    EXPECT_NEAR((m[1, 0]), 0.0f, kEps);
    EXPECT_NEAR((m[2, 0]), -1.0f, kEps);

    EXPECT_NEAR((m[0, 1]), 0.0f, kEps);
    EXPECT_NEAR((m[1, 1]), 1.0f, kEps);
    EXPECT_NEAR((m[2, 1]), 0.0f, kEps);

    EXPECT_NEAR((m[0, 2]), 1.0f, kEps);
    EXPECT_NEAR((m[1, 2]), 0.0f, kEps);
    EXPECT_NEAR((m[2, 2]), 0.0f, kEps);

    // Fourth row/column is identity-like
    EXPECT_NEAR((m[3, 0]), 0.0f, kEps);
    EXPECT_NEAR((m[3, 1]), 0.0f, kEps);
    EXPECT_NEAR((m[3, 2]), 0.0f, kEps);
    EXPECT_NEAR((m[0, 3]), 0.0f, kEps);
    EXPECT_NEAR((m[1, 3]), 0.0f, kEps);
    EXPECT_NEAR((m[2, 3]), 0.0f, kEps);
    EXPECT_NEAR((m[3, 3]), 1.0f, kEps);
}

// ==========================================================================
// toEulerXYZ — extrinsic XYZ Euler extraction
// ==========================================================================

TEST(QuaternionToEulerXYZ, IdentityReturnsZero)
{
    Vec3 e = Quaternion::identity().toEulerXYZ();
    EXPECT_NEAR(e.x(), 0.0f, kEps);
    EXPECT_NEAR(e.y(), 0.0f, kEps);
    EXPECT_NEAR(e.z(), 0.0f, kEps);
}

TEST(QuaternionToEulerXYZ, PureXAxisRotation)
{
    float angle = fire_engine::pi / 3.0f;
    Quaternion q = axisAngle(1.0f, 0.0f, 0.0f, angle);
    Vec3 e = q.toEulerXYZ();
    EXPECT_NEAR(e.x(), angle, kEps);
    EXPECT_NEAR(e.y(), 0.0f, kEps);
    EXPECT_NEAR(e.z(), 0.0f, kEps);
}

TEST(QuaternionToEulerXYZ, PureYAxisRotation)
{
    float angle = fire_engine::pi / 4.0f;
    Quaternion q = axisAngle(0.0f, 1.0f, 0.0f, angle);
    Vec3 e = q.toEulerXYZ();
    EXPECT_NEAR(e.x(), 0.0f, kEps);
    EXPECT_NEAR(e.y(), angle, kEps);
    EXPECT_NEAR(e.z(), 0.0f, kEps);
}

TEST(QuaternionToEulerXYZ, PureZAxisRotation)
{
    float angle = fire_engine::pi / 6.0f;
    Quaternion q = axisAngle(0.0f, 0.0f, 1.0f, angle);
    Vec3 e = q.toEulerXYZ();
    EXPECT_NEAR(e.x(), 0.0f, kEps);
    EXPECT_NEAR(e.y(), 0.0f, kEps);
    EXPECT_NEAR(e.z(), angle, kEps);
}

TEST(QuaternionToEulerXYZ, DecalBlendRotationExtractsXAxis)
{
    // AlphaBlendModeTest DecalBlend: qx=-0.47186, qy=0, qz=0, qw=0.88167
    Quaternion q{-0.47185850f, 0.0f, 0.0f, 0.88167441f};
    Vec3 e = q.toEulerXYZ();
    // Should be a pure X rotation of ~-56.3 degrees.
    EXPECT_NEAR(e.x(), -0.98279f, 1e-4f);
    EXPECT_NEAR(e.y(), 0.0f, kEps);
    EXPECT_NEAR(e.z(), 0.0f, kEps);
}

// ==========================================================================
// fromVectors
// ==========================================================================

namespace
{
// Rotate v by q via the rotation matrix derived from q (matches what
// callers using the quaternion as a transform will see).
Vec3 rotated(const Quaternion& q, Vec3 v)
{
    Vec4 r = q.toMat4() * Vec4{v};
    return {r.x(), r.y(), r.z()};
}
} // namespace

TEST(QuaternionFromVectors, ParallelInputsReturnIdentity)
{
    Vec3 v = Vec3::normalise(Vec3{0.3f, 0.7f, -0.2f});
    Quaternion q = Quaternion::fromVectors(v, v);
    expectNear(q, 0.0f, 0.0f, 0.0f, 1.0f);
}

TEST(QuaternionFromVectors, AntiparallelInputsRotate180)
{
    Vec3 from{0.0f, 0.0f, -1.0f};
    Vec3 to{0.0f, 0.0f, 1.0f};
    Quaternion q = Quaternion::fromVectors(from, to);

    // 180° rotation about an axis orthogonal to `from`, so w == 0 and the
    // axis lies in the X/Y plane.
    EXPECT_NEAR(q.w(), 0.0f, kEps);
    EXPECT_NEAR(q.z(), 0.0f, kEps);
    // Rotating `from` should still land on `to`.
    Vec3 rot = rotated(q, from);
    EXPECT_NEAR(rot.x(), to.x(), kEps);
    EXPECT_NEAR(rot.y(), to.y(), kEps);
    EXPECT_NEAR(rot.z(), to.z(), kEps);
}

TEST(QuaternionFromVectors, ArbitraryPairRotatesFromOntoTo)
{
    Vec3 from = Vec3::normalise(Vec3{0.0f, 0.0f, -1.0f});
    Vec3 to = Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f});
    Quaternion q = Quaternion::fromVectors(from, to);

    Vec3 rot = rotated(q, from);
    EXPECT_NEAR(rot.x(), to.x(), kEps);
    EXPECT_NEAR(rot.y(), to.y(), kEps);
    EXPECT_NEAR(rot.z(), to.z(), kEps);
}

TEST(QuaternionFromVectors, ResultIsUnitMagnitude)
{
    Vec3 from = Vec3::normalise(Vec3{0.0f, 0.0f, -1.0f});
    Vec3 to = Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f});
    Quaternion q = Quaternion::fromVectors(from, to);
    EXPECT_NEAR(q.magnitude(), 1.0f, kEps);
}
