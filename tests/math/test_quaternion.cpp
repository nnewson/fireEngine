#include <fire_engine/math/quaternion.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
    CHECK(q.x() == Catch::Approx(ex).margin(eps));
    CHECK(q.y() == Catch::Approx(ey).margin(eps));
    CHECK(q.z() == Catch::Approx(ez).margin(eps));
    CHECK(q.w() == Catch::Approx(ew).margin(eps));
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

TEST_CASE("QuaternionConstruction.DefaultIsIdentity", "[QuaternionConstruction]")
{
    Quaternion q{};
    expectNear(q, 0.0f, 0.0f, 0.0f, 1.0f);
}

TEST_CASE("QuaternionConstruction.ExplicitValues", "[QuaternionConstruction]")
{
    Quaternion q{0.1f, 0.2f, 0.3f, 0.4f};
    expectNear(q, 0.1f, 0.2f, 0.3f, 0.4f);
}

TEST_CASE("QuaternionConstruction.IdentityFactory", "[QuaternionConstruction]")
{
    Quaternion q = Quaternion::identity();
    expectNear(q, 0.0f, 0.0f, 0.0f, 1.0f);
}

TEST_CASE("QuaternionAccessors.Setters", "[QuaternionAccessors]")
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

TEST_CASE("QuaternionEquality.SameValuesAreEqual", "[QuaternionEquality]")
{
    Quaternion a{1.0f, 2.0f, 3.0f, 4.0f};
    Quaternion b{1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(a == b);
}

TEST_CASE("QuaternionEquality.DifferentValuesNotEqual", "[QuaternionEquality]")
{
    Quaternion a{1.0f, 2.0f, 3.0f, 4.0f};
    Quaternion b{1.0f, 2.0f, 3.0f, 4.1f};
    CHECK_FALSE(a == b);
}

TEST_CASE("QuaternionEquality.BitwiseEqualMatchesOperator", "[QuaternionEquality]")
{
    Quaternion a{0.1f, 0.2f, 0.3f, 0.4f};
    Quaternion b{0.1f, 0.2f, 0.3f, 0.4f};
    Quaternion c{0.1f, 0.2f, 0.3f, 0.5f};
    CHECK(a.bitwiseEqual(b));
    CHECK_FALSE(a.bitwiseEqual(c));
}

TEST_CASE("QuaternionEquality.ApproxEqualWithinTolerance", "[QuaternionEquality]")
{
    Quaternion a{0.1f, 0.2f, 0.3f, 0.4f};
    Quaternion b{0.1f + 1e-7f, 0.2f, 0.3f, 0.4f - 1e-7f};
    CHECK_FALSE(a == b);
    CHECK(a.approxEqual(b, 1e-6f));
    CHECK_FALSE(a.approxEqual(b, 1e-9f));
}

TEST_CASE("QuaternionUnaryMinus.NegatesAllComponents", "[QuaternionUnaryMinus]")
{
    Quaternion q{0.1f, -0.2f, 0.3f, -0.4f};
    Quaternion n = -q;
    expectNear(n, -0.1f, 0.2f, -0.3f, 0.4f);
}

// ==========================================================================
// Dot product
// ==========================================================================

TEST_CASE("QuaternionDotProduct.IdentityWithItselfIsOne", "[QuaternionDotProduct]")
{
    Quaternion id = Quaternion::identity();
    CHECK(Quaternion::dotProduct(id, id) == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("QuaternionDotProduct.HandComputed", "[QuaternionDotProduct]")
{
    Quaternion a{1.0f, 2.0f, 3.0f, 4.0f};
    Quaternion b{5.0f, 6.0f, 7.0f, 8.0f};
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    CHECK(Quaternion::dotProduct(a, b) == Catch::Approx(70.0f).margin(kEps));
    CHECK(a.dotProduct(b) == Catch::Approx(70.0f).margin(kEps));
}

// ==========================================================================
// Magnitude / normalise
// ==========================================================================

TEST_CASE("QuaternionMagnitude.UnitQuaternionIsOne", "[QuaternionMagnitude]")
{
    Quaternion id = Quaternion::identity();
    CHECK(id.magnitudeSquared() == Catch::Approx(1.0f).margin(kEps));
    CHECK(id.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("QuaternionMagnitude.NonUnit", "[QuaternionMagnitude]")
{
    Quaternion q{2.0f, 0.0f, 0.0f, 0.0f};
    CHECK(q.magnitudeSquared() == Catch::Approx(4.0f).margin(kEps));
    CHECK(q.magnitude() == Catch::Approx(2.0f).margin(kEps));
}

TEST_CASE("QuaternionNormalise.StaticProducesUnitLength", "[QuaternionNormalise]")
{
    Quaternion q{1.0f, 2.0f, 3.0f, 4.0f};
    Quaternion n = Quaternion::normalise(q);
    CHECK(n.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("QuaternionNormalise.InstanceProducesUnitLength", "[QuaternionNormalise]")
{
    Quaternion q{1.0f, 2.0f, 3.0f, 4.0f};
    q.normalise();
    CHECK(q.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

// ==========================================================================
// SLERP
// ==========================================================================

TEST_CASE("QuaternionSlerp.IdenticalInputs", "[QuaternionSlerp]")
{
    Quaternion q = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.25f);
    Quaternion result = Quaternion::slerp(q, q, 0.5f);
    expectNear(result, q.x(), q.y(), q.z(), q.w());
}

TEST_CASE("QuaternionSlerp.EndpointsAtT0AndT1", "[QuaternionSlerp]")
{
    Quaternion a = Quaternion::identity();
    Quaternion b = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.5f);

    Quaternion r0 = Quaternion::slerp(a, b, 0.0f);
    expectNear(r0, a.x(), a.y(), a.z(), a.w());

    Quaternion r1 = Quaternion::slerp(a, b, 1.0f);
    expectNear(r1, b.x(), b.y(), b.z(), b.w());
}

TEST_CASE("QuaternionSlerp.MidpointOf90DegYIs45DegY", "[QuaternionSlerp]")
{
    Quaternion a = Quaternion::identity();
    Quaternion b = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.5f);
    Quaternion mid = Quaternion::slerp(a, b, 0.5f);

    Quaternion expected = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.25f);
    expectNear(mid, expected.x(), expected.y(), expected.z(), expected.w());
}

TEST_CASE("QuaternionSlerp.ShortPathCorrection", "[QuaternionSlerp]")
{
    // slerp(q, -q, 0.5) should produce something parallel to q (up to sign)
    Quaternion q = axisAngle(0.0f, 0.0f, 1.0f, fire_engine::pi * 0.3f);
    Quaternion result = Quaternion::slerp(q, -q, 0.5f);
    // Since -q represents the same rotation, slerp should short-path and give q back.
    expectNear(result, q.x(), q.y(), q.z(), q.w());
}

TEST_CASE("QuaternionSlerp.NlerpFallbackUnitMagnitude", "[QuaternionSlerp]")
{
    // Two nearly-identical quaternions (dot > 0.9995)
    Quaternion a = Quaternion::identity();
    Quaternion b{0.001f, 0.0f, 0.0f, 0.9999995f};
    Quaternion result = Quaternion::slerp(a, b, 0.5f);
    CHECK(result.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

// ==========================================================================
// toMat4
// ==========================================================================

TEST_CASE("QuaternionToMat4.IdentityIsIdentityMatrix", "[QuaternionToMat4]")
{
    Mat4 m = Quaternion::identity().toMat4();
    Mat4 id = Mat4::identity();
    for (int c = 0; c < 4; ++c)
    {
        for (int r = 0; r < 4; ++r)
        {
            CHECK((m[r, c]) == Catch::Approx((id[r, c])).margin(kEps));
        }
    }
}

TEST_CASE("QuaternionToMat4.NinetyDegreeYRotation", "[QuaternionToMat4]")
{
    Quaternion q = axisAngle(0.0f, 1.0f, 0.0f, fire_engine::pi * 0.5f);
    Mat4 m = q.toMat4();

    // Expected column-major rotation matrix for 90° about Y:
    //  [  0  0  1  0 ]
    //  [  0  1  0  0 ]
    //  [ -1  0  0  0 ]
    //  [  0  0  0  1 ]
    CHECK((m[0, 0]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[1, 0]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[2, 0]) == Catch::Approx(-1.0f).margin(kEps));

    CHECK((m[0, 1]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[1, 1]) == Catch::Approx(1.0f).margin(kEps));
    CHECK((m[2, 1]) == Catch::Approx(0.0f).margin(kEps));

    CHECK((m[0, 2]) == Catch::Approx(1.0f).margin(kEps));
    CHECK((m[1, 2]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[2, 2]) == Catch::Approx(0.0f).margin(kEps));

    // Fourth row/column is identity-like
    CHECK((m[3, 0]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[3, 1]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[3, 2]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[0, 3]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[1, 3]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[2, 3]) == Catch::Approx(0.0f).margin(kEps));
    CHECK((m[3, 3]) == Catch::Approx(1.0f).margin(kEps));
}

// ==========================================================================
// toEulerXYZ — extrinsic XYZ Euler extraction
// ==========================================================================

TEST_CASE("QuaternionToEulerXYZ.IdentityReturnsZero", "[QuaternionToEulerXYZ]")
{
    Vec3 e = Quaternion::identity().toEulerXYZ();
    CHECK(e.x() == Catch::Approx(0.0f).margin(kEps));
    CHECK(e.y() == Catch::Approx(0.0f).margin(kEps));
    CHECK(e.z() == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("QuaternionToEulerXYZ.PureXAxisRotation", "[QuaternionToEulerXYZ]")
{
    float angle = fire_engine::pi / 3.0f;
    Quaternion q = axisAngle(1.0f, 0.0f, 0.0f, angle);
    Vec3 e = q.toEulerXYZ();
    CHECK(e.x() == Catch::Approx(angle).margin(kEps));
    CHECK(e.y() == Catch::Approx(0.0f).margin(kEps));
    CHECK(e.z() == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("QuaternionToEulerXYZ.PureYAxisRotation", "[QuaternionToEulerXYZ]")
{
    float angle = fire_engine::pi / 4.0f;
    Quaternion q = axisAngle(0.0f, 1.0f, 0.0f, angle);
    Vec3 e = q.toEulerXYZ();
    CHECK(e.x() == Catch::Approx(0.0f).margin(kEps));
    CHECK(e.y() == Catch::Approx(angle).margin(kEps));
    CHECK(e.z() == Catch::Approx(0.0f).margin(kEps));
}

TEST_CASE("QuaternionToEulerXYZ.PureZAxisRotation", "[QuaternionToEulerXYZ]")
{
    float angle = fire_engine::pi / 6.0f;
    Quaternion q = axisAngle(0.0f, 0.0f, 1.0f, angle);
    Vec3 e = q.toEulerXYZ();
    CHECK(e.x() == Catch::Approx(0.0f).margin(kEps));
    CHECK(e.y() == Catch::Approx(0.0f).margin(kEps));
    CHECK(e.z() == Catch::Approx(angle).margin(kEps));
}

TEST_CASE("QuaternionToEulerXYZ.DecalBlendRotationExtractsXAxis", "[QuaternionToEulerXYZ]")
{
    // AlphaBlendModeTest DecalBlend: qx=-0.47186, qy=0, qz=0, qw=0.88167
    Quaternion q{-0.47185850f, 0.0f, 0.0f, 0.88167441f};
    Vec3 e = q.toEulerXYZ();
    // Should be a pure X rotation of ~-56.3 degrees.
    CHECK(e.x() == Catch::Approx(-0.98279f).margin(1e-4f));
    CHECK(e.y() == Catch::Approx(0.0f).margin(kEps));
    CHECK(e.z() == Catch::Approx(0.0f).margin(kEps));
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

TEST_CASE("QuaternionFromVectors.ParallelInputsReturnIdentity", "[QuaternionFromVectors]")
{
    Vec3 v = Vec3::normalise(Vec3{0.3f, 0.7f, -0.2f});
    Quaternion q = Quaternion::fromVectors(v, v);
    expectNear(q, 0.0f, 0.0f, 0.0f, 1.0f);
}

TEST_CASE("QuaternionFromVectors.AntiparallelInputsRotate180", "[QuaternionFromVectors]")
{
    Vec3 from{0.0f, 0.0f, -1.0f};
    Vec3 to{0.0f, 0.0f, 1.0f};
    Quaternion q = Quaternion::fromVectors(from, to);

    // 180° rotation about an axis orthogonal to `from`, so w == 0 and the
    // axis lies in the X/Y plane.
    CHECK(q.w() == Catch::Approx(0.0f).margin(kEps));
    CHECK(q.z() == Catch::Approx(0.0f).margin(kEps));
    // Rotating `from` should still land on `to`.
    Vec3 rot = rotated(q, from);
    CHECK(rot.x() == Catch::Approx(to.x()).margin(kEps));
    CHECK(rot.y() == Catch::Approx(to.y()).margin(kEps));
    CHECK(rot.z() == Catch::Approx(to.z()).margin(kEps));
}

TEST_CASE("QuaternionFromVectors.ArbitraryPairRotatesFromOntoTo", "[QuaternionFromVectors]")
{
    Vec3 from = Vec3::normalise(Vec3{0.0f, 0.0f, -1.0f});
    Vec3 to = Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f});
    Quaternion q = Quaternion::fromVectors(from, to);

    Vec3 rot = rotated(q, from);
    CHECK(rot.x() == Catch::Approx(to.x()).margin(kEps));
    CHECK(rot.y() == Catch::Approx(to.y()).margin(kEps));
    CHECK(rot.z() == Catch::Approx(to.z()).margin(kEps));
}

TEST_CASE("QuaternionFromVectors.ResultIsUnitMagnitude", "[QuaternionFromVectors]")
{
    Vec3 from = Vec3::normalise(Vec3{0.0f, 0.0f, -1.0f});
    Vec3 to = Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f});
    Quaternion q = Quaternion::fromVectors(from, to);
    CHECK(q.magnitude() == Catch::Approx(1.0f).margin(kEps));
}

TEST_CASE("Quaternion.ConjugateAndRotate", "[Quaternion]")
{
    // 90° about Z maps +x → +y.
    const float h = std::sqrt(0.5f);
    const Quaternion qz{0.0f, 0.0f, h, h};
    CHECK(qz.rotate(Vec3{1.0f, 0.0f, 0.0f}).approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 1e-5f));

    // Conjugate is {-x,-y,-z,w} and undoes the rotation for a unit quaternion.
    const Quaternion c = qz.conjugate();
    CHECK(c.x() == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(c.z() == Catch::Approx(-h).margin(1e-6f));
    CHECK(c.w() == Catch::Approx(h).margin(1e-6f));
    CHECK(c.rotate(qz.rotate(Vec3{1.0f, 2.0f, 3.0f})).approxEqual(Vec3{1.0f, 2.0f, 3.0f}, 1e-5f));

    // rotate() agrees with the toMat4() path.
    const Vec3 v{0.3f, -0.7f, 1.1f};
    const Vec4 m = qz.toMat4() * Vec4{v.x(), v.y(), v.z(), 0.0f};
    CHECK(qz.rotate(v).approxEqual(Vec3{m.x(), m.y(), m.z()}, 1e-5f));
}
