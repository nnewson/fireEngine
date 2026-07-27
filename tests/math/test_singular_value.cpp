#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/singular_value.hpp>

using fire_engine::largestSingularValue;
using fire_engine::linearPart;
using fire_engine::Mat3;
using fire_engine::Mat4;
using fire_engine::Quaternion;
using fire_engine::Vec3;

// ---------------------------------------------------------------------------
// The conservative σ_max bound, shared by VDPM's per-instance refinement and the shadow-LOD
// projection. Both bound an object-space deviation radius into world space, so an UNDER-estimate
// silently under-refines — geometry that should have stayed detailed gets coarsened. These pin the
// exact cases where a cheaper implementation would be wrong.
// ---------------------------------------------------------------------------

TEST_CASE("SingularValue.IdentityAndUniformScaleAreExact", "[SingularValue]")
{
    CHECK(largestSingularValue(Mat3::identity()) == Catch::Approx(1.0f).epsilon(1e-5));
    CHECK(largestSingularValue(Mat3::diagonal(Vec3{3.0f, 3.0f, 3.0f})) ==
          Catch::Approx(3.0f).epsilon(1e-5));
}

TEST_CASE("SingularValue.NonUniformScaleTakesTheLargestAxis", "[SingularValue]")
{
    // The bound is exact for a diagonal matrix: mᵀm is diagonal, so its row sums ARE its
    // eigenvalues.
    CHECK(largestSingularValue(Mat3::diagonal(Vec3{2.0f, 0.5f, 1.0f})) ==
          Catch::Approx(2.0f).epsilon(1e-5));
    CHECK(largestSingularValue(Mat3::diagonal(Vec3{0.25f, 0.25f, 7.0f})) ==
          Catch::Approx(7.0f).epsilon(1e-5));
}

TEST_CASE("SingularValue.RotationDoesNotStretch", "[SingularValue]")
{
    const Mat3 rotation =
        Mat3::fromQuaternion(Quaternion::fromAxisAngle(Vec3{0.3f, 1.0f, -0.2f}.normalise(), 0.9f));

    CHECK(largestSingularValue(rotation) == Catch::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("SingularValue.ReflectionDoesNotStretch", "[SingularValue]")
{
    // A negative determinant is still an isometry — mirroring must not read as scaling.
    CHECK(largestSingularValue(Mat3::diagonal(Vec3{1.0f, -1.0f, 1.0f})) ==
          Catch::Approx(1.0f).epsilon(1e-5));
    CHECK(largestSingularValue(Mat3::diagonal(Vec3{-4.0f, 1.0f, 1.0f})) ==
          Catch::Approx(4.0f).epsilon(1e-5));
}

TEST_CASE("SingularValue.SingularMatrixHasFiniteStretch", "[SingularValue]")
{
    // A collapsed axis removes stretch; it does not create unbounded stretch. The caller must NOT
    // treat this as an error case — only non-finite input is meaningless.
    const float sigma = largestSingularValue(Mat3::diagonal(Vec3{5.0f, 0.0f, 2.0f}));

    CHECK(std::isfinite(sigma));
    CHECK(sigma == Catch::Approx(5.0f).epsilon(1e-5));
    CHECK(largestSingularValue(Mat3::diagonal(Vec3{0.0f, 0.0f, 0.0f})) ==
          Catch::Approx(0.0f).margin(1e-6));
}

TEST_CASE("SingularValue.InPlaneShearIsNotUnderEstimated", "[SingularValue]")
{
    // The case named in the implementation comment: σ_max = 10, but the dominant mᵀm eigenvector
    // (1,-1,0) is orthogonal to a (1,1,1) start vector, so power iteration returns 1 — a 10x
    // under-estimate that would silently under-refine every instance carrying this transform.
    const Mat3 shear =
        Mat3::fromColumns({5.5f, -4.5f, 0.0f}, {-4.5f, 5.5f, 0.0f}, {0.0f, 0.0f, 1.0f});

    const float sigma = largestSingularValue(shear);

    CHECK(sigma >= 10.0f); // never under-estimates the true value
    CHECK(sigma <= 20.0f); // and the Gershgorin slack stays sane rather than unusable
}

TEST_CASE("SingularValue.NeverUnderEstimatesAnyDirection", "[SingularValue]")
{
    // The defining property, sampled directly: no unit direction may be stretched by more than the
    // reported bound. This is what makes an object-space radius safe to scale into world space.
    const Mat3 awkward =
        Mat3::fromColumns({1.5f, 0.4f, -0.2f}, {-0.9f, 2.2f, 0.3f}, {0.1f, -0.7f, 0.6f});
    const float sigma = largestSingularValue(awkward);

    constexpr int steps = 24;
    for (int i = 0; i < steps; ++i)
    {
        for (int j = 0; j < steps; ++j)
        {
            const float theta = static_cast<float>(i) / steps * 2.0f * std::numbers::pi_v<float>;
            const float phi = static_cast<float>(j) / steps * std::numbers::pi_v<float>;
            const Vec3 dir{std::sin(phi) * std::cos(theta), std::cos(phi),
                           std::sin(phi) * std::sin(theta)};
            CHECK((awkward * dir).magnitude() <= sigma + 1e-4f);
        }
    }
}

TEST_CASE("SingularValue.LinearPartDropsTranslation", "[SingularValue]")
{
    // A translated instance is not a stretched one: only the upper-left 3x3 can scale a radius.
    Mat4 world = Mat4::identity();
    world[0, 3] = 100.0f;
    world[1, 3] = -50.0f;
    world[2, 3] = 7.0f;

    CHECK(largestSingularValue(linearPart(world)) == Catch::Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("SingularValue.NonFiniteElementsPropagateAsInfinity", "[SingularValue]")
{
    // Left to the arithmetic a NaN is SWALLOWED: std::max(finite, NaN) keeps the finite operand, so
    // a NaN column would yield a plausible scale and let one bad transform under-refine an
    // instance. The contract is that non-finite input is rejected outright.
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float inf = std::numeric_limits<float>::infinity();

    CHECK(std::isinf(largestSingularValue(
        Mat3::fromColumns({nan, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}))));
    CHECK(std::isinf(largestSingularValue(
        Mat3::fromColumns({1.0f, 0.0f, 0.0f}, {0.0f, nan, 0.0f}, {0.0f, 0.0f, 1.0f}))));
    CHECK(std::isinf(largestSingularValue(
        Mat3::fromColumns({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, inf}))));
}
