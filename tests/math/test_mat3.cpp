#include <fire_engine/math/mat3.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <optional>

using fire_engine::Mat3;
using fire_engine::Quaternion;
using fire_engine::Vec3;

TEST_CASE("Mat3.IdentityAndDiagonalApplyToVector", "[Mat3]")
{
    CHECK((Mat3::identity() * Vec3{1.0f, 2.0f, 3.0f}).approxEqual(Vec3{1.0f, 2.0f, 3.0f}, 1e-6f));

    const Mat3 d = Mat3::diagonal({2.0f, 3.0f, 4.0f});
    CHECK((d * Vec3{1.0f, 1.0f, 1.0f}).approxEqual(Vec3{2.0f, 3.0f, 4.0f}, 1e-6f));
    CHECK((d * Vec3{5.0f, 5.0f, 5.0f}).approxEqual(Vec3{10.0f, 15.0f, 20.0f}, 1e-6f));
}

TEST_CASE("Mat3.FromQuaternionMatchesToMat4AndRotatesVectors", "[Mat3]")
{
    const float h = std::sqrt(0.5f);
    const Quaternion qz{0.0f, 0.0f, h, h}; // 90° about Z
    const Mat3 r = Mat3::fromQuaternion(qz);
    const auto m4 = qz.toMat4();

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            CHECK(r[row, col] == Catch::Approx(m4[row, col]).margin(1e-5f));
        }
    }

    // 90° about Z maps +x → +y (same as the quaternion's own rotate).
    CHECK((r * Vec3{1.0f, 0.0f, 0.0f}).approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 1e-5f));
    CHECK((r * Vec3{0.2f, 0.5f, -0.9f}).approxEqual(qz.rotate(Vec3{0.2f, 0.5f, -0.9f}), 1e-5f));
}

TEST_CASE("Mat3.RotationTimesTransposeIsIdentity", "[Mat3]")
{
    // A normalised arbitrary quaternion ({1,2,3,4}/√30).
    const float inv = 1.0f / std::sqrt(30.0f);
    const Quaternion q{1.0f * inv, 2.0f * inv, 3.0f * inv, 4.0f * inv};
    const Mat3 r = Mat3::fromQuaternion(q);
    CHECK((r * r.transpose()).approxEqual(Mat3::identity(), 1e-4f));
}

TEST_CASE("Mat3.MultiplyAndTranspose", "[Mat3]")
{
    const Mat3 a = Mat3::fromColumns({1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 10.0f});

    CHECK(a.transpose().transpose() == a);
    CHECK((a * Mat3::identity()) == a);
    CHECK((Mat3::identity() * a) == a);

    // (A·B)·v == A·(B·v) for a concrete pair.
    const Mat3 b = Mat3::diagonal({2.0f, 0.5f, 3.0f});
    const Vec3 v{1.0f, -2.0f, 0.5f};
    CHECK(((a * b) * v).approxEqual(a * (b * v), 1e-5f));
}

TEST_CASE("Mat3.InverseTimesMatrixIsIdentity", "[Mat3]")
{
    // A non-symmetric invertible matrix (det != 0).
    const Mat3 a = Mat3::fromColumns({1.0f, 2.0f, 3.0f}, {0.0f, 1.0f, 4.0f}, {5.0f, 6.0f, 0.0f});
    const std::optional<Mat3> ai = a.tryInverse();
    REQUIRE(ai.has_value());
    CHECK((a * *ai).approxEqual(Mat3::identity(), 1e-4f));
    CHECK((*ai * a).approxEqual(Mat3::identity(), 1e-4f));

    // Identity inverts to itself; a diagonal inverts component-wise.
    REQUIRE(Mat3::identity().tryInverse().has_value());
    CHECK(Mat3::identity().tryInverse()->approxEqual(Mat3::identity(), 1e-6f));
    REQUIRE(Mat3::diagonal({2.0f, 4.0f, 0.5f}).tryInverse().has_value());
    CHECK(Mat3::diagonal({2.0f, 4.0f, 0.5f})
              .tryInverse()
              ->approxEqual(Mat3::diagonal({0.5f, 0.25f, 2.0f}), 1e-6f));

    // A symmetric positive-definite matrix (the shape of the joint's effective-mass K).
    const Mat3 k = Mat3::fromColumns({4.0f, 1.0f, 0.5f}, {1.0f, 3.0f, 0.2f}, {0.5f, 0.2f, 2.0f});
    REQUIRE(k.tryInverse().has_value());
    CHECK((k * *k.tryInverse()).approxEqual(Mat3::identity(), 1e-4f));
}

TEST_CASE("Mat3.TryInverseAcceptsTinyWellConditionedTransforms", "[Mat3]")
{
    // THE tier-0 finding. A uniform scale of 1e-5 is perfectly conditioned — its inverse is a
    // uniform 1e5 — but its determinant is 1e-15, which the old absolute threshold (|det| <= 1e-12)
    // called singular and answered with a zero matrix. Size is not conditioning.
    const Mat3 tiny = Mat3::diagonal({1.0e-5f, 1.0e-5f, 1.0e-5f});
    const std::optional<Mat3> inverse = tiny.tryInverse();
    REQUIRE(inverse.has_value());
    CHECK(inverse->approxEqual(Mat3::diagonal({1.0e5f, 1.0e5f, 1.0e5f}), 1.0f));
    CHECK((tiny * *inverse).approxEqual(Mat3::identity(), 1e-4f));

    // The same shape across several decades: the answer must not depend on absolute size at all.
    for (const float scale : {1.0e-8f, 1.0e-4f, 1.0f, 1.0e4f, 1.0e8f})
    {
        const Mat3 scaled = Mat3::diagonal({scale, scale * 2.0f, scale * 0.5f});
        const std::optional<Mat3> scaledInverse = scaled.tryInverse();
        REQUIRE(scaledInverse.has_value());
        CHECK((scaled * *scaledInverse).approxEqual(Mat3::identity(), 1e-3f));
    }
}

TEST_CASE("Mat3.TryInverseRejectsIllConditionedMatricesAtEveryScale", "[Mat3]")
{
    // Conditioning is a SHAPE question, so a matrix that is nearly rank-deficient must be refused
    // whatever its magnitude — the mirror of the case above, and the reason the threshold applies
    // to a normalised determinant rather than a raw one.
    for (const float scale : {1.0e-6f, 1.0f, 1.0e6f})
    {
        // Column 2 is (column 0 + column 1) to within a part in 1e11, so the NORMALISED determinant
        // is ~1e-11 — an order below the 1e-9 conditioning threshold, whatever `scale` is.
        const Mat3 nearlySingular = Mat3::fromColumns({scale, 0.0f, 0.0f}, {0.0f, scale, 0.0f},
                                                      {scale, scale, scale * 1.0e-11f});
        CHECK_FALSE(nearlySingular.tryInverse().has_value());

        // Calibration, so the threshold is a documented number rather than a mystery: the same
        // shape two decades better conditioned is ACCEPTED at every scale.
        const Mat3 conditioned = Mat3::fromColumns({scale, 0.0f, 0.0f}, {0.0f, scale, 0.0f},
                                                   {scale, scale, scale * 1.0e-7f});
        CHECK(conditioned.tryInverse().has_value());
    }

    // Exactly singular: column 2 = column 0 + column 1.
    const Mat3 singular =
        Mat3::fromColumns({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f});
    CHECK_FALSE(singular.tryInverse().has_value());

    // The zero matrix is the one singular case scaling cannot normalise, and it is refused rather
    // than dividing by its own zero scale.
    CHECK_FALSE(Mat3{}.tryInverse().has_value());
}

TEST_CASE("Mat3.TryInverseAcceptsReflectionsAndKeepsTheirSign", "[Mat3]")
{
    // A reflection is INVERTIBLE — its determinant is negative, not small — so a magnitude test is
    // what the conditioning question needs, and the sign belongs to orientation instead.
    const Mat3 reflection = Mat3::diagonal({1.0f, -1.0f, 1.0f});
    const std::optional<Mat3> inverse = reflection.tryInverse();
    REQUIRE(inverse.has_value());
    CHECK((reflection * *inverse).approxEqual(Mat3::identity(), 1e-6f));
    CHECK(reflection.determinant() < 0.0);

    // And a tiny reflection keeps that sign where a FLOAT determinant cannot. The underflow needs a
    // scale small enough that the CUBE leaves float's range: 1e-5 cubed is -1e-15, which float
    // represents perfectly well, but 1e-16 cubed is -1e-48 and float has nothing below about
    // 1.4e-45. The sign is then lost to -0.0f, which is NOT less than zero and compares `>= 0` as
    // true — so a caller reading the sign concludes the transform preserves winding when it
    // reverses it. VDPM reads exactly this sign to fold a reflection into its cone facing, and the
    // wrong answer culls the side that should be visible.
    const Mat3 tinyReflection = Mat3::diagonal({1.0e-16f, -1.0e-16f, 1.0e-16f});
    REQUIRE(tinyReflection.tryInverse().has_value()); // conditioning is scale-invariant: usable
    CHECK(tinyReflection.determinant() < 0.0);        // in double the sign survives
    const float asFloat = static_cast<float>(tinyReflection.determinant());
    CHECK(asFloat == 0.0f);      // in float the magnitude is gone...
    CHECK_FALSE(asFloat < 0.0f); // ...and with it the orientation VDPM needs
}

TEST_CASE("Mat3.TryInverseRefusesAnInvalidTolerance", "[Mat3]")
{
    // `magnitude > tolerance` is TRUE for a zero determinant against a negative tolerance, so
    // without validation a singular matrix would be accepted and then divided by its own zero
    // determinant. The tolerance is checked before the matrix is even examined.
    const Mat3 singular =
        Mat3::fromColumns({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f});
    CHECK_FALSE(singular.tryInverse(-1.0f).has_value());
    CHECK_FALSE(singular.tryInverse(std::numeric_limits<float>::quiet_NaN()).has_value());
    CHECK_FALSE(singular.tryInverse(std::numeric_limits<float>::infinity()).has_value());

    // A perfectly invertible matrix is refused too: a bad threshold is a caller defect, and the
    // answer to one is never "every matrix is invertible".
    CHECK_FALSE(Mat3::identity().tryInverse(-1.0f).has_value());
    CHECK_FALSE(Mat3::identity().tryInverse(std::numeric_limits<float>::quiet_NaN()).has_value());
}

TEST_CASE("Mat3.TryInverseRefusesAnInverseFloatCannotHold", "[Mat3]")
{
    // WELL CONDITIONED IS NOT REPRESENTABLE. A uniform scale of 1e-39 has a normalised determinant
    // of exactly 1 — it could not be better conditioned — and an inverse of 1e39, which is past
    // float's 3.4e38. Converting anyway yields infinities inside an ENGAGED optional, so the caller
    // believes it holds an inverse and propagates them. The optional means "a representable inverse
    // exists" or it means nothing.
    const Mat3 unrepresentable = Mat3::diagonal({1.0e-39f, 1.0e-39f, 1.0e-39f});
    CHECK_FALSE(unrepresentable.tryInverse().has_value());

    // The neighbouring scale that IS representable still works, so the refusal is about the answer
    // rather than about smallness.
    const Mat3 representable = Mat3::diagonal({1.0e-38f, 1.0e-38f, 1.0e-38f});
    const std::optional<Mat3> inverse = representable.tryInverse();
    REQUIRE(inverse.has_value());
    CHECK((*inverse)[0, 0] > 0.0f);
    CHECK(std::isfinite((*inverse)[0, 0]));

    // A mixed case: one axis fine, one beyond range. Any single unrepresentable element is enough,
    // because the caller gets one matrix and cannot use half of it.
    const Mat3 mixed = Mat3::diagonal({1.0f, 1.0e-39f, 1.0f});
    CHECK_FALSE(mixed.tryInverse().has_value());
}

TEST_CASE("Mat3.TryInverseRefusesNonFiniteMatrices", "[Mat3]")
{
    // An inverse built from a NaN is a matrix of NaNs that every later operation spreads silently.
    // Refusing is what lets a caller notice at the point of failure.
    Mat3 withNaN = Mat3::identity();
    withNaN[1, 1] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(withNaN.tryInverse().has_value());

    Mat3 withInf = Mat3::identity();
    withInf[0, 2] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(withInf.tryInverse().has_value());
}
