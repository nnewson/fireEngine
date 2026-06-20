#include <fire_engine/math/mat3.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
