#include <fire_engine/math/scalar.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

using namespace fire_engine;

namespace
{

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kMax = std::numeric_limits<float>::max();

} // namespace

TEST_CASE("a NaN is not approximately equal to anything, itself included", "[Scalar]")
{
    // THE defect this authority exists for (tier-0 finding 2). The previous form asked
    // `diff > eps || diff < -eps`, and both comparisons are false for a NaN difference, so every
    // approximate comparison in the engine answered "equal" for a NaN — including a test asserting
    // that a transform had stayed finite.
    CHECK_FALSE(almostEqual(kNaN, kNaN));
    CHECK_FALSE(almostEqual(kNaN, 0.0f));
    CHECK_FALSE(almostEqual(0.0f, kNaN));
    CHECK_FALSE(almostEqual(kNaN, kInf));
    // And no tolerance, however wide, may rescue it: NaN is not a value that is nearly something.
    CHECK_FALSE(almostEqual(kNaN, 1.0f, 1.0e30f, 1.0e30f));
}

TEST_CASE("equal infinities are equal and opposite ones are not", "[Scalar]")
{
    // Both fell out of the old subtraction identically (inf - inf is NaN, and NaN passed), so the
    // right answer and the wrong one were indistinguishable. Here they are decided before any
    // arithmetic happens: `a == b` is the first question.
    CHECK(almostEqual(kInf, kInf));
    CHECK(almostEqual(-kInf, -kInf));
    CHECK_FALSE(almostEqual(kInf, -kInf));
    // An infinity is not NEARLY a finite value, however large that value is.
    CHECK_FALSE(almostEqual(kInf, kMax));
    CHECK_FALSE(almostEqual(-kInf, -kMax));
}

TEST_CASE("the difference is computed where it cannot overflow", "[Scalar]")
{
    // `kMax - (-kMax)` is +inf in float. Computed that way the comparison would be reasoning about
    // a number neither caller passed; in double the subtraction of two floats is exact, so these
    // answer about the values themselves.
    CHECK_FALSE(almostEqual(kMax, -kMax));
    CHECK_FALSE(almostEqual(-kMax, kMax));
    // Two values a single ULP apart near FLT_MAX: far beyond any absolute tolerance, but well
    // inside the relative one — which is the case the relative term exists for.
    const float nextDown = std::nextafter(kMax, 0.0f);
    CHECK(almostEqual(kMax, nextDown));
    CHECK_FALSE(almostEqual(kMax, nextDown, float_epsilon, 0.0f));
}

TEST_CASE("the tolerances are absolute and relative, in that order", "[Scalar]")
{
    // The FIRST argument keeps its historical meaning. A caller that wrote `approxEqual(rhs, 0.1f)`
    // meant "within 0.1" and still does; reinterpreting it as 10% would have silently loosened
    // every existing call site by orders of magnitude at large values.
    CHECK(almostEqual(1.0f, 1.05f, 0.1f, 0.0f));
    CHECK_FALSE(almostEqual(1.0f, 1.5f, 0.1f, 0.0f));

    // Near zero the relative term is meaningless (any two small values are relatively far apart),
    // which is exactly where the absolute term carries the comparison.
    CHECK(almostEqual(0.0f, 1.0e-9f));
    CHECK_FALSE(almostEqual(0.0f, 1.0e-9f, 0.0f, float_relative_epsilon));

    // And far from zero the absolute term is meaningless, which is where the relative one does.
    CHECK(almostEqual(1.0e6f, 1.0e6f + 0.5f));
    CHECK_FALSE(almostEqual(1.0e6f, 1.0e6f + 0.5f, float_epsilon, 0.0f));
}

TEST_CASE("+0.0 and -0.0 are the same number", "[Scalar]")
{
    // They compare equal here because they ARE equal numerically; a caller wanting to tell the two
    // bit patterns apart wants something this function has never claimed to be.
    CHECK(almostEqual(0.0f, -0.0f, 0.0f, 0.0f));
}

TEST_CASE("every vector, matrix and quaternion comparison inherits the NaN rule", "[Scalar]")
{
    // The authority is only worth having if nothing bypasses it. One NaN component per type, each
    // of which used to compare equal to itself.
    const Vec3 nanVec{kNaN, 0.0f, 0.0f};
    CHECK_FALSE(nanVec.approxEqual(nanVec));
    CHECK_FALSE(nanVec.approxEqual(Vec3{0.0f, 0.0f, 0.0f}));

    Mat3 nanMat3 = Mat3::identity();
    nanMat3[0, 0] = kNaN;
    CHECK_FALSE(nanMat3.approxEqual(nanMat3));

    Mat4 nanMat4 = Mat4::identity();
    nanMat4[2, 3] = kNaN;
    CHECK_FALSE(nanMat4.approxEqual(nanMat4));

    const Quaternion nanQuat{kNaN, 0.0f, 0.0f, 1.0f};
    CHECK_FALSE(nanQuat.approxEqual(nanQuat));

    // Finite values still behave, so the fix is not simply "everything is unequal now".
    CHECK(Vec3{1.0f, 2.0f, 3.0f}.approxEqual(Vec3{1.0f, 2.0f, 3.0f}));
    CHECK(Mat4::identity().approxEqual(Mat4::identity()));
}

TEST_CASE("an explicit tolerance is the whole answer", "[Scalar]")
{
    // The overload set exists for this. A caller who writes a tolerance means it: the two-argument
    // form carries both defaults, the three-argument form is ABSOLUTE ONLY, and neither can be
    // quietly loosened by the other's default.
    CHECK(almostEqual(1.0f, 1.0f + 1.0e-7f));                // default relative admits it
    CHECK_FALSE(almostEqual(1.0f, 1.0f + 1.0e-7f, 1.0e-9f)); // an explicit 1e-9 does not
    CHECK(almostEqual(1.0f, 1.0f + 1.0e-7f, 1.0e-9f, 1.0e-6f));

    // At large magnitudes the default form still works where an absolute tolerance cannot.
    const float large = 1.0e7f;
    CHECK(almostEqual(large, large + 1.0f));
    CHECK_FALSE(almostEqual(large, large + 1.0f, float_epsilon));
}

TEST_CASE("an invalid tolerance is refused, not reinterpreted", "[Scalar]")
{
    // A negative, NaN or infinite tolerance is a caller defect — a bad constant, an uninitialised
    // field, a division that went wrong. Answering "false" makes it visible at the first
    // comparison; treating a negative as an unsatisfiable term would silently convert a broken
    // configuration into a stricter policy and let the run continue looking healthy.
    CHECK_FALSE(almostEqual(1.0f, 1.0f, -1.0f));
    CHECK_FALSE(almostEqual(1.0f, 1.0f, 1.0e-6f, -1.0f));
    CHECK_FALSE(almostEqual(1.0f, 1.0f, kNaN));
    CHECK_FALSE(almostEqual(1.0f, 1.0f, 1.0e-6f, kNaN));
    CHECK_FALSE(almostEqual(1.0f, 1.0f, kInf));
    CHECK_FALSE(almostEqual(1.0f, 1.0f, 1.0e-6f, kInf));

    // EQUAL OPERANDS TOO, which is why the validation runs before the `a == b` shortcut: the one
    // case most likely to be exercised by a smoke test is the one that would hide the defect.
    CHECK_FALSE(almostEqual(2.5f, 2.5f, -0.0001f));
    CHECK_FALSE(almostEqual(kInf, kInf, -1.0f));

    // And the aggregates inherit the refusal rather than validating separately.
    CHECK_FALSE(Vec3{1.0f, 2.0f, 3.0f}.approxEqual(Vec3{1.0f, 2.0f, 3.0f}, -1.0f));
    CHECK_FALSE(Mat4::identity().approxEqual(Mat4::identity(), kNaN));
}
