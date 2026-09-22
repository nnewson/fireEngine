#include <fire_engine/math/rotation3.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using namespace fire_engine;

namespace
{

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

[[nodiscard]] Rotation3 aboutZ(float angle)
{
    const auto r = Rotation3::tryFromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, angle);
    REQUIRE(r.has_value());
    return *r;
}

} // namespace

TEST_CASE("Rotation3.TheUnitToleranceIsSquaredDeviation", "[Rotation3]")
{
    // The convention is pinned here because the two spellings differ by a factor of two and nothing
    // in a type name says which one a constant means. `kRotationUnitToleranceSquared` is measured
    // on |‖q‖² − 1|, so a quaternion whose LENGTH is off by d registers as roughly 2d.
    const float linearDeviation = 2.0e-5f;
    const Quaternion slightlyLong{0.0f, 0.0f, 0.0f, 1.0f + linearDeviation};
    const float squaredDeviation = std::fabs(slightlyLong.magnitudeSquared() - 1.0f);
    // The RELATION is the claim, not the arithmetic: subtracting 1 from a number just above 1
    // cancels most of a float's significant digits, so this is checked to a few percent rather than
    // to the last bit.
    CHECK(squaredDeviation == Catch::Approx(2.0f * linearDeviation).epsilon(0.05));
    CHECK(Rotation3::isUnit(slightlyLong));

    // And a value outside it is rejected by the same measure, so `isUnit` and the constant cannot
    // drift apart.
    const Quaternion clearlyNotUnit{0.0f, 0.0f, 0.0f, 1.01f};
    CHECK(std::fabs(clearlyNotUnit.magnitudeSquared() - 1.0f) > kRotationUnitToleranceSquared);
    CHECK_FALSE(Rotation3::isUnit(clearlyNotUnit));

    // A non-finite quaternion is not unit, whatever the arithmetic would say.
    CHECK_FALSE(Rotation3::isUnit(Quaternion{kNaN, 0.0f, 0.0f, 1.0f}));
    CHECK_FALSE(Rotation3::isUnit(Quaternion{kInf, 0.0f, 0.0f, 1.0f}));
}

TEST_CASE("Rotation3.FactoriesNormaliseImpreciseInputAndRefuseDegenerateInput", "[Rotation3]")
{
    // IMPRECISE IS NOT INVALID. A keyframe authored to six decimals, or an orientation that has
    // drifted a few ulps, describes a rotation perfectly well — it is normalised, not rejected.
    const auto drifted = Rotation3::tryFromQuaternion(Quaternion{0.0f, 0.0f, 0.0f, 1.0f + 1.0e-4f});
    REQUIRE(drifted.has_value());
    CHECK(Rotation3::isUnit(drifted->quaternion()));

    // THE WORST CASE glTF PERMITS, pinned exactly. Rotation animation outputs may be normalised
    // signed BYTES, decoded as round(c·127)/127 — the coarsest encoding in the format. The worst
    // deviation is {0.5, 0.5, 0.5, 0.5}, which encodes as 64/127 per component and decodes to a
    // squared norm of 1.015810. An admission tolerance of 1e-2 would refuse this: a conforming
    // asset rejected by the loader, which is why the bound is derived from the encoding rather than
    // from the engine's own drift.
    constexpr float kByteQuantised = 64.0f / 127.0f;
    const Quaternion byteEncoded{kByteQuantised, kByteQuantised, kByteQuantised, kByteQuantised};
    CHECK(std::fabs(byteEncoded.magnitudeSquared() - 1.0f) ==
          Catch::Approx(0.015810).epsilon(1e-3));
    const auto authored = Rotation3::tryFromQuaternion(byteEncoded);
    REQUIRE(authored.has_value());
    CHECK(Rotation3::isUnit(authored->quaternion()));

    // Shorts are finer and therefore also accepted.
    const float shortQuantised = std::round(0.5f * 32767.0f) / 32767.0f;
    const auto fromShorts = Rotation3::tryFromQuaternion(
        Quaternion{shortQuantised, shortQuantised, shortQuantised, shortQuantised});
    REQUIRE(fromShorts.has_value());

    // A SCALED quaternion is refused. {0, 0, 10, 10} normalises to a perfectly good rotation, which
    // is exactly why accepting it would be wrong: this factory could then not tell a rotation from
    // a derivative or an unnormalised intermediate, and the constant promising an admission
    // tolerance would be decorative.
    CHECK_FALSE(Rotation3::tryFromQuaternion(Quaternion{0.0f, 0.0f, 10.0f, 10.0f}).has_value());
    CHECK_FALSE(Rotation3::tryFromQuaternion(Quaternion{0.0f, 0.0f, 0.0f, 0.5f}).has_value());

    // DEGENERATE AND NON-FINITE ARE REFUSED, because the alternative is laundering a producer bug
    // into the identity — a value that rotates nothing and looks deliberate.
    CHECK_FALSE(Rotation3::tryFromQuaternion(Quaternion{0.0f, 0.0f, 0.0f, 0.0f}).has_value());
    CHECK_FALSE(Rotation3::tryFromQuaternion(Quaternion{kNaN, 0.0f, 0.0f, 1.0f}).has_value());
    CHECK_FALSE(Rotation3::tryFromQuaternion(Quaternion{0.0f, kInf, 0.0f, 1.0f}).has_value());
}

TEST_CASE("Rotation3.AxisAngleNormalisesTheAxisAndRefusesDegenerateGeometry", "[Rotation3]")
{
    // A non-unit axis means the DIRECTION, so these must agree exactly in angle — the failure
    // condition is geometry with no direction, never "the caller did not pre-normalise".
    const auto unitAxis = Rotation3::tryFromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, 0.75f);
    const auto longAxis = Rotation3::tryFromAxisAngle(Vec3{0.0f, 0.0f, 7.0f}, 0.75f);
    REQUIRE(unitAxis.has_value());
    REQUIRE(longAxis.has_value());
    CHECK(unitAxis->approxEqual(*longAxis, 1.0e-6f));
    CHECK(Rotation3::isUnit(longAxis->quaternion()));

    // `fromAxisAngle({0,0,0}, angle)` was the review's example of a value that is not a rotation.
    CHECK_FALSE(Rotation3::tryFromAxisAngle(Vec3{}, 1.0f).has_value());
    CHECK_FALSE(Rotation3::tryFromAxisAngle(Vec3{1.0e-30f, 0.0f, 0.0f}, 1.0f).has_value());
    CHECK_FALSE(Rotation3::tryFromAxisAngle(Vec3{kNaN, 0.0f, 1.0f}, 1.0f).has_value());
    CHECK_FALSE(Rotation3::tryFromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, kNaN).has_value());
    CHECK_FALSE(Rotation3::tryFromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, kInf).has_value());
}

TEST_CASE("Rotation3.FromVectorsNormalisesAndHandlesAntiparallel", "[Rotation3]")
{
    const auto r = Rotation3::tryFromVectors(Vec3{3.0f, 0.0f, 0.0f}, Vec3{0.0f, 5.0f, 0.0f});
    REQUIRE(r.has_value());
    CHECK(r->rotate(Vec3{1.0f, 0.0f, 0.0f}).approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 1.0e-5f));

    // ANTIPARALLEL IS NOT DEGENERATE: it is a 180° rotation about some perpendicular axis, and the
    // result must take `from` to `to` like any other.
    const auto flipped = Rotation3::tryFromVectors(Vec3{1.0f, 0.0f, 0.0f}, Vec3{-1.0f, 0.0f, 0.0f});
    REQUIRE(flipped.has_value());
    CHECK(Rotation3::isUnit(flipped->quaternion()));
    CHECK(flipped->rotate(Vec3{1.0f, 0.0f, 0.0f}).approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1.0e-5f));

    CHECK_FALSE(Rotation3::tryFromVectors(Vec3{}, Vec3{0.0f, 1.0f, 0.0f}).has_value());
    CHECK_FALSE(Rotation3::tryFromVectors(Vec3{1.0f, 0.0f, 0.0f}, Vec3{}).has_value());
    CHECK_FALSE(
        Rotation3::tryFromVectors(Vec3{kInf, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}).has_value());
}

TEST_CASE("Rotation3.EveryOperationPreservesTheInvariant", "[Rotation3]")
{
    // The property the type exists for, and the one the reconnaissance did NOT prove: the engine's
    // bounded drift today comes from `integrate` and `slerp` normalising their own results, so a
    // rotation type must carry that responsibility rather than assume it. Composition especially —
    // a product of two unit quaternions is unit only in exact arithmetic, and a skeleton compounds
    // the error link by link.
    Rotation3 chained = Rotation3::identity();
    const Rotation3 step = aboutZ(0.017f);
    for (int i = 0; i < 4096; ++i)
    {
        chained = chained * step;
        REQUIRE(Rotation3::isUnit(chained.quaternion()));
    }

    Rotation3 integrated = Rotation3::identity();
    for (int i = 0; i < 4096; ++i)
    {
        integrated = integrated.integrate(Vec3{0.3f, -1.1f, 0.7f}, 1.0f / 120.0f);
        REQUIRE(Rotation3::isUnit(integrated.quaternion()));
    }

    const Rotation3 a = aboutZ(0.2f);
    const Rotation3 b = aboutZ(2.7f);
    for (const float t : {0.0f, 0.001f, 0.25f, 0.5f, 0.999f, 1.0f})
    {
        CHECK(Rotation3::isUnit(Rotation3::slerp(a, b, t).quaternion()));
    }
    CHECK(Rotation3::isUnit(a.inverse().quaternion()));
    CHECK(Rotation3::isUnit(a.alignedTo(b).quaternion()));
    CHECK(Rotation3::isUnit(Rotation3::identity().quaternion()));
}

TEST_CASE("Rotation3.EqualityIsAboutRotationsNotRepresentations", "[Rotation3]")
{
    // `q` and `-q` rotate every vector identically, so for a ROTATION type they are equal. A type
    // whose equality answered about its storage would make every caller learn about the double
    // cover, which is the knowledge this type exists to absorb.
    const Rotation3 r = aboutZ(1.3f);
    const Quaternion negated{-r.quaternion().x(), -r.quaternion().y(), -r.quaternion().z(),
                             -r.quaternion().w()};
    const auto mirrored = Rotation3::tryFromQuaternion(negated);
    REQUIRE(mirrored.has_value());

    CHECK(r == *mirrored);
    CHECK(r.angleTo(*mirrored) == Catch::Approx(0.0f).margin(1.0e-6));
    CHECK(r.rotate(Vec3{1.0f, 2.0f, 3.0f})
              .approxEqual(mirrored->rotate(Vec3{1.0f, 2.0f, 3.0f}), 1.0e-5f));

    // The representation still differs, and `sameComponents` is how a caller says it means that —
    // a serialiser checking round-trip fidelity, or a test pinning which hemisphere was chosen.
    CHECK_FALSE(r.sameComponents(*mirrored));
    CHECK(r.sameComponents(r));

    // `alignedTo` is the explicit hemisphere choice. There is deliberately no unary `operator-`:
    // negation does not produce a different rotation, so it would read as an inverse and silently
    // be a no-op.
    CHECK(mirrored->alignedTo(r).sameComponents(r));
    CHECK(r.alignedTo(r).sameComponents(r));
}

TEST_CASE("Rotation3.ApproximateEqualityIsAnAngle", "[Rotation3]")
{
    // Four independent component tolerances answer a question nobody asks: two rotations can differ
    // in every component and be a thousandth of a degree apart, or agree in three and be far apart.
    const Rotation3 a = aboutZ(1.0f);
    const Rotation3 b = aboutZ(1.0f + 1.0e-5f);
    CHECK(a.approxEqual(b, 1.0e-4f));
    CHECK_FALSE(a.approxEqual(aboutZ(1.1f), 1.0e-4f));

    CHECK(a.angleTo(a) == Catch::Approx(0.0f).margin(1.0e-6));
    CHECK(aboutZ(0.0f).angleTo(aboutZ(pi * 0.5f)) == Catch::Approx(pi * 0.5f).epsilon(1e-4));
    // Symmetric, and bounded by π: the distance between rotations, never between representations.
    CHECK(aboutZ(0.3f).angleTo(aboutZ(2.9f)) ==
          Catch::Approx(aboutZ(2.9f).angleTo(aboutZ(0.3f))).epsilon(1e-5));
    CHECK(aboutZ(0.0f).angleTo(aboutZ(pi * 1.999f)) <= pi + 1.0e-4f);
}

TEST_CASE("Rotation3.IdentityAndCompositionBehave", "[Rotation3]")
{
    const Rotation3 r = aboutZ(0.9f);

    // `operator==` is EXACT up to the double cover — it answers "the same rotation", not "near
    // enough". Composing with the identity is exact (multiplying by {0,0,0,1} perturbs nothing), so
    // these hold bit-for-bit...
    CHECK((r * Rotation3::identity()) == r);
    CHECK((Rotation3::identity() * r) == r);

    // ...while a rotation composed with its own inverse is identity only in exact arithmetic, and
    // computed rotations are what `approxEqual` is for. A caller comparing two independently
    // computed orientations wants the angle, not the bits; `==` is for the cases where one value
    // provably came from the other.
    CHECK(r.inverse().inverse() == r); // conjugation twice is exact
    CHECK((r * r.inverse()).approxEqual(Rotation3::identity(), 1.0e-6f));
    CHECK((r.inverse() * r).approxEqual(Rotation3::identity(), 1.0e-6f));

    // Composition applies the right-hand rotation first, matching the quaternion convention it is
    // built on: rotating a vector by (a * b) equals rotating it by b and then by a.
    const Rotation3 a = aboutZ(0.4f);
    const Rotation3 b = *Rotation3::tryFromAxisAngle(Vec3{1.0f, 0.0f, 0.0f}, 0.6f);
    const Vec3 v{0.3f, -0.7f, 1.1f};
    CHECK((a * b).rotate(v).approxEqual(a.rotate(b.rotate(v)), 1.0e-5f));
}

TEST_CASE("Rotation3.AngleResolvesDifferencesFinerThanItsOwnTolerance", "[Rotation3]")
{
    // `2·acos(|dot|)` cannot answer this. For a small angle θ the dot product is cos(θ/2), which
    // rounds to exactly 1.0f below about θ = 5e-4 — so every difference finer than that reports as
    // ZERO, and this type's own default comparison tolerance (1e-4 rad) sits inside the blind spot.
    // Reading the angle off the relative rotation's vector part has no such floor.
    const Rotation3 base = aboutZ(0.7f);
    const Rotation3 nudged = aboutZ(0.7f + 1.0e-5f);

    CHECK(base.angleTo(nudged) > 0.0f);
    CHECK(base.angleTo(nudged) == Catch::Approx(1.0e-5f).epsilon(0.05));
    // And the comparison built on it can tell them apart at a tolerance below the difference.
    CHECK_FALSE(base.approxEqual(nudged, 1.0e-6f));
    CHECK(base.approxEqual(nudged, 1.0e-4f));

    // Finer still: a tenth of the old resolution floor, and two decades below it.
    for (const float delta : {1.0e-4f, 1.0e-5f, 1.0e-6f})
    {
        const Rotation3 other = aboutZ(0.7f + delta);
        CHECK(base.angleTo(other) == Catch::Approx(delta).epsilon(0.1));
    }

    // The degenerate direction still answers exactly zero rather than a small noise floor.
    CHECK(base.angleTo(base) == Catch::Approx(0.0f).margin(1.0e-9));
}

TEST_CASE("Rotation3.ApproxEqualRefusesAnInvalidTolerance", "[Rotation3]")
{
    // Consistent with `almostEqual`: a negative, NaN or infinite tolerance is a caller defect, and
    // the answer to one is false — never "everything matches", which is what an infinite tolerance
    // would otherwise mean.
    const Rotation3 a = aboutZ(0.2f);
    const Rotation3 b = aboutZ(2.0f);
    CHECK_FALSE(a.approxEqual(a, -1.0f));
    CHECK_FALSE(a.approxEqual(a, kNaN));
    CHECK_FALSE(a.approxEqual(b, kInf));
    CHECK_FALSE(a.approxEqual(a, kInf));
    CHECK(a.approxEqual(a, 0.0f)); // zero is a legitimate, if strict, tolerance
}

TEST_CASE("Rotation3.OperationsNormaliseExactlyOnce", "[Rotation3]")
{
    // Not a performance point — an accuracy one. Normalising twice rounds twice, and phase 1's norm
    // work showed what an extra ulp per component costs a solver (a settling box stack went from
    // step 169 to 425). It also muddies attribution: if the goldens move during this migration, the
    // cause should be the conversion authority changing, not an operation quietly rounding twice.
    //
    // EACH CASE PROVES ITS OWN SENSITIVITY FIRST. Normalising an already-unit value is usually
    // idempotent to the bit, so a fixture chosen at random passes whether the implementation rounds
    // once or twice — the comparison would assert nothing at all. The `REQUIRE` in each section
    // establishes that its raw value DOES change under a second normalisation, and only then is the
    // operation's output compared against a single one. The constants were found by search; a
    // search over 200,000 quaternions found about 37% to be sensitive, so they are not rare, but
    // they do have to be chosen deliberately.
    const auto sensitive = [](const Quaternion& raw)
    {
        const Quaternion once = Quaternion::normalise(raw);
        return !(once == Quaternion::normalise(once));
    };

    SECTION("composition")
    {
        const auto a = Rotation3::tryFromAxisAngle(Vec3{0.3f, -0.8f, 0.5f}, 0.0091f);
        const auto b = Rotation3::tryFromAxisAngle(Vec3{1.0f, 0.2f, -0.4f}, 1.1f);
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        const Quaternion raw = a->quaternion() * b->quaternion();
        REQUIRE(sensitive(raw)); // this fixture can tell one normalisation from two
        CHECK((*a * *b).quaternion() == Quaternion::normalise(raw));
    }

    SECTION("integration")
    {
        // The path that used to round twice most clearly: `Quaternion::integrate` normalises its
        // own result, and passing that to the choke point normalised it again.
        const auto r = Rotation3::tryFromAxisAngle(Vec3{0.2f, 0.9f, -0.3f}, 0.61f);
        REQUIRE(r.has_value());
        const Vec3 omega{0.83f, -1.37f, 0.44f};
        const float dt = 0.0002329f;
        const Vec3 rotationVector = omega * dt;
        const float angle = rotationVector.magnitude();
        const float half = angle * 0.5f;
        const float scale = std::sin(half) / angle;
        const Quaternion delta{rotationVector.x() * scale, rotationVector.y() * scale,
                               rotationVector.z() * scale, std::cos(half)};
        const Quaternion raw = delta * r->quaternion();
        REQUIRE(sensitive(raw));
        CHECK(r->integrate(omega, dt).quaternion() == Quaternion::normalise(raw));
    }

    SECTION("slerp, general branch")
    {
        const auto a = Rotation3::tryFromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, 0.2f);
        const auto b = Rotation3::tryFromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, 2.7f);
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        const float t = 0.015f;
        const Quaternion qa = a->quaternion();
        const Quaternion qb = b->quaternion();
        const float dot = Quaternion::dotProduct(qa, qb);
        REQUIRE(dot <= 0.9995f); // the general branch, not the linear fallback
        const float theta = std::acos(dot);
        const float sinTheta = std::sin(theta);
        const float wa = std::sin((1.0f - t) * theta) / sinTheta;
        const float wb = std::sin(t * theta) / sinTheta;
        const Quaternion raw{qa.x() * wa + qb.x() * wb, qa.y() * wa + qb.y() * wb,
                             qa.z() * wa + qb.z() * wb, qa.w() * wa + qb.w() * wb};
        REQUIRE(sensitive(raw));
        CHECK(Rotation3::slerp(*a, *b, t).quaternion() == Quaternion::normalise(raw));
    }

    SECTION("slerp, linear branch — not provable by observation")
    {
        // AN HONEST LIMIT, stated rather than papered over. The linear fallback blends two
        // rotations that are already nearly equal, so its result is unit to the bit and a second
        // normalisation changes nothing: a search over 60,000 fixtures found no input where once
        // and twice differ. The single-normalisation property holds here by construction and is NOT
        // asserted — a comparison that cannot fail is worse than no comparison, because it reads as
        // coverage.
        //
        // What can be checked is that this branch is the one taken, and that it produces a unit
        // rotation genuinely between its inputs.
        const auto a = Rotation3::tryFromAxisAngle(Vec3{0.4f, -0.2f, 0.9f}, 0.33f);
        const auto b = Rotation3::tryFromAxisAngle(Vec3{0.4f, -0.2f, 0.9f}, 0.3305f);
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        REQUIRE(Quaternion::dotProduct(a->quaternion(), b->quaternion()) > 0.9995f);
        const Rotation3 blended = Rotation3::slerp(*a, *b, 0.5f);
        CHECK(Rotation3::isUnit(blended.quaternion()));
        CHECK(blended.angleTo(*a) == Catch::Approx(blended.angleTo(*b)).epsilon(0.05));
    }

    SECTION("factories")
    {
        const Quaternion drifted{0.0f, 0.0f, 0.0f, 1.0f + 1.0e-4f};
        const auto admitted = Rotation3::tryFromQuaternion(drifted);
        REQUIRE(admitted.has_value());
        CHECK(admitted->quaternion() == Quaternion::normalise(drifted));

        const Vec3 axis{0.3f, -0.8f, 0.5f};
        const float angle = 1.234f;
        const auto viaFactory = Rotation3::tryFromAxisAngle(axis, angle);
        REQUIRE(viaFactory.has_value());
        const Vec3 unitAxis = Vec3::normalise(axis);
        const float half = angle * 0.5f;
        const float s = std::sin(half);
        const Quaternion raw{unitAxis.x() * s, unitAxis.y() * s, unitAxis.z() * s, std::cos(half)};
        CHECK(viaFactory->quaternion() == Quaternion::normalise(raw));

        // From-vectors builds the half-angle form and hands it over unnormalised, so the result is
        // a single normalisation of that — not a normalised `fromVectors` normalised again.
        const Vec3 from{2.0f, 0.0f, 0.0f};
        const Vec3 to{0.0f, 3.0f, 0.0f};
        const auto fromVectors = Rotation3::tryFromVectors(from, to);
        REQUIRE(fromVectors.has_value());
        const Vec3 f = Vec3::normalise(from);
        const Vec3 t = Vec3::normalise(to);
        const Vec3 cross = Vec3::crossProduct(f, t);
        const Quaternion halfAngle{cross.x(), cross.y(), cross.z(), 1.0f + Vec3::dotProduct(f, t)};
        CHECK(fromVectors->quaternion() == Quaternion::normalise(halfAngle));
    }
}
