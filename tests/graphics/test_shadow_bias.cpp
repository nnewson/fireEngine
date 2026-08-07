#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <numbers>

#include <fire_engine/graphics/shadow_bias.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// SH-07: the shadow bias law and its per-projection metrics.
//
// The defect being closed: the receiver scaled bias and normal offset by `exp2(cascade)`, which
// asserts each cascade's world-per-texel doubles exactly. It does not — practical splits and a
// bounding-sphere fit set the footprint, and since SH-06 the depth SPAN is fitted per cascade too,
// so one constant was standing in for two independent scales. These cases pin the replacement in
// the terms that make it checkable: each metric in its own named unit, and the law monotonic in
// each.
//
// They are also the arbiter for `shaders/shadow_bias.glsl`, which runs the same law per fragment —
// the values below are the golden numbers if the two ever disagree.
// ---------------------------------------------------------------------------

namespace
{

constexpr ShadowBiasPolicy kPolicy{.slopeScale = 1.0f,
                                   .constantTexels = 1.0f,
                                   .normalOffsetTexels = 1.0f,
                                   .maxSlopeTangent = 8.0f};

// A cascade one world unit per texel across, spanning 100 world units of depth.
constexpr ShadowBiasInputs kFaceOnCascade{.worldUnitsPerTexel = 1.0f,
                                          .normalizedDepthPerWorldUnit = 1.0f / 100.0f,
                                          .nDotL = 1.0f,
                                          .filterRadiusTexels = 0.0f};

[[nodiscard]] float cosineOf(float degrees)
{
    return std::cos(degrees * std::numbers::pi_v<float> / 180.0f);
}

} // namespace

TEST_CASE("ShadowBias.AFaceOnReceiverPaysOnlyTheConstantTerm", "[ShadowBias]")
{
    // Head-on, tan(theta) is 0, so the whole bias is the floor: one texel of world footprint
    // converted into normalised depth. 1 world unit / 100 world units of span = 0.01.
    const ShadowBias bias = shadowBias(kFaceOnCascade, kPolicy);

    CHECK(bias.receiverDepthBias == Catch::Approx(0.01f).margin(1e-6f));
    // No slope means no in-texel depth range to escape, so nothing to offset along the normal.
    CHECK(bias.normalOffsetWorld == Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("ShadowBias.BiasTracksTheTexelFootprintNotTheCascadeIndex", "[ShadowBias]")
{
    // THE regression case. `exp2(cascade)` asserted a doubling per cascade; what actually matters
    // is this view's own footprint. A cascade with 3.7x the footprint gets 3.7x the world-space
    // slop — whatever its index, and whether or not its neighbour doubled.
    ShadowBiasInputs wide = kFaceOnCascade;
    wide.worldUnitsPerTexel = 3.7f;

    const ShadowBias narrow = shadowBias(kFaceOnCascade, kPolicy);
    const ShadowBias broad = shadowBias(wide, kPolicy);

    CHECK(broad.receiverDepthBias == Catch::Approx(narrow.receiverDepthBias * 3.7f).margin(1e-6f));
}

TEST_CASE("ShadowBias.TheDepthSpanConvertsSeparatelyFromTheFootprint", "[ShadowBias]")
{
    // The second scale `exp2(cascade)` was also standing in for. Same world footprint, a depth
    // range half as deep: the SAME world-space slop is a bigger fraction of the stored range, so
    // the normalised bias doubles — while the world-space normal offset does not move at all,
    // because it never enters depth space.
    ShadowBiasInputs shallow = kFaceOnCascade;
    shallow.nDotL = cosineOf(45.0f);
    ShadowBiasInputs deep = shallow;
    shallow.normalizedDepthPerWorldUnit = 1.0f / 50.0f;

    const ShadowBias shallowBias = shadowBias(shallow, kPolicy);
    const ShadowBias deepBias = shadowBias(deep, kPolicy);

    CHECK(shallowBias.receiverDepthBias ==
          Catch::Approx(deepBias.receiverDepthBias * 2.0f).margin(1e-6f));
    CHECK(shallowBias.normalOffsetWorld == Catch::Approx(deepBias.normalOffsetWorld).margin(1e-6f));
}

TEST_CASE("ShadowBias.TheSlopeTermIsBoundedSoShadowsCannotDetach", "[ShadowBias]")
{
    // tan(theta) diverges at grazing incidence. Unbounded, the bias would push the comparison depth
    // past the caster and the shadow would visibly float away from it — a worse artefact than the
    // acne the term prevents, and one that reads as a missing feature rather than a tuning error.
    ShadowBiasInputs grazing = kFaceOnCascade;
    grazing.nDotL = 1.0e-4f;
    ShadowBiasInputs perpendicular = kFaceOnCascade;
    perpendicular.nDotL = 0.0f;

    const ShadowBias grazingBias = shadowBias(grazing, kPolicy);
    const ShadowBias perpendicularBias = shadowBias(perpendicular, kPolicy);

    // Both saturate at the same ceiling: (maxSlopeTangent + constantTexels) texels, converted.
    const float ceiling = (kPolicy.maxSlopeTangent + kPolicy.constantTexels) * 0.01f;
    CHECK(grazingBias.receiverDepthBias == Catch::Approx(ceiling).margin(1e-6f));
    CHECK(perpendicularBias.receiverDepthBias == Catch::Approx(ceiling).margin(1e-6f));
}

TEST_CASE("ShadowBias.TheFilterRadiusWidensWhatMustBeCleared", "[ShadowBias]")
{
    // A PCF kernel reads a disc, so the slope term must clear the whole disc — otherwise the taps
    // at the edge of the kernel self-shadow while the centre tap does not, which is acne that
    // appears only once filtering is enabled and looks like a filtering bug.
    ShadowBiasInputs sloped = kFaceOnCascade;
    sloped.nDotL = cosineOf(45.0f); // tan = 1
    ShadowBiasInputs filtered = sloped;
    filtered.filterRadiusTexels = 3.0f;

    const ShadowBias single = shadowBias(sloped, kPolicy);
    const ShadowBias wide = shadowBias(filtered, kPolicy);

    // Slope term scales with (radius + 1) texels; the constant term does not.
    CHECK(single.receiverDepthBias == Catch::Approx((1.0f + 1.0f) * 0.01f).margin(1e-6f));
    CHECK(wide.receiverDepthBias == Catch::Approx((4.0f * 1.0f + 1.0f) * 0.01f).margin(1e-6f));
    CHECK(wide.normalOffsetWorld > single.normalOffsetWorld);
}

TEST_CASE("ShadowBias.DegenerateMetricsYieldNoBiasRatherThanInfinity", "[ShadowBias]")
{
    // Reachable: a zero-extent map, a collapsed depth span, a fragment at the light. Zero produces
    // acne — visible and locatable. An infinity would detach every shadow in the frame, which reads
    // as "shadows are broken" and hides which input was bad.
    for (const float bad : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity()})
    {
        ShadowBiasInputs badFootprint = kFaceOnCascade;
        badFootprint.worldUnitsPerTexel = bad;
        CHECK(shadowBias(badFootprint, kPolicy).receiverDepthBias == Catch::Approx(0.0f));

        ShadowBiasInputs badDepth = kFaceOnCascade;
        badDepth.normalizedDepthPerWorldUnit = bad;
        CHECK(shadowBias(badDepth, kPolicy).receiverDepthBias == Catch::Approx(0.0f));
    }
}

TEST_CASE("ShadowBias.OrthographicDepthConversionIsOneOverTheFittedSpan", "[ShadowBias]")
{
    CHECK(orthographicNormalizedDepthPerWorldUnit(80.0f) == Catch::Approx(1.0f / 80.0f));
    // A collapsed or absent span converts nothing rather than dividing by zero.
    CHECK(orthographicNormalizedDepthPerWorldUnit(0.0f) == Catch::Approx(0.0f));
    CHECK(orthographicNormalizedDepthPerWorldUnit(-5.0f) == Catch::Approx(0.0f));
}

TEST_CASE("ShadowBias.SpotMetricsFollowTheReceiversForwardDepth", "[ShadowBias]")
{
    // A 90-degree spot into a 1024-texel map: 2 * tan(45) / 1024 world units per texel per unit of
    // forward depth. Footprint grows linearly with depth; precision falls as 1/depth^2. Both are
    // why one constant cannot serve a cone.
    const float texelAngleScale = 2.0f * 1.0f / 1024.0f;
    CHECK(perspectiveWorldUnitsPerTexel(texelAngleScale, 10.0f) ==
          Catch::Approx(10.0f * texelAngleScale));
    CHECK(perspectiveWorldUnitsPerTexel(texelAngleScale, 20.0f) ==
          Catch::Approx(2.0f * perspectiveWorldUnitsPerTexel(texelAngleScale, 10.0f)));

    const float nearP = 0.1f;
    const float farP = 50.0f;
    // ON AXIS: radial == forward, so the ray cosine is 1 and only the perspective term applies.
    const float atTen = perspectiveNormalizedDepthPerWorldUnit(nearP, farP, 10.0f, 10.0f);
    const float atTwenty = perspectiveNormalizedDepthPerWorldUnit(nearP, farP, 20.0f, 20.0f);
    CHECK(atTen == Catch::Approx((nearP * farP) / ((farP - nearP) * 100.0f)));
    CHECK(atTwenty == Catch::Approx(atTen / 4.0f).margin(1e-9f)); // inverse square
}

TEST_CASE("ShadowBias.SpotConversionAccountsForTheRayForwardCosine", "[ShadowBias]")
{
    // The correction the simplified model missed. The law's slope term is measured along the LOCAL
    // LIGHT RAY (nDotL is against that ray), while the stored depth is a projection onto the spot's
    // forward axis — so one world unit along the ray advances forward depth by only
    // cos(ray, forward). Off-axis, ignoring that over-converts the bias.
    const float nearP = 0.1f;
    const float farP = 50.0f;
    const float forward = 10.0f;
    // 45 degrees off the cone axis: radial = forward * sqrt(2), cosine = 1/sqrt(2) ~ 0.7071.
    const float radial = forward * std::numbers::sqrt2_v<float>;

    const float onAxis = perspectiveNormalizedDepthPerWorldUnit(nearP, farP, forward, forward);
    const float offAxis = perspectiveNormalizedDepthPerWorldUnit(nearP, farP, forward, radial);

    CHECK(offAxis == Catch::Approx(onAxis / std::numbers::sqrt2_v<float>).margin(1e-9f));
    // Stated the way the defect would present: without the cosine this fragment would receive ~41%
    // more normalised bias than it should.
    CHECK(onAxis / offAxis == Catch::Approx(1.41421f).margin(1e-4f));
}

TEST_CASE("ShadowBias.SpotConversionRejectsAGrosslyImpossibleRadius", "[ShadowBias]")
{
    // Radial is a hypotenuse. MEANINGFULLY shorter than its own forward leg is not an extreme
    // angle, it is bad input — and silently clamping the cosine to 1 would hand back a plausible
    // number built from a geometry that cannot exist.
    CHECK(perspectiveNormalizedDepthPerWorldUnit(0.1f, 50.0f, 10.0f, 9.0f) == Catch::Approx(0.0f));
    CHECK(perspectiveNormalizedDepthPerWorldUnit(0.1f, 50.0f, 10.0f, 0.0f) == Catch::Approx(0.0f));
}

TEST_CASE("ShadowBias.SpotConversionToleratesOnAxisRounding", "[ShadowBias]")
{
    // The case the strict comparison broke. On the cone axis radial and forward SHOULD be equal,
    // but one comes from a `length` and the other from a `dot`, so they land an ULP apart in
    // whichever direction the rounding falls. Rejecting that would zero the bias straight down the
    // axis — the last place anyone would think to look — so a sub-ULP shortfall is tolerated and
    // the cosine is clamped to 1 rather than allowed slightly above it.
    const float forward = 10.0f;
    const float oneUlpShort = std::nextafter(forward, 0.0f);
    const float oneUlpLong = std::nextafter(forward, 100.0f);
    const float exact = perspectiveNormalizedDepthPerWorldUnit(0.1f, 50.0f, forward, forward);

    REQUIRE(oneUlpShort < forward);
    CHECK(perspectiveNormalizedDepthPerWorldUnit(0.1f, 50.0f, forward, oneUlpShort) ==
          Catch::Approx(exact).margin(1e-9f));
    CHECK(perspectiveNormalizedDepthPerWorldUnit(0.1f, 50.0f, forward, oneUlpLong) ==
          Catch::Approx(exact).margin(1e-9f));
}

TEST_CASE("ShadowBias.AMalformedPolicyYieldsNoBiasRatherThanAnInvertedOne", "[ShadowBias]")
{
    // A negative scale is not a smaller bias: it biases the receiver INTO its caster, so every lit
    // surface self-shadows. A NaN unlights whatever it touches. Sanitised to 0 here, and
    // identically in shaders/shadow_bias.glsl — the policy is compile-time today, but the two sides
    // must not disagree the day it is not.
    ShadowBiasInputs sloped = kFaceOnCascade;
    sloped.nDotL = cosineOf(45.0f);

    ShadowBiasPolicy negative = kPolicy;
    negative.slopeScale = -4.0f;
    negative.constantTexels = -1.0f;
    negative.normalOffsetTexels = -2.0f;
    const ShadowBias fromNegative = shadowBias(sloped, negative);
    CHECK(fromNegative.receiverDepthBias == Catch::Approx(0.0f));
    CHECK(fromNegative.normalOffsetWorld == Catch::Approx(0.0f));

    ShadowBiasPolicy notFinite = kPolicy;
    notFinite.slopeScale = std::numeric_limits<float>::quiet_NaN();
    notFinite.constantTexels = std::numeric_limits<float>::infinity();
    const ShadowBias fromNaN = shadowBias(sloped, notFinite);
    CHECK(std::isfinite(fromNaN.receiverDepthBias));
    CHECK(fromNaN.receiverDepthBias == Catch::Approx(0.0f));

    // A negative filter radius collapses to the single-tap footprint rather than shrinking it below
    // one texel.
    ShadowBiasInputs negativeRadius = sloped;
    negativeRadius.filterRadiusTexels = -5.0f;
    CHECK(shadowBias(negativeRadius, kPolicy).receiverDepthBias ==
          Catch::Approx(shadowBias(sloped, kPolicy).receiverDepthBias).margin(1e-6f));
}

TEST_CASE("ShadowBias.SpotMetricsRejectADegenerateFrustum", "[ShadowBias]")
{
    CHECK(perspectiveNormalizedDepthPerWorldUnit(0.1f, 50.0f, 0.0f, 0.0f) == Catch::Approx(0.0f));
    CHECK(perspectiveNormalizedDepthPerWorldUnit(0.0f, 50.0f, 10.0f, 10.0f) == Catch::Approx(0.0f));
    // far <= near is a collapsed or inverted frustum, not a very thin one.
    CHECK(perspectiveNormalizedDepthPerWorldUnit(50.0f, 50.0f, 10.0f, 10.0f) ==
          Catch::Approx(0.0f));
    CHECK(perspectiveNormalizedDepthPerWorldUnit(50.0f, 1.0f, 10.0f, 10.0f) == Catch::Approx(0.0f));
}

TEST_CASE("ShadowBias.CubeFaceFootprintFollowsTheMajorAxisNotTheRadius", "[ShadowBias]")
{
    // The distinction that keeps the shared law honest for point lights. A cube face divides by its
    // MAJOR AXIS, so that is what sets the footprint; radial distance is longer everywhere except
    // dead centre of the face, and using it would bias hardest exactly where the face is most
    // oblique. The stored comparison depth stays radial — a different question, deliberately.
    const float texelAxisScale = 2.0f / 512.0f;
    const float majorAxis = 6.0f;
    // A fragment 6 units along the face axis and 4 units off it: radius is ~7.2, major axis is 6.
    const float radial = std::sqrt(majorAxis * majorAxis + 4.0f * 4.0f);
    REQUIRE(radial > majorAxis);

    CHECK(cubeFaceWorldUnitsPerTexel(texelAxisScale, majorAxis) ==
          Catch::Approx(majorAxis * texelAxisScale));
    CHECK(cubeFaceWorldUnitsPerTexel(texelAxisScale, majorAxis) <
          cubeFaceWorldUnitsPerTexel(texelAxisScale, radial));
}

TEST_CASE("ShadowBias.CubeDepthConversionIsLinearInRange", "[ShadowBias]")
{
    // The point path stores radial distance / range (shaders/shadow_depth.glsl), so the conversion
    // is constant — no per-fragment term, unlike the spot's inverse square.
    CHECK(cubeNormalizedDepthPerWorldUnit(25.0f) == Catch::Approx(1.0f / 25.0f));
    CHECK(cubeNormalizedDepthPerWorldUnit(0.0f) == Catch::Approx(0.0f));
}

// ---------------------------------------------------------------------------
// SH-07 filtering: the PCF kernel's support.
//
// The sampler and the bias law must mean the same thing by "radius". The kernel in `shader.frag` is
// a fixed table, so its support is checkable here rather than only observable as acne: the law
// clears `filterRadiusTexels + 1` texels, and a table reaching beyond 1.0 would put its outer taps
// past that — self-shadowing that appears only once filtering is enabled and reads as a filter bug.
// ---------------------------------------------------------------------------

TEST_CASE("ShadowBias.ThePcfKernelHasUnitSupport", "[ShadowBias]")
{
    // THE PRODUCTION TAPS, not a copy of them. `shaders/poisson_taps.inl` is a list of
    // POISSON_TAP(x, y) invocations that shader.frag expands into a vec2[16] and this test expands
    // into the array below, so editing a tap out of unit support fails here. A transcribed table
    // would only ever have proved the transcription right.
#define POISSON_TAP(x, y)                                                                          \
    std::array<float, 2>                                                                           \
    {                                                                                              \
        static_cast<float>(x), static_cast<float>(y)                                               \
    }
    constexpr std::array<std::array<float, 2>, 16> kKernel{{
#include "../../shaders/poisson_taps.inl"
    }};
#undef POISSON_TAP

    float support = 0.0f;
    for (const auto& tap : kKernel)
    {
        support = std::max(support, std::hypot(tap[0], tap[1]));
    }

    // Exactly 1: the table is normalised BY its own support, so the outermost tap sits on the unit
    // circle. Below 1 would waste the radius; above 1 is the defect.
    CHECK(support == Catch::Approx(1.0f).margin(1e-5f));
    // And the law's footprint covers it at every radius, which is the property the normalisation
    // exists to guarantee.
    for (const float radius : {0.0f, 1.0f, 3.0f, 8.0f})
    {
        CAPTURE(radius);
        CHECK(support * radius <= radius + 1.0f);
    }
}
