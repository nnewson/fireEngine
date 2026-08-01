#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>
#include <fire_engine/math/view_basis.hpp>
#include <fire_engine/render/cascade_fit.hpp>
#include <fire_engine/render/constants.hpp>

using Catch::Approx;
using fire_engine::CascadeDepthFit;
using fire_engine::CascadeReceiverFit;
using fire_engine::CascadeReceiverInput;
using fire_engine::fitLegacyCascadeDepth;
using fire_engine::Mat4;
using fire_engine::Vec3;
using fire_engine::Vec4;

namespace
{

// The pre-extraction `fitCascade` lambda from `Renderer::computeShadowCascades`, copied verbatim
// down to the order of every operation. It is the whole point of this file: the extraction is only
// safe if the shipped matrices are unchanged, and "unchanged" here means BIT-identical, not close.
// When SH-06 replaces the depth policy, this reference stays as the record of what the fixed
// back-extension used to produce.
struct LegacyFit
{
    Mat4 viewProj;
    float worldPerTexel;
};

[[nodiscard]] LegacyFit legacyFitCascade(Vec3 cameraPosition, Vec3 cameraTarget, Vec3 lightDirIn,
                                         float aspect, float sliceNear, float sliceFar,
                                         std::uint32_t extent, float backExtend)
{
    const Vec3 lightDir = lightDirIn;
    const float tanHalfFov = std::tan(fire_engine::kCameraFovRadians * 0.5f);
    const fire_engine::ViewBasis basis = fire_engine::makeViewBasis(cameraPosition, cameraTarget);
    const Vec3 lightUp = fire_engine::stableUpForForward(lightDir);
    const Vec3 lightRight =
        fire_engine::normaliseOr(Vec3::crossProduct(lightDir, lightUp), {1.0f, 0.0f, 0.0f});
    const Vec3 lightUpOrtho =
        fire_engine::normaliseOr(Vec3::crossProduct(lightRight, lightDir), lightUp);
    const float shadowMapExtentF = static_cast<float>(extent);

    const float nearH = tanHalfFov * sliceNear;
    const float nearW = nearH * aspect;
    const float farH = tanHalfFov * sliceFar;
    const float farW = farH * aspect;

    const Vec3 sliceNearCentre = cameraPosition + basis.forward * sliceNear;
    const Vec3 sliceFarCentre = cameraPosition + basis.forward * sliceFar;

    const std::array<Vec3, 8> corners{sliceNearCentre - basis.right * nearW - basis.up * nearH,
                                      sliceNearCentre + basis.right * nearW - basis.up * nearH,
                                      sliceNearCentre + basis.right * nearW + basis.up * nearH,
                                      sliceNearCentre - basis.right * nearW + basis.up * nearH,
                                      sliceFarCentre - basis.right * farW - basis.up * farH,
                                      sliceFarCentre + basis.right * farW - basis.up * farH,
                                      sliceFarCentre + basis.right * farW + basis.up * farH,
                                      sliceFarCentre - basis.right * farW + basis.up * farH};

    Vec3 frustumCentre{0.0f, 0.0f, 0.0f};
    for (const auto& c : corners)
    {
        frustumCentre += c;
    }
    frustumCentre /= 8.0f;

    float radius = 0.0f;
    for (const auto& c : corners)
    {
        radius = std::max(radius, (c - frustumCentre).magnitude());
    }
    radius = std::ceil(radius * 16.0f) / 16.0f;

    const float worldPerTexel = (2.0f * radius) / shadowMapExtentF;
    const float centreU = Vec3::dotProduct(frustumCentre, lightRight);
    const float centreV = Vec3::dotProduct(frustumCentre, lightUpOrtho);
    const float centreW = Vec3::dotProduct(frustumCentre, lightDir);
    const float snappedU = std::floor(centreU / worldPerTexel) * worldPerTexel;
    const float snappedV = std::floor(centreV / worldPerTexel) * worldPerTexel;
    const Vec3 snappedCentre = lightRight * snappedU + lightUpOrtho * snappedV + lightDir * centreW;

    const Vec3 lightPos = snappedCentre - lightDir * (radius + backExtend);
    const Mat4 lightView = Mat4::lookAt(lightPos, snappedCentre, lightUpOrtho);
    const Mat4 lightProj =
        Mat4::ortho(-radius, radius, -radius, radius, 0.0f, 2.0f * radius + 2.0f * backExtend);
    return LegacyFit{lightProj * lightView, worldPerTexel};
}

// Element-wise on the bit patterns, not `memcmp` on the object: float has no unique object
// representation. Each element must also be FINITE — bit equality alone would certify two
// identically poisoned matrices as a match, which is the one way this equivalence check could pass
// while both sides were broken.
[[nodiscard]] bool bitIdentical(const Mat4& a, const Mat4& b)
{
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            if (!std::isfinite(a[row, col]) || !std::isfinite(b[row, col]))
            {
                return false;
            }
            if (std::bit_cast<std::uint32_t>(a[row, col]) !=
                std::bit_cast<std::uint32_t>(b[row, col]))
            {
                return false;
            }
        }
    }
    return true;
}

// The shipped cascade splits, so the equivalence check covers the slices that actually render
// rather than round numbers that happen to be well-conditioned.
[[nodiscard]] std::vector<std::pair<float, float>> shippedSlices()
{
    std::vector<std::pair<float, float>> slices;
    float sliceNear = fire_engine::kCameraNearPlane;
    for (std::uint32_t i = 0; i < fire_engine::kShadowCascadeCount; ++i)
    {
        const float p =
            static_cast<float>(i + 1) / static_cast<float>(fire_engine::kShadowCascadeCount);
        const float linear = fire_engine::kCameraNearPlane +
                             (fire_engine::kShadowFarPlane - fire_engine::kCameraNearPlane) * p;
        const float logSplit =
            fire_engine::kCameraNearPlane *
            std::pow(fire_engine::kShadowFarPlane / fire_engine::kCameraNearPlane, p);
        const float split = fire_engine::kShadowCascadeSplitLambda * logSplit +
                            (1.0f - fire_engine::kShadowCascadeSplitLambda) * linear;
        slices.emplace_back(sliceNear, split);
        sliceNear = split;
    }
    return slices;
}

struct Scenario
{
    const char* name;
    Vec3 cameraPosition;
    Vec3 cameraTarget;
    Vec3 lightDirection;
    float aspect;
};

[[nodiscard]] std::vector<Scenario> scenarios()
{
    return {
        {"sun down the diagonal",
         {0.0f, 2.0f, 8.0f},
         {0.0f, 1.0f, 0.0f},
         Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f}),
         16.0f / 9.0f},
        {"near-vertical sun",
         {12.0f, 3.0f, -4.0f},
         {0.0f, 0.5f, 0.0f},
         Vec3::normalise(Vec3{0.02f, -1.0f, 0.01f}),
         4.0f / 3.0f},
        {"low sun, camera looking up",
         {-6.0f, 1.0f, -6.0f},
         {2.0f, 6.0f, 3.0f},
         Vec3::normalise(Vec3{-0.9f, -0.1f, 0.4f}),
         1.0f},
        {"tall viewport",
         {3.0f, 40.0f, 3.0f},
         {3.0f, 0.0f, 3.5f},
         Vec3::normalise(Vec3{0.3f, -0.8f, -0.5f}),
         0.5f},
    };
}

[[nodiscard]] CascadeReceiverInput inputFor(const Scenario& s, float sliceNear, float sliceFar)
{
    return CascadeReceiverInput{.cameraPosition = s.cameraPosition,
                                .cameraTarget = s.cameraTarget,
                                .lightDirection = s.lightDirection,
                                .fovRadians = fire_engine::kCameraFovRadians,
                                .aspect = s.aspect,
                                .sliceNear = sliceNear,
                                .sliceFar = sliceFar,
                                .shadowMapExtent = fire_engine::kShadowMapExtent};
}

} // namespace

TEST_CASE("CascadeFit.LegacyDepthPolicyIsBitIdentical", "[CascadeFit]")
{
    for (const Scenario& s : scenarios())
    {
        for (const auto& [sliceNear, sliceFar] : shippedSlices())
        {
            INFO(s.name << " slice [" << sliceNear << ", " << sliceFar << "]");
            const auto receiver = CascadeReceiverFit::fit(inputFor(s, sliceNear, sliceFar));
            REQUIRE(receiver);
            const auto depth =
                fitLegacyCascadeDepth(*receiver, fire_engine::kShadowDepthBackExtend);
            REQUIRE(depth);
            const LegacyFit legacy = legacyFitCascade(
                s.cameraPosition, s.cameraTarget, s.lightDirection, s.aspect, sliceNear, sliceFar,
                fire_engine::kShadowMapExtent, fire_engine::kShadowDepthBackExtend);

            CHECK(bitIdentical(depth->viewProj, legacy.viewProj));
            CHECK(receiver->worldPerTexel() == legacy.worldPerTexel);
        }
    }
}

// Guards the reference itself: if `legacyFitCascade` had drifted into simply calling the new code,
// or into something insensitive to its inputs, the check above would pass vacuously.
TEST_CASE("CascadeFit.LegacyReferenceRespondsToItsInputs", "[CascadeFit]")
{
    const Scenario s = scenarios().front();
    const LegacyFit a =
        legacyFitCascade(s.cameraPosition, s.cameraTarget, s.lightDirection, s.aspect, 1.0f, 10.0f,
                         fire_engine::kShadowMapExtent, fire_engine::kShadowDepthBackExtend);
    const LegacyFit wider =
        legacyFitCascade(s.cameraPosition, s.cameraTarget, s.lightDirection, s.aspect, 1.0f, 20.0f,
                         fire_engine::kShadowMapExtent, fire_engine::kShadowDepthBackExtend);
    const LegacyFit deeper =
        legacyFitCascade(s.cameraPosition, s.cameraTarget, s.lightDirection, s.aspect, 1.0f, 10.0f,
                         fire_engine::kShadowMapExtent, fire_engine::kShadowDepthBackExtend * 2.0f);

    CHECK_FALSE(bitIdentical(a.viewProj, wider.viewProj));
    CHECK(a.worldPerTexel < wider.worldPerTexel);
    // A different back-extension must move the matrix even though the XY fit is untouched — that is
    // the exact axis SH-06 changes.
    CHECK_FALSE(bitIdentical(a.viewProj, deeper.viewProj));
    CHECK(a.worldPerTexel == deeper.worldPerTexel);
}

TEST_CASE("CascadeFit.ProjectionMapsFittedBoundsToClipEdges", "[CascadeFit]")
{
    for (const Scenario& s : scenarios())
    {
        INFO(s.name);
        const auto receiver = CascadeReceiverFit::fit(inputFor(s, 1.0f, 12.0f));
        REQUIRE(receiver);
        const auto depth = fitLegacyCascadeDepth(*receiver, fire_engine::kShadowDepthBackExtend);
        REQUIRE(depth);

        // A point is placed by its light-space (U, V, W) rather than by any world position, so the
        // assertion is about the projection's contract, not about a particular scene.
        auto atUvw = [&](float u, float v, float w)
        {
            const Vec3 p = receiver->lightRight() * u + receiver->lightUp() * v +
                           receiver->lightDirection() * w;
            const Vec4 clip = depth->viewProj * Vec4{p.x(), p.y(), p.z(), 1.0f};
            return clip;
        };

        const float midW = 0.5f * (depth->nearW + depth->farW);
        // Vulkan clip: x right-handed in [-1, 1], y FLIPPED by Mat4::ortho, z in [0, 1].
        CHECK(atUvw(receiver->minU(), 0.0f, midW).x() == Approx(-1.0f).margin(1e-4));
        CHECK(atUvw(receiver->maxU(), 0.0f, midW).x() == Approx(1.0f).margin(1e-4));
        CHECK(atUvw(0.0f, receiver->minV(), midW).y() == Approx(1.0f).margin(1e-4));
        CHECK(atUvw(0.0f, receiver->maxV(), midW).y() == Approx(-1.0f).margin(1e-4));

        CHECK(atUvw(0.0f, 0.0f, depth->nearW).z() == Approx(0.0f).margin(1e-4));
        CHECK(atUvw(0.0f, 0.0f, depth->farW).z() == Approx(1.0f).margin(1e-4));
        CHECK(depth->viewDepthSpan == Approx(depth->farW - depth->nearW));
        // The light sits ON the near plane: the legacy ortho near distance is zero.
        CHECK(Vec3::dotProduct(depth->lightPosition, receiver->lightDirection()) ==
              Approx(depth->nearW).margin(1e-3));
    }
}

TEST_CASE("CascadeFit.ReceiverDepthComesFromTheCornersNotTheSphere", "[CascadeFit]")
{
    for (const Scenario& s : scenarios())
    {
        INFO(s.name);
        const auto receiver = CascadeReceiverFit::fit(inputFor(s, 1.0f, 12.0f));
        REQUIRE(receiver);

        // Rebuild the slice corners independently and check every one lies within the reported
        // depth extent, with the extent touching the extremes rather than merely containing them.
        const fire_engine::ViewBasis basis =
            fire_engine::makeViewBasis(s.cameraPosition, s.cameraTarget);
        const float tanHalfFov = std::tan(fire_engine::kCameraFovRadians * 0.5f);
        float minW = std::numeric_limits<float>::max();
        float maxW = std::numeric_limits<float>::lowest();
        for (const float d : {1.0f, 12.0f})
        {
            const float h = tanHalfFov * d;
            const float w = h * s.aspect;
            const Vec3 centre = s.cameraPosition + basis.forward * d;
            for (const float sx : {-1.0f, 1.0f})
            {
                for (const float sy : {-1.0f, 1.0f})
                {
                    const Vec3 corner = centre + basis.right * (w * sx) + basis.up * (h * sy);
                    const float cw = Vec3::dotProduct(corner, receiver->lightDirection());
                    minW = std::min(minW, cw);
                    maxW = std::max(maxW, cw);
                }
            }
        }
        CHECK(receiver->receiverMinW() == Approx(minW).margin(1e-4));
        CHECK(receiver->receiverMaxW() == Approx(maxW).margin(1e-4));

        // The reason the corners are used at all: the sphere can never be TIGHTER in depth, so a
        // far plane fitted from it can only sit past the real receivers. This direction is the
        // universal one — it holds for any slice — and it is deliberately stated as a bound, not a
        // strict inequality: a slice whose extreme corners happen to lie on the light axis makes
        // the two extents equal. Strictness is checked below, on a case where it genuinely holds.
        CHECK(receiver->receiverMinW() >= receiver->centreW() - receiver->radius());
        CHECK(receiver->receiverMaxW() <= receiver->centreW() + receiver->radius());
        CHECK(receiver->receiverMaxW() - receiver->receiverMinW() <= 2.0f * receiver->radius());
    }

    // The representative case: an off-axis sun over a wide slice, where the corner extent really is
    // narrower than the sphere — which is what makes the corner-derived far plane worth having.
    const auto wide = CascadeReceiverFit::fit(inputFor(scenarios().front(), 1.0f, 24.0f));
    REQUIRE(wide);
    CHECK(wide->receiverMaxW() - wide->receiverMinW() < 2.0f * wide->radius());
}

TEST_CASE("CascadeFit.ReturnedBasisIsOrthonormalAndCentreIsSnapped", "[CascadeFit]")
{
    for (const Scenario& s : scenarios())
    {
        INFO(s.name);
        const auto receiver = CascadeReceiverFit::fit(inputFor(s, 1.0f, 12.0f));
        REQUIRE(receiver);

        CHECK(receiver->lightDirection().magnitude() == Approx(1.0f).margin(1e-5));
        CHECK(receiver->lightRight().magnitude() == Approx(1.0f).margin(1e-5));
        CHECK(receiver->lightUp().magnitude() == Approx(1.0f).margin(1e-5));
        CHECK(Vec3::dotProduct(receiver->lightDirection(), receiver->lightRight()) ==
              Approx(0.0f).margin(1e-5));
        CHECK(Vec3::dotProduct(receiver->lightDirection(), receiver->lightUp()) ==
              Approx(0.0f).margin(1e-5));
        CHECK(Vec3::dotProduct(receiver->lightRight(), receiver->lightUp()) ==
              Approx(0.0f).margin(1e-5));

        CHECK(
            receiver->worldPerTexel() ==
            Approx(2.0f * receiver->radius() / static_cast<float>(fire_engine::kShadowMapExtent)));
        CHECK(receiver->maxU() - receiver->minU() == Approx(2.0f * receiver->radius()));
        CHECK(receiver->maxV() - receiver->minV() == Approx(2.0f * receiver->radius()));

        // Snapping is what stops the cascade shimmering: the centre sits on a texel boundary in U
        // and V, and moves in W only.
        const float snappedU = 0.5f * (receiver->minU() + receiver->maxU());
        const float snappedV = 0.5f * (receiver->minV() + receiver->maxV());
        CHECK(std::abs(std::remainder(snappedU, receiver->worldPerTexel())) <
              receiver->worldPerTexel() * 1e-2f);
        CHECK(std::abs(std::remainder(snappedV, receiver->worldPerTexel())) <
              receiver->worldPerTexel() * 1e-2f);
        CHECK(Vec3::dotProduct(receiver->snappedCentre(), receiver->lightDirection()) ==
              Approx(receiver->centreW()).margin(1e-3));
    }
}

// The pin that matters most: corrupt render input must FAIL rather than be repaired into something
// plausible. `makeViewBasis` and `normaliseOr` deliberately manufacture fallbacks; the fit must not
// inherit that behaviour, because a cascade fitted around a fabricated basis shadows the wrong
// region with nothing pointing at the cause.
TEST_CASE("CascadeFit.RejectsCorruptInputRatherThanManufacturingABasis", "[CascadeFit]")
{
    const Scenario s = scenarios().front();
    const CascadeReceiverInput good = inputFor(s, 1.0f, 12.0f);
    REQUIRE(CascadeReceiverFit::fit(good));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    auto rejects = [](CascadeReceiverInput in) { return !CascadeReceiverFit::fit(in).has_value(); };

    SECTION("non-finite vectors")
    {
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.cameraPosition = Vec3{nan, 0.0f, 0.0f};
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.cameraTarget = Vec3{0.0f, inf, 0.0f};
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.lightDirection = Vec3{0.0f, 0.0f, nan};
                return in;
            }()));
    }
    SECTION("degenerate directions")
    {
        // A zero light direction would silently become (0, 0, -1) via `normaliseOr`.
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.lightDirection = Vec3{0.0f, 0.0f, 0.0f};
                return in;
            }()));
        // Non-unit is rejected rather than normalised: a scaled direction scales every light-space
        // depth the fit reports, and repairing it here would hide the producer that scaled it. The
        // SUBTLE scales are the ones that matter — a doubled direction is obvious, whereas 1.0001
        // is the kind of drift an accumulated transform produces, and the tolerance is sized in
        // float rounding precisely so it is still caught.
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.lightDirection = in.lightDirection * 1.0001f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.lightDirection = in.lightDirection * 0.9999f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.lightDirection = in.lightDirection * 2.0f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.lightDirection = in.lightDirection * 0.9f;
                return in;
            }()));
        // The other side of that tolerance: real `Vec3::normalise` output must still be accepted,
        // whatever ulp its squared length landed on, or the fit would reject every genuine caller.
        for (const Scenario& scenario : scenarios())
        {
            auto in = good;
            in.lightDirection = scenario.lightDirection;
            CHECK(CascadeReceiverFit::fit(in).has_value());
        }
        // A camera pointing at itself would get `makeViewBasis`'s fallback forward.
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.cameraTarget = in.cameraPosition;
                return in;
            }()));
    }
    SECTION("out-of-range scalars")
    {
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.fovRadians = 0.0f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.fovRadians = fire_engine::pi;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.fovRadians = nan;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.aspect = 0.0f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.aspect = -1.5f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.aspect = inf;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.sliceNear = 0.0f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.sliceNear = -1.0f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.sliceFar = in.sliceNear;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.sliceFar = in.sliceNear * 0.5f;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.sliceFar = nan;
                return in;
            }()));
        CHECK(rejects(
            [&]
            {
                auto in = good;
                in.shadowMapExtent = 0;
                return in;
            }()));
    }
    SECTION("finite input whose fit overflows")
    {
        auto in = good;
        in.sliceNear = std::numeric_limits<float>::max() * 0.5f;
        in.sliceFar = std::numeric_limits<float>::max();
        CHECK(rejects(in));
    }
}

// The receiver is TRUSTED here — only `CascadeReceiverFit::fit` can produce one — so the depth fit
// validates the one input that still arrives from outside. That trust is itself pinned below, in
// the type system rather than in a runtime check.
TEST_CASE("CascadeFit.DepthFitRejectsAnUnusableBackExtension", "[CascadeFit]")
{
    const Scenario s = scenarios().front();
    const auto good = CascadeReceiverFit::fit(inputFor(s, 1.0f, 12.0f));
    REQUIRE(good);
    REQUIRE(fitLegacyCascadeDepth(*good, fire_engine::kShadowDepthBackExtend));

    auto rejects = [&](float backExtend)
    { return !fitLegacyCascadeDepth(*good, backExtend).has_value(); };

    // Negative is the dangerous one, and it is dangerous at BOTH magnitudes — verified against the
    // raw expressions: a small negative value keeps the range ordered but pulls both planes inside
    // the fitted sphere (silently clipping the cascade's own contents), while anything past -radius
    // reverses it. Both yield a fully finite matrix, so neither is caught downstream.
    CHECK(rejects(-1.0f));
    CHECK(rejects(-(good->radius() + 10.0f)));
    CHECK(rejects(std::numeric_limits<float>::quiet_NaN()));
    CHECK(rejects(std::numeric_limits<float>::infinity()));
    // Zero is legitimate — that is a cascade fitted exactly to its bounding sphere.
    CHECK(fitLegacyCascadeDepth(*good, 0.0f).has_value());
    // A finite extension large enough to overflow the range still has to fail, since the failure
    // shows up in the OUTPUT rather than the input.
    CHECK(rejects(std::numeric_limits<float>::max()));
}

// The hole encapsulation closes, stated where it cannot rot: a `lightUp` equal to `lightDirection`
// is finite and passes every field-wise check worth writing, yet sends `Mat4::lookAt` to its own
// fallback up — manufacturing the plausible basis this API refuses. No runtime test can cover that
// now, because there is no longer a way to express it. These assertions are what say so.
TEST_CASE("CascadeFit.ReceiverFitCannotBeAssembledByHand", "[CascadeFit]")
{
    // Not default-constructible and not an aggregate: no `CascadeReceiverFit{}`, no designated
    // initialisers, no assignment to a member after the fact. The factory is the only door.
    STATIC_REQUIRE_FALSE(std::is_default_constructible_v<CascadeReceiverFit>);
    STATIC_REQUIRE_FALSE(std::is_aggregate_v<CascadeReceiverFit>);
    // A fitted receiver is an immutable value object and remains valid when copied — copying is not
    // a way to bypass the factory.
    STATIC_REQUIRE(std::is_copy_constructible_v<CascadeReceiverFit>);

    // And the invariant the type now guarantees, on a real fit: an orthonormal basis, so
    // `Mat4::lookAt` never has cause to substitute an up vector of its own.
    const auto fit = CascadeReceiverFit::fit(inputFor(scenarios().front(), 1.0f, 12.0f));
    REQUIRE(fit);
    CHECK(std::abs(Vec3::dotProduct(fit->lightUp(), fit->lightDirection())) < 1e-5f);
    CHECK(fit->lightUp().magnitude() == Approx(1.0f).margin(1e-5));
    CHECK(fit->radius() > 0.0f);
    CHECK(fit->worldPerTexel() > 0.0f);
}

// The equivalence check is only as good as its comparator, and the comparator's failure mode is
// certifying two identically broken matrices as a match.
TEST_CASE("CascadeFit.BitIdenticalRejectsPoisonedMatrices", "[CascadeFit]")
{
    const Mat4 identity = Mat4::identity();
    CHECK(bitIdentical(identity, identity));

    Mat4 poisoned = Mat4::identity();
    poisoned[2, 2] = std::numeric_limits<float>::quiet_NaN();
    Mat4 samePoison = Mat4::identity();
    samePoison[2, 2] = std::numeric_limits<float>::quiet_NaN();
    // Same bits on both sides, and still not a match: NaN never certifies anything.
    CHECK_FALSE(bitIdentical(poisoned, samePoison));
    CHECK_FALSE(bitIdentical(poisoned, identity));

    Mat4 infinite = Mat4::identity();
    infinite[0, 3] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(bitIdentical(infinite, infinite));

    // A one-ulp difference must still register — the whole reason the comparison is bitwise.
    Mat4 nudged = Mat4::identity();
    nudged[1, 1] = std::nextafter(1.0f, 2.0f);
    CHECK_FALSE(bitIdentical(identity, nudged));
}
