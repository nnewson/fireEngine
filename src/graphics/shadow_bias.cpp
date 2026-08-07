#include <fire_engine/graphics/shadow_bias.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace fire_engine
{

namespace
{

// Every metric below is a divisor away from a degenerate input, and each of them is reachable: a
// zero-extent map, a collapsed depth span, a fragment exactly at the light. Returning 0 rather than
// an infinity is deliberate — 0 propagates into "no bias", which produces visible acne, while an
// infinity propagates into a bias that detaches every shadow in the frame. The first is a bug you
// can see and locate; the second looks like the shadows were never implemented.
[[nodiscard]] float safeReciprocal(float value) noexcept
{
    if (!std::isfinite(value) || value <= 0.0f)
    {
        return 0.0f;
    }
    return 1.0f / value;
}

[[nodiscard]] float finiteOrZero(float value) noexcept
{
    return std::isfinite(value) && value > 0.0f ? value : 0.0f;
}

// Policy and filter-radius sanitisation: 0 is a legitimate value here rather than a rejection — a
// policy may switch a term off, and radius 0 is a single-tap comparison — while negatives and
// non-finites are not, for the reasons ShadowBiasPolicy documents.
[[nodiscard]] float nonNegativeOrZero(float value) noexcept
{
    return std::isfinite(value) && value > 0.0f ? value : 0.0f;
}

} // namespace

ShadowBias shadowBias(const ShadowBiasInputs& inputs, const ShadowBiasPolicy& policy) noexcept
{
    // Policy first, sanitised rather than trusted — mirrored in shaders/shadow_bias.glsl so the two
    // sides cannot disagree about a malformed policy.
    const float slopeScale = nonNegativeOrZero(policy.slopeScale);
    const float constantTexels = nonNegativeOrZero(policy.constantTexels);
    const float normalOffsetTexels = nonNegativeOrZero(policy.normalOffsetTexels);
    const float maxSlopeTangent = nonNegativeOrZero(policy.maxSlopeTangent);

    const float worldPerTexel = finiteOrZero(inputs.worldUnitsPerTexel);
    const float depthPerWorld = finiteOrZero(inputs.normalizedDepthPerWorldUnit);
    if (worldPerTexel == 0.0f || depthPerWorld == 0.0f)
    {
        return ShadowBias{};
    }

    // tan(theta) between the surface and the light: how much depth a surface at this angle
    // traverses across one texel of footprint. CLAMPED, because at grazing incidence it diverges
    // and an unbounded bias peter-pans the shadow off its caster — a worse artefact than the acne
    // the term exists to prevent, and one that looks like a missing feature rather than a tuning
    // error.
    const float cosTheta = std::clamp(inputs.nDotL, 0.0f, 1.0f);
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    const float tanTheta =
        cosTheta > 0.0f ? std::min(sinTheta / cosTheta, maxSlopeTangent) : maxSlopeTangent;

    // The kernel reads a disc, not a texel, so the footprint the bias has to clear is the radius
    // the filter samples plus the texel itself.
    const float footprintTexels = nonNegativeOrZero(inputs.filterRadiusTexels) + 1.0f;

    // WORLD units first, then one conversion into normalised depth. Doing it in this order is what
    // keeps the two scales from being applied twice or half — the mistake `exp2(cascade)` was
    // making by standing in for both at once.
    const float slopeWorld = worldPerTexel * footprintTexels * slopeScale * tanTheta;
    const float constantWorld = worldPerTexel * constantTexels;
    ShadowBias bias{};
    bias.receiverDepthBias = (slopeWorld + constantWorld) * depthPerWorld;
    // The normal offset is a WORLD-space nudge applied before projection, so it converts nothing:
    // it scales with the texel footprint and with how obliquely the light meets the surface, since
    // a face-on receiver has no in-texel depth range to escape.
    bias.normalOffsetWorld = worldPerTexel * footprintTexels * normalOffsetTexels * sinTheta;
    return bias;
}

float orthographicNormalizedDepthPerWorldUnit(float depthSpanWorld) noexcept
{
    return safeReciprocal(depthSpanWorld);
}

float perspectiveWorldUnitsPerTexel(float texelAngleScale, float forwardDepthWorld) noexcept
{
    return finiteOrZero(texelAngleScale) * finiteOrZero(forwardDepthWorld);
}

float perspectiveNormalizedDepthPerWorldUnit(float nearPlane, float farPlane,
                                             float forwardDepthWorld,
                                             float radialDepthWorld) noexcept
{
    const float depth = finiteOrZero(forwardDepthWorld);
    const float radial = finiteOrZero(radialDepthWorld);
    const float near = finiteOrZero(nearPlane);
    const float far = finiteOrZero(farPlane);
    if (depth == 0.0f || radial == 0.0f || near == 0.0f || far <= near)
    {
        return 0.0f;
    }
    // The radial distance is a hypotenuse, so shorter than its own forward leg is geometrically
    // impossible — but only GROSSLY shorter is bad input. Both values are derived from one vector
    // (`length` and a `dot`), and on-axis, where they should be equal, rounding routinely puts them
    // an ULP apart in either direction. Rejecting on the raw comparison would zero the bias for
    // fragments straight down the cone axis, which is the last place anyone would look for it. The
    // tolerance is relative, matching the unit-length check in `render/cascade_fit.cpp`.
    const float slack = 8.0f * std::numeric_limits<float>::epsilon() * std::max(depth, radial);
    if (radial + slack < depth)
    {
        return 0.0f;
    }
    // TWO factors. d(z_ndc)/d(forwardDepth) falls off as 1/depth^2 — why a single constant bias
    // cannot serve a cone, being adequate at its mouth and orders of magnitude too small at its far
    // end...
    const float perForwardUnit = (near * far) / ((far - near) * depth * depth);
    // ...and d(forwardDepth)/d(alongRay) = cos(ray, spotForward), because the law's slope term is
    // measured along the LOCAL LIGHT RAY — that is what nDotL is against — while the stored depth
    // is a projection onto the spot's forward axis. Off-axis those differ, and omitting the cosine
    // over-converts: ~41% too much bias at 45 degrees off-axis, worst at the rim.
    // Clamped for the same reason the slack exists: within tolerance the ratio can exceed 1 by an
    // ULP, and a cosine above 1 is not a small error but a nonsensical one.
    return perForwardUnit * std::min(1.0f, depth / radial);
}

float cubeFaceWorldUnitsPerTexel(float texelAxisScale, float majorAxisDepth) noexcept
{
    return finiteOrZero(texelAxisScale) * finiteOrZero(majorAxisDepth);
}

float cubeNormalizedDepthPerWorldUnit(float rangeWorld) noexcept
{
    // Linear, because the point path stores radial distance / range rather than a projected depth
    // (shaders/shadow_depth.glsl). No per-fragment term.
    return safeReciprocal(rangeWorld);
}

} // namespace fire_engine
