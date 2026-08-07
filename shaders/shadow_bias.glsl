// SH-07: THE shadow bias law, and the per-projection metrics that feed it — the single production
// implementation, shared by every receiver path.
//
// This is the mirror of `graphics/shadow_bias.hpp` / `shadow_bias.cpp`. The C++ side is the
// EXECUTABLE SPECIFICATION: it is unit-tested headlessly (tests/graphics/test_shadow_bias.cpp) and
// its golden values are the arbiter if the two ever disagree. This side is what actually runs, and it
// exists separately because the law's inputs are per fragment — nDotL, and the depths the punctual
// metrics derive from — so no amount of CPU precomputation can supply them.
//
// Both sides SANITISE identically. A negative policy scale is not a smaller bias but an inverted one
// (the receiver biases into its caster and self-shadows everywhere), and a non-finite value unlights
// whatever it touches; the policy is compile-time today, and this keeps the two honest the day it is
// not.
//
// Units, named because mixing them silently is the whole defect this replaced (`exp2(cascade)` stood
// in for a texel footprint AND a depth-range conversion at once):
//   worldUnitsPerTexel          — world-space size of one shadow-map texel at the receiver
//   normalizedDepthPerWorldUnit — stored-depth change per world unit ALONG THE LIGHT RAY
//   returned .depth             — normalised depth, subtract from the compare value
//   returned .normalOffsetWorld — world units, apply along the normal BEFORE projecting

#ifndef FIRE_ENGINE_SHADOW_BIAS_GLSL
#define FIRE_ENGINE_SHADOW_BIAS_GLSL

struct ShadowBias {
    float depth;
    float normalOffsetWorld;
};

// Mirrors nonNegativeOrZero / finiteOrZero: 0 is a legitimate value for a policy term or a filter
// radius, negatives and non-finites are not.
float shadowBiasSanitise(float v)
{
    return (isnan(v) || isinf(v) || v <= 0.0) ? 0.0 : v;
}

// THE LAW. policy = (slopeScale, constantTexels, normalOffsetTexels, maxSlopeTangent), which is
// LightUBO::shadowParams laid out by the renderer.
ShadowBias shadowBiasFor(float worldUnitsPerTexel, float normalizedDepthPerWorldUnit,
                         float nDotL, float filterRadiusTexels, vec4 policy)
{
    ShadowBias result;
    result.depth = 0.0;
    result.normalOffsetWorld = 0.0;

    float worldPerTexel = shadowBiasSanitise(worldUnitsPerTexel);
    float depthPerWorld = shadowBiasSanitise(normalizedDepthPerWorldUnit);
    if (worldPerTexel == 0.0 || depthPerWorld == 0.0) {
        return result; // no metrics (an inactive slot, a degenerate fit) — no bias, visibly
    }

    float slopeScale = shadowBiasSanitise(policy.x);
    float constantTexels = shadowBiasSanitise(policy.y);
    float normalOffsetTexels = shadowBiasSanitise(policy.z);
    float maxSlopeTangent = shadowBiasSanitise(policy.w);

    // tan(theta) between surface and light: the depth a surface at this angle traverses across one
    // texel. CLAMPED — at grazing incidence it diverges, and an unbounded bias detaches the shadow
    // from its caster, which is worse and less diagnosable than the acne it was avoiding.
    float cosTheta = clamp(nDotL, 0.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float tanTheta = cosTheta > 0.0 ? min(sinTheta / cosTheta, maxSlopeTangent) : maxSlopeTangent;

    // The kernel reads a disc, so the footprint to clear is its radius plus the texel itself.
    float footprintTexels = shadowBiasSanitise(filterRadiusTexels) + 1.0;

    // WORLD units first, ONE conversion at the end. That order is what stops the two scales being
    // applied twice or half.
    float slopeWorld = worldPerTexel * footprintTexels * slopeScale * tanTheta;
    float constantWorld = worldPerTexel * constantTexels;
    result.depth = (slopeWorld + constantWorld) * depthPerWorld;
    result.normalOffsetWorld = worldPerTexel * footprintTexels * normalOffsetTexels * sinTheta;
    return result;
}

// --- per-projection metrics (mirrors the free functions in graphics/shadow_bias.hpp) ------------

// SPOT: footprint grows with FORWARD depth; the depth conversion falls as its square AND is measured
// along the light ray, so it carries the ray-forward cosine. Omitting that cosine over-converts every
// off-axis fragment (~41% too much bias at 45 degrees off the cone axis).
float spotNormalizedDepthPerWorldUnit(float nearPlane, float farPlane, float forwardDepth,
                                      float radialDepth)
{
    float depth = shadowBiasSanitise(forwardDepth);
    float radial = shadowBiasSanitise(radialDepth);
    float nearP = shadowBiasSanitise(nearPlane);
    float farP = shadowBiasSanitise(farPlane);
    if (depth == 0.0 || radial == 0.0 || nearP == 0.0 || farP <= nearP) {
        return 0.0;
    }
    // Slack for ULP disagreement between a length and a dot taken from ONE vector: on-axis they
    // should be equal and routinely are not, and rejecting there would zero the bias straight down
    // the cone axis.
    float slack = 8.0 * 1.1920929e-7 * max(depth, radial);
    if (radial + slack < depth) {
        return 0.0;
    }
    float perForwardUnit = (nearP * farP) / ((farP - nearP) * depth * depth);
    return perForwardUnit * min(1.0, depth / radial);
}

// POINT: the face's footprint follows the MAJOR AXIS of the light-to-fragment vector, because that is
// the axis its 90-degree projection divides by. Radial distance would overstate it everywhere except
// dead centre of a face. The stored comparison depth stays radial — a different question.
float pointMajorAxisDepth(vec3 toFragment)
{
    vec3 a = abs(toFragment);
    return max(a.x, max(a.y, a.z));
}

#endif // FIRE_ENGINE_SHADOW_BIAS_GLSL
