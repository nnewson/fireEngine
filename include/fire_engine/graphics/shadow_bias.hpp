#pragma once

namespace fire_engine
{

// SH-07: the shadow bias law, and the per-projection metrics that feed it
// ([`docs/shadowplans.md`](../../../docs/shadowplans.md) § SH-07). Vulkan-free and
// headless-testable, like the SH-02 selection model beside it.
//
// The defect this exists to fix: `shader.frag` scaled both the depth bias and the normal offset by
// `exp2(cascade)`, asserting that each cascade's world-per-texel doubles exactly. Practical splits
// and a bounding-sphere fit do not guarantee that, and since SH-06 fitted the depth range per
// cascade the world-to-depth conversion differs per cascade too — so one hard-coded guess was
// covering two independent scales, in units nobody had written down.
//
// UNITS ARE THE WHOLE POINT HERE, so every quantity below names its own:
//
//   * `worldUnitsPerTexel` — the world-space footprint of ONE shadow-map texel at the receiver.
//   * `normalizedDepthPerWorldUnit` — how much stored depth changes per world unit ALONG THE LIGHT,
//     at the receiver. The conversion that turns a world-space slop into the [0,1] depth the
//     comparison sampler tests.
//   * `receiverDepthBias` — normalised depth, subtracted from the receiver's compare value.
//   * `normalOffsetWorld` — world units, applied along the surface normal BEFORE projection.
//
// Keeping the two metrics separate rather than folding them into one "bias scale" is what makes the
// law projection-independent: an orthographic cascade, a spot cone and one cube face disagree about
// BOTH numbers, in different ways, and each derives them below.

// What the law needs to know about one receiver sample.
struct ShadowBiasInputs
{
    // Both come from the projection-specific derivations below.
    float worldUnitsPerTexel{0.0f};
    float normalizedDepthPerWorldUnit{0.0f};
    // Surface-to-light cosine, already clamped to [0,1] by the caller (a back-facing receiver is
    // not lit and is not asking for a bias).
    float nDotL{1.0f};
    // The PCF kernel's radius in texels, so the offset clears the whole footprint the filter reads
    // rather than only the texel the sample lands in. 0 for a single-tap comparison.
    float filterRadiusTexels{0.0f};
};

// The tunables, passed explicitly rather than read from `constants.hpp`, so the law stays pure and
// a test can sweep them.
//
// PRECONDITION: every field is finite and >= 0. The law SANITISES rather than trusts — a negative
// scale is not a smaller bias but an INVERTED one (the receiver biases into its caster and
// self-shadows everywhere), and a NaN silently unlights whatever it touches.
// `shaders/shadow_bias.glsl` performs the identical sanitisation, so a policy that ever becomes
// runtime-editable cannot mean two different things on the two sides, and the shipped values in
// `render/constants.hpp` are static_asserted against this contract where they are defined.
struct ShadowBiasPolicy
{
    // Multiplies the slope term — the depth a surface at this angle traverses across one texel.
    float slopeScale{1.0f};
    // A floor in TEXELS of world footprint, covering the depth quantisation and interpolation slop
    // that survives even on a surface facing the light head-on.
    float constantTexels{1.0f};
    // Normal offset, in texels of world footprint, at grazing incidence.
    float normalOffsetTexels{1.0f};
    // How far the slope term may run before it is cut off. At grazing incidence tan(theta)
    // diverges, and an unbounded bias detaches the shadow from the caster (peter-panning) far more
    // visibly than the acne it was avoiding.
    float maxSlopeTangent{8.0f};
};

// What to apply at the receiver.
struct ShadowBias
{
    float receiverDepthBias{0.0f}; // normalised depth
    float normalOffsetWorld{0.0f}; // world units, along the surface normal
};

// THE LAW. Mirrored in `shaders/shadow_bias.glsl`, which is what the receiver actually runs — the
// per-fragment inputs (nDotL, and the receiver depth the punctual metrics need) only exist there.
// Both sides are three lines and name each other; the golden values in
// tests/graphics/test_shadow_bias.cpp are the arbiter if they ever disagree.
[[nodiscard]] ShadowBias shadowBias(const ShadowBiasInputs& inputs,
                                    const ShadowBiasPolicy& policy) noexcept;

// --- per-projection metrics -------------------------------------------------------------------
//
// The two numbers the law consumes, derived from what each projection actually is. These are the
// projection-specific half, and the reason the law itself does not need to know which kind of view
// it is serving.

// ORTHOGRAPHIC (directional cascades, world-only cascades, self-shadow layers). Texel footprint is
// constant across the layer — it is the fit's own `worldPerTexel` — and depth is linear in world
// units along the light, so the conversion is one over the fitted span. Both are per VIEW, not per
// fragment, which is why the renderer can compute them once and upload them.
[[nodiscard]] float orthographicNormalizedDepthPerWorldUnit(float depthSpanWorld) noexcept;

// SPOT. Texel footprint grows linearly with the receiver's depth ALONG THE SPOT FORWARD (not its
// radial distance — the projection is planar, so it is the forward component that sets the
// footprint). `texelAngleScale` is the per-view constant `2 * tan(fov/2) / extentTexels`.
[[nodiscard]] float perspectiveWorldUnitsPerTexel(float texelAngleScale,
                                                  float forwardDepthWorld) noexcept;

// SPOT depth conversion, per world unit ALONG THE LIGHT RAY — which is the unit the law's slope
// term is measured in, and not the same as per unit of forward depth.
//
// Two factors, and dropping the second is an error rather than a refinement. The stored value is
// the usual perspective mapping of FORWARD depth, whose precision collapses with distance:
// d(z_ndc)/d(forwardDepth) = near * far / ((far - near) * forwardDepth^2). But `nDotL` measures the
// surface against the local light RAY, and off-axis one world unit along that ray advances forward
// depth by only cos(ray, spotForward). Omitting it over-converts every off-axis fragment — about
// 41% too much normalised bias at 45 degrees off the cone axis, worst at the rim where the texel
// footprint is already widest.
//
// The cosine is DERIVED here from the two depths the caller already has, rather than accepted as a
// third parameter: both come from the same light-to-fragment vector, and a caller passing an
// inconsistent cosine beside them would produce a plausible, wrong bias. `radialDepthWorld` must be
// >= `forwardDepthWorld` (it is a hypotenuse); anything else is rejected as bad input.
[[nodiscard]] float perspectiveNormalizedDepthPerWorldUnit(float nearPlane, float farPlane,
                                                           float forwardDepthWorld,
                                                           float radialDepthWorld) noexcept;

// POINT CUBE FACE. The footprint follows the MAJOR AXIS distance — `max(|x|,|y|,|z|)` of the
// light-to-fragment vector — because that is the axis the face's 90-degree projection divides by.
// Radial distance would overstate the footprint everywhere except dead centre of a face, biasing
// hardest exactly where the face is most oblique and least where it is flattest. `texelAxisScale`
// is the per-view constant `2 / extentTexels` (a 90-degree face has tan(fov/2) == 1).
[[nodiscard]] float cubeFaceWorldUnitsPerTexel(float texelAxisScale, float majorAxisDepth) noexcept;

// POINT depth conversion. The point path does NOT store a perspective depth: `shadow_depth.glsl`
// writes radial distance / range, which is linear, so the conversion is a constant one over range —
// no per-fragment term at all, and no ray cosine either — unlike the spot above, the stored
// quantity is measured along the very ray the law's slope term uses. The comparison stays radial
// even though the footprint above is major-axis; they are answers to different questions,
// deliberately.
[[nodiscard]] float cubeNormalizedDepthPerWorldUnit(float rangeWorld) noexcept;

} // namespace fire_engine
