#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include <fire_engine/graphics/bounds.hpp>
#include <fire_engine/graphics/shadow_caster_bounds.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// SH-06: the cascade fit, split into the two halves the pipeline actually needs separately.
//
// The order of work is `receiver slice -> stable XY fit -> candidate query -> depth fit -> render
// matrix`. The candidate query — which casters can project into this slice — consumes the XY fit
// and the receiver's light-space extent, and its ANSWER is what the depth fit is supposed to use.
// A single `fitCascade` that produced the matrix in one pass therefore had to bake a depth range
// before it could know which casters existed, which is precisely why a fixed
// `kShadowDepthBackExtend` was needed at all. Keeping the halves apart is what lets the query sit
// between them.

// Everything the receiver fit reads. Grouped into one struct so a caller cannot supply half a
// frame's camera and half of another's: the validity check is over the whole set.
struct CascadeReceiverInput
{
    Vec3 cameraPosition{};
    Vec3 cameraTarget{};
    // MUST already be unit length, to within float rounding; a non-unit direction is rejected, not
    // normalised. Re-normalising is not free: `normalise` of an already-unit vector moves it by an
    // ulp whenever its squared length landed a bit below one, which is enough to change the shipped
    // cascade matrices. The fit refuses to repair a corrupt basis anywhere else, and this is the
    // same rule. The engine's directional light is normalised TWICE on its way here —
    // `Light::toLighting` normalises the node's forward, and `Renderer::updateLightData` normalises
    // again into `directionalLightDir_` — and it is that second one, the renderer's own, that is
    // the authority for what the fit receives; the tolerance is sized for the rounding those leave,
    // not for an intended scale.
    Vec3 lightDirection{};
    float fovRadians{0.0f};
    float aspect{0.0f};
    // Camera-space distances bounding this cascade's slice. `sliceFar > sliceNear > 0`.
    float sliceNear{0.0f};
    float sliceFar{0.0f};
    std::uint32_t shadowMapExtent{0};
};

// The stable half: the slice's light-space footprint and the texel grid it snaps to. Nothing here
// depends on any caster, so it is safe to compute before the candidate query and to hand TO that
// query.
//
// ENCAPSULATED for the same reason `ShadowView` is, and the reason is not theoretical. As a public
// aggregate this had a hole no downstream validator could close cheaply: `lightUp` set equal to
// `lightDirection` (or to zero) is finite, passes every field-wise check worth writing, and sends
// `Mat4::lookAt` to its own fallback up — manufacturing exactly the plausible basis this API says
// it refuses. The alternative, a shared full-carrier validator over finiteness, unit lengths,
// orthogonality, handedness, snapped-centre and U/V-width consistency, would work only for as long
// as every future consumer remembered to call it. Making the fit the single constructor removes the
// question: downstream policies validate their own external inputs and outputs, and TRUST this.
class CascadeReceiverFit
{
public:
    // The only way to obtain one. Returns nullopt for input that is non-finite or degenerate rather
    // than fitting something plausible-looking. `makeViewBasis` is deliberately total — it
    // manufactures a fallback forward and a stable up so a camera pointing at itself still renders
    // — which is right for a camera and wrong here: a NaN aspect or a zero light direction means
    // the render input is corrupt, and a cascade silently fitted around a fabricated basis would
    // shadow the wrong half of the scene with no symptom pointing back at the cause.
    [[nodiscard]] static std::optional<CascadeReceiverFit>
    fit(const CascadeReceiverInput& input) noexcept;

    // The slice and aspect this fit was built from, echoed back. Diagnostics read them from HERE
    // rather than from the caller's own variables: a log that re-derives its inputs can agree with
    // the code that produced it while both disagree with what actually rendered.
    [[nodiscard]] float sliceNear() const noexcept
    {
        return sliceNear_;
    }
    [[nodiscard]] float sliceFar() const noexcept
    {
        return sliceFar_;
    }
    [[nodiscard]] float aspect() const noexcept
    {
        return aspect_;
    }

    // Orthonormal light basis, guaranteed by the factory: `lightDirection` is the (already unit)
    // input unchanged, and `lightUp` is re-orthogonalised against it.
    [[nodiscard]] const Vec3& lightDirection() const noexcept
    {
        return lightDirection_;
    }
    [[nodiscard]] const Vec3& lightRight() const noexcept
    {
        return lightRight_;
    }
    [[nodiscard]] const Vec3& lightUp() const noexcept
    {
        return lightUp_;
    }

    // Centre of the eight slice corners, and that centre snapped to the texel grid along U/V (W is
    // untouched — snapping depth would move the near plane every frame for no benefit).
    [[nodiscard]] const Vec3& frustumCentre() const noexcept
    {
        return frustumCentre_;
    }
    [[nodiscard]] const Vec3& snappedCentre() const noexcept
    {
        return snappedCentre_;
    }
    // Bounding-sphere radius of the slice about `frustumCentre`, quantised upwards so a rotating
    // camera does not resize the cascade every frame. Always finite and > 0.
    [[nodiscard]] float radius() const noexcept
    {
        return radius_;
    }

    // The snapped ortho rectangle in light space: `snappedU/V +/- radius`. These are the bounds the
    // projection maps to clip [-1, 1], so a candidate test can use them directly rather than
    // re-deriving them from the matrix.
    [[nodiscard]] float minU() const noexcept
    {
        return minU_;
    }
    [[nodiscard]] float maxU() const noexcept
    {
        return maxU_;
    }
    [[nodiscard]] float minV() const noexcept
    {
        return minV_;
    }
    [[nodiscard]] float maxV() const noexcept
    {
        return maxV_;
    }

    // W of `frustumCentre`, i.e. the depth about which the legacy fit was symmetric.
    [[nodiscard]] float centreW() const noexcept
    {
        return centreW_;
    }
    // EXACT receiver depth extent, taken from the eight corners rather than `centreW +/- radius`.
    // The bounding sphere is the right shape to snap XY against (it is rotation-invariant) but it
    // is generally looser in depth, and the caster-aware far plane must reach the receiver VOLUME —
    // using the sphere there would push the far plane past every real receiver and waste depth
    // precision. "Generally", not "always": the corner extent can never EXCEED the sphere's, but a
    // slice whose extreme corners happen to sit on the light axis makes them equal, so treat this
    // as a bound, not a strict inequality.
    [[nodiscard]] float receiverMinW() const noexcept
    {
        return receiverMinW_;
    }
    [[nodiscard]] float receiverMaxW() const noexcept
    {
        return receiverMaxW_;
    }
    // World units per shadow-map texel. SH-02's selection error is expressed in these units, so it
    // is returned by the fit rather than recomputed: the two cannot drift.
    [[nodiscard]] float worldPerTexel() const noexcept
    {
        return worldPerTexel_;
    }

private:
    // Private, and there is no invalid state to represent: the factory returns nullopt instead, so
    // a `CascadeReceiverFit` that exists was fitted.
    CascadeReceiverFit() = default;

    float sliceNear_{0.0f};
    float sliceFar_{0.0f};
    float aspect_{0.0f};
    Vec3 lightDirection_{};
    Vec3 lightRight_{};
    Vec3 lightUp_{};
    Vec3 frustumCentre_{};
    Vec3 snappedCentre_{};
    float radius_{0.0f};
    float minU_{0.0f};
    float maxU_{0.0f};
    float minV_{0.0f};
    float maxV_{0.0f};
    float centreW_{0.0f};
    float receiverMinW_{0.0f};
    float receiverMaxW_{0.0f};
    float worldPerTexel_{0.0f};
};

// The caster-dependent half: where the light "camera" sits and how deep it sees. SH-06 replaces
// `fitLegacyCascadeDepth` with a policy that reads candidate caster bounds; the carrier itself does
// not change, so the render path and its tests are unaffected by that swap.
//
// Which policy produced a depth range. Part of the RESULT, not something a log or a panel
// reconstructs, so the reason shown is always the reason for the matrix actually selected.
//
// Three values, not two: `fitLegacyCascadeDepth` is still its own function with its own callers,
// and its direct result is neither caster-aware nor a fallback from anything — labelling it either
// would be false. Only the caster-aware policy can report a fallback, and only where stale geometry
// forced it.
enum class CascadeDepthFitMode : std::uint8_t
{
    // The pre-SH-06 fixed extension, asked for directly.
    LegacyFixedExtension,
    // Fitted to this frame's caster bounds and the receiver volume.
    CasterAware,
    // The caster-aware policy declined: the frame contains a caster whose bounds cannot bound it
    // (cloth), so the legacy range was used instead. An unresolved correctness fallback, not a
    // policy — see `ShadowCasterBoundsKind::Stale`.
    LegacyStaleFallback,
};

// For diagnostics. Named beside the enum so a new mode cannot be added without a name, and so the
// panel and the log read the same words.
[[nodiscard]] constexpr std::string_view cascadeDepthFitModeName(CascadeDepthFitMode mode) noexcept
{
    switch (mode)
    {
    case CascadeDepthFitMode::CasterAware:
        return "caster-aware";
    case CascadeDepthFitMode::LegacyStaleFallback:
        return "legacy (stale-caster fallback)";
    case CascadeDepthFitMode::LegacyFixedExtension:
        break;
    }
    return "legacy (fixed extension)";
}

// ENCAPSULATED, like `CascadeReceiverFit`, and for a reason that only appeared once the modes did:
// the mode must describe THIS matrix. As a public aggregate any caller could set
// `mode = CasterAware` on a legacy matrix, or move a plane without moving the projection, and the
// diagnostics would then report a policy that did not produce what rendered. `placeCaster` also
// consumes one, so it is an input to something else — the same test that made the receiver fit a
// class. Only the two fit functions can build one, and only they can convert a mode.
class CascadeDepthFit
{
public:
    [[nodiscard]] CascadeDepthFitMode mode() const noexcept
    {
        return mode_;
    }
    // Absolute light-space W of the near and far planes (not offsets from the eye). A point at
    // `nearW` lands on Vulkan depth 0 and one at `farW` on depth 1.
    [[nodiscard]] float nearW() const noexcept
    {
        return nearW_;
    }
    [[nodiscard]] float farW() const noexcept
    {
        return farW_;
    }
    [[nodiscard]] const Vec3& lightPosition() const noexcept
    {
        return lightPosition_;
    }
    // `farW - nearW`. The depth range the shadow map has to spend its precision over.
    [[nodiscard]] float viewDepthSpan() const noexcept
    {
        return viewDepthSpan_;
    }
    [[nodiscard]] const Mat4& viewProj() const noexcept
    {
        return viewProj_;
    }

private:
    // The factories, and nothing else. `fitCasterAwareCascadeDepth` needs to relabel a legacy
    // result as `LegacyStaleFallback`, which is the one mode conversion that exists — it happens
    // there, on a matrix that function has in hand, not in a caller.
    friend std::optional<CascadeDepthFit> fitLegacyCascadeDepth(const CascadeReceiverFit&,
                                                                float) noexcept;
    friend std::optional<CascadeDepthFit>
    fitCasterAwareCascadeDepth(const CascadeReceiverFit&, std::span<const ShadowCasterBounds>,
                               float) noexcept;

    CascadeDepthFit() = default;

    CascadeDepthFitMode mode_{CascadeDepthFitMode::LegacyFixedExtension};
    float nearW_{0.0f};
    float farW_{0.0f};
    Vec3 lightPosition_{};
    float viewDepthSpan_{0.0f};
    Mat4 viewProj_{Mat4::identity()};
};

// Pre-SH-06 depth policy, preserved bit-for-bit: centre the range on the slice's bounding sphere
// and extend both ends by `backExtend`. The near extension is the part SH-06 removes — it is a
// fixed world distance standing in for "how far behind the slice a caster might be", which no
// constant can answer for an arbitrary scene.
//
// The receiver is trusted (only `CascadeReceiverFit::fit` can produce one); `backExtend` is not,
// because it arrives from outside. It is the instructive case, too: a negative value always
// produces a FINITE matrix, which is precisely what the view set's non-finite validation waves
// through. A small negative value pulls both planes INSIDE the fitted sphere, clipping casters and
// receivers the cascade is supposed to cover; past `-radius` it reverses the range outright and
// every depth comparison in the map inverts. Neither reports itself, so both are rejected here.
[[nodiscard]] std::optional<CascadeDepthFit>
fitLegacyCascadeDepth(const CascadeReceiverFit& receiver, float backExtend) noexcept;

// How a caster's light-space footprint relates to the cascade's snapped rectangle.
//
// A relation rather than a bool because the candidate query SH-06 will build must be able to reject
// `Outside` and only `Outside`. Edge-touching is classified conservatively as `Straddles`: a caster
// exactly on the boundary may still contribute, and the cost of over-including one is a wasted
// draw, while the cost of wrongly excluding it is a missing shadow nobody can trace.
//
// Note what `Straddles` does NOT mean. Light rays in an orthographic directional map preserve U and
// V, so the part of a caster outside the rectangle cannot shadow any receiver inside it — a
// straddling caster is ordinary, not defective. The classification exists to make cases
// INSPECTABLE, not to accuse them.
enum class CascadeFootprintRelation : std::uint8_t
{
    // The caster had no valid bounds; nothing was placed.
    Invalid,
    // No overlap with the rectangle at all.
    Outside,
    // Wholly within, and not touching an edge.
    Inside,
    // Overlaps, and reaches or crosses at least one edge.
    Straddles,
};

// Where one caster sits relative to one cascade, in the cascade's own light space.
//
// This is the SH-06 evidence type, and deliberately not a verdict: it separates the two ways a
// shadow can go missing, which a single "was it drawn" boolean cannot — clipped in DEPTH by the
// fitted near/far planes, or outside the cascade's XY footprint entirely.
//
// The depth flags describe the CASTER, not the shadow it throws. What a depth clip costs is
// measurable and was measured (SH-06, `ShadowDepthClipDemo`): the shadow pass culls FRONT faces, so
// the surface a caster records is its far side, and the near plane removes a cap from that surface.
// For the fixture's sphere the projected silhouette then shrinks CONCENTRICALLY — 14% smaller
// linearly, 26% in area, matching the analytic cap prediction — rather than acquiring a straight
// edge. That result is about a sphere and a plane perpendicular to the light; asymmetric geometry
// can certainly present a straight projected boundary under the same clip, so do not generalise the
// shape, only the mechanism: a depth clip removes the part of the recorded surface beyond the
// plane.
struct CascadeCasterPlacement
{
    // The caster's world bounds projected onto the cascade's light basis.
    float minU{0.0f};
    float maxU{0.0f};
    float minV{0.0f};
    float maxV{0.0f};
    float minW{0.0f};
    float maxW{0.0f};
    CascadeFootprintRelation footprint{CascadeFootprintRelation::Invalid};
    // Entirely within the fitted depth range.
    bool insideDepth{false};
    // Extends nearer than `nearW` / further than `farW`.
    bool clippedNear{false};
    bool clippedFar{false};
    // Entirely outside the depth range — the whole caster is missing from this map rather than part
    // of it. Distinguished from partial clipping because the two look completely different on
    // screen: a missing shadow versus a shrunken one.
    bool outsideDepth{false};
};

// Whether a caster's footprint overlaps the cascade's rectangle. DEPTH-INDEPENDENT, deliberately:
// the question is about U and V only, and the depth policy has to ask it BEFORE it has a depth
// range — it is choosing the range. Building a throwaway legacy fit to answer it would have made
// the caster-aware path depend on `kShadowDepthBackExtend`, the very thing it retires, and a bad
// extension would then have failed a fit that never used one.
//
// Non-finite or invalid bounds report `Invalid` rather than `Outside`: NaN comparisons are all
// false, so a corrupt box would otherwise look like a caster that is simply elsewhere, and the
// depth range would tighten around geometry it never accounted for.
[[nodiscard]] CascadeFootprintRelation classifyFootprint(const CascadeReceiverFit& receiver,
                                                         const Bounds3& casterBounds) noexcept;

// Pure: no view set, no draw list, no GPU state — the geometric relationship only. Slice 4's
// candidate query is expected to be built from this same function, so a diagnostic and the policy
// it justifies cannot disagree about where a caster was.
//
// An INVALID or non-finite bounds yields a placement with every flag false, `Invalid` footprint and
// zero extents: a caster with no usable bounds has no position to report, and inventing one from
// the default min/max sentinels would place it at infinity.
[[nodiscard]] CascadeCasterPlacement placeCaster(const CascadeReceiverFit& receiver,
                                                 const CascadeDepthFit& depth,
                                                 const Bounds3& casterBounds) noexcept;

// SH-06's depth policy: fit the light-space near/far planes to the casters that can actually shadow
// this cascade, instead of extending a fixed distance in both directions and hoping.
//
// The near plane is the one that matters. The shadow pass culls FRONT faces, so a caster records
// its far side; the near plane removes a cap from that surface, and the shadow shrinks by exactly
// what was removed (measured on `ShadowDepthClipDemo`: 14% linearly, 26% by area). Reaching back to
// the furthest-upstream caster is what stops that. The far plane only has to cover the RECEIVER
// volume — geometry beyond it is further from the light than every receiver in this slice, so it
// cannot shadow anything here, and pushing the plane out to include it would spend depth precision
// on nothing.
//
// Candidates are casters whose light-space footprint is not `Outside` this cascade's rectangle:
// light rays preserve U and V in an orthographic map, so a caster outside the rectangle cannot
// shadow a receiver inside it and must not widen the range. Casters with invalid bounds are
// skipped; they have no extent to fit to.
//
// STALE casters (cloth: a compute pass rewrites the vertices the bounds were measured from) make
// the whole thing undecidable — their box neither bounds the geometry nor establishes which cascade
// it affects — so a single one anywhere in the frame makes this return the LEGACY range, marked
// `LegacyStaleFallback`. "Ignore them and fit the rest" is not the safe reading: it can produce a
// range NARROWER than one covering the cloth, which clips it, arriving at the same defect from the
// other side.
//
// `backExtend` is only used for that fallback. Returns nullopt on the same terms as
// `fitLegacyCascadeDepth`.
[[nodiscard]] std::optional<CascadeDepthFit>
fitCasterAwareCascadeDepth(const CascadeReceiverFit& receiver,
                           std::span<const ShadowCasterBounds> casters, float backExtend) noexcept;

} // namespace fire_engine
