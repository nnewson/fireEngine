#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include <fire_engine/graphics/shadow_diagnostics.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>
#include <fire_engine/graphics/shadow_view.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

// SH-07: the fitted metrics the RECEIVER's bias law needs from this view — the two quantities
// `graphics/shadow_bias.hpp` consumes, in the form each projection can supply them.
//
// They live here, beside the matrix and the projection descriptor, for the reason SH-03 put those
// two together: a parallel array in the renderer would be a second thing to keep in step with the
// fit, and the failure would be a bias computed from one cascade's footprint against another's
// depth range — plausible everywhere, correct nowhere. Produced from the same intermediates as the
// matrix, immutable afterwards.
//
// What each kind carries, and why they differ rather than sharing one "scale":
//
//   * ORTHOGRAPHIC (cascade / world-only / self): both quantities are constant across the layer, so
//     they are computed once at fit time — the footprint is the fit's own worldPerTexel, and the
//     depth conversion is one over its fitted span.
//   * SPOT: neither is constant. The footprint grows with the receiver's forward depth and the
//   depth
//     conversion falls as its square, so what is stored is the per-view CONSTANTS the receiver
//     needs to derive both per fragment: the texel angle scale, and the frustum's near/far.
//   * POINT: the footprint grows with the receiver's major-axis distance (per-fragment), while the
//     depth conversion is a constant one over range — the point path stores radial distance /
//     range, which is linear.
// One face of a point light's cube: the two things that DO vary per face. Everything else about the
// cube — identity, range, texel scale — belongs to the light and is passed once (setPointLight).
struct ShadowPointFace
{
    Mat4 viewProj;
    ShadowView projection;
};

enum class ShadowViewMetricsKind : std::uint8_t
{
    Orthographic,
    Spot,
    PointLight,
};

class ShadowViewMetrics
{
public:
    // No default construction, deliberately. Zeroed metrics would be accepted by the cascade and
    // self writers and would produce an ACTIVE raster view whose bias silently evaluates to nothing
    // — an inactive slot is already spelled `nullopt`, and the extractors below zero-fill those
    // themselves. The three factories are the only way in.
    // An orthographic layer. `worldUnitsPerTexel` and `depthSpanWorld` come from the same fit.
    [[nodiscard]] static ShadowViewMetrics orthographic(float worldUnitsPerTexel,
                                                        float depthSpanWorld) noexcept;
    // A spot cone. `texelAngleScale` is `2 * tan(fov/2) / extentTexels`.
    [[nodiscard]] static ShadowViewMetrics spot(float texelAngleScale, float nearPlane,
                                                float farPlane) noexcept;
    // A point light. `texelAxisScale` is `2 / extentTexels` (each face is 90 degrees, so
    // tan(fov/2) == 1). Shared by all six faces: the two quantities are properties of the LIGHT,
    // not of which face a fragment lands on.
    [[nodiscard]] static ShadowViewMetrics pointLight(float texelAxisScale,
                                                      float rangeWorld) noexcept;

    // The GPU payload, packed once HERE so the shader's reading of it has a single counterpart.
    // Per kind: orthographic (worldUnitsPerTexel, normalizedDepthPerWorldUnit, 0, 0);
    // spot (texelAngleScale, near, far, 0); point (texelAxisScale, 1/range, 0, 0).
    // `shaders/shadow_bias.glsl` documents the identical layout.
    [[nodiscard]] std::array<float, 4> packed() const noexcept
    {
        return packed_;
    }
    // Which reading of `packed()` is the correct one. Checked against the slot's family by the
    // set's writers: the three layouts share a shape, so a spot triple stored in a cascade slot
    // would be read as (worldPerTexel, normalizedDepthPerWorldUnit) — numbers that are the wrong
    // quantities entirely and yet produce a bias rather than a failure.
    [[nodiscard]] ShadowViewMetricsKind kind() const noexcept
    {
        return kind_;
    }

private:
    ShadowViewMetrics(ShadowViewMetricsKind kind, const std::array<float, 4>& packed) noexcept
        : packed_{packed},
          kind_{kind}
    {
    }

    std::array<float, 4> packed_{};
    ShadowViewMetricsKind kind_{ShadowViewMetricsKind::Orthographic};
};

// One shadow view: the matrix the pass rasterises with, the projection descriptor LOD selection
// reasons about, the fitted bias metrics the receiver converts with (SH-07), and the stable logical
// identity hysteresis keys on (SH-03).
//
// Stored together and READ-ONLY. That does not prove the matrix and the descriptor describe the
// same fit — no type can — but it removes the ways they drift APART: they are supplied together at
// construction, and neither can be mutated afterwards while the other stays put. Producing them
// from the same intermediates is the caller's job, and the family writers below are shaped to make
// that the natural thing to do.
class ShadowRenderView
{
public:
    [[nodiscard]] const Mat4& viewProj() const noexcept
    {
        return viewProj_;
    }
    // May be INVALID on an engaged entry: the view is rendering but its descriptor could not be
    // built, which selection must report as InvalidView rather than treat as inactive.
    [[nodiscard]] const ShadowView& projection() const noexcept
    {
        return projection_;
    }
    // Always valid on an engaged entry — the writers reject anything else.
    [[nodiscard]] const ShadowLogicalViewId& logicalId() const noexcept
    {
        return logicalId_;
    }
    // SH-07: what the receiver converts world-space slop into stored depth with. Supplied at
    // construction from the same fit that produced the matrix.
    [[nodiscard]] const ShadowViewMetrics& biasMetrics() const noexcept
    {
        return biasMetrics_;
    }

private:
    // Only the set constructs these, through its family writers, so no call site can assemble an
    // entry whose identity or projection kind contradicts the slot it lands in.
    friend class ShadowRenderViewSet;

    ShadowRenderView(const Mat4& viewProj, const ShadowView& projection,
                     const ShadowViewMetrics& biasMetrics,
                     const ShadowLogicalViewId& logicalId) noexcept
        : viewProj_{viewProj},
          projection_{projection},
          biasMetrics_{biasMetrics},
          logicalId_{logicalId}
    {
    }

    Mat4 viewProj_;
    ShadowView projection_;
    ShadowViewMetrics biasMetrics_;
    ShadowLogicalViewId logicalId_;
};

// The per-frame set of shadow views, addressed by PHYSICAL (group, slot) — the same addressing
// SH-01's diagnostics use, so an overlay row and an entry here are the same view.
//
// Absent (nullopt) means exactly one thing: this physical view is INACTIVE this frame and will not
// rasterise. An engaged entry whose projection is invalid is a DIFFERENT state — rendering, but
// with a descriptor that could not be built — and stays visible as an InvalidView selection instead
// of being quietly demoted to "inactive", which would hide the bug.
//
// Writers are per family rather than one generic setter. A generic `set(group, slot, view)` can
// only check the numeric slot, so it happily accepts a cascade identity in a spot slot or an
// orthographic projection for a point face — combinations that are not merely invalid but
// meaningless, and which would then be read back as authoritative. Each writer below builds the
// logical identity itself from the slot it is writing, so identity and slot cannot disagree, and
// rejects a projection of the wrong kind.
//
// Indexing is not exposed: a caller computing its own flat index is one arithmetic slip from
// billing one view's matrix to another. Out of range asserts in Dev; in release the read returns
// null and the write is dropped — never clamped into a neighbouring valid slot.
//
// EVERY writer returns whether the view is now active, and every result must be checked. A caller
// that requested a view and did not get one is holding a contradiction: its own counters say the
// pass runs, the set says the slot is inactive, and whichever one the pass consults decides whether
// a shadow map is rasterised from an identity matrix. The renderer's contract is that a rejection
// is TERMINAL — it aborts the frame rather than degrading through it — because everything rejected
// here (a non-finite matrix, an unkeyable identity, a descriptor of the wrong kind) is corrupt
// render input, not a survivable condition. In a Dev build the assertion beside each check fires
// first and the returned `false` is never observed; the return value is what carries the same
// refusal under NDEBUG, where those assertions are compiled out.
class ShadowRenderViewSet
{
public:
    // Disengages every entry. Called once per frame BEFORE anything populates it, so a view that
    // stopped being active cannot linger from the previous frame and be read as current.
    void reset() noexcept;

    // A directional cascade. `projection` must be orthographic; the identity is cascade(index).
    [[nodiscard]] bool setCascade(std::uint32_t index, const Mat4& viewProj,
                                  const ShadowView& projection,
                                  const ShadowViewMetrics& biasMetrics) noexcept;

    // Marks cascade `index`'s world-only pass active. It stores NO view of its own: the world-only
    // slot is an ALIAS of the cascade slot, so `find(WorldOnly, i)` reads the cascade's entry
    // live — matrix, projection and identity (worldOnly(i) IS cascade(i)). A copy would only be
    // equal at the instant it was taken, and a later setCascade would silently leave the two
    // rasterising different fits; aliasing makes "the two passes make the same choice for a rigid
    // caster" hold under any LATER re-fit, in either order, once activated. Activation itself is
    // still ordered: enabling before the cascade exists is rejected (after asserting), because the
    // alias would have nothing to point at.
    //
    // Returns whether the pass is now active, and the result MUST be checked: the set is the
    // authority on whether the world-only pass runs, so a caller that decides from its own state
    // instead would rasterise a pass the set reports as inactive.
    [[nodiscard]] bool enableWorldOnly(std::uint32_t index) noexcept;

    // A per-caster self-shadow layer. `projection` must be orthographic; `objectId` is the owning
    // object (0 is rejected — it means "never loaded").
    [[nodiscard]] bool setSelf(std::size_t slot, std::uint32_t objectId, const Mat4& viewProj,
                               const ShadowView& projection,
                               const ShadowViewMetrics& biasMetrics) noexcept;

    // A spot map. `projection` must be perspective.
    [[nodiscard]] bool setSpot(std::size_t slot, NodeId light, const Mat4& viewProj,
                               const ShadowView& projection,
                               const ShadowViewMetrics& biasMetrics) noexcept;

    // ONE point light's WHOLE CUBE, installed or cleared together.
    //
    // Atomic because everything about a point map is per LIGHT except the six matrices: the
    // identity, the range, the texel-axis scale. A per-face writer let callers supply six
    // independent metrics and six unrelated identities, then had extraction trust face 0 for the
    // light — so two faces could legitimately disagree about the range and nobody would find out,
    // and a cube missing exactly face 0 would upload zeros while still rasterising. Neither state
    // is representable now.
    //
    // `faces` is indexed by cube face, matching kCubemapFaceForward. Every face's matrix must be
    // finite and its projection perspective; `biasMetrics` must be PointLight metrics. If ANY of
    // that fails, all six slots are cleared and the call returns false — a five-face cube is not a
    // usable caster, and the "all six or none" contract was previously only a comment in the
    // renderer.
    [[nodiscard]] bool
    setPointLight(std::size_t lightSlot, NodeId light, const ShadowViewMetrics& biasMetrics,
                  std::span<const ShadowPointFace, kCubeFaceCount> faces) noexcept;

    // The view at a physical slot, or null when the slot is inactive or out of range.
    [[nodiscard]] const ShadowRenderView* find(ShadowViewGroup group,
                                               std::size_t slot) const noexcept;

    // Whether this physical view will rasterise this frame. Says nothing about whether its
    // projection descriptor is usable.
    [[nodiscard]] bool active(ShadowViewGroup group, std::size_t slot) const noexcept;

    // How many slots in the group are active. NOT a dense prefix: active slots may have gaps (a
    // self-shadow slot can be freed while a later one stays occupied), so a caller must iterate
    // every physical slot and skip the inactive ones. Never loop `0..activeCount()`.
    [[nodiscard]] std::size_t activeCount(ShadowViewGroup group) const noexcept;

private:
    [[nodiscard]] static bool inRange(ShadowViewGroup group, std::size_t slot) noexcept;
    // Every family writer funnels through this: it validates the address, then the content
    // (projection kind, identity, finite matrix) in one place, so no writer can forget a check or
    // reject a write without clearing the slot it addressed.
    [[nodiscard]] bool store(ShadowViewGroup group, std::size_t slot, ShadowViewKind expectedKind,
                             ShadowViewMetricsKind expectedMetricsKind, const Mat4& viewProj,
                             const ShadowView& projection, const ShadowViewMetrics& biasMetrics,
                             const ShadowLogicalViewId& logicalId) noexcept;

    // Only the four storing families occupy slots here; the WorldOnly range is never written (see
    // enableWorldOnly — it aliases the cascade), which `store` asserts.
    std::array<std::optional<ShadowRenderView>, kShadowViewCount> views_{};
    // Which cascades' world-only pass runs this frame. A bit, not a view: the view IS the
    // cascade's.
    std::array<bool, kShadowCascadeCount> worldOnlyActive_{};
};

// --- Shader-array extraction -------------------------------------------------------------------
//
// Every matrix any consumer sees is a PROJECTION of the set — there is no second store. Each
// extractor returns a FIXED-SIZE array by value: the sizes are compile-time constants, so accepting
// a dynamic span would admit an undersized-destination state whose only safe handling is to write
// nothing, silently leaving the previous frame's matrices in place. Returning the array removes
// that state rather than defending against it.
//
// The mapping is explicit per family, not a loop over all entries, because the destinations differ
// in both layout and meaning: a generic copy would put a spot matrix in a point slot.

// The ShadowUBO / push-constant matrix array: cascades at kShadowCascadeMatrixBase, spots at
// kShadowSpotMatrixBase, point faces at kShadowPointMatrixBase + the flat slot. Inactive slots are
// identity. Cascades are MANDATORY (the directional pass always runs) and their absence asserts;
// punctual and self slots are legitimately inactive.
//
// World-only contributes no slot: it rasterises with its cascade's matrix, and its entry IS that
// cascade's entry, so there is never a second value to reconcile.
[[nodiscard]] std::array<Mat4, static_cast<std::size_t>(kShadowTotalMatrixCount)>
shadowMatrixArray(const ShadowRenderViewSet& views) noexcept;

// LightUBO::cascadeViewProj — the forward shader's directional lookup.
[[nodiscard]] std::array<Mat4, kShadowCascadeCount>
cascadeViewProjArray(const ShadowRenderViewSet& views) noexcept;

// LightUBO::spotViewProj.
[[nodiscard]] std::array<Mat4, static_cast<std::size_t>(kMaxSpotShadowCasters)>
spotViewProjArray(const ShadowRenderViewSet& views) noexcept;

// LightUBO::selfShadowViewProj — the per-caster tightly-fit self-shadow transforms.
[[nodiscard]] std::array<Mat4, static_cast<std::size_t>(kMaxSkinnedSelfShadowCasters)>
selfShadowViewProjArray(const ShadowRenderViewSet& views) noexcept;

// SH-07 bias metrics, projected out of the same set for the same reason the matrices are: one
// authority, no parallel array to drift from the fit. An inactive slot packs zeros, which the
// receiver's law reads as "no metrics" and answers with no bias — the visible-and-locatable
// failure, not the silent one (see `graphics/shadow_bias.hpp`).
//
// Point metrics are per LIGHT, not per face: the texel-axis scale and the range are properties of
// the light, and all six faces of a cube share them.
[[nodiscard]] std::array<std::array<float, 4>, kShadowCascadeCount>
cascadeBiasMetricsArray(const ShadowRenderViewSet& views) noexcept;
[[nodiscard]] std::array<std::array<float, 4>,
                         static_cast<std::size_t>(kMaxSkinnedSelfShadowCasters)>
selfBiasMetricsArray(const ShadowRenderViewSet& views) noexcept;
[[nodiscard]] std::array<std::array<float, 4>, static_cast<std::size_t>(kMaxSpotShadowCasters)>
spotBiasMetricsArray(const ShadowRenderViewSet& views) noexcept;
[[nodiscard]] std::array<std::array<float, 4>, static_cast<std::size_t>(kMaxPointShadowCasters)>
pointBiasMetricsArray(const ShadowRenderViewSet& views) noexcept;

} // namespace fire_engine
