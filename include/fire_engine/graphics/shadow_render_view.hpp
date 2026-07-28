#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <fire_engine/graphics/shadow_diagnostics.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>
#include <fire_engine/graphics/shadow_view.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

// One shadow view: the matrix the pass rasterises with, the projection descriptor LOD selection
// reasons about, and the stable logical identity hysteresis keys on (SH-03).
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

private:
    // Only the set constructs these, through its family writers, so no call site can assemble an
    // entry whose identity or projection kind contradicts the slot it lands in.
    friend class ShadowRenderViewSet;

    ShadowRenderView(const Mat4& viewProj, const ShadowView& projection,
                     const ShadowLogicalViewId& logicalId) noexcept
        : viewProj_{viewProj},
          projection_{projection},
          logicalId_{logicalId}
    {
    }

    Mat4 viewProj_;
    ShadowView projection_;
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
                                  const ShadowView& projection) noexcept;

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
                               const ShadowView& projection) noexcept;

    // A spot map. `projection` must be perspective.
    [[nodiscard]] bool setSpot(std::size_t slot, NodeId light, const Mat4& viewProj,
                               const ShadowView& projection) noexcept;

    // ONE face of a point map. `projection` must be perspective; the entry lands at the flat slot
    // `shadowPointViewSlot(lightSlot, face)` and its identity carries that same face, so the two
    // cannot disagree. A point light is only usable when ALL SIX faces are accepted, so the caller
    // must treat any single face's rejection as the whole light failing.
    [[nodiscard]] bool setPoint(std::size_t lightSlot, std::uint8_t face, NodeId light,
                                const Mat4& viewProj, const ShadowView& projection) noexcept;

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
                             const Mat4& viewProj, const ShadowView& projection,
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

} // namespace fire_engine
