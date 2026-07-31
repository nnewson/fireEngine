#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>

namespace fire_engine
{

// SH-01 shadow diagnostics: the evidence layer the shadow-LOD work is built on
// ([`docs/shadowplans.md`](../../../docs/shadowplans.md) § Milestone 0). A single
// `ProfilePass::Shadow` total cannot say whether a change moved cost between cascades, punctual
// lights, or self-shadowing, and nothing today reports which geometry level a given shadow view
// actually rasterised — so no shadow-LOD change can be judged. Everything here is Vulkan-free and
// headless-testable: the selection side is filled in `graphics/object.cpp` where the decision is
// made, the counting side in `render/shadows.cpp` where a draw meets a specific view.

// Why a shadow draw ended up at the level it did, recorded AT the decision (never reconstructed
// from the level afterwards — level 0 chosen deliberately and level 0 forced are different facts,
// and only the decision site can tell them apart).
enum class ShadowLodReason : std::uint8_t
{
    // The selector ran and this level fit the shadow budget. Includes legitimately choosing level 0
    // (a near or coarse-budget caster) — NOT a fallback.
    Selected,
    // SHADOW LOD is switched off for the frame (`FrameInfo::shadowLodEnabled` — its own switch
    // since SH-03, not the forward `lodEnabled`), so every shadow draw is full detail.
    LodDisabled,
    // The geometry carries a single level, so there is nothing to select. A RIGID mesh below the
    // simplifier's threshold lands here. Cloth no longer does: it is storage-vertex geometry and
    // SH-04 classifies it Deformable, which is checked first and reports DeformableFallback — the
    // point being that its safety must come from the classification and not from the accident of
    // having one level.
    SingleLevel,
    // SH-02 forced fallbacks. Each is a level-0 result the selector could NOT justify, kept
    // distinct from `Selected` so the SH-01 panel reports a forced choice as such — a fallback
    // reported as a deliberate selection would make the diagnostics lie about why detail is high.
    //
    // The shadow view descriptor itself is unusable (degenerate texel size, zero-length or
    // non-finite light direction, non-positive fov/extent/near plane).
    InvalidView,
    // The caster's inputs are unusable: a non-finite transform, invalid world bounds, a
    // non-finite or negative recorded deviation, or a non-finite budget.
    InvalidCaster,
    // A perspective view whose caster reaches the light's near plane (or sits behind it). Texel
    // size diverges there, so no projected error is meaningful.
    NearPlane,
    // `previousLevel` did not name a level of this geometry — a plumbing error (hysteresis state
    // carried across a reassigned punctual slot, or a stale per-caster record). Deliberately NOT
    // clamped into range: clamping would hide the bug and apply one caster's history to another.
    InvalidPreviousLevel,
    // SH-04. The caster deforms after the geometry its error was measured on (skinned,
    // morph-capable or storage-vertex), so no level below full detail can be justified for it. Kept
    // distinct from `LodDisabled` on purpose: that reports a user's toggle, this reports the
    // absence of a valid error model, and a panel that showed one for the other would answer "why
    // is this caster at full detail?" with somebody else's reason.
    DeformableFallback,
    Count
};

inline constexpr std::size_t kShadowLodReasonCount =
    static_cast<std::size_t>(ShadowLodReason::Count);

// "This draw reports no shadow level." Carried on a FORWARD draw (and into its push constants) for
// the ShadowLod debug view, which tints a shaded mesh by the level its depth-only shadow draw
// picked. A mesh that casts no shadow this frame has no level to report, and level 0 would be a
// lie — it would read as "full detail chosen" in exactly the view built to find over-detailed
// shadow casters. The shader mirrors this value; keep the two in lockstep.
inline constexpr std::uint32_t kNoShadowLod = 0xFFFFFFFFu;

[[nodiscard]] std::string_view toString(ShadowLodReason reason) noexcept;

// The five shadow view families, each timed and counted separately.
enum class ShadowViewGroup : std::uint8_t
{
    Cascade,   // the main CSM (all casters)
    WorldOnly, // the second CSM excluding skinned casters, sampled by self-shadowed meshes
    Self,      // per-caster skinned self-shadow (rasterised TWICE — see `countSelection`)
    Spot,
    Point,
    Count
};

inline constexpr std::size_t kShadowViewGroupCount =
    static_cast<std::size_t>(ShadowViewGroup::Count);

[[nodiscard]] std::string_view toString(ShadowViewGroup group) noexcept;

// Per-group physical slot capacity. Keyed by the slot the renderer actually rasterises, not by a
// compacted "active" index, so a diagnostic row always refers to the same physical view across
// frames even as the active count changes.
[[nodiscard]] constexpr std::size_t shadowViewSlotCount(ShadowViewGroup group) noexcept
{
    switch (group)
    {
    case ShadowViewGroup::Cascade:
    case ShadowViewGroup::WorldOnly:
        return kShadowCascadeCount;
    case ShadowViewGroup::Self:
        return static_cast<std::size_t>(kMaxSkinnedSelfShadowCasters);
    case ShadowViewGroup::Spot:
        return static_cast<std::size_t>(kMaxSpotShadowCasters);
    case ShadowViewGroup::Point:
        // Flattened: lightSlot * kCubeFaceCount + face.
        return static_cast<std::size_t>(kMaxPointShadowCasters) * kCubeFaceCount;
    case ShadowViewGroup::Count:
        break;
    }
    return 0;
}

// Start of `group`'s slots in the flat view array.
[[nodiscard]] constexpr std::size_t shadowViewGroupBase(ShadowViewGroup group) noexcept
{
    std::size_t base = 0;
    for (std::size_t g = 0; g < static_cast<std::size_t>(group); ++g)
    {
        base += shadowViewSlotCount(static_cast<ShadowViewGroup>(g));
    }
    return base;
}

inline constexpr std::size_t kShadowViewCount = shadowViewGroupBase(ShadowViewGroup::Count);

// Flat index for (group, slot). `slot` is the physical slot within the group — cascade index,
// self-shadow caster slot, spot light slot, or `lightSlot * kCubeFaceCount + face` for point.
[[nodiscard]] constexpr std::size_t shadowViewIndex(ShadowViewGroup group,
                                                    std::size_t slot) noexcept
{
    return shadowViewGroupBase(group) + slot;
}

[[nodiscard]] constexpr std::size_t shadowPointViewSlot(std::size_t lightSlot,
                                                        std::size_t face) noexcept
{
    return lightSlot * kCubeFaceCount + face;
}

// Which kind of logical identity a group's views carry. The families are not interchangeable: a
// cascade identity can only name a cascade view, so a focus pairing one with the Spot group names
// nothing and must be rejected as malformed rather than searched for forever. WorldOnly maps to
// Cascade because `worldOnly(i)` IS `cascade(i)` — the two passes deliberately share one identity.
[[nodiscard]] constexpr ShadowLogicalViewKind shadowViewKindFor(ShadowViewGroup group) noexcept
{
    switch (group)
    {
    case ShadowViewGroup::Cascade:
    case ShadowViewGroup::WorldOnly:
        return ShadowLogicalViewKind::Cascade;
    case ShadowViewGroup::Self:
        return ShadowLogicalViewKind::Self;
    case ShadowViewGroup::Spot:
        return ShadowLogicalViewKind::Spot;
    case ShadowViewGroup::Point:
        return ShadowLogicalViewKind::Point;
    case ShadowViewGroup::Count:
        break;
    }
    return ShadowLogicalViewKind::Invalid;
}

// LOD histogram bins: levels 0, 1, 2, and "3 or coarser" lumped together (the deep tail carries no
// extra signal for a selection distribution, and a fixed width keeps the row printable).
inline constexpr std::size_t kShadowLodBinCount = 4;

[[nodiscard]] constexpr std::size_t shadowLodBin(std::uint32_t level) noexcept
{
    return level < kShadowLodBinCount - 1 ? level : kShadowLodBinCount - 1;
}

// One rasterised shadow view's frame cost. 64-bit because triangles accumulate across every view a
// caster appears in and these are also summed into scene totals.
//
// `candidate*` counts what was OFFERED to this view — every shadow draw the pass walked for it.
// `drawn*` counts what was actually recorded after the per-view filter (frustum reject, or a
// self-shadow slot belonging to a different caster) and after this view resolved its LOD.
//
// SH-03 changed what `candidateTriangles` MEANS, deliberately. A candidate's triangle count used to
// be known up front, because the level had already been chosen from the camera; now a rejected
// caster is never resolved at all, and asking for its triangles would mean selecting a level for a
// view that will not draw it — the very thing the filter-before-select order exists to avoid. So
// `candidateTriangles` is now the FULL-DETAIL (whole-mesh) triangle count of every draw walked: the
// cost this view would pay with neither culling nor LOD. `candidateTriangles − drawnTriangles` is
// therefore the combined saving of both, not the filter's yield alone. For the filter's yield in
// isolation, use `candidateDraws − drawnDraws`, which is still exactly that.
struct ShadowViewStats
{
    // Times this view was actually rasterised — incremented when the pass BEGINS recording it, so a
    // view that rendered and cleared with an empty draw span still reports as active. Normal views
    // report 1 per frame; the self-shadow families report 2 (first + second depth layer). This, not
    // a candidate count, is what "was this map rendered?" means.
    std::uint64_t rasterPasses{0};
    std::uint64_t candidateDraws{0};
    std::uint64_t drawnDraws{0};
    std::uint64_t candidateTriangles{0};
    std::uint64_t drawnTriangles{0};
    // Selected level of each DRAWN draw. Counted once per logical view: the self-shadow families
    // rasterise the same view twice and doubling a selection distribution would misreport it, so
    // the caller passes `countSelection` only on the first layer.
    std::array<std::uint64_t, kShadowLodBinCount> lodHistogram{};
    // Why each drawn draw got the level it did, counted PER VIEW (SH-03). It used to be a single
    // frame-wide tally taken once per command, which was true only while every view rasterised the
    // one camera-derived choice; now each view resolves its own, so a frame-wide count would be the
    // sum of several unrelated decisions with no way to tell which view forced what. Subject to the
    // same `countSelection` rule as the histogram.
    std::array<std::uint64_t, kShadowLodReasonCount> lodReasons{};
    // WHICH logical view these counters describe, recorded when the view is marked rasterised.
    //
    // A physical slot is not an identity: spot, point and self assignments are compacted in
    // scene-gather order every frame, so slot 1 can be a different light next frame. Rows are still
    // addressed by slot (that is what the renderer rasterises), but anything that must refer to the
    // SAME view across frames — a panel selection, and from slice 5 the tint — has to key on this.
    // Invalid only on a view that never rasterised.
    ShadowLogicalViewId logicalId{};

    // Marks this view rasterised, and states WHICH view it is. The identity is required rather than
    // optional: it is the only chance to record it, and a row that cannot say what it describes
    // cannot be selected reliably later.
    //
    // Returns false and changes NOTHING when the identity is invalid, or when this row already
    // holds a different one. A row is one logical view's counters: two identities rasterising into
    // the same physical slot in one frame would merge their draws, triangles and level
    // distributions and then label the total as whichever came second — a plausible row describing
    // no real view. Repeating the SAME identity is the normal case (a self-shadow slot's two depth
    // layers). The caller must treat a false return as terminal; the counters are unusable evidence
    // either way, and the renderer knows which view it was trying to record.
    [[nodiscard]] bool beginRasterPass(ShadowLogicalViewId view) noexcept;
    // THE FOUR OBSERVATION RULES. Every per-view number in the panel is only readable because these
    // hold together; each one exists because breaking it produced a plausible-looking wrong answer:
    //
    //  1. A view that was RASTERISED reports, even with nothing to draw. `beginRasterPass` runs
    //     before the span is walked, so an empty-but-rendered map — a real cost, and evidence a map
    //     is being rendered for no reason — is visible instead of vanishing.
    //  2. ONE observation per walked draw, carrying the filter's verdict. A separate
    //     add-candidate/add-drawn pair would let a caller record an accepted draw that was never
    //     offered, and `drawn <= candidate` would stop being structural.
    //  3. `accepted` is the FILTER's verdict alone, evaluated before resolution. Nothing else may
    //  be
    //     folded into it, or `candidateDraws - drawnDraws` stops being exactly the cull yield (a
    //     caster that failed to resolve is terminal in the pass, not quietly counted as culled).
    //  4. Selection is counted ONCE per logical view. A self-shadow slot rasterises the same view
    //     twice, so the second layer passes `countSelection = false`: its cost is real, its level
    //     is the same decision, and counting it again would double a distribution.
    //
    // `fullDetailTriangles` is the whole-mesh count (always known); `resolvedTriangles`, `lodLevel`
    // and `reason` describe this view's resolution and are read only when `accepted`, because a
    // rejected draw was never resolved and has no level or reason to report.
    void observe(std::uint64_t fullDetailTriangles, bool accepted, std::uint64_t resolvedTriangles,
                 std::uint32_t lodLevel, ShadowLodReason reason, bool countSelection) noexcept;
    // Rasterised at all this frame — independent of whether anything was drawn into it.
    [[nodiscard]] bool touched() const noexcept
    {
        return rasterPasses != 0;
    }
};

// WHICH view the diagnostics are being read for: one LOGICAL view, or the whole scene.
//
// Per-view selection made the scene rollup insufficient on its own. A rollup answers "how much
// shadow work happened", but the question SH-03 raises is "why did THIS map keep that much
// geometry" — and one caster now holds a different level in every view it appears in, so the sum
// over views is a distribution of unrelated decisions. This names the view being interrogated.
// (SH-03 slice 5 reuses it: the ShadowLod tint needs one view to tint by, for the same reason.)
//
// Keyed by IDENTITY, not by physical slot. Slots are compacted in scene-gather order each frame, so
// a slot-keyed focus silently retargets to whichever light replaced the one you selected — and the
// tint makes that worse than a mislabelled row, because the panel reports a COMPLETED ring frame
// while the tint would sample the CURRENT one, so the two could disagree about which view they
// mean. `group` is still needed: a cascade and its world-only twin deliberately share one logical
// id, and they are different maps with different counters.
struct ShadowViewFocus
{
    // False means the scene rollup. Kept as a flag rather than a sentinel so "no view chosen"
    // cannot be confused with a real group whose views happen to be inactive.
    bool perView{false};
    ShadowViewGroup group{ShadowViewGroup::Cascade};
    ShadowLogicalViewId view{};

    // Names a view that COULD exist. False for the scene rollup (it names no view at all), for an
    // unusable group or identity, and — the case independent checks miss — for a well-formed pair
    // that cannot go together, like a cascade identity in the Spot group. That combination would
    // otherwise look addressable and then never be found, which reads as "this view keeps not
    // rendering" instead of "this selection is malformed".
    //
    // Says nothing about whether the view ran this frame; that is a separate question with a
    // separate answer, because the diagnostics cannot tell a view that is absent this frame from
    // one that is gone for good — that would need scene liveness they do not have.
    [[nodiscard]] bool addressable() const noexcept
    {
        return perView && static_cast<std::size_t>(group) < kShadowViewGroupCount && view.valid() &&
               view.kind() == shadowViewKindFor(group);
    }
};

// A request to focus the view currently occupying one physical slot, parsed from `--shadow-focus`.
//
// A slot is the only handle a person has BEFORE the engine runs — logical identities are allocated
// at load — so the command line has to speak in slots. It is resolved ONCE, at startup, into the
// identity that slot holds, and the identity is what gets stored; the request itself is then
// discarded. Keeping the slot would reintroduce exactly the defect the identity-keyed focus exists
// to prevent: the selection would silently retarget when assignments compact.
struct ShadowViewSlotRequest
{
    ShadowViewGroup group{ShadowViewGroup::Cascade};
    std::size_t slot{0};
};

// Validates one SH-03 calibration override (`--shadow-budget` / `--shadow-ratio`) — the numeric
// half of the flags, kept here so it is headless-testable rather than buried in the renderer.
//
// Returns nothing for anything unusable: empty, malformed, a numeric PREFIX only ("4x"), non-finite
// ("nan"/"inf"), non-positive, or above `maximum`. The caller reports that by name and refuses to
// start — a calibration input that silently became the default would produce a sweep row that reads
// like a measurement of the value that was asked for.
//
// `maximum` is inclusive; the budget has none (pass infinity), the coarsening ratio must be <= 1
// because the selector treats anything outside (0, 1] as an invalid caster and would force LOD0 for
// the whole run while still producing an image and a table.
[[nodiscard]] std::optional<float> parseShadowCalibrationValue(std::string_view text,
                                                               float maximum) noexcept;

// Parses `<group>:<slot>` — `cascade:2`, `world-only:0`, `self:0`, `spot:1` — plus `point:<light>`
// `:<face>` for a cube face (`point:0:4`), since a flat point slot is an implementation detail no
// one should have to compute. Returns nothing for an unknown group, a malformed number, or a slot
// outside the group's capacity; the caller reports that by name rather than falling back, because a
// silent fallback would produce a capture of the wrong view that looks perfectly fine.
[[nodiscard]] std::optional<ShadowViewSlotRequest>
parseShadowViewSlotRequest(std::string_view text) noexcept;

// Where a focused view was found this frame: its counters and the physical slot carrying them (the
// slot is what the row label says, and it can differ from frame to frame for one identity).
struct FocusedShadowView
{
    const ShadowViewStats* stats{nullptr};
    std::size_t slot{0};

    [[nodiscard]] bool found() const noexcept
    {
        return stats != nullptr;
    }
};

// How the per-view LOD decisions MOVED when this frame's levels were committed (SH-03 slice 6).
//
// Counted per (caster, logical view) at COMMIT, so a decision only lands here once the frame it
// belongs to was actually submitted.
//
// TRANSITIONS ARE NOT CHATTER, and conflating them is the mistake this type is shaped to prevent. A
// caster moving away from a light legitimately steps L0 → L1 → L2 and stays there: three
// transitions, no instability. Chatter is a caster REVERSING — L1 → L2 → L1 — which is what a
// viewer sees as a silhouette flickering in place, and what the hysteresis dead band exists to
// stop. A dead band cannot remove legitimate transitions and should not be judged by them.
struct ShadowLodTransitions
{
    // Held the level it already had. The quiet, expected case.
    std::uint64_t held{0};
    // Moved to a different level than the one committed previously. Includes both legitimate
    // motion and chatter; `reversed` below is the part that indicates instability.
    std::uint64_t transitions{0};
    // Moved BACK to the level held before the last one — L1 → L2 → L1. This is the chatter signal:
    // a caster oscillating across a threshold rather than travelling through it. Sustained non-zero
    // counts are what justify widening the dead band (a SMALLER coarsen ratio).
    std::uint64_t reversed{0};
    // Seen for the first time (no previous level for that view). Neither good nor bad: a caster
    // entering a view, or the first frame after a reset. Kept separate so it cannot be mistaken for
    // movement, which is what a plain "decisions" count would do.
    std::uint64_t firstSeen{0};
};

// A whole frame's shadow diagnostics. Held in a frame-indexed ring by the renderer and published
// only when the matching GPU timestamp slot resolves — the overlay is built BEFORE the current
// frame records its shadow pass, so writing live counters next to completed-frame timings would
// mix two different frames (most misleading in exactly the moving-light stability case SH-01
// exists to measure).
struct ShadowFrameStats
{
    std::array<ShadowViewStats, kShadowViewCount> views{};
    // How this frame's committed levels moved (SH-03 slice 6). Filled by the renderer from the
    // resolver after the frame is submitted, so it rides the same publish discipline as the
    // counters above rather than mixing a live number into a completed frame's report.
    ShadowLodTransitions lodMovement{};

    void reset() noexcept;
    [[nodiscard]] ShadowViewStats& view(ShadowViewGroup group, std::size_t slot) noexcept;
    [[nodiscard]] const ShadowViewStats& view(ShadowViewGroup group,
                                              std::size_t slot) const noexcept;
    // Group rollup: the sum over the group's physical slots.
    [[nodiscard]] ShadowViewStats groupTotal(ShadowViewGroup group) const noexcept;
    [[nodiscard]] ShadowViewStats sceneTotal() const noexcept;
    // Slots this frame actually rasterised — the rows worth printing. A map that rendered and
    // cleared with nothing to draw counts; it cost a pass and its emptiness is a finding.
    [[nodiscard]] std::size_t activeViewCount(ShadowViewGroup group) const noexcept;
    // Searches `focus.group` for the view carrying `focus.view`, and reports where it was found.
    //
    // A SEARCH, not an index: the identity may sit in a different physical slot than it did last
    // frame, which is the whole reason the focus is keyed by identity rather than by slot.
    //
    // For a well-formed focus, not found means simply "not present in this frame" — and that is all
    // it means. A view whose light was removed and a view that merely did not rasterise are
    // indistinguishable here (both are valid identities that were not found); separating them would
    // need scene liveness the diagnostics do not have, so neither this function nor its callers may
    // claim whether the view will return. Not found is also deliberately distinct from a zeroed
    // ShadowViewStats, which describes a view that DID run and drew nothing — a finding in its own
    // right. An unaddressable focus, including the scene rollup, is likewise not found; the caller
    // separates that case by asking `focus.addressable()`, because "this selection can never name a
    // view" and "that view is not in this frame" are different things to tell a reader.
    [[nodiscard]] FocusedShadowView focused(ShadowViewFocus focus) const noexcept;
};

} // namespace fire_engine
