#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include <fire_engine/graphics/gpu_limits.hpp>

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
    // Mesh LOD is switched off for the frame (`FrameInfo::lodEnabled`), so every draw is full
    // detail.
    LodDisabled,
    // The geometry carries a single level, so there is nothing to select. Cloth / storage-vertex
    // geometry lands here: it never builds coarser levels.
    SingleLevel,
    Count
    // SH-04 adds an explicit deformable-fallback reason. Today skinned and morphed geometry DO
    // build and select simplified levels, so they are `Selected` like any rigid mesh — recording
    // that honestly now is what will make SH-04's change visible.
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
        // Flattened: lightSlot * 6 + face.
        return static_cast<std::size_t>(kMaxPointShadowCasters) * 6;
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
// self-shadow caster slot, spot light slot, or `lightSlot * 6 + face` for point.
[[nodiscard]] constexpr std::size_t shadowViewIndex(ShadowViewGroup group,
                                                    std::size_t slot) noexcept
{
    return shadowViewGroupBase(group) + slot;
}

[[nodiscard]] constexpr std::size_t shadowPointViewSlot(std::size_t lightSlot,
                                                        std::size_t face) noexcept
{
    return lightSlot * 6 + face;
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
// self-shadow slot belonging to a different caster). candidate − drawn is therefore the filter's
// yield, and `drawn` is the real GPU cost.
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

    void beginRasterPass() noexcept;
    // ONE operation per draw the pass walked for this view, so `drawn` can never exceed
    // `candidate`: a separate add-candidate/add-drawn pair let a caller (or a test) record an
    // accepted draw that was never offered, which would corrupt the promised
    // `candidate − drawn = filter yield`. `accepted` is the per-view filter's verdict.
    void observe(std::uint64_t triangles, bool accepted, std::uint32_t lodLevel,
                 bool countSelection) noexcept;
    // Rasterised at all this frame — independent of whether anything was drawn into it.
    [[nodiscard]] bool touched() const noexcept
    {
        return rasterPasses != 0;
    }
};

// A whole frame's shadow diagnostics. Held in a frame-indexed ring by the renderer and published
// only when the matching GPU timestamp slot resolves — the overlay is built BEFORE the current
// frame records its shadow pass, so writing live counters next to completed-frame timings would
// mix two different frames (most misleading in exactly the moving-light stability case SH-01
// exists to measure).
struct ShadowFrameStats
{
    std::array<ShadowViewStats, kShadowViewCount> views{};
    // Why each shadow draw got its level, counted at the decision site once per draw (not per view
    // it is replayed into).
    std::array<std::uint64_t, kShadowLodReasonCount> lodReasons{};

    void reset() noexcept;
    void addLodReason(ShadowLodReason reason) noexcept;
    [[nodiscard]] ShadowViewStats& view(ShadowViewGroup group, std::size_t slot) noexcept;
    [[nodiscard]] const ShadowViewStats& view(ShadowViewGroup group,
                                              std::size_t slot) const noexcept;
    // Group rollup: the sum over the group's physical slots.
    [[nodiscard]] ShadowViewStats groupTotal(ShadowViewGroup group) const noexcept;
    [[nodiscard]] ShadowViewStats sceneTotal() const noexcept;
    // Slots this frame actually rasterised — the rows worth printing. A map that rendered and
    // cleared with nothing to draw counts; it cost a pass and its emptiness is a finding.
    [[nodiscard]] std::size_t activeViewCount(ShadowViewGroup group) const noexcept;
};

} // namespace fire_engine
