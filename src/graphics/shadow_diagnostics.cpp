#include <fire_engine/graphics/shadow_diagnostics.hpp>

#include <cassert>

namespace fire_engine
{

std::string_view toString(ShadowLodReason reason) noexcept
{
    switch (reason)
    {
    case ShadowLodReason::Selected:
        return "selected";
    case ShadowLodReason::LodDisabled:
        return "LOD disabled";
    case ShadowLodReason::SingleLevel:
        return "single-level geometry";
    case ShadowLodReason::InvalidView:
        return "invalid shadow view";
    case ShadowLodReason::InvalidCaster:
        return "invalid caster input";
    case ShadowLodReason::NearPlane:
        return "near-plane intersection";
    case ShadowLodReason::InvalidPreviousLevel:
        return "invalid previous level";
    case ShadowLodReason::Count:
        break;
    }
    return "unknown";
}

std::string_view toString(ShadowViewGroup group) noexcept
{
    switch (group)
    {
    case ShadowViewGroup::Cascade:
        return "cascade";
    case ShadowViewGroup::WorldOnly:
        return "world-only";
    case ShadowViewGroup::Self:
        return "self";
    case ShadowViewGroup::Spot:
        return "spot";
    case ShadowViewGroup::Point:
        return "point";
    case ShadowViewGroup::Count:
        break;
    }
    return "unknown";
}

bool ShadowViewStats::beginRasterPass(ShadowLogicalViewId view) noexcept
{
    // VALIDATE FIRST, mutate second. Counting the pass before checking would leave a row that
    // rejected the identity still claiming to have rasterised it.
    assert(view.valid() && "a rasterised shadow view must say which logical view it is");
    if (!view.valid())
    {
        return false;
    }
    // An engaged row belongs to ONE logical view. The self-shadow families legitimately begin twice
    // with the same identity (two depth layers, one view); a DIFFERENT identity arriving at the
    // same physical slot would merge two views' counters under one name.
    assert((!touched() || logicalId == view) &&
           "two logical views cannot share one diagnostic row in a frame");
    if (touched() && !(logicalId == view))
    {
        return false;
    }
    logicalId = view;
    ++rasterPasses;
    return true;
}

void ShadowViewStats::observe(std::uint64_t fullDetailTriangles, bool accepted,
                              std::uint64_t resolvedTriangles, std::uint32_t lodLevel,
                              ShadowLodReason reason, bool countSelection) noexcept
{
    // A draw can only be observed for a view the pass is rasterising. Debug trips at the source;
    // release repairs the count to 1 rather than leaving a view that holds draws yet reports
    // inactive — an inconsistency that would read as a diagnostics bug in the panel.
    assert(rasterPasses != 0 && "observe() before beginRasterPass() for this view");
    if (rasterPasses == 0)
    {
        rasterPasses = 1;
    }
    ++candidateDraws;
    candidateTriangles += fullDetailTriangles;
    if (!accepted)
    {
        // Rejected before resolution, so there is deliberately no level and no reason to record —
        // see the `candidateTriangles` note in the header.
        return;
    }
    ++drawnDraws;
    drawnTriangles += resolvedTriangles;
    if (!countSelection)
    {
        return;
    }
    ++lodHistogram[shadowLodBin(lodLevel)];
    // Debug trips at the source; release drops the sample rather than writing past the array (the
    // `writeMapped` policy — a diagnostic must never be the thing that corrupts memory).
    const auto index = static_cast<std::size_t>(reason);
    assert(index < kShadowLodReasonCount && "an accepted draw must carry a real reason");
    if (index < kShadowLodReasonCount)
    {
        ++lodReasons[index];
    }
}

void ShadowFrameStats::reset() noexcept
{
    *this = ShadowFrameStats{};
}

namespace
{

// Shared by both view() overloads: debug-assert the (group, slot) pair, and in release clamp into
// the group so a bad slot mis-attributes a counter instead of indexing out of bounds.
[[nodiscard]] std::size_t checkedViewIndex(ShadowViewGroup group, std::size_t slot) noexcept
{
    // Validate the UNDERLYING value, not just `!= Count`: any out-of-range enumerator has slot
    // capacity 0, and `capacity - 1` on that would underflow to a huge index in release.
    const bool validGroup = static_cast<std::size_t>(group) < kShadowViewGroupCount;
    assert(validGroup && "shadow view group is not a real group");
    const ShadowViewGroup safeGroup = validGroup ? group : ShadowViewGroup::Cascade;
    const std::size_t capacity = shadowViewSlotCount(safeGroup);
    assert(slot < capacity && "shadow view slot outside its group's capacity");
    return shadowViewIndex(safeGroup, slot < capacity ? slot : capacity - 1);
}

} // namespace

ShadowViewStats& ShadowFrameStats::view(ShadowViewGroup group, std::size_t slot) noexcept
{
    return views[checkedViewIndex(group, slot)];
}

const ShadowViewStats& ShadowFrameStats::view(ShadowViewGroup group,
                                              std::size_t slot) const noexcept
{
    return views[checkedViewIndex(group, slot)];
}

ShadowViewStats ShadowFrameStats::groupTotal(ShadowViewGroup group) const noexcept
{
    ShadowViewStats total{};
    const std::size_t base = shadowViewGroupBase(group);
    for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
    {
        const ShadowViewStats& v = views[base + slot];
        total.rasterPasses += v.rasterPasses;
        total.candidateDraws += v.candidateDraws;
        total.drawnDraws += v.drawnDraws;
        total.candidateTriangles += v.candidateTriangles;
        total.drawnTriangles += v.drawnTriangles;
        for (std::size_t bin = 0; bin < kShadowLodBinCount; ++bin)
        {
            total.lodHistogram[bin] += v.lodHistogram[bin];
        }
        for (std::size_t reason = 0; reason < kShadowLodReasonCount; ++reason)
        {
            total.lodReasons[reason] += v.lodReasons[reason];
        }
    }
    return total;
}

ShadowViewStats ShadowFrameStats::sceneTotal() const noexcept
{
    ShadowViewStats total{};
    for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
    {
        const ShadowViewStats groupSum = groupTotal(static_cast<ShadowViewGroup>(g));
        total.rasterPasses += groupSum.rasterPasses;
        total.candidateDraws += groupSum.candidateDraws;
        total.drawnDraws += groupSum.drawnDraws;
        total.candidateTriangles += groupSum.candidateTriangles;
        total.drawnTriangles += groupSum.drawnTriangles;
        for (std::size_t bin = 0; bin < kShadowLodBinCount; ++bin)
        {
            total.lodHistogram[bin] += groupSum.lodHistogram[bin];
        }
        for (std::size_t reason = 0; reason < kShadowLodReasonCount; ++reason)
        {
            total.lodReasons[reason] += groupSum.lodReasons[reason];
        }
    }
    return total;
}

std::size_t ShadowFrameStats::activeViewCount(ShadowViewGroup group) const noexcept
{
    std::size_t active = 0;
    const std::size_t base = shadowViewGroupBase(group);
    for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
    {
        active += views[base + slot].touched() ? 1 : 0;
    }
    return active;
}

FocusedShadowView ShadowFrameStats::focused(ShadowViewFocus focus) const noexcept
{
    // No assert on an unaddressable focus: it is user state (a panel selection that outlived the
    // light it named), not a producer bug. It simply finds nothing, which is true.
    if (!focus.addressable())
    {
        return FocusedShadowView{};
    }
    const std::size_t base = shadowViewGroupBase(focus.group);
    for (std::size_t slot = 0; slot < shadowViewSlotCount(focus.group); ++slot)
    {
        const ShadowViewStats& stats = views[base + slot];
        // `touched()` first: an untouched row's identity is stale by definition — it is whatever
        // the slot last described, possibly frames ago.
        if (stats.touched() && stats.logicalId == focus.view)
        {
            return FocusedShadowView{.stats = &stats, .slot = slot};
        }
    }
    return FocusedShadowView{};
}

} // namespace fire_engine
