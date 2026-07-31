#include <fire_engine/graphics/shadow_diagnostics.hpp>

#include <cassert>
#include <charconv>
#include <cmath>
#include <system_error>

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
    case ShadowLodReason::DeformableFallback:
        return "deformable caster";
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

namespace
{

// The spellings `--shadow-focus` accepts, matched against the same families the panel labels use.
// `world-only` also accepts `worldonly`, because the hyphen is easy to drop from a shell.
[[nodiscard]] std::optional<ShadowViewGroup> parseShadowViewGroup(std::string_view text) noexcept
{
    if (text == "cascade")
    {
        return ShadowViewGroup::Cascade;
    }
    if (text == "world-only" || text == "worldonly")
    {
        return ShadowViewGroup::WorldOnly;
    }
    if (text == "self")
    {
        return ShadowViewGroup::Self;
    }
    if (text == "spot")
    {
        return ShadowViewGroup::Spot;
    }
    if (text == "point")
    {
        return ShadowViewGroup::Point;
    }
    return std::nullopt;
}

// A whole non-negative number and nothing else. `from_chars` accepts a numeric PREFIX, so "2x"
// would otherwise parse as 2 and the run would focus a view nobody asked for.
[[nodiscard]] std::optional<std::size_t> parseWholeNumber(std::string_view text) noexcept
{
    std::size_t value = 0;
    const char* const end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != end)
    {
        return std::nullopt;
    }
    return value;
}

} // namespace

std::optional<float> parseShadowCalibrationValue(std::string_view text, float maximum) noexcept
{
    float value = 0.0f;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    if (text.empty())
    {
        return std::nullopt;
    }
    const auto result = std::from_chars(begin, end, value);
    // `result.ptr != end` rejects a numeric PREFIX ("4x"), which from_chars would otherwise accept
    // as 4 — a sweep step that measured a different budget than the one written down.
    if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(value) || value <= 0.0f ||
        value > maximum)
    {
        return std::nullopt;
    }
    return value;
}

std::optional<ShadowViewSlotRequest> parseShadowViewSlotRequest(std::string_view text) noexcept
{
    const std::size_t firstColon = text.find(':');
    if (firstColon == std::string_view::npos)
    {
        return std::nullopt;
    }
    const std::optional<ShadowViewGroup> group = parseShadowViewGroup(text.substr(0, firstColon));
    if (!group)
    {
        return std::nullopt;
    }
    const std::string_view rest = text.substr(firstColon + 1);

    std::size_t slot = 0;
    if (const std::size_t secondColon = rest.find(':'); secondColon != std::string_view::npos)
    {
        // `point:<light>:<face>` — the friendly spelling, since a flat point slot is an
        // implementation detail nobody should have to compute. Only point views have a face, so the
        // three-part form is meaningless elsewhere and is rejected rather than half-honoured.
        if (*group != ShadowViewGroup::Point)
        {
            return std::nullopt;
        }
        const std::optional<std::size_t> light = parseWholeNumber(rest.substr(0, secondColon));
        const std::optional<std::size_t> face = parseWholeNumber(rest.substr(secondColon + 1));
        // BOTH validated before flattening. `light * kCubeFaceCount + face` is unsigned arithmetic,
        // so a large enough light index wraps back into the valid range — 2^63 lands on slot 0 —
        // and a request for a nonsense light would silently focus a real one. Same defect, and the
        // same fix, as ShadowRenderViewSet::setPoint.
        if (!light || !face || *face >= kCubeFaceCount ||
            *light >= static_cast<std::size_t>(kMaxPointShadowCasters))
        {
            return std::nullopt;
        }
        slot = shadowPointViewSlot(*light, *face);
    }
    else
    {
        const std::optional<std::size_t> parsed = parseWholeNumber(rest);
        if (!parsed)
        {
            return std::nullopt;
        }
        slot = *parsed;
    }

    if (slot >= shadowViewSlotCount(*group))
    {
        return std::nullopt;
    }
    return ShadowViewSlotRequest{.group = *group, .slot = slot};
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
