#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>

#include <fire_engine/graphics/bounds.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/shadow_geometry_request.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>
#include <fire_engine/graphics/shadow_render_view.hpp>
#include <fire_engine/graphics/shadow_view.hpp>

namespace fire_engine
{

// One view's answer for one caster: what to draw, and the evidence for it.
struct ResolvedShadowDraw
{
    BufferHandle indexBuffer{NullBuffer};
    std::uint32_t indexCount{0};
    std::size_t level{0};
    ShadowLodReason reason{ShadowLodReason::Count};
    // Projected texel error of the chosen level (0 for LOD0, infinity for a forced fallback), so a
    // forced LOD0 stays distinguishable from a deliberate one in diagnostics.
    float projectedTexels{0.0f};

    [[nodiscard]] bool drawable() const noexcept
    {
        return indexBuffer != NullBuffer && indexCount > 0;
    }
};

// Resolves an unresolved caster against ONE shadow view, with no state at all. The stateful entry
// point below adds caching and hysteresis on top; this is what it calls, and what the tests pin.
//
// `previousLevel` must come from THIS logical view's history or be kNoPreviousShadowLod — feeding
// another view's level in is the plumbing bug `InvalidPreviousLevel` exists to surface.
[[nodiscard]] ResolvedShadowDraw resolveShadowDraw(const ShadowGeometryRequest& request,
                                                   const ShadowView& projection,
                                                   const Bounds3& worldBounds, float budgetTexels,
                                                   ShadowLodHysteresis hysteresis,
                                                   std::size_t previousLevel) noexcept;

// Per-frame resolution cache plus the cross-frame hysteresis history (SH-03).
//
// Two stores, deliberately not one:
//
//  * The FRAME cache exists because several physical views are the same logical view. A cascade and
//    its world-only twin carry one identity, and a self-shadow slot's two depth layers are one
//    view rasterised twice — resolving those separately would let one caster draw two different
//    levels into passes that must agree. It keys on the FULL
//    `(ShadowCasterId, generation, ShadowLogicalViewId)` — the same key as the history, since a
//    cache keyed on the view alone would return one caster's answer for another — so the agreement
//    is a consequence of the lookup rather than a rule someone has to remember.
//
//  * The HISTORY is the dead band, and it must survive the frame. It is STAGED during resolution
//    and only committed once the frame is known to have been submitted: a level recorded for a
//    frame that was then thrown away would apply a dead band to geometry the GPU never saw.
//
// Nothing but a genuinely SELECTED level ever reaches the history. A forced fallback, a disabled
// LOD system or a single-level mesh says nothing about where the caster sits relative to its
// budget, so overwriting a justified level with one would silently discard the evidence the dead
// band is built on.
class ShadowLodResolver
{
public:
    // Drops the frame cache and any staged history that was never committed. Called once per frame
    // before any resolution.
    void beginFrame() noexcept;

    // Resolve `request` for `view`, reusing this frame's answer when the same logical view has
    // already asked. Returns a draw that is always drawable when the request is valid.
    [[nodiscard]] ResolvedShadowDraw resolve(const ShadowGeometryRequest& request,
                                             const ShadowRenderView& view,
                                             const Bounds3& worldBounds, float budgetTexels,
                                             ShadowLodHysteresis hysteresis) noexcept;

    // Promote this frame's staged levels into the history. Call ONLY after the frame has been
    // submitted; entries not seen for `kUnseenHistoryFrames` commits are dropped, so a caster that
    // leaves the scene stops paying rent.
    void commitFrame() noexcept;
    // Throw the staged levels away — the frame was abandoned (a lost swapchain, a failed submit),
    // so nothing it decided should influence the next one.
    void discardFrame() noexcept;

    [[nodiscard]] std::size_t historySize() const noexcept
    {
        return history_.size();
    }
    [[nodiscard]] std::size_t frameCacheSize() const noexcept
    {
        return frameCache_.size();
    }
    // The level this key holds, or kNoPreviousShadowLod. Committed history only — staged entries
    // are not visible here, because they are not yet true.
    [[nodiscard]] std::size_t historyLevel(const ShadowLodStateKey& key) const noexcept;

private:
    struct HistoryEntry
    {
        std::size_t level{0};
        std::uint64_t lastSeenCommit{0};
    };

    // Commits a caster may go unseen before its history is dropped. Small: the state is one level,
    // and re-deriving it costs a single frame of dead band on a caster that came back.
    static constexpr std::uint64_t kUnseenHistoryFrames = 120;

    std::unordered_map<ShadowLodStateKey, ResolvedShadowDraw> frameCache_{};
    std::unordered_map<ShadowLodStateKey, HistoryEntry> history_{};
    std::unordered_map<ShadowLodStateKey, std::size_t> staged_{};
    std::uint64_t commits_{0};
};

} // namespace fire_engine
