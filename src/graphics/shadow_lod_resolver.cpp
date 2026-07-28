#include <fire_engine/graphics/shadow_lod_resolver.hpp>

#include <algorithm>
#include <cassert>

namespace fire_engine
{

namespace
{

// The whole-mesh answer, carrying the reason that explains why nothing was selected.
[[nodiscard]] ResolvedShadowDraw wholeMesh(const ShadowGeometryRequest& request,
                                           ShadowLodReason reason) noexcept
{
    return ResolvedShadowDraw{.indexBuffer = request.baseIndexBuffer,
                              .indexCount = request.baseIndexCount,
                              .level = 0,
                              .reason = reason,
                              .projectedTexels = 0.0f};
}

} // namespace

ResolvedShadowDraw resolveShadowDraw(const ShadowGeometryRequest& request,
                                     const ShadowView& projection, const Bounds3& worldBounds,
                                     float budgetTexels, ShadowLodHysteresis hysteresis,
                                     std::size_t previousLevel) noexcept
{
    // An unfilled request is a producer bug, not a caster property. It resolves to whatever
    // geometry it does carry, reported as InvalidCaster — never as a plausible LOD0.
    assert(request.valid() && "an unresolved shadow command must carry drawable base geometry");
    if (!request.valid())
    {
        return wholeMesh(request, ShadowLodReason::InvalidCaster);
    }
    if (!request.lodEnabled)
    {
        // Checked before the chain: with LOD off the whole mesh is the answer even for a caster
        // that has levels, and reporting SingleLevel there would misattribute the cause.
        return wholeMesh(request, ShadowLodReason::LodDisabled);
    }
    if (request.lods.size() <= 1)
    {
        return wholeMesh(request, ShadowLodReason::SingleLevel);
    }

    const ShadowLodSelection selection =
        selectShadowLod(request.lods, projection, request.worldScale, worldBounds, budgetTexels,
                        hysteresis, previousLevel);
    // A forced fallback names LOD0, but LOD0 is `lods[0]`, whose buffers are the whole mesh —
    // resolve through the same path so a fallback and a deliberate LOD0 bind identical geometry.
    if (selection.level >= request.lods.size())
    {
        assert(false && "selection returned a level outside the chain");
        return wholeMesh(request, selection.reason);
    }
    const GeometryLod& lod = request.lods[selection.level];
    // The chain is validated for its ERROR values (lodChainUsable, inside the selector) but nothing
    // upstream promises the CARRIERS: a level with a null buffer or a zero count is a build that
    // half-failed. Recording it would bind nothing and skip the draw — a silently missing caster in
    // that view's shadow map — so fall back to the whole mesh and say the caster was unusable. The
    // reason must not stay `Selected`: this level was not selectable, whatever the maths said.
    if (lod.indexBuffer == NullBuffer || lod.indexCount == 0)
    {
        assert(false && "a selectable LOD level must carry drawable geometry");
        return wholeMesh(request, ShadowLodReason::InvalidCaster);
    }
    // The level, its buffers and its reason are assigned together, so a diagnostic can never
    // describe a different mesh than the one bound.
    return ResolvedShadowDraw{.indexBuffer = lod.indexBuffer,
                              .indexCount = lod.indexCount,
                              .level = selection.level,
                              .reason = selection.reason,
                              .projectedTexels = selection.projectedTexels};
}

void ShadowLodResolver::beginFrame() noexcept
{
    frameCache_.clear();
    // Staged levels from a frame that was never committed are dropped here rather than carried:
    // they describe decisions the GPU never acted on.
    staged_.clear();
}

std::size_t ShadowLodResolver::historyLevel(const ShadowLodStateKey& key) const noexcept
{
    if (!key.valid())
    {
        return kNoPreviousShadowLod;
    }
    const auto it = history_.find(key);
    return it == history_.end() ? kNoPreviousShadowLod : it->second.level;
}

ResolvedShadowDraw ShadowLodResolver::resolve(const ShadowGeometryRequest& request,
                                              const ShadowRenderView& view,
                                              const Bounds3& worldBounds, float budgetTexels,
                                              ShadowLodHysteresis hysteresis) noexcept
{
    const ShadowLodStateKey key{request.casterId, request.generation, view.logicalId()};
    // An unkeyable request still has to draw — a caster with no identity is a bug in the producer,
    // not a reason to leave a hole in a shadow map — but it takes part in NEITHER store. Caching it
    // would collide with every other unkeyable caster; recording history for it would attach a dead
    // band to something that cannot be looked up again.
    if (!key.valid())
    {
        assert(false && "a shadow caster must be keyable before it can be resolved");
        return resolveShadowDraw(request, view.projection(), worldBounds, budgetTexels, hysteresis,
                                 kNoPreviousShadowLod);
    }

    if (const auto cached = frameCache_.find(key); cached != frameCache_.end())
    {
        // The same logical view asking again: a cascade's world-only twin, or the second depth
        // layer of a self-shadow slot. One decision, reused — not re-derived and hoped to match.
        return cached->second;
    }

    const ResolvedShadowDraw resolved = resolveShadowDraw(
        request, view.projection(), worldBounds, budgetTexels, hysteresis, historyLevel(key));
    frameCache_.emplace(key, resolved);
    if (resolved.reason == ShadowLodReason::Selected)
    {
        // ONLY a selected level is evidence about where this caster sits relative to its budget.
        // Staging a forced fallback would erase a justified level and hand the next frame a dead
        // band derived from a failure.
        staged_[key] = resolved.level;
    }
    return resolved;
}

void ShadowLodResolver::commitFrame() noexcept
{
    ++commits_;
    for (const auto& [key, level] : staged_)
    {
        assert(key.valid() && "an invalid key must never reach the history");
        history_[key] = HistoryEntry{.level = level, .lastSeenCommit = commits_};
    }
    staged_.clear();

    // Age out casters that stopped being resolved — a scene that streams objects in and out would
    // otherwise accumulate a level per caster per view forever.
    if (commits_ <= kUnseenHistoryFrames)
    {
        return;
    }
    const std::uint64_t oldest = commits_ - kUnseenHistoryFrames;
    std::erase_if(history_,
                  [oldest](const auto& entry) { return entry.second.lastSeenCommit < oldest; });
}

void ShadowLodResolver::discardFrame() noexcept
{
    staged_.clear();
}

} // namespace fire_engine
