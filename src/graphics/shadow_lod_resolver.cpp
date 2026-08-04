#include <fire_engine/graphics/shadow_lod_resolver.hpp>

#include <algorithm>
#include <cassert>
#include <limits>

namespace fire_engine
{

namespace
{

// The whole-mesh answer, carrying the reason that explains why nothing was selected.
//
// `projectedTexels` defaults to 0 because for most fallbacks no projection was attempted and zero
// reads as "nothing measured". A deformable caster is different and passes infinity: there the
// error is not unmeasured but UNBOUNDED, and reporting 0 would read as the most accurate caster in
// the frame — the precise inversion of the truth.
[[nodiscard]] ResolvedShadowDraw wholeMesh(const ShadowGeometryRequest& request,
                                           ShadowLodReason reason,
                                           float projectedTexels = 0.0f) noexcept
{
    return ResolvedShadowDraw{.indexBuffer = request.baseIndexBuffer,
                              .indexCount = request.baseIndexCount,
                              .level = 0,
                              .reason = reason,
                              .projectedTexels = projectedTexels};
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
    if (request.deformation == ShadowCasterDeformation::Deformable)
    {
        // SH-04. Checked BEFORE the chain length, so a deformable caster that happens to carry one
        // level still reports why it may not select — cloth is single-level today, and reporting
        // SingleLevel would make it look safe for the wrong reason, quietly reopening this hole the
        // day storage-vertex geometry grows an LOD chain.
        //
        // Ordered AFTER LodDisabled because when the user has switched shadow LOD off, that is the
        // operative fact about every caster in the frame; a deformable one is not more disabled
        // than the rest. Ordered after InvalidCaster for the same reason: a malformed request is a
        // producer bug and outranks a property of well-formed geometry.
        return wholeMesh(request, ShadowLodReason::DeformableFallback,
                         std::numeric_limits<float>::infinity());
    }
    if (request.alpha == ShadowCasterAlpha::Masked)
    {
        // SH-05. A cutout's silhouette is decided by base-colour alpha sampled through the level's
        // UVs, and no channel the simplifier records measures that: a collapse can hold the surface
        // inside the budget while moving the cutout boundary anywhere. So the shadow LOD for a
        // masked caster BEGINS at level 0, and says so — infinity for the same reason a deformable
        // caster reports it, not 0: the error is unbounded, not measured-as-zero, and a masked
        // caster ranked as the frame's most accurate would invert exactly what is known about it.
        //
        // Ordered AFTER DeformableFallback and before the chain length: a caster that is both
        // deformable and masked gets one whole-mesh answer either way, and reporting the
        // deformation keeps the precedence "most-specific missing evidence first" that SH-04
        // established (a mesh that moves after measurement has no valid error model at all, which
        // is the stronger statement). Before SingleLevel for the reason SH-04 gave: a masked caster
        // with one level must report why it may not select, so its safety cannot come from the
        // accident of the chain's length.
        return wholeMesh(request, ShadowLodReason::AlphaMaskedFallback,
                         std::numeric_limits<float>::infinity());
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
    // Decisions AND their provenance, in one clear: last frame's membership would otherwise let a
    // consumer tint a caster a view has since stopped drawing.
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

namespace
{

// One bit per family. Small and fixed — the group count is a compile-time constant — so the
// membership of every family that drew a caster fits in one map entry.
[[nodiscard]] std::uint32_t groupBit(ShadowViewGroup group) noexcept
{
    const auto index = static_cast<std::size_t>(group);
    assert(index < kShadowViewGroupCount && "shadow view group is not a real group");
    return index < kShadowViewGroupCount ? (1U << index) : 0U;
}

} // namespace

void ShadowLodResolver::noteDrawn(ShadowViewGroup group, const ShadowLodStateKey& key) noexcept
{
    if (!key.valid())
    {
        // An unkeyable caster is in no store (see resolve), so it has no entry to mark — and asking
        // about it later returns nothing, which is the honest answer.
        return;
    }
    const auto it = frameCache_.find(key);
    // A draw can only be marked on a decision that exists. Reaching here without one means a pass
    // drew a caster it never resolved, which the pass itself treats as terminal; creating an entry
    // would manufacture provenance for a level nobody chose.
    assert(it != frameCache_.end() && "a drawn shadow caster must have been resolved first");
    if (it == frameCache_.end())
    {
        return;
    }
    it->second.drawnGroups |= groupBit(group);
}

const ResolvedShadowDraw*
ShadowLodResolver::drawnResolution(ShadowViewGroup group,
                                   const ShadowLodStateKey& key) const noexcept
{
    if (!key.valid())
    {
        return nullptr;
    }
    const auto it = frameCache_.find(key);
    if (it == frameCache_.end() || (it->second.drawnGroups & groupBit(group)) == 0U)
    {
        return nullptr;
    }
    return &it->second.resolved;
}

const ResolvedShadowDraw*
ShadowLodResolver::frameResolution(const ShadowLodStateKey& key) const noexcept
{
    if (!key.valid())
    {
        // An unkeyable caster never entered the cache (see resolve), so there is nothing to find —
        // and no assert: asking about one is a legitimate question with the answer "no".
        return nullptr;
    }
    const auto it = frameCache_.find(key);
    return it == frameCache_.end() ? nullptr : &it->second.resolved;
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
        return cached->second.resolved;
    }

    const ResolvedShadowDraw resolved = resolveShadowDraw(
        request, view.projection(), worldBounds, budgetTexels, hysteresis, historyLevel(key));
    frameCache_.emplace(key, FrameEntry{.resolved = resolved, .drawnGroups = 0});
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
    // Measured HERE, against the level being replaced, because this is the only place both are in
    // hand. Doing it at resolve time would count decisions the frame never submitted.
    lastCommitMovement_ = ShadowLodTransitions{};
    for (const auto& [key, level] : staged_)
    {
        assert(key.valid() && "an invalid key must never reach the history");
        const auto existing = history_.find(key);
        if (existing == history_.end())
        {
            ++lastCommitMovement_.firstSeen;
            history_[key] = HistoryEntry{
                .level = level, .previousLevel = kNoPreviousShadowLod, .lastSeenCommit = commits_};
            continue;
        }
        HistoryEntry& entry = existing->second;
        if (entry.level == level)
        {
            ++lastCommitMovement_.held;
            entry.lastSeenCommit = commits_;
            continue;
        }
        ++lastCommitMovement_.transitions;
        // CHATTER is a reversal that undoes a RECENT transition — the caster is sitting on a
        // threshold, flipping across it. A reversal after a long hold is ordinary motion: a
        // periodic animation walks a caster L1 -> L2 and back every few seconds, and counting that
        // would report the animation as instability and invite widening a dead band that cannot fix
        // it.
        const bool recent = commits_ - entry.levelAdoptedCommit <= kReversalWindowCommits;
        if (entry.previousLevel == level && recent)
        {
            ++lastCommitMovement_.reversed;
        }
        entry = HistoryEntry{.level = level,
                             .previousLevel = entry.level,
                             .levelAdoptedCommit = commits_,
                             .lastSeenCommit = commits_};
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
