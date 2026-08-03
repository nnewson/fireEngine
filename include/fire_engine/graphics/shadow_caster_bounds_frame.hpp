#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <fire_engine/graphics/shadow_caster_bounds.hpp>

namespace fire_engine
{

// One frame's shadow-caster bounds, and the single authority on them (SH-06).
//
// Built once per frame by `RenderableScene::gatherShadowCasters`, before the cascade fit; read by
// the fit, by the draw build, and by the diagnostics. That "once" is the point. Computing a
// caster's world bounds means walking its vertices with its current skin and morph weights, so a
// second computation is both expensive and an opportunity to disagree — the draw path used to
// recompute an object-WIDE union and stamp it onto every binding's command, which cost a full
// second pass and threw away exactly the per-binding precision the depth fit needs.
//
// Keyed by (`ShadowCasterId`, `ShadowCasterGeneration`), the same pair the shadow LOD state is
// keyed on, so a reloaded or replaced caster cannot inherit the previous one's bounds.
//
// Lifetime is one frame and nothing more: `reset()` at the start of the prepass, and no entry
// survives into the next frame. Nothing is cached on `Object`, so there is no per-object state
// whose correctness depends on which walk ran first.
class ShadowCasterBoundsFrame
{
public:
    // Start a frame. Keeps capacity — the caster set is stable frame to frame.
    void reset() noexcept;

    // Record one shadow-casting binding. A DUPLICATE key is terminal: two bindings claiming one
    // identity means the shadow state (hysteresis, drawn history, and now bounds) is being shared
    // by casters that are not the same caster, and the resulting shadow would be fitted to one and
    // drawn from the other.
    //
    // An entry whose bounds are invalid (a binding with no vertices) is still RECORDED, so the set
    // of entries matches the set of shadow-casting bindings exactly. Consumers skip invalid bounds
    // explicitly; they are not silently absent.
    void add(const ShadowCasterBounds& caster);

    // The bounds for one caster, or terminal if the key is absent. Terminal rather than a recompute
    // or an empty box, because absence means the prepass and the draw walk disagree about what the
    // scene contains — and a silently empty bound would place a caster at the origin, where it
    // would be fitted and culled against geometry it has nothing to do with.
    [[nodiscard]] const ShadowCasterBounds& require(ShadowCasterId casterId,
                                                    ShadowCasterGeneration generation) const;

    // Non-terminal lookup, for diagnostics that legitimately ask about a caster that may not be in
    // this frame's set.
    [[nodiscard]] const ShadowCasterBounds* find(ShadowCasterId casterId,
                                                 ShadowCasterGeneration generation) const noexcept;

    [[nodiscard]] std::span<const ShadowCasterBounds> entries() const noexcept
    {
        return entries_;
    }
    [[nodiscard]] std::size_t size() const noexcept
    {
        return entries_.size();
    }
    [[nodiscard]] bool empty() const noexcept
    {
        return entries_.empty();
    }

private:
    std::vector<ShadowCasterBounds> entries_;
    // Keyed by the real {id, generation} pair. Packing both 64-bit halves into one integer would
    // make distinct casters collide, and a collision here is not a slow lookup — it is one caster
    // silently receiving another's bounds.
    std::unordered_map<ShadowCasterKey, std::size_t> index_;
};

} // namespace fire_engine
