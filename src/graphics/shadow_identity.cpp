#include <fire_engine/graphics/shadow_identity.hpp>

#include <atomic>
#include <cstdlib>
#include <limits>

#include <fire_engine/core/log.hpp>

namespace fire_engine
{

ShadowCasterId allocateShadowCasterId() noexcept
{
    // Starts at 1 so ShadowCasterId::Invalid (0) is never handed out.
    static std::atomic<std::uint64_t> next{1};
    const std::uint64_t id = next.fetch_add(1, std::memory_order_relaxed);
    if (id == 0)
    {
        // Wrapped: continuing would hand out Invalid and then REUSE live identities, so a destroyed
        // caster's hysteresis history would be inherited by whatever was created next. Unreachable
        // in practice at ~2^64 allocations; stopping is the only honest response.
        log::error(log::category::render,
                   "shadow caster id counter wrapped — identities are no longer unique");
        std::abort();
    }
    return static_cast<ShadowCasterId>(id);
}

ShadowCasterGeneration nextShadowCasterGeneration(ShadowCasterGeneration current) noexcept
{
    const auto value = static_cast<std::uint64_t>(current);
    if (value == std::numeric_limits<std::uint64_t>::max())
    {
        // Wrapping would return to `First`, where this caster could find history recorded against
        // a chain replaced 2^64 generations ago. Unreachable in practice; stopping is the only
        // honest response, matching the id allocators.
        log::error(log::category::render,
                   "shadow caster generation exhausted — replaced-chain history is no longer "
                   "distinguishable");
        std::abort();
    }
    return static_cast<ShadowCasterGeneration>(value + 1);
}

ShadowLogicalViewId ShadowLogicalViewId::cascade(std::uint32_t index) noexcept
{
    ShadowLogicalViewId view;
    if (index >= kShadowCascadeCount)
    {
        return view; // invalid: there is no such cascade to hold history for
    }
    view.kind_ = ShadowLogicalViewKind::Cascade;
    view.id_ = index;
    return view;
}

ShadowLogicalViewId ShadowLogicalViewId::worldOnly(std::uint32_t cascadeIndex) noexcept
{
    return cascade(cascadeIndex);
}

ShadowLogicalViewId ShadowLogicalViewId::self(std::uint32_t objectId) noexcept
{
    ShadowLogicalViewId view;
    if (objectId == 0)
    {
        return view; // 0 means "never loaded" — not an object that can own a self-shadow map
    }
    view.kind_ = ShadowLogicalViewKind::Self;
    view.id_ = objectId;
    return view;
}

ShadowLogicalViewId ShadowLogicalViewId::spot(NodeId lightNodeId) noexcept
{
    ShadowLogicalViewId view;
    if (lightNodeId == NodeId::Invalid)
    {
        return view;
    }
    view.kind_ = ShadowLogicalViewKind::Spot;
    view.id_ = static_cast<std::uint64_t>(lightNodeId);
    return view;
}

ShadowLogicalViewId ShadowLogicalViewId::point(NodeId lightNodeId, std::uint8_t face) noexcept
{
    ShadowLogicalViewId view;
    if (lightNodeId == NodeId::Invalid || face >= kCubeFaceCount)
    {
        return view;
    }
    view.kind_ = ShadowLogicalViewKind::Point;
    view.id_ = static_cast<std::uint64_t>(lightNodeId);
    view.face_ = face;
    return view;
}

} // namespace fire_engine
