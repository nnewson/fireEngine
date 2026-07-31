#pragma once

#include <cstdint>
#include <functional>

#include <fire_engine/core/node_id.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>

namespace fire_engine
{

// Identities for shadow-LOD hysteresis (SH-03). Hysteresis is state carried BETWEEN frames, so both
// halves of its key have to name the same thing next frame as they did last frame. A physical slot
// does not: spot and point slots are handed out per frame in whatever order the active light set
// produces, so keying on one applies a departed light's dead band to its replacement.

// A single shadow-casting geometry binding, stable for its lifetime.
//
// `Object::objectId` is NOT sufficient: an Object holds one binding per geometry, so a
// multi-geometry mesh emits several shadow commands sharing one object id, and they would overwrite
// each other's history — the finer caster dragging the coarser one's level around.
//
// A STRONG type, like NodeId: as bare uint64_t aliases these two identity domains are freely
// interchangeable, and `ShadowLogicalViewId::spot(someCasterId)` would compile while naming
// nothing.
enum class ShadowCasterId : std::uint64_t
{
    Invalid = 0,
};

// Terminates rather than wrapping — see allocateNodeId for why reuse is unrecoverable.
[[nodiscard]] ShadowCasterId allocateShadowCasterId() noexcept;

// A caster's LOD-chain generation. Bumped when its shadow geometry is replaced, so history built
// against the old chain cannot be applied to the new one — even when the new chain happens to have
// the same number of levels, where a stored level index would still "fit" while meaning something
// different.
enum class ShadowCasterGeneration : std::uint64_t
{
    First = 0,
};

// Advances a generation, checked. Raw `+ 1` arithmetic would wrap at UINT64_MAX back to `First`,
// where a caster could then find ANCIENT history keyed against a long-replaced chain — the exact
// failure the generation exists to prevent. Same policy as the id allocators: stop rather than
// silently reuse.
//
// NO production caller since SH-04 removed the shadow-proxy setter: a binding's geometry no longer
// changes after construction, so nothing needs to invalidate a caster's dead band yet. Kept, and
// kept tested, because the validated proxy API that replaces that setter is exactly the caller it
// was written for — and a wrap-checked advancer is the wrong thing to delete now and reconstruct
// from memory later.
[[nodiscard]] ShadowCasterGeneration
nextShadowCasterGeneration(ShadowCasterGeneration current) noexcept;

// Which kind of thing a logical view belongs to.
enum class ShadowLogicalViewKind : std::uint8_t
{
    // Deliberately the default, and NOT a usable view: a default-constructed key must not silently
    // mean "cascade 0", which is a real view whose history it would then corrupt.
    Invalid,
    // A directional cascade, identified by its INDEX. The world-only cascade deliberately shares
    // this identity with the full cascade of the same index: the two passes must make the same
    // choice for a given rigid caster, and sharing the key is what guarantees it rather than hoping
    // two independent computations agree.
    Cascade,
    // A per-caster self-shadow map, identified by the OWNING OBJECT. Its physical slot churns frame
    // to frame, but the slot's occupant is chosen by object id, so the object is the stable
    // identity — and both depth passes of one slot therefore share one entry.
    Self,
    // A spot map, identified by the light's node id.
    Spot,
    // ONE face of a point map: the light's node id plus the face index. Kept distinct from Spot
    // rather than sharing a "punctual" kind — a light is only ever one or the other, so the
    // collision would be unreachable today, but that is an invariant held elsewhere and a key type
    // should not depend on it.
    Point,
};

// The stable identity of a shadow view, as opposed to the physical (group, slot) it occupies this
// frame.
//
// ENCAPSULATED, like ShadowView: as a public aggregate it could be assembled with a point face of
// 9, a nonzero face on a cascade, a zero light id, or left default-constructed meaning cascade 0.
// Each of those produces a key that looks valid and names the wrong view — or no view at all.
class ShadowLogicalViewId
{
public:
    // Default is INVALID, never a real view.
    ShadowLogicalViewId() = default;

    [[nodiscard]] static ShadowLogicalViewId cascade(std::uint32_t index) noexcept;
    // The SAME identity the full cascade uses — see ShadowLogicalViewKind::Cascade.
    [[nodiscard]] static ShadowLogicalViewId worldOnly(std::uint32_t cascadeIndex) noexcept;
    // `objectId` is Object::objectId, which is 0 for an object that was never loaded.
    [[nodiscard]] static ShadowLogicalViewId self(std::uint32_t objectId) noexcept;
    [[nodiscard]] static ShadowLogicalViewId spot(NodeId lightNodeId) noexcept;
    [[nodiscard]] static ShadowLogicalViewId point(NodeId lightNodeId, std::uint8_t face) noexcept;

    [[nodiscard]] ShadowLogicalViewKind kind() const noexcept
    {
        return kind_;
    }
    // Cascade: the cascade index. Self: the owning object id. Spot/Point: the light's NodeId.
    [[nodiscard]] std::uint64_t id() const noexcept
    {
        return id_;
    }
    // Point-light cube face; 0 for every other kind.
    [[nodiscard]] std::uint8_t face() const noexcept
    {
        return face_;
    }
    // False for a default-constructed value, and for any factory given out-of-range input.
    [[nodiscard]] bool valid() const noexcept
    {
        return kind_ != ShadowLogicalViewKind::Invalid;
    }

    [[nodiscard]] friend bool operator==(const ShadowLogicalViewId&,
                                         const ShadowLogicalViewId&) = default;

private:
    ShadowLogicalViewKind kind_{ShadowLogicalViewKind::Invalid};
    std::uint64_t id_{0};
    std::uint8_t face_{0};
};

// The persistent hysteresis key: one caster's history, for one LOD-chain generation, in one logical
// view. Never a physical slot.
//
// The generation is part of the KEY rather than a field compared at each lookup: as a field, every
// future call site would have to remember to check it, and the one that forgot would apply stale
// history silently. In the key, a replaced chain simply finds nothing.
class ShadowLodStateKey
{
public:
    ShadowLodStateKey() = default;
    ShadowLodStateKey(ShadowCasterId caster, ShadowCasterGeneration generation,
                      const ShadowLogicalViewId& view) noexcept
        : caster_{caster},
          generation_{generation},
          view_{view}
    {
    }

    [[nodiscard]] ShadowCasterId caster() const noexcept
    {
        return caster_;
    }
    [[nodiscard]] ShadowCasterGeneration generation() const noexcept
    {
        return generation_;
    }
    [[nodiscard]] const ShadowLogicalViewId& view() const noexcept
    {
        return view_;
    }
    [[nodiscard]] bool valid() const noexcept
    {
        return caster_ != ShadowCasterId::Invalid && view_.valid();
    }

    [[nodiscard]] friend bool operator==(const ShadowLodStateKey&,
                                         const ShadowLodStateKey&) = default;

private:
    ShadowCasterId caster_{ShadowCasterId::Invalid};
    ShadowCasterGeneration generation_{ShadowCasterGeneration::First};
    ShadowLogicalViewId view_{};
};

} // namespace fire_engine

template <>
struct std::hash<fire_engine::ShadowLogicalViewId>
{
    [[nodiscard]] std::size_t operator()(const fire_engine::ShadowLogicalViewId& v) const noexcept
    {
        const std::size_t kindAndFace =
            (static_cast<std::size_t>(v.kind()) << 8U) | static_cast<std::size_t>(v.face());
        return std::hash<std::uint64_t>{}(v.id()) ^ (kindAndFace * 0x9E3779B97F4A7C15ULL);
    }
};

template <>
struct std::hash<fire_engine::ShadowLodStateKey>
{
    [[nodiscard]] std::size_t operator()(const fire_engine::ShadowLodStateKey& k) const noexcept
    {
        // Asymmetric combine, so (caster a, view b) and (caster b, view a) tend not to land in one
        // bucket. DISTRIBUTION only: equality decides membership, and no finite hash can promise
        // collision freedom — two colliding keys still coexist and retrieve independently.
        std::size_t seed = std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(k.caster()));
        const auto mix = [&seed](std::size_t value)
        { seed ^= value + 0x9E3779B97F4A7C15ULL + (seed << 6U) + (seed >> 2U); };
        mix(std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(k.generation())));
        mix(std::hash<fire_engine::ShadowLogicalViewId>{}(k.view()));
        return seed;
    }
};
