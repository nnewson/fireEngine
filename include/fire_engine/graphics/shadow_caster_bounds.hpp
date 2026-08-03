#pragma once

#include <cstdint>
#include <functional>

#include <fire_engine/graphics/bounds.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>

namespace fire_engine
{

// How much a caster's reported world bounds can be trusted to contain what will actually rasterise.
//
// The distinction exists because the shadow depth range is about to be FITTED to these bounds
// (SH-06). A range fitted to bounds that understate the geometry clips it, which is the defect the
// fixed extension was hiding — so a caster whose bounds are not authoritative must not be allowed
// to tighten the range as if they were.
enum class ShadowCasterBoundsKind : std::uint8_t
{
    // The bounds were computed from the vertices that will be drawn, in their current pose —
    // including skinning and morph weights. Safe to fit against.
    Exact,
    // The CPU-side vertices are NOT what renders: a compute pass rewrites the storage vertex buffer
    // (cloth). The bounds describe the bind pose, so the deformed geometry can leave them in any
    // direction and by any amount. Usable as a hint, never as a bound.
    Stale,
};

// The identity a caster's bounds are recorded under: the same pair the shadow LOD state is keyed
// on, so a reloaded or replaced caster cannot inherit the previous one's bounds.
//
// A real key rather than the two values packed into one integer. Both halves are 64-bit, so any
// packing is lossy — `(id << 32) | generation` silently equates (id 2, generation 2^32) with
// (id 3, generation 0) and discards the top half of every id. Equality decides membership here;
// the hash below only distributes.
struct ShadowCasterKey
{
    ShadowCasterId casterId{ShadowCasterId::Invalid};
    ShadowCasterGeneration generation{ShadowCasterGeneration::First};

    [[nodiscard]] friend bool operator==(const ShadowCasterKey&, const ShadowCasterKey&) = default;
};

// One shadow caster's world-space extent, as the scene reports it BEFORE any draw commands exist.
//
// The prepass this belongs to is what lets the cascade depth range depend on the casters: the fit
// runs before draw construction (a cascade finalised afterwards would leave the frame's shadow
// matrices describing a different fit than the one the draws were culled against), so it cannot ask
// the draw list what it will contain.
struct ShadowCasterBounds
{
    Bounds3 world{};
    ShadowCasterId casterId{ShadowCasterId::Invalid};
    ShadowCasterGeneration generation{ShadowCasterGeneration::First};
    ShadowCasterBoundsKind kind{ShadowCasterBoundsKind::Stale};

    [[nodiscard]] ShadowCasterKey key() const noexcept
    {
        return ShadowCasterKey{.casterId = casterId, .generation = generation};
    }
};

} // namespace fire_engine

template <>
struct std::hash<fire_engine::ShadowCasterKey>
{
    [[nodiscard]] std::size_t operator()(const fire_engine::ShadowCasterKey& k) const noexcept
    {
        // Both halves are hashed at FULL width and combined, rather than shifted into one word
        // where the wider one would lose bits. Distribution only — equality decides membership, and
        // two colliding keys still coexist and retrieve independently.
        std::size_t seed = std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(k.casterId));
        seed ^= std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(k.generation)) +
                0x9E3779B97F4A7C15ULL + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};
