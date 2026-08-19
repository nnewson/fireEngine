#pragma once

#include <cstdint>

// Which faces a shadow draw keeps — the family's policy, and the EFFECTIVE answer it produces once
// resolved against the caster's own sidedness.
//
// Vulkan-free and here rather than in `render/shadows.hpp` because two consumers need it and only
// one of them may see Vulkan: the recorder turns the effective answer into a `vk::CullModeFlags`,
// and the cache's content descriptor (`graphics/shadow_pass_plan.hpp`) has to record what was
// actually rasterised. Deriving the effective mode twice — once for the draw, once for the
// descriptor — would let a cached map be reused for content it no longer matches.

namespace fire_engine
{

// SH-05: which faces one shadow family keeps. A property of the PASS, not of the pipeline: the
// shadow pipelines declare cull mode dynamic, so this is set at record time and every family must
// name its policy — there is no static fallback to inherit if one forgets.
enum class ShadowFaceCull : std::uint8_t
{
    // Cascade / spot / point: the CASTER decides. Single-sided casters cull front faces (back faces
    // carry the depth, which is what keeps receiver acne off); a double-sided material culls
    // nothing, because front-culling a sheet authored face-on to the light discards the only faces
    // it has and it casts no shadow at all.
    PerCaster,
    // Self-shadow FIRST layer: keep everything, so the first light-facing surface is captured
    // whatever its winding. Was its own pipeline before SH-05 made cull mode dynamic.
    AllFaces,
    // Self-shadow SECOND layer: cull front faces so only back faces rasterise, which is what makes
    // the dual-depth rejection well-founded rather than a coin-flip on marginal fragments.
    BackFacesOnly,
};

// The resolved answer for ONE draw. Only two states exist in the shadow pass — no back-face culling
// variant — so this is the whole codomain, and it is a closed enum rather than a bitmask so a
// content descriptor comparing it cannot be fooled by an equal-but-differently-spelled flag set.
enum class ShadowEffectiveCull : std::uint8_t
{
    None,
    FrontFaces,
};

// SH-05: the family's policy resolved against the caster's sidedness.
//
// Pure, and public for the same reason the pipeline choice is: every shadow pipeline declares cull
// mode dynamic, so this function IS the cull policy, and swapping two of its answers would silently
// restore the defect the item fixed (a double-sided sheet front-culled into casting nothing) or
// break the dual-depth self-shadow layer. Takes the caster's `doubleSided` flag rather than a
// DrawCommand so the mapping can be exercised exhaustively without building a draw.
[[nodiscard]] constexpr ShadowEffectiveCull shadowEffectiveCull(ShadowFaceCull policy,
                                                                bool casterIsDoubleSided) noexcept
{
    switch (policy)
    {
    case ShadowFaceCull::PerCaster:
        // A double-sided caster culls NOTHING. Front-culling one authored face-on to the light
        // discards the only faces it has, and it casts no shadow at all.
        return casterIsDoubleSided ? ShadowEffectiveCull::None : ShadowEffectiveCull::FrontFaces;
    case ShadowFaceCull::AllFaces:
        return ShadowEffectiveCull::None;
    case ShadowFaceCull::BackFacesOnly:
        return ShadowEffectiveCull::FrontFaces;
    }
    // Unreachable for a valid policy; the switch is exhaustive over the enum.
    return ShadowEffectiveCull::FrontFaces;
}

} // namespace fire_engine
