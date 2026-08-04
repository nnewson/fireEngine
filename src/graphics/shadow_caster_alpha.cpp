#include <fire_engine/graphics/shadow_caster_alpha.hpp>

#include <fire_engine/graphics/material.hpp>

namespace fire_engine
{

ShadowCasterAlpha shadowCasterAlpha(const Material& material) noexcept
{
    // MASK is the only mode whose coverage is per-fragment. BLEND is mapped to Opaque on purpose:
    // its shadow semantics (opaque silhouette / dithered / transmittance) are an open design
    // decision, and the cutout path would settle it by accident — a BLEND material publishes
    // alphaCutoff 0 to the bindless authority, so the mask test could never discard anything and
    // the only visible effect would be the fetch it costs.
    return material.alphaMode() == AlphaMode::Mask ? ShadowCasterAlpha::Masked
                                                   : ShadowCasterAlpha::Opaque;
}

} // namespace fire_engine
