#include <fire_engine/graphics/vdpm_material.hpp>

#include <algorithm>
#include <cstddef>

#include <fire_engine/graphics/material.hpp>

namespace fire_engine
{

VdpmChannelScales vdpmChannelScales(const Material& material) noexcept
{
    VdpmChannelScales scales;

    // Unlit: the shader skips the BRDF/IBL entirely, so neither the interpolated shading normal nor
    // the tangent frame is ever evaluated — nothing to preserve.
    if (material.unlit())
    {
        scales.normal = 0.0f;
        scales.tangent = 0.0f;
    }

    // The tangent frame is read ONLY to sample a tangent-space normal map (base or clearcoat). With
    // neither map, the frame never reaches a texture fetch, so protecting it is triangles spent on
    // detail nothing samples — the common case for an asset that ships tangents but no normal map.
    const bool hasNormalMap = material.texture(MaterialTextureSlot::Normal).has();
    const bool hasClearcoatNormalMap = material.clearcoat().has_value() &&
                                       material.texture(MaterialTextureSlot::ClearcoatNormal).has();
    if (!hasNormalMap && !hasClearcoatNormalMap)
    {
        scales.tangent = 0.0f;
    }

    // No textures on any slot: UV stretch is invisible because nothing samples UV, so the UV
    // channel need not drive refinement.
    bool anyTexture = false;
    for (std::size_t slot = 0; slot < materialTextureSlotCount; ++slot)
    {
        if (material.texture(static_cast<MaterialTextureSlot>(slot)).has())
        {
            anyTexture = true;
            break;
        }
    }
    if (!anyTexture)
    {
        scales.uv = 0.0f;
    }

    // Gloss: a shading-normal error is far more visible in a sharp specular highlight than on a
    // diffuse surface, so refine the normal channel harder as roughness drops. Linear ramp from 1.0
    // (fully rough) to kVdpmGlossyNormalBoost (fully glossy). Skipped when the channel is already
    // off.
    if (scales.normal > 0.0f)
    {
        const float gloss = std::clamp(1.0f - material.roughness(), 0.0f, 1.0f);
        scales.normal *= 1.0f + ((kVdpmGlossyNormalBoost - 1.0f) * gloss);
    }

    return scales;
}

} // namespace fire_engine
