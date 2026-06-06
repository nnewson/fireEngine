#include <fire_engine/graphics/material_binding.hpp>

#include <cmath>
#include <cstring>

#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

// GPU-side sentinel for "no Beer–Lambert attenuation". Large enough that
// exp(-d / distance) ≈ 1 across any plausible scene depth, but finite so the
// shader can branch on it without an infinity guard. The CPU-side spec default
// is std::numeric_limits<float>::infinity(); this constant is the value we
// shove into the UBO when CPU says "no attenuation".
inline constexpr float kMaxAttenuationDistance = 1.0e6f;

[[nodiscard]]
bool sameTextureSlot(const TextureSlot& a, const TextureSlot& b) noexcept
{
    if (a.has() != b.has())
    {
        return false;
    }
    if (!a.has())
    {
        return true;
    }
    return a.texture->handle() == b.texture->handle();
}

void writeUv(UvXform& dst, const UvTransform& transform) noexcept
{
    dst.offsetScale[0] = transform.offsetX;
    dst.offsetScale[1] = transform.offsetY;
    dst.offsetScale[2] = transform.scaleX;
    dst.offsetScale[3] = transform.scaleY;
    dst.rotation = transform.rotation;
}

} // namespace

MaterialUBO toMaterialUBO(const Material& mat)
{
    MaterialUBO ubo{};
    ubo.diffuseAlpha[0] = mat.baseColor().r();
    ubo.diffuseAlpha[1] = mat.baseColor().g();
    ubo.diffuseAlpha[2] = mat.baseColor().b();
    ubo.diffuseAlpha[3] = mat.alpha();
    ubo.emissiveRoughness[0] = mat.emissive().r();
    ubo.emissiveRoughness[1] = mat.emissive().g();
    ubo.emissiveRoughness[2] = mat.emissive().b();
    ubo.emissiveRoughness[3] = mat.roughness();
    ubo.materialParams[0] = mat.metallic();
    ubo.materialParams[1] = mat.normalScale();
    ubo.materialParams[2] = mat.alphaMode() == AlphaMode::Mask ? mat.alphaCutoff() : 0.0f;
    ubo.materialParams[3] = mat.occlusionStrength();
    using Slot = MaterialTextureSlot;
    ubo.textureFlags[0] = mat.texture(Slot::BaseColour).has() ? 1 : 0;
    ubo.textureFlags[1] = mat.texture(Slot::Emissive).has() ? 1 : 0;
    ubo.textureFlags[2] = mat.texture(Slot::Normal).has() ? 1 : 0;
    ubo.textureFlags[3] = mat.texture(Slot::MetallicRoughness).has() ? 1 : 0;
    ubo.extraFlags[0] = mat.texture(Slot::Occlusion).has() ? 1 : 0;
    ubo.extraFlags[1] = mat.texture(Slot::Occlusion).texCoord;
    ubo.extraFlags[2] = mat.unlit() ? 1 : 0;
    ubo.texCoordIndices[0] = mat.texture(Slot::BaseColour).texCoord;
    ubo.texCoordIndices[1] = mat.texture(Slot::Emissive).texCoord;
    ubo.texCoordIndices[2] = mat.texture(Slot::Normal).texCoord;
    ubo.texCoordIndices[3] = mat.texture(Slot::MetallicRoughness).texCoord;

    // Pack every slot's KHR_texture_transform, indexed by MaterialTextureSlot.
    for (std::size_t i = 0; i < materialTextureSlotCount; ++i)
    {
        writeUv(ubo.uv[i], mat.texture(static_cast<Slot>(i)).transform);
    }

    // Optional extension blocks: value_or({}) reproduces the old always-present
    // defaults (transmission factor 0 / ior 1.5, clearcoat 0, thickness 0, etc.)
    // so the packed UBO is unchanged for materials without the extension.
    const TransmissionParams tr = mat.transmission().value_or(TransmissionParams{});
    ubo.transmissionParams[0] = tr.factor;
    ubo.transmissionParams[1] = mat.texture(Slot::Transmission).has() ? 1.0f : 0.0f;
    ubo.transmissionParams[2] = static_cast<float>(mat.texture(Slot::Transmission).texCoord);
    ubo.transmissionParams[3] = tr.ior;

    const ClearcoatParams cc = mat.clearcoat().value_or(ClearcoatParams{});
    ubo.clearcoatParams[0] = cc.factor;
    ubo.clearcoatParams[1] = cc.roughness;
    ubo.clearcoatParams[2] = cc.normalScale;
    ubo.clearcoatFlags[0] = mat.texture(Slot::Clearcoat).has() ? 1.0f : 0.0f;
    ubo.clearcoatFlags[1] = mat.texture(Slot::ClearcoatRoughness).has() ? 1.0f : 0.0f;
    ubo.clearcoatFlags[2] = mat.texture(Slot::ClearcoatNormal).has() ? 1.0f : 0.0f;
    ubo.clearcoatTexCoords[0] = static_cast<float>(mat.texture(Slot::Clearcoat).texCoord);
    ubo.clearcoatTexCoords[1] = static_cast<float>(mat.texture(Slot::ClearcoatRoughness).texCoord);
    ubo.clearcoatTexCoords[2] = static_cast<float>(mat.texture(Slot::ClearcoatNormal).texCoord);

    const VolumeParams vol = mat.volume().value_or(VolumeParams{});
    ubo.volumeParams[0] = vol.thicknessFactor;
    ubo.volumeParams[1] = mat.texture(Slot::Thickness).has() ? 1.0f : 0.0f;
    ubo.volumeParams[2] = static_cast<float>(mat.texture(Slot::Thickness).texCoord);
    // volumeParams[3] is reserved; the thickness UV rotation is part of the
    // unified uv[Thickness] slot written by the loop above.
    ubo.attenuation[0] = vol.attenuationColor.r();
    ubo.attenuation[1] = vol.attenuationColor.g();
    ubo.attenuation[2] = vol.attenuationColor.b();

    const float attenuationDistance = vol.attenuationDistance;
    ubo.attenuation[3] = attenuationDistance <= 0.0f || !std::isfinite(attenuationDistance)
                             ? kMaxAttenuationDistance
                             : attenuationDistance;
    return ubo;
}

bool materialsEquivalent(const Material& lhs, const Material& rhs)
{
    const MaterialUBO lhsUbo = toMaterialUBO(lhs);
    const MaterialUBO rhsUbo = toMaterialUBO(rhs);
    if (std::memcmp(&lhsUbo, &rhsUbo, sizeof(MaterialUBO)) != 0)
    {
        return false;
    }

    for (std::size_t i = 0; i < materialTextureSlotCount; ++i)
    {
        const auto slot = static_cast<MaterialTextureSlot>(i);
        if (!sameTextureSlot(lhs.texture(slot), rhs.texture(slot)))
        {
            return false;
        }
    }
    return true;
}

MaterialTextureHandles materialTextureHandles(const Material& material) noexcept
{
    MaterialTextureHandles handles{};
    handles.fill(NullTexture);

    for (std::size_t i = 0; i < materialTextureSlotCount; ++i)
    {
        const TextureSlot& slot = material.texture(static_cast<MaterialTextureSlot>(i));
        if (slot.has())
        {
            handles[i] = slot.texture->handle();
        }
    }

    return handles;
}

} // namespace fire_engine
