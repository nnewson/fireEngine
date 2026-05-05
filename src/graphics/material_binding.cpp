#include <fire_engine/graphics/material_binding.hpp>

#include <cmath>
#include <cstring>

#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]]
bool sameTextureSlot(bool hasA, bool hasB, const Texture& texA, const Texture& texB) noexcept
{
    if (hasA != hasB)
    {
        return false;
    }

    if (!hasA)
    {
        return true;
    }

    return texA.handle() == texB.handle();
}

void packUv(float* dst, const UvTransform& transform) noexcept
{
    dst[0] = transform.offsetX;
    dst[1] = transform.offsetY;
    dst[2] = transform.scaleX;
    dst[3] = transform.scaleY;
}

} // namespace

MaterialUBO toMaterialUBO(const Material& mat)
{
    MaterialUBO ubo{};
    ubo.diffuseAlpha[0] = mat.diffuse().r();
    ubo.diffuseAlpha[1] = mat.diffuse().g();
    ubo.diffuseAlpha[2] = mat.diffuse().b();
    ubo.diffuseAlpha[3] = mat.alpha();
    ubo.emissiveRoughness[0] = mat.emissive().r();
    ubo.emissiveRoughness[1] = mat.emissive().g();
    ubo.emissiveRoughness[2] = mat.emissive().b();
    ubo.emissiveRoughness[3] = mat.roughness();
    ubo.materialParams[0] = mat.metallic();
    ubo.materialParams[1] = mat.normalScale();
    ubo.materialParams[2] = mat.alphaMode() == AlphaMode::Mask ? mat.alphaCutoff() : 0.0f;
    ubo.materialParams[3] = mat.occlusionStrength();
    ubo.textureFlags[0] = mat.hasTexture() ? 1 : 0;
    ubo.textureFlags[1] = mat.hasEmissiveTexture() ? 1 : 0;
    ubo.textureFlags[2] = mat.hasNormalTexture() ? 1 : 0;
    ubo.textureFlags[3] = mat.hasMetallicRoughnessTexture() ? 1 : 0;
    ubo.extraFlags[0] = mat.hasOcclusionTexture() ? 1 : 0;
    ubo.extraFlags[1] = mat.occlusionTexCoord();
    ubo.extraFlags[2] = mat.unlit() ? 1 : 0;
    ubo.texCoordIndices[0] = mat.baseColorTexCoord();
    ubo.texCoordIndices[1] = mat.emissiveTexCoord();
    ubo.texCoordIndices[2] = mat.normalTexCoord();
    ubo.texCoordIndices[3] = mat.metallicRoughnessTexCoord();

    packUv(ubo.uvBaseColor, mat.baseColorUvTransform());
    packUv(ubo.uvEmissive, mat.emissiveUvTransform());
    packUv(ubo.uvNormal, mat.normalUvTransform());
    packUv(ubo.uvMetallicRoughness, mat.metallicRoughnessUvTransform());
    packUv(ubo.uvOcclusion, mat.occlusionUvTransform());
    packUv(ubo.uvTransmission, mat.transmissionUvTransform());
    ubo.uvRotations[0] = mat.baseColorUvTransform().rotation;
    ubo.uvRotations[1] = mat.emissiveUvTransform().rotation;
    ubo.uvRotations[2] = mat.normalUvTransform().rotation;
    ubo.uvRotations[3] = mat.metallicRoughnessUvTransform().rotation;
    ubo.uvRotationsExtra[0] = mat.occlusionUvTransform().rotation;
    ubo.uvRotationsExtra[1] = mat.transmissionUvTransform().rotation;

    ubo.transmissionParams[0] = mat.transmissionFactor();
    ubo.transmissionParams[1] = mat.hasTransmissionTexture() ? 1.0f : 0.0f;
    ubo.transmissionParams[2] = static_cast<float>(mat.transmissionTexCoord());
    ubo.transmissionParams[3] = mat.ior();

    ubo.clearcoatParams[0] = mat.clearcoatFactor();
    ubo.clearcoatParams[1] = mat.clearcoatRoughness();
    ubo.clearcoatParams[2] = mat.clearcoatNormalScale();
    ubo.clearcoatFlags[0] = mat.hasClearcoatTexture() ? 1.0f : 0.0f;
    ubo.clearcoatFlags[1] = mat.hasClearcoatRoughnessTexture() ? 1.0f : 0.0f;
    ubo.clearcoatFlags[2] = mat.hasClearcoatNormalTexture() ? 1.0f : 0.0f;
    ubo.clearcoatTexCoords[0] = static_cast<float>(mat.clearcoatTexCoord());
    ubo.clearcoatTexCoords[1] = static_cast<float>(mat.clearcoatRoughnessTexCoord());
    ubo.clearcoatTexCoords[2] = static_cast<float>(mat.clearcoatNormalTexCoord());
    packUv(ubo.uvClearcoat, mat.clearcoatUvTransform());
    packUv(ubo.uvClearcoatRoughness, mat.clearcoatRoughnessUvTransform());
    packUv(ubo.uvClearcoatNormal, mat.clearcoatNormalUvTransform());
    ubo.clearcoatRotations[0] = mat.clearcoatUvTransform().rotation;
    ubo.clearcoatRotations[1] = mat.clearcoatRoughnessUvTransform().rotation;
    ubo.clearcoatRotations[2] = mat.clearcoatNormalUvTransform().rotation;

    ubo.volumeParams[0] = mat.thicknessFactor();
    ubo.volumeParams[1] = mat.hasThicknessTexture() ? 1.0f : 0.0f;
    ubo.volumeParams[2] = static_cast<float>(mat.thicknessTexCoord());
    ubo.volumeParams[3] = mat.thicknessUvTransform().rotation;
    ubo.attenuation[0] = mat.attenuationColor().r();
    ubo.attenuation[1] = mat.attenuationColor().g();
    ubo.attenuation[2] = mat.attenuationColor().b();

    const float attenuationDistance = mat.attenuationDistance();
    ubo.attenuation[3] = attenuationDistance <= 0.0f || !std::isfinite(attenuationDistance)
                             ? 1.0e6f
                             : attenuationDistance;
    packUv(ubo.uvThickness, mat.thicknessUvTransform());
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

    return sameTextureSlot(lhs.hasTexture(), rhs.hasTexture(), lhs.texture(), rhs.texture()) &&
           sameTextureSlot(lhs.hasEmissiveTexture(), rhs.hasEmissiveTexture(),
                           lhs.emissiveTexture(), rhs.emissiveTexture()) &&
           sameTextureSlot(lhs.hasNormalTexture(), rhs.hasNormalTexture(), lhs.normalTexture(),
                           rhs.normalTexture()) &&
           sameTextureSlot(lhs.hasMetallicRoughnessTexture(), rhs.hasMetallicRoughnessTexture(),
                           lhs.metallicRoughnessTexture(), rhs.metallicRoughnessTexture()) &&
           sameTextureSlot(lhs.hasOcclusionTexture(), rhs.hasOcclusionTexture(),
                           lhs.occlusionTexture(), rhs.occlusionTexture()) &&
           sameTextureSlot(lhs.hasTransmissionTexture(), rhs.hasTransmissionTexture(),
                           lhs.transmissionTexture(), rhs.transmissionTexture()) &&
           sameTextureSlot(lhs.hasClearcoatTexture(), rhs.hasClearcoatTexture(),
                           lhs.clearcoatTexture(), rhs.clearcoatTexture()) &&
           sameTextureSlot(lhs.hasClearcoatRoughnessTexture(), rhs.hasClearcoatRoughnessTexture(),
                           lhs.clearcoatRoughnessTexture(), rhs.clearcoatRoughnessTexture()) &&
           sameTextureSlot(lhs.hasClearcoatNormalTexture(), rhs.hasClearcoatNormalTexture(),
                           lhs.clearcoatNormalTexture(), rhs.clearcoatNormalTexture()) &&
           sameTextureSlot(lhs.hasThicknessTexture(), rhs.hasThicknessTexture(),
                           lhs.thicknessTexture(), rhs.thicknessTexture());
}

MaterialTextureHandles materialTextureHandles(const Material& material) noexcept
{
    MaterialTextureHandles handles{};
    handles.fill(NullTexture);

    if (material.hasTexture())
    {
        handles[slotIndex(MaterialTextureSlot::BaseColour)] = material.texture().handle();
    }
    if (material.hasEmissiveTexture())
    {
        handles[slotIndex(MaterialTextureSlot::Emissive)] = material.emissiveTexture().handle();
    }
    if (material.hasNormalTexture())
    {
        handles[slotIndex(MaterialTextureSlot::Normal)] = material.normalTexture().handle();
    }
    if (material.hasMetallicRoughnessTexture())
    {
        handles[slotIndex(MaterialTextureSlot::MetallicRoughness)] =
            material.metallicRoughnessTexture().handle();
    }
    if (material.hasOcclusionTexture())
    {
        handles[slotIndex(MaterialTextureSlot::Occlusion)] = material.occlusionTexture().handle();
    }
    if (material.hasTransmissionTexture())
    {
        handles[slotIndex(MaterialTextureSlot::Transmission)] =
            material.transmissionTexture().handle();
    }
    if (material.hasClearcoatTexture())
    {
        handles[slotIndex(MaterialTextureSlot::Clearcoat)] = material.clearcoatTexture().handle();
    }
    if (material.hasClearcoatRoughnessTexture())
    {
        handles[slotIndex(MaterialTextureSlot::ClearcoatRoughness)] =
            material.clearcoatRoughnessTexture().handle();
    }
    if (material.hasClearcoatNormalTexture())
    {
        handles[slotIndex(MaterialTextureSlot::ClearcoatNormal)] =
            material.clearcoatNormalTexture().handle();
    }
    if (material.hasThicknessTexture())
    {
        handles[slotIndex(MaterialTextureSlot::Thickness)] = material.thicknessTexture().handle();
    }

    return handles;
}

} // namespace fire_engine
