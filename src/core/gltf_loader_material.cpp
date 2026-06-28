#include <fire_engine/core/gltf_loader.hpp>

#include <optional>
#include <string>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <fire_engine/graphics/material.hpp>

namespace fire_engine
{
namespace
{

UvTransform readUvTransform(const fastgltf::TextureInfo& info) noexcept
{
    UvTransform transform;
    if (info.transform)
    {
        const auto& source = *info.transform;
        transform.offsetX = static_cast<float>(source.uvOffset.x());
        transform.offsetY = static_cast<float>(source.uvOffset.y());
        transform.scaleX = static_cast<float>(source.uvScale.x());
        transform.scaleY = static_cast<float>(source.uvScale.y());
        transform.rotation = static_cast<float>(source.rotation);
    }
    return transform;
}

void applyTextureSlotUv(Material& material, MaterialTextureSlot slot,
                        const fastgltf::TextureInfo& info) noexcept
{
    material.texture(slot).texCoord = static_cast<int>(info.texCoordIndex);
    material.texture(slot).transform = readUvTransform(info);
}

void applyBaseMaterialFields(Material& material, const fastgltf::Material& gltfMat)
{
    material.name(std::string(gltfMat.name));

    const auto& pbr = gltfMat.pbrData;
    material.baseColor({static_cast<float>(pbr.baseColorFactor.x()),
                        static_cast<float>(pbr.baseColorFactor.y()),
                        static_cast<float>(pbr.baseColorFactor.z())});
    material.alpha(static_cast<float>(pbr.baseColorFactor.w()));
    material.metallic(static_cast<float>(pbr.metallicFactor));
    material.roughness(static_cast<float>(pbr.roughnessFactor));

    const float emissiveStrength = static_cast<float>(gltfMat.emissiveStrength);
    material.emissive({static_cast<float>(gltfMat.emissiveFactor.x()) * emissiveStrength,
                       static_cast<float>(gltfMat.emissiveFactor.y()) * emissiveStrength,
                       static_cast<float>(gltfMat.emissiveFactor.z()) * emissiveStrength});

    if (gltfMat.normalTexture.has_value())
    {
        material.normalScale(static_cast<float>(gltfMat.normalTexture.value().scale));
    }
    if (gltfMat.occlusionTexture.has_value())
    {
        material.occlusionStrength(static_cast<float>(gltfMat.occlusionTexture.value().strength));
    }
}

void applyBaseTextureSlotUv(Material& material, const fastgltf::Material& gltfMat)
{
    using Slot = MaterialTextureSlot;
    if (gltfMat.pbrData.baseColorTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::BaseColour, gltfMat.pbrData.baseColorTexture.value());
    }
    if (gltfMat.emissiveTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::Emissive, gltfMat.emissiveTexture.value());
    }
    if (gltfMat.normalTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::Normal, gltfMat.normalTexture.value());
    }
    if (gltfMat.pbrData.metallicRoughnessTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::MetallicRoughness,
                           gltfMat.pbrData.metallicRoughnessTexture.value());
    }
    if (gltfMat.occlusionTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::Occlusion, gltfMat.occlusionTexture.value());
    }
}

void applyTransmission(Material& material, const fastgltf::Material& gltfMat)
{
    const float ior = static_cast<float>(gltfMat.ior);
    if (gltfMat.transmission != nullptr || ior != 1.5f)
    {
        TransmissionParams transmission;
        transmission.ior = ior;
        if (gltfMat.transmission != nullptr)
        {
            transmission.factor = static_cast<float>(gltfMat.transmission->transmissionFactor);
        }
        material.transmission(transmission);
    }
    if (gltfMat.transmission != nullptr && gltfMat.transmission->transmissionTexture.has_value())
    {
        applyTextureSlotUv(material, MaterialTextureSlot::Transmission,
                           gltfMat.transmission->transmissionTexture.value());
    }
}

void applyClearcoat(Material& material, const fastgltf::Material& gltfMat)
{
    if (gltfMat.clearcoat == nullptr)
    {
        return;
    }

    const auto& source = *gltfMat.clearcoat;
    ClearcoatParams clearcoat;
    clearcoat.factor = static_cast<float>(source.clearcoatFactor);
    clearcoat.roughness = static_cast<float>(source.clearcoatRoughnessFactor);
    if (source.clearcoatTexture.has_value())
    {
        applyTextureSlotUv(material, MaterialTextureSlot::Clearcoat,
                           source.clearcoatTexture.value());
    }
    if (source.clearcoatRoughnessTexture.has_value())
    {
        applyTextureSlotUv(material, MaterialTextureSlot::ClearcoatRoughness,
                           source.clearcoatRoughnessTexture.value());
    }
    if (source.clearcoatNormalTexture.has_value())
    {
        const auto& info = source.clearcoatNormalTexture.value();
        applyTextureSlotUv(material, MaterialTextureSlot::ClearcoatNormal, info);
        clearcoat.normalScale = static_cast<float>(info.scale);
    }
    material.clearcoat(clearcoat);
}

void applyVolume(Material& material, const fastgltf::Material& gltfMat)
{
    if (gltfMat.volume == nullptr)
    {
        return;
    }

    const auto& source = *gltfMat.volume;
    VolumeParams volume;
    volume.thicknessFactor = static_cast<float>(source.thicknessFactor);
    volume.attenuationColor = Colour3{static_cast<float>(source.attenuationColor.x()),
                                      static_cast<float>(source.attenuationColor.y()),
                                      static_cast<float>(source.attenuationColor.z())};
    volume.attenuationDistance = static_cast<float>(source.attenuationDistance);
    material.volume(volume);
    if (source.thicknessTexture.has_value())
    {
        applyTextureSlotUv(material, MaterialTextureSlot::Thickness,
                           source.thicknessTexture.value());
    }
}

void applyAlphaFields(Material& material, const fastgltf::Material& gltfMat) noexcept
{
    switch (gltfMat.alphaMode)
    {
    case fastgltf::AlphaMode::Opaque:
        material.alphaMode(AlphaMode::Opaque);
        break;
    case fastgltf::AlphaMode::Mask:
        material.alphaMode(AlphaMode::Mask);
        break;
    case fastgltf::AlphaMode::Blend:
        material.alphaMode(AlphaMode::Blend);
        break;
    }
    material.alphaCutoff(static_cast<float>(gltfMat.alphaCutoff));
    material.doubleSided(gltfMat.doubleSided);
}

} // namespace

Material GltfLoader::loadMaterial(const fastgltf::Asset& asset,
                                  std::optional<std::size_t> materialIndex)
{
    Material material;
    if (!materialIndex.has_value())
    {
        return material;
    }

    const auto& gltfMat = asset.materials[materialIndex.value()];
    applyBaseMaterialFields(material, gltfMat);
    applyBaseTextureSlotUv(material, gltfMat);
    applyTransmission(material, gltfMat);
    applyClearcoat(material, gltfMat);
    applyVolume(material, gltfMat);
    material.unlit(gltfMat.unlit);
    applyAlphaFields(material, gltfMat);

    return material;
}

} // namespace fire_engine
