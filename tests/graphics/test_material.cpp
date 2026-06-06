#include <gtest/gtest.h>

#include <cmath>

#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/material_binding.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/render/ubo.hpp>

using namespace fire_engine;

using Slot = MaterialTextureSlot;

TEST(Material, DefaultConstructionUsesPbrDefaults)
{
    Material mat;

    EXPECT_TRUE(mat.name().empty());
    EXPECT_FALSE(mat.texture(Slot::BaseColour).has());
    EXPECT_FALSE(mat.texture(Slot::Emissive).has());
    EXPECT_FALSE(mat.texture(Slot::Normal).has());
    EXPECT_FALSE(mat.texture(Slot::MetallicRoughness).has());
    EXPECT_FALSE(mat.texture(Slot::Occlusion).has());

    EXPECT_FLOAT_EQ(mat.baseColor().r(), 0.0f);
    EXPECT_FLOAT_EQ(mat.baseColor().g(), 0.0f);
    EXPECT_FLOAT_EQ(mat.baseColor().b(), 0.0f);
    EXPECT_FLOAT_EQ(mat.emissive().r(), 0.0f);
    EXPECT_FLOAT_EQ(mat.roughness(), 0.0f);
    EXPECT_FLOAT_EQ(mat.metallic(), 0.0f);
    EXPECT_FLOAT_EQ(mat.alpha(), 1.0f);
    EXPECT_FLOAT_EQ(mat.normalScale(), 1.0f);
    EXPECT_EQ(mat.alphaMode(), AlphaMode::Opaque);
    EXPECT_FLOAT_EQ(mat.alphaCutoff(), 0.5f);
    EXPECT_FALSE(mat.doubleSided());
}

TEST(Material, DefaultExtensionBlocksAreAbsent)
{
    Material mat;
    EXPECT_FALSE(mat.transmission().has_value());
    EXPECT_FALSE(mat.clearcoat().has_value());
    EXPECT_FALSE(mat.volume().has_value());
}

TEST(Material, SetAndGetCorePbrFields)
{
    Material mat;

    mat.name("helmet");
    mat.baseColor({0.8f, 0.6f, 0.4f});
    mat.emissive({0.1f, 0.2f, 0.3f});
    mat.roughness(0.75f);
    mat.metallic(0.5f);
    mat.alpha(0.25f);
    mat.normalScale(0.9f);
    mat.alphaMode(AlphaMode::Blend);
    mat.alphaCutoff(0.2f);
    mat.doubleSided(true);

    EXPECT_EQ(mat.name(), "helmet");
    EXPECT_FLOAT_EQ(mat.baseColor().r(), 0.8f);
    EXPECT_FLOAT_EQ(mat.baseColor().g(), 0.6f);
    EXPECT_FLOAT_EQ(mat.baseColor().b(), 0.4f);
    EXPECT_FLOAT_EQ(mat.emissive().r(), 0.1f);
    EXPECT_FLOAT_EQ(mat.emissive().g(), 0.2f);
    EXPECT_FLOAT_EQ(mat.emissive().b(), 0.3f);
    EXPECT_FLOAT_EQ(mat.roughness(), 0.75f);
    EXPECT_FLOAT_EQ(mat.metallic(), 0.5f);
    EXPECT_FLOAT_EQ(mat.alpha(), 0.25f);
    EXPECT_FLOAT_EQ(mat.normalScale(), 0.9f);
    EXPECT_EQ(mat.alphaMode(), AlphaMode::Blend);
    EXPECT_FLOAT_EQ(mat.alphaCutoff(), 0.2f);
    EXPECT_TRUE(mat.doubleSided());
}

TEST(Material, TextureSlotPointersRoundTrip)
{
    Material mat;
    Texture base;
    Texture emissive;
    Texture normal;
    Texture mr;
    Texture occlusion;
    Texture transmission;

    mat.texture(Slot::BaseColour).texture = &base;
    mat.texture(Slot::Emissive).texture = &emissive;
    mat.texture(Slot::Normal).texture = &normal;
    mat.texture(Slot::MetallicRoughness).texture = &mr;
    mat.texture(Slot::Occlusion).texture = &occlusion;
    mat.texture(Slot::Transmission).texture = &transmission;

    EXPECT_TRUE(mat.texture(Slot::BaseColour).has());
    EXPECT_TRUE(mat.texture(Slot::Emissive).has());
    EXPECT_TRUE(mat.texture(Slot::Normal).has());
    EXPECT_TRUE(mat.texture(Slot::MetallicRoughness).has());
    EXPECT_TRUE(mat.texture(Slot::Occlusion).has());
    EXPECT_TRUE(mat.texture(Slot::Transmission).has());
    EXPECT_EQ(mat.texture(Slot::BaseColour).texture, &base);
}

TEST(Material, TextureSlotTexCoordAndTransformRoundTrip)
{
    Material mat;
    mat.texture(Slot::Normal).texCoord = 1;
    mat.texture(Slot::Normal).transform = UvTransform{0.5f, 0.25f, 2.0f, 3.0f, 0.4f};

    EXPECT_EQ(mat.texture(Slot::Normal).texCoord, 1);
    EXPECT_FLOAT_EQ(mat.texture(Slot::Normal).transform.offsetX, 0.5f);
    EXPECT_FLOAT_EQ(mat.texture(Slot::Normal).transform.scaleY, 3.0f);
    EXPECT_FLOAT_EQ(mat.texture(Slot::Normal).transform.rotation, 0.4f);
}

TEST(MaterialBinding, ToMaterialUboPacksCoreFields)
{
    Material mat;
    mat.baseColor({0.8f, 0.6f, 0.4f});
    mat.emissive({0.1f, 0.2f, 0.3f});
    mat.roughness(0.75f);
    mat.metallic(0.5f);
    mat.alpha(0.25f);
    mat.normalScale(0.9f);
    mat.alphaMode(AlphaMode::Mask);
    mat.alphaCutoff(0.2f);
    mat.occlusionStrength(0.7f);

    const MaterialUBO ubo = toMaterialUBO(mat);

    EXPECT_FLOAT_EQ(ubo.diffuseAlpha[0], 0.8f);
    EXPECT_FLOAT_EQ(ubo.diffuseAlpha[3], 0.25f);
    EXPECT_FLOAT_EQ(ubo.emissiveRoughness[2], 0.3f);
    EXPECT_FLOAT_EQ(ubo.emissiveRoughness[3], 0.75f);
    EXPECT_FLOAT_EQ(ubo.materialParams[0], 0.5f);
    EXPECT_FLOAT_EQ(ubo.materialParams[1], 0.9f);
    EXPECT_FLOAT_EQ(ubo.materialParams[2], 0.2f);
    EXPECT_FLOAT_EQ(ubo.materialParams[3], 0.7f);
}

TEST(MaterialBinding, ToMaterialUboUsesExtensionDefaultsWhenAbsent)
{
    // Absent optional blocks must pack the same defaults the old always-present
    // members produced: transmission factor 0 / ior 1.5, clearcoat 0, etc.
    const MaterialUBO ubo = toMaterialUBO(Material{});
    EXPECT_FLOAT_EQ(ubo.transmissionParams[0], 0.0f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[3], 1.5f);
    EXPECT_FLOAT_EQ(ubo.clearcoatParams[0], 0.0f);
    EXPECT_FLOAT_EQ(ubo.clearcoatParams[2], 1.0f);
    EXPECT_FLOAT_EQ(ubo.volumeParams[0], 0.0f);
}

TEST(MaterialBinding, MissingTextureHandlesPackAsNull)
{
    const auto handles = materialTextureHandles(Material{});

    for (TextureHandle handle : handles)
    {
        EXPECT_EQ(handle, NullTexture);
    }
}

TEST(Material, TransmissionBlockRoundTrip)
{
    Material mat;
    EXPECT_FALSE(mat.transmission().has_value());

    mat.transmission(TransmissionParams{0.65f, 1.33f});
    ASSERT_TRUE(mat.transmission().has_value());
    EXPECT_FLOAT_EQ(mat.transmission()->factor, 0.65f);
    EXPECT_FLOAT_EQ(mat.transmission()->ior, 1.33f);
}

TEST(Material, TransmissionDefaultsMatchSpec)
{
    // Spec defaults when the optional is unset: factor 0, ior 1.5.
    const TransmissionParams defaults = Material{}.transmission().value_or(TransmissionParams{});
    EXPECT_FLOAT_EQ(defaults.factor, 0.0f);
    EXPECT_FLOAT_EQ(defaults.ior, 1.5f);
}

TEST(Material, TransmissionTextureSlotRoundTrip)
{
    Material mat;
    EXPECT_EQ(mat.texture(Slot::Transmission).texCoord, 0);
    mat.texture(Slot::Transmission).texCoord = 1;
    mat.texture(Slot::Transmission).transform = UvTransform{0.5f, 0.25f, 2.0f, 3.0f, 0.4f};
    EXPECT_EQ(mat.texture(Slot::Transmission).texCoord, 1);
    EXPECT_FLOAT_EQ(mat.texture(Slot::Transmission).transform.offsetX, 0.5f);
    EXPECT_FLOAT_EQ(mat.texture(Slot::Transmission).transform.rotation, 0.4f);
}

TEST(Material, MoveConstructionPreservesCoreFields)
{
    Material original;
    original.name("moved");
    original.baseColor({0.2f, 0.4f, 0.6f});
    original.emissive({0.7f, 0.1f, 0.0f});
    original.roughness(0.8f);
    original.metallic(0.9f);
    original.alpha(0.3f);
    original.normalScale(0.6f);
    original.clearcoat(ClearcoatParams{0.8f, 0.25f, 0.6f});

    Material moved(std::move(original));

    EXPECT_EQ(moved.name(), "moved");
    EXPECT_FLOAT_EQ(moved.baseColor().r(), 0.2f);
    EXPECT_FLOAT_EQ(moved.emissive().r(), 0.7f);
    EXPECT_FLOAT_EQ(moved.roughness(), 0.8f);
    EXPECT_FLOAT_EQ(moved.metallic(), 0.9f);
    EXPECT_FLOAT_EQ(moved.alpha(), 0.3f);
    EXPECT_FLOAT_EQ(moved.normalScale(), 0.6f);
    ASSERT_TRUE(moved.clearcoat().has_value());
    EXPECT_FLOAT_EQ(moved.clearcoat()->factor, 0.8f);
}

TEST(Material, ClearcoatBlockRoundTrips)
{
    Material mat;
    EXPECT_FALSE(mat.clearcoat().has_value());

    mat.clearcoat(ClearcoatParams{0.8f, 0.25f, 0.6f});
    ASSERT_TRUE(mat.clearcoat().has_value());
    EXPECT_FLOAT_EQ(mat.clearcoat()->factor, 0.8f);
    EXPECT_FLOAT_EQ(mat.clearcoat()->roughness, 0.25f);
    EXPECT_FLOAT_EQ(mat.clearcoat()->normalScale, 0.6f);
}

TEST(Material, ClearcoatDefaultsMatchSpec)
{
    const ClearcoatParams defaults = Material{}.clearcoat().value_or(ClearcoatParams{});
    EXPECT_FLOAT_EQ(defaults.factor, 0.0f);
    EXPECT_FLOAT_EQ(defaults.roughness, 0.0f);
    EXPECT_FLOAT_EQ(defaults.normalScale, 1.0f);
}

TEST(Material, ClearcoatTextureSlotsRoundTrip)
{
    Material mat;
    Texture tFactor;
    Texture tRough;
    Texture tNormal;
    mat.texture(Slot::Clearcoat).texture = &tFactor;
    mat.texture(Slot::ClearcoatRoughness).texture = &tRough;
    mat.texture(Slot::ClearcoatNormal).texture = &tNormal;
    mat.texture(Slot::Clearcoat).texCoord = 1;
    mat.texture(Slot::ClearcoatNormal).transform = UvTransform{0.5f, 0.6f, 1.5f, 2.5f, 1.5f};

    EXPECT_TRUE(mat.texture(Slot::Clearcoat).has());
    EXPECT_TRUE(mat.texture(Slot::ClearcoatRoughness).has());
    EXPECT_TRUE(mat.texture(Slot::ClearcoatNormal).has());
    EXPECT_EQ(mat.texture(Slot::Clearcoat).texCoord, 1);
    EXPECT_FLOAT_EQ(mat.texture(Slot::ClearcoatNormal).transform.rotation, 1.5f);
}

TEST(Material, VolumeBlockRoundTrips)
{
    Material mat;
    EXPECT_FALSE(mat.volume().has_value());

    mat.volume(VolumeParams{0.7f, Colour3(0.2f, 0.6f, 0.4f), 2.5f});
    ASSERT_TRUE(mat.volume().has_value());
    EXPECT_FLOAT_EQ(mat.volume()->thicknessFactor, 0.7f);
    EXPECT_EQ(mat.volume()->attenuationColor, Colour3(0.2f, 0.6f, 0.4f));
    EXPECT_FLOAT_EQ(mat.volume()->attenuationDistance, 2.5f);
}

TEST(Material, VolumeDefaultsMatchSpec)
{
    const VolumeParams defaults = Material{}.volume().value_or(VolumeParams{});
    EXPECT_FLOAT_EQ(defaults.thicknessFactor, 0.0f);
    EXPECT_EQ(defaults.attenuationColor, Colour3(1.0f, 1.0f, 1.0f));
    EXPECT_TRUE(std::isinf(defaults.attenuationDistance));
}

TEST(Material, ThicknessTextureSlotRoundTrip)
{
    Material mat;
    Texture t;
    mat.texture(Slot::Thickness).texture = &t;
    mat.texture(Slot::Thickness).texCoord = 1;
    EXPECT_TRUE(mat.texture(Slot::Thickness).has());
    EXPECT_EQ(mat.texture(Slot::Thickness).texCoord, 1);
}
