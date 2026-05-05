#include <gtest/gtest.h>

#include <cmath>

#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/material_binding.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/render/ubo.hpp>

using namespace fire_engine;

TEST(Material, DefaultConstructionUsesPbrDefaults)
{
    Material mat;

    EXPECT_TRUE(mat.name().empty());
    EXPECT_FALSE(mat.hasTexture());
    EXPECT_FALSE(mat.hasEmissiveTexture());
    EXPECT_FALSE(mat.hasNormalTexture());
    EXPECT_FALSE(mat.hasMetallicRoughnessTexture());
    EXPECT_FALSE(mat.hasOcclusionTexture());

    EXPECT_FLOAT_EQ(mat.diffuse().r(), 0.0f);
    EXPECT_FLOAT_EQ(mat.diffuse().g(), 0.0f);
    EXPECT_FLOAT_EQ(mat.diffuse().b(), 0.0f);
    EXPECT_FLOAT_EQ(mat.emissive().r(), 0.0f);
    EXPECT_FLOAT_EQ(mat.roughness(), 0.0f);
    EXPECT_FLOAT_EQ(mat.metallic(), 0.0f);
    EXPECT_FLOAT_EQ(mat.alpha(), 1.0f);
    EXPECT_FLOAT_EQ(mat.normalScale(), 1.0f);
    EXPECT_EQ(mat.alphaMode(), AlphaMode::Opaque);
    EXPECT_FLOAT_EQ(mat.alphaCutoff(), 0.5f);
    EXPECT_FALSE(mat.doubleSided());
}

TEST(Material, SetAndGetCorePbrFields)
{
    Material mat;

    mat.name("helmet");
    mat.diffuse({0.8f, 0.6f, 0.4f});
    mat.emissive({0.1f, 0.2f, 0.3f});
    mat.roughness(0.75f);
    mat.metallic(0.5f);
    mat.alpha(0.25f);
    mat.normalScale(0.9f);
    mat.alphaMode(AlphaMode::Blend);
    mat.alphaCutoff(0.2f);
    mat.doubleSided(true);

    EXPECT_EQ(mat.name(), "helmet");
    EXPECT_FLOAT_EQ(mat.diffuse().r(), 0.8f);
    EXPECT_FLOAT_EQ(mat.diffuse().g(), 0.6f);
    EXPECT_FLOAT_EQ(mat.diffuse().b(), 0.4f);
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

TEST(Material, TexturePointersRoundTrip)
{
    Material mat;
    Texture base;
    Texture emissive;
    Texture normal;
    Texture mr;
    Texture occlusion;
    Texture transmission;

    mat.texture(&base);
    mat.emissiveTexture(&emissive);
    mat.normalTexture(&normal);
    mat.metallicRoughnessTexture(&mr);
    mat.occlusionTexture(&occlusion);
    mat.transmissionTexture(&transmission);

    EXPECT_TRUE(mat.hasTexture());
    EXPECT_TRUE(mat.hasEmissiveTexture());
    EXPECT_TRUE(mat.hasNormalTexture());
    EXPECT_TRUE(mat.hasMetallicRoughnessTexture());
    EXPECT_TRUE(mat.hasOcclusionTexture());
    EXPECT_TRUE(mat.hasTransmissionTexture());
}

TEST(MaterialBinding, ToMaterialUboPacksCoreFields)
{
    Material mat;
    mat.diffuse({0.8f, 0.6f, 0.4f});
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

TEST(MaterialBinding, MissingTextureHandlesPackAsNull)
{
    const auto handles = materialTextureHandles(Material{});

    for (TextureHandle handle : handles)
    {
        EXPECT_EQ(handle, NullTexture);
    }
}

TEST(Material, TransmissionFactorRoundTrip)
{
    Material mat;
    EXPECT_FLOAT_EQ(mat.transmissionFactor(), 0.0f);
    mat.transmissionFactor(0.65f);
    EXPECT_FLOAT_EQ(mat.transmissionFactor(), 0.65f);
}

TEST(Material, TransmissionTexCoordRoundTrip)
{
    Material mat;
    EXPECT_EQ(mat.transmissionTexCoord(), 0);
    mat.transmissionTexCoord(1);
    EXPECT_EQ(mat.transmissionTexCoord(), 1);
}

TEST(Material, TransmissionUvTransformRoundTrip)
{
    Material mat;
    UvTransform t{0.5f, 0.25f, 2.0f, 3.0f, 0.4f};
    mat.transmissionUvTransform(t);
    EXPECT_FLOAT_EQ(mat.transmissionUvTransform().offsetX, 0.5f);
    EXPECT_FLOAT_EQ(mat.transmissionUvTransform().scaleY, 3.0f);
    EXPECT_FLOAT_EQ(mat.transmissionUvTransform().rotation, 0.4f);
}

TEST(Material, MoveConstructionPreservesCoreFields)
{
    Material original;
    original.name("moved");
    original.diffuse({0.2f, 0.4f, 0.6f});
    original.emissive({0.7f, 0.1f, 0.0f});
    original.roughness(0.8f);
    original.metallic(0.9f);
    original.alpha(0.3f);
    original.normalScale(0.6f);

    Material moved(std::move(original));

    EXPECT_EQ(moved.name(), "moved");
    EXPECT_FLOAT_EQ(moved.diffuse().r(), 0.2f);
    EXPECT_FLOAT_EQ(moved.emissive().r(), 0.7f);
    EXPECT_FLOAT_EQ(moved.roughness(), 0.8f);
    EXPECT_FLOAT_EQ(moved.metallic(), 0.9f);
    EXPECT_FLOAT_EQ(moved.alpha(), 0.3f);
    EXPECT_FLOAT_EQ(moved.normalScale(), 0.6f);
}

TEST(Material, IorDefaultsToFifteen)
{
    Material mat;
    EXPECT_FLOAT_EQ(mat.ior(), 1.5f);
}

TEST(Material, IorRoundTrip)
{
    Material mat;
    mat.ior(1.33f);
    EXPECT_FLOAT_EQ(mat.ior(), 1.33f);
}

TEST(Material, ClearcoatDefaultsAreOff)
{
    Material mat;
    EXPECT_FLOAT_EQ(mat.clearcoatFactor(), 0.0f);
    EXPECT_FLOAT_EQ(mat.clearcoatRoughness(), 0.0f);
    EXPECT_FLOAT_EQ(mat.clearcoatNormalScale(), 1.0f);
    EXPECT_FALSE(mat.hasClearcoatTexture());
    EXPECT_FALSE(mat.hasClearcoatRoughnessTexture());
    EXPECT_FALSE(mat.hasClearcoatNormalTexture());
}

TEST(Material, ClearcoatScalarRoundTrips)
{
    Material mat;
    mat.clearcoatFactor(0.8f);
    mat.clearcoatRoughness(0.25f);
    mat.clearcoatNormalScale(0.6f);
    EXPECT_FLOAT_EQ(mat.clearcoatFactor(), 0.8f);
    EXPECT_FLOAT_EQ(mat.clearcoatRoughness(), 0.25f);
    EXPECT_FLOAT_EQ(mat.clearcoatNormalScale(), 0.6f);
}

TEST(Material, ClearcoatTexCoordsRoundTrip)
{
    Material mat;
    mat.clearcoatTexCoord(1);
    mat.clearcoatRoughnessTexCoord(0);
    mat.clearcoatNormalTexCoord(1);
    EXPECT_EQ(mat.clearcoatTexCoord(), 1);
    EXPECT_EQ(mat.clearcoatRoughnessTexCoord(), 0);
    EXPECT_EQ(mat.clearcoatNormalTexCoord(), 1);
}

TEST(Material, ClearcoatUvTransformsRoundTrip)
{
    Material mat;
    UvTransform t1{0.1f, 0.2f, 2.0f, 3.0f, 0.5f};
    UvTransform t2{0.3f, 0.4f, 0.5f, 0.5f, 1.0f};
    UvTransform t3{0.5f, 0.6f, 1.5f, 2.5f, 1.5f};
    mat.clearcoatUvTransform(t1);
    mat.clearcoatRoughnessUvTransform(t2);
    mat.clearcoatNormalUvTransform(t3);
    EXPECT_FLOAT_EQ(mat.clearcoatUvTransform().offsetX, 0.1f);
    EXPECT_FLOAT_EQ(mat.clearcoatRoughnessUvTransform().scaleY, 0.5f);
    EXPECT_FLOAT_EQ(mat.clearcoatNormalUvTransform().rotation, 1.5f);
}

TEST(Material, ClearcoatTexturePointersRoundTrip)
{
    Material mat;
    Texture tFactor;
    Texture tRough;
    Texture tNormal;
    mat.clearcoatTexture(&tFactor);
    mat.clearcoatRoughnessTexture(&tRough);
    mat.clearcoatNormalTexture(&tNormal);
    EXPECT_TRUE(mat.hasClearcoatTexture());
    EXPECT_TRUE(mat.hasClearcoatRoughnessTexture());
    EXPECT_TRUE(mat.hasClearcoatNormalTexture());
}

TEST(Material, VolumeDefaultsAreOff)
{
    Material mat;
    EXPECT_FLOAT_EQ(mat.thicknessFactor(), 0.0f);
    EXPECT_EQ(mat.attenuationColor(), Colour3(1.0f, 1.0f, 1.0f));
    EXPECT_TRUE(std::isinf(mat.attenuationDistance()));
    EXPECT_FALSE(mat.hasThicknessTexture());
}

TEST(Material, VolumeScalarRoundTrips)
{
    Material mat;
    mat.thicknessFactor(0.7f);
    mat.attenuationColor(Colour3(0.2f, 0.6f, 0.4f));
    mat.attenuationDistance(2.5f);
    EXPECT_FLOAT_EQ(mat.thicknessFactor(), 0.7f);
    EXPECT_EQ(mat.attenuationColor(), Colour3(0.2f, 0.6f, 0.4f));
    EXPECT_FLOAT_EQ(mat.attenuationDistance(), 2.5f);
}

TEST(Material, ThicknessTextureRoundTrip)
{
    Material mat;
    Texture t;
    mat.thicknessTexture(&t);
    mat.thicknessTexCoord(1);
    EXPECT_TRUE(mat.hasThicknessTexture());
    EXPECT_EQ(mat.thicknessTexCoord(), 1);
}
