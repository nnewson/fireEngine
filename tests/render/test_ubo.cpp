#include <gtest/gtest.h>

#include <cstring>

#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/ubo.hpp>

using fire_engine::EnvironmentCaptureUBO;
using fire_engine::ForwardPushConstants;
using fire_engine::LightUBO;
using fire_engine::MaterialUBO;
using fire_engine::MorphUBO;
using fire_engine::ShadowPushConstants;
using fire_engine::ShadowUBO;
using fire_engine::SkinUBO;
using fire_engine::UniformBufferObject;

// ---------------------------------------------------------------------------
// MorphUBO structure tests
// ---------------------------------------------------------------------------

TEST(MorphUBO, DefaultInitialisation)
{
    MorphUBO ubo{};
    EXPECT_EQ(ubo.hasMorph, 0);
    EXPECT_EQ(ubo.morphTargetCount, 0);
    EXPECT_EQ(ubo.vertexCount, 0);
    for (int i = 0; i < fire_engine::kMaxMorphTargets; ++i)
    {
        EXPECT_FLOAT_EQ(ubo.weights[i], 0.0f);
    }
}

TEST(MorphUBO, MaxMorphTargetsConstant)
{
    EXPECT_GE(fire_engine::kMaxMorphTargets, 2);
    EXPECT_EQ(fire_engine::kMaxMorphTargets, 8);
}

TEST(MorphUBO, WeightsArraySize)
{
    MorphUBO ubo{};
    EXPECT_EQ(sizeof(ubo.weights) / sizeof(float),
              static_cast<std::size_t>(fire_engine::kMaxMorphTargets));
}

TEST(MorphUBO, SetWeights)
{
    MorphUBO ubo{};
    ubo.hasMorph = 1;
    ubo.morphTargetCount = 2;
    ubo.vertexCount = 24;
    ubo.weights[0] = 0.5f;
    ubo.weights[1] = 0.8f;

    EXPECT_EQ(ubo.hasMorph, 1);
    EXPECT_EQ(ubo.morphTargetCount, 2);
    EXPECT_EQ(ubo.vertexCount, 24);
    EXPECT_FLOAT_EQ(ubo.weights[0], 0.5f);
    EXPECT_FLOAT_EQ(ubo.weights[1], 0.8f);
}

// ---------------------------------------------------------------------------
// UBO alignment sanity
// ---------------------------------------------------------------------------

TEST(UBO, UniformBufferObjectSize)
{
    // Must be compatible with std140 layout
    EXPECT_EQ(sizeof(UniformBufferObject) % 16, 0u);
}

TEST(UBO, MaterialUBOSize)
{
    EXPECT_EQ(sizeof(MaterialUBO) % 16, 0u);
}

TEST(UBO, MaterialUBOHasTextureDefaultsToZero)
{
    MaterialUBO ubo{};
    EXPECT_EQ(ubo.textureFlags[0], 0);
}

TEST(UBO, MaterialUBOFieldOrder)
{
    static_assert(offsetof(MaterialUBO, diffuseAlpha) < offsetof(MaterialUBO, emissiveRoughness),
                  "diffuseAlpha must precede emissiveRoughness to match shader layout");
    static_assert(offsetof(MaterialUBO, emissiveRoughness) < offsetof(MaterialUBO, materialParams),
                  "emissiveRoughness must precede materialParams to match shader layout");
    static_assert(offsetof(MaterialUBO, materialParams) < offsetof(MaterialUBO, textureFlags),
                  "materialParams must precede textureFlags to match shader layout");
    static_assert(offsetof(MaterialUBO, extraFlags) < offsetof(MaterialUBO, texCoordIndices),
                  "extraFlags must precede texCoordIndices to match shader layout");
    static_assert(offsetof(MaterialUBO, textureFlags) < offsetof(MaterialUBO, extraFlags),
                  "textureFlags must precede extraFlags to match shader layout");
    SUCCEED();
}

TEST(UBO, MaterialUBOAlphaCutoffRoundTrip)
{
    MaterialUBO ubo{};
    ubo.materialParams[2] = 0.25f;
    EXPECT_FLOAT_EQ(ubo.materialParams[2], 0.25f);
}

TEST(UBO, MaterialUBOAlphaRoundTrip)
{
    MaterialUBO ubo{};
    ubo.diffuseAlpha[3] = 0.75f;
    EXPECT_FLOAT_EQ(ubo.diffuseAlpha[3], 0.75f);
}

TEST(UBO, MaterialUBOHasTextureCanBeSet)
{
    MaterialUBO ubo{};
    ubo.textureFlags[0] = 1;
    EXPECT_EQ(ubo.textureFlags[0], 1);
}

TEST(UBO, MaterialUBOTexCoordIndicesDefaultToZero)
{
    MaterialUBO ubo{};
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(ubo.texCoordIndices[i], 0);
    }
    // Occlusion's UV-set index also lives in extraFlags.y.
    EXPECT_EQ(ubo.extraFlags[1], 0);
}

TEST(UBO, MaterialUBOTexCoordIndicesRoundTrip)
{
    MaterialUBO ubo{};
    ubo.texCoordIndices[0] = 1; // baseColor on TEXCOORD_1
    ubo.texCoordIndices[1] = 0;
    ubo.texCoordIndices[2] = 1; // normal on TEXCOORD_1
    ubo.texCoordIndices[3] = 0;
    ubo.extraFlags[1] = 1; // occlusion on TEXCOORD_1
    EXPECT_EQ(ubo.texCoordIndices[0], 1);
    EXPECT_EQ(ubo.texCoordIndices[2], 1);
    EXPECT_EQ(ubo.extraFlags[1], 1);
}

TEST(UBO, MaterialUBOHasEmissiveTextureCanBeSet)
{
    MaterialUBO ubo{};
    ubo.textureFlags[1] = 1;
    EXPECT_EQ(ubo.textureFlags[1], 1);
}

TEST(UBO, MaterialUBOHasNormalTextureCanBeSet)
{
    MaterialUBO ubo{};
    ubo.textureFlags[2] = 1;
    EXPECT_EQ(ubo.textureFlags[2], 1);
}

TEST(UBO, MaterialUBOHasMetallicRoughnessTextureCanBeSet)
{
    MaterialUBO ubo{};
    ubo.textureFlags[3] = 1;
    EXPECT_EQ(ubo.textureFlags[3], 1);
}

TEST(UBO, MaterialUBOHasOcclusionTextureCanBeSet)
{
    MaterialUBO ubo{};
    ubo.extraFlags[0] = 1;
    EXPECT_EQ(ubo.extraFlags[0], 1);
}

TEST(UBO, MaterialUBOTransmissionDefaultsMatchShaderExpectations)
{
    MaterialUBO ubo{};
    const auto t = fire_engine::slotIndex(fire_engine::MaterialTextureSlot::Transmission);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[0], 0.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[1], 0.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[2], 1.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[3], 1.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].rotation, 0.0f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[0], 0.0f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[1], 0.0f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[2], 0.0f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[3], 1.5f);
}

TEST(UBO, MaterialUBOTransmissionFieldsRoundTrip)
{
    MaterialUBO ubo{};
    const auto t = fire_engine::slotIndex(fire_engine::MaterialTextureSlot::Transmission);
    ubo.uv[t].offsetScale[0] = 0.25f;
    ubo.uv[t].offsetScale[1] = 0.5f;
    ubo.uv[t].offsetScale[2] = 0.75f;
    ubo.uv[t].offsetScale[3] = 1.25f;
    ubo.uv[t].rotation = 0.6f;
    ubo.transmissionParams[0] = 1.0f;
    ubo.transmissionParams[1] = 1.0f;
    ubo.transmissionParams[2] = 1.0f;
    ubo.transmissionParams[3] = 1.0f;

    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[0], 0.25f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[1], 0.5f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[2], 0.75f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[3], 1.25f);
    EXPECT_FLOAT_EQ(ubo.uv[t].rotation, 0.6f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[0], 1.0f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[1], 1.0f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[2], 1.0f);
    EXPECT_FLOAT_EQ(ubo.transmissionParams[3], 1.0f);
}

TEST(UBO, MaterialUBOVolumeDefaultsMatchShaderExpectations)
{
    MaterialUBO ubo{};
    const auto t = fire_engine::slotIndex(fire_engine::MaterialTextureSlot::Thickness);
    EXPECT_FLOAT_EQ(ubo.volumeParams[0], 0.0f);
    EXPECT_FLOAT_EQ(ubo.volumeParams[1], 0.0f);
    EXPECT_FLOAT_EQ(ubo.volumeParams[2], 0.0f);
    EXPECT_FLOAT_EQ(ubo.volumeParams[3], 0.0f);
    EXPECT_FLOAT_EQ(ubo.attenuation[0], 1.0f);
    EXPECT_FLOAT_EQ(ubo.attenuation[1], 1.0f);
    EXPECT_FLOAT_EQ(ubo.attenuation[2], 1.0f);
    EXPECT_FLOAT_EQ(ubo.attenuation[3], 1.0e6f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[0], 0.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[1], 0.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[2], 1.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[3], 1.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].rotation, 0.0f);
}

TEST(UBO, MaterialUBOVolumeFieldsRoundTrip)
{
    MaterialUBO ubo{};
    const auto t = fire_engine::slotIndex(fire_engine::MaterialTextureSlot::Thickness);
    ubo.volumeParams[0] = 0.125f;
    ubo.volumeParams[1] = 1.0f;
    ubo.volumeParams[2] = 1.0f;
    ubo.attenuation[0] = 0.3f;
    ubo.attenuation[1] = 0.5f;
    ubo.attenuation[2] = 0.7f;
    ubo.attenuation[3] = 4.0f;
    ubo.uv[t].offsetScale[0] = 0.1f;
    ubo.uv[t].offsetScale[1] = 0.2f;
    ubo.uv[t].offsetScale[2] = 0.8f;
    ubo.uv[t].offsetScale[3] = 0.9f;
    ubo.uv[t].rotation = 0.35f;

    EXPECT_FLOAT_EQ(ubo.volumeParams[0], 0.125f);
    EXPECT_FLOAT_EQ(ubo.volumeParams[1], 1.0f);
    EXPECT_FLOAT_EQ(ubo.volumeParams[2], 1.0f);
    EXPECT_FLOAT_EQ(ubo.attenuation[0], 0.3f);
    EXPECT_FLOAT_EQ(ubo.attenuation[1], 0.5f);
    EXPECT_FLOAT_EQ(ubo.attenuation[2], 0.7f);
    EXPECT_FLOAT_EQ(ubo.attenuation[3], 4.0f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[0], 0.1f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[1], 0.2f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[2], 0.8f);
    EXPECT_FLOAT_EQ(ubo.uv[t].offsetScale[3], 0.9f);
    EXPECT_FLOAT_EQ(ubo.uv[t].rotation, 0.35f);
}

TEST(UBO, MaterialUBOTextureFlagsFieldOrder)
{
    static_assert(offsetof(MaterialUBO, textureFlags) % 16 == 0,
                  "textureFlags must be 16-byte aligned for std140 ivec4");
    static_assert(offsetof(MaterialUBO, extraFlags) % 16 == 0,
                  "extraFlags must be 16-byte aligned for std140 ivec4");
    SUCCEED();
}

TEST(UBO, MorphUBOSize)
{
    EXPECT_EQ(sizeof(MorphUBO) % 16, 0u);
}

TEST(UBO, SkinUBOSize)
{
    EXPECT_EQ(sizeof(SkinUBO) % 16, 0u);
}

TEST(UBO, LightUBOSize)
{
    EXPECT_EQ(sizeof(LightUBO) % 16, 0u);
}

TEST(UBO, EnvironmentCaptureUBOSize)
{
    EXPECT_EQ(sizeof(EnvironmentCaptureUBO) % 16, 0u);
}

TEST(UBO, EnvironmentCaptureUBODefaultFaceIndexIsZero)
{
    EnvironmentCaptureUBO ubo{};
    EXPECT_EQ(ubo.faceIndex, 0);
    EXPECT_EQ(ubo.faceExtent, 0);
}

TEST(UBO, LightUBODefaults)
{
    LightUBO ubo{};
    EXPECT_EQ(ubo.lightCount, 0);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(ubo.iblParams[i], 0.0f);
        EXPECT_FLOAT_EQ(ubo.shadowParams[i], 0.0f);
        EXPECT_FLOAT_EQ(ubo.environmentParams[i], 0.0f);
    }
    for (int i = 0; i < fire_engine::kMaxLights; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_FLOAT_EQ(ubo.lights[i].position[j], 0.0f);
            EXPECT_FLOAT_EQ(ubo.lights[i].direction[j], 0.0f);
            EXPECT_FLOAT_EQ(ubo.lights[i].colour[j], 0.0f);
        }
    }
}

TEST(UBO, LightUBOFieldOrder)
{
    static_assert(offsetof(LightUBO, cascadeViewProj) < offsetof(LightUBO, cascadeSplits),
                  "cascadeViewProj must precede cascadeSplits to match shader layout");
    static_assert(offsetof(LightUBO, spotViewProj) < offsetof(LightUBO, selfShadowViewProj),
                  "spotViewProj must precede selfShadowViewProj to match shader layout");
    static_assert(offsetof(LightUBO, selfShadowViewProj) < offsetof(LightUBO, cascadeSplits),
                  "selfShadowViewProj must precede cascadeSplits to match shader layout");
    static_assert(offsetof(LightUBO, cascadeSplits) < offsetof(LightUBO, iblParams),
                  "cascadeSplits must precede iblParams to match shader layout");
    static_assert(offsetof(LightUBO, iblParams) < offsetof(LightUBO, shadowParams),
                  "iblParams must precede shadowParams to match shader layout");
    static_assert(offsetof(LightUBO, shadowParams) < offsetof(LightUBO, environmentParams),
                  "shadowParams must precede environmentParams to match shader layout");
    static_assert(offsetof(LightUBO, environmentParams) < offsetof(LightUBO, lightCount),
                  "environmentParams must precede lightCount to match shader layout");
    static_assert(offsetof(LightUBO, lightCount) < offsetof(LightUBO, lights),
                  "lightCount must precede lights[] to match shader layout");
    SUCCEED();
}

TEST(UBO, LightUBOFieldsAligned16)
{
    static_assert(offsetof(LightUBO, cascadeViewProj) % 16 == 0,
                  "cascadeViewProj must be 16-byte aligned for std140 mat4[]");
    static_assert(offsetof(LightUBO, spotViewProj) % 16 == 0,
                  "spotViewProj must be 16-byte aligned for std140 mat4[]");
    static_assert(offsetof(LightUBO, selfShadowViewProj) % 16 == 0,
                  "selfShadowViewProj must be 16-byte aligned for std140 mat4[]");
    static_assert(offsetof(LightUBO, cascadeSplits) % 16 == 0,
                  "cascadeSplits must be 16-byte aligned for std140 vec4");
    static_assert(offsetof(LightUBO, iblParams) % 16 == 0,
                  "iblParams must be 16-byte aligned for std140 vec4");
    static_assert(offsetof(LightUBO, shadowParams) % 16 == 0,
                  "shadowParams must be 16-byte aligned for std140 vec4");
    static_assert(offsetof(LightUBO, environmentParams) % 16 == 0,
                  "environmentParams must be 16-byte aligned for std140 vec4");
    static_assert(offsetof(LightUBO, lightCount) % 16 == 0,
                  "lightCount must be 16-byte aligned for std140 int + padding");
    static_assert(offsetof(LightUBO, lights) % 16 == 0,
                  "lights[] must be 16-byte aligned for std140 LightData[]");
    SUCCEED();
}

TEST(UBO, LightDataSizeAligned)
{
    EXPECT_EQ(sizeof(fire_engine::LightData) % 16, 0u);
    EXPECT_EQ(sizeof(fire_engine::LightData), 64u);
}

TEST(UBO, ForwardPushConstantsDefaultsToNoSelfShadowSlot)
{
    ForwardPushConstants pc{};
    EXPECT_EQ(pc.selfShadowSlot, -1);
    EXPECT_EQ(sizeof(ForwardPushConstants), 16u);
}

TEST(UBO, ShadowPushConstantsCanCarryInlineLightMatrix)
{
    ShadowPushConstants pc{};
    EXPECT_EQ(pc.matrixIndex, 0);
    EXPECT_EQ(pc.selfShadowSlot, -1);
    EXPECT_FLOAT_EQ(pc.selfShadowDepthEpsilon, fire_engine::kSkinnedSelfShadowDepthEpsilon);
    EXPECT_EQ(sizeof(ShadowPushConstants) % 16, 0u);
    EXPECT_EQ(pc.lightViewProj, fire_engine::Mat4::identity());
}

// ---------------------------------------------------------------------------
// ShadowUBO structure tests
// ---------------------------------------------------------------------------

TEST(UBO, ShadowUBOSize)
{
    EXPECT_EQ(sizeof(ShadowUBO) % 16, 0u);
}

TEST(UBO, ShadowUBODefaultHasSkinIsZero)
{
    ShadowUBO ubo{};
    EXPECT_EQ(ubo.hasSkin, 0);
}

TEST(UBO, ShadowUBOHasSkinCanBeSet)
{
    ShadowUBO ubo{};
    ubo.hasSkin = 1;
    EXPECT_EQ(ubo.hasSkin, 1);
}

TEST(UBO, ShadowUBOFieldOrder)
{
    static_assert(offsetof(ShadowUBO, model) < offsetof(ShadowUBO, lightViewProj),
                  "model must precede lightViewProj to match shader layout");
    static_assert(offsetof(ShadowUBO, lightViewProj) < offsetof(ShadowUBO, hasSkin),
                  "lightViewProj must precede hasSkin to match shader layout");
    SUCCEED();
}

TEST(UBO, ShadowUBOMatricesAligned16)
{
    static_assert(offsetof(ShadowUBO, model) % 16 == 0,
                  "model must be 16-byte aligned for std140 mat4");
    static_assert(offsetof(ShadowUBO, lightViewProj) % 16 == 0,
                  "lightViewProj must be 16-byte aligned for std140 mat4");
    SUCCEED();
}
