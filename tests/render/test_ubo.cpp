#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("MorphUBO.DefaultInitialisation", "[MorphUBO]")
{
    MorphUBO ubo{};
    CHECK(ubo.hasMorph == 0);
    CHECK(ubo.morphTargetCount == 0);
    CHECK(ubo.vertexCount == 0);
    for (int i = 0; i < fire_engine::kMaxMorphTargets; ++i)
    {
        CHECK(ubo.weights[i] == Catch::Approx(0.0f).margin(1e-5f));
    }
}

TEST_CASE("MorphUBO.MaxMorphTargetsConstant", "[MorphUBO]")
{
    CHECK(fire_engine::kMaxMorphTargets >= 2);
    CHECK(fire_engine::kMaxMorphTargets == 8);
}

TEST_CASE("MorphUBO.WeightsArraySize", "[MorphUBO]")
{
    MorphUBO ubo{};
    CHECK(sizeof(ubo.weights) / sizeof(float) ==
          static_cast<std::size_t>(fire_engine::kMaxMorphTargets));
}

TEST_CASE("MorphUBO.SetWeights", "[MorphUBO]")
{
    MorphUBO ubo{};
    ubo.hasMorph = 1;
    ubo.morphTargetCount = 2;
    ubo.vertexCount = 24;
    ubo.weights[0] = 0.5f;
    ubo.weights[1] = 0.8f;

    CHECK(ubo.hasMorph == 1);
    CHECK(ubo.morphTargetCount == 2);
    CHECK(ubo.vertexCount == 24);
    CHECK(ubo.weights[0] == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(ubo.weights[1] == Catch::Approx(0.8f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// UBO alignment sanity
// ---------------------------------------------------------------------------

TEST_CASE("UBO.UniformBufferObjectSize", "[UBO]")
{
    // Must be compatible with std140 layout
    CHECK(sizeof(UniformBufferObject) % 16 == 0u);
}

TEST_CASE("UBO.MaterialUBOSize", "[UBO]")
{
    CHECK(sizeof(MaterialUBO) % 16 == 0u);
}

TEST_CASE("UBO.MaterialUBOHasTextureDefaultsToZero", "[UBO]")
{
    MaterialUBO ubo{};
    CHECK(ubo.textureFlags[0] == 0);
}

TEST_CASE("UBO.MaterialUBOFieldOrder", "[UBO]")
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

TEST_CASE("UBO.MaterialUBOAlphaCutoffRoundTrip", "[UBO]")
{
    MaterialUBO ubo{};
    ubo.materialParams[2] = 0.25f;
    CHECK(ubo.materialParams[2] == Catch::Approx(0.25f).margin(1e-5f));
}

TEST_CASE("UBO.MaterialUBOAlphaRoundTrip", "[UBO]")
{
    MaterialUBO ubo{};
    ubo.diffuseAlpha[3] = 0.75f;
    CHECK(ubo.diffuseAlpha[3] == Catch::Approx(0.75f).margin(1e-5f));
}

TEST_CASE("UBO.MaterialUBOHasTextureCanBeSet", "[UBO]")
{
    MaterialUBO ubo{};
    ubo.textureFlags[0] = 1;
    CHECK(ubo.textureFlags[0] == 1);
}

TEST_CASE("UBO.MaterialUBOTexCoordIndicesDefaultToZero", "[UBO]")
{
    MaterialUBO ubo{};
    for (int i = 0; i < 4; ++i)
    {
        CHECK(ubo.texCoordIndices[i] == 0);
    }
    // Occlusion's UV-set index also lives in extraFlags.y.
    CHECK(ubo.extraFlags[1] == 0);
}

TEST_CASE("UBO.MaterialUBOTexCoordIndicesRoundTrip", "[UBO]")
{
    MaterialUBO ubo{};
    ubo.texCoordIndices[0] = 1; // baseColor on TEXCOORD_1
    ubo.texCoordIndices[1] = 0;
    ubo.texCoordIndices[2] = 1; // normal on TEXCOORD_1
    ubo.texCoordIndices[3] = 0;
    ubo.extraFlags[1] = 1; // occlusion on TEXCOORD_1
    CHECK(ubo.texCoordIndices[0] == 1);
    CHECK(ubo.texCoordIndices[2] == 1);
    CHECK(ubo.extraFlags[1] == 1);
}

TEST_CASE("UBO.MaterialUBOHasEmissiveTextureCanBeSet", "[UBO]")
{
    MaterialUBO ubo{};
    ubo.textureFlags[1] = 1;
    CHECK(ubo.textureFlags[1] == 1);
}

TEST_CASE("UBO.MaterialUBOHasNormalTextureCanBeSet", "[UBO]")
{
    MaterialUBO ubo{};
    ubo.textureFlags[2] = 1;
    CHECK(ubo.textureFlags[2] == 1);
}

TEST_CASE("UBO.MaterialUBOHasMetallicRoughnessTextureCanBeSet", "[UBO]")
{
    MaterialUBO ubo{};
    ubo.textureFlags[3] = 1;
    CHECK(ubo.textureFlags[3] == 1);
}

TEST_CASE("UBO.MaterialUBOHasOcclusionTextureCanBeSet", "[UBO]")
{
    MaterialUBO ubo{};
    ubo.extraFlags[0] = 1;
    CHECK(ubo.extraFlags[0] == 1);
}

TEST_CASE("UBO.MaterialUBOTransmissionDefaultsMatchShaderExpectations", "[UBO]")
{
    MaterialUBO ubo{};
    const auto t = fire_engine::slotIndex(fire_engine::MaterialTextureSlot::Transmission);
    CHECK(ubo.uv[t].offsetScale[0] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[1] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[2] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[3] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.uv[t].rotation == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.transmissionParams[0] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.transmissionParams[1] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.transmissionParams[2] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.transmissionParams[3] == Catch::Approx(1.5f).margin(1e-5f));
}

TEST_CASE("UBO.MaterialUBOTransmissionFieldsRoundTrip", "[UBO]")
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

    CHECK(ubo.uv[t].offsetScale[0] == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[1] == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[2] == Catch::Approx(0.75f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[3] == Catch::Approx(1.25f).margin(1e-5f));
    CHECK(ubo.uv[t].rotation == Catch::Approx(0.6f).margin(1e-5f));
    CHECK(ubo.transmissionParams[0] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.transmissionParams[1] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.transmissionParams[2] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.transmissionParams[3] == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("UBO.MaterialUBOVolumeDefaultsMatchShaderExpectations", "[UBO]")
{
    MaterialUBO ubo{};
    const auto t = fire_engine::slotIndex(fire_engine::MaterialTextureSlot::Thickness);
    CHECK(ubo.volumeParams[0] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.volumeParams[1] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.volumeParams[2] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.volumeParams[3] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.attenuation[0] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.attenuation[1] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.attenuation[2] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.attenuation[3] == Catch::Approx(1.0e6f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[0] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[1] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[2] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[3] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.uv[t].rotation == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("UBO.MaterialUBOVolumeFieldsRoundTrip", "[UBO]")
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

    CHECK(ubo.volumeParams[0] == Catch::Approx(0.125f).margin(1e-5f));
    CHECK(ubo.volumeParams[1] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.volumeParams[2] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.attenuation[0] == Catch::Approx(0.3f).margin(1e-5f));
    CHECK(ubo.attenuation[1] == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(ubo.attenuation[2] == Catch::Approx(0.7f).margin(1e-5f));
    CHECK(ubo.attenuation[3] == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[0] == Catch::Approx(0.1f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[1] == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[2] == Catch::Approx(0.8f).margin(1e-5f));
    CHECK(ubo.uv[t].offsetScale[3] == Catch::Approx(0.9f).margin(1e-5f));
    CHECK(ubo.uv[t].rotation == Catch::Approx(0.35f).margin(1e-5f));
}

TEST_CASE("UBO.MaterialUBOTextureFlagsFieldOrder", "[UBO]")
{
    static_assert(offsetof(MaterialUBO, textureFlags) % 16 == 0,
                  "textureFlags must be 16-byte aligned for std140 ivec4");
    static_assert(offsetof(MaterialUBO, extraFlags) % 16 == 0,
                  "extraFlags must be 16-byte aligned for std140 ivec4");
    SUCCEED();
}

TEST_CASE("UBO.MorphUBOSize", "[UBO]")
{
    CHECK(sizeof(MorphUBO) % 16 == 0u);
}

TEST_CASE("UBO.SkinUBOSize", "[UBO]")
{
    CHECK(sizeof(SkinUBO) % 16 == 0u);
}

TEST_CASE("UBO.LightUBOSize", "[UBO]")
{
    CHECK(sizeof(LightUBO) % 16 == 0u);
}

TEST_CASE("UBO.EnvironmentCaptureUBOSize", "[UBO]")
{
    CHECK(sizeof(EnvironmentCaptureUBO) % 16 == 0u);
}

TEST_CASE("UBO.EnvironmentCaptureUBODefaultFaceIndexIsZero", "[UBO]")
{
    EnvironmentCaptureUBO ubo{};
    CHECK(ubo.faceIndex == 0);
    CHECK(ubo.faceExtent == 0);
}

TEST_CASE("UBO.LightUBODefaults", "[UBO]")
{
    LightUBO ubo{};
    CHECK(ubo.lightCount == 0);
    for (int i = 0; i < 4; ++i)
    {
        CHECK(ubo.iblParams[i] == Catch::Approx(0.0f).margin(1e-5f));
        CHECK(ubo.shadowParams[i] == Catch::Approx(0.0f).margin(1e-5f));
        CHECK(ubo.environmentParams[i] == Catch::Approx(0.0f).margin(1e-5f));
    }
    for (int i = 0; i < fire_engine::kMaxLights; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            CHECK(ubo.lights[i].position[j] == Catch::Approx(0.0f).margin(1e-5f));
            CHECK(ubo.lights[i].direction[j] == Catch::Approx(0.0f).margin(1e-5f));
            CHECK(ubo.lights[i].colour[j] == Catch::Approx(0.0f).margin(1e-5f));
        }
    }
}

TEST_CASE("UBO.LightUBOFieldOrder", "[UBO]")
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

TEST_CASE("UBO.LightUBOFieldsAligned16", "[UBO]")
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

TEST_CASE("UBO.LightDataSizeAligned", "[UBO]")
{
    CHECK(sizeof(fire_engine::LightData) % 16 == 0u);
    CHECK(sizeof(fire_engine::LightData) == 64u);
}

TEST_CASE("UBO.ForwardPushConstantsDefaultsToNoSelfShadowSlot", "[UBO]")
{
    ForwardPushConstants pc{};
    CHECK(pc.selfShadowSlot == -1);
    CHECK(sizeof(ForwardPushConstants) == 16u);
}

TEST_CASE("UBO.ShadowPushConstantsCanCarryInlineLightMatrix", "[UBO]")
{
    ShadowPushConstants pc{};
    CHECK(pc.matrixIndex == 0);
    CHECK(pc.selfShadowSlot == -1);
    CHECK(pc.selfShadowDepthEpsilon ==
          Catch::Approx(fire_engine::kSkinnedSelfShadowDepthEpsilon).margin(1e-5f));
    CHECK(sizeof(ShadowPushConstants) % 16 == 0u);
    CHECK(pc.lightViewProj == fire_engine::Mat4::identity());
}

// ---------------------------------------------------------------------------
// ShadowUBO structure tests
// ---------------------------------------------------------------------------

TEST_CASE("UBO.ShadowUBOSize", "[UBO]")
{
    CHECK(sizeof(ShadowUBO) % 16 == 0u);
}

TEST_CASE("UBO.ShadowUBODefaultHasSkinIsZero", "[UBO]")
{
    ShadowUBO ubo{};
    CHECK(ubo.hasSkin == 0);
}

TEST_CASE("UBO.ShadowUBOHasSkinCanBeSet", "[UBO]")
{
    ShadowUBO ubo{};
    ubo.hasSkin = 1;
    CHECK(ubo.hasSkin == 1);
}

TEST_CASE("UBO.ShadowUBOFieldOrder", "[UBO]")
{
    static_assert(offsetof(ShadowUBO, model) < offsetof(ShadowUBO, lightViewProj),
                  "model must precede lightViewProj to match shader layout");
    static_assert(offsetof(ShadowUBO, lightViewProj) < offsetof(ShadowUBO, hasSkin),
                  "lightViewProj must precede hasSkin to match shader layout");
    SUCCEED();
}

TEST_CASE("UBO.ShadowUBOMatricesAligned16", "[UBO]")
{
    static_assert(offsetof(ShadowUBO, model) % 16 == 0,
                  "model must be 16-byte aligned for std140 mat4");
    static_assert(offsetof(ShadowUBO, lightViewProj) % 16 == 0,
                  "lightViewProj must be 16-byte aligned for std140 mat4");
    SUCCEED();
}
