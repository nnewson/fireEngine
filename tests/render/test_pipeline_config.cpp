#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/ubo.hpp>

using fire_engine::bindingIndex;
using fire_engine::ForwardBinding;
using fire_engine::ForwardPushConstants;
using fire_engine::Pipeline;
using fire_engine::ShadowBinding;

TEST_CASE("PipelineConfig.ForwardConfigBindingsSplitBetweenSets", "[PipelineConfig]")
{
    using fire_engine::ForwardGlobalBinding;
    auto config = Pipeline::forwardConfig();

    // Set 0 — per-object + per-frame push UBOs/SSBO: frame (per-object), camera (per-frame), skin,
    // prev-skin (per-frame joints for skinned motion vectors), morph UBO + morph-targets SSBO +
    // VIPM geomorph SSBO (7 total). Camera + prev-skin are in set 0 (not the global set 1) so the
    // depth prepass — which reuses shader.vert but binds no globals — gets them via the same push.
    // Material data (textures + scalars) is fully bindless (set 2).
    CHECK(config.bindings.size() == 7u);
    // Set 1 — forward globals: light UBO + 5 shadow maps + debug image + 2
    // standalone samplers + 3 IBL textures + sceneColor + ssao.
    CHECK(config.globalBindings.size() == 14u);
    // Set 2 — bindless materials (texture array + materials SSBO).
    CHECK(config.bindlessSet);

    auto hasObjectBinding = [&](ForwardBinding binding)
    {
        return std::any_of(config.bindings.begin(), config.bindings.end(), [&](const auto& entry)
                           { return entry.binding == bindingIndex(binding); });
    };
    auto hasGlobalBinding = [&](ForwardGlobalBinding binding)
    {
        return std::any_of(config.globalBindings.begin(), config.globalBindings.end(),
                           [&](const auto& entry)
                           { return entry.binding == bindingIndex(binding); });
    };

    // Per-object set 0 bindings (material UBO + textures are no longer here).
    CHECK(hasObjectBinding(ForwardBinding::Frame));
    CHECK(hasObjectBinding(ForwardBinding::Camera));
    CHECK(hasObjectBinding(ForwardBinding::Skin));
    CHECK(hasObjectBinding(ForwardBinding::PrevSkin));
    CHECK(hasObjectBinding(ForwardBinding::Morph));
    CHECK(hasObjectBinding(ForwardBinding::MorphTargets));
    CHECK(hasObjectBinding(ForwardBinding::VipmMorph));
    CHECK_FALSE(hasObjectBinding(ForwardBinding::Material));
    CHECK_FALSE(hasObjectBinding(ForwardBinding::BaseColourTexture));
    CHECK_FALSE(hasObjectBinding(ForwardBinding::ThicknessTexture));

    // Forward globals set 1 bindings.
    CHECK(hasGlobalBinding(ForwardGlobalBinding::Light));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::IrradianceMap));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::PrefilteredMap));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::BrdfLut));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::ShadowCompareSampler));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::SceneColour));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::SpotShadowMap));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::PointShadowMap));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::ShadowDebugSampler));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::ShadowDebugImage));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::WorldShadowMap));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::SelfShadowMap));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::ShadowMap));
    CHECK(hasGlobalBinding(ForwardGlobalBinding::SsaoMap));
}

TEST_CASE("PipelineConfig.ForwardConfigIncludesSelfShadowPushConstant", "[PipelineConfig]")
{
    auto config = Pipeline::forwardConfig();

    REQUIRE(config.pushConstantRanges.size() == 1u);
    CHECK(config.pushConstantRanges[0].stageFlags == vk::ShaderStageFlagBits::eFragment);
    CHECK(config.pushConstantRanges[0].offset == 0u);
    CHECK(config.pushConstantRanges[0].size ==
          static_cast<uint32_t>(sizeof(fire_engine::ForwardPushConstants)));
}

TEST_CASE("PipelineConfig.ForwardConfigUsesDynamicCullMode", "[PipelineConfig]")
{
    // Opaque + double-sided share one forward pipeline: cull mode is dynamic
    // (set per draw). BLEND keeps a static cull mode (eNone) since dynamic blend
    // state is unsupported on MoltenVK, so the blend pipeline stays separate.
    auto forward = Pipeline::forwardConfig();
    CHECK(forward.dynamicCullMode);

    auto blend = Pipeline::forwardBlendConfig();
    CHECK_FALSE(blend.dynamicCullMode);
    CHECK(blend.cullMode == vk::CullModeFlagBits::eNone);
    CHECK(blend.blendEnable);
}

TEST_CASE("PipelineConfig.ForwardConfigPushesSet0", "[PipelineConfig]")
{
    // Per-object set 0 (UBOs + morph SSBO) is a push-descriptor layout — pushed
    // inline per draw, never allocated. The forward pass (blend inherits it) and
    // the shadow pass (self-shadow first/second inherit it) both push. Skybox
    // keeps a regular allocated set-0 layout.
    CHECK(Pipeline::forwardConfig().pushDescriptorSet0);
    CHECK(Pipeline::forwardBlendConfig().pushDescriptorSet0);
    CHECK(Pipeline::shadowConfig().pushDescriptorSet0);
    CHECK(Pipeline::shadowMaskedConfig().pushDescriptorSet0);
    CHECK(Pipeline::selfShadowSecondConfig().pushDescriptorSet0);
    CHECK(Pipeline::selfShadowSecondMaskedConfig().pushDescriptorSet0);
    CHECK_FALSE(Pipeline::skyboxConfig().pushDescriptorSet0);
}

TEST_CASE("PipelineConfig.DepthPrepassReadsTheMaterialAuthority", "[PipelineConfig]")
{
    // The prepass applies the alpha cutout, so it needs the same bindless material set (2) and a
    // push block to index it with. Without either, a MASK material writes depth through its holes
    // and the forward pass depth-rejects whatever stands behind them — the defect this pins.
    const auto prepass = Pipeline::depthPrepassConfig();
    const auto forward = Pipeline::forwardConfig();

    CHECK(prepass.bindlessSet);
    REQUIRE(prepass.pushConstantRanges.size() == 1u);
    CHECK(prepass.pushConstantRanges[0].stageFlags == vk::ShaderStageFlagBits::eFragment);
    CHECK(prepass.pushConstantRanges[0].offset == 0u);
    // The WHOLE block, even though the stage reads one field of it: the range must cover what the
    // shader declares, and both stages share one declaration (shaders/forward_push.glsl).
    CHECK(prepass.pushConstantRanges[0].size == sizeof(ForwardPushConstants));
    // Same vertex path as the forward pass, so the depth it writes and the UVs it tests are the
    // forward pass' own — that identity is what makes the two discard the same fragments.
    CHECK(prepass.vertShaderPath == forward.vertShaderPath);
    CHECK(prepass.fragShaderPath == "depth_prepass.frag.spv");
    CHECK(prepass.dynamicCullMode);
    CHECK(prepass.depthWrite);
}

TEST_CASE("PipelineConfig.ShadowConfigLeavesCullingToTheRecorder", "[PipelineConfig]")
{
    auto config = Pipeline::shadowConfig();

    // SH-05: cull mode is DYNAMIC, so the pipeline carries no static answer — a single-sided caster
    // (cull front), a double-sided one (cull nothing, which is what stops a sheet authored face-on
    // to the light from casting nothing at all) and the self-shadow first layer are all one
    // pipeline now. Shadows::recordPass sets it per draw from an explicit per-family policy.
    CHECK(config.dynamicCullMode);
    CHECK(config.depthBiasEnable);
    // The masked fragment path reads the bindless material authority, and the set index must be the
    // same 2 the forward pipelines use — see the empty set-1 layout in Pipeline's constructor.
    CHECK(config.bindlessSet);

    auto hasBinding = [&](ShadowBinding binding)
    {
        return std::any_of(config.bindings.begin(), config.bindings.end(), [&](const auto& entry)
                           { return entry.binding == bindingIndex(binding); });
    };

    CHECK(hasBinding(ShadowBinding::SelfShadowFirstMap));
    CHECK(hasBinding(ShadowBinding::SelfShadowDepthSampler));
}

TEST_CASE("PipelineConfig.ShadowMaterialModesAreTwoPipelinesEach", "[PipelineConfig]")
{
    // SH-05's four shadow material modes are (alpha x sidedness), and only the ALPHA half needs a
    // pipeline: a different fragment shader, everything else identical. The sidedness half is
    // dynamic cull state, which is why there are two pipelines here and not four.
    const auto opaque = Pipeline::shadowConfig();
    const auto masked = Pipeline::shadowMaskedConfig();

    CHECK(opaque.fragShaderPath == "shadow.frag.spv");
    CHECK(masked.fragShaderPath == "shadow_masked.frag.spv");
    CHECK(masked.vertShaderPath == opaque.vertShaderPath);
    CHECK(masked.dynamicCullMode == opaque.dynamicCullMode);
    CHECK(masked.bindlessSet == opaque.bindlessSet);
    CHECK(masked.depthFormat == opaque.depthFormat);
    CHECK(masked.depthCompare == opaque.depthCompare);
    CHECK(masked.depthBiasEnable == opaque.depthBiasEnable);
    CHECK(masked.bindings.size() == opaque.bindings.size());
}

TEST_CASE("PipelineConfig.SelfShadowSecondLayerHasItsOwnFragmentPaths", "[PipelineConfig]")
{
    // The FIRST self-shadow layer no longer has a config: it differed from shadowConfig in cull
    // mode alone, and SH-05 made that dynamic state, so the pass records the main pipelines with an
    // all-faces policy. The SECOND layer keeps its own pair — its fragment shader samples the first
    // layer's depth and rejects same-surface fragments, and its masked variant applies the cutout
    // before that test.
    const auto second = Pipeline::selfShadowSecondConfig();
    const auto secondMasked = Pipeline::selfShadowSecondMaskedConfig();

    CHECK(second.fragShaderPath == "self_shadow_second.frag.spv");
    CHECK(secondMasked.fragShaderPath == "self_shadow_second_masked.frag.spv");
    CHECK(second.dynamicCullMode);
    CHECK(secondMasked.dynamicCullMode);
    CHECK(second.bindlessSet);
    CHECK(secondMasked.bindlessSet);
}

TEST_CASE("PipelineConfig.SkyboxConfigIncludesCubemapSamplerBinding", "[PipelineConfig]")
{
    auto config = Pipeline::skyboxConfig();

    REQUIRE(config.bindings.size() == 3u);
    CHECK(config.bindings[0].binding == 0u);
    CHECK(config.bindings[1].binding == 1u);
    CHECK(config.bindings[1].descriptorType == vk::DescriptorType::eCombinedImageSampler);
    CHECK(config.bindings[2].binding == 2u);
    CHECK(config.bindings[2].descriptorType == vk::DescriptorType::eUniformBuffer);
}

TEST_CASE("PipelineConfig.EnvironmentConvertConfigIncludesPanoramaSamplerBinding",
          "[PipelineConfig]")
{
    auto config = Pipeline::environmentConvertConfig({});

    REQUIRE(config.bindings.size() == 1u);
    CHECK(config.vertShaderPath == "skybox.vert.spv");
    CHECK(config.fragShaderPath == "environment_convert.frag.spv");
    CHECK_FALSE(config.useVertexInput);
    CHECK_FALSE(config.depthTestEnable);
    CHECK_FALSE(config.depthWrite);
    CHECK(config.bindings[0].binding == 0u);
    CHECK(config.bindings[0].descriptorType == vk::DescriptorType::eCombinedImageSampler);
    REQUIRE(config.pushConstantRanges.size() == 1u);
    CHECK(config.pushConstantRanges[0].stageFlags == vk::ShaderStageFlagBits::eFragment);
    CHECK(config.pushConstantRanges[0].offset == 0u);
    CHECK(config.pushConstantRanges[0].size ==
          static_cast<uint32_t>(sizeof(fire_engine::EnvironmentCaptureUBO)));
}

TEST_CASE("PipelineConfig.IrradianceConvolutionConfigIncludesCubemapSamplerBinding",
          "[PipelineConfig]")
{
    auto config = Pipeline::irradianceConvolutionConfig({});

    REQUIRE(config.bindings.size() == 1u);
    CHECK(config.vertShaderPath == "skybox.vert.spv");
    CHECK(config.fragShaderPath == "irradiance_convolution.frag.spv");
    CHECK_FALSE(config.useVertexInput);
    CHECK_FALSE(config.depthTestEnable);
    CHECK_FALSE(config.depthWrite);
    CHECK(config.bindings[0].binding == 0u);
    CHECK(config.bindings[0].descriptorType == vk::DescriptorType::eCombinedImageSampler);
    REQUIRE(config.pushConstantRanges.size() == 1u);
    CHECK(config.pushConstantRanges[0].stageFlags == vk::ShaderStageFlagBits::eFragment);
    CHECK(config.pushConstantRanges[0].offset == 0u);
    CHECK(config.pushConstantRanges[0].size ==
          static_cast<uint32_t>(sizeof(fire_engine::EnvironmentCaptureUBO)));
}

TEST_CASE("PipelineConfig.PrefilterEnvironmentConfigIncludesCubemapSamplerBinding",
          "[PipelineConfig]")
{
    auto config = Pipeline::prefilterEnvironmentConfig({});

    REQUIRE(config.bindings.size() == 1u);
    CHECK(config.vertShaderPath == "skybox.vert.spv");
    CHECK(config.fragShaderPath == "prefilter_environment.frag.spv");
    CHECK_FALSE(config.useVertexInput);
    CHECK_FALSE(config.depthTestEnable);
    CHECK_FALSE(config.depthWrite);
    CHECK(config.bindings[0].binding == 0u);
    CHECK(config.bindings[0].descriptorType == vk::DescriptorType::eCombinedImageSampler);
    REQUIRE(config.pushConstantRanges.size() == 1u);
    CHECK(config.pushConstantRanges[0].stageFlags == vk::ShaderStageFlagBits::eFragment);
    CHECK(config.pushConstantRanges[0].offset == 0u);
    CHECK(config.pushConstantRanges[0].size ==
          static_cast<uint32_t>(sizeof(fire_engine::EnvironmentPrefilterPushConstants)));
}

TEST_CASE("PipelineConfig.BrdfIntegrationConfigUsesNoBindings", "[PipelineConfig]")
{
    auto config = Pipeline::brdfIntegrationConfig({});

    CHECK(config.bindings.empty());
    CHECK(config.vertShaderPath == "postprocess.vert.spv");
    CHECK(config.fragShaderPath == "brdf_integration.frag.spv");
    CHECK_FALSE(config.useVertexInput);
    CHECK_FALSE(config.depthTestEnable);
    CHECK_FALSE(config.depthWrite);
    CHECK(config.pushConstantRanges.empty());
}
