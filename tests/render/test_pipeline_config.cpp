#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/ubo.hpp>

using fire_engine::bindingIndex;
using fire_engine::ForwardBinding;
using fire_engine::Pipeline;
using fire_engine::ShadowBinding;

TEST_CASE("PipelineConfig.ForwardConfigBindingsSplitBetweenSets", "[PipelineConfig]")
{
    using fire_engine::ForwardGlobalBinding;
    auto config = Pipeline::forwardConfig();

    // Set 0 — per-object vertex-stage UBOs/SSBO only now: frame, skin, morph UBO +
    // morph-targets SSBO (4 total). Material data (textures + scalars) is fully
    // bindless (set 2).
    CHECK(config.bindings.size() == 4u);
    // Set 1 — forward globals: light UBO + 5 shadow maps + debug image + 2
    // standalone samplers + 3 IBL textures + sceneColor.
    CHECK(config.globalBindings.size() == 13u);
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
    CHECK(hasObjectBinding(ForwardBinding::Skin));
    CHECK(hasObjectBinding(ForwardBinding::Morph));
    CHECK(hasObjectBinding(ForwardBinding::MorphTargets));
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
    CHECK(Pipeline::selfShadowFirstConfig().pushDescriptorSet0);
    CHECK(Pipeline::selfShadowSecondConfig().pushDescriptorSet0);
    CHECK_FALSE(Pipeline::skyboxConfig().pushDescriptorSet0);
}

TEST_CASE("PipelineConfig.ShadowConfigCullsFrontFaces", "[PipelineConfig]")
{
    auto config = Pipeline::shadowConfig();

    CHECK(config.cullMode == vk::CullModeFlagBits::eFront);
    CHECK(config.depthBiasEnable);

    auto hasBinding = [&](ShadowBinding binding)
    {
        return std::any_of(config.bindings.begin(), config.bindings.end(), [&](const auto& entry)
                           { return entry.binding == bindingIndex(binding); });
    };

    CHECK(hasBinding(ShadowBinding::SelfShadowFirstMap));
    CHECK(hasBinding(ShadowBinding::SelfShadowDepthSampler));
}

TEST_CASE("PipelineConfig.SelfShadowConfigsCulling", "[PipelineConfig]")
{
    // First pass rasterises both faces so it captures the light-facing depth.
    // Second pass culls front faces so only back-facing fragments survive,
    // which keeps the in-shader discard threshold from flipping on marginal
    // fragments and producing per-pixel flicker.
    auto first = Pipeline::selfShadowFirstConfig();
    auto second = Pipeline::selfShadowSecondConfig();

    CHECK(first.cullMode == vk::CullModeFlagBits::eNone);
    CHECK(first.fragShaderPath == "shadow.frag.spv");
    CHECK(second.cullMode == vk::CullModeFlagBits::eFront);
    CHECK(second.fragShaderPath == "self_shadow_second.frag.spv");
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
