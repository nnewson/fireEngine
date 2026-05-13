#include <gtest/gtest.h>

#include <algorithm>

#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/ubo.hpp>

using fire_engine::bindingIndex;
using fire_engine::ForwardBinding;
using fire_engine::Pipeline;
using fire_engine::ShadowBinding;

TEST(PipelineConfig, ForwardConfigBindingsSplitBetweenSets)
{
    using fire_engine::ForwardGlobalBinding;
    auto config = Pipeline::forwardConfig({});

    // Set 0 — per-object / per-material bindings: frame, material, skin,
    // morph UBO + SSBO, base + 9 material textures (10 total).
    EXPECT_EQ(config.bindings.size(), 15u);
    // Set 1 — forward globals: light UBO + 5 shadow maps + debug image + 2
    // standalone samplers + 3 IBL textures + sceneColor.
    EXPECT_EQ(config.globalBindings.size(), 13u);

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

    // Per-object set 0 bindings.
    EXPECT_TRUE(hasObjectBinding(ForwardBinding::TransmissionTexture));
    EXPECT_TRUE(hasObjectBinding(ForwardBinding::ClearcoatTexture));
    EXPECT_TRUE(hasObjectBinding(ForwardBinding::ClearcoatRoughnessTexture));
    EXPECT_TRUE(hasObjectBinding(ForwardBinding::ClearcoatNormalTexture));
    EXPECT_TRUE(hasObjectBinding(ForwardBinding::ThicknessTexture));

    // Forward globals set 1 bindings.
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::Light));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::IrradianceMap));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::PrefilteredMap));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::BrdfLut));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::ShadowCompareSampler));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::SceneColour));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::SpotShadowMap));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::PointShadowMap));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::ShadowDebugSampler));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::ShadowDebugImage));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::WorldShadowMap));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::SelfShadowMap));
    EXPECT_TRUE(hasGlobalBinding(ForwardGlobalBinding::ShadowMap));
}

TEST(PipelineConfig, ForwardConfigIncludesSelfShadowPushConstant)
{
    auto config = Pipeline::forwardConfig({});

    ASSERT_EQ(config.pushConstantRanges.size(), 1u);
    EXPECT_EQ(config.pushConstantRanges[0].stageFlags, vk::ShaderStageFlagBits::eFragment);
    EXPECT_EQ(config.pushConstantRanges[0].offset, 0u);
    EXPECT_EQ(config.pushConstantRanges[0].size,
              static_cast<uint32_t>(sizeof(fire_engine::ForwardPushConstants)));
}

TEST(PipelineConfig, ShadowConfigCullsFrontFaces)
{
    auto config = Pipeline::shadowConfig({});

    EXPECT_EQ(config.cullMode, vk::CullModeFlagBits::eFront);
    EXPECT_TRUE(config.depthBiasEnable);

    auto hasBinding = [&](ShadowBinding binding)
    {
        return std::any_of(config.bindings.begin(), config.bindings.end(), [&](const auto& entry)
                           { return entry.binding == bindingIndex(binding); });
    };

    EXPECT_TRUE(hasBinding(ShadowBinding::SelfShadowFirstMap));
    EXPECT_TRUE(hasBinding(ShadowBinding::SelfShadowDepthSampler));
}

TEST(PipelineConfig, SelfShadowConfigsCulling)
{
    // First pass rasterises both faces so it captures the light-facing depth.
    // Second pass culls front faces so only back-facing fragments survive,
    // which keeps the in-shader discard threshold from flipping on marginal
    // fragments and producing per-pixel flicker.
    auto first = Pipeline::selfShadowFirstConfig({});
    auto second = Pipeline::selfShadowSecondConfig({});

    EXPECT_EQ(first.cullMode, vk::CullModeFlagBits::eNone);
    EXPECT_EQ(first.fragShaderPath, "shadow.frag.spv");
    EXPECT_EQ(second.cullMode, vk::CullModeFlagBits::eFront);
    EXPECT_EQ(second.fragShaderPath, "self_shadow_second.frag.spv");
}

TEST(PipelineConfig, SkyboxConfigIncludesCubemapSamplerBinding)
{
    auto config = Pipeline::skyboxConfig({});

    ASSERT_EQ(config.bindings.size(), 3u);
    EXPECT_EQ(config.bindings[0].binding, 0u);
    EXPECT_EQ(config.bindings[1].binding, 1u);
    EXPECT_EQ(config.bindings[1].descriptorType, vk::DescriptorType::eCombinedImageSampler);
    EXPECT_EQ(config.bindings[2].binding, 2u);
    EXPECT_EQ(config.bindings[2].descriptorType, vk::DescriptorType::eUniformBuffer);
}

TEST(PipelineConfig, EnvironmentConvertConfigIncludesPanoramaSamplerBinding)
{
    auto config = Pipeline::environmentConvertConfig({});

    ASSERT_EQ(config.bindings.size(), 1u);
    EXPECT_EQ(config.vertShaderPath, "skybox.vert.spv");
    EXPECT_EQ(config.fragShaderPath, "environment_convert.frag.spv");
    EXPECT_FALSE(config.useVertexInput);
    EXPECT_FALSE(config.depthTestEnable);
    EXPECT_FALSE(config.depthWrite);
    EXPECT_EQ(config.bindings[0].binding, 0u);
    EXPECT_EQ(config.bindings[0].descriptorType, vk::DescriptorType::eCombinedImageSampler);
    ASSERT_EQ(config.pushConstantRanges.size(), 1u);
    EXPECT_EQ(config.pushConstantRanges[0].stageFlags, vk::ShaderStageFlagBits::eFragment);
    EXPECT_EQ(config.pushConstantRanges[0].offset, 0u);
    EXPECT_EQ(config.pushConstantRanges[0].size,
              static_cast<uint32_t>(sizeof(fire_engine::EnvironmentCaptureUBO)));
}

TEST(PipelineConfig, IrradianceConvolutionConfigIncludesCubemapSamplerBinding)
{
    auto config = Pipeline::irradianceConvolutionConfig({});

    ASSERT_EQ(config.bindings.size(), 1u);
    EXPECT_EQ(config.vertShaderPath, "skybox.vert.spv");
    EXPECT_EQ(config.fragShaderPath, "irradiance_convolution.frag.spv");
    EXPECT_FALSE(config.useVertexInput);
    EXPECT_FALSE(config.depthTestEnable);
    EXPECT_FALSE(config.depthWrite);
    EXPECT_EQ(config.bindings[0].binding, 0u);
    EXPECT_EQ(config.bindings[0].descriptorType, vk::DescriptorType::eCombinedImageSampler);
    ASSERT_EQ(config.pushConstantRanges.size(), 1u);
    EXPECT_EQ(config.pushConstantRanges[0].stageFlags, vk::ShaderStageFlagBits::eFragment);
    EXPECT_EQ(config.pushConstantRanges[0].offset, 0u);
    EXPECT_EQ(config.pushConstantRanges[0].size,
              static_cast<uint32_t>(sizeof(fire_engine::EnvironmentCaptureUBO)));
}

TEST(PipelineConfig, PrefilterEnvironmentConfigIncludesCubemapSamplerBinding)
{
    auto config = Pipeline::prefilterEnvironmentConfig({});

    ASSERT_EQ(config.bindings.size(), 1u);
    EXPECT_EQ(config.vertShaderPath, "skybox.vert.spv");
    EXPECT_EQ(config.fragShaderPath, "prefilter_environment.frag.spv");
    EXPECT_FALSE(config.useVertexInput);
    EXPECT_FALSE(config.depthTestEnable);
    EXPECT_FALSE(config.depthWrite);
    EXPECT_EQ(config.bindings[0].binding, 0u);
    EXPECT_EQ(config.bindings[0].descriptorType, vk::DescriptorType::eCombinedImageSampler);
    ASSERT_EQ(config.pushConstantRanges.size(), 1u);
    EXPECT_EQ(config.pushConstantRanges[0].stageFlags, vk::ShaderStageFlagBits::eFragment);
    EXPECT_EQ(config.pushConstantRanges[0].offset, 0u);
    EXPECT_EQ(config.pushConstantRanges[0].size,
              static_cast<uint32_t>(sizeof(fire_engine::EnvironmentPrefilterPushConstants)));
}

TEST(PipelineConfig, BrdfIntegrationConfigUsesNoBindings)
{
    auto config = Pipeline::brdfIntegrationConfig({});

    EXPECT_TRUE(config.bindings.empty());
    EXPECT_EQ(config.vertShaderPath, "postprocess.vert.spv");
    EXPECT_EQ(config.fragShaderPath, "brdf_integration.frag.spv");
    EXPECT_FALSE(config.useVertexInput);
    EXPECT_FALSE(config.depthTestEnable);
    EXPECT_FALSE(config.depthWrite);
    EXPECT_TRUE(config.pushConstantRanges.empty());
}
