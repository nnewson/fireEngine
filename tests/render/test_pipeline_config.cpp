#include <gtest/gtest.h>

#include <algorithm>

#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/ubo.hpp>

using fire_engine::bindingIndex;
using fire_engine::ForwardBinding;
using fire_engine::Pipeline;

TEST(PipelineConfig, ForwardConfigIncludesIblBindings)
{
    auto config = Pipeline::forwardConfig({});

    EXPECT_EQ(config.bindings.size(), 26u);

    auto hasBinding = [&](ForwardBinding binding)
    {
        return std::any_of(config.bindings.begin(), config.bindings.end(), [&](const auto& entry)
                           { return entry.binding == bindingIndex(binding); });
    };

    EXPECT_TRUE(hasBinding(ForwardBinding::IrradianceMap));
    EXPECT_TRUE(hasBinding(ForwardBinding::PrefilteredMap));
    EXPECT_TRUE(hasBinding(ForwardBinding::BrdfLut));
    // Shared comparison sampler — used with the SampledImage shadow bindings
    // (10/22/23) via GLSL sampler*() constructors. Apple caps per-stage
    // samplers at 16.
    EXPECT_TRUE(hasBinding(ForwardBinding::ShadowCompareSampler));
    // KHR_materials_transmission texture.
    EXPECT_TRUE(hasBinding(ForwardBinding::TransmissionTexture));
    // KHR_materials_clearcoat: factor (R), roughness (G), normal (RGB).
    EXPECT_TRUE(hasBinding(ForwardBinding::ClearcoatTexture));
    EXPECT_TRUE(hasBinding(ForwardBinding::ClearcoatRoughnessTexture));
    EXPECT_TRUE(hasBinding(ForwardBinding::ClearcoatNormalTexture));
    // KHR_materials_transmission F3 — sceneColor mip chain.
    EXPECT_TRUE(hasBinding(ForwardBinding::SceneColour));
    // KHR_materials_volume — thickness texture (G channel).
    EXPECT_TRUE(hasBinding(ForwardBinding::ThicknessTexture));
    // Punctual light shadows: spot 2D-array + point cubemap-array depth maps.
    EXPECT_TRUE(hasBinding(ForwardBinding::SpotShadowMap));
    EXPECT_TRUE(hasBinding(ForwardBinding::PointShadowMap));
    EXPECT_TRUE(hasBinding(ForwardBinding::ShadowDebugSampler));
    EXPECT_TRUE(hasBinding(ForwardBinding::ShadowDebugImage));
}

TEST(PipelineConfig, ShadowConfigCullsFrontFaces)
{
    auto config = Pipeline::shadowConfig({});

    EXPECT_EQ(config.cullMode, vk::CullModeFlagBits::eFront);
    EXPECT_TRUE(config.depthBiasEnable);
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
