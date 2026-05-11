#include <cstddef>

#include <fire_engine/core/shader_loader.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

Pipeline::Pipeline(const Device& device, const PipelineConfig& config)
    : device_(&device.device())
{
    createDescriptorSetLayout(config.bindings);
    createGraphicsPipeline(config);
}

PipelineConfig Pipeline::forwardConfig(vk::RenderPass renderPass)
{
    auto uniform = [](ForwardBinding binding, vk::ShaderStageFlags stages)
    {
        return vk::DescriptorSetLayoutBinding{bindingIndex(binding),
                                              vk::DescriptorType::eUniformBuffer, 1, stages};
    };
    auto sampler = [](ForwardBinding binding)
    {
        return vk::DescriptorSetLayoutBinding{bindingIndex(binding),
                                              vk::DescriptorType::eCombinedImageSampler, 1,
                                              vk::ShaderStageFlagBits::eFragment};
    };
    auto sampledImage = [](ForwardBinding binding)
    {
        return vk::DescriptorSetLayoutBinding{bindingIndex(binding),
                                              vk::DescriptorType::eSampledImage, 1,
                                              vk::ShaderStageFlagBits::eFragment};
    };
    auto plainSampler = [](ForwardBinding binding)
    {
        return vk::DescriptorSetLayoutBinding{bindingIndex(binding), vk::DescriptorType::eSampler,
                                              1, vk::ShaderStageFlagBits::eFragment};
    };

    PipelineConfig config;
    config.vertShaderPath = "shader.vert.spv";
    config.fragShaderPath = "shader.frag.spv";
    config.bindings = {
        uniform(ForwardBinding::Frame,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
        uniform(ForwardBinding::Material, vk::ShaderStageFlagBits::eFragment),
        sampler(ForwardBinding::BaseColourTexture),
        uniform(ForwardBinding::Skin, vk::ShaderStageFlagBits::eVertex),
        uniform(ForwardBinding::Morph, vk::ShaderStageFlagBits::eVertex),
        {bindingIndex(ForwardBinding::MorphTargets), vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eVertex},
        sampler(ForwardBinding::EmissiveTexture),
        sampler(ForwardBinding::NormalTexture),
        sampler(ForwardBinding::MetallicRoughnessTexture),
        sampler(ForwardBinding::OcclusionTexture),
        sampledImage(ForwardBinding::ShadowMap),
        uniform(ForwardBinding::Light, vk::ShaderStageFlagBits::eFragment),
        sampler(ForwardBinding::IrradianceMap),
        sampler(ForwardBinding::PrefilteredMap),
        sampler(ForwardBinding::BrdfLut),
        plainSampler(ForwardBinding::ShadowCompareSampler),
        sampler(ForwardBinding::TransmissionTexture),
        sampler(ForwardBinding::ClearcoatTexture),
        sampler(ForwardBinding::ClearcoatRoughnessTexture),
        sampler(ForwardBinding::ClearcoatNormalTexture),
        sampler(ForwardBinding::SceneColour),
        sampler(ForwardBinding::ThicknessTexture),
        sampledImage(ForwardBinding::SpotShadowMap),
        sampledImage(ForwardBinding::PointShadowMap),
        plainSampler(ForwardBinding::ShadowDebugSampler),
        sampledImage(ForwardBinding::ShadowDebugImage),
        sampledImage(ForwardBinding::WorldShadowMap),
        sampledImage(ForwardBinding::SelfShadowMap),
    };
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eFragment, 0,
                                           static_cast<uint32_t>(sizeof(ForwardPushConstants)));
    config.renderPass = renderPass;
    return config;
}

PipelineConfig Pipeline::forwardDoubleSidedConfig(vk::RenderPass renderPass)
{
    PipelineConfig config = forwardConfig(renderPass);
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::forwardBlendConfig(vk::RenderPass renderPass)
{
    PipelineConfig config = forwardConfig(renderPass);
    config.cullMode = vk::CullModeFlagBits::eNone;
    config.depthWrite = false;
    config.blendEnable = true;
    config.srcColourBlend = vk::BlendFactor::eSrcAlpha;
    config.dstColourBlend = vk::BlendFactor::eOneMinusSrcAlpha;
    config.srcAlphaBlend = vk::BlendFactor::eOne;
    config.dstAlphaBlend = vk::BlendFactor::eOneMinusSrcAlpha;
    return config;
}

PipelineConfig Pipeline::shadowConfig(vk::RenderPass renderPass)
{
    PipelineConfig config;
    config.vertShaderPath = "shadow.vert.spv";
    config.fragShaderPath = "shadow.frag.spv";
    config.bindings = {
        {bindingIndex(ShadowBinding::Shadow), vk::DescriptorType::eUniformBuffer, 1,
         vk::ShaderStageFlagBits::eVertex},
        {bindingIndex(ShadowBinding::Skin), vk::DescriptorType::eUniformBuffer, 1,
         vk::ShaderStageFlagBits::eVertex},
        {bindingIndex(ShadowBinding::Morph), vk::DescriptorType::eUniformBuffer, 1,
         vk::ShaderStageFlagBits::eVertex},
        {bindingIndex(ShadowBinding::MorphTargets), vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eVertex},
        {bindingIndex(ShadowBinding::SelfShadowFirstMap), vk::DescriptorType::eSampledImage, 1,
         vk::ShaderStageFlagBits::eFragment},
        {bindingIndex(ShadowBinding::SelfShadowDepthSampler), vk::DescriptorType::eSampler, 1,
         vk::ShaderStageFlagBits::eFragment},
    };
    // matrixIndex picks lightViewProj[] in the vertex stage. lightPosRange is
    // consumed by the fragment shader's point-shadow branch (linear distance
    // depth), so the push constant must be visible to both stages.
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eVertex |
                                               vk::ShaderStageFlagBits::eFragment,
                                           0, static_cast<uint32_t>(sizeof(ShadowPushConstants)));
    config.renderPass = renderPass;
    config.depthWrite = true;
    config.depthCompare = vk::CompareOp::eLessOrEqual;
    config.cullMode = vk::CullModeFlagBits::eFront;
    config.writeColour = true;
    config.depthBiasEnable = true;
    config.depthBiasConstant = 0.0f;
    config.depthBiasSlope = 0.0f;
    return config;
}

PipelineConfig Pipeline::selfShadowSecondConfig(vk::RenderPass renderPass)
{
    PipelineConfig config = shadowConfig(renderPass);
    config.fragShaderPath = "self_shadow_second.frag.spv";
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::selfShadowFirstConfig(vk::RenderPass renderPass)
{
    PipelineConfig config = shadowConfig(renderPass);
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::postProcessConfig(vk::RenderPass renderPass)
{
    PipelineConfig config;
    config.vertShaderPath = "postprocess.vert.spv";
    config.fragShaderPath = "postprocess.frag.spv";
    config.bindings = {
        {bindingIndex(PostProcessBinding::HdrInput), vk::DescriptorType::eCombinedImageSampler, 1,
         vk::ShaderStageFlagBits::eFragment},
        // 1: bloom mip 0 — added by Stage 6.
        {bindingIndex(PostProcessBinding::BloomInput), vk::DescriptorType::eCombinedImageSampler, 1,
         vk::ShaderStageFlagBits::eFragment},
    };
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eFragment, 0,
                                           static_cast<uint32_t>(sizeof(PostProcessPushConstants)));
    config.renderPass = renderPass;
    config.useVertexInput = false;
    config.depthTestEnable = false;
    config.depthWrite = false;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::bloomDownsampleConfig(vk::RenderPass renderPass)
{
    PipelineConfig config;
    config.vertShaderPath = "postprocess.vert.spv";
    config.fragShaderPath = "bloom_downsample.frag.spv";
    config.bindings = {
        {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
    };
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eFragment, 0,
                                           static_cast<uint32_t>(sizeof(BloomPushConstants)));
    config.renderPass = renderPass;
    config.useVertexInput = false;
    config.depthTestEnable = false;
    config.depthWrite = false;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::bloomUpsampleConfig(vk::RenderPass renderPass)
{
    PipelineConfig config = bloomDownsampleConfig(renderPass);
    config.fragShaderPath = "bloom_upsample.frag.spv";
    // Additive blend so each upsample step adds its tent contribution onto
    // the existing mip content (which was loaded via render-pass eLoad).
    config.blendEnable = true;
    config.srcColourBlend = vk::BlendFactor::eOne;
    config.dstColourBlend = vk::BlendFactor::eOne;
    config.srcAlphaBlend = vk::BlendFactor::eOne;
    config.dstAlphaBlend = vk::BlendFactor::eOne;
    return config;
}

PipelineConfig Pipeline::skyboxConfig(vk::RenderPass renderPass)
{
    PipelineConfig config;
    config.vertShaderPath = "skybox.vert.spv";
    config.fragShaderPath = "skybox.frag.spv";
    config.bindings = {
        {bindingIndex(SkyboxBinding::Skybox), vk::DescriptorType::eUniformBuffer, 1,
         vk::ShaderStageFlagBits::eFragment},
        {bindingIndex(SkyboxBinding::Cubemap), vk::DescriptorType::eCombinedImageSampler, 1,
         vk::ShaderStageFlagBits::eFragment},
        {bindingIndex(SkyboxBinding::Light), vk::DescriptorType::eUniformBuffer, 1,
         vk::ShaderStageFlagBits::eFragment},
    };
    config.renderPass = renderPass;
    config.useVertexInput = false;
    config.depthWrite = false;
    config.depthCompare = vk::CompareOp::eLessOrEqual;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::environmentConvertConfig(vk::RenderPass renderPass)
{
    PipelineConfig config;
    config.vertShaderPath = "skybox.vert.spv";
    config.fragShaderPath = "environment_convert.frag.spv";
    config.bindings = {
        {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
    };
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eFragment, 0,
                                           static_cast<uint32_t>(sizeof(EnvironmentCaptureUBO)));
    config.renderPass = renderPass;
    config.useVertexInput = false;
    config.depthTestEnable = false;
    config.depthWrite = false;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::irradianceConvolutionConfig(vk::RenderPass renderPass)
{
    PipelineConfig config;
    config.vertShaderPath = "skybox.vert.spv";
    config.fragShaderPath = "irradiance_convolution.frag.spv";
    config.bindings = {
        {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
    };
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eFragment, 0,
                                           static_cast<uint32_t>(sizeof(EnvironmentCaptureUBO)));
    config.renderPass = renderPass;
    config.useVertexInput = false;
    config.depthTestEnable = false;
    config.depthWrite = false;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::prefilterEnvironmentConfig(vk::RenderPass renderPass)
{
    PipelineConfig config;
    config.vertShaderPath = "skybox.vert.spv";
    config.fragShaderPath = "prefilter_environment.frag.spv";
    config.bindings = {
        {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
    };
    config.pushConstantRanges.emplace_back(
        vk::ShaderStageFlagBits::eFragment, 0,
        static_cast<uint32_t>(sizeof(EnvironmentPrefilterPushConstants)));
    config.renderPass = renderPass;
    config.useVertexInput = false;
    config.depthTestEnable = false;
    config.depthWrite = false;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::brdfIntegrationConfig(vk::RenderPass renderPass)
{
    PipelineConfig config;
    config.vertShaderPath = "postprocess.vert.spv";
    config.fragShaderPath = "brdf_integration.frag.spv";
    config.renderPass = renderPass;
    config.useVertexInput = false;
    config.depthTestEnable = false;
    config.depthWrite = false;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

void Pipeline::createDescriptorSetLayout(
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings)
{
    vk::DescriptorSetLayoutCreateInfo ci{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    descSetLayout_ = vk::raii::DescriptorSetLayout(*device_, ci);
}

void Pipeline::createGraphicsPipeline(const PipelineConfig& config)
{
    vk::raii::ShaderModule vertMod{nullptr};
    vk::raii::ShaderModule fragMod{nullptr};
    auto stages = createShaderStages(config, vertMod, fragMod);

    vk::VertexInputBindingDescription bindDesc{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
    std::array<vk::VertexInputAttributeDescription, 8> attrDesc = {{
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(Vertex, position_)),
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(Vertex, colour_)),
        },
        vk::VertexInputAttributeDescription{
            .location = 2,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(Vertex, normal_)),
        },
        vk::VertexInputAttributeDescription{
            .location = 3,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(Vertex, texCoord_)),
        },
        vk::VertexInputAttributeDescription{
            .location = 4,
            .binding = 0,
            .format = vk::Format::eR32G32B32A32Uint,
            .offset = static_cast<uint32_t>(offsetof(Vertex, joints_)),
        },
        vk::VertexInputAttributeDescription{
            .location = 5,
            .binding = 0,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(Vertex, weights_)),
        },
        vk::VertexInputAttributeDescription{
            .location = 6,
            .binding = 0,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(Vertex, tangent_)),
        },
        vk::VertexInputAttributeDescription{
            .location = 8,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(Vertex, texCoord1_)),
        },
    }};
    vk::PipelineVertexInputStateCreateInfo vertInput{};
    if (config.useVertexInput)
    {
        vertInput = vk::PipelineVertexInputStateCreateInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindDesc,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size()),
            .pVertexAttributeDescriptions = attrDesc.data(),
        };
    }

    vk::PipelineInputAssemblyStateCreateInfo inputAsm{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    vk::PipelineViewportStateCreateInfo vpState{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    std::array dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor,
                                vk::DynamicState::eDepthBias};
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    vk::PipelineRasterizationStateCreateInfo raster{
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = config.cullMode,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = config.depthBiasEnable,
        .depthBiasConstantFactor = config.depthBiasConstant,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = config.depthBiasSlope,
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo ms{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = config.depthTestEnable,
        .depthWriteEnable = config.depthWrite,
        .depthCompareOp = config.depthCompare,
    };

    vk::ColorComponentFlags writeMask =
        config.writeColour ? vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                 vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                           : vk::ColorComponentFlags{};
    vk::PipelineColorBlendAttachmentState colourBlendAtt{
        .blendEnable = config.blendEnable,
        .srcColorBlendFactor = config.srcColourBlend,
        .dstColorBlendFactor = config.dstColourBlend,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = config.srcAlphaBlend,
        .dstAlphaBlendFactor = config.dstAlphaBlend,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = writeMask,
    };

    vk::PipelineColorBlendStateCreateInfo colourBlend{
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &colourBlendAtt,
    };

    createPipelineLayout(config);

    const vk::PipelineColorBlendStateCreateInfo* colourBlendPtr = &colourBlend;

    vk::GraphicsPipelineCreateInfo pci{
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertInput,
        .pInputAssemblyState = &inputAsm,
        .pViewportState = &vpState,
        .pRasterizationState = &raster,
        .pMultisampleState = &ms,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = colourBlendPtr,
        .pDynamicState = &dynamicState,
        .layout = *pipelineLayout_,
        .renderPass = config.renderPass,
        .subpass = 0,
    };

    pipeline_ = vk::raii::Pipeline(*device_, nullptr, pci);
}

std::array<vk::PipelineShaderStageCreateInfo, 2>
Pipeline::createShaderStages(const PipelineConfig& config, vk::raii::ShaderModule& vertMod,
                             vk::raii::ShaderModule& fragMod) const
{
    auto vertCode = ShaderLoader::load_from_file(config.vertShaderPath);
    auto fragCode = ShaderLoader::load_from_file(config.fragShaderPath);
    vk::ShaderModuleCreateInfo vertCi{
        .codeSize = vertCode.size(),
        .pCode = reinterpret_cast<const uint32_t*>(vertCode.data()),
    };
    vk::ShaderModuleCreateInfo fragCi{
        .codeSize = fragCode.size(),
        .pCode = reinterpret_cast<const uint32_t*>(fragCode.data()),
    };
    vertMod = vk::raii::ShaderModule(*device_, vertCi);
    fragMod = vk::raii::ShaderModule(*device_, fragCi);

    return {{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *vertMod,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *fragMod,
            .pName = "main",
        },
    }};
}

void Pipeline::createPipelineLayout(const PipelineConfig& config)
{
    vk::PipelineLayoutCreateInfo plci{
        .setLayoutCount = 1,
        .pSetLayouts = &*descSetLayout_,
        .pushConstantRangeCount = static_cast<uint32_t>(config.pushConstantRanges.size()),
        .pPushConstantRanges = config.pushConstantRanges.data(),
    };
    pipelineLayout_ = vk::raii::PipelineLayout(*device_, plci);
}

} // namespace fire_engine
