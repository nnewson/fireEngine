#include <cstddef>
#include <utility>

#include <fire_engine/core/shader_loader.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

Pipeline::Pipeline(const Device& device, const PipelineConfig& config)
    : device_(&device.device()),
      pipelineCache_(&device.pipelineCache())
{
    createDescriptorSetLayout(config.bindings, config.pushDescriptorSet0);
    if (!config.globalBindings.empty())
    {
        vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = static_cast<uint32_t>(config.globalBindings.size()),
            .pBindings = config.globalBindings.data(),
        };
        globalDescSetLayout_ = vk::raii::DescriptorSetLayout(*device_, ci);
    }
    if (config.bindlessSet)
    {
        createBindlessDescriptorSetLayout();
    }
    createGraphicsPipeline(config);
}

namespace
{

[[nodiscard]]
vk::DescriptorSetLayoutBinding fragmentBinding(uint32_t binding, vk::DescriptorType type)
{
    return vk::DescriptorSetLayoutBinding{binding, type, 1, vk::ShaderStageFlagBits::eFragment};
}

[[nodiscard]]
vk::DescriptorSetLayoutBinding combinedImageSamplerBinding(uint32_t binding)
{
    return fragmentBinding(binding, vk::DescriptorType::eCombinedImageSampler);
}

[[nodiscard]]
PipelineConfig fullscreenConfig(std::string vertShaderPath, std::string fragShaderPath,
                                vk::Format colourFormat)
{
    PipelineConfig config;
    config.vertShaderPath = std::move(vertShaderPath);
    config.fragShaderPath = std::move(fragShaderPath);
    config.colourFormats = {colourFormat};
    config.useVertexInput = false;
    config.depthTestEnable = false;
    config.depthWrite = false;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

void addFragmentPushConstant(PipelineConfig& config, uint32_t size)
{
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eFragment, 0, size);
}

void enableAdditiveBlend(PipelineConfig& config)
{
    config.blendEnable = true;
    config.srcColourBlend = vk::BlendFactor::eOne;
    config.dstColourBlend = vk::BlendFactor::eOne;
    config.srcAlphaBlend = vk::BlendFactor::eOne;
    config.dstAlphaBlend = vk::BlendFactor::eOne;
}

} // namespace

void Pipeline::createBindlessDescriptorSetLayout()
{
    // Binding 0: a partially-bound, update-after-bind combined-image-sampler array
    // indexed in-shader by texture handle (partially-bound lets the array be sparse;
    // update-after-bind lets Resources write a texture into a slot after the set has
    // already been bound for earlier frames). Binding 1: the global materials[]
    // SSBO, indexed by the per-draw material index.
    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
        {bindingIndex(BindlessBinding::Textures), vk::DescriptorType::eCombinedImageSampler,
         kMaxBindlessTextures, vk::ShaderStageFlagBits::eFragment},
        {bindingIndex(BindlessBinding::Materials), vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eFragment},
    }};
    const std::array<vk::DescriptorBindingFlags, 2> bindingFlags{
        // Textures: sparse + written as textures load (after the set is bound).
        vk::DescriptorBindingFlagBits::ePartiallyBound |
            vk::DescriptorBindingFlagBits::eUpdateAfterBind,
        // Materials SSBO: written once at startup before the set is ever bound, so
        // it needs no binding flags (no update-after-bind feature dependency).
        vk::DescriptorBindingFlags{},
    };
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        .bindingCount = static_cast<uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };
    const vk::DescriptorSetLayoutCreateInfo ci{
        .pNext = &flagsInfo,
        .flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    bindlessDescSetLayout_ = vk::raii::DescriptorSetLayout(*device_, ci);
}

PipelineConfig Pipeline::forwardConfig()
{
    auto uniform = [](ForwardBinding binding, vk::ShaderStageFlags stages)
    {
        return vk::DescriptorSetLayoutBinding{bindingIndex(binding),
                                              vk::DescriptorType::eUniformBuffer, 1, stages};
    };
    auto globalSampler = [](ForwardGlobalBinding binding)
    {
        return vk::DescriptorSetLayoutBinding{bindingIndex(binding),
                                              vk::DescriptorType::eCombinedImageSampler, 1,
                                              vk::ShaderStageFlagBits::eFragment};
    };
    auto globalSampledImage = [](ForwardGlobalBinding binding)
    {
        return vk::DescriptorSetLayoutBinding{bindingIndex(binding),
                                              vk::DescriptorType::eSampledImage, 1,
                                              vk::ShaderStageFlagBits::eFragment};
    };
    auto globalPlainSampler = [](ForwardGlobalBinding binding)
    {
        return vk::DescriptorSetLayoutBinding{bindingIndex(binding), vk::DescriptorType::eSampler,
                                              1, vk::ShaderStageFlagBits::eFragment};
    };

    PipelineConfig config;
    config.vertShaderPath = "shader.vert.spv";
    config.fragShaderPath = "shader.frag.spv";
    // Set 0 — per-object vertex-stage UBOs/SSBO only. Material data (textures +
    // scalars) is fully bindless now: textures in the set-2 array, scalars in the
    // set-2 materials SSBO indexed by push constant. Shared globals (light, shadow
    // maps, IBL, sceneColor) live on set 1 below. Bindings 1 + 2 (old Material UBO
    // and BaseColourTexture) are intentional gaps — gaps are legal and avoid
    // renumbering Skin/Morph/MorphTargets.
    config.bindings = {
        uniform(ForwardBinding::Frame,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
        uniform(ForwardBinding::Skin, vk::ShaderStageFlagBits::eVertex),
        uniform(ForwardBinding::Morph, vk::ShaderStageFlagBits::eVertex),
        {bindingIndex(ForwardBinding::MorphTargets), vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eVertex},
    };
    // Set 1 — forward globals. Bound once per frame in Renderer; survives
    // pipeline transitions within the forward bucket. See ForwardGlobalBinding
    // enum for the canonical numbering.
    config.globalBindings = {
        vk::DescriptorSetLayoutBinding{bindingIndex(ForwardGlobalBinding::Light),
                                       vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment},
        globalSampledImage(ForwardGlobalBinding::ShadowMap),
        globalSampledImage(ForwardGlobalBinding::WorldShadowMap),
        globalSampledImage(ForwardGlobalBinding::SelfShadowMap),
        globalSampledImage(ForwardGlobalBinding::SpotShadowMap),
        globalSampledImage(ForwardGlobalBinding::PointShadowMap),
        globalSampledImage(ForwardGlobalBinding::ShadowDebugImage),
        globalPlainSampler(ForwardGlobalBinding::ShadowCompareSampler),
        globalPlainSampler(ForwardGlobalBinding::ShadowDebugSampler),
        globalSampler(ForwardGlobalBinding::IrradianceMap),
        globalSampler(ForwardGlobalBinding::PrefilteredMap),
        globalSampler(ForwardGlobalBinding::BrdfLut),
        globalSampler(ForwardGlobalBinding::SceneColour),
    };
    // Set 0 is pushed inline per draw (VK_KHR_push_descriptor) — no per-object
    // descriptor-set allocation. forwardBlendConfig inherits this (it copies
    // forwardConfig), so blend draws push their set 0 too.
    config.pushDescriptorSet0 = true;
    // Set 2 — bindless material textures + materials SSBO, indexed in-shader.
    config.bindlessSet = true;
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eFragment, 0,
                                           static_cast<uint32_t>(sizeof(ForwardPushConstants)));
    // Forward target: HDR offscreen colour + RG16F motion vectors (TAA) + shared
    // D32 depth. The velocity format must match Resources::createVelocityTarget.
    config.colourFormats = {vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16Sfloat};
    config.depthFormat = vk::Format::eD32Sfloat;
    // Cull mode is dynamic so this one pipeline covers both single-sided
    // (cull back) and double-sided (cull none) materials; the renderer sets it
    // per draw from DrawCommand::doubleSided.
    config.dynamicCullMode = true;
    return config;
}

PipelineConfig Pipeline::forwardBlendConfig()
{
    PipelineConfig config = forwardConfig();
    // Blend draws keep a static cull mode (eNone) — the renderer only sets the
    // dynamic cull mode for the merged opaque pipeline, not for blend.
    config.dynamicCullMode = false;
    config.cullMode = vk::CullModeFlagBits::eNone;
    config.depthWrite = false;
    config.blendEnable = true;
    config.srcColourBlend = vk::BlendFactor::eSrcAlpha;
    config.dstColourBlend = vk::BlendFactor::eOneMinusSrcAlpha;
    config.srcAlphaBlend = vk::BlendFactor::eOne;
    config.dstAlphaBlend = vk::BlendFactor::eOneMinusSrcAlpha;
    return config;
}

PipelineConfig Pipeline::shadowConfig()
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
    // Set 0 is pushed inline per draw (VK_KHR_push_descriptor) — no per-object
    // descriptor-set allocation, mirroring the forward pass. The per-object
    // ShadowUBO/skin/morph buffers and the shared self-shadow image+sampler are
    // pushed by pushShadowObjectDescriptors. selfShadowFirst/Second inherit this
    // (they copy shadowConfig).
    config.pushDescriptorSet0 = true;
    // matrixIndex picks lightViewProj[] in the vertex stage. lightPosRange is
    // consumed by the fragment shader's point-shadow branch (linear distance
    // depth), so the push constant must be visible to both stages.
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eVertex |
                                               vk::ShaderStageFlagBits::eFragment,
                                           0, static_cast<uint32_t>(sizeof(ShadowPushConstants)));
    // Depth-only shadow pass: a single sampled D32 depth attachment, no colour.
    // (Historically a throwaway B8G8R8A8 colour attachment was attached to work
    // around a MoltenVK TBDR depth-store quirk; dynamic rendering on current
    // MoltenVK commits depth-only stores, so the colour target is dropped.)
    config.depthFormat = vk::Format::eD32Sfloat;
    config.depthWrite = true;
    config.depthCompare = vk::CompareOp::eLessOrEqual;
    config.cullMode = vk::CullModeFlagBits::eFront;
    config.writeColour = true;
    config.depthBiasEnable = true;
    config.depthBiasConstant = 0.0f;
    config.depthBiasSlope = 0.0f;
    return config;
}

PipelineConfig Pipeline::selfShadowSecondConfig()
{
    PipelineConfig config = shadowConfig();
    config.fragShaderPath = "self_shadow_second.frag.spv";
    // Cull front faces so only back faces rasterise. Back faces are always
    // behind the first-pass front-face depth by definition, so the per-fragment
    // `currentDepth <= firstDepth + ε` discard inside the shader stops being a
    // coin-flip on marginal fragments. The discard test remains as a safety
    // net but should never fire on properly-oriented back faces.
    config.cullMode = vk::CullModeFlagBits::eFront;
    return config;
}

PipelineConfig Pipeline::selfShadowFirstConfig()
{
    PipelineConfig config = shadowConfig();
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::postProcessConfig(vk::Format colourFormat)
{
    PipelineConfig config =
        fullscreenConfig("postprocess.vert.spv", "postprocess.frag.spv", colourFormat);
    config.bindings = {
        combinedImageSamplerBinding(bindingIndex(PostProcessBinding::HdrInput)),
        // 1: bloom mip 0 (additively composited over the tone-mapped HDR input).
        combinedImageSamplerBinding(bindingIndex(PostProcessBinding::BloomInput)),
    };
    addFragmentPushConstant(config, static_cast<uint32_t>(sizeof(PostProcessPushConstants)));
    return config;
}

PipelineConfig Pipeline::taaResolveConfig(vk::Format colourFormat)
{
    PipelineConfig config = fullscreenConfig("postprocess.vert.spv", "taa.frag.spv", colourFormat);
    config.bindings = {
        combinedImageSamplerBinding(bindingIndex(TaaBinding::CurrentColor)),
        combinedImageSamplerBinding(bindingIndex(TaaBinding::Velocity)),
        combinedImageSamplerBinding(bindingIndex(TaaBinding::History)),
    };
    addFragmentPushConstant(config, static_cast<uint32_t>(sizeof(TaaResolvePushConstants)));
    return config;
}

PipelineConfig Pipeline::bloomDownsampleConfig(vk::Format colourFormat)
{
    PipelineConfig config =
        fullscreenConfig("postprocess.vert.spv", "bloom_downsample.frag.spv", colourFormat);
    config.bindings = {
        combinedImageSamplerBinding(0),
    };
    addFragmentPushConstant(config, static_cast<uint32_t>(sizeof(BloomPushConstants)));
    return config;
}

PipelineConfig Pipeline::bloomUpsampleConfig(vk::Format colourFormat)
{
    PipelineConfig config = bloomDownsampleConfig(colourFormat);
    config.fragShaderPath = "bloom_upsample.frag.spv";
    // Additive blend so each upsample step adds its tent contribution onto
    // the existing mip content (which was loaded via render-pass eLoad).
    enableAdditiveBlend(config);
    return config;
}

PipelineConfig Pipeline::particleConfig(vk::Format colourFormat)
{
    PipelineConfig config =
        fullscreenConfig("particle.vert.spv", "particle.frag.spv", colourFormat);
    config.bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        // Scene depth, sampled in the fragment shader for soft-particle fade.
        combinedImageSamplerBinding(2),
    };
    addFragmentPushConstant(config, static_cast<uint32_t>(sizeof(ParticleSoftPushConstants)));
    // Additive: particles add HDR glow on top of the scene (like bloomUpsample).
    enableAdditiveBlend(config);
    return config;
}

PipelineConfig Pipeline::skyboxConfig()
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
    // Shares the forward HDR colour + RG16F velocity + D32 depth target.
    config.colourFormats = {vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16Sfloat};
    config.depthFormat = vk::Format::eD32Sfloat;
    config.useVertexInput = false;
    config.depthWrite = false;
    config.depthCompare = vk::CompareOp::eLessOrEqual;
    config.cullMode = vk::CullModeFlagBits::eNone;
    return config;
}

PipelineConfig Pipeline::environmentConvertConfig(vk::Format colourFormat)
{
    PipelineConfig config =
        fullscreenConfig("skybox.vert.spv", "environment_convert.frag.spv", colourFormat);
    config.bindings = {
        combinedImageSamplerBinding(0),
    };
    addFragmentPushConstant(config, static_cast<uint32_t>(sizeof(EnvironmentCaptureUBO)));
    return config;
}

PipelineConfig Pipeline::irradianceConvolutionConfig(vk::Format colourFormat)
{
    PipelineConfig config =
        fullscreenConfig("skybox.vert.spv", "irradiance_convolution.frag.spv", colourFormat);
    config.bindings = {
        combinedImageSamplerBinding(0),
    };
    addFragmentPushConstant(config, static_cast<uint32_t>(sizeof(EnvironmentCaptureUBO)));
    return config;
}

PipelineConfig Pipeline::prefilterEnvironmentConfig(vk::Format colourFormat)
{
    PipelineConfig config =
        fullscreenConfig("skybox.vert.spv", "prefilter_environment.frag.spv", colourFormat);
    config.bindings = {
        combinedImageSamplerBinding(0),
    };
    addFragmentPushConstant(config,
                            static_cast<uint32_t>(sizeof(EnvironmentPrefilterPushConstants)));
    return config;
}

PipelineConfig Pipeline::brdfIntegrationConfig(vk::Format colourFormat)
{
    return fullscreenConfig("postprocess.vert.spv", "brdf_integration.frag.spv", colourFormat);
}

void Pipeline::createDescriptorSetLayout(
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings, bool pushDescriptor)
{
    vk::DescriptorSetLayoutCreateInfo ci{
        // A push-descriptor set 0 is written inline with vkCmdPushDescriptorSetKHR
        // at draw time; it is never allocated from a pool.
        .flags = pushDescriptor ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                                : vk::DescriptorSetLayoutCreateFlags{},
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

    std::vector<vk::DynamicState> dynamicStates{
        vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eDepthBias};
    // Cull mode is core-1.3 dynamic state; opt in so the merged opaque pipeline
    // can switch between back-face and no culling per draw (config.cullMode is
    // then ignored — the renderer drives it).
    if (config.dynamicCullMode)
    {
        dynamicStates.push_back(vk::DynamicState::eCullMode);
    }
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
    // One blend state per colour attachment. Attachment 0 uses the config's
    // blend (HDR colour); any additional attachments (e.g. the TAA velocity
    // buffer) overwrite with no blend — motion vectors must not be alpha-mixed.
    const vk::ColorComponentFlags allChannels =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    std::vector<vk::PipelineColorBlendAttachmentState> blendAtts;
    blendAtts.reserve(config.colourFormats.size());
    for (std::size_t i = 0; i < config.colourFormats.size(); ++i)
    {
        if (i == 0)
        {
            blendAtts.push_back(vk::PipelineColorBlendAttachmentState{
                .blendEnable = config.blendEnable,
                .srcColorBlendFactor = config.srcColourBlend,
                .dstColorBlendFactor = config.dstColourBlend,
                .colorBlendOp = vk::BlendOp::eAdd,
                .srcAlphaBlendFactor = config.srcAlphaBlend,
                .dstAlphaBlendFactor = config.dstAlphaBlend,
                .alphaBlendOp = vk::BlendOp::eAdd,
                .colorWriteMask = writeMask,
            });
        }
        else
        {
            blendAtts.push_back(vk::PipelineColorBlendAttachmentState{
                .blendEnable = vk::False,
                .colorWriteMask = allChannels,
            });
        }
    }

    // Depth-only pipelines (no colour formats, e.g. shadow passes) carry no
    // blend attachment — the count must match colorAttachmentCount = 0.
    vk::PipelineColorBlendStateCreateInfo colourBlend{
        .logicOpEnable = false,
        .attachmentCount = static_cast<uint32_t>(blendAtts.size()),
        .pAttachments = blendAtts.empty() ? nullptr : blendAtts.data(),
    };

    createPipelineLayout(config);

    const vk::PipelineColorBlendStateCreateInfo* colourBlendPtr = &colourBlend;

    // Dynamic rendering: chain a VkPipelineRenderingCreateInfo carrying the
    // attachment formats the pipeline renders into (no VkRenderPass).
    vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount = static_cast<uint32_t>(config.colourFormats.size()),
        .pColorAttachmentFormats = config.colourFormats.data(),
        .depthAttachmentFormat = config.depthFormat,
    };

    vk::GraphicsPipelineCreateInfo pci{
        .pNext = &renderingInfo,
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
        .subpass = 0,
    };

    pipeline_ = vk::raii::Pipeline(*device_, *pipelineCache_, pci);
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
    // setLayouts[0] = per-object set 0, always present. setLayouts[1] = forward
    // globals set 1, only when the config declared globalBindings (forward
    // pipelines opt in; skybox / post-process / shadow / IBL precompute don't).
    // setLayouts[2] = bindless materials set 2, only when the config opted in
    // (forward pipelines). Sets must be contiguous: a forward pipeline always has
    // set 1 too, so the order is set0, set1, set2.
    std::array<vk::DescriptorSetLayout, 3> setLayouts{};
    uint32_t setCount = 1;
    setLayouts[0] = *descSetLayout_;
    if (static_cast<bool>(*globalDescSetLayout_))
    {
        setLayouts[setCount++] = *globalDescSetLayout_;
    }
    if (static_cast<bool>(*bindlessDescSetLayout_))
    {
        setLayouts[setCount++] = *bindlessDescSetLayout_;
    }
    vk::PipelineLayoutCreateInfo plci{
        .setLayoutCount = setCount,
        .pSetLayouts = setLayouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(config.pushConstantRanges.size()),
        .pPushConstantRanges = config.pushConstantRanges.data(),
    };
    pipelineLayout_ = vk::raii::PipelineLayout(*device_, plci);
}

} // namespace fire_engine
