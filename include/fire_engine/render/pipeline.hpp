#pragma once

#include <array>
#include <string>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/render/device.hpp>

namespace fire_engine
{

struct PipelineConfig
{
    std::string vertShaderPath;
    std::string fragShaderPath;
    // Set 0 — per-object vertex-stage bindings (frame / skin / morph UBOs + morph
    // SSBO), allocated per object. Material data is bindless (set 2).
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    // When true, set 0 is created as a push-descriptor layout
    // (VK_KHR_push_descriptor): its buffers are pushed inline at draw time
    // rather than allocated as a per-object descriptor set. Forward pipelines
    // only — such a layout cannot be used to allocate descriptor sets.
    bool pushDescriptorSet0{false};
    // Set 1 — forward-pipeline globals (light UBO, shadow maps, IBL, scene
    // colour). Allocated once per frame, bound once per forward pass. Empty
    // for pipelines that don't need a second descriptor set (skybox,
    // post-process, shadow, environment precompute).
    std::vector<vk::DescriptorSetLayoutBinding> globalBindings;
    // Set 2 — bindless materials: one global combined-image-sampler array
    // (textures) + the global materials SSBO, shared by every forward draw and
    // indexed in-shader. Opt-in (forward pipelines only); created with
    // partially-bound + update-after-bind binding flags.
    bool bindlessSet{false};
    std::vector<vk::PushConstantRange> pushConstantRanges;
    // Dynamic-rendering attachment formats: the pipeline is created with a
    // VkPipelineRenderingCreateInfo built from these (no VkRenderPass).
    std::vector<vk::Format> colourFormats;
    vk::Format depthFormat{vk::Format::eUndefined};
    bool useVertexInput{true};
    bool depthTestEnable{true};
    bool depthWrite{true};
    vk::CompareOp depthCompare{vk::CompareOp::eLess};
    vk::CullModeFlags cullMode{vk::CullModeFlagBits::eBack};
    bool blendEnable{false};
    vk::BlendFactor srcColourBlend{vk::BlendFactor::eOne};
    vk::BlendFactor dstColourBlend{vk::BlendFactor::eZero};
    vk::BlendFactor srcAlphaBlend{vk::BlendFactor::eOne};
    vk::BlendFactor dstAlphaBlend{vk::BlendFactor::eZero};
    bool writeColour{true};
    // When true the pipeline declares VK_DYNAMIC_STATE_CULL_MODE (core 1.3) and
    // the static cullMode above is ignored — the renderer sets cull mode per
    // draw. Lets opaque + double-sided geometry share one forward pipeline.
    bool dynamicCullMode{false};
    bool depthBiasEnable{false};
    float depthBiasConstant{0.0f};
    float depthBiasSlope{0.0f};
};

class Pipeline
{
public:
    Pipeline(const Device& device, const PipelineConfig& config);
    ~Pipeline() = default;

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept = default;
    Pipeline& operator=(Pipeline&&) noexcept = default;

    [[nodiscard]] vk::DescriptorSetLayout descriptorSetLayout() const noexcept
    {
        return *descSetLayout_;
    }
    // Set 1 layout (forward globals). nullptr when the pipeline declared no
    // globalBindings in its PipelineConfig.
    [[nodiscard]] vk::DescriptorSetLayout globalDescriptorSetLayout() const noexcept
    {
        return *globalDescSetLayout_;
    }
    [[nodiscard]] bool hasGlobalDescriptorSetLayout() const noexcept
    {
        return static_cast<bool>(*globalDescSetLayout_);
    }
    // Set 2 layout (bindless materials). nullptr unless the config opted in via
    // bindlessSet. Resources allocates + writes the actual set from this layout.
    [[nodiscard]] vk::DescriptorSetLayout bindlessDescriptorSetLayout() const noexcept
    {
        return *bindlessDescSetLayout_;
    }
    [[nodiscard]] bool hasBindlessDescriptorSetLayout() const noexcept
    {
        return static_cast<bool>(*bindlessDescSetLayout_);
    }
    [[nodiscard]] vk::PipelineLayout pipelineLayout() const noexcept
    {
        return *pipelineLayout_;
    }
    [[nodiscard]] vk::Pipeline pipeline() const noexcept
    {
        return *pipeline_;
    }

    // Factory producing the PipelineConfig for the forward-lit pipeline,
    // targeting the HDR offscreen colour + shared D32 depth via dynamic
    // rendering. Serves both opaque and double-sided glTF materials — cull mode
    // is a dynamic state (dynamicCullMode), set per draw by the renderer.
    [[nodiscard]] static PipelineConfig forwardConfig();

    // Variant of forwardConfig for glTF BLEND materials: cullMode=None,
    // depthWrite disabled, straight-alpha blending
    // (SRC_ALPHA / ONE_MINUS_SRC_ALPHA on colour; ONE / ONE_MINUS_SRC_ALPHA on
    // alpha).
    [[nodiscard]] static PipelineConfig forwardBlendConfig();

    // Factory producing the PipelineConfig for the environment skybox
    // pipeline. Shares the forward render pass, uses no vertex buffers
    // (fullscreen triangle via gl_VertexIndex), and disables depth writes
    // with LEQUAL compare so it only writes where no forward geometry has
    // drawn.
    [[nodiscard]] static PipelineConfig skyboxConfig();

    // Factory producing the PipelineConfig for equirectangular HDR ->
    // cubemap conversion. Draws a fullscreen triangle with no vertex input,
    // sampling a 2D panorama and writing one cubemap face at a time.
    [[nodiscard]] static PipelineConfig environmentConvertConfig(vk::Format colourFormat);

    // Factory producing the PipelineConfig for cubemap -> irradiance cubemap
    // convolution. Draws a fullscreen triangle with no vertex input,
    // sampling an environment cubemap and writing one low-resolution
    // irradiance face at a time.
    [[nodiscard]] static PipelineConfig irradianceConvolutionConfig(vk::Format colourFormat);

    // Factory producing the PipelineConfig for specular environment
    // prefiltering. Draws a fullscreen triangle with no vertex input,
    // sampling an environment cubemap and writing one cubemap face / mip
    // level at a time.
    [[nodiscard]] static PipelineConfig prefilterEnvironmentConfig(vk::Format colourFormat);

    // Factory producing the PipelineConfig for BRDF LUT integration. Draws a
    // fullscreen triangle into a 2D floating-point render target with no
    // descriptor bindings.
    [[nodiscard]] static PipelineConfig brdfIntegrationConfig(vk::Format colourFormat);

    // Factory producing the PipelineConfig for the camera depth prepass: reuses
    // the forward vertex shader (shader.vert, skin/morph aware) with an empty
    // fragment shader, writing only depth into the shared D32 buffer before the
    // forward pass. Same push-descriptor set 0 as the forward pipeline so the
    // shared pushForwardObjectDescriptors works; cull mode dynamic (per draw).
    // The forward pass then loads this depth with a LESS_OR_EQUAL test.
    [[nodiscard]] static PipelineConfig depthPrepassConfig();

    // Factory producing the PipelineConfig for a depth-only shadow pipeline.
    // Writes only depth into an offscreen D32_SFLOAT attachment (no colour
    // attachment). Uses front-face culling and depth bias to eliminate
    // receiver shadow acne. Bindings 0..3 are ShadowUBO / SkinUBO / MorphUBO /
    // MorphTargets SSBO, all vertex stage.
    [[nodiscard]] static PipelineConfig shadowConfig();

    // First pass for skinned self-shadow maps. Same shader/layout as the
    // regular shadow path but no face culling so the first visible
    // light-facing surface is captured for dual-depth rejection.
    [[nodiscard]] static PipelineConfig selfShadowFirstConfig();

    // Second pass for skinned self-shadow maps. Uses the same vertex path and
    // descriptor layout as shadowConfig, but the fragment shader samples the
    // first-depth self map and discards same-surface fragments.
    [[nodiscard]] static PipelineConfig selfShadowSecondConfig();

    // Factory producing the PipelineConfig for the post-process pass.
    // Draws a fullscreen triangle via gl_VertexIndex sampling the offscreen
    // HDR forward target at binding 0, applying ACES tone mapping + gamma,
    // and writing the result into the swapchain colour attachment. Depth
    // test disabled, no culling, no vertex input.
    [[nodiscard]] static PipelineConfig postProcessConfig(vk::Format colourFormat);

    // Factory producing the PipelineConfig for the TAA resolve pass. Fullscreen
    // triangle (postprocess.vert), sampling current scene colour + velocity +
    // previous-frame history, writing the resolved colour into a history target.
    // No vertex input, no depth.
    [[nodiscard]] static PipelineConfig taaResolveConfig(vk::Format colourFormat);

    // Factory producing the PipelineConfig for the SSAO + contact-shadow pass.
    // Fullscreen triangle (postprocess.vert) sampling scene depth (binding 0) +
    // an SsaoUBO (binding 1), writing the R8 AO target. No vertex input, no depth.
    [[nodiscard]] static PipelineConfig ssaoConfig(vk::Format colourFormat);

    // Factory for the bilateral AO blur pass. Fullscreen triangle sampling the
    // raw AO target (binding 0) + scene depth (binding 1), writing the smoothed
    // R8G8 AO. Push constant carries texel size + the proj terms for depth
    // linearisation. No vertex input, no depth.
    [[nodiscard]] static PipelineConfig ssaoBlurConfig(vk::Format colourFormat);

    // Bloom downsample: fullscreen triangle, samples one input mip, writes
    // the next coarser mip. Push constant carries inverse-input-resolution
    // and a first-pass flag (Karis-average to suppress firefly halos).
    [[nodiscard]] static PipelineConfig bloomDownsampleConfig(vk::Format colourFormat);

    // Bloom upsample: fullscreen triangle, samples a coarser mip with a tent
    // filter, additively blends into the next finer mip. Push constant carries
    // inverse-input-resolution.
    [[nodiscard]] static PipelineConfig bloomUpsampleConfig(vk::Format colourFormat);

    // Factory for the GPU particle billboard pipeline. No vertex input
    // (gl_VertexIndex builds the quad, gl_InstanceIndex reads the pool SSBO at
    // binding 0; frame UBO at binding 1). Additive blend into the HDR target,
    // depth test/write off (occlusion + soft fade are done in-shader against
    // sampled scene depth).
    [[nodiscard]] static PipelineConfig particleConfig(vk::Format colourFormat);

private:
    void createDescriptorSetLayout(const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                                   bool pushDescriptor);
    void createGraphicsPipeline(const PipelineConfig& config);

    [[nodiscard]] std::array<vk::PipelineShaderStageCreateInfo, 2>
    createShaderStages(const PipelineConfig& config, vk::raii::ShaderModule& vertMod,
                       vk::raii::ShaderModule& fragMod) const;
    void createPipelineLayout(const PipelineConfig& config);

    void createBindlessDescriptorSetLayout();

    const vk::raii::Device* device_{nullptr};
    const vk::raii::PipelineCache* pipelineCache_{nullptr};
    vk::raii::DescriptorSetLayout descSetLayout_{nullptr};
    vk::raii::DescriptorSetLayout globalDescSetLayout_{nullptr};
    vk::raii::DescriptorSetLayout bindlessDescSetLayout_{nullptr};
    vk::raii::PipelineLayout pipelineLayout_{nullptr};
    vk::raii::Pipeline pipeline_{nullptr};
};

} // namespace fire_engine
