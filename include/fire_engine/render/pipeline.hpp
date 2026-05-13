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
    // Set 0 — per-object / per-material bindings, allocated once per object
    // and rewritten only when materials change.
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    // Set 1 — forward-pipeline globals (light UBO, shadow maps, IBL, scene
    // colour). Allocated once per frame, bound once per forward pass. Empty
    // for pipelines that don't need a second descriptor set (skybox,
    // post-process, shadow, environment precompute).
    std::vector<vk::DescriptorSetLayoutBinding> globalBindings;
    std::vector<vk::PushConstantRange> pushConstantRanges;
    vk::RenderPass renderPass{};
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
    [[nodiscard]] vk::PipelineLayout pipelineLayout() const noexcept
    {
        return *pipelineLayout_;
    }
    [[nodiscard]] vk::Pipeline pipeline() const noexcept
    {
        return *pipeline_;
    }

    // Factory producing the PipelineConfig for the existing forward-lit
    // pipeline. The render pass handle comes from the caller (usually a
    // RenderPass::createForward() result).
    [[nodiscard]] static PipelineConfig forwardConfig(vk::RenderPass renderPass);

    // Variant of forwardConfig with cullMode=None for glTF materials flagged
    // doubleSided. Otherwise identical to forwardConfig (shaders, bindings,
    // depth state).
    [[nodiscard]] static PipelineConfig forwardDoubleSidedConfig(vk::RenderPass renderPass);

    // Variant of forwardConfig for glTF BLEND materials: cullMode=None,
    // depthWrite disabled, straight-alpha blending
    // (SRC_ALPHA / ONE_MINUS_SRC_ALPHA on colour; ONE / ONE_MINUS_SRC_ALPHA on
    // alpha).
    [[nodiscard]] static PipelineConfig forwardBlendConfig(vk::RenderPass renderPass);

    // Factory producing the PipelineConfig for the environment skybox
    // pipeline. Shares the forward render pass, uses no vertex buffers
    // (fullscreen triangle via gl_VertexIndex), and disables depth writes
    // with LEQUAL compare so it only writes where no forward geometry has
    // drawn.
    [[nodiscard]] static PipelineConfig skyboxConfig(vk::RenderPass renderPass);

    // Factory producing the PipelineConfig for equirectangular HDR ->
    // cubemap conversion. Draws a fullscreen triangle with no vertex input,
    // sampling a 2D panorama and writing one cubemap face at a time.
    [[nodiscard]] static PipelineConfig environmentConvertConfig(vk::RenderPass renderPass);

    // Factory producing the PipelineConfig for cubemap -> irradiance cubemap
    // convolution. Draws a fullscreen triangle with no vertex input,
    // sampling an environment cubemap and writing one low-resolution
    // irradiance face at a time.
    [[nodiscard]] static PipelineConfig irradianceConvolutionConfig(vk::RenderPass renderPass);

    // Factory producing the PipelineConfig for specular environment
    // prefiltering. Draws a fullscreen triangle with no vertex input,
    // sampling an environment cubemap and writing one cubemap face / mip
    // level at a time.
    [[nodiscard]] static PipelineConfig prefilterEnvironmentConfig(vk::RenderPass renderPass);

    // Factory producing the PipelineConfig for BRDF LUT integration. Draws a
    // fullscreen triangle into a 2D floating-point render target with no
    // descriptor bindings.
    [[nodiscard]] static PipelineConfig brdfIntegrationConfig(vk::RenderPass renderPass);

    // Factory producing the PipelineConfig for a depth-only shadow pipeline.
    // Writes only depth into an offscreen D32_SFLOAT attachment (no colour
    // attachment). Uses front-face culling and depth bias to eliminate
    // receiver shadow acne. Bindings 0..3 are ShadowUBO / SkinUBO / MorphUBO /
    // MorphTargets SSBO, all vertex stage.
    [[nodiscard]] static PipelineConfig shadowConfig(vk::RenderPass renderPass);

    // First pass for skinned self-shadow maps. Same shader/layout as the
    // regular shadow path but no face culling so the first visible
    // light-facing surface is captured for dual-depth rejection.
    [[nodiscard]] static PipelineConfig selfShadowFirstConfig(vk::RenderPass renderPass);

    // Second pass for skinned self-shadow maps. Uses the same vertex path and
    // descriptor layout as shadowConfig, but the fragment shader samples the
    // first-depth self map and discards same-surface fragments.
    [[nodiscard]] static PipelineConfig selfShadowSecondConfig(vk::RenderPass renderPass);

    // Factory producing the PipelineConfig for the post-process pass.
    // Draws a fullscreen triangle via gl_VertexIndex sampling the offscreen
    // HDR forward target at binding 0, applying ACES tone mapping + gamma,
    // and writing the result into the swapchain colour attachment. Depth
    // test disabled, no culling, no vertex input.
    [[nodiscard]] static PipelineConfig postProcessConfig(vk::RenderPass renderPass);

    // Bloom downsample: fullscreen triangle, samples one input mip, writes
    // the next coarser mip. Push constant carries inverse-input-resolution
    // and a first-pass flag (Karis-average to suppress firefly halos).
    [[nodiscard]] static PipelineConfig bloomDownsampleConfig(vk::RenderPass renderPass);

    // Bloom upsample: fullscreen triangle, samples a coarser mip with a tent
    // filter, additively blends into the next finer mip. Push constant carries
    // inverse-input-resolution.
    [[nodiscard]] static PipelineConfig bloomUpsampleConfig(vk::RenderPass renderPass);

private:
    void createDescriptorSetLayout(const std::vector<vk::DescriptorSetLayoutBinding>& bindings);
    void createGraphicsPipeline(const PipelineConfig& config);

    [[nodiscard]] std::array<vk::PipelineShaderStageCreateInfo, 2>
    createShaderStages(const PipelineConfig& config, vk::raii::ShaderModule& vertMod,
                       vk::raii::ShaderModule& fragMod) const;
    void createPipelineLayout(const PipelineConfig& config);

    const vk::raii::Device* device_{nullptr};
    vk::raii::DescriptorSetLayout descSetLayout_{nullptr};
    vk::raii::DescriptorSetLayout globalDescSetLayout_{nullptr};
    vk::raii::PipelineLayout pipelineLayout_{nullptr};
    vk::raii::Pipeline pipeline_{nullptr};
};

} // namespace fire_engine
