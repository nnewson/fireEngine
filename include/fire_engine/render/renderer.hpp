#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/frame.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/post_processing.hpp>
#include <fire_engine/render/render_pass.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/shadows.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/transmission.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

class SceneGraph;

class Renderer
{
public:
    explicit Renderer(const Window& window, std::string environmentPath = {});
    ~Renderer() = default;

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) noexcept = default;
    Renderer& operator=(Renderer&&) noexcept = default;

    void drawFrame(Window& display, SceneGraph& scene, Vec3 cameraPosition, Vec3 cameraTarget);

    void waitIdle() const
    {
        device_.device().waitIdle();
    }

    [[nodiscard]] const Device& device() const noexcept
    {
        return device_;
    }

    [[nodiscard]] const Swapchain& swapchain() const noexcept
    {
        return swapchain_;
    }

    [[nodiscard]] Swapchain& swapchain() noexcept
    {
        return swapchain_;
    }

    [[nodiscard]] const Pipeline& pipeline() const noexcept
    {
        return pipelineOpaque_;
    }

    [[nodiscard]] const Frame& frame() const noexcept
    {
        return frame_;
    }

    [[nodiscard]] Frame& frame() noexcept
    {
        return frame_;
    }

    [[nodiscard]] Resources& resources() noexcept
    {
        return resources_;
    }

    [[nodiscard]] const Resources& resources() const noexcept
    {
        return resources_;
    }

private:
    struct DrawBuckets
    {
        std::vector<DrawCommand> shadow;
        std::vector<DrawCommand> opaque;
        std::vector<DrawCommand> blend;
        // KHR_materials_transmission F3 — draws deferred to the second forward
        // sub-pass so they can sample the captured sceneColor target.
        std::vector<DrawCommand> transmissive;
    };

    void updateLightData(Vec3 cameraPosition, Vec3 cameraTarget, float aspect,
                         std::span<const Lighting> lights);
    [[nodiscard]] DrawBuckets buildDrawBuckets(const std::vector<DrawCommand>& drawCommands) const;
    void recordDrawBucket(vk::CommandBuffer cmd, const std::vector<DrawCommand>& bucket,
                          PipelineHandle& lastBoundPipeline) const;
    void recordForwardPass(vk::CommandBuffer cmd, const DrawBuckets& buckets);
    void recreateSwapchain(const Window& display);
    [[nodiscard]] std::optional<uint32_t> acquireNextImage(Window& display);
    void beginRenderPass(vk::CommandBuffer cmd);
    void submitAndPresent(Window& display, vk::CommandBuffer cmd, uint32_t imageIndex);
    void recordSkybox(Vec3 cameraPosition, Vec3 cameraTarget,
                      std::vector<DrawCommand>& drawCommands);

    Device device_;
    Swapchain swapchain_;
    RenderPass forwardPass_;
    Pipeline pipelineOpaque_;
    Pipeline pipelineOpaqueDoubleSided_;
    Pipeline pipelineBlend_;
    Pipeline skyboxPipeline_;
    Frame frame_;
    Resources resources_;
    PostProcessing postProcessing_;
    Transmission transmission_;
    Shadows shadows_;
    PipelineHandle forwardOpaqueHandle_{NullPipeline};
    PipelineHandle forwardOpaqueDoubleSidedHandle_{NullPipeline};
    PipelineHandle forwardBlendHandle_{NullPipeline};
    PipelineHandle skyboxPipelineHandle_{NullPipeline};
    TextureHandle skyboxCubemapHandle_{NullTexture};
    TextureHandle irradianceCubemapHandle_{NullTexture};
    TextureHandle prefilteredCubemapHandle_{NullTexture};
    TextureHandle brdfLutHandle_{NullTexture};
    Resources::MappedBufferSet skyboxUbo_;
    std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT> skyboxDescSets_{};
    BufferHandle skyboxIndexBuffer_{NullBuffer};
    Resources::MappedBufferSet lightUbo_;
    std::array<Mat4, SHADOW_TOTAL_MATRIX_COUNT> shadowViewProjs_{};
    int activeSpotCasters_{0};
    int activePointCasters_{0};
    std::array<PointShadowCaster, MAX_POINT_SHADOW_CASTERS> pointCasters_{};
    std::vector<vk::Fence> imagesInFlight_{};
    uint32_t currentFrame_{0};
    // Flip to true and rebuild to visualise cascade regions: red→green→blue→yellow
    // for cascades 0..3, near to far. Surfaced via LightUBO.environmentParams.w.
    bool cascadeDebugTint_{false};
    std::string environmentPath_;
};

} // namespace fire_engine
