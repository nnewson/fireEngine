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
#include <fire_engine/render/particle_system.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/post_processing.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/shadows.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/transmission.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

class SceneGraph;

// Selects which debug-view branch the forward fragment shader runs. Mapped
// 1:1 to LightUBO::environmentParams[2] (0..4) — keep the enum and shader
// constants in lockstep.
enum class DebugView : int
{
    None = 0,
    Normals = 1,
    NdotL = 2,
    Shadow = 3,
    ShadowDepth = 4,
};

struct RendererDebug
{
    DebugView view{DebugView::None};
    // Disables every shadow-map visibility lookup (cascade, spot, point) in
    // the forward shader. Surfaces look fully lit by direct lighting.
    bool noShadows{false};
};

class Renderer
{
public:
    explicit Renderer(const Window& window, std::string environmentPath = {},
                      RendererDebug debug = {});
    ~Renderer() = default;

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) noexcept = default;
    Renderer& operator=(Renderer&&) noexcept = default;

    void drawFrame(Window& display, SceneGraph& scene, Vec3 cameraPosition, Vec3 cameraTarget,
                   float dt);

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
        std::vector<DrawCommand> worldShadow;
        std::vector<DrawCommand> selfShadow;
        std::vector<DrawCommand> opaque;
        std::vector<DrawCommand> blend;
        // KHR_materials_transmission F3 — draws deferred to the second forward
        // sub-pass so they can sample the captured sceneColor target.
        std::vector<DrawCommand> transmissive;
    };

    void updateLightData(Vec3 cameraPosition, Vec3 cameraTarget, float aspect,
                         std::span<const Lighting> lights);
    // Resets shadowViewProjs_ and writes the per-cascade matrix slots. Fills
    // out.cascadeViewProj / out.cascadeSplits. Reads directionalLightDir_ for
    // the light basis, so the caller must set it before calling.
    void computeShadowCascades(LightUBO& out, Vec3 cameraPosition, Vec3 cameraTarget, float aspect);
    // Registers the packed light as a spot caster if there is room (advancing
    // activeSpotCasters_) and writes the matching matrices into out and
    // shadowViewProjs_. No-op if the spot caster cap is hit.
    void assignSpotShadow(LightUBO& out, int packedSlot, const Lighting& light);
    // Registers the packed light as a point caster if there is room (advancing
    // activePointCasters_ and pointCasters_) and writes its six cube-face
    // matrices into shadowViewProjs_. No-op if the point caster cap is hit.
    void assignPointShadow(LightUBO& out, int packedSlot, const Lighting& light);
    // Fills out.iblParams / out.shadowParams / out.environmentParams from the
    // engine-wide constants plus the debug-flag members.
    void writeIblAndDebugParams(LightUBO& out) const;
    void assignSelfShadowSlots(std::vector<DrawCommand>& drawCommands);
    [[nodiscard]] DrawBuckets buildDrawBuckets(const std::vector<DrawCommand>& drawCommands) const;
    void recordDrawBucket(vk::CommandBuffer cmd, const std::vector<DrawCommand>& bucket,
                          PipelineHandle& lastBoundPipeline) const;

    // drawFrame() phases, in per-frame execution order. Each records into the
    // supplied command buffer (already in the recording state).
    void updateFrameLighting(SceneGraph& scene, Vec3 cameraPosition, Vec3 cameraTarget);
    [[nodiscard]] DrawBuckets collectDrawCommands(vk::CommandBuffer cmd, SceneGraph& scene,
                                                  Vec3 cameraPosition, Vec3 cameraTarget);
    void recordShadowPass(vk::CommandBuffer cmd, const DrawBuckets& buckets);
    void recordForwardPass(vk::CommandBuffer cmd, const DrawBuckets& buckets);
    void recordTransmissionPass(vk::CommandBuffer cmd, const DrawBuckets& buckets);
    void recordParticlePass(vk::CommandBuffer cmd);
    void recordPostProcessing(vk::CommandBuffer cmd, uint32_t imageIndex);

    void recreateSwapchain(const Window& display);
    // Snapshots the current state of sharedTextures + lightUbo_ buffers into a
    // GlobalDescriptorRequest. Used at ctor time to populate the set 1
    // descriptors and on swapchain resize to rebind them after any
    // releaseTexture/createTexture cycle has invalidated the old samplers.
    [[nodiscard]] GlobalDescriptorRequest buildGlobalDescriptorRequest() const;
    [[nodiscard]] std::optional<uint32_t> acquireNextImage(Window& display);
    void beginForwardRendering(vk::CommandBuffer cmd);
    void endForwardRendering(vk::CommandBuffer cmd);
    void submitAndPresent(Window& display, vk::CommandBuffer cmd, uint32_t imageIndex);
    void recordSkybox(Vec3 cameraPosition, Vec3 cameraTarget,
                      std::vector<DrawCommand>& drawCommands);

    Device device_;
    Swapchain swapchain_;
    Pipeline pipelineOpaque_;
    Pipeline pipelineOpaqueDoubleSided_;
    Pipeline pipelineBlend_;
    Pipeline skyboxPipeline_;
    Frame frame_;
    Resources resources_;
    PostProcessing postProcessing_;
    Transmission transmission_;
    Shadows shadows_;
    ParticleSystem particles_;
    PipelineHandle forwardOpaqueHandle_{NullPipeline};
    PipelineHandle forwardOpaqueDoubleSidedHandle_{NullPipeline};
    PipelineHandle forwardBlendHandle_{NullPipeline};
    PipelineHandle skyboxPipelineHandle_{NullPipeline};
    TextureHandle skyboxCubemapHandle_{NullTexture};
    TextureHandle irradianceCubemapHandle_{NullTexture};
    TextureHandle prefilteredCubemapHandle_{NullTexture};
    TextureHandle brdfLutHandle_{NullTexture};
    Resources::MappedBufferSet skyboxUbo_;
    std::array<DescriptorSetHandle, kMaxFramesInFlight> skyboxDescSets_{};
    BufferHandle skyboxIndexBuffer_{NullBuffer};
    Resources::MappedBufferSet lightUbo_;
    // Forward pipeline globals (descriptor set 1) — one set per frame-in-flight,
    // bound once at the start of every forward pass.
    std::array<DescriptorSetHandle, kMaxFramesInFlight> globalDescSets_{};
    std::array<Mat4, kShadowTotalMatrixCount> shadowViewProjs_{};
    LightUBO lightData_{};
    Vec3 directionalLightDir_{1.0f, -1.0f, 1.0f};
    int activeSpotCasters_{0};
    int activePointCasters_{0};
    std::array<PointShadowCaster, kMaxPointShadowCasters> pointCasters_{};
    std::vector<vk::Fence> imagesInFlight_{};
    uint32_t currentFrame_{0};
    std::string environmentPath_;
    RendererDebug debug_{};
};

} // namespace fire_engine
