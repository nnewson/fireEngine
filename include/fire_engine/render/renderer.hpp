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
#include <fire_engine/render/debug_draw.hpp>
#include <fire_engine/render/debug_overlay.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/frame.hpp>
#include <fire_engine/render/gpu_profiler.hpp>
#include <fire_engine/render/particle_system.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/post_processing.hpp>
#include <fire_engine/render/render_tunables.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/shadows.hpp>
#include <fire_engine/render/soft_body_system.hpp>
#include <fire_engine/render/ssao.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/taa.hpp>
#include <fire_engine/render/transmission.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

class SceneGraph;

// DebugView lives in render_tunables.hpp (included below) so the overlay can
// reference it without pulling in the renderer.

struct RendererDebug
{
    DebugView view{DebugView::None};
    // Disables every shadow-map visibility lookup (cascade, spot, point) in
    // the forward shader. Surfaces look fully lit by direct lighting.
    bool noShadows{false};
    // Temporal anti-aliasing. When false (--no-taa) the projection jitter and
    // the resolve pass are both skipped, reverting to the raw aliased image —
    // the A/B reference for confirming TAA is doing the work.
    bool taa{true};
    // Start with the ImGui debug overlay visible (--overlay). Off by default so
    // normal runs and screenshots are unaffected; toggled at runtime with F1.
    bool overlayVisible{false};
    // Start with physics debug wireframes on (--debug-physics): broadphase AABBs,
    // collider shapes, and contacts. Off by default; toggled per-category in the
    // overlay's "Physics debug" panel.
    bool physicsDebug{false};
};

class Renderer
{
public:
    explicit Renderer(const Window& window, std::string environmentPath = {},
                      RendererDebug debug = {});
    ~Renderer() = default;

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    // Non-movable: owns the ImGui overlay (global state) and subsystems that hold
    // back-references. The Renderer lives behind a unique_ptr and is never moved.
    Renderer(Renderer&&) noexcept = delete;
    Renderer& operator=(Renderer&&) noexcept = delete;

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

    // Register a cloth with the GPU soft-body solver (see SoftBodySystem). The
    // mesh supplies the particle/constraint data; vertexBuffer is the cloth
    // Geometry's storage vertex buffer that the solver writes each frame.
    void addCloth(const ClothMesh& mesh, BufferHandle vertexBuffer)
    {
        softBody_.addCloth(mesh, vertexBuffer);
    }

    // World-space colliders the cloth solver projects out of, refreshed each frame
    // by the app (PhysicsWorld gather + ground plane) before drawFrame.
    void setClothColliders(std::span<const ClothCollider> colliders)
    {
        clothColliders_.assign(colliders.begin(), colliders.end());
    }

    // Physics debug-draw data, pushed by the app each frame (only when wanted).
    void setPhysicsDebug(PhysicsDebugData data)
    {
        physicsDebug_ = std::move(data);
    }

    // True when any physics debug-draw category is enabled — the app uses this to
    // skip gathering debug data when nothing will be drawn.
    [[nodiscard]] bool physicsDebugWanted() const noexcept
    {
        return tunables_.debugDrawAabbs || tunables_.debugDrawColliders ||
               tunables_.debugDrawContacts;
    }

    [[nodiscard]] Resources& resources() noexcept
    {
        return resources_;
    }

    [[nodiscard]] const Resources& resources() const noexcept
    {
        return resources_;
    }

    // Debug-overlay control surface for the main loop: toggle visibility (F1) and
    // query whether the overlay is currently capturing input (so the camera
    // doesn't move while the user drives a widget).
    void toggleOverlay() noexcept
    {
        overlay_.toggle();
    }

    [[nodiscard]] bool overlayWantsMouse() const noexcept
    {
        return overlay_.wantsMouse();
    }

    [[nodiscard]] bool overlayWantsKeyboard() const noexcept
    {
        return overlay_.wantsKeyboard();
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
    [[nodiscard]] DrawBuckets buildDrawBuckets(std::span<const DrawCommand> drawCommands) const;
    void recordDrawBucket(vk::CommandBuffer cmd, std::span<const DrawCommand> bucket,
                          PipelineHandle& lastBoundPipeline) const;

    // drawFrame() phases, in per-frame execution order. Each records into the
    // supplied command buffer (already in the recording state).
    void updateFrameLighting(SceneGraph& scene, Vec3 cameraPosition, Vec3 cameraTarget);
    [[nodiscard]] DrawBuckets collectDrawCommands(vk::CommandBuffer cmd, SceneGraph& scene,
                                                  Vec3 cameraPosition, Vec3 cameraTarget);
    void recordShadowPass(vk::CommandBuffer cmd, const DrawBuckets& buckets);
    // Depth-only prepass over the opaque bucket, before the forward pass, so the
    // shared depth buffer is filled for SSAO and the forward pass gets early-Z.
    void recordDepthPrepass(vk::CommandBuffer cmd, const DrawBuckets& buckets);
    // SSAO + contact-shadow pass (after the prepass): writes the AO target the
    // forward shader samples to modulate ambient / sun visibility.
    void recordSsaoPass(vk::CommandBuffer cmd);
    // Physics debug wireframes into the HDR target, after particles / before post.
    void recordDebugDrawPass(vk::CommandBuffer cmd);
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
    // Blocks the CPU until the frame-pacing timeline semaphore reaches `value`
    // (no-op for value 0 — nothing has been submitted for that slot/image yet).
    void waitTimeline(uint64_t value) const;
    void beginForwardRendering(vk::CommandBuffer cmd);
    void endForwardRendering(vk::CommandBuffer cmd);
    // Final ColorAttachmentOptimal → PresentSrcKHR transition, recorded after the
    // overlay (post-process leaves the swap image in ColorAttachmentOptimal).
    void transitionSwapchainToPresent(vk::CommandBuffer cmd, uint32_t imageIndex);
    void submitAndPresent(Window& display, vk::CommandBuffer cmd, uint32_t imageIndex);
    void recordSkybox(Vec3 cameraPosition, Vec3 cameraTarget,
                      std::vector<DrawCommand>& drawCommands);

    Device device_;
    Swapchain swapchain_;
    // Forward pipeline for opaque + double-sided materials (cull mode set per
    // draw via dynamic state). BLEND materials use pipelineBlend_.
    Pipeline pipelineOpaque_;
    Pipeline pipelineBlend_;
    Pipeline skyboxPipeline_;
    Pipeline depthPrepassPipeline_;
    Frame frame_;
    Resources resources_;
    PostProcessing postProcessing_;
    Transmission transmission_;
    Shadows shadows_;
    ParticleSystem particles_;
    Taa taa_;
    Ssao ssao_;
    DebugDraw debugDraw_;
    SoftBodySystem softBody_;
    GpuProfiler profiler_;
    DebugOverlay overlay_;
    FrameStats stats_{};
    RenderTunables tunables_{};
    std::vector<ClothCollider> clothColliders_;
    PhysicsDebugData physicsDebug_;
    PipelineHandle forwardOpaqueHandle_{NullPipeline};
    PipelineHandle forwardBlendHandle_{NullPipeline};
    PipelineHandle skyboxPipelineHandle_{NullPipeline};
    PipelineHandle depthPrepassHandle_{NullPipeline};
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
    // Timeline-semaphore frame pacing. timelineValue_ is the last value signalled;
    // frameTimelineValue_[slot] is the value the last submit using that
    // frame-in-flight slot signalled (gate cmd-buffer / per-frame-UBO reuse);
    // imageTimelineValue_[image] is the value the last submit that rendered to a
    // swapchain image signalled (gate image reuse when image count != frames in
    // flight). Replaces the inFlight fences + imagesInFlight fence tracking.
    uint64_t timelineValue_{0};
    std::array<uint64_t, kMaxFramesInFlight> frameTimelineValue_{};
    std::vector<uint64_t> imageTimelineValue_{};
    uint32_t currentFrame_{0};
    // Per-frame camera matrices (set at the top of drawFrame). view_ + jitteredProj_
    // drive rasterisation; currentViewProj_/previousViewProj_ are jitter-free for
    // TAA motion vectors. previousViewProj_ persists across frames.
    Mat4 view_{Mat4::identity()};
    Mat4 jitteredProj_{Mat4::identity()};
    Mat4 currentViewProj_{Mat4::identity()};
    Mat4 previousViewProj_{Mat4::identity()};
    uint32_t taaJitterIndex_{0};
    std::string environmentPath_;
};

} // namespace fire_engine
