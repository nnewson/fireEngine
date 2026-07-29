#pragma once

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_capture.hpp>
#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/graphics/particle.hpp>
#include <fire_engine/graphics/shadow_lod_resolver.hpp>
#include <fire_engine/graphics/shadow_render_view.hpp>
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
#include <fire_engine/render/vdpm_gpu_manager.hpp>

namespace fire_engine
{

class RenderableScene;

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
    // Initial LOD mode (--lod-mode discrete|continuous|view-dependent). Default Discrete matches
    // RenderTunables; the overlay's LOD combo still switches at runtime. `view-dependent` activates
    // VDPM (and its indirect draws) at launch, so that path is exercisable without the overlay.
    LodMode lodMode{LodMode::Discrete};
    // GPU-driven VDPM backend request (rendering-spine #3). Tri-state: `true` (--vdpm-gpu), `false`
    // (--no-vdpm-gpu), or unset (nullopt) — resolved in the Renderer against device capability, so
    // unset defaults the backend ON wherever the device supports it (B5c-4 default flip), explicit
    // values always win, and repeated flags are last-one-wins. Only takes effect with lodMode
    // view-dependent; the overlay checkbox toggles it at runtime thereafter.
    std::optional<bool> vdpmGpuBackend{};
    // --no-lod: start with mesh LOD switched off, so every draw is full detail. Seeds
    // RenderTunables::lodEnabled, which both the forward and the shadow selection read — so this is
    // the "full detail" half of an acceptance A/B, not a forward-only override.
    bool lod{true};
    // --capture <path>: write the frame numbered `captureFrame` to `path` as a PNG and exit
    // successfully. Empty ⇒ no capture. The image is the final SWAPCHAIN content — post-process and
    // overlay included, exactly what the user sees — copied out immediately before present.
    //
    // Frame-numbered, never time-based: a wall-clock delay captures a different frame on a
    // different machine, which is the opposite of a reproducible reference image. It points at
    // argv, which outlives the parse.
    std::string_view capturePath{};
    // Which frame to capture (1-based). Enough frames must pass for the swapchain, IBL and any
    // temporal accumulation to settle; 16 is comfortably past that at any frame rate.
    int captureFrame{16};
    // --shadow-focus <group>:<slot>: focus one shadow view for the diagnostics panel and the
    // ShadowLod tint, so a capture can name its view instead of needing a click.
    //
    // TRI-STATE, deliberately: `nullopt` is "flag absent", an ENGAGED-but-empty value is "flag
    // given with no usable value" (which is terminal), and an engaged non-empty value is the
    // request. Two states would collapse `--shadow-focus` with a missing or flag-shaped value into
    // "no request", and the run would then capture the default view while looking exactly like a
    // correct capture of whatever was intended.
    //
    // A slot is the only handle available before the engine runs (identities are allocated at
    // load), so it is resolved ONCE at startup into the identity occupying that slot, and only the
    // identity is stored. An absent slot fails by name rather than falling back. It points at argv,
    // which outlives the parse.
    std::optional<std::string_view> shadowFocus{};
    // --require-validation: refuse to start unless the Vulkan validation layer is actually active.
    // For the render smoke and any automated run where "zero VUIDs" must mean "checked and clean"
    // rather than "nothing was checking". Off by default so a machine without the SDK still runs.
    bool requireValidation{false};
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

    // The active camera comes from the scene through the RenderableScene seam
    // (scene.activeCamera()), not as an argument — the scene owns all transforms.
    void drawFrame(Window& display, RenderableScene& scene, float dt);

    void waitIdle() const
    {
        device_.device().waitIdle();
    }

    [[nodiscard]] const Device& device() const noexcept
    {
        return device_;
    }

    // The GPU-driven VDPM registration seam (rendering-spine #3, Stage B5b), or null ONLY when the
    // device lacks the compute/scan capability (the manager is built whenever the device is
    // capable, independent of the runtime RenderTunables::vdpmGpuBackend selector, so the backend
    // can be toggled at runtime without a reload). The scene load path passes it to Geometry::load
    // / Object::load so meshes/fronts register with the backend; a null return keeps every instance
    // on the CPU front.
    [[nodiscard]] VdpmGpuRegistry* vdpmRegistry() noexcept
    {
        return vdpmManager_.get();
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

    // True when the ragdoll joint debug view is selected — the app gathers per-joint markers +
    // labels, and the forward pass suppresses the scene geometry, only then.
    [[nodiscard]] bool jointDebugWanted() const noexcept
    {
        return tunables_.debugView == DebugView::Joints;
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

    // True once a requested --capture has been written. The main loop ends on it, so a capture
    // command returns as soon as the file exists. A FAILED capture never gets here — it throws
    // out to the top-level handler, which logs and exits non-zero.
    [[nodiscard]] bool captureComplete() const noexcept
    {
        return captureDone_;
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
        // True when any collected draw this frame carries a skin. Only skinned
        // fragments sample the world-only CSM (shader.frag gates on hasSkin), so
        // when this is false the shadow pass skips the world-shadow iterations
        // entirely — nobody reads the map, and the frame that reintroduces a
        // skinned mesh re-renders it before any fragment samples it.
        bool anySkinned{false};
    };

    void updateLightData(Vec3 cameraPosition, Vec3 cameraTarget, float aspect,
                         std::span<const Lighting> lights);
    // RESETS shadowViews_ (so a view that stopped being active cannot linger from the previous
    // frame) and populates the cascade slots, then derives out.cascadeViewProj from the set. Fills
    // out.cascadeSplits. Reads directionalLightDir_ for the light basis, so the caller must set it
    // before calling. Runs before anything else populates the set.
    void computeShadowCascades(LightUBO& out, Vec3 cameraPosition, Vec3 cameraTarget, float aspect);
    // Registers the packed light as a spot caster if there is room (advancing
    // activeSpotCasters_) and populates that spot slot in shadowViews_. No-op if the
    // spot caster cap is hit.
    void assignSpotShadow(LightUBO& out, int packedSlot, const Lighting& light);
    // Registers the packed light as a point caster if there is room (advancing
    // activePointCasters_ and pointCasters_) and populates its six cube-face
    // slots in shadowViews_. No-op if the point caster cap is hit.
    void assignPointShadow(LightUBO& out, int packedSlot, const Lighting& light);
    // Fills out.iblParams / out.shadowParams / out.environmentParams from the
    // engine-wide constants plus the debug-flag members.
    void writeIblAndDebugParams(LightUBO& out) const;
    void assignSelfShadowSlots(std::span<DrawCommand> drawCommands);
    static void clearDrawBuckets(DrawBuckets& buckets) noexcept;
    void buildDrawBuckets(std::span<const DrawCommand> drawCommands, DrawBuckets& buckets) const;
    void recordDrawBucket(vk::CommandBuffer cmd, std::span<const DrawCommand> bucket,
                          PipelineHandle& lastBoundPipeline) const;

    // drawFrame() phases, in per-frame execution order. Each records into the
    // supplied command buffer (already in the recording state).
    void updateFrameLighting(RenderableScene& scene, Vec3 cameraPosition, Vec3 cameraTarget);
    [[nodiscard]] const DrawBuckets& collectDrawCommands(RenderableScene& scene,
                                                         Vec3 cameraPosition, Vec3 cameraTarget);
    void recordShadowPass(vk::CommandBuffer cmd, const DrawBuckets& buckets);
    // SH-03 slice 5: colour the ShadowLod debug view by the level the FOCUSED shadow view chose.
    //
    // Runs BETWEEN the shadow pass and the forward pass, which is the only window in which the
    // answer exists: the levels are decided per view during shadow recording, and the forward
    // draws' push constants are written afterwards. It patches this frame's forward buckets, so it
    // must be called after recordShadowPass and before recordForwardPass / recordDepthPrepass.
    //
    // A no-op unless the ShadowLod view is active — the tint is the only consumer, and walking the
    // buckets to compute a value nothing reads would be pure cost.
    void applyShadowLodTint(DrawBuckets& buckets) const;
    // Honours a pending --shadow-focus once, translating its slot into the identity that slot holds
    // in THIS frame's view set. Called after the set is fully populated; throws (naming the
    // request) if the slot is inactive, because continuing would capture a different view than was
    // asked for while looking perfectly correct.
    void resolveShadowFocusRequest();
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
    // `afterCapture` starts the transition from TransferSrcOptimal, where recordCaptureCopy
    // left the image, instead of ColorAttachmentOptimal.
    void transitionSwapchainToPresent(vk::CommandBuffer cmd, uint32_t imageIndex,
                                      bool afterCapture = false);
    // Copies the presented swapchain image into the host-visible capture buffer (--capture),
    // snapshotting the extent + format it was copied with.
    void recordCaptureCopy(vk::CommandBuffer cmd, uint32_t imageIndex);
    // Waits for that copy, converts, and writes the PNG. THROWS if the conversion or the write
    // fails, so a capture command cannot report success without a file.
    void writeCapture();
    // Swapchain format → capture channel order, rejecting anything unsupported.
    [[nodiscard]] static CaptureFormat resolveCaptureFormat(vk::Format format);
    [[nodiscard]] bool captureWanted() const noexcept
    {
        return !capturePath_.empty();
    }
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
    // GPU-driven VDPM front manager (rendering-spine #3, Stage B5b). Null ONLY when the device
    // lacks the compute/scan capability; otherwise built in the ctor regardless of the runtime
    // backend selector (so it can be toggled without a reload), and an unsupported device never
    // fails Renderer construction. Behind a unique_ptr because it is non-movable (owns device-bound
    // pipelines) and conditionally constructed.
    std::unique_ptr<VdpmGpuManager> vdpmManager_;
    // Per-frame VDPM work requests (Object appends camera+shadow-visible fronts during collection)
    // and the deduped camera-visible subset the compute is recorded for. Members so they keep their
    // capacity across frames.
    std::vector<VdpmWorkRequest> vdpmRequestScratch_;
    std::vector<VdpmWorkRequest> vdpmRecordScratch_;
    // Camera-visible GPU-VDPM front handles harvested from the forward buckets, handed to
    // selectVisibleVdpmRequests to filter + dedup the sink. Members (with vdpmSelectScratch_) so
    // the whole per-frame selection retains capacity and allocates nothing steady-state.
    std::vector<VdpmFrontHandle> vdpmVisibleScratch_;
    VdpmRequestSelectScratch vdpmSelectScratch_;
    // Per-front forward-draw counts (B5c-1 health emitted-triangle weighting), rebuilt each frame
    // from the visible forward buckets and handed to recordRequests. Member for steady-state
    // capacity.
    std::vector<VdpmFrontDrawCount> vdpmDrawCounts_;
    // Throttle for the periodic VDPM perf sample log (CPU record vs GPU compute ms) — the headless
    // baseline complement to the overlay's live readout.
    std::uint32_t vdpmPerfLogCounter_{0};
    // Frame capture (--capture). `framesRendered_` counts presented frames, so the capture is keyed
    // to a RENDER ORDINAL rather than elapsed time — every machine captures the same frame number.
    // That is not the same as the same picture: the main loop advances animation and physics from
    // wall-clock dt, so identical content additionally requires a static scene (which the SH-01
    // baseline is). `captureFormat_` is resolved at startup for an early, clear failure AND again
    // when the copy is recorded, since a swapchain recreation can change it in between.
    std::string capturePath_;
    std::uint64_t captureFrame_{0};
    std::uint64_t framesRendered_{0};
    // Extent and format the capture was actually COPIED with. Snapshotted when the copy is
    // recorded, because the swapchain can be recreated (resize / out-of-date present) between
    // that submit and the write — decoding against the new extent would garble the old pixels.
    vk::Extent2D captureExtent_{};
    CaptureFormat captureFormat_{CaptureFormat::Bgra8};
    MappedBufferSet captureBuffer_{};
    bool captureDone_{false};
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
    // Per-frame camera UBO (forward set 1, binding Camera) — view/proj/cameraPos/view-projections,
    // written once per frame here instead of duplicated into every object's set-0 UBO.
    Resources::MappedBufferSet cameraUbo_;
    // Forward pipeline globals (descriptor set 1) — one set per frame-in-flight,
    // bound once at the start of every forward pass.
    std::array<DescriptorSetHandle, kMaxFramesInFlight> globalDescSets_{};
    // SH-03: the single authority for every shadow view this frame — matrix, projection descriptor
    // and stable logical identity together. Every consumer (the ShadowUBO matrix array, the
    // LightUBO arrays, the coarse cull frustums, the shadow pass) is a PROJECTION of this set; no
    // producer writes a shadow matrix anywhere else. Populated in view order: reset + cascades and
    // punctual views in updateFrameLighting, self layers once the draws are known, world-only once
    // `anySkinned` is.
    ShadowRenderViewSet shadowViews_;
    // SH-03: per-frame LOD resolution cache + the cross-frame hysteresis history. Staged during
    // recording and committed only once the frame has been submitted, so a frame that was thrown
    // away leaves no dead band behind.
    ShadowLodResolver shadowLodResolver_;
    // The pending --shadow-focus request, held only until the first frame that can resolve it into
    // a logical identity (see resolveShadowFocusRequest). Cleared once honoured, so it can never
    // re-apply and re-anchor the focus to a slot after the user has clicked elsewhere.
    std::optional<ShadowViewSlotRequest> pendingShadowFocus_{};
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
    // SH-01 shadow diagnostics, FRAME-INDEXED. The overlay is built before this frame records its
    // shadow pass, while the GPU timings it sits beside come from a completed ring slot — so the
    // counters are collected into slot `currentFrame_` and published only when that slot's
    // timestamps resolve. Writing them straight into `stats_` would pair this frame's counts with
    // an older frame's times, which is most misleading in exactly the moving-light stability case
    // SH-01 exists to measure.
    std::array<ShadowFrameStats, kMaxFramesInFlight> shadowStatsRing_{};
    // Per-slot "collected AND submitted" bit, consumed on publication. Independent of the GPU
    // timestamp validity: a device without timestamp support still produces valid CPU counters.
    std::array<bool, kMaxFramesInFlight> shadowStatsSlotUsed_{};
    // Per-frame camera matrices (set at the top of drawFrame). view_ + jitteredProj_
    // drive rasterisation; currentViewProj_/previousViewProj_ are jitter-free for
    // TAA motion vectors. previousViewProj_ persists across frames.
    Mat4 view_{Mat4::identity()};
    Mat4 jitteredProj_{Mat4::identity()};
    Mat4 currentViewProj_{Mat4::identity()};
    Mat4 previousViewProj_{Mat4::identity()};
    uint32_t taaJitterIndex_{0};
    std::vector<DrawCommand> drawCommandScratch_;
    DrawBuckets drawBucketsScratch_;
    std::vector<Frustum> frustumScratch_;
    std::vector<Lighting> lightScratch_;
    std::vector<EmitterState> emitterScratch_;
    std::unordered_map<uint32_t, int> selfShadowSlotsScratch_;
    std::string environmentPath_;
    // Draw the environment cubemap as a background. False when no skybox was requested (empty
    // environmentPath_): the scene is still lit by the default environment's IBL, but nothing is
    // drawn behind the geometry — the scene target's neutral clear shows through.
    bool drawSkybox_{true};
};

} // namespace fire_engine
