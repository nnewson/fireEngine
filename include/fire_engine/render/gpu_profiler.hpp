#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/gpu_limits.hpp>

namespace fire_engine
{

class Device;

// Per-pass GPU timing via a timestamp query pool. Each pass writes a begin/end
// timestamp pair; results are read back one cycle later (the acquire fence
// guarantees the slot is complete), so there's no stall. Degrades gracefully to
// disabled when the device/queue doesn't support timestamps (e.g. some MoltenVK
// configurations) — callers fall back to CPU frame timing.
enum class ProfilePass : uint32_t
{
    VdpmCompute, // GPU-driven VDPM front lifecycle (recorded before the shadow pass)
    Shadow,
    DepthPrepass,
    Ssao,
    Forward,
    Transmission,
    Taa,
    Particles,
    DebugDraw,
    Bloom,
    Post,
    Count
};

inline constexpr uint32_t kProfilePassCount = std::to_underlying(ProfilePass::Count);

// Resolved per-frame timings consumed by the overlay.
struct FrameStats
{
    // Wall-clock frame time (from the main-loop dt). Always valid.
    float cpuFrameMs{0.0f};
    // Per-pass GPU milliseconds; only meaningful when gpuValid. A pass that
    // didn't run this cycle (e.g. transmission with no transmissive draws)
    // reports 0.
    std::array<float, kProfilePassCount> passMs{};
    float gpuTotalMs{0.0f};
    bool gpuValid{false};
    // Scene frustum-cull results from the most recent collectDrawCommands (the overlay
    // shows them a frame later). trackedNodes counts rigid renderables in the cull BVH;
    // culledNodes is how many of those fell outside every frustum this frame. Both 0 when
    // culling is disabled.
    int trackedNodes{0};
    int culledNodes{0};
    // Triangles actually submitted in the forward opaque bucket this frame (post LOD + cull), so
    // the overlay can show the LOD saving.
    int trianglesDrawn{0};
    // VDPM per-frame repair work summed over every instance (vertices each pass pulled back in) — a
    // diagnostic so a repair-count regression is visible in the overlay. 0 outside ViewDependent
    // LOD.
    int vdpmFoldoversRepaired{0};
    int vdpmCoverageRepaired{0};
    // VDPM per-channel refine attribution (which metric channel won each over-budget trigger,
    // summed over instances) + the largest score/budget ratio each channel reached (max over
    // instances) — the counts say what drove detail, the ratios say how close an under-firing
    // channel came. 0 outside ViewDependent.
    int vdpmGeometryTriggers{0};
    int vdpmUvTriggers{0};
    int vdpmNormalTriggers{0};
    int vdpmTangentTriggers{0};
    float vdpmMaxGeometryRatio{0.0f};
    float vdpmMaxUvRatio{0.0f};
    float vdpmMaxNormalRatio{0.0f};
    float vdpmMaxTangentRatio{0.0f};

    // GPU-driven-front (B5b) performance instrumentation — populated only when the GPU VDPM backend
    // recorded compute this frame (else all 0). CPU wall time is measured around the manager's
    // recordRequests; the GPU compute time is the ProfilePass::VdpmCompute timestamp above
    // (passMs). The dispatch/barrier totals are ANALYTIC (computed from each front's rank count +
    // the repair round budget, per the recorder structure) — they quantify the command-count cost
    // this arc targets, independent of the shader time.
    float vdpmRecordCpuMs{0.0f};   // CPU cost of recording all fronts' compute
    int vdpmFrontsRecorded{0};     // fronts whose lifecycle was recorded this frame
    int vdpmMaxRankCount{0};       // largest per-front rank count R across those fronts
    int vdpmRepairRoundBudget{0};  // the repair round budget B those dispatches assumed
    int vdpmAnalyticDispatches{0}; // Σ over fronts of the analytic compute-dispatch count
    int vdpmAnalyticBarriers{0};   // Σ over fronts of the analytic pipeline-barrier count
};

class GpuProfiler
{
public:
    explicit GpuProfiler(const Device& device);
    ~GpuProfiler() = default;

    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;
    GpuProfiler(GpuProfiler&&) noexcept = default;
    GpuProfiler& operator=(GpuProfiler&&) noexcept = default;

    // Reset this frame's query range. Must be recorded outside a render pass,
    // before any begin()/end() for the frame.
    void beginFrame(vk::CommandBuffer cmd, uint32_t frameIndex);
    void begin(vk::CommandBuffer cmd, uint32_t frameIndex, ProfilePass pass) const;
    void end(vk::CommandBuffer cmd, uint32_t frameIndex, ProfilePass pass) const;

    // Read back the results currently held in `frameIndex`'s slot (written a full
    // ring-cycle ago) into out.passMs / gpuTotalMs / gpuValid. No-op when timing
    // is unsupported.
    void resolve(uint32_t frameIndex, FrameStats& out) const;

    [[nodiscard]] bool enabled() const noexcept
    {
        return enabled_;
    }

private:
    [[nodiscard]] uint32_t queryIndex(uint32_t frameIndex, ProfilePass pass,
                                      bool end) const noexcept
    {
        return frameIndex * kQueriesPerFrame + static_cast<uint32_t>(pass) * 2 + (end ? 1u : 0u);
    }

    static constexpr uint32_t kQueriesPerFrame = kProfilePassCount * 2;

    const vk::raii::Device* device_{nullptr};
    vk::raii::QueryPool pool_{nullptr};
    float timestampPeriodNs_{0.0f};
    bool enabled_{false};
    // A slot must be reset (beginFrame) at least once before its results may be
    // read, or validation flags reading uninitialised queries on the first cycle.
    std::array<bool, kMaxFramesInFlight> slotUsed_{};
};

} // namespace fire_engine
