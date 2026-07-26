#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/shadow_diagnostics.hpp>

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
    // Per-stage breakdown of VdpmCompute (score/apply/repair/emit), written by recordFrame ONLY
    // when a single front is recorded (the query slots are one-shot per frame, so bracketing N>1
    // fronts would double-write them). Report 0 on frames with 0 or >1 fronts. The apply-kernel
    // arc's checkpoint measures which stage dominates GPU time.
    VdpmScore,
    VdpmApply,
    VdpmRepair,
    VdpmEmit,
    // The five shadow families, timed separately (SH-01): a single Shadow total could not say
    // whether a change moved cost between cascades, punctual lights, or self-shadowing. They are
    // disjoint spans — there is no outer Shadow timer to overlap them — so the frame total sums
    // them like any other pass.
    ShadowCascades,
    ShadowWorldOnly,
    ShadowSelf,
    ShadowSpot,
    ShadowPoint,
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

// Display names, kept BESIDE the enum so adding a pass without naming it fails to compile. The
// previous table lived in the overlay with fewer initializers than passes, which silently
// misaligned every name after the gap and left the tail null.
inline constexpr std::array kProfilePassNames{
    "VDPM compute", "VDPM score",   "VDPM apply",  "VDPM repair",  "VDPM emit",
    "Shadow CSM",   "Shadow world", "Shadow self", "Shadow spot",  "Shadow point",
    "Depth",        "SSAO",         "Forward",     "Transmission", "TAA",
    "Particles",    "Debug",        "Bloom",       "Post"};
static_assert(kProfilePassNames.size() == kProfilePassCount,
              "every ProfilePass needs a display name, in enum order");

// Whether a pass's time belongs in the frame total. The four VDPM breakdown rows are SUBRANGES of
// VdpmCompute, so summing them alongside it double-counts that work; every other pass is a disjoint
// span and contributes. Pure, so the overlay and any future consumer share one policy.
[[nodiscard]] constexpr bool profilePassContributesToTotal(ProfilePass pass) noexcept
{
    switch (pass)
    {
    case ProfilePass::VdpmScore:
    case ProfilePass::VdpmApply:
    case ProfilePass::VdpmRepair:
    case ProfilePass::VdpmEmit:
        return false;
    default:
        return true;
    }
}

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
    // SH-01 shadow diagnostics from the COMPLETED frame whose ring slot this is (see
    // Renderer::shadowStatsRing_). `shadowValid` is false during ring warm-up and is entirely
    // independent of `gpuValid` — the counters are CPU-side and survive a device with no timestamp
    // support.
    ShadowFrameStats shadow{};
    bool shadowValid{false};
    // Scene frustum-cull results from the most recent collectDrawCommands (the overlay
    // shows them a frame later). trackedNodes counts rigid renderables in the cull BVH;
    // culledNodes is how many of those fell outside every frustum this frame. Both 0 when
    // culling is disabled.
    int trackedNodes{0};
    int culledNodes{0};
    // Triangles actually submitted this frame (post LOD + cull), so the overlay can show the LOD
    // saving. 64-bit: a GPU-front scene's combined emitted total (delayed, frame-consistent) can in
    // principle exceed 32 bits. `trianglesOverflow` flags a (unreachable) 64-bit wrap of the
    // combine.
    std::uint64_t trianglesDrawn{0};
    bool trianglesOverflow{false};
    // True when the GPU VDPM backend recorded fronts this frame but their emitted total isn't back
    // yet (ring warming / just after an inactive gap): `trianglesDrawn` then holds only the
    // CPU-front subtotal, which is INCOMPLETE — the overlay shows "pending" rather than a plausible
    // partial.
    bool trianglesGpuPending{false};
    // True iff the device supports the GPU-driven VDPM front (the manager was constructed) —
    // independent of whether the runtime selector currently has it ACTIVE. Drives the overlay's
    // backend checkbox: enabled when available, an explicit "unsupported" label otherwise.
    bool vdpmGpuAvailable{false};
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
    float vdpmRecordCpuMs{0.0f};  // CPU cost of recording all fronts' compute
    int vdpmFrontsRecorded{0};    // fronts whose lifecycle was recorded this frame
    int vdpmMaxRankCount{0};      // largest per-front rank count R across those fronts
    int vdpmRepairRoundBudget{0}; // the repair round budget B those dispatches assumed
    int vdpmAnalyticDispatches{
        0}; // analytic compute-dispatch count (batched aggregate / per-front Σ)
    int vdpmAnalyticBarriers{0}; // analytic pipeline-barrier count
    int vdpmApplyJobs{0};        // batched Na (compacted apply jobs); 0 on the per-front path
    int vdpmRepairJobs{0};       // batched Nr (compacted repair jobs)
    // Delayed SCENE-WIDE GPU-front health (B5c-1), read back ~kMaxFramesInFlight frames late from
    // the health reduction. Health-oriented, NOT the CPU foldover/coverage counts. -1 / 0 until the
    // first slot completes. `trianglesDrawn` above already carries the frame-consistent combined
    // total when the GPU backend is active.
    int vdpmRepairFronts{0};     // fronts that ran repair this frame
    int vdpmMaxMarkedRounds{-1}; // max leading marked-round count across them (of the round budget)
    int vdpmSumMarkedRounds{0};  // Σ leading marked-round counts
    int vdpmFallbackFronts{0};   // fronts whose full-detail fallback fired (a convergence stress)
    int vdpmNonCleanPrefix{0};   // fronts with a marked round after a clean one (a repair/sync bug)
    int vdpmAncestorFailures{0}; // fronts with an ancestor-resolve failure
    int vdpmFailFlagFronts{0};   // fronts with a B3 refine/coarsen failure flag
    bool vdpmEmittedOverflow{false}; // the 64-bit emitted-index total wrapped (guard)
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

    // Write a pass's begin (`end`=false) or end (`end`=true) query at BOTTOM-of-pipe. begin() uses
    // top-of-pipe, which is fine for a coarse whole-pass span but bleeds badly across adjacent
    // SUB-millisecond stages (a stage's top-of-pipe begin fires while the prior stage is still
    // draining). For the VDPM per-stage breakdown the boundaries are stamped bottom-of-pipe on BOTH
    // sides — the shared boundary written as one stage's end AND the next stage's begin — so each
    // resolved passMs is a clean consecutive delta. resolve() is unchanged (end − begin per pass).
    void stampBottom(vk::CommandBuffer cmd, uint32_t frameIndex, ProfilePass pass, bool end) const;

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
