#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/generational_slot_pool.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/vdpm_gpu_registry.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/compute_pipeline.hpp>
#include <fire_engine/render/descriptors.hpp>
#include <fire_engine/render/vdpm_gpu.hpp>

namespace fire_engine
{

class Device;
class Resources;

// The frame-global inputs the manager pairs with each work request's per-instance data to derive
// the per-front VdpmViewParams + VdpmRepairParams. The Renderer owns these (camera + viewport +
// budgets) and supplies them ONCE per frame, never per instance — the same values the CPU front
// reads from FrameInfo, so the two backends stay bit-for-bit comparable. `viewProj` is the
// JITTER-FREE view-projection (TAA jitter would thrash the coverage test); `projScaleY` is
// |proj[1][1]| (world length → NDC height), kept separate because it can't be recovered from the
// combined viewProj. The refine pixel budget arrives here; the coarsen budget (kVdpmCoarsenRatio ×
// budget) is derived by the manager.
struct VdpmFrameGlobals
{
    Mat4 viewProj;
    Vec3 cameraPos;
    float projScaleY{1.0f};
    float viewportWidth{0.0f};
    float viewportHeight{0.0f};
    float pixelBudget{0.0f};
    std::uint32_t frameIndex{0};
    // Optional per-stage timing (apply-kernel arc checkpoint). When set, recordRequests measures
    // each front's score/apply/repair/emit CPU record cost, and — ONLY when a single front is
    // recorded — writes GPU timestamps into the VdpmScore/Apply/Repair/Emit profiler passes.
    // nullptr ⇒ off.
    const GpuProfiler* stageProfiler{nullptr};
};

// One GPU-front's forward-draw count this frame (B5c-1): how many camera-visible forward
// DrawCommands a given front backs. The health reduction weights each front's emitted-index total
// by this so the scene triangle total carries submitted-draw semantics (a front backing >1 forward
// command counts per command). Coupled to the handle — NOT a positional span — so later
// filtering/reordering can't desync it from the request. Normally the count is 1.
struct VdpmFrontDrawCount
{
    VdpmFrontHandle front{NullVdpmFront};
    std::uint32_t drawCount{1};
};

// Owns every Vulkan object the GPU-driven VDPM front needs — the reusable pipeline bundles (score /
// refine / repair / emit) and the per-mesh + per-instance GPU-front tables — and implements the
// Vulkan-free VdpmGpuRegistry the load path registers geometry/instances through. Constructed by
// the Renderer ONLY when the device meets the compute/scan capability (VdpmScan::deviceSupported);
// on unsupported hardware the Renderer never builds it and every instance stays on the CPU front,
// so Renderer construction never fails over a missing GPU-front capability. Non-movable (the
// pipelines hold a device pointer).
class VdpmGpuManager final : public VdpmGpuRegistry
{
public:
    // Precondition: VdpmScan::deviceSupported(device) — the Renderer checks it before constructing.
    VdpmGpuManager(const Device& device, Resources& resources);

    VdpmGpuManager(const VdpmGpuManager&) = delete;
    VdpmGpuManager& operator=(const VdpmGpuManager&) = delete;
    VdpmGpuManager(VdpmGpuManager&&) = delete;
    VdpmGpuManager& operator=(VdpmGpuManager&&) = delete;
    ~VdpmGpuManager() override = default;

    // --- VdpmGpuRegistry (load-time; never per-frame)
    // ---------------------------------------------
    [[nodiscard]] bool available() const noexcept override
    {
        return true; // constructed only on capable hardware
    }
    [[nodiscard]] VdpmMeshHandle registerMesh(std::span<const Vertex> vertices,
                                              std::span<const std::uint32_t> indices,
                                              std::span<const MeshCollapse> collapses) override;
    [[nodiscard]] VdpmFrontHandle createFront(VdpmMeshHandle mesh) override;

    // --- Per-frame
    // -------------------------------------------------------------------------------- Record the
    // full GPU front lifecycle (score → refine/coarsen → repair → emit) for each request into frame
    // slot `globals.frameIndex`. `requests` MUST be pre-deduped by the caller (Renderer): each live
    // front appears at most once, or its persistent state would be scored twice in a frame. Records
    // NO consumer barrier after the last emit — the caller adds the compute→(index + indirect read)
    // barrier before the draw. A request whose front handle is stale/invalid is skipped.
    // Frame-global camera/budget data comes from `globals`.
    // `drawCounts` (B5c-1): per-front forward-draw counts for the emitted-triangle weighting, keyed
    // by handle. When supplied it MUST be a complete 1:1 keyed cover of the request set — a
    // zero-count, unknown, duplicate, or missing entry throws std::logic_error (a filtering
    // mismatch must fail loudly, not silently weight 1). Empty ⇒ every front weights 1 (the test
    // convenience).
    void recordRequests(vk::CommandBuffer cmd, std::span<const VdpmWorkRequest> requests,
                        const VdpmFrameGlobals& globals,
                        std::span<const VdpmFrontDrawCount> drawCounts = {});

    // Per-frame compute-command instrumentation from the most recent recordRequests (perf arc, no
    // behaviour change): how many fronts recorded, the largest rank count among them, the repair
    // round budget those dispatches assumed, and the ANALYTIC dispatch/barrier totals — the
    // STAGE-MAJOR aggregate (VdpmGpuFront::analyticBatchedCost) on the batched path, or the Σ of
    // per-front VdpmGpuFront::analyticComputeCost on the per-front path. Zeroed at the top of each
    // recordRequests.
    struct ComputeStats
    {
        std::uint32_t frontsRecorded{0};
        std::uint32_t maxRankCount{0};
        std::uint32_t roundBudget{0};
        std::uint32_t analyticDispatches{0};
        std::uint32_t analyticBarriers{0};
        // Compacted batched job counts (front-batching arc): Na apply jobs (split-bearing fronts),
        // Nr repair jobs (additionally finestFaceCount>0). Set on the batched path; 0 on the
        // per-front path. Make the analytic dispatch composition observable (score+emit are still
        // per-front, so e.g. 13 fronts → 13 score + Σemit + Na-chunks + Nr-chunks).
        std::uint32_t applyJobs{0};
        std::uint32_t repairJobs{0};
        // Per-stage CPU record ms summed across the fronts recorded this frame ([score, apply,
        // repair, emit]) — the apply-kernel checkpoint's CPU evidence; populated when
        // VdpmFrameGlobals::stageProfiler is set. GPU per-stage ms come from the profiler passes.
        std::array<float, 4> stageCpuMs{};
    };
    [[nodiscard]] const ComputeStats& lastComputeStats() const noexcept
    {
        return lastComputeStats_;
    }

    // Record the delayed scene-health readback (B5c-1). Called by the Renderer AFTER the
    // VdpmCompute profiler pass ends (so the copy doesn't perturb the compute timestamp): it (1)
    // parses the host ring slot written a full frames-in-flight cycle ago (already complete — no
    // stall) together with that same frame's stored `cpuTriangleSubtotal` into lastSceneHealth();
    // (2) records the compute→transfer copy of THIS frame's reduced scene-health buffer into that
    // slot (+ a transfer-write→host-read barrier); (3) stores this frame's `cpuTriangleSubtotal`
    // for the next ring cycle. `cpuTriangleSubtotal` is this frame's triangle count from NON-GPU
    // draws (GPU draws carry indexCount 0), combined with the delayed GPU emitted total so the
    // displayed total is internally from one frame. No-op if no front recorded this frame.
    void recordDiagnosticReadback(vk::CommandBuffer cmd, std::uint32_t frameIndex,
                                  std::uint64_t cpuTriangleSubtotal);

    // Invalidate frame slot `frameIndex`'s readback ring entry (B5c-1). The Renderer calls this on
    // a frame where the GPU backend records NO fronts (backend toggled off, or nothing visible):
    // that frame writes no reduction, so its slot must not be parsed a cycle later as if it carried
    // the normal frames-in-flight delay. A slot is only ever parsed when the frame that wrote it (a
    // full cycle ago) was ACTIVE — so after any inactive gap the readback stays invalid until the
    // ring re-warms with live data.
    void invalidateHealthSlot(std::uint32_t frameIndex) noexcept
    {
        diagSlotWritten_[frameIndex] = false;
    }

    // The most recently READ-BACK scene-wide GPU-front HEALTH (from ~kMaxFramesInFlight frames ago)
    // — health-oriented, NOT the CPU foldover/coverage vertex counts. `valid` is false until the
    // first slot completes. `triangleTotal` combines the delayed GPU emitted triangles with the
    // SAME frame's CPU-front triangle subtotal (frame-consistent). The repair fields summarise
    // convergence across all repair-bearing fronts; any non-zero
    // fallback/non-clean/ancestor/fail-flag count is a health signal worth surfacing.
    struct SceneHealth
    {
        bool valid{false};
        std::uint64_t triangleTotal{0};    // GPU emitted (delayed) + CPU-front subtotal, same frame
        bool emittedOverflow{false};       // the 64-bit GPU emitted-index total wrapped (guard)
        std::uint32_t repairFronts{0};     // fronts that ran repair this frame
        std::uint32_t maxMarkedRounds{0};  // max leading marked-round count across them
        std::uint32_t sumMarkedRounds{0};  // Σ leading marked-round counts
        std::uint32_t roundBudget{0};      // the per-front repair round budget
        std::uint32_t fallbackFronts{0};   // fronts whose full-detail fallback fired
        std::uint32_t nonCleanPrefix{0};   // fronts with a marked round after a clean one
        std::uint32_t ancestorFailures{0}; // fronts with an ancestor-resolve failure
        std::uint32_t failFlagFronts{0};   // fronts with a B3 refine/coarsen failure flag
    };
    [[nodiscard]] const SceneHealth& lastSceneHealth() const noexcept
    {
        return lastSceneHealth_;
    }

    // A GPU-driven front's draw-consumed buffers for one frame slot: the GPU-emitted index stream
    // (always uint32) + the GPU-written indirect command. The count lives in the indirect command.
    struct DrawBuffers
    {
        BufferHandle index{NullBuffer};
        BufferHandle indirect{NullBuffer};
    };

    // Resolve a GPU-driven front's draw buffers for `frameIndex` in ONE handle lookup — the B5b-2
    // draw switch points a tagged forward DrawCommand at these. The handle came from createFront,
    // so it MUST resolve to a live front with allocated buffers: a null lookup or a null buffer is
    // an INVARIANT VIOLATION (eligibility was decided at registration, and a GPU-backed draw
    // carries indexCount 0 — a silent miss would issue a zero-count draw, not a finest fallback).
    // So this THROWS std::logic_error rather than letting that through. Legitimate CPU fallback is
    // chosen upstream in Object (a NullVdpmFront binding never reaches here). frameIndex must be a
    // valid frame-in-flight slot.
    [[nodiscard]] DrawBuffers resolveDrawBuffers(VdpmFrontHandle front,
                                                 std::uint32_t frameIndex) const;

    // Soft query of a front's ring buffers for `frameIndex` — returns NullBuffer on a stale/invalid
    // handle (the resolve-or-throw contract lives in resolveDrawBuffers). Kept for tests + probing.
    [[nodiscard]] BufferHandle frontIndexBuffer(VdpmFrontHandle front,
                                                std::uint32_t frameIndex) const;
    [[nodiscard]] BufferHandle frontIndirectBuffer(VdpmFrontHandle front,
                                                   std::uint32_t frameIndex) const;

    // TEST-ONLY read-only introspection: the live front behind a handle, so the
    // batched-vs-per-front cross-check ([.][gpu]) can read its already-public state buffers
    // (active/refined/dependents/ required/failFlags) after a batched recordRequests. nullptr on a
    // stale/invalid handle. Not for production use — the draw path goes through resolveDrawBuffers.
    [[nodiscard]] const VdpmGpuFront* frontForTest(VdpmFrontHandle front) const noexcept
    {
        return resolveFront(front);
    }

    // TEST-ONLY: force the batched-dispatch 1-D group cap so the chunked advanced-BDA dispatch path
    // (⌈N/cap⌉ chunks advancing the job BDA by firstJob*stride) is actually exercised — the real
    // device cap is enormous, so N never chunks in practice. 0 ⇒ use the device cap (production).
    void setTestGroupCapOverride(std::uint32_t cap) noexcept
    {
        testGroupCapOverride_ = cap;
    }

private:
    struct FrontSlot
    {
        std::optional<VdpmGpuFront> front;
        std::uint32_t meshIndex{0}; // the mesh this front was built over
    };

    // One resolved work request (front-batching arc): the live front + its derived per-frame params
    // + the stage work flags, resolved ONCE per recordRequests into the reused `resolveScratch_` so
    // no stage re-resolves handles or allocates. `hasApply` (splitCount>0 ⇒ score + an apply job);
    // `hasRepair` (splitCount>0 && finestFaceCount>0 ⇒ a repair job) — the compaction predicates.
    struct ResolvedRequest
    {
        VdpmGpuFront* front{nullptr};
        VdpmViewParams view{};
        VdpmRepairParams repair{};
        bool hasApply{false};
        bool hasRepair{false};
        std::uint32_t drawMultiplier{
            1}; // forward draws this front backs (health emitted weighting)
    };

    [[nodiscard]] VdpmGpuFront* resolveFront(VdpmFrontHandle front) noexcept;
    [[nodiscard]] const VdpmGpuFront* resolveFront(VdpmFrontHandle front) const noexcept;

    // Grow the batched job arrays to hold >= `frontCount` compacted jobs (geometric). A replaced
    // array's buffer is not freed (Resources is session-lifetime), so an in-flight frame's dispatch
    // referencing the old BDA stays valid. Called from recordBatched before any dispatch references
    // an array. No-op once capacity suffices.
    void ensureJobCapacity(std::uint32_t frontCount);

    // Grow the health-job array to hold >= `frontCount` jobs (geometric; same session-lifetime
    // reasoning as ensureJobCapacity). The health reduction runs on BOTH record paths, so this is
    // called for every recorded frame (not only the batched path).
    void ensureHealthCapacity(std::uint32_t frontCount);

    // Record the scene-wide health reduction (B5c-1) at the END of recordRequests, inside the
    // VdpmCompute timestamp: an emit→health compute barrier, fill the health-job array from
    // `resolveScratch_` (each front's prepareHealthJob with its drawMultiplier), then a
    // single-invocation dispatch writing this frame's `sceneHealthDeviceRing_` slot. No-op if no
    // front recorded.
    void recordHealthReduction(vk::CommandBuffer cmd, std::uint32_t frameIndex);

    // The STAGE-MAJOR batched record path (front-batching arc), driven from `resolveScratch_`: one
    // lifecycle barrier → score every front → one dispatch(Na) apply → one dispatch(Nr) repair
    // (both over compacted, chunked job arrays) → one final front-state→emit barrier → emit every
    // front. Taken when BOTH kernels are available and the frame is not a profiled single front
    // (see recordRequests). Fills lastComputeStats_ from the BATCHED aggregate
    // (analyticBatchedCost).
    void recordBatched(vk::CommandBuffer cmd, const VdpmFrameGlobals& globals, float coarsenBudget);

    // The per-front reference path (recordFrame per resolved request) — the original behaviour,
    // used when a kernel is missing or the frame is a single profiled front (preserves per-stage
    // GPU timestamps). Fills lastComputeStats_ from the per-front analyticComputeCost sum.
    void recordPerFront(vk::CommandBuffer cmd, const VdpmFrameGlobals& globals, float coarsenBudget,
                        const VdpmStageProfile* stageProfilePtr);

    Resources& resources_;

    // The reusable pipeline bundles, built once. Non-movable, so the manager is too. The
    // multi-dispatch repair recorder (repairPipelines_) stays built even when the kernel is present
    // — it is the intentional reference/fallback for a device that fails VdpmRepairKernel support.
    ComputePipeline scorePipeline_;
    VdpmRefinePipelines refinePipelines_;
    VdpmRepairPipelines repairPipelines_;
    VdpmEmitPipelines emitPipelines_;
    // The single-dispatch persistent repair kernel (Stage 3) — emplaced ONLY when the device meets
    // VdpmRepairKernel::deviceSupported (independent of the VdpmScan gate). recordFrame is handed
    // `repairKernel_ ? &*repairKernel_ : nullptr`, so an unsupported device keeps the recorder.
    std::optional<VdpmRepairKernel> repairKernel_;
    // The single-dispatch persistent APPLY kernel (apply-kernel arc) — same pattern: emplaced iff
    // VdpmApplyKernel::deviceSupported, else recordFrame gets nullptr and keeps the recorder
    // (recordApplyScoredView). The recorder pipelines (refinePipelines_) stay built regardless.
    std::optional<VdpmApplyKernel> applyKernel_;

    // Per-mesh forests (shared by every instance of a geometry) and per-instance fronts, each keyed
    // by a generational handle. The optional lets a recycled slot be re-emplaced.
    GenerationalSlotPool meshPool_;
    std::vector<std::optional<VdpmGpuMesh>> meshes_;
    GenerationalSlotPool frontPool_;
    std::vector<FrontSlot> fronts_;

    // Reused per-frame scratch (front-batching arc) — resolved once per recordRequests, never
    // per-stage, so no stage re-resolves handles or allocates. `dimsScratch_` feeds the batched
    // aggregate accounting (analyticBatchedCost).
    std::vector<ResolvedRequest> resolveScratch_;
    std::vector<VdpmGpuFront::FrontDims> dimsScratch_;

    // Stage-major batched job arrays (front-batching arc): one host-visible device-address buffer
    // PER frame slot (a MappedBufferSet each), holding up to `jobCapacity_` COMPACTED jobs.
    // recordBatched fills slot [frameIndex] and issues one dispatch(N) per stage. Grown
    // geometrically by ensureJobCapacity when the live front count exceeds capacity. LIFETIME:
    // `Resources` never frees a buffer within a session (there is no releaseBuffer — buffers live
    // until Resources is destroyed), so a grown-over array's underlying buffer stays valid for any
    // in-flight frame whose dispatch still references its BDA; reassigning applyJobArray_ drops
    // only the handle copy, not the resource. Growth is load-time-rare (fronts register at load).
    MappedBufferSet applyJobArray_{};
    MappedBufferSet repairJobArray_{};
    std::uint32_t jobCapacity_{0};
    // TEST-ONLY chunk-cap override (0 ⇒ device cap); see setTestGroupCapOverride.
    std::uint32_t testGroupCapOverride_{0};

    ComputeStats lastComputeStats_{};

    // Scene-wide health reduction (B5c-1). The single-invocation `vdpm_health_reduce.comp` folds
    // every recorded front's health (via a per-front VdpmHealthJobGpu array) into ONE
    // VdpmSceneHealthGpu in the per-frame `sceneHealthDeviceRing_` slot (device-local, ringed so a
    // later frame's reduction can't clobber an unread slot). recordDiagnosticReadback copies that
    // into `diagReadback_` (the host-visible readback ring) and parses the slot from a full cycle
    // ago, combined with that frame's `cpuTriangleSubtotal_`.
    ComputePipeline healthReducePipeline_;
    MappedBufferSet healthJobArray_{}; // per-front VdpmHealthJobGpu, one slot per frame-in-flight
    std::uint32_t healthJobCapacity_{0};
    std::array<BufferHandle, kMaxFramesInFlight> sceneHealthDeviceRing_{};
    std::array<std::uint64_t, kMaxFramesInFlight> sceneHealthDeviceAddress_{};
    MappedBufferSet diagReadback_{}; // host ring: VdpmSceneHealthGpu per slot
    std::array<std::uint64_t, kMaxFramesInFlight>
        cpuTriangleSubtotal_{};                              // same-frame CPU-front tris
    std::array<bool, kMaxFramesInFlight> diagSlotWritten_{}; // slot's data is from an ACTIVE frame
    SceneHealth lastSceneHealth_{};
    // Last-logged discrete health signature {repair, fallback, non-clean, ancestor, B3-fail} — logs
    // on ANY transition, not just a repair-front count change. Sentinel so the first frame logs.
    std::array<std::uint32_t, 5> lastLoggedHealth_{static_cast<std::uint32_t>(-1), 0, 0, 0, 0};

    // Log a dispatch-limit ineligibility fallback only once (else one line per ineligible mesh).
    bool loggedDispatchFallback_{false};
    // The front count last logged by recordRequests, so the debug trace fires only when the count
    // changes (an activation/count-change proof) rather than every frame. SIZE_MAX ⇒ never logged,
    // so the first recorded frame always traces. Retired when B5c adds real diagnostics.
    std::size_t lastLoggedRequestCount_{static_cast<std::size_t>(-1)};
};

} // namespace fire_engine
