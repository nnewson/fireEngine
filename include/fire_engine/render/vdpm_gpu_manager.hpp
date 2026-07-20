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
    void recordRequests(vk::CommandBuffer cmd, std::span<const VdpmWorkRequest> requests,
                        const VdpmFrameGlobals& globals);

    // Per-frame compute-command instrumentation from the most recent recordRequests (perf arc, no
    // behaviour change): how many fronts recorded, the largest rank count among them, the repair
    // round budget those dispatches assumed, and the ANALYTIC dispatch/barrier totals (Σ over
    // fronts of VdpmGpuFront::analyticComputeCost). Zeroed at the top of each recordRequests.
    struct ComputeStats
    {
        std::uint32_t frontsRecorded{0};
        std::uint32_t maxRankCount{0};
        std::uint32_t roundBudget{0};
        std::uint32_t analyticDispatches{0};
        std::uint32_t analyticBarriers{0};
        // Per-stage CPU record ms summed across the fronts recorded this frame ([score, apply,
        // repair, emit]) — the apply-kernel checkpoint's CPU evidence; populated when
        // VdpmFrameGlobals::stageProfiler is set. GPU per-stage ms come from the profiler passes.
        std::array<float, 4> stageCpuMs{};
    };
    [[nodiscard]] const ComputeStats& lastComputeStats() const noexcept
    {
        return lastComputeStats_;
    }

    // Record the delayed diagnostic readback for a representative front (perf arc, item 4). Called
    // by the Renderer AFTER the VdpmCompute profiler pass ends (so the tiny copies don't perturb
    // the compute timestamp): it parses the ring slot written a full frames-in-flight cycle ago
    // (already complete — no stall) into lastDiagnostics(), then records the copy of this frame's
    // front convergence buffers into that slot. No-op if no front recorded this frame.
    void recordDiagnosticReadback(vk::CommandBuffer cmd, std::uint32_t frameIndex);

    // The most recently READ-BACK per-round convergence diagnostics (from ~kMaxFramesInFlight
    // frames ago). `valid` is false until the first slot completes. `markedRounds` is the count of
    // leading repair rounds whose detect found a violation (the convergence point); `cleanPrefix`
    // is false if a marked round followed a clean one (a repair/sync bug under a fixed view).
    // finalAnyMarked = the post-budget detect still found a violation; fallbackFired = the
    // full-detail seed ran; emittedIndexCount = the drawn index count.
    struct Diagnostics
    {
        bool valid{false};
        std::uint32_t roundBudget{0};
        std::uint32_t markedRounds{0};
        bool cleanPrefix{true};
        std::uint32_t finalAnyMarked{0};
        std::uint32_t fallbackFired{0};
        std::uint32_t emittedIndexCount{0};
    };
    [[nodiscard]] const Diagnostics& lastDiagnostics() const noexcept
    {
        return lastDiagnostics_;
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

private:
    struct FrontSlot
    {
        std::optional<VdpmGpuFront> front;
        std::uint32_t meshIndex{0}; // the mesh this front was built over
    };

    [[nodiscard]] VdpmGpuFront* resolveFront(VdpmFrontHandle front) noexcept;
    [[nodiscard]] const VdpmGpuFront* resolveFront(VdpmFrontHandle front) const noexcept;

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

    ComputeStats lastComputeStats_{};

    // Delayed per-round convergence diagnostics (perf arc, item 4). `diagReadback_` is a
    // host-visible ring (one slot per frame-in-flight) each holding [roundHistory[0..B),
    // repairControl[0..4), counters[0..3)] copied from `diagFront_` — the first front recorded this
    // frame; reading slot `frameIndex` recovers the copy from a full cycle ago (already complete).
    // Parsed into lastDiagnostics_.
    MappedBufferSet diagReadback_{};
    VdpmGpuFront* diagFront_{nullptr};
    std::array<bool, kMaxFramesInFlight> diagSlotWritten_{}; // slot has real data to parse
    Diagnostics lastDiagnostics_{};
    std::uint32_t lastLoggedMarkedRounds_{static_cast<std::uint32_t>(-1)};

    // Log a dispatch-limit ineligibility fallback only once (else one line per ineligible mesh).
    bool loggedDispatchFallback_{false};
    // The front count last logged by recordRequests, so the debug trace fires only when the count
    // changes (an activation/count-change proof) rather than every frame. SIZE_MAX ⇒ never logged,
    // so the first recorded frame always traces. Retired when B5c adds real diagnostics.
    std::size_t lastLoggedRequestCount_{static_cast<std::size_t>(-1)};
};

} // namespace fire_engine
