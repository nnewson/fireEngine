#include <fire_engine/render/vdpm_gpu_manager.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>

#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/mapped_buffer.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{
// Chunked batched dispatch advances a job array's BDA by firstJob * sizeof(Job); the kernels
// declare the JobBuf buffer reference `buffer_reference_align = 8`, so each job stride must be an
// 8-multiple for every chunk's base address to keep that promise (front-batching arc).
static_assert(sizeof(VdpmApplyJobGpu) % 8 == 0,
              "apply job stride must satisfy buffer_reference_align=8");
static_assert(sizeof(VdpmRepairJobGpu) % 8 == 0,
              "repair job stride must satisfy buffer_reference_align=8");
} // namespace

VdpmGpuManager::VdpmGpuManager(const Device& device, Resources& resources)
    : resources_(resources),
      scorePipeline_(device, vdpmScorePipelineConfig()),
      refinePipelines_(device),
      repairPipelines_(device),
      emitPipelines_(device),
      healthReducePipeline_(device, vdpmHealthReducePipelineConfig())
{
    // Scene-health readback (B5c-1): the reduction writes one VdpmSceneHealthGpu into a
    // device-local ring slot (per frame-in-flight, eTransferSrc for the copy);
    // recordDiagnosticReadback copies it into `diagReadback_` (host-visible) and reads the slot
    // from a full cycle ago.
    diagReadback_ = resources_.createMappedReadbackBuffers(sizeof(VdpmSceneHealthGpu));
    for (std::uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        sceneHealthDeviceRing_[i] = resources_.createDeviceLocalStorageBuffer(
            sizeof(VdpmSceneHealthGpu), nullptr, vk::BufferUsageFlagBits::eTransferSrc);
        sceneHealthDeviceAddress_[i] = resources_.bufferAddress(sceneHealthDeviceRing_[i]);
    }

    // The single-dispatch persistent repair kernel (Stage 3) — built only when the device supports
    // it (checked independently of the VdpmScan gate that let this manager be constructed).
    // Unsupported hardware leaves it empty and every front repairs through the multi-dispatch
    // recorder.
    if (VdpmRepairKernel::deviceSupported(device))
    {
        repairKernel_.emplace(device);
    }
    // The persistent apply kernel (apply-kernel arc) — same independent capability gate. Built at
    // the default workgroup size (VdpmApplyKernel::kLocalSize, baked from the size sweep).
    if (VdpmApplyKernel::deviceSupported(device))
    {
        applyKernel_.emplace(device);
    }
}

VdpmMeshHandle VdpmGpuManager::registerMesh(std::span<const Vertex> vertices,
                                            std::span<const std::uint32_t> indices,
                                            std::span<const MeshCollapse> collapses)
{
    // The forest is derived here from the same inputs the CPU front uses, so the two backends can
    // never disagree on the topology. buildVertexForest / VdpmGpuMesh::build run every validation
    // (throwing) BEFORE any GPU allocation, so a malformed forest surfaces rather than
    // half-uploads.
    VertexForest forest = buildVertexForest(vertices, collapses);

    // Hard-capability gate: a forest/index count whose static dispatches would exceed the device's
    // 1-D group-count cap can't run on the GPU front at all. Fall back to the CPU front (logged
    // once — one line per ineligible mesh would be noise) instead of faulting at frame-record time.
    if (!VdpmGpuMesh::fitsComputeDispatchLimits(resources_, forest, indices.size()))
    {
        if (!loggedDispatchFallback_)
        {
            log::warn(log::category::render,
                      "VDPM GPU: a mesh exceeds this device's compute dispatch limits; it (and any "
                      "further such mesh) falls back to the CPU front");
            loggedDispatchFallback_ = true;
        }
        return NullVdpmMesh;
    }

    VdpmGpuMesh mesh = VdpmGpuMesh::build(resources_, vertices, indices, forest);
    const GenerationalSlotPool::Slot slot = meshPool_.acquire();
    if (slot.index >= meshes_.size())
    {
        meshes_.resize(slot.index + 1);
    }
    meshes_[slot.index].emplace(std::move(mesh)); // emplace resets any recycled slot's old mesh
    return makeHandle<VdpmMeshHandle>(slot.index, slot.generation);
}

VdpmFrontHandle VdpmGpuManager::createFront(VdpmMeshHandle mesh)
{
    if (mesh == NullVdpmMesh)
    {
        return NullVdpmFront;
    }
    const std::uint32_t mi = handleIndex(mesh);
    if (!meshPool_.valid(mi, handleGeneration(mesh)) || mi >= meshes_.size())
    {
        return NullVdpmFront;
    }
    // Separate has_value() guard before the dereference — the optional-access checker can't track
    // the check through the compound condition above.
    const std::optional<VdpmGpuMesh>& meshSlot = meshes_[mi];
    if (!meshSlot.has_value())
    {
        return NullVdpmFront;
    }

    VdpmGpuFront front = VdpmGpuFront::buildRuntime(resources_, *meshSlot);
    const GenerationalSlotPool::Slot slot = frontPool_.acquire();
    if (slot.index >= fronts_.size())
    {
        fronts_.resize(slot.index + 1);
    }
    fronts_[slot.index].front.emplace(std::move(front));
    fronts_[slot.index].meshIndex = mi;
    return makeHandle<VdpmFrontHandle>(slot.index, slot.generation);
}

VdpmGpuFront* VdpmGpuManager::resolveFront(VdpmFrontHandle front) noexcept
{
    if (front == NullVdpmFront)
    {
        return nullptr;
    }
    const std::uint32_t i = handleIndex(front);
    if (i >= fronts_.size() || !frontPool_.valid(i, handleGeneration(front)))
    {
        return nullptr;
    }
    // Separate has_value() guard immediately before the dereference — the optional-access checker
    // can't track the check through the compound condition above.
    std::optional<VdpmGpuFront>& slot = fronts_[i].front;
    return slot.has_value() ? &*slot : nullptr;
}

const VdpmGpuFront* VdpmGpuManager::resolveFront(VdpmFrontHandle front) const noexcept
{
    if (front == NullVdpmFront)
    {
        return nullptr;
    }
    const std::uint32_t i = handleIndex(front);
    if (i >= fronts_.size() || !frontPool_.valid(i, handleGeneration(front)))
    {
        return nullptr;
    }
    // Separate has_value() guard immediately before the dereference — the optional-access checker
    // can't track the check through the compound condition above.
    const std::optional<VdpmGpuFront>& slot = fronts_[i].front;
    return slot.has_value() ? &*slot : nullptr;
}

void VdpmGpuManager::ensureJobCapacity(std::uint32_t frontCount)
{
    if (frontCount <= jobCapacity_)
    {
        return;
    }
    std::uint32_t cap = jobCapacity_ == 0 ? 1 : jobCapacity_;
    while (cap < frontCount)
    {
        cap *= 2; // geometric growth (load-time-rare; fronts register at load)
    }
    // Reassigning the arrays drops only the old handle copies; `Resources` never frees a buffer
    // within a session (no releaseBuffer), so an OLD array's buffer stays alive for any in-flight
    // frame whose recorded dispatch still references its BDA — no retirement bookkeeping needed.
    applyJobArray_ = resources_.createMappedDeviceAddressBuffers(static_cast<std::size_t>(cap) *
                                                                 sizeof(VdpmApplyJobGpu));
    repairJobArray_ = resources_.createMappedDeviceAddressBuffers(static_cast<std::size_t>(cap) *
                                                                  sizeof(VdpmRepairJobGpu));
    jobCapacity_ = cap;
}

void VdpmGpuManager::ensureHealthCapacity(std::uint32_t frontCount)
{
    if (frontCount <= healthJobCapacity_)
    {
        return;
    }
    std::uint32_t cap = healthJobCapacity_ == 0 ? 1 : healthJobCapacity_;
    while (cap < frontCount)
    {
        cap *= 2;
    }
    // Same session-lifetime reasoning as ensureJobCapacity: the old array's buffer is never freed,
    // so an in-flight frame's reduction referencing its BDA stays valid.
    healthJobArray_ = resources_.createMappedDeviceAddressBuffers(static_cast<std::size_t>(cap) *
                                                                  sizeof(VdpmHealthJobGpu));
    healthJobCapacity_ = cap;
}

void VdpmGpuManager::recordHealthReduction(vk::CommandBuffer cmd, std::uint32_t frameIndex)
{
    const std::uint32_t n = static_cast<std::uint32_t>(resolveScratch_.size());
    for (std::uint32_t i = 0; i < n; ++i)
    {
        const ResolvedRequest& r = resolveScratch_[i];
        const VdpmHealthJobGpu job = r.front->prepareHealthJob(frameIndex, r.drawMultiplier);
        writeMapped(healthJobArray_.mapped[frameIndex].subspan(i * sizeof(VdpmHealthJobGpu)), job);
    }
    // emit → health: the reduction reads each front's counters[2] (emit wrote the emitted-index
    // count) + its repair buffers.
    VdpmGpuFront::recordComputeStageBoundary(cmd);
    const VdpmHealthReducePush push{
        .jobsAddress = resources_.bufferAddress(healthJobArray_.buffers[frameIndex]),
        .sceneHealthAddress = sceneHealthDeviceAddress_[frameIndex],
        .jobCount = n,
        .pad = 0};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, healthReducePipeline_.pipeline());
    cmd.pushConstants<VdpmHealthReducePush>(healthReducePipeline_.pipelineLayout(),
                                            vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(1, 1, 1); // one workgroup, one invocation — sequential fold over the N jobs
}

void VdpmGpuManager::recordRequests(vk::CommandBuffer cmd,
                                    std::span<const VdpmWorkRequest> requests,
                                    const VdpmFrameGlobals& globals,
                                    std::span<const VdpmFrontDrawCount> drawCounts)
{
    // The coarsen budget is the refine budget scaled by the persistent-front hysteresis ratio — the
    // same relationship the CPU refineForView uses internally.
    const float coarsenBudget = kVdpmCoarsenRatio * globals.pixelBudget;

    // Perf instrumentation (no behaviour change): tally the analytic compute-command cost as we
    // record. Reset here; the chosen record path fills it.
    lastComputeStats_ = ComputeStats{.roundBudget = kVdpmGpuRepairRoundBudget};

    // Draw-count validation (B5c-1): when supplied it MUST be a complete 1:1 keyed cover of the
    // request set — no zero-count, duplicate, unknown (front not in the requests), or missing
    // entry. A silent default-to-1 would let a future filtering mismatch produce
    // plausible-but-false totals; this fails loudly instead. Empty is the explicit "all multipliers
    // 1" convenience (tests).
    if (!drawCounts.empty())
    {
        for (const VdpmFrontDrawCount& dc : drawCounts)
        {
            if (dc.drawCount == 0)
            {
                throw std::logic_error("VdpmGpuManager::recordRequests: draw-count entry has a "
                                       "zero count");
            }
            std::size_t inRequests = 0;
            for (const VdpmWorkRequest& req : requests)
            {
                inRequests += (req.front == dc.front) ? 1 : 0;
            }
            if (inRequests == 0)
            {
                throw std::logic_error("VdpmGpuManager::recordRequests: draw-count entry for a "
                                       "front not in the request set");
            }
            std::size_t inCounts = 0;
            for (const VdpmFrontDrawCount& other : drawCounts)
            {
                inCounts += (other.front == dc.front) ? 1 : 0;
            }
            if (inCounts != 1)
            {
                throw std::logic_error("VdpmGpuManager::recordRequests: duplicate draw-count entry "
                                       "for a front");
            }
        }
        for (const VdpmWorkRequest& req : requests)
        {
            bool matched = false;
            for (const VdpmFrontDrawCount& dc : drawCounts)
            {
                if (dc.front == req.front)
                {
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                throw std::logic_error("VdpmGpuManager::recordRequests: request front has no "
                                       "draw-count entry");
            }
        }
    }

    // Resolve every request ONCE into the reused scratch: front* + derived per-frame params + stage
    // work flags. No stage below re-resolves a handle or allocates a temporary (front-batching
    // arc).
    resolveScratch_.clear();
    for (const VdpmWorkRequest& req : requests)
    {
        VdpmGpuFront* front = resolveFront(req.front);
        if (front == nullptr)
        {
            continue; // stale/invalid handle — skip
        }
        // Score view (cone predicate + screen-space error scale). makeVdpmViewParams is the SAME
        // pure helper the CPU front calls, so scoring inputs match bit-for-bit.
        const VdpmViewParams view = makeVdpmViewParams(
            req.world, globals.cameraPos, globals.projScaleY, globals.viewportHeight,
            kVdpmSilhouetteBoost, req.rasterBackfaceCulling, req.uvScale, req.normalScale,
            req.tangentScale);
        // Repair params (full jitter-free viewProj for the coverage projection). Packed field-by-
        // field into the std430 struct — z of viewport carries the raster cull policy (see
        // ubo.hpp).
        VdpmRepairParams repair{};
        repair.world = req.world;
        repair.viewProj = globals.viewProj;
        repair.cameraPos[0] = globals.cameraPos.x();
        repair.cameraPos[1] = globals.cameraPos.y();
        repair.cameraPos[2] = globals.cameraPos.z();
        repair.viewport[0] = globals.viewportWidth;
        repair.viewport[1] = globals.viewportHeight;
        repair.viewport[2] = req.rasterBackfaceCulling ? 1.0f : 0.0f;
        // Compaction predicates: score/apply run for a split-bearing front (rankCount>0 ⇔
        // splitCount>0); repair additionally needs post-weld faces (finestFaceCount>0).
        const bool hasApply = front->rankCount() > 0;
        const bool hasRepair = hasApply && front->finestFaceCount() > 0;
        // Draw multiplier for the emitted-triangle weighting, matched BY HANDLE (validated above as
        // a complete 1:1 cover when non-empty, so the match is guaranteed). Empty ⇒ 1.
        std::uint32_t drawMultiplier = 1;
        for (const VdpmFrontDrawCount& dc : drawCounts)
        {
            if (dc.front == req.front)
            {
                drawMultiplier = dc.drawCount;
                break;
            }
        }
        resolveScratch_.push_back({.front = front,
                                   .view = view,
                                   .repair = repair,
                                   .hasApply = hasApply,
                                   .hasRepair = hasRepair,
                                   .drawMultiplier = drawMultiplier});
    }

    // Per-stage timing (apply-kernel checkpoint), off unless the renderer supplied a profiler. GPU
    // timestamps use one-shot query slots, so they're only meaningful when a SINGLE front records
    // this frame (N>1 would double-write them). CPU per-stage timing accumulates for any count.
    const VdpmStageProfile stageProfile{.gpu = (resolveScratch_.size() == 1) ? globals.stageProfiler
                                                                             : nullptr,
                                        .gpuFrameIndex = globals.frameIndex,
                                        .cpuMs = &lastComputeStats_.stageCpuMs};
    const VdpmStageProfile* const stageProfilePtr =
        globals.stageProfiler != nullptr ? &stageProfile : nullptr;

    // Route: BATCH when both kernels are available and this is not a profiled SINGLE front (a
    // single profiled front keeps the per-front path for recordFrame's per-stage GPU timestamps —
    // which the dispatch(N) topology can't stamp). Otherwise (missing kernel, empty, or single
    // profiled front) the per-front reference path.
    const bool canBatch = applyKernel_.has_value() && repairKernel_.has_value();
    const bool singleFrontProfiled =
        resolveScratch_.size() == 1 && globals.stageProfiler != nullptr;
    if (canBatch && !resolveScratch_.empty() && !singleFrontProfiled)
    {
        recordBatched(cmd, globals, coarsenBudget);
    }
    else
    {
        recordPerFront(cmd, globals, coarsenBudget, stageProfilePtr);
    }

    // Scene-wide health reduction (B5c-1) — recorded AFTER all emits but still inside the
    // VdpmCompute timestamp (the renderer brackets recordRequests). Runs on BOTH record paths
    // (every recorded front carries the health buffers).
    if (!resolveScratch_.empty())
    {
        ensureHealthCapacity(static_cast<std::uint32_t>(resolveScratch_.size()));
        recordHealthReduction(cmd, globals.frameIndex);
        // Keep the analytic totals exact: the reduction adds exactly one emit→health barrier + one
        // dispatch on top of whichever record path filled the per-lifecycle totals above.
        lastComputeStats_.analyticDispatches += 1;
        lastComputeStats_.analyticBarriers += 1;
    }

    // Debug trace only when the front count changes (an activation / count-change proof, not a
    // per-frame flood). Reports the analytic command cost so the dispatch-bound diagnosis is
    // visible headless (perf arc); real per-round GPU diagnostics land later.
    if (requests.size() != lastLoggedRequestCount_)
    {
        log::debug(log::category::render,
                   "VDPM GPU: {} front(s), {} ranks, repairBudget {} → ~{} dispatches + ~{} "
                   "barriers/frame (analytic)",
                   lastComputeStats_.frontsRecorded, lastComputeStats_.maxRankCount,
                   lastComputeStats_.roundBudget, lastComputeStats_.analyticDispatches,
                   lastComputeStats_.analyticBarriers);
        lastLoggedRequestCount_ = requests.size();
    }
}

void VdpmGpuManager::recordPerFront(vk::CommandBuffer cmd, const VdpmFrameGlobals& globals,
                                    float coarsenBudget, const VdpmStageProfile* stageProfilePtr)
{
    const VdpmApplyKernel* const applyKernel = applyKernel_ ? &*applyKernel_ : nullptr;
    const VdpmRepairKernel* const kernel = repairKernel_ ? &*repairKernel_ : nullptr;
    for (ResolvedRequest& r : resolveScratch_)
    {
        r.front->recordFrame(cmd, scorePipeline_, refinePipelines_, repairPipelines_,
                             emitPipelines_, resources_, globals.frameIndex, r.view, r.repair,
                             globals.pixelBudget, coarsenBudget, kVdpmGpuRepairRoundBudget,
                             applyKernel, kernel, stageProfilePtr);

        const std::uint32_t rank = r.front->rankCount();
        const VdpmGpuFront::ComputeCost cost = VdpmGpuFront::analyticComputeCost(
            rank, r.front->faceCount(), r.front->finestFaceCount(), kVdpmGpuRepairRoundBudget,
            /*persistentApply=*/applyKernel != nullptr, /*persistentRepair=*/kernel != nullptr);
        ++lastComputeStats_.frontsRecorded;
        lastComputeStats_.maxRankCount = std::max(lastComputeStats_.maxRankCount, rank);
        lastComputeStats_.analyticDispatches += cost.dispatches;
        lastComputeStats_.analyticBarriers += cost.barriers;
    }
}

void VdpmGpuManager::recordBatched(vk::CommandBuffer cmd, const VdpmFrameGlobals& globals,
                                   float coarsenBudget)
{
    const std::uint32_t fi = globals.frameIndex;
    // Grow the job arrays to hold every resolved front BEFORE any dispatch references them — only
    // the batched path needs them (the per-front / profiled-single-front routes never touch them).
    ensureJobCapacity(static_cast<std::uint32_t>(resolveScratch_.size()));
    // The 1-D group cap for chunking a batched dispatch — the device's, or a TEST override so the
    // chunked advanced-BDA path is exercised (the real cap is far above any live front count). The
    // override is CLAMPED to the physical cap so an accidental oversized test value can never
    // record a dispatch beyond the device limit.
    const std::uint32_t deviceCap = resources_.maxComputeWorkGroupCountX();
    const std::uint32_t groupCap =
        testGroupCapOverride_ != 0 ? std::min(testGroupCapOverride_, deviceCap) : deviceCap;

    // Optional per-stage CPU record timing (perf arc); a no-op unless the renderer supplied a
    // profiler. Times each stage's recording into the SAME [score, apply, repair, emit] slots
    // recordFrame's stageProfile writes, so the two paths report comparably.
    const bool timeCpu = globals.stageProfiler != nullptr;
    auto cpuStage = [&](std::size_t idx, auto&& fn)
    {
        if (!timeCpu)
        {
            fn();
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        lastComputeStats_.stageCpuMs[idx] +=
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    // Dispatch a compacted N-job array in <= groupCap chunks: advance the BDA by firstJob * stride
    // (each chunk's jobs are DISJOINT, so NO barrier between chunks). The base is buffer-aligned
    // and the stride is an 8-multiple (asserted above), so every chunk base keeps
    // buffer_reference_align.
    auto dispatchChunks = [&](const auto& kernel, const MappedBufferSet& array, std::uint32_t count,
                              std::size_t stride)
    {
        const std::uint64_t base = resources_.bufferAddress(array.buffers[fi]);
        const std::uint32_t cap = groupCap == 0 ? 1 : groupCap;
        for (std::uint32_t first = 0; first < count;)
        {
            const std::uint32_t chunk = std::min(cap, count - first);
            kernel.recordDispatch(cmd, base + static_cast<std::uint64_t>(first) * stride, chunk);
            first += chunk;
        }
    };

    // (0) ONE lifecycle-boundary barrier — global, so it covers EVERY front's cross-frame WAR on
    // its single-buffered score output + emit scratch (the same boundary recordFrame opens with).
    VdpmGpuFront::recordLifecycleBoundary(cmd);

    // (1) SCORE every front (independent score outputs — no inter-front barrier). A zero-split
    // front records nothing (recordScore early-outs).
    cpuStage(0,
             [&]
             {
                 for (ResolvedRequest& r : resolveScratch_)
                 {
                     r.front->recordScore(cmd, scorePipeline_, fi, r.view);
                 }
             });

    // (2) APPLY batch: compact the split-bearing fronts' jobs, one dispatch(Na) (chunked).
    std::uint32_t na = 0;
    cpuStage(1,
             [&]
             {
                 for (ResolvedRequest& r : resolveScratch_)
                 {
                     if (!r.hasApply)
                     {
                         continue;
                     }
                     const VdpmApplyJobGpu job =
                         r.front->prepareApplyJob(fi, globals.pixelBudget, coarsenBudget);
                     writeMapped(applyJobArray_.mapped[fi].subspan(na * sizeof(VdpmApplyJobGpu)),
                                 job);
                     ++na;
                 }
                 if (na > 0)
                 {
                     VdpmGpuFront::recordComputeStageBoundary(cmd); // score → apply
                     dispatchChunks(*applyKernel_, applyJobArray_, na, sizeof(VdpmApplyJobGpu));
                 }
             });

    // (3) REPAIR batch: compact the repair-bearing fronts' jobs, one dispatch(Nr) (chunked).
    std::uint32_t nr = 0;
    cpuStage(2,
             [&]
             {
                 for (ResolvedRequest& r : resolveScratch_)
                 {
                     if (!r.hasRepair)
                     {
                         continue;
                     }
                     const VdpmRepairJobGpu job =
                         r.front->prepareRepairJob(fi, r.repair, kVdpmGpuRepairRoundBudget);
                     writeMapped(repairJobArray_.mapped[fi].subspan(nr * sizeof(VdpmRepairJobGpu)),
                                 job);
                     ++nr;
                 }
                 if (nr > 0)
                 {
                     VdpmGpuFront::recordComputeStageBoundary(cmd); // apply → repair
                     dispatchChunks(*repairKernel_, repairJobArray_, nr, sizeof(VdpmRepairJobGpu));
                 }
             });

    // (4) ONE final front-state→emit barrier: repair→emit if any repair ran, else apply→emit if any
    // apply wrote `active` (adj 5). Global, so it also covers apply-only fronts omitted from the
    // repair batch. None when nothing wrote front state (all zero-split) — the lifecycle barrier
    // suffices, matching recordFrame.
    if (na > 0 || nr > 0)
    {
        VdpmGpuFront::recordComputeStageBoundary(cmd);
    }

    // (5) EMIT every front into its frame-slot ring output (each front's emit scratch/output is its
    // own — no inter-front barrier). No consumer barrier — the caller adds the draw barrier.
    cpuStage(3,
             [&]
             {
                 for (ResolvedRequest& r : resolveScratch_)
                 {
                     r.front->recordEmitFromFront(cmd, emitPipelines_, resources_, fi);
                 }
             });

    // Accounting from the BATCHED aggregate (compacted counts + chunks) — NOT the per-front sum.
    dimsScratch_.clear();
    std::uint32_t maxRank = 0;
    for (const ResolvedRequest& r : resolveScratch_)
    {
        const std::uint32_t rank = r.front->rankCount();
        dimsScratch_.push_back({.rankCount = rank,
                                .faceCount = r.front->faceCount(),
                                .finestFaceCount = r.front->finestFaceCount()});
        maxRank = std::max(maxRank, rank);
    }
    const VdpmGpuFront::ComputeCost cost =
        VdpmGpuFront::analyticBatchedCost(dimsScratch_, groupCap);
    lastComputeStats_.frontsRecorded = static_cast<std::uint32_t>(resolveScratch_.size());
    lastComputeStats_.maxRankCount = maxRank;
    lastComputeStats_.analyticDispatches = cost.dispatches;
    lastComputeStats_.analyticBarriers = cost.barriers;
    lastComputeStats_.applyJobs = na;  // Na — makes the dispatch composition observable
    lastComputeStats_.repairJobs = nr; // Nr
}

void VdpmGpuManager::recordDiagnosticReadback(vk::CommandBuffer cmd, std::uint32_t frameIndex,
                                              std::uint64_t cpuTriangleSubtotal)
{
    // (1) Parse the host slot written a FULL frames-in-flight cycle ago (this frame's fence
    // guarantees it is complete — no stall), combined with THAT frame's stored CPU triangle
    // subtotal so the total is internally from one frame.
    const std::span<std::byte> slot = diagReadback_.mapped[frameIndex];
    // A slot invalidated across an inactive gap (or not yet warm) must not leave lastSceneHealth_
    // showing the previous, now-stale value — reset it to invalid until the ring re-warms.
    lastSceneHealth_ = SceneHealth{};
    if (diagSlotWritten_[frameIndex] && slot.size() >= sizeof(VdpmSceneHealthGpu))
    {
        VdpmSceneHealthGpu h{};
        std::memcpy(&h, slot.data(), sizeof(h));
        const std::uint64_t gpuEmitted =
            (static_cast<std::uint64_t>(h.emittedIndexTotalHi) << 32) | h.emittedIndexTotalLo;
        const std::uint64_t cpuSub = cpuTriangleSubtotal_[frameIndex];
        const std::uint64_t combined = cpuSub + gpuEmitted / 3;
        // 64-bit wrap guard: the GPU total already overflowed, OR the CPU+GPU sum itself wrapped.
        const bool overflow = h.emittedIndexOverflow != 0 || combined < cpuSub;
        lastSceneHealth_ = SceneHealth{.valid = true,
                                       .triangleTotal = combined,
                                       .emittedOverflow = overflow,
                                       .repairFronts = h.repairFronts,
                                       .maxMarkedRounds = h.maxMarkedRounds,
                                       .sumMarkedRounds = h.sumMarkedRounds,
                                       .roundBudget = kVdpmGpuRepairRoundBudget,
                                       .fallbackFronts = h.fallbackFronts,
                                       .nonCleanPrefix = h.nonCleanPrefixFronts,
                                       .ancestorFailures = h.ancestorFailureFronts,
                                       .failFlagFronts = h.failFlagFronts};

        // Log on any HEALTH-STATE transition (not just a front-count change) — a fallback/failure
        // appearing while repairFronts holds steady must still surface headless.
        const std::array<std::uint32_t, 5> sig{h.repairFronts, h.fallbackFronts,
                                               h.nonCleanPrefixFronts, h.ancestorFailureFronts,
                                               h.failFlagFronts};
        if (sig != lastLoggedHealth_)
        {
            log::debug(log::category::render,
                       "VDPM GPU health: {} repair front(s), max {}/{} marked rounds, {} fallback, "
                       "{} non-clean, {} ancestor-fail, {} B3-fail",
                       h.repairFronts, h.maxMarkedRounds, kVdpmGpuRepairRoundBudget,
                       h.fallbackFronts, h.nonCleanPrefixFronts, h.ancestorFailureFronts,
                       h.failFlagFronts);
            lastLoggedHealth_ = sig;
        }
    }

    // (2) Copy THIS frame's reduced scene-health (written by recordHealthReduction inside the
    // VdpmCompute timestamp) into the slot just parsed. No-op if no front recorded this frame.
    if (resolveScratch_.empty())
    {
        return;
    }
    const vk::Buffer dst = resources_.vulkanBuffer(diagReadback_.buffers[frameIndex]);
    const vk::MemoryBarrier2 computeToCopy{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &computeToCopy});
    cmd.copyBuffer(resources_.vulkanBuffer(sceneHealthDeviceRing_[frameIndex]), dst,
                   vk::BufferCopy{.size = sizeof(VdpmSceneHealthGpu)});
    // Explicit transfer-write → host-read memory dependency: the later fence provides completion,
    // but recording the dependency is correct + hardens the delayed readback.
    const vk::MemoryBarrier2 copyToHost{
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &copyToHost});

    // (3) Store this frame's CPU subtotal for the next ring cycle (parsed a cycle from now
    // alongside the scene-health just copied).
    cpuTriangleSubtotal_[frameIndex] = cpuTriangleSubtotal;
    diagSlotWritten_[frameIndex] = true;
}

VdpmGpuManager::DrawBuffers VdpmGpuManager::resolveDrawBuffers(VdpmFrontHandle front,
                                                               std::uint32_t frameIndex) const
{
    const VdpmGpuFront* f = resolveFront(front);
    if (f == nullptr)
    {
        throw std::logic_error(
            "VdpmGpuManager::resolveDrawBuffers: front handle did not resolve to a live front (a "
            "GPU-backed draw was tagged with a stale/invalid handle)");
    }
    const DrawBuffers out{.index = f->emittedIndicesBuffer(frameIndex),
                          .indirect = f->emittedIndirectBuffer(frameIndex)};
    if (out.index == NullBuffer || out.indirect == NullBuffer)
    {
        throw std::logic_error(
            "VdpmGpuManager::resolveDrawBuffers: a live front has a null draw buffer (its runtime "
            "ring was not allocated)");
    }
    return out;
}

BufferHandle VdpmGpuManager::frontIndexBuffer(VdpmFrontHandle front, std::uint32_t frameIndex) const
{
    const VdpmGpuFront* f = resolveFront(front);
    return f != nullptr ? f->emittedIndicesBuffer(frameIndex) : NullBuffer;
}

BufferHandle VdpmGpuManager::frontIndirectBuffer(VdpmFrontHandle front,
                                                 std::uint32_t frameIndex) const
{
    const VdpmGpuFront* f = resolveFront(front);
    return f != nullptr ? f->emittedIndirectBuffer(frameIndex) : NullBuffer;
}

} // namespace fire_engine
