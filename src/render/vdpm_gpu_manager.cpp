#include <fire_engine/render/vdpm_gpu_manager.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>

#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{
// Diagnostic readback layout (uint32 words): roundHistory[0..B) | repairControl[0..4) |
// counters[0..3).
constexpr std::uint32_t kVdpmDiagRoundBase = 0;
constexpr std::uint32_t kVdpmDiagControlBase = kVdpmGpuRepairRoundBudget;
constexpr std::uint32_t kVdpmDiagCountersBase = kVdpmGpuRepairRoundBudget + 4;
constexpr std::uint32_t kVdpmDiagWordCount = kVdpmGpuRepairRoundBudget + 7;
} // namespace

VdpmGpuManager::VdpmGpuManager(const Device& device, Resources& resources)
    : resources_(resources),
      scorePipeline_(device, vdpmScorePipelineConfig()),
      refinePipelines_(device),
      repairPipelines_(device),
      emitPipelines_(device)
{
    // Delayed diagnostics readback ring (perf arc, item 4): [roundHistory[0..B),
    // repairControl[0..4), counters[0..3)] per frame slot.
    diagReadback_ =
        resources_.createMappedReadbackBuffers(kVdpmDiagWordCount * sizeof(std::uint32_t));
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

void VdpmGpuManager::recordRequests(vk::CommandBuffer cmd,
                                    std::span<const VdpmWorkRequest> requests,
                                    const VdpmFrameGlobals& globals)
{
    // The coarsen budget is the refine budget scaled by the persistent-front hysteresis ratio — the
    // same relationship the CPU refineForView uses internally.
    const float coarsenBudget = kVdpmCoarsenRatio * globals.pixelBudget;

    // Perf instrumentation (no behaviour change): tally the analytic compute-command cost as we
    // record. Reset here; accumulated per resolved front below.
    lastComputeStats_ = ComputeStats{.roundBudget = kVdpmGpuRepairRoundBudget};
    diagFront_ = nullptr; // set to the first resolved front for the delayed convergence readback

    for (const VdpmWorkRequest& req : requests)
    {
        VdpmGpuFront* front = resolveFront(req.front);
        if (front == nullptr)
        {
            continue; // stale/invalid handle — skip
        }
        if (diagFront_ == nullptr)
        {
            diagFront_ = front; // representative front for the delayed convergence readback
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

        front->recordFrame(cmd, scorePipeline_, refinePipelines_, repairPipelines_, emitPipelines_,
                           resources_, globals.frameIndex, view, repair, globals.pixelBudget,
                           coarsenBudget, kVdpmGpuRepairRoundBudget);

        const std::uint32_t rank = front->rankCount();
        const VdpmGpuFront::ComputeCost cost =
            VdpmGpuFront::analyticComputeCost(rank, kVdpmGpuRepairRoundBudget);
        ++lastComputeStats_.frontsRecorded;
        lastComputeStats_.maxRankCount = std::max(lastComputeStats_.maxRankCount, rank);
        lastComputeStats_.analyticDispatches += cost.dispatches;
        lastComputeStats_.analyticBarriers += cost.barriers;
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

void VdpmGpuManager::recordDiagnosticReadback(vk::CommandBuffer cmd, std::uint32_t frameIndex)
{
    // Parse the slot written a FULL frames-in-flight cycle ago (this frame's fence guarantees it is
    // complete — no stall). Only once it has real data.
    const std::span<std::byte> slot = diagReadback_.mapped[frameIndex];
    if (diagSlotWritten_[frameIndex] && slot.size() >= kVdpmDiagWordCount * sizeof(std::uint32_t))
    {
        std::array<std::uint32_t, kVdpmDiagWordCount> w{};
        std::memcpy(w.data(), slot.data(), sizeof(w));

        std::uint32_t marked = 0;
        bool seenZero = false;
        bool cleanPrefix = true;
        for (std::uint32_t r = 0; r < kVdpmGpuRepairRoundBudget; ++r)
        {
            if (w[kVdpmDiagRoundBase + r] != 0)
            {
                ++marked;
                if (seenZero)
                {
                    cleanPrefix = false; // a marked round after a clean one — repair/sync anomaly
                }
            }
            else
            {
                seenZero = true;
            }
        }
        lastDiagnostics_ = Diagnostics{.valid = true,
                                       .roundBudget = kVdpmGpuRepairRoundBudget,
                                       .markedRounds = marked,
                                       .cleanPrefix = cleanPrefix,
                                       .finalAnyMarked = w[kVdpmDiagControlBase + 0],
                                       .fallbackFired = w[kVdpmDiagControlBase + 2],
                                       .emittedIndexCount = w[kVdpmDiagCountersBase + 2]};

        if (marked != lastLoggedMarkedRounds_)
        {
            log::debug(log::category::render,
                       "VDPM GPU repair: converged after {}/{} rounds (cleanPrefix {}, "
                       "finalAnyMarked {}, fallbackFired {}, emitted {})",
                       marked, kVdpmGpuRepairRoundBudget, cleanPrefix,
                       lastDiagnostics_.finalAnyMarked, lastDiagnostics_.fallbackFired,
                       lastDiagnostics_.emittedIndexCount);
            lastLoggedMarkedRounds_ = marked;
        }
    }

    // Record this frame's copy into the same slot (for a later frame to read). No-op if no front
    // recorded or the representative front has no repair buffers.
    if (diagFront_ == nullptr || diagFront_->roundHistoryBuffer() == NullBuffer)
    {
        return;
    }
    const vk::Buffer dst = resources_.vulkanBuffer(diagReadback_.buffers[frameIndex]);
    // Order the compute writes (roundHistory / repairControl / counters) before the transfer reads.
    const vk::MemoryBarrier2 computeToCopy{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &computeToCopy});
    cmd.copyBuffer(resources_.vulkanBuffer(diagFront_->roundHistoryBuffer()), dst,
                   vk::BufferCopy{.srcOffset = 0,
                                  .dstOffset = kVdpmDiagRoundBase * sizeof(std::uint32_t),
                                  .size = kVdpmGpuRepairRoundBudget * sizeof(std::uint32_t)});
    cmd.copyBuffer(resources_.vulkanBuffer(diagFront_->repairControlBuffer()), dst,
                   vk::BufferCopy{.srcOffset = 0,
                                  .dstOffset = kVdpmDiagControlBase * sizeof(std::uint32_t),
                                  .size = 4 * sizeof(std::uint32_t)});
    cmd.copyBuffer(resources_.vulkanBuffer(diagFront_->countersBuffer(frameIndex)), dst,
                   vk::BufferCopy{.srcOffset = 0,
                                  .dstOffset = kVdpmDiagCountersBase * sizeof(std::uint32_t),
                                  .size = 3 * sizeof(std::uint32_t)});
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
