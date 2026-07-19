#include <fire_engine/render/vdpm_gpu_manager.hpp>

#include <utility>

#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

VdpmGpuManager::VdpmGpuManager(const Device& device, Resources& resources)
    : resources_(resources),
      scorePipeline_(device, vdpmScorePipelineConfig()),
      refinePipelines_(device),
      repairPipelines_(device),
      emitPipelines_(device)
{
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
    if (!meshPool_.valid(mi, handleGeneration(mesh)) || mi >= meshes_.size() || !meshes_[mi])
    {
        return NullVdpmFront;
    }

    VdpmGpuFront front = VdpmGpuFront::buildRuntime(resources_, *meshes_[mi]);
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
    if (i >= fronts_.size() || !frontPool_.valid(i, handleGeneration(front)) || !fronts_[i].front)
    {
        return nullptr;
    }
    return &*fronts_[i].front;
}

const VdpmGpuFront* VdpmGpuManager::resolveFront(VdpmFrontHandle front) const noexcept
{
    if (front == NullVdpmFront)
    {
        return nullptr;
    }
    const std::uint32_t i = handleIndex(front);
    if (i >= fronts_.size() || !frontPool_.valid(i, handleGeneration(front)) || !fronts_[i].front)
    {
        return nullptr;
    }
    return &*fronts_[i].front;
}

void VdpmGpuManager::recordRequests(vk::CommandBuffer cmd,
                                    std::span<const VdpmWorkRequest> requests,
                                    const VdpmFrameGlobals& globals)
{
    // The coarsen budget is the refine budget scaled by the persistent-front hysteresis ratio — the
    // same relationship the CPU refineForView uses internally.
    const float coarsenBudget = kVdpmCoarsenRatio * globals.pixelBudget;

    // Debug trace only when the front count changes (an activation / count-change proof, not a
    // per-frame flood). Placeholder until B5c's real diagnostics land.
    if (requests.size() != lastLoggedRequestCount_)
    {
        log::debug(log::category::render, "VDPM GPU: now recording {} front(s) per frame",
                   requests.size());
        lastLoggedRequestCount_ = requests.size();
    }

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

        front->recordFrame(cmd, scorePipeline_, refinePipelines_, repairPipelines_, emitPipelines_,
                           resources_, globals.frameIndex, view, repair, globals.pixelBudget,
                           coarsenBudget, kVdpmGpuRepairRoundBudget);
    }
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
