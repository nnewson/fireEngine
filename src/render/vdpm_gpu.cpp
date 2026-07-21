#include <fire_engine/render/vdpm_gpu.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/mapped_buffer.hpp>
#include <fire_engine/graphics/mesh_topology.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/graphics/vdpm_wedge_choices.hpp>
#include <fire_engine/render/gpu_profiler.hpp>

namespace fire_engine
{

namespace
{

// A buffer_reference block promises its base address is aligned to `buffer_reference_align`; that
// is the APP's promise, not something Vulkan derives from the C++ struct. Verify it (VMA
// device-local allocations are 16-aligned in practice, so this only ever fires on a
// driver/allocator surprise).
void requireAligned(std::uint64_t address, std::uint64_t alignment, const char* what)
{
    if ((address % alignment) != 0)
    {
        throw std::runtime_error(std::string("VDPM GPU buffer address for ") + what +
                                 " is not aligned to the buffer_reference promise");
    }
}

// Global compute-write → compute-read/write barrier between emit passes. A global memory barrier
// (not per-buffer) because a pass touches several buffers reached only by device address here — the
// same idiom the scan primitive uses between its levels.
void emitComputeBarrier(vk::CommandBuffer cmd)
{
    const vk::MemoryBarrier2 mb{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
}

constexpr std::uint32_t kEmitLocalSize = 64;

// ceil(n / kEmitLocalSize) in 64-bit: `n + 63` wraps as uint32 near UINT32_MAX and would yield a
// zero / wrong dispatch count (same guard the scan primitive uses).
[[nodiscard]] std::uint32_t emitGroups(std::uint32_t n)
{
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(n) + kEmitLocalSize - 1) /
                                      kEmitLocalSize);
}

// Reject a 1-D dispatch whose group count exceeds the device cap rather than fault the driver.
void requireDispatchable(std::uint32_t groups, std::uint32_t maxGroupsX, const char* what)
{
    if (groups > maxGroupsX)
    {
        throw std::runtime_error(std::string("VDPM GPU dispatch for ") + what +
                                 " exceeds maxComputeWorkGroupCount[0]");
    }
}

// Record the SHARED bounded ancestor-resolution pass (`vdpm_ancestor.comp`): per canonical vertex,
// walk removalParent to its active ancestor + depth, atomic-incrementing `push.counters[0]` on a
// bad chain. Reused by B2 emit AND B4 repair (each round, against the live front). Records NO
// barrier — the caller orders the consumer read.
void recordAncestorResolve(vk::CommandBuffer cmd, const ComputePipeline& ancestorPipeline,
                           const VdpmAncestorPush& push)
{
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ancestorPipeline.pipeline());
    cmd.pushConstants<VdpmAncestorPush>(ancestorPipeline.pipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(emitGroups(push.vertexCount), 1, 1);
}

} // namespace

VdpmSplitGpu packVdpmSplit(const VertexSplit& split) noexcept
{
    VdpmSplitGpu g;
    g.coneAxisCos[0] = split.normalConeAxis.x();
    g.coneAxisCos[1] = split.normalConeAxis.y();
    g.coneAxisCos[2] = split.normalConeAxis.z();
    g.coneAxisCos[3] = split.normalConeCos;
    g.supportRadius = split.supportRadius;
    g.error = split.error;
    g.uvError = split.uvError;
    g.normalError = split.normalError;
    g.tangentError = split.tangentError;
    g.parentId = split.parent;
    g.childId = split.child;
    return g;
}

VdpmPositionGpu packVdpmPosition(const Vec3& position) noexcept
{
    VdpmPositionGpu g;
    g.position[0] = position.x();
    g.position[1] = position.y();
    g.position[2] = position.z();
    g.position[3] = 0.0f;
    return g;
}

VdpmScoreParams packVdpmScoreParams(const VdpmViewParams& view, std::uint64_t splitsAddress,
                                    std::uint64_t positionsAddress, std::uint64_t outputsAddress,
                                    std::uint32_t splitCount) noexcept
{
    VdpmScoreParams p;
    // GLSL mat3 is column-major, matching Mat3::fromColumns: column c = (worldLinear[0,c],
    // worldLinear[1,c], worldLinear[2,c]). The w of each padded column is unused.
    for (int c = 0; c < 3; ++c)
    {
        float* col = c == 0 ? p.worldLinearCol0 : (c == 1 ? p.worldLinearCol1 : p.worldLinearCol2);
        col[0] = view.worldLinear[0, c];
        col[1] = view.worldLinear[1, c];
        col[2] = view.worldLinear[2, c];
        col[3] = 0.0f;
    }
    p.worldTranslationMinusCamera[0] = view.worldTranslationMinusCamera.x();
    p.worldTranslationMinusCamera[1] = view.worldTranslationMinusCamera.y();
    p.worldTranslationMinusCamera[2] = view.worldTranslationMinusCamera.z();
    p.cameraObj[0] = view.cameraObj.x();
    p.cameraObj[1] = view.cameraObj.y();
    p.cameraObj[2] = view.cameraObj.z();
    p.worldLengthScale = view.worldLengthScale;
    p.facingSign = view.facingSign;
    p.projScaleY = view.projScaleY;
    p.halfViewport = view.halfViewport;
    p.silhouetteBoost = view.silhouetteBoost;
    p.uvScale = view.uvScale;
    p.normalScale = view.normalScale;
    p.tangentScale = view.tangentScale;
    p.coneUsable = view.coneUsable ? 1u : 0u;
    p.coneCullEnabled = view.coneCullEnabled ? 1u : 0u;
    p.splitCount = splitCount;
    p.splitsAddress = splitsAddress;
    p.positionsAddress = positionsAddress;
    p.outputsAddress = outputsAddress;
    return p;
}

ComputePipelineConfig vdpmScorePipelineConfig()
{
    // No descriptor bindings — every buffer reaches the shader as a device-address
    // (buffer_reference) carried in the params block, whose own address is the only push constant.
    ComputePipelineConfig config;
    config.compShaderPath = "vdpm_score.comp.spv";
    config.pushConstantRanges.emplace_back(
        vk::ShaderStageFlagBits::eCompute, 0,
        static_cast<std::uint32_t>(sizeof(VdpmScorePushConstants)));
    return config;
}

namespace
{
// Shared shape for the descriptor-free emit pipelines: one compute shader reached entirely by
// buffer_reference, one push-constant range carrying its ABI block.
template <typename Push>
[[nodiscard]] ComputePipelineConfig emitConfig(const char* spv)
{
    ComputePipelineConfig config;
    config.compShaderPath = spv;
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eCompute, 0,
                                           static_cast<std::uint32_t>(sizeof(Push)));
    return config;
}
} // namespace

ComputePipelineConfig vdpmAncestorPipelineConfig()
{
    return emitConfig<VdpmAncestorPush>("vdpm_ancestor.comp.spv");
}

ComputePipelineConfig vdpmSurvivalPipelineConfig()
{
    return emitConfig<VdpmSurvivalPush>("vdpm_survival.comp.spv");
}

ComputePipelineConfig vdpmScatterPipelineConfig()
{
    return emitConfig<VdpmScatterPush>("vdpm_scatter.comp.spv");
}

ComputePipelineConfig vdpmEmitFinalizePipelineConfig()
{
    return emitConfig<VdpmEmitFinalizePush>("vdpm_emit_finalize.comp.spv");
}

ComputePipelineConfig vdpmMarkPipelineConfig()
{
    return emitConfig<VdpmMarkPush>("vdpm_mark.comp.spv");
}

ComputePipelineConfig vdpmClosePipelineConfig()
{
    return emitConfig<VdpmClosePush>("vdpm_close.comp.spv");
}

ComputePipelineConfig vdpmRefinePipelineConfig()
{
    return emitConfig<VdpmRefinePush>("vdpm_refine.comp.spv");
}

ComputePipelineConfig vdpmCoarsenPipelineConfig()
{
    return emitConfig<VdpmCoarsenPush>("vdpm_coarsen.comp.spv");
}

ComputePipelineConfig vdpmRepairDetectPipelineConfig()
{
    return emitConfig<VdpmRepairDetectPush>("vdpm_repair_detect.comp.spv");
}

ComputePipelineConfig vdpmRepairFallbackPipelineConfig()
{
    return emitConfig<VdpmRepairFallbackPush>("vdpm_repair_fallback.comp.spv");
}

ComputePipelineConfig vdpmRepairKernelPipelineConfig()
{
    return emitConfig<VdpmRepairKernelPush>("vdpm_repair_kernel.comp.spv");
}

ComputePipelineConfig vdpmApplyKernelPipelineConfig(std::uint32_t workgroupSize)
{
    ComputePipelineConfig config = emitConfig<VdpmApplyKernelPush>("vdpm_apply_kernel.comp.spv");
    config.specConstants = {{.id = 0, .value = workgroupSize}}; // local_size_x
    return config;
}

bool VdpmRepairKernel::deviceSupported(const Device& device)
{
    const vk::PhysicalDeviceLimits& limits = device.physicalDevice().getProperties().limits;
    return limits.maxComputeWorkGroupInvocations >= kLocalSize &&
           limits.maxComputeWorkGroupSize[0] >= kLocalSize &&
           limits.maxComputeSharedMemorySize >= sizeof(std::uint32_t); // s_anyMarked
}

VdpmRepairKernel::Checked VdpmRepairKernel::requireSupported(const Device& device)
{
    if (!deviceSupported(device))
    {
        throw std::runtime_error(
            "VDPM repair kernel: device does not meet the workgroup/shared-memory limits (256)");
    }
    return {};
}

VdpmRepairKernel::VdpmRepairKernel(const Device& device)
    : VdpmRepairKernel(device, requireSupported(device))
{
}

VdpmRepairKernel::VdpmRepairKernel(const Device& device, Checked)
    : pipeline_(device, vdpmRepairKernelPipelineConfig())
{
}

void VdpmRepairKernel::recordDispatch(vk::CommandBuffer cmd, std::uint64_t jobsAddress,
                                      std::uint32_t jobCount) const
{
    const VdpmRepairKernelPush push{.jobsAddress = jobsAddress, .jobCount = jobCount, .pad = 0};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_.pipeline());
    cmd.pushConstants<VdpmRepairKernelPush>(pipeline_.pipelineLayout(),
                                            vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(jobCount, 1, 1); // one workgroup per job — each runs its front's whole fixpoint
}

bool VdpmApplyKernel::deviceSupported(const Device& device, std::uint32_t workgroupSize)
{
    if (workgroupSize == 0)
    {
        return false; // a zero local_size_x is an invalid pipeline, never "supported"
    }
    const vk::PhysicalDeviceLimits& limits = device.physicalDevice().getProperties().limits;
    // Apply uses NO shared memory (unlike repair's s_anyMarked) — just a full workgroup of the
    // SELECTED size (spec constant), validated against both the per-dimension and total-invocation
    // caps.
    return limits.maxComputeWorkGroupInvocations >= workgroupSize &&
           limits.maxComputeWorkGroupSize[0] >= workgroupSize;
}

VdpmApplyKernel::Checked VdpmApplyKernel::requireSupported(const Device& device,
                                                           std::uint32_t workgroupSize)
{
    if (!deviceSupported(device, workgroupSize))
    {
        throw std::runtime_error("VDPM apply kernel: device does not meet the workgroup limits for "
                                 "the selected size");
    }
    return {};
}

VdpmApplyKernel::VdpmApplyKernel(const Device& device, std::uint32_t workgroupSize)
    : VdpmApplyKernel(device, workgroupSize, requireSupported(device, workgroupSize))
{
}

VdpmApplyKernel::VdpmApplyKernel(const Device& device, std::uint32_t workgroupSize, Checked)
    : pipeline_(device, vdpmApplyKernelPipelineConfig(workgroupSize))
{
}

void VdpmApplyKernel::recordDispatch(vk::CommandBuffer cmd, std::uint64_t jobsAddress,
                                     std::uint32_t jobCount) const
{
    const VdpmApplyKernelPush push{.jobsAddress = jobsAddress, .jobCount = jobCount, .pad = 0};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_.pipeline());
    cmd.pushConstants<VdpmApplyKernelPush>(pipeline_.pipelineLayout(),
                                           vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(jobCount, 1, 1); // one workgroup per job — each runs its front's whole apply
}

VdpmGpuMesh VdpmGpuMesh::build(Resources& resources, std::span<const Vertex> vertices,
                               const VertexForest& forest)
{
    // Validation boundary: a shader must never see a malformed forest. validateForest covers the
    // 32-bit count limits + all structural references; the vertex-array size must match too (the
    // positions buffer is one entry per original vertex, indexed by canonical IDs in [0,
    // vertexCount)).
    validateForest(forest);
    if (vertices.size() != forest.vertexCount)
    {
        throw std::runtime_error("VdpmGpuMesh::build: vertex count != forest.vertexCount");
    }
    // Reject an over-large score dispatch BEFORE allocating (validateForest already bounded
    // splits.size() to 32-bit). The B5 selector queries fitsComputeDispatchLimits up front to pick
    // the CPU fallback instead; this throw is the enforcement boundary.
    if (emitGroups(static_cast<std::uint32_t>(forest.splits.size())) >
        resources.maxComputeWorkGroupCountX())
    {
        throw std::runtime_error(
            "VdpmGpuMesh::build: score dispatch exceeds maxComputeWorkGroupCount[0]");
    }
    // Build + validate the dependency DAG (throws on a cycle) BEFORE any upload — the "all throwing
    // derivation before the first upload" contract, so an invalid forest orphans no GPU resources.
    const DependencyDag dag = buildDependencyDag(forest);

    VdpmGpuMesh mesh;
    uploadScoreData(mesh, resources, vertices, forest, dag);
    return mesh;
}

void validateVdpmRankRanges(std::span<const RankRange> ranges, std::uint32_t splitCount)
{
    // A contiguous partition of [0, splitCount): rank 0 at offset 0, each rank starting where the
    // previous ended, the last ending exactly at splitCount. The running offset accumulates in
    // 64-bit so a wild `count` can't wrap around into a spuriously valid total.
    std::uint64_t expected = 0;
    for (const RankRange& rr : ranges)
    {
        if (rr.offset != expected)
        {
            throw std::runtime_error(
                "validateVdpmRankRanges: rank ranges are not a contiguous partition (gap/overlap)");
        }
        expected += rr.count;
    }
    if (expected != splitCount)
    {
        throw std::runtime_error(
            "validateVdpmRankRanges: rank ranges do not cover exactly splitCount splits");
    }
}

bool VdpmGpuMesh::fitsComputeDispatchLimits(const Resources& resources, const VertexForest& forest,
                                            std::size_t indexCount) noexcept
{
    const std::uint64_t maxX = resources.maxComputeWorkGroupCountX();
    auto groups64 = [](std::uint64_t n) { return (n + kEmitLocalSize - 1) / kEmitLocalSize; };
    // score (splitCount), ancestor (vertexCount), survival/scatter (faceCount = indexCount / 3).
    // The scan's largest dispatch is ceil(faceCount / kScanElementsPerBlock) <= ceil(faceCount /
    // 64), so the face check conservatively covers the scan too. All in 64-bit — no wrap on a huge
    // input.
    return groups64(forest.splits.size()) <= maxX && groups64(forest.vertexCount) <= maxX &&
           groups64(indexCount / 3) <= maxX;
}

void VdpmGpuMesh::uploadScoreData(VdpmGpuMesh& mesh, Resources& resources,
                                  std::span<const Vertex> vertices, const VertexForest& forest,
                                  const DependencyDag& dag)
{
    mesh.splitCount_ = static_cast<std::uint32_t>(forest.splits.size());

    // Per-rank dispatch ranges from the CSR rank offsets. DERIVED + VALIDATED up front — BEFORE any
    // GPU upload below — so a malformed partition throws without orphaning resource-table entries
    // (the "all throwing derivation before the first upload" contract). A zero-split forest has NO
    // ranks (not one zero-count range) — the documented empty representation.
    if (mesh.splitCount_ > 0)
    {
        mesh.rankRanges_.reserve(dag.maxRank + 1);
        for (std::uint32_t r = 0; r <= dag.maxRank; ++r)
        {
            mesh.rankRanges_.push_back(
                {dag.rankOffsets[r], dag.rankOffsets[r + 1] - dag.rankOffsets[r]});
        }
        validateVdpmRankRanges(mesh.rankRanges_, mesh.splitCount_);
    }

    // Per-split metric records (static, device-local, uploaded once).
    std::vector<VdpmSplitGpu> splits;
    splits.reserve(forest.splits.size());
    for (const VertexSplit& s : forest.splits)
    {
        splits.push_back(packVdpmSplit(s));
    }
    // Canonical positions: one entry per ORIGINAL vertex (canonical IDs index into this array).
    std::vector<VdpmPositionGpu> positions;
    positions.reserve(vertices.size());
    for (const Vertex& v : vertices)
    {
        positions.push_back(packVdpmPosition(v.position()));
    }

    // A zero-byte VMA allocation is invalid, so an empty forest / empty mesh leaves the buffers
    // null (splitCount_ == 0 ⇒ VdpmGpuFront records no dispatch).
    if (!splits.empty())
    {
        mesh.splits_ = resources.createDeviceLocalStorageBuffer(
            splits.size() * sizeof(VdpmSplitGpu), splits.data());
        mesh.splitsAddress_ = resources.bufferAddress(mesh.splits_);
        requireAligned(mesh.splitsAddress_, 16, "splits");
    }
    if (!positions.empty())
    {
        mesh.positions_ = resources.createDeviceLocalStorageBuffer(
            positions.size() * sizeof(VdpmPositionGpu), positions.data());
        mesh.positionsAddress_ = resources.bufferAddress(mesh.positions_);
        requireAligned(mesh.positionsAddress_, 16, "positions");
    }

    // Assemble the score portion of the binding (emit portion stays null; hasEmitData false).
    mesh.binding_.splitsAddress = mesh.splitsAddress_;
    mesh.binding_.positionsAddress = mesh.positionsAddress_;
    mesh.binding_.splitCount = mesh.splitCount_;
    mesh.binding_.vertexCount = forest.vertexCount;

    // Front topology (Stage B3): the caller-built dependency DAG gives the per-split mutation
    // record
    // + the rank-ordered dispatch layout. The DAG was built + validated (acyclicity) BEFORE any
    // upload, so a malformed forest never reaches here (the "validate before upload" contract).
    mesh.binding_.maxRank = dag.maxRank;

    // Coarsest front's initial active flags: roots (never removed) active, everything else
    // inactive.
    mesh.initialActive_.assign(forest.vertexCount, 0);
    for (std::uint32_t v = 0; v < forest.vertexCount; ++v)
    {
        if (forest.removingSplit[v] == kNoSplit)
        {
            mesh.initialActive_[v] = 1;
        }
    }

    // Per-split mutation record: vertex slots (for dependents / child activation) + the DAG's
    // dependency-split triple (for closure), 1:1 with DependencyDag::dependencies.
    std::vector<VdpmFrontSplitGpu> frontSplits(mesh.splitCount_);
    for (std::uint32_t s = 0; s < mesh.splitCount_; ++s)
    {
        const VertexSplit& sp = forest.splits[s];
        const SplitDependencies& d = dag.dependencies[s];
        frontSplits[s] = {sp.parent, sp.child, sp.vl, sp.vr, d.parent, d.vl, d.vr, 0};
    }

    if (!frontSplits.empty())
    {
        mesh.frontSplits_ = resources.createDeviceLocalStorageBuffer(
            frontSplits.size() * sizeof(VdpmFrontSplitGpu), frontSplits.data());
        mesh.binding_.frontSplitsAddress = resources.bufferAddress(mesh.frontSplits_);
        requireAligned(mesh.binding_.frontSplitsAddress, 4, "frontSplits");
    }
    if (!dag.splitsByRank.empty())
    {
        mesh.splitsByRank_ = resources.createDeviceLocalStorageBuffer(
            dag.splitsByRank.size() * sizeof(std::uint32_t), dag.splitsByRank.data());
        mesh.binding_.splitsByRankAddress = resources.bufferAddress(mesh.splitsByRank_);
        requireAligned(mesh.binding_.splitsByRankAddress, 4, "splitsByRank");
    }

    // Upload the per-rank ranges device-local (the persistent repair kernel walks them
    // in-workgroup; the CPU recorder keeps using rankRanges_ as push constants). Already derived +
    // VALIDATED at the top of this function, before any upload.
    if (!mesh.rankRanges_.empty())
    {
        mesh.rankRangesBuffer_ = resources.createDeviceLocalStorageBuffer(
            mesh.rankRanges_.size() * sizeof(RankRange), mesh.rankRanges_.data());
        mesh.binding_.rankRangesAddress = resources.bufferAddress(mesh.rankRangesBuffer_);
        requireAligned(mesh.binding_.rankRangesAddress, 4, "rankRanges");
    }
}

VdpmGpuMesh VdpmGpuMesh::build(Resources& resources, std::span<const Vertex> vertices,
                               std::span<const std::uint32_t> indices, const VertexForest& forest)
{
    // ALL validation + throwing CPU derivation runs BEFORE the first upload, so malformed input
    // leaves no orphaned resource-table entries.
    validateForest(forest);
    if (vertices.size() != forest.vertexCount)
    {
        throw std::runtime_error("VdpmGpuMesh::build: vertex count != forest.vertexCount");
    }
    // Index validation: a shader indexes weld/positions with these, so reject a malformed buffer
    // here rather than fault a GPU thread. Every corner references an original vertex in
    // [0, vertexCount); the finest index count must fit 32-bit GPU indexing.
    if ((indices.size() % 3) != 0)
    {
        throw std::runtime_error("VdpmGpuMesh::build: index count is not a multiple of 3");
    }
    if (indices.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::runtime_error("VdpmGpuMesh::build: index count exceeds 32-bit GPU indexing");
    }
    for (const std::uint32_t idx : indices)
    {
        if (idx >= forest.vertexCount)
        {
            throw std::runtime_error("VdpmGpuMesh::build: index references an out-of-range vertex");
        }
    }

    // Derive the emit acceleration structures (these can throw — a cycle / a 32-bit overflow). weld
    // is built here (not passed) so it can't disagree with the mesh; wedge choices + removal parent
    // are the precomputed GPU walk data (Vulkan-free). All BEFORE any upload.
    const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(vertices);
    const WedgeChoices wc = buildWedgeChoices(vertices, forest, weld);
    const std::vector<std::uint32_t> removalParent = buildRemovalParent(forest);
    const std::vector<std::uint32_t> indexVec{indices.begin(), indices.end()};
    const DependencyDag dag = buildDependencyDag(forest); // throws on a cycle — before any upload
    // Canonical finest faces (post-weld, degenerate faces dropped) — the B4 repair detector's
    // input; identical to ParallelFront::finestFaces() (same corner order + winding). Flattened for
    // upload.
    const std::vector<std::array<std::uint32_t, 3>> finestFaces =
        mesh_topology::canonicalFaces(weld, indices);
    std::vector<std::uint32_t> finestFacesFlat;
    finestFacesFlat.reserve(finestFaces.size() * 3);
    for (const std::array<std::uint32_t, 3>& f : finestFaces)
    {
        finestFacesFlat.insert(finestFacesFlat.end(), {f[0], f[1], f[2]});
    }

    // Reject an over-large static emit/score dispatch BEFORE allocating, so a mesh the device can't
    // dispatch never gets GPU buffers (the B5 selector queries this same predicate to pick the CPU
    // fallback up front; recordEmit re-checks as defence-in-depth).
    if (!fitsComputeDispatchLimits(resources, forest, indices.size()))
    {
        throw std::runtime_error(
            "VdpmGpuMesh::build: emit dispatch exceeds maxComputeWorkGroupCount[0]");
    }

    // Validation + derivation complete — now the uploads (nothing below throws for bad input).
    VdpmGpuMesh mesh;
    uploadScoreData(mesh, resources, vertices, forest, dag);

    auto uploadU32 = [&](const std::vector<std::uint32_t>& data, BufferHandle& handle,
                         std::uint64_t& address, const char* what)
    {
        if (data.empty())
        {
            return; // zero-byte VMA allocations are invalid; a null address is never dereferenced
        }
        handle = resources.createDeviceLocalStorageBuffer(data.size() * sizeof(std::uint32_t),
                                                          data.data());
        address = resources.bufferAddress(handle);
        requireAligned(address, 4, what);
    };

    std::uint64_t indicesAddress = 0;
    std::uint64_t weldAddress = 0;
    std::uint64_t removalParentAddress = 0;
    std::uint64_t choicesAddress = 0;
    std::uint64_t offsetsAddress = 0;
    std::uint64_t finestFacesAddress = 0;
    std::uint64_t removingSplitAddress = 0;
    uploadU32(indexVec, mesh.indices_, indicesAddress, "indices");
    uploadU32(weld, mesh.weld_, weldAddress, "weld");
    uploadU32(removalParent, mesh.removalParent_, removalParentAddress, "removalParent");
    uploadU32(wc.choices, mesh.wedgeChoices_, choicesAddress, "wedgeChoices");
    uploadU32(wc.offsets, mesh.wedgeOffsets_, offsetsAddress, "wedgeOffsets");
    uploadU32(finestFacesFlat, mesh.finestFaces_, finestFacesAddress, "finestFaces");
    uploadU32(forest.removingSplit, mesh.removingSplit_, removingSplitAddress, "removingSplit");

    mesh.binding_.indicesAddress = indicesAddress;
    mesh.binding_.weldAddress = weldAddress;
    mesh.binding_.removalParentAddress = removalParentAddress;
    mesh.binding_.wedgeChoicesAddress = choicesAddress;
    mesh.binding_.wedgeOffsetsAddress = offsetsAddress;
    mesh.binding_.finestFacesAddress = finestFacesAddress;
    mesh.binding_.removingSplitAddress = removingSplitAddress;
    mesh.binding_.finestFaceCount = static_cast<std::uint32_t>(finestFaces.size());
    mesh.binding_.faceCount = static_cast<std::uint32_t>(indices.size() / 3);
    mesh.binding_.maxDepth = wc.maxDepth;
    mesh.binding_.hasEmitData = true;
    return mesh;
}

VdpmGpuFront VdpmGpuFront::build(Resources& resources, const VdpmGpuMesh& mesh)
{
    VdpmGpuFront front;
    front.binding_ = mesh.binding(); // copy the immutable binding — no pointer into the mesh
    front.maxWorkGroupCountX_ = resources.maxComputeWorkGroupCountX();
    if (front.binding_.splitCount == 0)
    {
        return front; // nothing to score
    }

    // Device-local score output, readback-enabled (eTransferSrc) so the harness can copy it back.
    front.output_ = resources.createDeviceLocalStorageBuffer(
        mesh.splitCount() * sizeof(VdpmScoreOut), nullptr, vk::BufferUsageFlagBits::eTransferSrc);
    front.outputAddress_ = resources.bufferAddress(front.output_);
    requireAligned(front.outputAddress_, 4, "output");

    // Per-frame mapped params (host-visible + BDA). Coherent host writes become visible to the
    // device at queue submission — the same pattern as the dynamic index/UBO buffers.
    const MappedBufferSet params =
        resources.createMappedDeviceAddressBuffers(sizeof(VdpmScoreParams));
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        front.paramsMapped_[i] = params.mapped[i];
        front.paramsAddress_[i] = resources.bufferAddress(params.buffers[i]);
        requireAligned(front.paramsAddress_[i], 16, "params");
    }
    return front;
}

VdpmGpuFront VdpmGpuFront::buildWithEmit(Resources& resources, const VdpmGpuMesh& mesh)
{
    if (!mesh.binding().hasEmitData)
    {
        throw std::logic_error("VdpmGpuFront::buildWithEmit: mesh has no emit data (score-only)");
    }
    VdpmGpuFront front = build(resources, mesh); // score workspace + binding copy
    front.hasEmit_ = true;

    const VdpmGpuMeshBinding& b = front.binding_;
    const std::size_t vBytes = static_cast<std::size_t>(b.vertexCount) * sizeof(std::uint32_t);
    const std::size_t fBytes = static_cast<std::size_t>(b.faceCount) * sizeof(std::uint32_t);

    // Host-visible + BDA `active` (uploaded per emit). Slot 0 of the per-frame set —
    // single-buffered for B2; the GPU serialises on one queue and the harness fence-waits before
    // reuse.
    if (b.vertexCount > 0)
    {
        const MappedBufferSet active = resources.createMappedDeviceAddressBuffers(vBytes);
        front.activeMapped_ = active.mapped[0];
        front.activeAddress_ = resources.bufferAddress(active.buffers[0]);
        requireAligned(front.activeAddress_, 4, "active");
    }

    auto deviceLocal = [&](std::size_t bytes, BufferHandle& handle, std::uint64_t& address,
                           vk::BufferUsageFlags extraUsage, const char* what)
    {
        if (bytes == 0)
        {
            return; // never dispatched over (0 groups); the null address is never dereferenced
        }
        handle = resources.createDeviceLocalStorageBuffer(bytes, nullptr, extraUsage);
        address = resources.bufferAddress(handle);
        requireAligned(address, 4, what);
    };

    deviceLocal(vBytes, front.ancestorId_, front.ancestorIdAddress_, {}, "ancestorId");
    deviceLocal(vBytes, front.ancestorDepth_, front.ancestorDepthAddress_, {}, "ancestorDepth");
    deviceLocal(fBytes, front.survive_, front.surviveAddress_, {}, "survive");
    deviceLocal(fBytes, front.outSlot_, front.outSlotAddress_, {}, "outSlot");
    // Emitted indices: 3 corners per surviving face; sized for the worst case (all faces survive).
    // eIndexBuffer so B5 can bind it as the draw's index source; eTransferSrc for the harness
    // readback.
    deviceLocal(fBytes * 3, front.emittedIndices_, front.emittedIndicesAddress_,
                vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                "emittedIndices");
    // Counters [ancestorFailures, survivingFaces, emittedIndexCount] — always present + readable.
    deviceLocal(3 * sizeof(std::uint32_t), front.counters_, front.countersAddress_,
                vk::BufferUsageFlagBits::eTransferSrc, "counters");
    // The GPU-written draw indirect command — eIndirectBuffer (B5 draw) + eTransferSrc (readback).
    deviceLocal(sizeof(DrawIndexedIndirectCommand), front.emittedIndirect_,
                front.emittedIndirectAddress_,
                vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                "emittedIndirect");

    // Exclusive-scan per-level scratch (empty when faceCount fits one block).
    const std::vector<std::uint32_t> levels = VdpmScan::scratchElementCounts(b.faceCount);
    front.scanScratch_.resize(levels.size());
    front.scanScratchAddress_.resize(levels.size());
    for (std::size_t i = 0; i < levels.size(); ++i)
    {
        const std::size_t bytes = static_cast<std::size_t>(levels[i]) * sizeof(std::uint32_t);
        front.scanScratch_[i] = resources.createDeviceLocalStorageBuffer(bytes, nullptr);
        front.scanScratchAddress_[i] = resources.bufferAddress(front.scanScratch_[i]);
        requireAligned(front.scanScratchAddress_[i], 4, "scanScratch");
    }
    return front;
}

void VdpmGpuFront::recordEmit(vk::CommandBuffer cmd, const VdpmEmitPipelines& pipelines,
                              Resources& resources, std::span<const std::uint32_t> active)
{
    if (!hasEmit_)
    {
        throw std::logic_error(
            "VdpmGpuFront::recordEmit: front has no emit workspace (score-only)");
    }
    if (active.size() != binding_.vertexCount)
    {
        throw std::runtime_error("VdpmGpuFront::recordEmit: active size != vertexCount");
    }
    // Upload the settled front (host-visible coherent → visible at queue submit), then emit into
    // the single (non-ringed) B2 workspace output.
    if (binding_.vertexCount > 0)
    {
        writeMapped(activeMapped_, active.data(), active.size_bytes());
    }
    recordEmitImpl(cmd, pipelines, resources, activeAddress_,
                   FrameOutput{.emittedIndices = emittedIndices_,
                               .counters = counters_,
                               .indirect = emittedIndirect_,
                               .emittedIndicesAddress = emittedIndicesAddress_,
                               .countersAddress = countersAddress_,
                               .indirectAddress = emittedIndirectAddress_});
}

void VdpmGpuFront::recordEmitFromFront(vk::CommandBuffer cmd, const VdpmEmitPipelines& pipelines,
                                       Resources& resources, std::uint32_t frameIndex)
{
    if (!hasRuntime_)
    {
        throw std::logic_error("VdpmGpuFront::recordEmitFromFront: front is not a runtime front");
    }
    // Read the LIVE refine/coarsen state (no CPU upload); write the frame slot's ringed output.
    recordEmitImpl(cmd, pipelines, resources, activeStateAddress_, frameOutputs_[frameIndex]);
}

void VdpmGpuFront::recordEmitImpl(vk::CommandBuffer cmd, const VdpmEmitPipelines& pipelines,
                                  Resources& resources, std::uint64_t activeAddress,
                                  const FrameOutput& out)
{
    // The per-canonical (ancestor) + per-face (survival/scatter) dispatches must fit the device cap
    // — the scan validates its own levels internally.
    requireDispatchable(emitGroups(binding_.vertexCount), maxWorkGroupCountX_, "ancestor");
    requireDispatchable(emitGroups(binding_.faceCount), maxWorkGroupCountX_, "survival/scatter");

    // Clear the whole counters buffer ONCE: ancestor's atomic (counters[0]), the scan's total
    // (counters[1] — pre-zeroed so a zero-face scan needs no dispatch), and finalize's index count
    // (counters[2]) all start defined. fillBuffer is a CLEAR (eClear stage / eTransferWrite).
    cmd.fillBuffer(resources.vulkanBuffer(out.counters), 0, 3 * sizeof(std::uint32_t), 0);
    const vk::MemoryBarrier2 clearToCompute{
        .srcStageMask = vk::PipelineStageFlagBits2::eClear,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});

    // Pass 1 — ancestor resolution (per canonical vertex; the shared pass, failures → counters[0]).
    recordAncestorResolve(cmd, pipelines.ancestor(),
                          VdpmAncestorPush{.activeAddress = activeAddress,
                                           .removalParentAddress = binding_.removalParentAddress,
                                           .ancestorIdAddress = ancestorIdAddress_,
                                           .ancestorDepthAddress = ancestorDepthAddress_,
                                           .countersAddress = out.countersAddress,
                                           .vertexCount = binding_.vertexCount,
                                           .maxDepth = binding_.maxDepth});
    emitComputeBarrier(cmd);

    // Pass 2 — per-face survival flag.
    {
        const VdpmSurvivalPush push{.indicesAddress = binding_.indicesAddress,
                                    .weldAddress = binding_.weldAddress,
                                    .ancestorIdAddress = ancestorIdAddress_,
                                    .surviveAddress = surviveAddress_,
                                    .faceCount = binding_.faceCount};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.survival().pipeline());
        cmd.pushConstants<VdpmSurvivalPush>(pipelines.survival().pipelineLayout(),
                                            vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(emitGroups(binding_.faceCount), 1, 1);
    }
    emitComputeBarrier(cmd);

    // Pass 3 — exclusive scan of the survival flags → per-face output slot; grand total to
    // counters[1]. (No dispatch when faceCount == 0; counters[1] stays the pre-zeroed 0.)
    pipelines.scan().recordScan(cmd, surviveAddress_, outSlotAddress_, scanScratchAddress_,
                                out.countersAddress + sizeof(std::uint32_t), binding_.faceCount);
    // One barrier after the scan, before BOTH consumers (scatter reads outSlot; finalize reads
    // counters[1]).
    emitComputeBarrier(cmd);

    // Pass 4 — stable scatter of the surviving faces' restored-wedge corners.
    {
        const VdpmScatterPush push{.indicesAddress = binding_.indicesAddress,
                                   .weldAddress = binding_.weldAddress,
                                   .ancestorDepthAddress = ancestorDepthAddress_,
                                   .surviveAddress = surviveAddress_,
                                   .outSlotAddress = outSlotAddress_,
                                   .wedgeChoicesAddress = binding_.wedgeChoicesAddress,
                                   .wedgeOffsetsAddress = binding_.wedgeOffsetsAddress,
                                   .emittedIndicesAddress = out.emittedIndicesAddress,
                                   .faceCount = binding_.faceCount};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.scatter().pipeline());
        cmd.pushConstants<VdpmScatterPush>(pipelines.scatter().pipelineLayout(),
                                           vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(emitGroups(binding_.faceCount), 1, 1);
    }

    // Pass 5 — finalize: counters[2] = 3 * counters[1] + the full 5-word draw indirect command. One
    // invocation, always dispatched (defines the count + indirect even when every face collapses).
    {
        const VdpmEmitFinalizePush push{.countersAddress = out.countersAddress,
                                        .indirectAddress = out.indirectAddress};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.finalize().pipeline());
        cmd.pushConstants<VdpmEmitFinalizePush>(pipelines.finalize().pipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(1, 1, 1);
    }
    // No consumer barrier — the caller synchronises the emitted-index / counters reads.
}

VdpmGpuFront VdpmGpuFront::buildWithFront(Resources& resources, const VdpmGpuMesh& mesh,
                                          bool withClassificationReadback)
{
    VdpmGpuFront front;
    front.binding_ = mesh.binding();
    front.maxWorkGroupCountX_ = resources.maxComputeWorkGroupCountX();
    front.hasFront_ = true;
    front.rankRanges_.assign(mesh.rankRanges().begin(), mesh.rankRanges().end());

    const VdpmGpuMeshBinding& b = front.binding_;

    auto stateBuf = [&](std::size_t bytes, const void* init, BufferHandle& handle,
                        std::uint64_t& address, const char* what)
    {
        if (bytes == 0)
        {
            return;
        }
        handle = resources.createDeviceLocalStorageBuffer(bytes, init,
                                                          vk::BufferUsageFlagBits::eTransferSrc);
        address = resources.bufferAddress(handle);
        requireAligned(address, 4, what);
    };

    // The invariant-failure flags [refineFailure, underflow] are ALWAYS allocated (a fixed 2 uints,
    // independent of mesh size) — the diagnostic is meaningful even for a split-free/empty front,
    // and the harness reads it unconditionally.
    const std::array<std::uint32_t, 2> zeroFlags{0, 0};
    stateBuf(2 * sizeof(std::uint32_t), zeroFlags.data(), front.failFlags_, front.failFlagsAddress_,
             "failFlags");
    if (b.vertexCount == 0)
    {
        return front; // truly empty mesh: no per-vertex/per-split state, but failFlags exists
    }

    const std::size_t vBytes = static_cast<std::size_t>(b.vertexCount) * sizeof(std::uint32_t);
    const std::size_t sBytes = static_cast<std::size_t>(b.splitCount) * sizeof(std::uint32_t);
    const std::vector<std::uint32_t> zerosV(b.vertexCount, 0);
    const std::vector<std::uint32_t> zerosS(b.splitCount, 0);

    // Persistent state (readback-enabled): active = the coarsest front (roots), the rest zeroed.
    stateBuf(vBytes, mesh.initialActive().data(), front.activeState_, front.activeStateAddress_,
             "activeState");
    stateBuf(vBytes, zerosV.data(), front.dependentsState_, front.dependentsStateAddress_,
             "dependentsState");
    stateBuf(sBytes, zerosS.data(), front.refinedState_, front.refinedStateAddress_,
             "refinedState");
    stateBuf(sBytes, zerosS.data(), front.requiredState_, front.requiredStateAddress_,
             "requiredState");

    // Per-split score input (host-visible + BDA, uploaded per applyView). Single-buffered for B3.
    if (b.splitCount > 0)
    {
        const MappedBufferSet scores = resources.createMappedDeviceAddressBuffers(
            static_cast<std::size_t>(b.splitCount) * sizeof(VdpmScoreOut));
        front.scoresMapped_ = scores.mapped[0];
        front.scoresAddress_ = resources.bufferAddress(scores.buffers[0]);
        requireAligned(front.scoresAddress_, 4, "scores");
    }

    // Repair scratch (Stage B4) — only when the mesh carries the finest faces (a full-build mesh).
    // The ancestor cache is per canonical vertex; repairControl is a 4-uint diagnostic buffer
    // SEPARATE from failFlags; repairParams is host-visible (uploaded per repair).
    if (b.finestFacesAddress != 0 && b.vertexCount > 0)
    {
        front.hasRepair_ = true;
        // stateBuf gives eTransferSrc (harmless on the ancestor caches) + eTransferDst (the
        // fillBuffer clear of repairControl) via the device-local factory.
        stateBuf(vBytes, nullptr, front.repairAncestorId_, front.repairAncestorIdAddress_,
                 "repairAncestorId");
        stateBuf(vBytes, nullptr, front.repairAncestorDepth_, front.repairAncestorDepthAddress_,
                 "repairAncestorDepth");
        const std::array<std::uint32_t, 4> zeroCtrl{0, 0, 0, 0};
        stateBuf(4 * sizeof(std::uint32_t), zeroCtrl.data(), front.repairControl_,
                 front.repairControlAddress_, "repairControl");
        // Per-face classification readback (opt-in, test/debug only — a PRODUCTION front leaves
        // this null so it spends no per-face memory).
        if (withClassificationReadback)
        {
            stateBuf(static_cast<std::size_t>(b.finestFaceCount) * sizeof(std::uint32_t), nullptr,
                     front.repairClassification_, front.repairClassificationAddress_,
                     "repairClassification");
        }
        const MappedBufferSet params =
            resources.createMappedDeviceAddressBuffers(sizeof(VdpmRepairParams));
        front.repairParamsMapped_ = params.mapped[0];
        front.repairParamsAddress_ = resources.bufferAddress(params.buffers[0]);
        requireAligned(front.repairParamsAddress_, 16, "repairParams");
    }
    return front;
}

VdpmGpuFront VdpmGpuFront::buildRuntime(Resources& resources, const VdpmGpuMesh& mesh)
{
    const VdpmGpuMeshBinding& b = mesh.binding();
    // Requires a FULL mesh (emit data) with vertices. Does NOT require a non-null
    // finestFacesAddress: `finestFaceCount == 0` is a VALID runtime state — an empty canonical
    // repair-face array (all raw faces welded to degenerates). Repair recorders early-out at
    // finestFaceCount == 0; raw-face emission still runs (over `faceCount`), producing a canonical
    // zero-index indirect command; and recordFrame's conditional apply→emit barrier supplies the
    // sync. hasEmitData already means "full mesh", so gating on the empty repair-face array here
    // would be an accidental invariant.
    if (!b.hasEmitData || b.vertexCount == 0)
    {
        throw std::logic_error("VdpmGpuFront::buildRuntime: requires a full mesh (emit data) with "
                               "a non-empty vertex set");
    }

    VdpmGpuFront front = build(resources, mesh); // score output + params ring + binding + limit
    front.hasFront_ = true;
    front.hasEmit_ = true;
    front.hasRepair_ = true;
    front.hasRuntime_ = true;
    front.rankRanges_.assign(mesh.rankRanges().begin(), mesh.rankRanges().end());

    const std::size_t vBytes = static_cast<std::size_t>(b.vertexCount) * sizeof(std::uint32_t);
    const std::size_t sBytes = static_cast<std::size_t>(b.splitCount) * sizeof(std::uint32_t);
    const std::size_t fBytes = static_cast<std::size_t>(b.faceCount) * sizeof(std::uint32_t);
    const std::vector<std::uint32_t> zerosV(b.vertexCount, 0);
    const std::vector<std::uint32_t> zerosS(b.splitCount, 0);

    auto dev = [&](std::size_t bytes, const void* init, BufferHandle& h, std::uint64_t& addr,
                   vk::BufferUsageFlags extra, const char* what)
    {
        if (bytes == 0)
        {
            return; // zero-byte VMA allocations are invalid; a null address is never dereferenced
        }
        h = resources.createDeviceLocalStorageBuffer(bytes, init, extra);
        addr = resources.bufferAddress(h);
        requireAligned(addr, 4, what);
    };
    const auto srcFlag = vk::BufferUsageFlagBits::eTransferSrc;

    // Persistent front state (B3) — readback-enabled for diagnostics/harness.
    dev(vBytes, mesh.initialActive().data(), front.activeState_, front.activeStateAddress_, srcFlag,
        "activeState");
    dev(vBytes, zerosV.data(), front.dependentsState_, front.dependentsStateAddress_, srcFlag,
        "dependentsState");
    dev(sBytes, zerosS.data(), front.refinedState_, front.refinedStateAddress_, srcFlag,
        "refinedState");
    // requiredState is readback-enabled on the runtime front so the Stage-3 cross-check can compare
    // the two fronts' pre-repair `required` too (it's transient mark scratch, but proving it
    // matches strengthens the identical-pre-state gate).
    dev(sBytes, zerosS.data(), front.requiredState_, front.requiredStateAddress_, srcFlag,
        "requiredState");
    const std::array<std::uint32_t, 2> zeroFlags{0, 0};
    dev(2 * sizeof(std::uint32_t), zeroFlags.data(), front.failFlags_, front.failFlagsAddress_,
        srcFlag, "failFlags");

    // Emit workspace (B2) — internal, single (compute-only, serialised).
    dev(vBytes, nullptr, front.ancestorId_, front.ancestorIdAddress_, {}, "ancestorId");
    dev(vBytes, nullptr, front.ancestorDepth_, front.ancestorDepthAddress_, {}, "ancestorDepth");
    dev(fBytes, nullptr, front.survive_, front.surviveAddress_, {}, "survive");
    dev(fBytes, nullptr, front.outSlot_, front.outSlotAddress_, {}, "outSlot");
    const std::vector<std::uint32_t> levels = VdpmScan::scratchElementCounts(b.faceCount);
    front.scanScratch_.resize(levels.size());
    front.scanScratchAddress_.resize(levels.size());
    for (std::size_t i = 0; i < levels.size(); ++i)
    {
        dev(static_cast<std::size_t>(levels[i]) * sizeof(std::uint32_t), nullptr,
            front.scanScratch_[i], front.scanScratchAddress_[i], {}, "scanScratch");
    }

    // Repair scratch (B4). The repair ancestor cache SHARES the emit's ancestor buffer — repair
    // completes (its close+refine trailing barrier) before emit's ancestor pass in recordFrame, so
    // the transient cache never overlaps; sharing saves a per-instance vertex buffer. (The isolated
    // B4 buildWithFront keeps a separate allocation since it has no emit workspace.) Only the
    // control diagnostic (SEPARATE from failFlags) is new here.
    front.repairAncestorId_ = front.ancestorId_;
    front.repairAncestorIdAddress_ = front.ancestorIdAddress_;
    front.repairAncestorDepth_ = front.ancestorDepth_;
    front.repairAncestorDepthAddress_ = front.ancestorDepthAddress_;
    const std::array<std::uint32_t, 4> zeroCtrl{0, 0, 0, 0};
    dev(4 * sizeof(std::uint32_t), zeroCtrl.data(), front.repairControl_,
        front.repairControlAddress_, srcFlag, "repairControl");
    // Per-round convergence history (perf instrumentation) — kVdpmGpuRepairRoundBudget slots, one
    // per bounded repair round; each round's detect atomic-ORs its anyMarked here via address
    // redirection (see recordRepairImpl). Readback-enabled for the manager's delayed diagnostics.
    const std::vector<std::uint32_t> zeroHistory(kVdpmGpuRepairRoundBudget, 0);
    dev(kVdpmGpuRepairRoundBudget * sizeof(std::uint32_t), zeroHistory.data(), front.roundHistory_,
        front.roundHistoryAddress_, srcFlag, "roundHistory");

    // RINGED draw-consumed outputs (emitted indices / counters / indirect) + host-written repair
    // params — one per frame-in-flight, so a draw can read slot N while slot N+1 is computed.
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        FrameOutput& out = front.frameOutputs_[i];
        dev(fBytes * 3, nullptr, out.emittedIndices, out.emittedIndicesAddress,
            vk::BufferUsageFlagBits::eIndexBuffer | srcFlag, "emittedIndices");
        dev(3 * sizeof(std::uint32_t), nullptr, out.counters, out.countersAddress, srcFlag,
            "counters");
        dev(sizeof(DrawIndexedIndirectCommand), nullptr, out.indirect, out.indirectAddress,
            vk::BufferUsageFlagBits::eIndirectBuffer | srcFlag, "indirect");
    }
    const MappedBufferSet repairParams =
        resources.createMappedDeviceAddressBuffers(sizeof(VdpmRepairParams));
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        front.repairParamsRing_[i] = repairParams.mapped[i];
        front.repairParamsRingAddress_[i] = resources.bufferAddress(repairParams.buffers[i]);
        requireAligned(front.repairParamsRingAddress_[i], 16, "repairParams");
    }
    // Persistent-kernel job ring (Stage 2): one host-visible VdpmRepairJobGpu per frame slot.
    const MappedBufferSet jobs =
        resources.createMappedDeviceAddressBuffers(sizeof(VdpmRepairJobGpu));
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        front.jobRing_[i] = jobs.mapped[i];
        front.jobRingAddress_[i] = resources.bufferAddress(jobs.buffers[i]);
        requireAligned(front.jobRingAddress_[i], 8, "repairJob");
    }
    // Persistent apply-kernel job ring (apply-kernel arc): one host-visible VdpmApplyJobGpu per
    // slot.
    const MappedBufferSet applyJobs =
        resources.createMappedDeviceAddressBuffers(sizeof(VdpmApplyJobGpu));
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        front.applyJobRing_[i] = applyJobs.mapped[i];
        front.applyJobRingAddress_[i] = resources.bufferAddress(applyJobs.buffers[i]);
        requireAligned(front.applyJobRingAddress_[i], 8, "applyJob");
    }
    return front;
}

VdpmRepairJobGpu VdpmGpuFront::makeRepairJob(std::uint32_t frameIndex,
                                             std::uint32_t roundBudget) const
{
    if (!hasRuntime_ || !hasRepair_)
    {
        throw std::logic_error("VdpmGpuFront::makeRepairJob: not a runtime repair front");
    }
    if (roundBudget > kVdpmGpuRepairRoundBudget)
    {
        throw std::logic_error("VdpmGpuFront::makeRepairJob: roundBudget exceeds history capacity");
    }
    const std::uint32_t rankCount = static_cast<std::uint32_t>(rankRanges_.size());
    return VdpmRepairJobGpu{.activeAddress = activeStateAddress_,
                            .refinedAddress = refinedStateAddress_,
                            .requiredAddress = requiredStateAddress_,
                            .dependentsAddress = dependentsStateAddress_,
                            .failFlagsAddress = failFlagsAddress_,
                            .ancestorIdAddress = repairAncestorIdAddress_,
                            .ancestorDepthAddress = repairAncestorDepthAddress_,
                            .repairControlAddress = repairControlAddress_,
                            .roundHistoryAddress = roundHistoryAddress_,
                            .finestFacesAddress = binding_.finestFacesAddress,
                            .positionsAddress = binding_.positionsAddress,
                            .removalParentAddress = binding_.removalParentAddress,
                            .removingSplitAddress = binding_.removingSplitAddress,
                            .frontSplitsAddress = binding_.frontSplitsAddress,
                            .splitsByRankAddress = binding_.splitsByRankAddress,
                            .rankRangesAddress = binding_.rankRangesAddress,
                            .paramsAddress = repairParamsRingAddress_[frameIndex],
                            .vertexCount = binding_.vertexCount,
                            .finestFaceCount = binding_.finestFaceCount,
                            .splitCount = binding_.splitCount,
                            .maxDepth = binding_.maxDepth,
                            .rankCount = rankCount,
                            .roundBudget = roundBudget,
                            .roundHistoryCapacity = kVdpmGpuRepairRoundBudget,
                            .pad = 0};
}

VdpmApplyJobGpu VdpmGpuFront::prepareApplyJob(std::uint32_t /*frameIndex*/, float pixelBudget,
                                              float coarsenBudget) const
{
    if (!hasRuntime_ || !hasFront_)
    {
        throw std::logic_error("VdpmGpuFront::prepareApplyJob: not a runtime refine/coarsen front");
    }
    const std::uint32_t rankCount = static_cast<std::uint32_t>(rankRanges_.size());
    return VdpmApplyJobGpu{.scoresAddress = outputAddress_, // the front's own GPU score output
                           .activeAddress = activeStateAddress_,
                           .refinedAddress = refinedStateAddress_,
                           .requiredAddress = requiredStateAddress_,
                           .dependentsAddress = dependentsStateAddress_,
                           .failFlagsAddress = failFlagsAddress_,
                           .frontSplitsAddress = binding_.frontSplitsAddress,
                           .splitsByRankAddress = binding_.splitsByRankAddress,
                           .rankRangesAddress = binding_.rankRangesAddress,
                           .vertexCount = binding_.vertexCount,
                           .splitCount = binding_.splitCount,
                           .rankCount = rankCount,
                           .pad = 0,
                           .pixelBudget = pixelBudget,
                           .coarsenBudget = coarsenBudget};
}

VdpmRepairJobGpu VdpmGpuFront::prepareRepairJob(std::uint32_t frameIndex,
                                                const VdpmRepairParams& params,
                                                std::uint32_t roundBudget)
{
    if (!hasRuntime_ || !hasRepair_)
    {
        throw std::logic_error("VdpmGpuFront::prepareRepairJob: not a runtime repair front");
    }
    // Pack (and validate the budget) BEFORE mutating the params ring, so an oversized budget throws
    // without leaving a half-written ring slot.
    const VdpmRepairJobGpu job = makeRepairJob(frameIndex, roundBudget);
    writeMapped(repairParamsRing_[frameIndex], params);
    return job;
}

void VdpmGpuFront::recordRepairKernel(vk::CommandBuffer cmd, const VdpmRepairKernel& kernel,
                                      std::uint32_t frameIndex, const VdpmRepairParams& params,
                                      std::uint32_t roundBudget)
{
    if (!hasRuntime_ || !hasRepair_)
    {
        throw std::logic_error("VdpmGpuFront::recordRepairKernel: not a runtime repair front");
    }
    if (binding_.splitCount == 0 || binding_.finestFaceCount == 0)
    {
        return; // nothing to repair (no splits / no faces)
    }

    // Host-upload the params + the packed job (persistent mapping; the submit makes the writes
    // available to the dispatch, exactly like the CPU indirect write). The kernel reaches every
    // buffer via BDA, so no Resources needed here.
    const VdpmRepairJobGpu job = prepareRepairJob(frameIndex, params, roundBudget);
    writeMapped(jobRing_[frameIndex], job);

    // Leading barrier: applyView's coarsen (compute, no trailing barrier) → the kernel's reads of
    // the settled front state. recordFrame owns the cross-frame lifecycle barrier.
    recordComputeStageBoundary(cmd);
    kernel.recordDispatch(cmd, jobRingAddress_[frameIndex], 1); // ONE workgroup — whole fixpoint
    // No consumer barrier — the caller synchronises the state read-back.
}

void VdpmGpuFront::recordApplyKernel(vk::CommandBuffer cmd, const VdpmApplyKernel& kernel,
                                     std::uint32_t frameIndex, float pixelBudget,
                                     float coarsenBudget)
{
    if (!hasRuntime_ || !hasFront_)
    {
        throw std::logic_error(
            "VdpmGpuFront::recordApplyKernel: not a runtime refine/coarsen front");
    }
    if (binding_.splitCount == 0)
    {
        return; // nothing to apply (no splits) — matches recordApplyScoredView's early-out
    }

    // Pack + host-upload the job (budgets are in it — no separate params block). The submit makes
    // the write available to the dispatch, like the CPU indirect write.
    const VdpmApplyJobGpu job = prepareApplyJob(frameIndex, pixelBudget, coarsenBudget);
    writeMapped(applyJobRing_[frameIndex], job);

    // Leading barrier: the score dispatch (compute, no trailing barrier) → the kernel's mark read
    // of the score output. recordFrame owns the cross-frame lifecycle barrier.
    recordComputeStageBoundary(cmd);
    kernel.recordDispatch(cmd, applyJobRingAddress_[frameIndex], 1); // ONE workgroup — whole apply
    // No consumer barrier — the caller synchronises the state read-back.
}

void VdpmGpuFront::recordRepairRuntime(vk::CommandBuffer cmd,
                                       const VdpmRefinePipelines& refinePipelines,
                                       const VdpmRepairPipelines& repairPipelines,
                                       Resources& resources, std::uint32_t frameIndex,
                                       const VdpmRepairParams& params, std::uint32_t roundBudget)
{
    if (!hasRuntime_ || !hasRepair_)
    {
        throw std::logic_error("VdpmGpuFront::recordRepairRuntime: not a runtime repair front");
    }
    writeMapped(repairParamsRing_[frameIndex], params);
    recordRepairImpl(cmd, refinePipelines, repairPipelines, resources,
                     repairParamsRingAddress_[frameIndex], roundBudget);
}

void VdpmGpuFront::recordLifecycleBoundary(vk::CommandBuffer cmd)
{
    const vk::MemoryBarrier2 mb{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
}

void VdpmGpuFront::recordComputeStageBoundary(vk::CommandBuffer cmd)
{
    const vk::MemoryBarrier2 mb{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
}

void VdpmGpuFront::recordFrame(
    vk::CommandBuffer cmd, const ComputePipeline& scorePipeline,
    const VdpmRefinePipelines& refinePipelines, const VdpmRepairPipelines& repairPipelines,
    const VdpmEmitPipelines& emitPipelines, Resources& resources, std::uint32_t frameIndex,
    const VdpmViewParams& scoreView, const VdpmRepairParams& repairParams, float pixelBudget,
    float coarsenBudget, std::uint32_t repairRoundBudget, const VdpmApplyKernel* applyKernel,
    const VdpmRepairKernel* repairKernel, const VdpmStageProfile* stageProfile)
{
    if (!hasRuntime_)
    {
        throw std::logic_error("VdpmGpuFront::recordFrame: front is not a runtime front");
    }

    // Per-stage timing (no-ops unless stageProfile is set). The GPU boundaries are stamped
    // BOTTOM-of-pipe and SHARED — the timestamp after stage i is written as both stage i's end and
    // stage i+1's begin — so each resolved passMs is a clean consecutive delta with no top-of-pipe
    // bleed across these sub-millisecond stages. `gpuBoundary(pass, end)` writes one such stamp
    // (only when a single front is recorded — the query slots are one-shot per frame). CPU timing
    // is independent: steady_clock around each record call, accumulated into cpuMs[idx].
    const bool gpuStage = stageProfile != nullptr && stageProfile->gpu != nullptr;
    auto gpuBoundary = [&](ProfilePass pass, bool end)
    {
        if (gpuStage)
        {
            stageProfile->gpu->stampBottom(cmd, stageProfile->gpuFrameIndex, pass, end);
        }
    };
    auto cpuAccumulate = [&](std::size_t idx, std::chrono::steady_clock::time_point t0)
    {
        if (stageProfile != nullptr && stageProfile->cpuMs != nullptr)
        {
            (*stageProfile->cpuMs)[idx] +=
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    };
    // LIFECYCLE BOUNDARY barrier: the score buffer + the internal emit scratch are SINGLE-buffered
    // (only the draw-consumed outputs ring), so a prior frame's mark/coarsen READS of the score
    // output (and its emit-scratch reads/writes) must be ordered before THIS frame's recordScore
    // WRITE + emit-scratch reuse — a write-after-read the per-pass barriers (which sit AFTER the
    // score dispatch) can't cover. src names READS + WRITES; a no-op for the first frame on a fresh
    // front. (Also the only synchronisation on a zero-split mesh, where score/apply/repair all
    // early-out and two back-to-back emits would otherwise race the shared scratch.)
    recordLifecycleBoundary(cmd);

    // Boundary b0: bottom-of-pipe before the score dispatch (VdpmScore's begin).
    gpuBoundary(ProfilePass::VdpmScore, /*end=*/false);
    // (1) Score → the front's score output (score params written to this frame's ring slot). No
    // barrier here — recordApplyScoredView's leading barrier orders the score write → the mark
    // read.
    {
        const auto t = std::chrono::steady_clock::now();
        recordScore(cmd, scorePipeline, frameIndex, scoreView);
        cpuAccumulate(0, t);
    }
    // Boundary b1 (after score): VdpmScore's end AND VdpmApply's begin.
    gpuBoundary(ProfilePass::VdpmScore, /*end=*/true);
    gpuBoundary(ProfilePass::VdpmApply, /*end=*/false);
    // (2) Refine/coarsen reading that GPU score output (no host round-trip). The single-dispatch
    // persistent kernel (when the device supports it) or the multi-dispatch recorder. The kernel's
    // own leading score→kernel barrier orders the score write → its mark read; the recorder's
    // leading barrier does the same. Either way it writes active/refined/dependents, which repair's
    // leading barrier orders → repair's reads.
    {
        const auto t = std::chrono::steady_clock::now();
        if (applyKernel != nullptr)
        {
            recordApplyKernel(cmd, *applyKernel, frameIndex, pixelBudget, coarsenBudget);
        }
        else
        {
            recordApplyScoredView(cmd, refinePipelines, resources, pixelBudget, coarsenBudget);
        }
        cpuAccumulate(1, t);
    }
    // Boundary b2 (after apply): VdpmApply's end AND VdpmRepair's begin.
    gpuBoundary(ProfilePass::VdpmApply, /*end=*/true);
    gpuBoundary(ProfilePass::VdpmRepair, /*end=*/false);
    // (3) Repair, reading this frame's ring repair params. The single-dispatch kernel (when the
    // device supports it) or the multi-dispatch recorder.
    {
        const auto t = std::chrono::steady_clock::now();
        if (repairKernel != nullptr)
        {
            recordRepairKernel(cmd, *repairKernel, frameIndex, repairParams, repairRoundBudget);
        }
        else
        {
            recordRepairRuntime(cmd, refinePipelines, repairPipelines, resources, frameIndex,
                                repairParams, repairRoundBudget);
        }
        // Order the front-state writes (`active`/`refined`/`dependents`) → emit's ancestor read of
        // `active` (recordEmitFromFront's leading fillBuffer + clearToCompute orders only the
        // counter CLEAR, not these). Exactly ONE barrier is needed, and only sometimes:
        //  - splitCount == 0: score/apply/repair ALL early-out — nothing wrote the state, so the
        //    frame lifecycle barrier suffices; record none. (Reachable — a zero-split-but-faced
        //    mesh.)
        //  - kernel repair (splitCount > 0): the kernel records no trailing barrier, so add it here
        //  —
        //    as repair→emit when repair ran (finestFaceCount > 0), else as apply→emit (repair
        //    skipped but the apply DID write `active`).
        //  - recorder repair: recordRepairImpl's trailing close/refine barrier already orders
        //    repair→emit WHEN it ran; only when it early-outs (finestFaceCount == 0) — apply still
        //    having written — do we add the apply→emit barrier here.
        // All four cases are REACHABLE for a runtime front — finestFaceCount == 0 is a valid empty
        // repair-face array (a degenerate but raw-faced mesh; pinned by a `[.][gpu]` test).
        if (binding_.splitCount > 0 && (repairKernel != nullptr || binding_.finestFaceCount == 0))
        {
            recordComputeStageBoundary(cmd); // front-state (apply/repair) writes → emit's reads
        }
        cpuAccumulate(2, t);
    }
    // Boundary b3 (after repair): VdpmRepair's end AND VdpmEmit's begin.
    gpuBoundary(ProfilePass::VdpmRepair, /*end=*/true);
    gpuBoundary(ProfilePass::VdpmEmit, /*end=*/false);
    // (4) Emit the live settled front into this frame's ring output. No consumer barrier — the
    // caller (renderer) adds the compute→(index-read + indirect-read) barrier before the draw.
    {
        const auto t = std::chrono::steady_clock::now();
        recordEmitFromFront(cmd, emitPipelines, resources, frameIndex);
        cpuAccumulate(3, t);
    }
    // Boundary b4: bottom-of-pipe after emit (VdpmEmit's end).
    gpuBoundary(ProfilePass::VdpmEmit, /*end=*/true);
}

void VdpmGpuFront::recordCloseAndRefineRequired(vk::CommandBuffer cmd,
                                                const VdpmRefinePipelines& pipelines)
{
    const std::uint32_t maxRank = binding_.maxRank;

    // Leading barrier: the seed write (mark's `required`, or B4's detect) → the closure's read.
    emitComputeBarrier(cmd);

    // CLOSE in DESCENDING rank: each dispatch barriers after it (between close ranks, and the last
    // serves as close→refine).
    for (std::uint32_t r = maxRank + 1; r-- > 0;)
    {
        const RankRange& rr = rankRanges_[r];
        const VdpmClosePush push{.splitsByRankAddress = binding_.splitsByRankAddress,
                                 .frontSplitsAddress = binding_.frontSplitsAddress,
                                 .requiredAddress = requiredStateAddress_,
                                 .rankOffset = rr.offset,
                                 .rankCount = rr.count};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.close().pipeline());
        cmd.pushConstants<VdpmClosePush>(pipelines.close().pipelineLayout(),
                                         vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(emitGroups(rr.count), 1, 1);
        emitComputeBarrier(cmd);
    }

    // REFINE in ASCENDING rank (dependencies first): barrier between ranks; a trailing barrier
    // hands the refined state to the consumer (coarsen, or B4's next detect).
    for (std::uint32_t r = 0; r <= maxRank; ++r)
    {
        const RankRange& rr = rankRanges_[r];
        const VdpmRefinePush push{.splitsByRankAddress = binding_.splitsByRankAddress,
                                  .frontSplitsAddress = binding_.frontSplitsAddress,
                                  .requiredAddress = requiredStateAddress_,
                                  .refinedAddress = refinedStateAddress_,
                                  .activeAddress = activeStateAddress_,
                                  .dependentsAddress = dependentsStateAddress_,
                                  .failFlagsAddress = failFlagsAddress_,
                                  .rankOffset = rr.offset,
                                  .rankCount = rr.count};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.refine().pipeline());
        cmd.pushConstants<VdpmRefinePush>(pipelines.refine().pipelineLayout(),
                                          vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(emitGroups(rr.count), 1, 1);
        if (r < maxRank)
        {
            emitComputeBarrier(cmd);
        }
    }
    emitComputeBarrier(cmd);
}

void VdpmGpuFront::recordApplyView(vk::CommandBuffer cmd, const VdpmRefinePipelines& pipelines,
                                   Resources& resources, std::span<const VdpmScoreOut> scores,
                                   float pixelBudget, float coarsenBudget)
{
    if (!hasFront_)
    {
        throw std::logic_error("VdpmGpuFront::recordApplyView: front has no refine/coarsen state");
    }
    if (scores.size() != binding_.splitCount)
    {
        throw std::runtime_error("VdpmGpuFront::recordApplyView: scores size != splitCount");
    }
    if (binding_.splitCount == 0)
    {
        return; // no splits: the front is fixed at the coarsest = finest state
    }
    // Upload the per-split scores (host-visible coherent → visible at queue submit), then run the
    // shared body reading the uploaded buffer.
    writeMapped(scoresMapped_, scores.data(), scores.size_bytes());
    recordApplyViewImpl(cmd, pipelines, resources, scoresAddress_, pixelBudget, coarsenBudget);
}

void VdpmGpuFront::recordApplyScoredView(vk::CommandBuffer cmd,
                                         const VdpmRefinePipelines& pipelines, Resources& resources,
                                         float pixelBudget, float coarsenBudget)
{
    if (!hasFront_)
    {
        throw std::logic_error(
            "VdpmGpuFront::recordApplyScoredView: front has no refine/coarsen state");
    }
    if (binding_.splitCount == 0)
    {
        return;
    }
    // Read the front's own GPU score output (recordScore wrote it; the caller ordered a barrier).
    recordApplyViewImpl(cmd, pipelines, resources, outputAddress_, pixelBudget, coarsenBudget);
}

void VdpmGpuFront::recordApplyViewImpl(vk::CommandBuffer cmd, const VdpmRefinePipelines& pipelines,
                                       Resources& resources, std::uint64_t scoresAddress,
                                       float pixelBudget, float coarsenBudget)
{
    // Defence-in-depth dispatch validation (build already rejected an ineligible mesh).
    requireDispatchable(emitGroups(binding_.splitCount), maxWorkGroupCountX_, "mark");
    for (const RankRange& rr : rankRanges_)
    {
        requireDispatchable(emitGroups(rr.count), maxWorkGroupCountX_, "rank");
    }

    // Leading barrier: order any PRIOR frame's compute writes (persistent state, `required`,
    // `failFlags`) before this frame's failFlags clear (transfer) and mark (compute). This makes
    // two recordApplyView calls back-to-back in ONE submit correctly serialised (the
    // persistent-front contract — frame N+1 reads frame N's settled state); a no-op for the first
    // call on a fresh front. (Cross-frame the caller must also keep the score buffer stable until
    // the prior frame's reads complete — a compute-read→compute-write barrier, or per-frame score
    // buffers, in B5.)
    const vk::MemoryBarrier2 priorToThis{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask =
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eClear,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                         vk::AccessFlagBits2::eShaderStorageWrite |
                         vk::AccessFlagBits2::eTransferWrite,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &priorToThis});

    // Clear the invariant-failure flags for THIS frame (a persistent front would otherwise carry a
    // prior frame's flag). fillBuffer is a CLEAR (eClear / eTransferWrite).
    cmd.fillBuffer(resources.vulkanBuffer(failFlags_), 0, 2 * sizeof(std::uint32_t), 0);
    const vk::MemoryBarrier2 clearToCompute{
        .srcStageMask = vk::PipelineStageFlagBits2::eClear,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});

    // (1) MARK — full-overwrite the required seed over all splits.
    {
        const VdpmMarkPush push{.scoresAddress = scoresAddress,
                                .requiredAddress = requiredStateAddress_,
                                .splitCount = binding_.splitCount,
                                .pixelBudget = pixelBudget};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.mark().pipeline());
        cmd.pushConstants<VdpmMarkPush>(pipelines.mark().pipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(emitGroups(binding_.splitCount), 1, 1);
    }

    // (2) CLOSE + REFINE (the shared recorder owns its boundary barriers, incl. the trailing
    // refine→consumer one that hands the refined state to coarsen below).
    recordCloseAndRefineRequired(cmd, pipelines);

    // (3) COARSEN in DESCENDING rank (dependents first), barrier between ranks; none after the last
    // (the caller synchronises the state read-back).
    for (std::uint32_t r = binding_.maxRank + 1; r-- > 0;)
    {
        const RankRange& rr = rankRanges_[r];
        const VdpmCoarsenPush push{.splitsByRankAddress = binding_.splitsByRankAddress,
                                   .frontSplitsAddress = binding_.frontSplitsAddress,
                                   .scoresAddress = scoresAddress,
                                   .refinedAddress = refinedStateAddress_,
                                   .activeAddress = activeStateAddress_,
                                   .dependentsAddress = dependentsStateAddress_,
                                   .failFlagsAddress = failFlagsAddress_,
                                   .rankOffset = rr.offset,
                                   .rankCount = rr.count,
                                   .coarsenBudget = coarsenBudget};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.coarsen().pipeline());
        cmd.pushConstants<VdpmCoarsenPush>(pipelines.coarsen().pipelineLayout(),
                                           vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(emitGroups(rr.count), 1, 1);
        if (r > 0)
        {
            emitComputeBarrier(cmd);
        }
    }
}

void VdpmGpuFront::recordRepair(vk::CommandBuffer cmd, const VdpmRefinePipelines& refinePipelines,
                                const VdpmRepairPipelines& repairPipelines, Resources& resources,
                                const VdpmRepairParams& params, std::uint32_t roundBudget)
{
    if (!hasRepair_)
    {
        throw std::logic_error("VdpmGpuFront::recordRepair: front has no repair state (full mesh + "
                               "buildWithFront required)");
    }
    // Upload to the single (non-ringed) repair-params buffer, then run the shared body.
    writeMapped(repairParamsMapped_, params);
    recordRepairImpl(cmd, refinePipelines, repairPipelines, resources, repairParamsAddress_,
                     roundBudget);
}

void VdpmGpuFront::recordRepairImpl(vk::CommandBuffer cmd,
                                    const VdpmRefinePipelines& refinePipelines,
                                    const VdpmRepairPipelines& repairPipelines,
                                    Resources& resources, std::uint64_t paramsAddress,
                                    std::uint32_t roundBudget)
{
    if (binding_.splitCount == 0 || binding_.finestFaceCount == 0)
    {
        return; // nothing to repair (no splits / no faces)
    }
    requireDispatchable(emitGroups(binding_.vertexCount), maxWorkGroupCountX_, "repair ancestor");
    requireDispatchable(emitGroups(binding_.finestFaceCount), maxWorkGroupCountX_, "repair detect");
    requireDispatchable(emitGroups(binding_.splitCount), maxWorkGroupCountX_, "repair fallback");

    const vk::Buffer requiredBuf = resources.vulkanBuffer(requiredState_);
    const vk::Buffer controlBuf = resources.vulkanBuffer(repairControl_);
    const auto sBytes = static_cast<vk::DeviceSize>(binding_.splitCount) * sizeof(std::uint32_t);

    // Leading barrier: applyView's coarsen (compute, no trailing barrier) → this repair's first
    // reads (active/refined) + clears. Covers applyView → repair generally.
    const vk::MemoryBarrier2 lead{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask =
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eClear,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                         vk::AccessFlagBits2::eShaderStorageWrite |
                         vk::AccessFlagBits2::eTransferWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &lead});

    // Clear the whole repair-control buffer once ([anyMarked, ancestorFailure, fallbackFired,
    // pad]).
    cmd.fillBuffer(controlBuf, 0, 4 * sizeof(std::uint32_t), 0);
    const vk::MemoryBarrier2 clearToCompute{
        .srcStageMask = vk::PipelineStageFlagBits2::eClear,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});

    // Order a prior compute write to `required` (the last close/refine) before a fillBuffer clear,
    // then expose the clear to the next compute — the compute→eClear→compute reset the reviewer
    // pinned (B3's trailing barrier targets compute, not eClear).
    auto resetRequired = [&]()
    {
        const vk::MemoryBarrier2 computeToClear{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eClear,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &computeToClear});
        cmd.fillBuffer(requiredBuf, 0, sBytes, 0);
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});
    };

    // Ancestor resolve (live front → cache) + detect (mark violations). NO trailing barrier — the
    // caller adds detect→consumer (close, or the fallback). `anyMarkedAddress` is where the
    // detect's atomic `anyMarked` OR lands — repairControl[0] normally, or a per-round history slot
    // for the bounded rounds (perf instrumentation, via address redirection; the SAME atomic the
    // round already did, just aimed at its own slot — the accumulated bounded value in
    // repairControl[0] was unused; only the final detect drives the fallback).
    auto ancestorThenDetect = [&](std::uint64_t anyMarkedAddress)
    {
        recordAncestorResolve(
            cmd, repairPipelines.ancestor(),
            VdpmAncestorPush{.activeAddress = activeStateAddress_,
                             .removalParentAddress = binding_.removalParentAddress,
                             .ancestorIdAddress = repairAncestorIdAddress_,
                             .ancestorDepthAddress = repairAncestorDepthAddress_,
                             // ancestor failures → repairControl[1] (its counters[0] == control+1).
                             .countersAddress = repairControlAddress_ + sizeof(std::uint32_t),
                             .vertexCount = binding_.vertexCount,
                             .maxDepth = binding_.maxDepth});
        emitComputeBarrier(cmd); // ancestor cache → detect read
        const VdpmRepairDetectPush push{.finestFacesAddress = binding_.finestFacesAddress,
                                        .positionsAddress = binding_.positionsAddress,
                                        .activeAddress = activeStateAddress_,
                                        .ancestorIdAddress = repairAncestorIdAddress_,
                                        .removingSplitAddress = binding_.removingSplitAddress,
                                        .requiredAddress = requiredStateAddress_,
                                        .repairControlAddress = anyMarkedAddress,
                                        .paramsAddress = paramsAddress,
                                        .classificationAddress = 0, // production: no classification
                                        .faceCount = binding_.finestFaceCount,
                                        .writeClassification = 0};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, repairPipelines.detect().pipeline());
        cmd.pushConstants<VdpmRepairDetectPush>(repairPipelines.detect().pipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(emitGroups(binding_.finestFaceCount), 1, 1);
    };

    // Clear the per-round convergence history once (runtime front only), before the bounded rounds
    // redirect their detect atomic into per-round slots. eClear→compute so the first round's detect
    // sees a defined slot.
    const bool captureHistory = roundHistoryAddress_ != 0;
    if (captureHistory)
    {
        cmd.fillBuffer(resources.vulkanBuffer(roundHistory_), 0,
                       kVdpmGpuRepairRoundBudget * sizeof(std::uint32_t), 0);
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});
    }

    // Bounded snapshot rounds: reset required → detect → close+refine (the shared recorder barriers
    // detect→close and refine→consumer). The detect's anyMarked goes to this round's history slot
    // (or repairControl[0] on an isolated front / a budget past the history size).
    for (std::uint32_t round = 0; round < roundBudget; ++round)
    {
        resetRequired();
        const std::uint64_t anyMarkedAddress =
            (captureHistory && round < kVdpmGpuRepairRoundBudget)
                ? roundHistoryAddress_ + static_cast<std::uint64_t>(round) * sizeof(std::uint32_t)
                : repairControlAddress_;
        ancestorThenDetect(anyMarkedAddress);
        recordCloseAndRefineRequired(cmd, refinePipelines);
    }

    // Final detection: clear `anyMarked` (repairControl[0]) so it reflects ONLY this pass, reset
    // required, detect. Then the fallback reads anyMarked with no CPU round-trip.
    {
        const vk::MemoryBarrier2 computeToClear{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eClear,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &computeToClear});
        cmd.fillBuffer(controlBuf, 0, sizeof(std::uint32_t), 0); // just anyMarked
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});
    }
    resetRequired();
    ancestorThenDetect(
        repairControlAddress_); // final detect → repairControl[0] (fallback reads it)
    emitComputeBarrier(cmd);    // detect (anyMarked + required) → fallback reads

    // Fallback: seed every unrefined split iff anyMarked, else clear required — then close+refine
    // drives to full detail (guaranteed hole-free) or no-ops.
    {
        const VdpmRepairFallbackPush push{.requiredAddress = requiredStateAddress_,
                                          .refinedAddress = refinedStateAddress_,
                                          .repairControlAddress = repairControlAddress_,
                                          .splitCount = binding_.splitCount};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, repairPipelines.fallback().pipeline());
        cmd.pushConstants<VdpmRepairFallbackPush>(repairPipelines.fallback().pipelineLayout(),
                                                  vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(emitGroups(binding_.splitCount), 1, 1);
    }
    recordCloseAndRefineRequired(cmd, refinePipelines);
    // No consumer barrier — the caller synchronises the state read-back.
}

void VdpmGpuFront::recordDetectClassify(vk::CommandBuffer cmd,
                                        const VdpmRepairPipelines& repairPipelines,
                                        Resources& resources, const VdpmRepairParams& params)
{
    if (!hasRepair_ || repairClassification_ == NullBuffer)
    {
        throw std::logic_error("VdpmGpuFront::recordDetectClassify: front has no classification "
                               "buffer (buildWithFront withClassificationReadback)");
    }
    if (binding_.finestFaceCount == 0)
    {
        return;
    }
    writeMapped(repairParamsMapped_, params);

    // Leading barrier (prior compute → this clear + compute), then clear required + repairControl.
    const vk::MemoryBarrier2 lead{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask =
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eClear,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                         vk::AccessFlagBits2::eShaderStorageWrite |
                         vk::AccessFlagBits2::eTransferWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &lead});
    cmd.fillBuffer(resources.vulkanBuffer(repairControl_), 0, 4 * sizeof(std::uint32_t), 0);
    // A split-free (already-full-detail) mesh has no `required` buffer; the detect never marks
    // (every corner is active), so skip the clear rather than fillBuffer a null zero-size buffer.
    if (binding_.splitCount > 0)
    {
        cmd.fillBuffer(resources.vulkanBuffer(requiredState_), 0,
                       static_cast<vk::DeviceSize>(binding_.splitCount) * sizeof(std::uint32_t), 0);
    }
    const vk::MemoryBarrier2 clearToCompute{
        .srcStageMask = vk::PipelineStageFlagBits2::eClear,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});

    recordAncestorResolve(
        cmd, repairPipelines.ancestor(),
        VdpmAncestorPush{.activeAddress = activeStateAddress_,
                         .removalParentAddress = binding_.removalParentAddress,
                         .ancestorIdAddress = repairAncestorIdAddress_,
                         .ancestorDepthAddress = repairAncestorDepthAddress_,
                         .countersAddress = repairControlAddress_ + sizeof(std::uint32_t),
                         .vertexCount = binding_.vertexCount,
                         .maxDepth = binding_.maxDepth});
    emitComputeBarrier(cmd);

    const VdpmRepairDetectPush push{.finestFacesAddress = binding_.finestFacesAddress,
                                    .positionsAddress = binding_.positionsAddress,
                                    .activeAddress = activeStateAddress_,
                                    .ancestorIdAddress = repairAncestorIdAddress_,
                                    .removingSplitAddress = binding_.removingSplitAddress,
                                    .requiredAddress = requiredStateAddress_,
                                    .repairControlAddress = repairControlAddress_,
                                    .paramsAddress = repairParamsAddress_,
                                    .classificationAddress = repairClassificationAddress_,
                                    .faceCount = binding_.finestFaceCount,
                                    .writeClassification = 1};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, repairPipelines.detect().pipeline());
    cmd.pushConstants<VdpmRepairDetectPush>(repairPipelines.detect().pipelineLayout(),
                                            vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(emitGroups(binding_.finestFaceCount), 1, 1);
    // No consumer barrier — the caller synchronises the classification/required read-back.
}

void VdpmGpuFront::recordScore(vk::CommandBuffer cmd, const ComputePipeline& pipeline,
                               std::uint32_t frameIndex, const VdpmViewParams& view)
{
    if (binding_.splitCount == 0)
    {
        return; // zero-split: no buffers, no dispatch
    }
    // Write this frame slot's params (the three buffer addresses + splitCount + the view image).
    const VdpmScoreParams params =
        packVdpmScoreParams(view, binding_.splitsAddress, binding_.positionsAddress, outputAddress_,
                            binding_.splitCount);
    writeMapped(paramsMapped_[frameIndex], params);

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline());
    const VdpmScorePushConstants push{.paramsAddress = paramsAddress_[frameIndex]};
    cmd.pushConstants<VdpmScorePushConstants>(pipeline.pipelineLayout(),
                                              vk::ShaderStageFlagBits::eCompute, 0, push);
    const std::uint32_t groups = emitGroups(binding_.splitCount); // 64-wide, 64-bit ceil
    requireDispatchable(groups, maxWorkGroupCountX_, "score");
    cmd.dispatch(groups, 1, 1);
}

} // namespace fire_engine
