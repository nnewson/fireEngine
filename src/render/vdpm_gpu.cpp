#include <fire_engine/render/vdpm_gpu.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <fire_engine/graphics/mapped_buffer.hpp>

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

    VdpmGpuMesh mesh;
    mesh.splitCount_ = static_cast<std::uint32_t>(forest.splits.size());

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
    return mesh;
}

VdpmGpuFront VdpmGpuFront::build(Resources& resources, const VdpmGpuMesh& mesh)
{
    VdpmGpuFront front;
    front.binding_ = mesh.binding(); // copy the immutable binding — no pointer into the mesh
    if (front.binding_.splitCount == 0)
    {
        return front; // nothing to score
    }

    // Device-local score output, readback-enabled (eTransferSrc) so the harness can copy it back.
    front.output_ = resources.createDeviceLocalStorageBuffer(
        mesh.splitCount() * sizeof(VdpmScoreOut), nullptr, /*allowReadback=*/true);
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
    constexpr std::uint32_t kLocalSize = 64;
    const std::uint32_t groups = (binding_.splitCount + kLocalSize - 1) / kLocalSize;
    cmd.dispatch(groups, 1, 1);
}

} // namespace fire_engine
