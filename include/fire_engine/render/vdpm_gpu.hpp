#pragma once

#include <array>
#include <cstdint>
#include <span>

#include <vulkan/vulkan.hpp>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/compute_pipeline.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

// CPU → GPU packing for the VDPM GPU-front scoring ABI (Stage B1). These convert the Vulkan-free
// scoring authority (VdpmViewParams / VertexSplit, graphics/vdpm.hpp) into the std430 SSBO images
// (render/ubo.hpp) the compute shader reads. Keeping the conversion here — a thin, tested field
// copy — is what stops the oracle, the uploader, and the shader from drifting into three different
// scorings. (The score pipeline + dispatch live in vdpm_gpu.cpp alongside these.)

// Static per-split metric record. parent/child become IDs into the canonical-position buffer.
[[nodiscard]] VdpmSplitGpu packVdpmSplit(const VertexSplit& split) noexcept;

// One canonical vertex's object-space position (padded to a vec4).
[[nodiscard]] VdpmPositionGpu packVdpmPosition(const Vec3& position) noexcept;

// Per-instance params: the std430 image of `view` plus the three buffer_reference device addresses
// and the split count. `worldLinear` is written as three padded vec4 columns (a GLSL mat3).
[[nodiscard]] VdpmScoreParams packVdpmScoreParams(const VdpmViewParams& view,
                                                  std::uint64_t splitsAddress,
                                                  std::uint64_t positionsAddress,
                                                  std::uint64_t outputsAddress,
                                                  std::uint32_t splitCount) noexcept;

// The score compute pipeline: `shaders/vdpm_score.comp`. No descriptor sets (everything is reached
// by buffer_reference); one push-constant range carrying the params block's device address.
[[nodiscard]] ComputePipelineConfig vdpmScorePipelineConfig();

// STATIC per-geometry GPU data — the per-split metric records + the canonical-vertex positions,
// device-local and uploaded ONCE. Shared by every instance of the geometry (never duplicated per
// instance); the later repair/emit stages reuse the same positions. parent/child IDs in the split
// records index the positions array.
class VdpmGpuMesh
{
public:
    // `vertices` is the ORIGINAL vertex array (canonical IDs index into it); `forest` supplies the
    // splits. Both uploaded device-local via staging.
    [[nodiscard]] static VdpmGpuMesh build(Resources& resources, std::span<const Vertex> vertices,
                                           const VertexForest& forest);

    [[nodiscard]] std::uint64_t splitsAddress() const noexcept
    {
        return splitsAddress_;
    }
    [[nodiscard]] std::uint64_t positionsAddress() const noexcept
    {
        return positionsAddress_;
    }
    [[nodiscard]] std::uint32_t splitCount() const noexcept
    {
        return splitCount_;
    }

private:
    BufferHandle splits_{NullBuffer};
    BufferHandle positions_{NullBuffer};
    std::uint64_t splitsAddress_{0};
    std::uint64_t positionsAddress_{0};
    std::uint32_t splitCount_{0};
};

// PER-INSTANCE GPU front state — the per-split score/backface output + the per-frame mapped params
// block. References its shared VdpmGpuMesh. (Stage B1 carries only scoring; the active-front state
// joins here in later stages.)
class VdpmGpuFront
{
public:
    [[nodiscard]] static VdpmGpuFront build(Resources& resources, const VdpmGpuMesh& mesh);

    // Record ONLY the score dispatch for frame `frameIndex` (writes that slot's mapped params from
    // `view`, pushes its address, dispatches ceil(splitCount / 64)). NO barriers — the consumer
    // owns synchronisation (the harness a compute→transfer→host readback; later stages
    // compute→compute). A zero-split front records nothing.
    void recordScore(vk::CommandBuffer cmd, const ComputePipeline& pipeline,
                     std::uint32_t frameIndex, const VdpmViewParams& view);

    // The device-local score output (one VdpmScoreOut per split), created with eTransferSrc so a
    // test can copy it back.
    [[nodiscard]] BufferHandle outputBuffer() const noexcept
    {
        return output_;
    }
    [[nodiscard]] std::uint32_t splitCount() const noexcept
    {
        return mesh_ != nullptr ? mesh_->splitCount() : 0u;
    }

private:
    const VdpmGpuMesh* mesh_{nullptr};
    BufferHandle output_{NullBuffer};
    std::array<std::span<std::byte>, kMaxFramesInFlight> paramsMapped_{};
    std::array<std::uint64_t, kMaxFramesInFlight> paramsAddress_{};
    std::uint64_t outputAddress_{0};
};

} // namespace fire_engine
