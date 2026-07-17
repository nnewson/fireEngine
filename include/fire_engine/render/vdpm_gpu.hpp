#pragma once

#include <cstdint>

#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/math/vec3.hpp>
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

} // namespace fire_engine
