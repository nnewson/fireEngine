#pragma once

#include <fire_engine/graphics/lod.hpp>

namespace fire_engine
{

class Material;

// Per-channel VDPM refine scales for a material's `refineForView` call. A scale of 0 disables a
// channel (its screen score never exceeds the pixel budget), so the front spends no triangles
// preserving detail the material can't show — an unlit surface has no shading to protect, a mesh
// with tangents but no normal map never samples its tangent frame, an untextured material never
// samples UV. Derived at REFINE time from the material, not baked into the collapse stream, so the
// stream stays material-agnostic and reusable across material variants. Vulkan-free.
struct VdpmChannelScales
{
    float uv{kVdpmUvScale};
    float normal{kVdpmNormalScale};
    float tangent{kVdpmTangentScale};
};

// Derive the per-channel refine scales for `material` (see VdpmChannelScales). Pure; unit-tested.
[[nodiscard]] VdpmChannelScales vdpmChannelScales(const Material& material) noexcept;

} // namespace fire_engine
