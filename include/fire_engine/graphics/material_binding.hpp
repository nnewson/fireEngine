#pragma once

#include <array>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/material.hpp>

namespace fire_engine
{

struct MaterialUBO;

using MaterialTextureHandles = std::array<TextureHandle, materialTextureSlotCount>;

[[nodiscard]]
MaterialUBO toMaterialUBO(const Material& material);

// Which of the GPU material's alpha ranges this material violates, if any. `toMaterialUBO`
// normalises both — packed alpha into glTF's [0,1], packed cutoff to >= 0 — because
// `depth_prepass.frag` skips its cutout test when the packed cutoff is 0, which is equivalent to
// running it only while alpha >= 0 (see material_binding.cpp for the full argument).
//
// PURE and noexcept, and separate from the warning below on purpose: `toMaterialUBO` is reachable
// through `materialsEquivalent` from `Object::wouldChangeVariant` and
// `Mesh::isSelectableVariantState`, both `noexcept`, so nothing on that path may throw — and a
// formatting or allocation failure inside a log call would terminate the process rather than report
// anything.
struct MaterialAlphaRangeIssues
{
    bool alpha{false};
    bool cutoff{false};

    [[nodiscard]] bool any() const noexcept
    {
        return alpha || cutoff;
    }
};

[[nodiscard]]
MaterialAlphaRangeIssues materialAlphaRangeIssues(const Material& material) noexcept;

// Reports the above. NOT noexcept, and deliberately NOT called from the packing path: call it from
// a non-noexcept site that runs ONCE per material — `Resources::registerMaterial`'s first-sight
// branch — so a per-frame variant-selection query can neither terminate the process nor re-emit the
// same warning every frame.
void warnOnMaterialAlphaRangeIssues(const Material& material);

[[nodiscard]]
bool materialsEquivalent(const Material& lhs, const Material& rhs);

[[nodiscard]]
MaterialTextureHandles materialTextureHandles(const Material& material) noexcept;

} // namespace fire_engine
