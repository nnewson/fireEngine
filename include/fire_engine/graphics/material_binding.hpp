#pragma once

#include <array>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/render/descriptor_bindings.hpp>

namespace fire_engine
{

class Material;
struct MaterialUBO;

using MaterialTextureHandles = std::array<TextureHandle, materialTextureSlotCount>;

[[nodiscard]]
MaterialUBO toMaterialUBO(const Material& material);

[[nodiscard]]
bool materialsEquivalent(const Material& lhs, const Material& rhs);

[[nodiscard]]
MaterialTextureHandles materialTextureHandles(const Material& material) noexcept;

} // namespace fire_engine
