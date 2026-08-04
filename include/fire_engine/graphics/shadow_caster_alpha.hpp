#pragma once

#include <fire_engine/graphics/shadow_geometry_request.hpp>

namespace fire_engine
{

class Material;

// SH-05. Classify a shadow caster's fragment coverage from its material — the one place that
// decides whether a caster's shadow is its triangles or its cutout.
//
// A free function on the same reasoning as `shadowCasterDeformation`: it is a property of the
// material, it must be testable without a GPU, and `object.cpp` is its only production caller, so
// the classification and the two responses to it (the resolver's LOD pin, the shadow pass' fragment
// path) are tested independently rather than each proving the other.
//
// Deliberately keyed on the material's declared alpha mode ALONE — not on whether a base-colour
// texture happens to be bound. A MASK material with no texture still tests its base-colour factor's
// alpha against its cutoff, and a shadow that ignored that would occlude where the forward pass
// draws nothing at all.
[[nodiscard]] ShadowCasterAlpha shadowCasterAlpha(const Material& material) noexcept;

} // namespace fire_engine
