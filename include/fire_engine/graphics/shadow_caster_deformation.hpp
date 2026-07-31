#pragma once

#include <fire_engine/graphics/shadow_geometry_request.hpp>

namespace fire_engine
{

class Geometry;

// SH-04. Classify a shadow caster as Rigid or Deformable — the one place that decides whether a
// caster's recorded error describes the geometry that will actually be rasterised.
//
// A free function rather than an `Object` member because it is a property of the pairing (this
// geometry, deformed by this instance), and because it must be testable against a real `Geometry`
// without a GPU. `object.cpp` is the only production caller; the classification and the resolver's
// response to it are then tested independently, so neither can be proven only by the other.
//
// `objectDeforms` is the instance's `Object::deformable()` — a skin or morph weights bound to THIS
// instance. The geometry contributes the two facts an instance cannot know: whether it carries
// morph targets at all (independent of current weights) and whether a compute pass owns its vertex
// buffer.
[[nodiscard]] ShadowCasterDeformation shadowCasterDeformation(const Geometry& geometry,
                                                              bool objectDeforms) noexcept;

} // namespace fire_engine
