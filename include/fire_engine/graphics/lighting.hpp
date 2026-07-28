#pragma once

#include <fire_engine/core/node_id.hpp>
#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// World-space resolved light data emitted by Light components during the
// scene gather pass. Renderer consumes this each frame to populate LightUBO.
// Type matches Light::Type (0 = Directional, 1 = Point, 2 = Spot).
struct Lighting
{
    // The owning scene node's process-unique id — the LOGICAL identity of this light, stable across
    // frames and across the physical shadow-slot reassignment that happens when the active light
    // set changes. Shadow-LOD hysteresis keys on it (SH-03): keyed on a slot instead, one light's
    // dead band would silently be applied to whichever light took its slot next.
    NodeId nodeId{NodeId::Invalid};
    int type{0};
    Vec3 worldPosition{};
    Vec3 worldDirection{};
    Colour3 colour{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};
    float range{0.0f};
    float innerConeCos{1.0f};
    float outerConeCos{0.0f};
};

} // namespace fire_engine
