#pragma once

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// World-space resolved emitter data produced by ParticleEmitter components during
// the scene gather pass. The renderer's ParticleSystem consumes one of these per
// active emitter each frame to drive GPU simulation. Vulkan-free — mirrors
// graphics/lighting.hpp (the Light → Lighting gather analogue).
struct EmitterState
{
    Vec3 worldPosition{};
    // Mean spawn velocity (world space). Particles are emitted in a cone of
    // half-angle coneAngle around this direction, at this speed.
    Vec3 baseVelocity{0.0f, 4.0f, 0.0f};
    float coneAngle{0.35f};  // radians
    float lifetime{2.5f};    // seconds
    float size{0.06f};       // world-space billboard half-size
    float spawnRate{600.0f}; // particles per second
    float gravity{-6.0f};    // m/s^2 applied to velocity.y
    Colour3 colour{1.0f, 0.55f, 0.15f};
    float intensity{4.0f}; // HDR colour multiplier (drives bloom)
};

} // namespace fire_engine
