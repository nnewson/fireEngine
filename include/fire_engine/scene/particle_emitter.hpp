#pragma once

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/graphics/particle.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

// Scene component for a GPU particle emitter. Transform-driven like Light: it
// holds authored emission parameters and is *gathered* (not drawn) each frame
// into a world-space EmitterState that the renderer's ParticleSystem consumes.
class ParticleEmitter
{
public:
    // Pure: resolve an emitter + node world matrix into world-space EmitterState.
    // Position is the translation column; baseVelocity is rotated by the node's
    // upper-3x3 so an emitter can be aimed via its node orientation.
    [[nodiscard]] static EmitterState toEmitterState(const ParticleEmitter& emitter,
                                                     const Mat4& world) noexcept;

    ParticleEmitter() = default;
    ~ParticleEmitter() = default;

    ParticleEmitter(const ParticleEmitter&) = default;
    ParticleEmitter& operator=(const ParticleEmitter&) = default;
    ParticleEmitter(ParticleEmitter&&) noexcept = default;
    ParticleEmitter& operator=(ParticleEmitter&&) noexcept = default;

    // No-op; emitters are transform-driven and resolved at gather time. Present
    // so the component satisfies the scenegraph's update visitor (like Light).
    void update(const InputState& input_state, const Transform& transform);

    [[nodiscard]] Vec3 baseVelocity() const noexcept
    {
        return baseVelocity_;
    }
    void baseVelocity(Vec3 v) noexcept
    {
        baseVelocity_ = v;
    }

    [[nodiscard]] float coneAngle() const noexcept
    {
        return coneAngle_;
    }
    void coneAngle(float radians) noexcept
    {
        coneAngle_ = radians;
    }

    [[nodiscard]] float lifetime() const noexcept
    {
        return lifetime_;
    }
    void lifetime(float seconds) noexcept
    {
        lifetime_ = seconds;
    }

    [[nodiscard]] float size() const noexcept
    {
        return size_;
    }
    void size(float halfSize) noexcept
    {
        size_ = halfSize;
    }

    [[nodiscard]] float spawnRate() const noexcept
    {
        return spawnRate_;
    }
    void spawnRate(float perSecond) noexcept
    {
        spawnRate_ = perSecond;
    }

    [[nodiscard]] float gravity() const noexcept
    {
        return gravity_;
    }
    void gravity(float g) noexcept
    {
        gravity_ = g;
    }

    [[nodiscard]] Colour3 colour() const noexcept
    {
        return colour_;
    }
    void colour(Colour3 c) noexcept
    {
        colour_ = c;
    }

    [[nodiscard]] float intensity() const noexcept
    {
        return intensity_;
    }
    void intensity(float i) noexcept
    {
        intensity_ = i;
    }

private:
    Vec3 baseVelocity_{0.0f, 4.0f, 0.0f};
    float coneAngle_{0.35f};
    float lifetime_{2.5f};
    float size_{0.06f};
    float spawnRate_{600.0f};
    float gravity_{-6.0f};
    Colour3 colour_{1.0f, 0.55f, 0.15f};
    float intensity_{4.0f};
};

} // namespace fire_engine
