#include <fire_engine/scene/particle_emitter.hpp>

namespace fire_engine
{

EmitterState ParticleEmitter::toEmitterState(const ParticleEmitter& emitter,
                                             const Mat4& world) noexcept
{
    EmitterState state;
    state.worldPosition = Vec3{world[0, 3], world[1, 3], world[2, 3]};

    // Rotate the authored local velocity by the node's upper-3x3 (direction
    // only) so the emitter can be aimed by orienting its node.
    const Vec3 v = emitter.baseVelocity_;
    state.baseVelocity = Vec3{world[0, 0] * v.x() + world[0, 1] * v.y() + world[0, 2] * v.z(),
                              world[1, 0] * v.x() + world[1, 1] * v.y() + world[1, 2] * v.z(),
                              world[2, 0] * v.x() + world[2, 1] * v.y() + world[2, 2] * v.z()};

    state.coneAngle = emitter.coneAngle_;
    state.lifetime = emitter.lifetime_;
    state.size = emitter.size_;
    state.spawnRate = emitter.spawnRate_;
    state.gravity = emitter.gravity_;
    state.colour = emitter.colour_;
    state.intensity = emitter.intensity_;
    return state;
}

void ParticleEmitter::update(const InputState& /*input_state*/, const Transform& /*transform*/)
{
}

} // namespace fire_engine
