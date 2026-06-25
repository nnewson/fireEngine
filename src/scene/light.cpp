#include <cmath>
#include <utility>

#include <fire_engine/input/input_state.hpp>
#include <fire_engine/scene/light.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

Lighting Light::toLighting(const Light& light, const Mat4& world) noexcept
{
    Lighting inst;
    inst.type = std::to_underlying(light.type_);
    inst.worldPosition = Vec3{world[0, 3], world[1, 3], world[2, 3]};

    // KHR_lights_punctual: light forward is the node's local -Z transformed
    // by its world matrix. -Z column of upper-3x3 = -(world[*, 2]).
    Vec3 forward{-world[0, 2], -world[1, 2], -world[2, 2]};
    if (forward.magnitudeSquared() > float_epsilon)
    {
        forward.normalise();
    }
    else
    {
        forward = Vec3{0.0f, 0.0f, -1.0f};
    }
    inst.worldDirection = forward;

    inst.colour = light.colour_;
    inst.intensity = light.intensity_;
    inst.range = light.range_;
    inst.innerConeCos = std::cos(light.innerConeRad_);
    inst.outerConeCos = std::cos(light.outerConeRad_);
    return inst;
}

void Light::update(const InputState& /*input_state*/, const Transform& /*transform*/)
{
}

} // namespace fire_engine
