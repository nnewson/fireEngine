#pragma once

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

class Light
{
public:
    enum class Type
    {
        Directional,
        Point,
        Spot,
    };

    // Pure: resolve a Light + node world matrix into the world-space data the
    // renderer actually wants. Position is the translation column; direction
    // is the node's local -Z transformed and renormalised (matches the
    // KHR_lights_punctual convention).
    [[nodiscard]] static Lighting toLighting(const Light& light, const Mat4& world) noexcept;

    Light() = default;
    ~Light() = default;

    Light(const Light&) = default;
    Light& operator=(const Light&) = default;
    Light(Light&&) noexcept = default;
    Light& operator=(Light&&) noexcept = default;

    void update(const InputState& input_state, const Transform& transform);

    [[nodiscard]]
    Type type() const noexcept
    {
        return type_;
    }

    void type(Type t) noexcept
    {
        type_ = t;
    }

    [[nodiscard]]
    Colour3 colour() const noexcept
    {
        return colour_;
    }

    void colour(Colour3 c) noexcept
    {
        colour_ = c;
    }

    [[nodiscard]]
    float intensity() const noexcept
    {
        return intensity_;
    }

    void intensity(float i) noexcept
    {
        intensity_ = i;
    }

    [[nodiscard]]
    float range() const noexcept
    {
        return range_;
    }

    void range(float r) noexcept
    {
        range_ = r;
    }

    [[nodiscard]]
    float innerConeRad() const noexcept
    {
        return innerConeRad_;
    }

    void innerConeRad(float rad) noexcept
    {
        innerConeRad_ = rad;
        if (innerConeRad_ > outerConeRad_)
        {
            outerConeRad_ = innerConeRad_;
        }
    }

    [[nodiscard]]
    float outerConeRad() const noexcept
    {
        return outerConeRad_;
    }

    void outerConeRad(float rad) noexcept
    {
        outerConeRad_ = rad;
        if (outerConeRad_ < innerConeRad_)
        {
            innerConeRad_ = outerConeRad_;
        }
    }

private:
    Type type_{Type::Directional};
    Colour3 colour_{1.0f, 1.0f, 1.0f};
    float intensity_{1.0f};
    float range_{0.0f};
    float innerConeRad_{pi / 8.0f};
    float outerConeRad_{pi / 4.0f};
};

} // namespace fire_engine
