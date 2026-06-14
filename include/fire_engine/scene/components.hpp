#pragma once

#include <string_view>
#include <type_traits>
#include <variant>

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/scene/animator.hpp>
#include <fire_engine/scene/camera.hpp>
#include <fire_engine/scene/empty.hpp>
#include <fire_engine/scene/light.hpp>
#include <fire_engine/scene/mesh.hpp>
#include <fire_engine/scene/particle_emitter.hpp>

namespace fire_engine
{

using Components = std::variant<Empty, Animator, Camera, Mesh, Light, ParticleEmitter>;

[[nodiscard]] inline std::string_view componentName(const Components& component) noexcept
{
    return std::visit(
        [](const auto& c) -> std::string_view
        {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, Empty>)
            {
                return "Empty";
            }
            else if constexpr (std::is_same_v<T, Animator>)
            {
                return "Animator";
            }
            else if constexpr (std::is_same_v<T, Camera>)
            {
                return "Camera";
            }
            else if constexpr (std::is_same_v<T, Mesh>)
            {
                return "Mesh";
            }
            else if constexpr (std::is_same_v<T, Light>)
            {
                return "Light";
            }
            else if constexpr (std::is_same_v<T, ParticleEmitter>)
            {
                return "ParticleEmitter";
            }
        },
        component);
}

[[nodiscard]] inline Mat4 componentModelMatrix(const Components& component) noexcept
{
    return std::visit(
        [](const auto& c) -> Mat4
        {
            if constexpr (requires { c.modelMatrix(); })
            {
                return c.modelMatrix();
            }
            else
            {
                return Mat4::identity();
            }
        },
        component);
}

} // namespace fire_engine
