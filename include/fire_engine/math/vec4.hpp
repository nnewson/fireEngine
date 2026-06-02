#pragma once

#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec_base.hpp>

namespace fire_engine
{

class Vec4 : public VecBase<Vec4, 4>
{
public:
    constexpr Vec4(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f) noexcept
    {
        data_[0] = x;
        data_[1] = y;
        data_[2] = z;
        data_[3] = w;
    }

    // Lifts a Vec3 to a homogeneous Vec4 with w=1. Explicit because the
    // shape change is semantically meaningful — losing or gaining a w
    // component shouldn't happen implicitly via vec4 + vec3 etc.
    constexpr explicit Vec4(const Vec3& vec3) noexcept
    {
        data_[0] = vec3.x();
        data_[1] = vec3.y();
        data_[2] = vec3.z();
        data_[3] = 1.0f;
    }

    ~Vec4() = default;

    Vec4(const Vec4&) = default;
    Vec4& operator=(const Vec4&) = default;
    Vec4(Vec4&&) noexcept = default;
    Vec4& operator=(Vec4&&) noexcept = default;

    [[nodiscard]]
    constexpr float x() const noexcept
    {
        return data_[0];
    }

    constexpr void x(float x) noexcept
    {
        data_[0] = x;
    }

    [[nodiscard]]
    constexpr float y() const noexcept
    {
        return data_[1];
    }

    constexpr void y(float y) noexcept
    {
        data_[1] = y;
    }

    [[nodiscard]]
    constexpr float z() const noexcept
    {
        return data_[2];
    }

    constexpr void z(float z) noexcept
    {
        data_[2] = z;
    }

    [[nodiscard]]
    constexpr float w() const noexcept
    {
        return data_[3];
    }

    constexpr void w(float w) noexcept
    {
        data_[3] = w;
    }

    // Drops the w component. Explicit because silent truncation is exactly
    // the bug we want to avoid (e.g. vec3 = matrix * vec4 quietly losing
    // the homogeneous coordinate).
    [[nodiscard]]
    constexpr explicit operator Vec3() const noexcept
    {
        return {data_[0], data_[1], data_[2]};
    }
};

} // namespace fire_engine
