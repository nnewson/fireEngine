#pragma once

#include <fire_engine/math/vec_base.hpp>

namespace fire_engine
{

class Vec2 : public VecBase<Vec2, 2>
{
public:
    constexpr Vec2() noexcept = default;

    constexpr Vec2(float s, float t) noexcept
    {
        data_[0] = s;
        data_[1] = t;
    }

    ~Vec2() = default;

    Vec2(const Vec2&) = default;
    Vec2& operator=(const Vec2&) = default;
    Vec2(Vec2&&) noexcept = default;
    Vec2& operator=(Vec2&&) noexcept = default;

    [[nodiscard]]
    constexpr float s() const noexcept
    {
        return data_[0];
    }

    constexpr void s(float s) noexcept
    {
        data_[0] = s;
    }

    [[nodiscard]]
    constexpr float t() const noexcept
    {
        return data_[1];
    }

    constexpr void t(float t) noexcept
    {
        data_[1] = t;
    }
};

} // namespace fire_engine
