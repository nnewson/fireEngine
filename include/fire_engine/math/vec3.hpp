#pragma once

#include <fire_engine/math/vec_base.hpp>

namespace fire_engine
{

class Vec3 : public VecBase<Vec3, 3>
{
public:
    constexpr Vec3() noexcept = default;

    constexpr Vec3(float x, float y, float z) noexcept
    {
        data_[0] = x;
        data_[1] = y;
        data_[2] = z;
    }

    ~Vec3() = default;

    Vec3(const Vec3&) = default;
    Vec3& operator=(const Vec3&) = default;
    Vec3(Vec3&&) noexcept = default;
    Vec3& operator=(Vec3&&) noexcept = default;

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
    static constexpr Vec3 crossProduct(const Vec3& lhs, const Vec3& rhs) noexcept
    {
        return {lhs.data_[1] * rhs.data_[2] - lhs.data_[2] * rhs.data_[1],
                lhs.data_[2] * rhs.data_[0] - lhs.data_[0] * rhs.data_[2],
                lhs.data_[0] * rhs.data_[1] - lhs.data_[1] * rhs.data_[0]};
    }

    [[nodiscard]]
    constexpr Vec3 crossProduct(const Vec3& rhs) const noexcept
    {
        return Vec3::crossProduct(*this, rhs);
    }
};

} // namespace fire_engine
