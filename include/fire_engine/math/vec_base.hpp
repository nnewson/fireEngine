#pragma once

#include <cmath>
#include <cstddef>

#include <fire_engine/math/constants.hpp>

namespace fire_engine
{

template <typename Derived, std::size_t N>
class VecBase
{
public:
    [[nodiscard]]
    constexpr Derived operator-(const Derived& rhs) const noexcept
    {
        Derived result{self()};
        result -= rhs;
        return result;
    }

    constexpr Derived& operator-=(const Derived& rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            data_[i] -= rhs.data_[i];
        }
        return self();
    }

    [[nodiscard]]
    constexpr Derived operator+(const Derived& rhs) const noexcept
    {
        Derived result{self()};
        result += rhs;
        return result;
    }

    constexpr Derived& operator+=(const Derived& rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            data_[i] += rhs.data_[i];
        }
        return self();
    }

    [[nodiscard]]
    constexpr Derived operator*(const float rhs) const noexcept
    {
        Derived result{self()};
        result *= rhs;
        return result;
    }

    constexpr Derived& operator*=(const float rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            data_[i] *= rhs;
        }
        return self();
    }

    [[nodiscard]]
    constexpr Derived operator/(const float rhs) const noexcept
    {
        Derived result{self()};
        result /= rhs;
        return result;
    }

    constexpr Derived& operator/=(const float rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            data_[i] /= rhs;
        }
        return self();
    }

    [[nodiscard]]
    static constexpr float dotProduct(const Derived& lhs, const Derived& rhs) noexcept
    {
        float sum = 0.0f;
        for (std::size_t i = 0; i < N; ++i)
        {
            sum += lhs.data_[i] * rhs.data_[i];
        }
        return sum;
    }

    [[nodiscard]]
    constexpr float dotProduct(const Derived& rhs) const noexcept
    {
        return Derived::dotProduct(self(), rhs);
    }

    // std::sqrt is not constexpr before C++26, so magnitude (and the
    // normalise helpers that depend on it) cannot be constexpr either.
    [[nodiscard]] float magnitude() const noexcept
    {
        return std::sqrt(magnitudeSquared());
    }

    [[nodiscard]] constexpr float magnitudeSquared() const noexcept
    {
        float sum = 0.0f;
        for (std::size_t i = 0; i < N; ++i)
        {
            sum += data_[i] * data_[i];
        }
        return sum;
    }

    [[nodiscard]]
    static Derived normalise(const Derived& v) noexcept
    {
        float len = v.magnitude();
        if (len < float_epsilon)
        {
            return Derived{};
        }

        Derived result{v};
        for (std::size_t i = 0; i < N; ++i)
        {
            result.data_[i] /= len;
        }
        return result;
    }

    Derived& normalise() noexcept
    {
        self() = normalise(self());
        return self();
    }

    // Strict bit-for-bit equality. Two values that differ by a single ULP
    // compare not-equal — use approxEqual when you want tolerance.
    [[nodiscard]]
    friend constexpr bool operator==(const Derived& lhs, const Derived& rhs) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            if (lhs.data_[i] != rhs.data_[i])
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]]
    constexpr bool bitwiseEqual(const Derived& rhs) const noexcept
    {
        return self() == rhs;
    }

    [[nodiscard]]
    constexpr bool approxEqual(const Derived& rhs, float eps = float_epsilon) const noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            const float diff = data_[i] - rhs.data_[i];
            if (diff > eps || diff < -eps)
            {
                return false;
            }
        }
        return true;
    }

protected:
    float data_[N]{};

private:
    [[nodiscard]] constexpr Derived& self() noexcept
    {
        return static_cast<Derived&>(*this);
    }

    [[nodiscard]] constexpr const Derived& self() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }
};

} // namespace fire_engine
