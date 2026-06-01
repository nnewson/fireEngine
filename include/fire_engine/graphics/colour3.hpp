#pragma once

namespace fire_engine
{

class Colour3
{
public:
    constexpr Colour3(float r = 0.0f, float g = 0.0f, float b = 0.0f) noexcept
        : r_(r),
          g_(g),
          b_(b)
    {
    }

    ~Colour3() = default;

    Colour3(const Colour3&) = default;
    Colour3& operator=(const Colour3&) = default;
    Colour3(Colour3&&) noexcept = default;
    Colour3& operator=(Colour3&&) noexcept = default;

    [[nodiscard]]
    constexpr float r() const noexcept
    {
        return r_;
    }

    constexpr void r(float r) noexcept
    {
        r_ = r;
    }

    [[nodiscard]]
    constexpr float g() const noexcept
    {
        return g_;
    }

    constexpr void g(float g) noexcept
    {
        g_ = g;
    }

    [[nodiscard]]
    constexpr float b() const noexcept
    {
        return b_;
    }

    constexpr void b(float b) noexcept
    {
        b_ = b;
    }

    [[nodiscard]]
    constexpr bool operator==(const Colour3& other) const noexcept
    {
        return r_ == other.r_ && g_ == other.g_ && b_ == other.b_;
    }

private:
    float r_;
    float g_;
    float b_;
};

} // namespace fire_engine
