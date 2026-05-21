#pragma once

namespace fire_engine
{

class Window;

class Keyboard
{
public:
    Keyboard() = default;
    ~Keyboard() = default;

    Keyboard(const Keyboard&) = default;
    Keyboard& operator=(const Keyboard&) = default;
    Keyboard(Keyboard&&) noexcept = default;
    Keyboard& operator=(Keyboard&&) noexcept = default;

    void poll(const Window& window);

    [[nodiscard]] bool escape() const noexcept
    {
        return escape_;
    }
    [[nodiscard]] bool w() const noexcept
    {
        return w_;
    }
    [[nodiscard]] bool s() const noexcept
    {
        return s_;
    }
    [[nodiscard]] bool a() const noexcept
    {
        return a_;
    }
    [[nodiscard]] bool d() const noexcept
    {
        return d_;
    }
    [[nodiscard]] bool e() const noexcept
    {
        return e_;
    }
    [[nodiscard]] bool f() const noexcept
    {
        return f_;
    }
    [[nodiscard]] bool v() const noexcept
    {
        return v_;
    }
    [[nodiscard]] bool one() const noexcept
    {
        return one_;
    }
    [[nodiscard]] bool two() const noexcept
    {
        return two_;
    }
    [[nodiscard]] bool three() const noexcept
    {
        return three_;
    }
    [[nodiscard]] bool shift() const noexcept
    {
        return leftShift_ || rightShift_;
    }
    [[nodiscard]] bool left() const noexcept
    {
        return left_;
    }
    [[nodiscard]] bool right() const noexcept
    {
        return right_;
    }

private:
    bool escape_{false};
    bool w_{false};
    bool s_{false};
    bool a_{false};
    bool d_{false};
    bool e_{false};
    bool f_{false};
    bool v_{false};
    bool one_{false};
    bool two_{false};
    bool three_{false};
    bool leftShift_{false};
    bool rightShift_{false};
    bool left_{false};
    bool right_{false};
};

} // namespace fire_engine
