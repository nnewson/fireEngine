#pragma once

#include <array>
#include <cstddef>

namespace fire_engine
{

class Window;

// Logical keys the engine polls each frame. Order is the single source of truth
// for the pressed-state array and the GLFW keycode table in keyboard.cpp — keep
// the two in lockstep. `Count` must stay last so it sizes the array.
enum class Key : std::size_t
{
    Escape,
    W,
    S,
    A,
    D,
    E,
    F,
    V,
    One,
    Two,
    Three,
    LeftShift,
    RightShift,
    Left,
    Right,
    Count,
};

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

    [[nodiscard]] bool pressed(Key key) const noexcept
    {
        return pressed_[static_cast<std::size_t>(key)];
    }

    // Either shift counts as "shift held" for modifier checks.
    [[nodiscard]] bool shift() const noexcept
    {
        return pressed(Key::LeftShift) || pressed(Key::RightShift);
    }

private:
    std::array<bool, static_cast<std::size_t>(Key::Count)> pressed_{};
};

} // namespace fire_engine
