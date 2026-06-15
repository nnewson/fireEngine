#pragma once

#include <fire_engine/input/input_state.hpp>
#include <fire_engine/platform/keyboard.hpp>
#include <fire_engine/platform/mouse.hpp>
#include <fire_engine/platform/window.hpp>

namespace fire_engine
{

class Input
{
public:
    Input() = default;
    ~Input() = default;

    Input(const Input&) = default;
    Input& operator=(const Input&) = default;
    Input(Input&&) noexcept = default;
    Input& operator=(Input&&) noexcept = default;

    // Non-const: draining the window's accumulated scroll delta mutates it.
    // suppressMouse / suppressKeyboard neutralise the respective input-driven
    // deltas (camera pan/rotate/zoom, WASD/animation/variant) when the debug
    // overlay is capturing that input, so widgets don't also fly the camera.
    [[nodiscard]] InputState update(Window& window, float deltaTime, bool suppressMouse = false,
                                    bool suppressKeyboard = false);

private:
    Keyboard keyboard_;
    Mouse mouse_;
    bool previousVariantKey_{false};

    static constexpr float speed_{10.0f};
    static constexpr float sensitivity_{0.003f};
    static constexpr float panSensitivity_{0.01f};
    static constexpr float zoomSpeed_{1.0f};
};

} // namespace fire_engine
