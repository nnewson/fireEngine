#include <fire_engine/platform/keyboard.hpp>

#include <array>
#include <cstddef>

#include <fire_engine/platform/window.hpp>

namespace fire_engine
{

namespace
{

// GLFW keycode for each Key, indexed by enum order. Must stay in lockstep with
// enum Key in keyboard.hpp; the array size is tied to Key::Count so a mismatched
// element count is a compile error.
constexpr std::array<int, static_cast<std::size_t>(Key::Count)> kGlfwKeyCodes{
    GLFW_KEY_ESCAPE, GLFW_KEY_W,          GLFW_KEY_S,           GLFW_KEY_A,    GLFW_KEY_D,
    GLFW_KEY_E,      GLFW_KEY_F,          GLFW_KEY_V,           GLFW_KEY_1,    GLFW_KEY_2,
    GLFW_KEY_3,      GLFW_KEY_LEFT_SHIFT, GLFW_KEY_RIGHT_SHIFT, GLFW_KEY_LEFT, GLFW_KEY_RIGHT,
};

} // namespace

void Keyboard::poll(const Window& window)
{
    GLFWwindow* w = window.handle();
    for (std::size_t i = 0; i < kGlfwKeyCodes.size(); ++i)
    {
        pressed_[i] = glfwGetKey(w, kGlfwKeyCodes[i]) == GLFW_PRESS;
    }
}

} // namespace fire_engine
