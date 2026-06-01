#include <fire_engine/platform/keyboard.hpp>

#include <fire_engine/platform/window.hpp>

namespace fire_engine
{

void Keyboard::poll(const Window& window)
{
    GLFWwindow* w = window.handle();
    escape_ = glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    w_ = glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS;
    s_ = glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS;
    a_ = glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS;
    d_ = glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS;
    e_ = glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS;
    f_ = glfwGetKey(w, GLFW_KEY_F) == GLFW_PRESS;
    v_ = glfwGetKey(w, GLFW_KEY_V) == GLFW_PRESS;
    one_ = glfwGetKey(w, GLFW_KEY_1) == GLFW_PRESS;
    two_ = glfwGetKey(w, GLFW_KEY_2) == GLFW_PRESS;
    three_ = glfwGetKey(w, GLFW_KEY_3) == GLFW_PRESS;
    leftShift_ = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
    rightShift_ = glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    left_ = glfwGetKey(w, GLFW_KEY_LEFT) == GLFW_PRESS;
    right_ = glfwGetKey(w, GLFW_KEY_RIGHT) == GLFW_PRESS;
}

} // namespace fire_engine
