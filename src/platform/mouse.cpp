#include <fire_engine/platform/mouse.hpp>

#include <fire_engine/platform/window.hpp>

namespace fire_engine
{

void Mouse::poll(const Window& window)
{
    GLFWwindow* w = window.handle();
    double currentX = 0.0;
    double currentY = 0.0;
    glfwGetCursorPos(w, &currentX, &currentY);

    if (firstMouse_)
    {
        lastX_ = currentX;
        lastY_ = currentY;
        firstMouse_ = false;
    }

    deltaX_ = currentX - lastX_;
    deltaY_ = currentY - lastY_;

    lastX_ = currentX;
    lastY_ = currentY;

    leftButton_ = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    rightButton_ = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
}

void Mouse::registerScrollCallback(const Window& window)
{
    glfwSetScrollCallback(window.handle(), scrollCallback);
}

void Mouse::scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset)
{
    scrollAccumulator_ += yoffset;
}

} // namespace fire_engine
