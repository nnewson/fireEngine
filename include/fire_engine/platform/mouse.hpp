#pragma once

struct GLFWwindow;

namespace fire_engine
{

class Window;

class Mouse
{
public:
    Mouse() = default;
    ~Mouse() = default;

    Mouse(const Mouse&) = default;
    Mouse& operator=(const Mouse&) = default;
    Mouse(Mouse&&) noexcept = default;
    Mouse& operator=(Mouse&&) noexcept = default;

    void poll(const Window& window);

    void registerScrollCallback(const Window& window);

    [[nodiscard]] double consumeScrollDelta() noexcept
    {
        double delta = scrollAccumulator_;
        scrollAccumulator_ = 0.0;
        return delta;
    }

    [[nodiscard]] double x() const noexcept
    {
        return lastX_;
    }
    [[nodiscard]] double y() const noexcept
    {
        return lastY_;
    }
    [[nodiscard]] double deltaX() const noexcept
    {
        return deltaX_;
    }
    [[nodiscard]] double deltaY() const noexcept
    {
        return deltaY_;
    }
    [[nodiscard]] bool leftButton() const noexcept
    {
        return leftButton_;
    }
    [[nodiscard]] bool rightButton() const noexcept
    {
        return rightButton_;
    }

private:
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    static inline double scrollAccumulator_{0.0};

    double lastX_{0.0};
    double lastY_{0.0};
    double deltaX_{0.0};
    double deltaY_{0.0};
    bool firstMouse_{true};
    bool leftButton_{false};
    bool rightButton_{false};
};

} // namespace fire_engine
