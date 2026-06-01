#pragma once

#include <string_view>
#include <utility>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan_raii.hpp>

namespace fire_engine
{

class Window
{
public:
    explicit Window(size_t width, size_t height, std::string_view title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&) noexcept = default;
    Window& operator=(Window&&) noexcept = default;

    [[nodiscard]]
    bool shouldClose() const noexcept
    {
        return glfwWindowShouldClose(window_);
    }

    [[nodiscard]]
    GLFWwindow* handle() const noexcept
    {
        return window_;
    }

    [[nodiscard]]
    bool framebufferResized() const noexcept
    {
        return framebufferResized_;
    }

    void framebufferResized(bool resized) noexcept
    {
        framebufferResized_ = resized;
    }

    static void pollEvents();
    static void waitEvents();

    [[nodiscard]]
    std::pair<int, int> framebufferSize() const;

    [[nodiscard]]
    static std::vector<const char*> requiredVulkanExtensions();

    [[nodiscard]]
    vk::raii::SurfaceKHR createVulkanSurface(const vk::raii::Instance& instance) const;

private:
    GLFWwindow* window_ = nullptr;
    bool framebufferResized_ = false;
};

} // namespace fire_engine
