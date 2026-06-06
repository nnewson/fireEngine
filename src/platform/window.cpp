#include <fire_engine/platform/window.hpp>

#include <stdexcept>

namespace fire_engine
{

Window::Window(size_t width, size_t height, std::string_view title)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title.data(),
                               nullptr, nullptr);
    if (window_ == nullptr)
    {
        throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_,
                                   [](GLFWwindow* w, int, int)
                                   {
                                       auto* self =
                                           static_cast<Window*>(glfwGetWindowUserPointer(w));
                                       self->framebufferResized_ = true;
                                   });
    glfwSetScrollCallback(window_,
                          [](GLFWwindow* w, double, double yoffset)
                          {
                              auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
                              self->scrollDelta_ += yoffset;
                          });
}

Window::~Window()
{
    glfwDestroyWindow(window_);
}

void Window::pollEvents()
{
    glfwPollEvents();
}

void Window::waitEvents()
{
    glfwWaitEvents();
}

std::pair<int, int> Window::framebufferSize() const
{
    int w = 0;
    int h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    return {w, h};
}

std::vector<const char*> Window::requiredVulkanExtensions()
{
    uint32_t count = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&count);
    return {exts, exts + count};
}

vk::raii::SurfaceKHR Window::createVulkanSurface(const vk::raii::Instance& instance) const
{
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(*instance, window_, nullptr, &rawSurface) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create window surface");
    }
    return vk::raii::SurfaceKHR(instance, rawSurface);
}

} // namespace fire_engine
