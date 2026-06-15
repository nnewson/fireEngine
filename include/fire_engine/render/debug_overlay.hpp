#pragma once

#include <array>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/render/gpu_profiler.hpp>
#include <fire_engine/render/render_tunables.hpp>

namespace fire_engine
{

class Device;
class Swapchain;
class Window;

// Dear ImGui debug overlay. Owns the ImGui context + the GLFW platform backend
// and the Vulkan renderer backend (dynamic rendering), and draws an ImGui frame
// into the swapchain image (loadOp Load) after post-process and before present.
// Hidden by default; toggled with F1. ImGui keeps global state, so this type is
// a non-movable singleton-in-practice owned by the Renderer.
class DebugOverlay
{
public:
    DebugOverlay(const Device& device, const Swapchain& swapchain, const Window& window,
                 bool startVisible);
    ~DebugOverlay();

    DebugOverlay(const DebugOverlay&) = delete;
    DebugOverlay& operator=(const DebugOverlay&) = delete;
    DebugOverlay(DebugOverlay&&) = delete;
    DebugOverlay& operator=(DebugOverlay&&) = delete;

    // Begin a new ImGui frame. Always called (even when hidden) so the
    // NewFrame/Render pairing stays balanced.
    void beginFrame();
    // Build the overlay panels from the latest frame stats; widgets write back
    // into `tunables`. No-op (apart from frame-time history) when hidden.
    void buildUi(const FrameStats& stats, RenderTunables& tunables);
    // Render the ImGui draw data into the swap image (must already be in
    // ColorAttachmentOptimal). Draws nothing when hidden (empty draw list).
    void record(vk::CommandBuffer cmd, vk::ImageView swapView, vk::Extent2D extent);

    void toggle() noexcept
    {
        visible_ = !visible_;
    }

    // True only while visible AND ImGui wants the corresponding input — the main
    // loop uses these to suppress camera movement while the user drives widgets.
    [[nodiscard]] bool wantsMouse() const noexcept;
    [[nodiscard]] bool wantsKeyboard() const noexcept;

private:
    bool visible_{false};
    // Rolling history of CPU frame times (ms) for the overlay plot.
    static constexpr int kFrameHistory = 120;
    std::array<float, kFrameHistory> frameTimes_{};
    int frameTimeHead_{0};
};

} // namespace fire_engine
