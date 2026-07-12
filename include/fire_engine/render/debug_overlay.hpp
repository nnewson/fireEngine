#pragma once

#include <array>
#include <span>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/render/debug_draw.hpp>
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
    // Draw world-anchored text labels (ragdoll joint index:name) into the ImGui foreground draw
    // list, projected to screen via `viewProj`. Off-screen / behind-camera labels are culled. Call
    // between beginFrame and record (like buildUi) — and after the frame's viewProj is finalised,
    // so labels track the joints as the camera moves. No-op when the overlay is hidden. Screen
    // mapping uses ImGui's DisplaySize (logical points), not the swapchain pixel extent, so labels
    // land on the geometry on high-DPI/retina displays where the two differ.
    void drawWorldLabels(std::span<const DebugLabel> labels, const Mat4& viewProj);

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
