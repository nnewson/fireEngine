#include <fire_engine/render/debug_overlay.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <fire_engine/platform/window.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

DebugOverlay::DebugOverlay(const Device& device, const Swapchain& swapchain, const Window& window,
                           bool startVisible)
    : visible_{startVisible}
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr; // don't litter an imgui.ini next to the binary

    ImGui_ImplGlfw_InitForVulkan(window.handle(), true); // true = chain (not clobber) callbacks

    // Dynamic-rendering setup: ImGui_ImplVulkan_Init builds its pipeline against
    // the swapchain colour format (no VkRenderPass). The format pointer only has
    // to outlive the Init call below.
    const VkFormat swapFormat = static_cast<VkFormat>(swapchain.format());
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapFormat;

    const auto imageCount = static_cast<uint32_t>(swapchain.images().size());
    ImGui_ImplVulkan_InitInfo init{};
    init.ApiVersion = VK_API_VERSION_1_3;
    init.Instance = static_cast<VkInstance>(*device.instance());
    init.PhysicalDevice = static_cast<VkPhysicalDevice>(*device.physicalDevice());
    init.Device = static_cast<VkDevice>(*device.device());
    init.QueueFamily = device.graphicsFamily();
    init.Queue = static_cast<VkQueue>(*device.graphicsQueue());
    // Non-zero DescriptorPoolSize lets the backend own its descriptor pool (font
    // atlas + any AddTexture calls), so we don't manage one here. Use a small fixed
    // headroom (the backend asserts > 1).
    init.DescriptorPoolSize = 8;
    init.MinImageCount = imageCount;
    init.ImageCount = imageCount;
    init.UseDynamicRendering = true;
    init.PipelineRenderingCreateInfo = renderingInfo;

    ImGui_ImplVulkan_Init(&init);
}

DebugOverlay::~DebugOverlay()
{
    // Caller (Renderer) waits for device idle before destruction, so the backend
    // Vulkan objects are safe to tear down here.
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void DebugOverlay::beginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void DebugOverlay::buildUi(const FrameStats& stats, RenderTunables& tunables)
{
    // Record the frame time even while hidden so the plot has history on open.
    frameTimes_[frameTimeHead_] = stats.cpuFrameMs;
    frameTimeHead_ = (frameTimeHead_ + 1) % kFrameHistory;

    if (!visible_)
    {
        return;
    }

    static constexpr std::array<const char*, kProfilePassCount> kPassNames{
        "Shadow", "Forward", "Transmission", "TAA", "Particles", "Bloom", "Post"};

    ImGui::Begin("Fire Engine - Debug");

    const float fps = stats.cpuFrameMs > 0.0f ? 1000.0f / stats.cpuFrameMs : 0.0f;
    ImGui::Text("CPU frame: %.2f ms (%.0f FPS)", stats.cpuFrameMs, fps);
    ImGui::PlotLines("##frametime", frameTimes_.data(), kFrameHistory, frameTimeHead_, nullptr,
                     0.0f, 33.3f, ImVec2(0.0f, 50.0f));

    ImGui::Separator();
    if (stats.gpuValid)
    {
        ImGui::Text("GPU passes (ms):");
        if (ImGui::BeginTable("gpu", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
        {
            for (uint32_t p = 0; p < kProfilePassCount; ++p)
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(kPassNames[p]);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", stats.passMs[p]);
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("Total");
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.gpuTotalMs);
            ImGui::EndTable();
        }
    }
    else
    {
        ImGui::TextDisabled("GPU timestamps unavailable");
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("TAA", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Enabled", &tunables.taaEnabled);
        ImGui::BeginDisabled(!tunables.taaEnabled);
        ImGui::SliderFloat("History blend", &tunables.taaHistoryBlend, 0.0f, 0.98f);
        ImGui::SliderFloat("Sharpen", &tunables.taaSharpen, 0.0f, 1.0f);
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Debug view", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static constexpr const char* kViews[] = {"None",   "Normals",      "N·L",
                                                 "Shadow", "Shadow depth", "Velocity"};
        int view = static_cast<int>(tunables.debugView);
        if (ImGui::Combo("View", &view, kViews, IM_ARRAYSIZE(kViews)))
        {
            tunables.debugView = static_cast<DebugView>(view);
        }
        ImGui::Checkbox("No shadows", &tunables.noShadows);
    }

    if (ImGui::CollapsingHeader("Lighting / Post"))
    {
        ImGui::SliderFloat("Bloom", &tunables.bloomStrength, 0.0f, 0.2f);
        ImGui::SliderFloat("Diffuse IBL", &tunables.diffuseIbl, 0.0f, 2.0f);
        ImGui::SliderFloat("Specular IBL", &tunables.specularIbl, 0.0f, 2.0f);
        ImGui::SliderFloat("Sun intensity", &tunables.directionalIntensityScale, 0.0f, 4.0f);
    }

    if (ImGui::CollapsingHeader("Particles"))
    {
        ImGui::SliderFloat("Rate", &tunables.particleRateScale, 0.0f, 4.0f);
        ImGui::SliderFloat("Lifetime", &tunables.particleLifetimeScale, 0.1f, 4.0f);
        ImGui::SliderFloat("Size", &tunables.particleSizeScale, 0.1f, 4.0f);
    }

    if (ImGui::CollapsingHeader("Cloth"))
    {
        ImGui::SliderInt("Substeps", &tunables.clothSubsteps, 1, 40);
        // Global multiplier on each constraint's authored (per-type) compliance:
        // 1.0 = as authored, lower = stiffer, higher = softer.
        ImGui::SliderFloat("Compliance x", &tunables.clothComplianceScale, 0.0f, 8.0f, "%.2f");
        ImGui::SliderFloat("Damping", &tunables.clothDamping, 0.8f, 1.0f);
        ImGui::SliderFloat("Gravity", &tunables.clothGravity, -20.0f, 0.0f);
        ImGui::SliderFloat3("Wind", tunables.clothWind, -10.0f, 10.0f);
    }

    ImGui::End();
}

void DebugOverlay::record(vk::CommandBuffer cmd, vk::ImageView swapView, vk::Extent2D extent)
{
    ImGui::Render();

    vk::RenderingAttachmentInfo colour{
        .imageView = swapView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    vk::Rect2D area{.offset = vk::Offset2D{.x = 0, .y = 0}, .extent = extent};
    cmd.beginRendering(makeRenderingInfo(area, {&colour, 1}, nullptr));
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), static_cast<VkCommandBuffer>(cmd));
    cmd.endRendering();
}

bool DebugOverlay::wantsMouse() const noexcept
{
    return visible_ && ImGui::GetIO().WantCaptureMouse;
}

bool DebugOverlay::wantsKeyboard() const noexcept
{
    return visible_ && ImGui::GetIO().WantCaptureKeyboard;
}

} // namespace fire_engine
