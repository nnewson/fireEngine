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
        "VDPM compute", "Shadow",    "Depth", "SSAO",  "Forward", "Transmission",
        "TAA",          "Particles", "Debug", "Bloom", "Post"};

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
        // `##taa` keeps the visible label "Enabled" but gives a unique ImGui id (a CollapsingHeader
        // opens no id scope, so this would otherwise clash with the Mesh LOD "Enabled" checkbox).
        ImGui::Checkbox("Enabled##taa", &tunables.taaEnabled);
        ImGui::BeginDisabled(!tunables.taaEnabled);
        ImGui::SliderFloat("History blend", &tunables.taaHistoryBlend, 0.0f, 0.98f);
        ImGui::SliderFloat("Sharpen", &tunables.taaSharpen, 0.0f, 1.0f);
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Culling", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Frustum culling", &tunables.cullingEnabled);
        ImGui::BeginDisabled(!tunables.cullingEnabled);
        const int visible = stats.trackedNodes - stats.culledNodes;
        ImGui::Text("Tracked: %d   visible: %d   culled: %d", stats.trackedNodes, visible,
                    stats.culledNodes);
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Mesh LOD", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Enabled##lod", &tunables.lodEnabled);
        ImGui::BeginDisabled(!tunables.lodEnabled);
        static constexpr const char* kLodModes[] = {
            "Discrete (hard swap)", "Continuous (VIPM geomorph)", "View-dependent (VDPM)"};
        int lodMode = static_cast<int>(tunables.lodMode);
        if (ImGui::Combo("Mode", &lodMode, kLodModes, IM_ARRAYSIZE(kLodModes)))
        {
            tunables.lodMode = static_cast<LodMode>(lodMode);
        }
        ImGui::SliderFloat("Pixel error budget", &tunables.lodPixelErrorBudget, 0.25f, 16.0f,
                           "%.2f");
        ImGui::EndDisabled();
        if (stats.trianglesGpuPending)
        {
            ImGui::Text("Triangles drawn: pending GPU readback");
        }
        else
        {
            ImGui::Text("Triangles drawn: %llu%s",
                        static_cast<unsigned long long>(stats.trianglesDrawn),
                        stats.trianglesOverflow ? " (overflow!)" : "");
        }
        if (tunables.lodMode == LodMode::ViewDependent)
        {
            // Per-frame VDPM repair work (vertices each pass pulled back in). CPU-driven fronts
            // only — a GPU-backed front's CPU counters are stale (its lifecycle was skipped), so
            // they are excluded; the GPU-front health summary is separate below.
            ImGui::Text("VDPM repairs (verts, CPU fronts): foldover %d, coverage %d",
                        stats.vdpmFoldoversRepaired, stats.vdpmCoverageRepaired);
            // Per-channel refine attribution: which metric channel won each over-budget trigger,
            // and the largest score/budget ratio each channel reached this frame. The ratios expose
            // an under-firing channel the counts can't — a smooth interior with normal triggers 0
            // but a normal ratio near 1 is a hair under budget; near 0 means the channel is
            // genuinely blind.
            ImGui::Text("VDPM triggers (CPU fronts): geom %d, uv %d, normal %d, tangent %d",
                        stats.vdpmGeometryTriggers, stats.vdpmUvTriggers, stats.vdpmNormalTriggers,
                        stats.vdpmTangentTriggers);
            ImGui::Text("VDPM max score/budget: geom %.2f, uv %.2f, normal %.2f, tangent %.2f",
                        static_cast<double>(stats.vdpmMaxGeometryRatio),
                        static_cast<double>(stats.vdpmMaxUvRatio),
                        static_cast<double>(stats.vdpmMaxNormalRatio),
                        static_cast<double>(stats.vdpmMaxTangentRatio));
            // GPU-driven-front (B5b) command-count instrumentation. The compute cost is dispatch-
            // bound: the analytic total is dominated by the repair term B·(2R+2), so the CPU record
            // time (MoltenVK command translation) tracks it. Compare with the "VDPM compute" GPU
            // pass above. Only populated with --vdpm-gpu.
            if (stats.vdpmFrontsRecorded > 0)
            {
                ImGui::Text("VDPM GPU: %d front(s), %d ranks, repairBudget %d",
                            stats.vdpmFrontsRecorded, stats.vdpmMaxRankCount,
                            stats.vdpmRepairRoundBudget);
                ImGui::Text(
                    "VDPM GPU: record %.3f ms CPU | ~%d dispatches, ~%d barriers (analytic)",
                    static_cast<double>(stats.vdpmRecordCpuMs), stats.vdpmAnalyticDispatches,
                    stats.vdpmAnalyticBarriers);
                // Channel-trigger attribution is CPU-only (the GPU score picks a channel per split
                // but doesn't aggregate it back) — shown n/a for GPU fronts.
                ImGui::TextDisabled("VDPM GPU channel attribution: n/a");
                // Delayed SCENE-WIDE health readback (a few frames late): repair convergence across
                // all GPU repair fronts. A low max-marked vs the budget is the
                // economical-convergence signal; any fallback / non-clean / ancestor / B3-fail
                // count is a health flag.
                if (stats.vdpmMaxMarkedRounds >= 0)
                {
                    ImGui::Text(
                        "VDPM GPU health: %d repair front(s), max %d/%d marked rounds (Σ %d)",
                        stats.vdpmRepairFronts, stats.vdpmMaxMarkedRounds,
                        stats.vdpmRepairRoundBudget, stats.vdpmSumMarkedRounds);
                    ImGui::Text("VDPM GPU flags: fallback %d, non-clean %d, ancestor-fail %d, "
                                "B3-fail %d%s",
                                stats.vdpmFallbackFronts, stats.vdpmNonCleanPrefix,
                                stats.vdpmAncestorFailures, stats.vdpmFailFlagFronts,
                                stats.vdpmEmittedOverflow ? " | EMITTED OVERFLOW" : "");
                }
            }
        }
        ImGui::TextDisabled("View > 'LOD tint' colours by level (green/yellow/red)");
    }

    if (ImGui::CollapsingHeader("Debug view", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static constexpr const char* kViews[] = {"None",   "Normals",      "N·L",
                                                 "Shadow", "Shadow depth", "Velocity",
                                                 "SSAO",   "LOD tint",     "Joints"};
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

    if (ImGui::CollapsingHeader("SSAO / contact shadows"))
    {
        ImGui::Checkbox("SSAO", &tunables.ssaoEnabled);
        ImGui::BeginDisabled(!tunables.ssaoEnabled);
        ImGui::SliderFloat("Radius", &tunables.ssaoRadius, 0.05f, 2.0f, "%.2f");
        ImGui::SliderFloat("Bias", &tunables.ssaoBias, 0.0f, 0.1f, "%.3f");
        ImGui::SliderFloat("Intensity", &tunables.ssaoIntensity, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Power", &tunables.ssaoPower, 0.5f, 4.0f, "%.2f");
        ImGui::EndDisabled();
        ImGui::Checkbox("Contact shadows", &tunables.contactShadowsEnabled);
        ImGui::BeginDisabled(!tunables.contactShadowsEnabled);
        ImGui::SliderFloat("Contact length", &tunables.contactShadowLength, 0.05f, 2.0f, "%.2f");
        // Edge guard: fades contact shadows out at depth silhouettes (kills the
        // screen-space "hair"). Lower = guard more aggressively.
        ImGui::SliderFloat("Contact edge", &tunables.contactEdgeThreshold, 0.02f, 0.5f, "%.3f");
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Physics debug"))
    {
        ImGui::Checkbox("Broadphase AABBs", &tunables.debugDrawAabbs);
        ImGui::Checkbox("Collider shapes", &tunables.debugDrawColliders);
        ImGui::Checkbox("Contacts", &tunables.debugDrawContacts);
        ImGui::Checkbox("Depth-tested (off = x-ray)", &tunables.debugDepthTest);
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

void DebugOverlay::drawWorldLabels(std::span<const DebugLabel> labels, const Mat4& viewProj)
{
    if (!visible_ || labels.empty())
    {
        return;
    }
    ImDrawList* drawList = ImGui::GetForegroundDrawList();
    // The foreground draw list is in ImGui's coordinate space (DisplaySize, logical points), which
    // on retina is the swapchain pixel extent / DisplayFramebufferScale — using the pixel extent
    // here would scale the labels off the geometry as the camera pans.
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    const float w = display.x;
    const float h = display.y;
    for (const DebugLabel& label : labels)
    {
        const Vec3& p = label.worldPosition;
        const Vec4 clip = viewProj * Vec4{p.x(), p.y(), p.z(), 1.0f};
        if (clip.w() <= 1.0e-4f)
        {
            continue; // behind the camera
        }
        const float ndcX = clip.x() / clip.w();
        const float ndcY = clip.y() / clip.w();
        if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f)
        {
            continue; // off-screen
        }
        // Vulkan NDC (y already flipped in the projection) → framebuffer pixels (ImGui's top-left
        // origin), matching where the scene rasterised.
        const ImVec2 screen{(ndcX * 0.5f + 0.5f) * w, (ndcY * 0.5f + 0.5f) * h};
        drawList->AddText(screen, IM_COL32(255, 235, 60, 255), label.text.c_str());
    }
}

void DebugOverlay::record(vk::CommandBuffer cmd, vk::ImageView swapView, vk::Extent2D extent)
{
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!visible_ || drawData == nullptr || drawData->TotalVtxCount == 0)
    {
        return;
    }

    vk::RenderingAttachmentInfo colour{
        .imageView = swapView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    vk::Rect2D area{.offset = vk::Offset2D{.x = 0, .y = 0}, .extent = extent};
    cmd.beginRendering(makeRenderingInfo(area, {&colour, 1}, nullptr));
    ImGui_ImplVulkan_RenderDrawData(drawData, static_cast<VkCommandBuffer>(cmd));
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
