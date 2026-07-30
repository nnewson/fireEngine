#include <fire_engine/render/debug_overlay.hpp>

#include <cstddef>
#include <cstdio>
#include <string_view>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <fire_engine/platform/window.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

namespace
{

// One row of the SH-01 shadow table. `timing` is pre-formatted by the caller: milliseconds for a
// family row, an explicit non-number for rows that have no measurement of their own.
//
// A `focusable` row is CLICKABLE and returns whether it was clicked this frame (SH-03 slice 4): the
// per-view reason breakdown needs a view chosen, and choosing it by clicking the row whose numbers
// raised the question is the shortest path from "this map keeps too much" to "here is why". Group
// rollup rows are not focusable — a group's reasons are a SUBSET of the scene total (a useful one:
// "the point family forces LOD0 and the cascades don't"), but a group has no single logical view,
// which is what the focus identifies and what the ShadowLod tint needs. Worth revisiting if reading
// a family's mix on its own turns out to be the common question.
bool shadowStatsRow(const char* label, const ShadowViewStats& stats, const char* timing,
                    bool focusable = false, bool focused = false)
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    bool clicked = false;
    if (focusable)
    {
        clicked = ImGui::Selectable(label, focused, ImGuiSelectableFlags_SpanAllColumns);
    }
    else
    {
        ImGui::TextUnformatted(label);
    }
    ImGui::TableNextColumn();
    ImGui::Text("%llu", static_cast<unsigned long long>(stats.rasterPasses));
    ImGui::TableNextColumn();
    ImGui::Text("%llu / %llu", static_cast<unsigned long long>(stats.drawnDraws),
                static_cast<unsigned long long>(stats.candidateDraws));
    ImGui::TableNextColumn();
    ImGui::Text("%llu / %llu", static_cast<unsigned long long>(stats.drawnTriangles),
                static_cast<unsigned long long>(stats.candidateTriangles));
    for (std::size_t bin = 0; bin < kShadowLodBinCount; ++bin)
    {
        ImGui::TableNextColumn();
        ImGui::Text("%llu", static_cast<unsigned long long>(stats.lodHistogram[bin]));
    }
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(timing);
    return clicked;
}

// Names the LOGICAL view an identity refers to — the thing a focus actually selects, as opposed to
// the physical row it happens to occupy. The family alone is not a name: "cascade" describes four
// different views and "point" describes every face of every point light, so a label that stops
// there cannot tell a reader which one they are looking at (or tinting by).
void formatShadowViewIdentity(char* out, std::size_t size, ShadowViewGroup group,
                              const ShadowLogicalViewId& view)
{
    const std::string_view family = toString(group);
    const auto familyChars = static_cast<int>(family.size());
    switch (view.kind())
    {
    case ShadowLogicalViewKind::Cascade:
        std::snprintf(out, size, "%.*s %llu", familyChars, family.data(),
                      static_cast<unsigned long long>(view.id()));
        return;
    case ShadowLogicalViewKind::Self:
        // The owning object, not a slot: the slot is reassigned per frame, the object is not.
        std::snprintf(out, size, "%.*s object %llu", familyChars, family.data(),
                      static_cast<unsigned long long>(view.id()));
        return;
    case ShadowLogicalViewKind::Spot:
        std::snprintf(out, size, "%.*s light %llu", familyChars, family.data(),
                      static_cast<unsigned long long>(view.id()));
        return;
    case ShadowLogicalViewKind::Point:
        std::snprintf(out, size, "%.*s light %llu face %u", familyChars, family.data(),
                      static_cast<unsigned long long>(view.id()),
                      static_cast<unsigned>(view.face()));
        return;
    case ShadowLogicalViewKind::Invalid:
        break;
    }
    std::snprintf(out, size, "%.*s (no view)", familyChars, family.data());
}

// Row label for one physical slot of a family. Deliberately says "slot", not "light": slots are
// assignment order into a fixed array, so the same row can describe a different light next frame —
// a row is a MAP, not an identity. Point views are stored flat as lightSlot * kCubeFaceCount +
// face, so they decode back to both numbers here (the same split the renderer used to pick the
// cube layer).
void formatShadowSlotLabel(char* out, std::size_t size, ShadowViewGroup group, std::size_t slot)
{
    switch (group)
    {
    case ShadowViewGroup::Cascade:
    case ShadowViewGroup::WorldOnly:
        std::snprintf(out, size, "  cascade %zu", slot);
        return;
    case ShadowViewGroup::Point:
        std::snprintf(out, size, "  slot %zu face %zu", slot / kCubeFaceCount,
                      slot % kCubeFaceCount);
        return;
    case ShadowViewGroup::Self:
    case ShadowViewGroup::Spot:
    case ShadowViewGroup::Count:
        break;
    }
    std::snprintf(out, size, "  slot %zu", slot);
}

// The SH-01 evidence panel: what each shadow view rasterised this frame, and at which levels.
//
// Two different quantities share the table, and mixing them is the mistake it is laid out to
// prevent. "Passes" and the draw/triangle columns are RASTER WORK — the same caster counts once per
// view it appears in, which is the real GPU cost. The L0..L3+ columns are LOD SELECTIONS OF DRAWN
// CASTERS: one per drawn caster per logical view (a rejected candidate is never resolved and has no
// level; the self families rasterise twice but are sampled once), so they describe the level
// distribution, not the work.
void drawShadowDiagnostics(const FrameStats& stats, RenderTunables& tunables)
{
    if (!stats.shadowValid)
    {
        // The counters are published only when their frame's ring slot completes, so the first
        // frames after start-up (and after a resize) genuinely have nothing to report. Say so
        // rather than printing a zeroed table that reads like "no shadows rendered".
        ImGui::TextDisabled("Shadow diagnostics: pending (ring warm-up)");
        return;
    }

    const auto& shadow = stats.shadow;
    // SH-03 slice 4: reasons are recorded PER VIEW, at the resolution that produced them, so the
    // panel has to say WHICH view it is reporting. The scene rollup still answers "how much shadow
    // work happened", but the question this arc raises — why did this map keep that much geometry —
    // is per view, and summing it across views mixes unrelated decisions.
    const ShadowViewFocus& focus = tunables.shadowViewFocus;
    const FocusedShadowView focusedView = shadow.focused(focus);
    const ShadowViewStats sceneTotal = shadow.sceneTotal();
    // FOUR states, said differently because a reader acts on them differently:
    //  * the scene rollup (nothing selected);
    //  * a selected view that rasterised — its numbers follow;
    //  * a selection that is STRUCTURALLY malformed (no identity, or an identity that cannot occur
    //    in this group) — no frame will ever satisfy it, so the message says to re-select;
    //  * a well-formed selection not found in this frame.
    //
    // That last message deliberately does NOT say whether the view will come back. A light removed
    // from the scene and a light that simply did not rasterise this frame are the same thing here —
    // both are valid identities that were not found — and telling them apart needs scene liveness
    // the diagnostics do not have. Claiming either way would be a guess presented as a measurement.
    char reasonScope[128] = "Reasons: scene total (every view)";
    const bool focusRan = focusedView.found();
    if (focus.perView && !focus.addressable())
    {
        std::snprintf(reasonScope, sizeof(reasonScope),
                      "Reasons: selection is not a valid view — pick a row");
    }
    else if (focus.perView)
    {
        // Named by IDENTITY (which view), then located by the slot it was FOUND in this frame
        // (which row) — an identity can move slots between frames, so the slot it was selected in
        // would label a row it no longer occupies.
        char view[64];
        formatShadowViewIdentity(view, sizeof(view), focus.group, focus.view);
        // `formatShadowSlotLabel` indents for the table, so the leading spaces are skipped here —
        // without that the dash and the label ran together ("— cascade slot 3" became "—cascade").
        char where[48] = "(not present in this frame)";
        if (focusRan)
        {
            char slotLabel[48];
            formatShadowSlotLabel(slotLabel, sizeof(slotLabel), focus.group, focusedView.slot);
            const std::string_view trimmed = std::string_view{slotLabel}.substr(
                std::string_view{slotLabel}.find_first_not_of(' '));
            std::snprintf(where, sizeof(where), "%.*s", static_cast<int>(trimmed.size()),
                          trimmed.data());
        }
        std::snprintf(reasonScope, sizeof(reasonScope), "Reasons: %s — %s", view, where);
    }
    ImGui::TextUnformatted(reasonScope);
    // SH-03 slice 6: the dead band's calibration instrument, and a headline rather than a footnote
    // — it describes the frame as a whole. TRANSITIONS are movement (a caster receding legitimately
    // steps L0 -> L1 -> L2); REVERSED is chatter (L1 -> L2 -> L1), a caster oscillating across a
    // threshold, and the only one of the two a dead band can fix. `first` is separated so a caster
    // entering a view cannot be mistaken for either.
    ImGui::Text("LOD movement: %llu transitions (%llu reversed), %llu held, %llu first",
                static_cast<unsigned long long>(shadow.lodMovement.transitions),
                static_cast<unsigned long long>(shadow.lodMovement.reversed),
                static_cast<unsigned long long>(shadow.lodMovement.held),
                static_cast<unsigned long long>(shadow.lodMovement.firstSeen));

    // Every reason, by iterating the enum rather than naming a subset: a hard-coded list silently
    // hides new ones (the SH-02 fallbacks were invisible here until this changed), and a forced
    // LOD0 that looks like a deliberate one defeats the point of having reasons at all.
    //
    // A two-column table rather than one line: the fallback names are long ("near-plane
    // intersection", "invalid previous level") and would clip off the panel edge, which would make
    // "visible" true only in the source.
    const bool reportReasons = !focus.perView || focusRan;
    if (reportReasons &&
        ImGui::BeginTable("shadowreasons", 2,
                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
    {
        const ShadowViewStats& reported = focusRan ? *focusedView.stats : sceneTotal;
        for (std::size_t reason = 0; reason < kShadowLodReasonCount; ++reason)
        {
            const auto label = toString(static_cast<ShadowLodReason>(reason));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%.*s", static_cast<int>(label.size()), label.data());
            ImGui::TableNextColumn();
            ImGui::Text("%llu", static_cast<unsigned long long>(reported.lodReasons[reason]));
        }
        ImGui::EndTable();
    }

    constexpr int kColumns = 5 + static_cast<int>(kShadowLodBinCount);
    if (ImGui::BeginTable("shadowviews", kColumns,
                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersInnerV))
    {
        // Explicit weights, because proportional-by-default gave the wide "Tris d/c" column enough
        // room to squeeze the level columns down to a single ellipsised character — the LOD
        // distribution, which is the one thing this table exists to show, was unreadable.
        ImGui::TableSetupColumn("View", ImGuiTableColumnFlags_WidthStretch, 2.8f);
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthStretch, 0.55f);
        ImGui::TableSetupColumn("Draws d/c", ImGuiTableColumnFlags_WidthStretch, 1.35f);
        ImGui::TableSetupColumn("Tris d/c", ImGuiTableColumnFlags_WidthStretch, 2.3f);
        ImGui::TableSetupColumn("L0", ImGuiTableColumnFlags_WidthStretch, 0.5f);
        ImGui::TableSetupColumn("L1", ImGuiTableColumnFlags_WidthStretch, 0.5f);
        ImGui::TableSetupColumn("L2", ImGuiTableColumnFlags_WidthStretch, 0.5f);
        ImGui::TableSetupColumn("L3+", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableHeadersRow();

        for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
        {
            const auto group = static_cast<ShadowViewGroup>(g);
            const ShadowViewStats total = shadow.groupTotal(group);

            // "n/a", not the word: the column is narrow enough that "unavailable" clipped, and a
            // clipped diagnostic is one nobody reads. The banner above already says why.
            char timing[32] = "n/a";
            if (stats.gpuValid)
            {
                // A family that never rasterised has no bracketed span this frame — its resolved
                // time is 0 because nothing ran, which is a different fact from "not measured".
                const auto pass = static_cast<std::size_t>(shadowProfilePass(group));
                std::snprintf(timing, sizeof(timing), "%.3f",
                              static_cast<double>(stats.passMs[pass]));
            }

            char groupLabel[48];
            const std::string_view name = toString(group);
            std::snprintf(groupLabel, sizeof(groupLabel), "%.*s", static_cast<int>(name.size()),
                          name.data());
            shadowStatsRow(groupLabel, total, total.touched() ? timing : "idle");

            for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
            {
                const ShadowViewStats& view = shadow.view(group, slot);
                if (!view.touched())
                {
                    continue; // a slot nothing rasterised into costs nothing and says nothing
                }
                char slotLabel[48];
                formatShadowSlotLabel(slotLabel, sizeof(slotLabel), group, slot);
                // Highlighted by IDENTITY, not by slot: if the selected view moved slots this
                // frame, the highlight moves with it rather than staying on whatever now occupies
                // the old row.
                const bool isFocused =
                    focus.perView && focus.group == group && focus.view == view.logicalId;
                // Timestamps bracket a whole family, not one map, so a slot row has no time of its
                // own — an em dash, never a share of the family's number.
                if (shadowStatsRow(slotLabel, view, "—", /*focusable=*/true, isFocused))
                {
                    // The row's IDENTITY is what gets remembered. Remembering the slot would
                    // silently retarget to a different light the moment assignments compact.
                    tunables.shadowViewFocus =
                        ShadowViewFocus{.perView = true, .group = group, .view = view.logicalId};
                }
            }
        }

        // The five families are disjoint spans (there is no outer shadow timer), so their sum IS
        // the frame's shadow time.
        char totalTiming[32] = "n/a";
        if (stats.gpuValid)
        {
            float shadowMs = 0.0f;
            for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
            {
                shadowMs += stats.passMs[static_cast<std::size_t>(
                    shadowProfilePass(static_cast<ShadowViewGroup>(g)))];
            }
            std::snprintf(totalTiming, sizeof(totalTiming), "%.3f", static_cast<double>(shadowMs));
        }
        if (shadowStatsRow("Scene total", sceneTotal, totalTiming, /*focusable=*/true,
                           !tunables.shadowViewFocus.perView))
        {
            tunables.shadowViewFocus = ShadowViewFocus{}; // back to the rollup
        }
        ImGui::EndTable();
    }

    ImGui::TextDisabled(
        "Click a slot row to read its reasons; 'Scene total' returns to the rollup");
    ImGui::TextDisabled("Draws d/c: drawn / offered — the difference is this view's cull yield");
    ImGui::TextDisabled("Tris d/c: drawn / FULL DETAIL — the difference is culling AND LOD");
    ImGui::TextDisabled("L0..L3+ = LOD selections of DRAWN casters, once per logical view");
}

} // namespace

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

    // A FIRST-USE size only (ImGui remembers whatever the user drags it to). The default was
    // narrow enough that the shadow table's drawn/candidate pairs clipped mid-number, which is the
    // one place a diagnostic must not be approximate — the whole panel exists to compare those two
    // numbers per view.
    ImGui::SetNextWindowSize(ImVec2{520.0f, 900.0f}, ImGuiCond_FirstUseEver);
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
                ImGui::TextUnformatted(kProfilePassNames[p]);
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
    if (ImGui::CollapsingHeader("Shadows (SH-01)", ImGuiTreeNodeFlags_DefaultOpen))
    {
        drawShadowDiagnostics(stats, tunables);
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

        // SHADOW LOD is its own switch, not a rider on the forward one (SH-03 slice 6). Turning it
        // off leaves the visible geometry selecting normally, which is what an A/B of shadow LOD
        // needs; gating these sliders on `lodEnabled` would also grey them out for a run that only
        // meant to hold the camera geometry still.
        ImGui::Checkbox("Shadow LOD##shadowlod", &tunables.shadowLodEnabled);
        ImGui::BeginDisabled(!tunables.shadowLodEnabled);
        // A SEPARATE budget in shadow-map texels, deliberately not the camera one — different
        // units, different evidence (see tools/shadow_lod_sweep.sh).
        ImGui::SliderFloat("Shadow texel budget", &tunables.shadowLodPixelBudget, 0.25f, 16.0f,
                           "%.2f");
        // The dead band. 1.0 disables hysteresis; SMALLER widens the band, holding finer geometry
        // longer. Calibration sweeps the budget at 1.0 first, so a wide band cannot hide a budget
        // that is slightly too tight.
        ImGui::SliderFloat("Shadow coarsen ratio", &tunables.shadowLodCoarsenRatio, 0.25f, 1.0f,
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
            // GPU-driven front backend toggle (B5c-3). The manager is constructed whenever the
            // device supports it, independent of this selector, so flipping it takes effect the
            // next frame with NO reload and an unsupported device falls back to the CPU front. When
            // the device can't support it, show an explicit "unsupported" label rather than a
            // silent disabled checkbox.
            if (stats.vdpmGpuAvailable)
            {
                ImGui::Checkbox("GPU-driven front##vdpm", &tunables.vdpmGpuBackend);
            }
            else
            {
                ImGui::BeginDisabled(true);
                bool off = false;
                ImGui::Checkbox("GPU-driven front##vdpm", &off);
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::TextDisabled("(unsupported on this device)");
            }
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
        int view = static_cast<int>(tunables.debugView);
        if (ImGui::Combo("View", &view, kDebugViewNames.data(),
                         static_cast<int>(kDebugViewNames.size())))
        {
            tunables.debugView = static_cast<DebugView>(view);
        }
        // The Shadow LOD tint colours meshes by the level ONE shadow view chose (SH-03 slice 5), so
        // it has to name that view — the same mesh legitimately tints differently under a different
        // one, and a reader who does not know which view they are looking at cannot act on it. Said
        // here rather than only in the Shadows panel, because the tint is usable without opening
        // it.
        if (tunables.debugView == DebugView::ShadowLod)
        {
            if (tunables.shadowViewFocus.addressable())
            {
                char view[64];
                formatShadowViewIdentity(view, sizeof(view), tunables.shadowViewFocus.group,
                                         tunables.shadowViewFocus.view);
                ImGui::TextDisabled("Tinting by %s (focused in Shadows)", view);
            }
            else
            {
                ImGui::TextDisabled("Tinting by cascade 0 (default) — click a row in Shadows");
            }
            ImGui::TextDisabled(
                "Grey = no level from THAT view (no shadow, or it culled the mesh)");
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
