#pragma once

#include <array>
#include <cstdint>
#include <span>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/shadow_diagnostics.hpp>
#include <fire_engine/graphics/shadow_lod_resolver.hpp>
#include <fire_engine/graphics/shadow_map_validity.hpp>
#include <fire_engine/graphics/shadow_pass_plan.hpp>
#include <fire_engine/graphics/shadow_render_view.hpp>

// Turning a frame's shadow casters into a `ShadowFramePlan` (arc 2 #4 / §2.1).
//
// This is the half of the shadow pass that decides; `render/shadows.cpp` is the half that records.
// Everything that used to happen inside recording — the per-view filter, the LOD resolution, the
// diagnostic row claim and observations — happens HERE, because a shadow map can only be reused if
// the frame knows what it would have drawn BEFORE deciding whether to draw it. A recorder that
// still resolved as it went could not answer that question without doing the work it was trying to
// avoid.
//
// Vulkan-free and therefore headless-testable, which is the point: the cases that matter (a caster
// the filter drops, a family that must not be prepared at all, a self-shadow view's two layers) are
// exercised without a device.
//
// It MUTATES three things, all of them per-frame state the plan is derived from:
//
//   * the RESOLVER — each accepted caster is resolved through it, so its frame cache and its STAGED
//     hysteresis fill up here rather than during recording. The staged levels are still committed
//     only after the frame is submitted; moving selection earlier does not move that boundary.
//   * the STATS — preparation claims each row (naming the logical view it describes) and observes
//     every draw it walks. Raster passes stay with the recorder, where the GPU work is.
//   * the PLAN — the single object recording then consumes.

namespace fire_engine
{

// One family's raster parameters. They belong to the plan's comparison (both reach the rasteriser:
// the extent is the viewport and scissor, the biases are `vkCmdSetDepthBias`), but they live in
// `render/constants.hpp`, which the Vulkan-free graphics layer cannot see — so the caller supplies
// them and preparation records them into each prepared view.
struct ShadowFamilyRaster
{
    std::uint32_t extent{0};
    float depthBiasConstant{0.0f};
    float depthBiasSlope{0.0f};
};

// What preparation reads. Everything else it needs comes from the view SET, which is the authority
// on which physical views exist this frame and what each rasterises with.
//
// There is deliberately no "active spot count" or "active self count" here. Those were bounds the
// recorder carried beside the set, and a caller whose count disagreed with the set would either
// skip a view the set says is active or walk one it says is not. Preparation iterates every
// physical slot and asks the set, which is what makes "absent means inactive" true at the point of
// use.
struct ShadowPreparationInputs
{
    // Every shadow caster in the frame — what the cascade, spot and point families each filter for
    // themselves.
    std::span<const DrawCommand> shadowDraws{};
    // The same set minus skinned casters, for the world-only CSM that self-shadowed meshes sample.
    std::span<const DrawCommand> worldOnlyShadowDraws{};
    // Casters carrying a self-shadow slot; each self view keeps the one draw that names its slot.
    std::span<const DrawCommand> selfShadowDraws{};

    float lodBudgetTexels{0.0f};
    ShadowLodHysteresis hysteresis{};
    // When false no frustum is built, and every family's filter passes everything through — the
    // `--no-cull` path, which must produce the same IMAGE with more draws.
    bool cullingEnabled{true};

    // Indexed by ShadowViewGroup.
    std::array<ShadowFamilyRaster, kShadowViewGroupCount> raster{};
};

// Builds `plan` from the completed view set.
//
// `eligible` is the ELIGIBILITY answer (`ShadowFamilyEligibility::eligible()`), taken from the view
// set BEFORE this call. A family whose bit is clear is not prepared at all — not filtered, not
// resolved, not claimed — because resolving stages hysteresis, and a suppressed family's staged
// decisions would be committed for views whose maps are neither recorded nor sampled. The caller
// then derives the CONFIRMED validity from the finished plan (`shadowMapValidityFromPlan`), which
// is what the receiver is told.
//
// `plan` is reset first, so a caller cannot accumulate two frames into one.
//
// THROWS on a contradiction between what the frame thinks it is preparing and what the shadow state
// says — a diagnostic row claimed by two logical views, a caster that resolves to no geometry, a
// caster with no stated pose, or a prepared view the plan refuses. Each is corrupt render input or
// a producer bug, and both ways of continuing are worse than stopping: dropping the view leaves a
// shadow map holding another frame's content with nothing to say so, and degrading through it
// produces counters that read like a measurement and are not.
void prepareShadowFrame(const ShadowPreparationInputs& inputs, const ShadowRenderViewSet& views,
                        ShadowMapValidity eligible, ShadowLodResolver& resolver,
                        ShadowFrameStats& stats, ShadowFramePlan& plan);

} // namespace fire_engine
