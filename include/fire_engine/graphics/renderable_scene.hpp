#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/graphics/particle.hpp>
#include <fire_engine/graphics/shadow_caster_bounds_frame.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Per-frame VDPM refine attribution summed over every instance: how many over-budget *triggers*
// each metric channel *won* (was the dominant reason to refine — trigger counts, not
// resulting-refine counts, since one decision pulls in dependency splits), plus the largest
// score/budget ratio each channel reached (max-reduced, not summed). The ratios are what
// distinguish "genuinely zero" from "reached 99% of budget" — an under-firing channel is only
// visible with them. Exposes the metric itself in the overlay — the first thing needed to see why a
// region is (or isn't) refining.
struct VdpmChannelStats
{
    uint32_t geometryTriggers{0};
    uint32_t uvTriggers{0};
    uint32_t normalTriggers{0};
    uint32_t tangentTriggers{0};
    float maxGeometryRatio{0.0f};
    float maxUvRatio{0.0f};
    float maxNormalRatio{0.0f};
    float maxTangentRatio{0.0f};
};

// Per-frame scene-side accounting the renderer surfaces in the overlay. Coarse-cull counts (how
// many renderables the scene tracked and dropped as outside every frustum) plus VDPM repair work
// (the vertices each per-frame repair pass pulled back in, summed over every VDPM instance) and the
// per-channel refine attribution — diagnostics so a repair-count/refine regression is visible.
struct CullStats
{
    std::size_t tracked{0};
    std::size_t culled{0};
    uint32_t vdpmFoldoversRepaired{0};
    uint32_t vdpmCoverageRepaired{0};
    VdpmChannelStats vdpmChannels;
};

// The active camera's world-space pose, as the scene reports it to the renderer through the seam.
// Position + look-at target only — the renderer derives view/projection (and owns FOV/near/far). A
// future camera-types pass can widen this if a camera legitimately owns its own lens.
struct CameraView
{
    Vec3 position;
    Vec3 target;
};

// The render-facing view of a scene, and the Vulkan-free seam between the scene and render
// layers (CR-09). The renderer holds a RenderableScene& and pulls per-frame draw data through it
// without ever touching the scene graph's node tree; the scene never sees a Vulkan handle. Every
// type crossing this interface lives in graphics/ (or math/), so neither `render/` nor `scene/`
// depends on the other — both depend only on this shared layer.
//
// Matrix ownership: the scene owns all transforms (camera + per-object). Per-object world matrices
// flow to the renderer inside the emitted DrawCommands; the active camera's pose flows through
// `activeCamera()`. The renderer owns only the render-pipeline matrices (view/projection/shadow) it
// derives from those inputs — the camera no longer crosses the seam as a bespoke drawFrame
// argument.
class RenderableScene
{
public:
    RenderableScene() = default;
    virtual ~RenderableScene() = default;

    RenderableScene(const RenderableScene&) = default;
    RenderableScene& operator=(const RenderableScene&) = default;
    RenderableScene(RenderableScene&&) noexcept = default;
    RenderableScene& operator=(RenderableScene&&) noexcept = default;

    // This frame's active-camera pose (position + look-at target). Must reflect the current frame,
    // so the scene's per-frame update (which refreshes camera world transforms) has to run before
    // the renderer reads this. An implementation with no camera authored returns a fixed debug
    // fallback (not a real scene entity) — see SceneGraph.
    [[nodiscard]] virtual CameraView activeCamera() const = 0;

    // Resolve this frame's world-space lights into `out` (cleared first).
    virtual void gatherLights(std::vector<Lighting>& out) const = 0;

    // Resolve this frame's active particle emitters into `out` (cleared first).
    virtual void gatherEmitters(std::vector<EmitterState>& out) const = 0;

    // SH-06: every shadow caster's world bounds for THIS frame, into `out` (reset first), before
    // any draw command is built.
    //
    // A prepass rather than a read of the draw list, because of an ordering constraint that cannot
    // be worked around: the cascade depth range is fitted from these bounds, the fitted matrices
    // decide the shadow frustums, and those frustums are what the draw walk culls against. A
    // cascade finalised after draw collection would leave the frame's matrices describing a
    // different fit than the one its draws were selected for.
    //
    // `out` is then the frame's ONLY authority on caster bounds: `buildDrawCommands` receives it
    // and each shadow command looks up its own binding's entry, rather than anything recomputing.
    virtual void gatherShadowCasters(ShadowCasterBoundsFrame& out) const = 0;

    // Build this frame's draw commands, appending them to `out`. Renderables are culled against
    // `frustums` (the camera frustum plus any shadow-caster frustums) using the scene's own spatial
    // bounds; an empty `frustums` span means culling is disabled and everything is drawn.
    // Per-object data (world matrix, skin/morph, material) is baked from `frame`. Returns
    // coarse-cull counts.
    // `casterBounds` is this frame's prepass result, passed explicitly rather than stashed
    // anywhere: every shadow draw takes its bounds from that record, so the geometry the cascade
    // was fitted to and the geometry the pass culls are by construction the same measurement.
    [[nodiscard]]
    virtual CullStats buildDrawCommands(const FrameInfo& frame, std::span<const Frustum> frustums,
                                        const ShadowCasterBoundsFrame& casterBounds,
                                        std::vector<DrawCommand>& out) = 0;
};

} // namespace fire_engine
