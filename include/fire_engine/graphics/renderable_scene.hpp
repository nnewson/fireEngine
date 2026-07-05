#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/graphics/particle.hpp>

namespace fire_engine
{

// Coarse-cull accounting for the overlay: how many renderables the scene tracked this frame and
// how many it dropped as outside every frustum.
struct CullStats
{
    std::size_t tracked{0};
    std::size_t culled{0};
};

// The render-facing view of a scene, and the Vulkan-free seam between the scene and render
// layers (CR-09). The renderer holds a RenderableScene& and pulls per-frame draw data through it
// without ever touching the scene graph's node tree; the scene never sees a Vulkan handle. Every
// type crossing this interface lives in graphics/ (or math/), so neither `render/` nor `scene/`
// depends on the other — both depend only on this shared layer.
//
// Matrix ownership: the scene owns all transforms (camera + per-object). Per-object world
// matrices flow to the renderer inside the emitted DrawCommands; the camera transform is supplied
// back as the `frame` argument the renderer derives its view/projection from. The renderer owns
// only the render-pipeline matrices (view/projection/shadow) it derives from those inputs.
class RenderableScene
{
public:
    RenderableScene() = default;
    virtual ~RenderableScene() = default;

    RenderableScene(const RenderableScene&) = default;
    RenderableScene& operator=(const RenderableScene&) = default;
    RenderableScene(RenderableScene&&) noexcept = default;
    RenderableScene& operator=(RenderableScene&&) noexcept = default;

    // Resolve this frame's world-space lights into `out` (cleared first).
    virtual void gatherLights(std::vector<Lighting>& out) const = 0;

    // Resolve this frame's active particle emitters into `out` (cleared first).
    virtual void gatherEmitters(std::vector<EmitterState>& out) const = 0;

    // Build this frame's draw commands, appending them to `out`. Renderables are culled against
    // `frustums` (the camera frustum plus any shadow-caster frustums) using the scene's own spatial
    // bounds; an empty `frustums` span means culling is disabled and everything is drawn.
    // Per-object data (world matrix, skin/morph, material) is baked from `frame`. Returns
    // coarse-cull counts.
    [[nodiscard]]
    virtual CullStats buildDrawCommands(const FrameInfo& frame, std::span<const Frustum> frustums,
                                        std::vector<DrawCommand>& out) = 0;
};

} // namespace fire_engine
