#pragma once

#include <unordered_set>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/graphics/renderable_scene.hpp>

namespace fire_engine
{

class Node;

// Vulkan-free context threaded through the scene's draw-building traversal (CR-09). Carries
// exactly the three things the node / mesh / animator render paths read — the frame's camera +
// pipeline data, the coarse culled-node set (null ⇒ render everything), and the output draw-command
// list — with no reference to any Vulkan or `render/` type, so the scene layer stays
// backend-agnostic. Built once per frame by SceneGraph::buildDrawCommands.
struct SceneDrawContext
{
    const FrameInfo& frame;
    // This frame's caster-bounds authority (SH-06). Every shadow draw reads its own binding's entry
    // from here; nothing recomputes bounds during the draw walk, so the cascade fit and the draws
    // cannot describe different geometry.
    //
    // A REFERENCE, like `frame`: it is mandatory, and a nullable field would only move the
    // requirement into a runtime check at every consumer. A traversal without it cannot be
    // constructed.
    const ShadowCasterBoundsFrame& shadowCasterBounds;
    const std::unordered_set<const Node*>* culledNodes{nullptr};
    std::vector<DrawCommand>* drawCommands{nullptr};
    // Per-frame VDPM repair accumulators (null ⇒ not gathered): the mesh render path adds each
    // instance's repair work so SceneGraph can report a scene total for the overlay diagnostic.
    uint32_t* vdpmFoldoversRepaired{nullptr};
    uint32_t* vdpmCoverageRepaired{nullptr};
    // Per-frame VDPM per-channel refine attribution (null ⇒ not gathered), same accumulation.
    VdpmChannelStats* vdpmChannels{nullptr};
};

} // namespace fire_engine
