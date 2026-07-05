#pragma once

#include <unordered_set>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>

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
    const std::unordered_set<const Node*>* culledNodes{nullptr};
    std::vector<DrawCommand>* drawCommands{nullptr};
};

} // namespace fire_engine
