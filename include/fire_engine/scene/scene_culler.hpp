#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fire_engine/collision/aabb_bvh.hpp>
#include <fire_engine/graphics/frustum.hpp>

namespace fire_engine
{

class Node;

// Persistent spatial index over the scene's rigid renderable nodes, used to skip
// draw-building for nodes outside *every* active frustum (camera + shadow casters).
//
// Each tracked node owns a fat-AABB BVH proxy whose tight bounds are the node's
// local-space mesh AABB transformed by its composed world matrix. `sync()` refreshes
// those bounds (cheap — eight corners + a fat-box containment check) and reconciles the
// proxy set as nodes appear/disappear or stop being cullable. `cull()` queries the BVH
// once per frustum, unions the visible leaves, and returns the tracked nodes that landed
// in no frustum — the set Node::render may safely skip.
//
// Deformable nodes (skinned/morph) are never tracked: their bind-pose AABB under-covers
// the animated mesh, so they are always drawn and left to the precise per-draw cull.
class SceneCuller
{
public:
    SceneCuller() = default;

    // Refresh proxy bounds for every rigid renderable node reachable from `roots`,
    // creating proxies for newly cullable nodes and destroying proxies for nodes that
    // disappeared or became deformable/non-renderable.
    void sync(std::span<const std::unique_ptr<Node>> roots);

    // Tracked nodes outside all `frustums`. The result is owned by the culler and reused
    // each frame; it stays valid until the next sync()/cull().
    [[nodiscard]] const std::unordered_set<const Node*>& cull(std::span<const Frustum> frustums);

    [[nodiscard]] std::size_t trackedCount() const noexcept
    {
        return proxies_.size();
    }
    [[nodiscard]] std::size_t culledCount() const noexcept
    {
        return culled_.size();
    }

    void clear();

private:
    struct Proxy
    {
        int id{0};
        std::uint64_t worldRevision{0};
        std::uint64_t seenGeneration{0};
    };

    void syncNode(Node& node);

    AabbBvh<Node*> bvh_{0.1f};
    std::unordered_map<const Node*, Proxy> proxies_;
    std::unordered_set<const Node*> culled_;
    std::unordered_set<const Node*> visible_;
    std::uint64_t syncGeneration_{0};
};

} // namespace fire_engine
