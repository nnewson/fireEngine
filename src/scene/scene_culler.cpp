#include <fire_engine/scene/scene_culler.hpp>

#include <algorithm>

#include <fire_engine/graphics/bounds.hpp>
#include <fire_engine/math/vec4.hpp>
#include <fire_engine/scene/mesh.hpp>
#include <fire_engine/scene/node.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] Vec3 transformPoint(const Mat4& m, Vec3 p)
{
    const Vec4 h = m * Vec4{p.x(), p.y(), p.z(), 1.0f};
    const float w = h.w() != 0.0f ? h.w() : 1.0f;
    return Vec3{h.x() / w, h.y() / w, h.z() / w};
}

// World-space AABB of `local` transformed by `m` (tight bound of the eight corners).
[[nodiscard]] AABB transformBounds(const Mat4& m, const Bounds3& local)
{
    Vec3 lo;
    Vec3 hi;
    for (int i = 0; i < 8; ++i)
    {
        const Vec3 corner{(i & 1) ? local.max.x() : local.min.x(),
                          (i & 2) ? local.max.y() : local.min.y(),
                          (i & 4) ? local.max.z() : local.min.z()};
        const Vec3 world = transformPoint(m, corner);
        if (i == 0)
        {
            lo = world;
            hi = world;
        }
        else
        {
            lo = {std::min(lo.x(), world.x()), std::min(lo.y(), world.y()),
                  std::min(lo.z(), world.z())};
            hi = {std::max(hi.x(), world.x()), std::max(hi.y(), world.y()),
                  std::max(hi.z(), world.z())};
        }
    }
    return AABB{lo, hi};
}

// A rigid renderable node is one carrying a Mesh whose geometry does not deform and has
// a valid local bound — exactly the nodes the BVH can cull by a transformed AABB.
[[nodiscard]] const Mesh* cullableMesh(const Node& node)
{
    const Mesh* mesh = node.componentAs<Mesh>();
    if (mesh == nullptr || mesh->object().deformable() || !mesh->object().localBounds().valid)
    {
        return nullptr;
    }
    return mesh;
}

} // namespace

void SceneCuller::syncNode(Node& node, std::unordered_set<const Node*>& seen)
{
    if (const Mesh* mesh = cullableMesh(node); mesh != nullptr)
    {
        seen.insert(&node);
        const AABB worldBox = transformBounds(node.composedWorld(), mesh->object().localBounds());
        if (const auto it = proxies_.find(&node); it != proxies_.end())
        {
            bvh_.moveProxy(it->second, worldBox);
        }
        else
        {
            proxies_.emplace(&node, bvh_.createProxy(worldBox, &node));
        }
    }

    for (const auto& child : node.children())
    {
        syncNode(*child, seen);
    }
}

void SceneCuller::sync(std::span<const std::unique_ptr<Node>> roots)
{
    std::unordered_set<const Node*> seen;
    for (const auto& root : roots)
    {
        syncNode(*root, seen);
    }

    // Drop proxies for nodes that vanished or stopped being cullable this frame.
    for (auto it = proxies_.begin(); it != proxies_.end();)
    {
        if (!seen.contains(it->first))
        {
            bvh_.destroyProxy(it->second);
            it = proxies_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

const std::unordered_set<const Node*>& SceneCuller::cull(std::span<const Frustum> frustums)
{
    visible_.clear();
    culled_.clear();

    for (const Frustum& frustum : frustums)
    {
        bvh_.traverse([&frustum](const AABB& box)
                      { return frustum.intersects(Bounds3{box.min, box.max, true}); },
                      [this](int proxy) { visible_.insert(bvh_.payload(proxy)); });
    }

    for (const auto& [node, proxy] : proxies_)
    {
        (void)proxy;
        if (!visible_.contains(node))
        {
            culled_.insert(node);
        }
    }
    return culled_;
}

void SceneCuller::clear()
{
    bvh_.clear();
    proxies_.clear();
    culled_.clear();
    visible_.clear();
}

} // namespace fire_engine
