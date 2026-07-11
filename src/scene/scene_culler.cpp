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

// World-space AABB of `local` transformed by `m` (tight bound of the eight corners). `m` is a
// node's composed world matrix (affine), so Mat4::transformPoint's affine (no-perspective-divide)
// semantics are exactly right here.
[[nodiscard]] AABB transformBounds(const Mat4& m, const Bounds3& local)
{
    const auto corners = local.corners();
    Vec3 lo = m.transformPoint(corners.front());
    Vec3 hi = lo;
    for (std::size_t i = 1; i < corners.size(); ++i)
    {
        const Vec3 world = m.transformPoint(corners[i]);
        lo = {std::min(lo.x(), world.x()), std::min(lo.y(), world.y()),
              std::min(lo.z(), world.z())};
        hi = {std::max(hi.x(), world.x()), std::max(hi.y(), world.y()),
              std::max(hi.z(), world.z())};
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

void SceneCuller::syncNode(Node& node)
{
    if (const Mesh* mesh = cullableMesh(node); mesh != nullptr)
    {
        if (const auto it = proxies_.find(&node); it != proxies_.end())
        {
            Proxy& proxy = it->second;
            proxy.seenGeneration = syncGeneration_;
            if (proxy.worldRevision != node.worldRevision())
            {
                const AABB worldBox =
                    transformBounds(node.composedWorld(), mesh->object().localBounds());
                bvh_.moveProxy(proxy.id, worldBox);
                proxy.worldRevision = node.worldRevision();
            }
        }
        else
        {
            const AABB worldBox =
                transformBounds(node.composedWorld(), mesh->object().localBounds());
            proxies_.emplace(&node, Proxy{.id = bvh_.createProxy(worldBox, &node),
                                          .worldRevision = node.worldRevision(),
                                          .seenGeneration = syncGeneration_});
        }
    }

    for (const auto& child : node.children())
    {
        syncNode(*child);
    }
}

void SceneCuller::sync(std::span<const std::unique_ptr<Node>> roots)
{
    ++syncGeneration_;
    for (const auto& root : roots)
    {
        syncNode(*root);
    }

    // Drop proxies for nodes that vanished or stopped being cullable this frame.
    for (auto it = proxies_.begin(); it != proxies_.end();)
    {
        if (it->second.seenGeneration != syncGeneration_)
        {
            bvh_.destroyProxy(it->second.id);
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
    syncGeneration_ = 0;
}

} // namespace fire_engine
