#include <cassert>

#include <fire_engine/core/log.hpp>
#include <fire_engine/scene/camera.hpp>
#include <fire_engine/scene/light.hpp>
#include <fire_engine/scene/particle_emitter.hpp>
#include <fire_engine/scene/scene_draw_context.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{

namespace
{
void gatherLightsRecursive(const Node& node, std::vector<Lighting>& out)
{
    if (const auto* light = node.componentAs<Light>())
    {
        out.push_back(Light::toLighting(*light, node.composedWorld(), node.id()));
    }
    for (const auto& child : node.children())
    {
        gatherLightsRecursive(*child, out);
    }
}

void gatherEmittersRecursive(const Node& node, std::vector<EmitterState>& out)
{
    if (const auto* emitter = node.componentAs<ParticleEmitter>())
    {
        out.push_back(ParticleEmitter::toEmitterState(*emitter, node.composedWorld()));
    }
    for (const auto& child : node.children())
    {
        gatherEmittersRecursive(*child, out);
    }
}

bool hasDirectionalLightRecursive(const Node& node)
{
    if (const auto* light = node.componentAs<Light>())
    {
        if (light->type() == Light::Type::Directional)
        {
            return true;
        }
    }
    for (const auto& child : node.children())
    {
        if (hasDirectionalLightRecursive(*child))
        {
            return true;
        }
    }
    return false;
}

void submitPhysicsRecursive(const Node& node, PhysicsWorld& physics)
{
    const PhysicsBodyHandle handle = node.physicsBodyHandle();
    if (handle.valid())
    {
        const PhysicsBody* body = physics.body(handle);
        if (body != nullptr && body->type() != PhysicsBodyType::Dynamic)
        {
            physics.setBodyTransform(handle, node.transform());
        }
    }

    for (const auto& child : node.children())
    {
        submitPhysicsRecursive(*child, physics);
    }
}

void applyPhysicsRecursive(Node& node, const PhysicsWorld& physics, float alpha)
{
    const PhysicsBodyHandle handle = node.physicsBodyHandle();
    if (handle.valid())
    {
        const PhysicsBody* body = physics.body(handle);
        auto transform = physics.interpolatedBodyTransform(handle, alpha);
        if (body != nullptr && transform.has_value() && body->type() != PhysicsBodyType::Static)
        {
            if (node.hasWorldOverride())
            {
                // Ragdoll-driven bone: the body's world transform *is* the bone's
                // world pose (no parent composition), so write it straight into the
                // override that resolve() reads.
                node.worldOverride(transform->world());
            }
            else
            {
                node.transform().position(transform->position());
                node.transform().rotation(transform->rotation());
                node.transform().scale(transform->scale());
            }
        }
    }

    for (const auto& child : node.children())
    {
        applyPhysicsRecursive(*child, physics, alpha);
    }
}
} // namespace

Node& SceneGraph::addNode(std::unique_ptr<Node> node)
{
    nodes_.push_back(std::move(node));
    return *nodes_.back();
}

void SceneGraph::update(const InputState& input_state)
{
    for (auto& node : nodes_)
    {
        node->update(input_state, rootTransform_);
    }
}

void SceneGraph::resolve()
{
    for (auto& node : nodes_)
    {
        node->resolve(rootTransform_);
    }
}

void SceneGraph::submitPhysics(PhysicsWorld& physics) const
{
    for (const auto& node : nodes_)
    {
        submitPhysicsRecursive(*node, physics);
    }
}

void SceneGraph::applyPhysics(const PhysicsWorld& physics, float alpha)
{
    for (auto& node : nodes_)
    {
        applyPhysicsRecursive(*node, physics, alpha);
    }
    resolve();
}

CullStats SceneGraph::buildDrawCommands(const FrameInfo& frame, std::span<const Frustum> frustums,
                                        std::vector<DrawCommand>& out)
{
    // Cull only when the renderer supplied frustums; an empty span means culling is disabled, so
    // every renderable is drawn (culled == nullptr). The culled-node set never leaves the scene.
    const std::unordered_set<const Node*>* culled = nullptr;
    CullStats stats;
    if (!frustums.empty())
    {
        culler_.sync(nodes_);
        culled = &culler_.cull(frustums);
        stats.tracked = culler_.trackedCount();
        stats.culled = culler_.culledCount();
    }

    const SceneDrawContext ctx{frame,
                               culled,
                               &out,
                               &stats.vdpmFoldoversRepaired,
                               &stats.vdpmCoverageRepaired,
                               &stats.vdpmChannels};
    for (auto& node : nodes_)
    {
        node->render(ctx, rootTransform_);
    }
    return stats;
}

CameraView SceneGraph::activeCamera() const
{
    if (activeCameraNode_ == nullptr)
    {
        // No camera authored. Return a fixed debug pose — deliberately NOT a scene node, so it
        // can't be moved or drawn and never becomes a crutch that papers over an unauthored scene.
        // Warn once so the missing camera is visible in the log rather than silently worked around.
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            log::warn(
                log::category::general,
                "No active camera set; rendering from the debug fallback pose. Author a Camera "
                "node and pass it to SceneGraph::activeCamera(node).");
        }
        return CameraView{Vec3{2.0f, 2.0f, 2.0f}, Vec3{0.0f, 0.0f, 0.0f}};
    }
    const Camera* camera = activeCameraNode_->componentAs<Camera>();
    assert(camera != nullptr && "active camera node lost its Camera component");
    return CameraView{camera->worldPosition(), camera->worldTarget()};
}

void SceneGraph::activeCamera(Node* cameraNode) noexcept
{
    assert((cameraNode == nullptr || cameraNode->componentAs<Camera>() != nullptr) &&
           "SceneGraph::activeCamera: node must carry a Camera component");
    activeCameraNode_ = cameraNode;
}

void SceneGraph::gatherLights(std::vector<Lighting>& out) const
{
    out.clear();
    for (const auto& node : nodes_)
    {
        gatherLightsRecursive(*node, out);
    }
}

std::vector<Lighting> SceneGraph::gatherLights() const
{
    std::vector<Lighting> out;
    gatherLights(out);
    return out;
}

void SceneGraph::gatherEmitters(std::vector<EmitterState>& out) const
{
    out.clear();
    for (const auto& node : nodes_)
    {
        gatherEmittersRecursive(*node, out);
    }
}

std::vector<EmitterState> SceneGraph::gatherEmitters() const
{
    std::vector<EmitterState> out;
    gatherEmitters(out);
    return out;
}

bool SceneGraph::hasDirectionalLight() const
{
    for (const auto& node : nodes_)
    {
        if (hasDirectionalLightRecursive(*node))
        {
            return true;
        }
    }
    return false;
}

} // namespace fire_engine
