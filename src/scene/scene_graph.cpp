#include <fire_engine/scene/light.hpp>
#include <fire_engine/scene/particle_emitter.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{

namespace
{
void gatherLightsRecursive(const Node& node, std::vector<Lighting>& out)
{
    if (const auto* light = node.componentAs<Light>())
    {
        out.push_back(Light::toLighting(*light, node.composedWorld()));
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

void applyPhysicsRecursive(Node& node, const PhysicsWorld& physics)
{
    const PhysicsBodyHandle handle = node.physicsBodyHandle();
    if (handle.valid())
    {
        const PhysicsBody* body = physics.body(handle);
        auto transform = physics.bodyTransform(handle);
        if (body != nullptr && transform.has_value() && body->type() != PhysicsBodyType::Static)
        {
            node.transform().position(transform->position());
            node.transform().rotation(transform->rotation());
            node.transform().scale(transform->scale());
        }
    }

    for (const auto& child : node.children())
    {
        applyPhysicsRecursive(*child, physics);
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

void SceneGraph::applyPhysics(const PhysicsWorld& physics)
{
    for (auto& node : nodes_)
    {
        applyPhysicsRecursive(*node, physics);
    }
    resolve();
}

void SceneGraph::render(const RenderContext& ctx)
{
    for (auto& node : nodes_)
    {
        node->render(ctx, rootTransform_);
    }
}

std::vector<Lighting> SceneGraph::gatherLights() const
{
    std::vector<Lighting> out;
    for (const auto& node : nodes_)
    {
        gatherLightsRecursive(*node, out);
    }
    return out;
}

std::vector<EmitterState> SceneGraph::gatherEmitters() const
{
    std::vector<EmitterState> out;
    for (const auto& node : nodes_)
    {
        gatherEmittersRecursive(*node, out);
    }
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
