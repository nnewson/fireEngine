#pragma once

#include <memory>
#include <vector>

#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/graphics/particle.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/render/render_context.hpp>
#include <fire_engine/scene/node.hpp>

namespace fire_engine
{

class SceneGraph
{
public:
    SceneGraph() = default;
    ~SceneGraph() = default;

    SceneGraph(const SceneGraph&) = delete;
    SceneGraph& operator=(const SceneGraph&) = delete;
    SceneGraph(SceneGraph&&) noexcept = default;
    SceneGraph& operator=(SceneGraph&&) noexcept = default;

    Node& addNode(std::unique_ptr<Node> node);

    [[nodiscard]] const std::vector<std::unique_ptr<Node>>& nodes() const noexcept
    {
        return nodes_;
    }

    [[nodiscard]] Mat4 rootTransform() const noexcept
    {
        return rootTransform_;
    }
    void rootTransform(Mat4 t) noexcept
    {
        rootTransform_ = t;
    }

    void update(const InputState& input_state);
    void resolve();
    void submitPhysics(PhysicsWorld& physics) const;
    void applyPhysics(const PhysicsWorld& physics);
    void render(const RenderContext& ctx);

    // Walk the scene tree and resolve every Light component into a world-space
    // Lighting. Composed world matrices are taken from each Node's cached
    // composedWorld_ (populated by the most recent update() call). Cheap —
    // light counts are tiny compared to draw counts.
    [[nodiscard]] std::vector<Lighting> gatherLights() const;

    // Walk the scene tree and resolve every ParticleEmitter component into a
    // world-space EmitterState (translation + node-rotated velocity from the
    // cached composedWorld_). Mirrors gatherLights; the renderer's ParticleSystem
    // consumes the result each frame.
    [[nodiscard]] std::vector<EmitterState> gatherEmitters() const;

    // True when at least one node in the tree carries a directional Light.
    // Used so FireEngine can avoid seeding its default Sun when a glTF asset
    // has already authored one (KHR_lights_punctual). Cheap; walks the tree
    // and short-circuits on first hit.
    [[nodiscard]] bool hasDirectionalLight() const;

private:
    std::vector<std::unique_ptr<Node>> nodes_;
    Mat4 rootTransform_{Mat4::identity()};
};

} // namespace fire_engine
