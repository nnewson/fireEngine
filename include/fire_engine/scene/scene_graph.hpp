#pragma once

#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/graphics/particle.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/render/render_context.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_culler.hpp>

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
    // Write body-bound node poses from the physics world, interpolated by `alpha` towards the
    // latest simulated pose (CR-20). `alpha = accumulator / fixedDt`; pass 1.0f to snap to the
    // last simulated state.
    void applyPhysics(const PhysicsWorld& physics, float alpha = 1.0f);
    void render(const RenderContext& ctx);

    // Refresh the scene culler and return the rigid renderable nodes outside every
    // frustum (camera + shadow casters). The renderer hands the result back through
    // RenderContext::culledNodes so render() can skip their draw-building. The returned
    // reference is owned by the culler and valid until the next cull()/render().
    [[nodiscard]] const std::unordered_set<const Node*>& cull(std::span<const Frustum> frustums);

    [[nodiscard]] const SceneCuller& culler() const noexcept
    {
        return culler_;
    }

    // Walk the scene tree and resolve every Light component into a world-space
    // Lighting. Composed world matrices are taken from each Node's cached
    // composedWorld_ (populated by the most recent update() call). Cheap —
    // light counts are tiny compared to draw counts.
    void gatherLights(std::vector<Lighting>& out) const;
    [[nodiscard]] std::vector<Lighting> gatherLights() const;

    // Walk the scene tree and resolve every ParticleEmitter component into a
    // world-space EmitterState (translation + node-rotated velocity from the
    // cached composedWorld_). Mirrors gatherLights; the renderer's ParticleSystem
    // consumes the result each frame.
    void gatherEmitters(std::vector<EmitterState>& out) const;
    [[nodiscard]] std::vector<EmitterState> gatherEmitters() const;

    // True when at least one node in the tree carries a directional Light.
    // Used so FireEngine can avoid seeding its default Sun when a glTF asset
    // has already authored one (KHR_lights_punctual). Cheap; walks the tree
    // and short-circuits on first hit.
    [[nodiscard]] bool hasDirectionalLight() const;

private:
    std::vector<std::unique_ptr<Node>> nodes_;
    Mat4 rootTransform_{Mat4::identity()};
    SceneCuller culler_;
};

} // namespace fire_engine
