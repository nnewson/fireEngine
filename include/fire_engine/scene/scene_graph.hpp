#pragma once

#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/graphics/particle.hpp>
#include <fire_engine/graphics/renderable_scene.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_culler.hpp>

namespace fire_engine
{

class SceneGraph : public RenderableScene
{
public:
    SceneGraph() = default;
    ~SceneGraph() override = default;

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

    // RenderableScene (the Vulkan-free render seam, CR-09). buildDrawCommands walks the tree
    // emitting draws into `out`; it culls internally against `frustums` (camera + shadow-caster
    // frustums) using the scene's own bounds, an empty span meaning "cull disabled, render all".
    // The culled-node set stays entirely internal; only DrawCommands cross the boundary.
    void gatherLights(std::vector<Lighting>& out) const override;
    void gatherShadowCasters(ShadowCasterBoundsFrame& out) const override;
    void gatherEmitters(std::vector<EmitterState>& out) const override;
    [[nodiscard]] CullStats buildDrawCommands(const FrameInfo& frame,
                                              std::span<const Frustum> frustums,
                                              const ShadowCasterBoundsFrame& casterBounds,
                                              std::vector<DrawCommand>& out) override;
    [[nodiscard]] CameraView activeCamera() const override;

    // Select the node the renderer views the scene from. The node must carry a Camera component
    // (asserted); nullptr clears the selection, after which activeCamera() returns a fixed debug
    // fallback. The camera is a normal scene node — moving it (e.g. a future route system animating
    // its Transform) moves the view automatically, since activeCamera() reads its live world pose.
    void activeCamera(Node* cameraNode) noexcept;

    [[nodiscard]] std::vector<Lighting> gatherLights() const;
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
    // The node the renderer views from (non-owning; the node lives in nodes_). nullptr = no camera
    // authored → activeCamera() returns a fixed debug fallback.
    Node* activeCameraNode_{nullptr};
};

} // namespace fire_engine
