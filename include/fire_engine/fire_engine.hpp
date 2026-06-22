#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/input/input.hpp>
#include <fire_engine/physics/character_controller.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/platform/window.hpp>
#include <fire_engine/render/renderer.hpp>
#include <fire_engine/scene/camera.hpp>
#include <fire_engine/scene/ragdoll.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------
class FireEngine
{
public:
    explicit FireEngine();
    ~FireEngine();

    FireEngine(const FireEngine&) = delete;
    FireEngine& operator=(const FireEngine&) = delete;
    FireEngine(FireEngine&&) noexcept = delete;
    FireEngine& operator=(FireEngine&&) noexcept = delete;

    void run(size_t width, size_t height, std::string_view app_name,
             std::string_view scene_path = "", std::string_view skybox_path = "",
             bool addFloor = false, bool addParticles = false, bool addCloth = false,
             bool addCharacter = false, RendererDebug debug = {});

private:
    std::unique_ptr<Window> window_;
    std::unique_ptr<Renderer> renderer_;
    Input input_;
    SceneGraph scene_;
    Assets assets_;
    PhysicsWorld physics_;
    Camera* camera_{nullptr};
    // Ragdolls auto-built from `extras.Ragdoll` skinned nodes. Retained for the
    // app's lifetime: they hold the bone-node ↔ body bindings (the bodies live in
    // physics_, but the Ragdoll owns the activation/override state).
    std::vector<Ragdoll> ragdolls_;

    // Floor plane — kept outside `assets_` because
    // resizing the asset vector after the glTF loader populated it would
    // invalidate every Object's cached Geometry pointer.
    std::unique_ptr<Geometry> floorGeometry_;
    // Demo cloth geometry (-c). Kept alive here for the same reason as the floor:
    // Object caches a Geometry pointer, so it must not move.
    std::unique_ptr<Geometry> clothGeometry_;
    // Demo collision sphere the cloth drapes over (-c).
    std::unique_ptr<Geometry> sphereGeometry_;

    // Character-controller demo (-k): the kinematic capsule (driven by character_), its
    // visible node, and the obstacle-course geometry. Geometries are kept here (not in
    // assets_) so Object's cached Geometry pointers stay stable.
    std::optional<CharacterController> character_;
    Node* characterNode_{nullptr};
    float characterVerticalVelocity_{0.0f};
    Vec3 characterWalkDir_{1.0f, 0.0f, 0.0f};
    std::unique_ptr<Geometry> characterGeometry_;
    std::vector<std::unique_ptr<Geometry>> courseGeometries_;

    void loadScene(std::string_view scene_path);
    void addFloorPlane();
    void addParticleFountain();
    void addClothDemo();
    void addCharacterDemo();
    void updateCharacter(float dt);
    void addTestCube();
    void mainLoop();
};

} // namespace fire_engine
