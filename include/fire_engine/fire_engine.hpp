#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
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

struct RunOptions
{
    std::string_view scenePath{};
    std::string_view skyboxPath{};
    bool addFloor{false};
    bool addParticles{false};
    bool addCloth{false};
    bool addCharacter{false};
    bool addQueryProbe{false};
    bool startMaximized{false};
    RendererDebug debug{};
};

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

    void run(size_t width, size_t height, std::string_view appName, const RunOptions& options = {});

private:
    std::unique_ptr<Window> window_;
    std::unique_ptr<Renderer> renderer_;
    Input input_;
    // assets_ is declared before scene_ so it outlives it: scene_'s Objects cache Geometry*/
    // Material* pointers into assets_ (kept stable by its deques — see Assets), and must not
    // outlive what they point at.
    Assets assets_;
    SceneGraph scene_;
    PhysicsWorld physics_;
    Camera* camera_{nullptr};
    // Ragdolls auto-built from `extras.Ragdoll` skinned nodes. Retained for the
    // app's lifetime: they hold the bone-node ↔ body bindings (the bodies live in
    // physics_, but the Ragdoll owns the activation/override state).
    std::vector<Ragdoll> ragdolls_;

    // Character-controller demo (-k): the kinematic capsule (driven by character_) and its
    // visible node. Demo geometry (floor, cloth, sphere, capsule, obstacle course) lives in
    // assets_ now — its deques keep Object's cached Geometry* stable across additions.
    std::optional<CharacterController> character_;
    Node* characterNode_{nullptr};
    float characterVerticalVelocity_{0.0f};
    bool characterGrounded_{true};
    Vec3 characterWalkDir_{1.0f, 0.0f, 0.0f};

    // Query-probe demo (-q): a ring of static bodies queried each frame.
    bool queryProbeActive_{false};

    void loadScene(std::string_view scene_path);
    void addFloorPlane();
    void addParticleFountain();
    void addClothDemo();
    void addCharacterDemo();
    void addQueryProbeDemo();
    [[nodiscard]] std::vector<DebugLine> queryProbeLines(double time) const;
    void updateCharacter(float dt);
    void addTestCube();
    void mainLoop();
    void stepSimulation(float dt, float& accumulator);
    void syncRenderState(double time);
};

} // namespace fire_engine
