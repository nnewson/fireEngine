#pragma once

#include <cstddef>
#include <memory>

#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/input/input.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/platform/window.hpp>
#include <fire_engine/render/renderer.hpp>
#include <fire_engine/scene/camera.hpp>
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

    void run(size_t width, size_t height, std::string_view app_name,
             std::string_view scene_path = "", std::string_view skybox_path = "",
             bool addFloor = false, bool debugNormals = false, bool debugNdotL = false,
             bool debugShadow = false, bool debugShadowDepth = false, bool noShadows = false);

private:
    std::unique_ptr<Window> window_;
    std::unique_ptr<Renderer> renderer_;
    Input input_;
    SceneGraph scene_;
    Assets assets_;
    PhysicsWorld physics_;
    Camera* camera_{nullptr};

    // Floor plane — kept outside `assets_` because
    // resizing the asset vector after the glTF loader populated it would
    // invalidate every Object's cached Geometry pointer.
    std::unique_ptr<Geometry> floorGeometry_;

    void loadScene(std::string_view scene_path);
    void addFloorPlane();
    void addTestCube();
    void mainLoop();
};

} // namespace fire_engine
