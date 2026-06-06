#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>

#include <fire_engine/fire_engine.hpp>

#include <fire_engine/core/gltf_loader.hpp>
#include <fire_engine/core/system.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/object.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/scene/mesh.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_graph_format.hpp>

namespace fire_engine
{

// ---------------------------------------------------------------------------
// FireEngine
// ---------------------------------------------------------------------------

FireEngine::FireEngine()
{
    System::init();
}

FireEngine::~FireEngine()
{
    System::destroy();
}

void FireEngine::run(size_t width, size_t height, std::string_view app_name,
                     std::string_view scene_path, std::string_view skybox_path, bool addFloor,
                     RendererDebug debug)
{
    window_ = std::make_unique<Window>(width, height, app_name);

    renderer_ = std::make_unique<Renderer>(*window_, std::string(skybox_path), debug);

    loadScene(scene_path);
    if (addFloor)
    {
        addFloorPlane();
    }
    mainLoop();
}

void FireEngine::addFloorPlane()
{
    constexpr float halfSize = 5.0f;
    constexpr Colour3 white{1.0f, 1.0f, 1.0f};
    constexpr Vec3 normal{0.0f, 1.0f, 0.0f};

    // Material is fine to push into assets_ — backed by std::deque so existing
    // pointers stay valid across the insert.
    auto& mat = assets_.addMaterial(Material{});
    mat.name("FloorWhite");
    mat.baseColor(white);
    mat.alpha(1.0f);
    mat.roughness(1.0f);
    mat.metallic(0.0f);

    floorGeometry_ = std::make_unique<Geometry>();
    std::vector<Vertex> verts{
        Vertex{Vec3{-halfSize, 0.0f, -halfSize}, white, normal, Vec2{0.0f, 0.0f}},
        Vertex{Vec3{halfSize, 0.0f, -halfSize}, white, normal, Vec2{1.0f, 0.0f}},
        Vertex{Vec3{halfSize, 0.0f, halfSize}, white, normal, Vec2{1.0f, 1.0f}},
        Vertex{Vec3{-halfSize, 0.0f, halfSize}, white, normal, Vec2{0.0f, 1.0f}},
    };
    // CCW from above (+y) so the front face survives the forward pass'
    // back-face cull. The floor is a receiver-only debug plane; if it casts
    // into the CSM it self-shadows and greys out the whole scene.
    std::vector<uint32_t> indices{0, 2, 1, 0, 3, 2};
    floorGeometry_->vertices(std::move(verts));
    floorGeometry_->indices(std::move(indices));
    floorGeometry_->material(&mat);
    floorGeometry_->castsShadow(false);
    floorGeometry_->load(renderer_->resources());

    Object floorObject;
    floorObject.addGeometry(*floorGeometry_);
    floorObject.load(renderer_->resources());

    auto floorNode = std::make_unique<Node>("Floor");
    floorNode->component().emplace<Mesh>(std::move(floorObject));
    scene_.addNode(std::move(floorNode));
}

void FireEngine::loadScene(std::string_view scene_path)
{
    // Load glTF scene (CLI arg overrides default)
    constexpr std::string_view default_scene = "RiggedSimple/RiggedSimple.gltf";
    std::string_view path = scene_path.empty() ? default_scene : scene_path;
    Node* activeCamera =
        GltfLoader::loadScene(std::string(path), scene_, renderer_->resources(), assets_, physics_);

    if (activeCamera != nullptr)
    {
        camera_ = activeCamera->componentAs<Camera>();
    }

    if (camera_ == nullptr)
    {
        auto cameraNode = std::make_unique<Node>("Camera");
        auto& camera = cameraNode->component().emplace<Camera>();
        camera.localPosition({2.0f, 2.0f, 2.0f});
        camera.localPitch(-0.615f);
        camera.localYaw(-2.356f);
        scene_.addNode(std::move(cameraNode));
        camera_ = &camera;
    }

    // Seed default directional only when the asset didn't author its own
    // (KHR_lights_punctual). Aim local -Z along normalise(1, -1, 1) so the
    // sun appears above + behind the camera's typical starting orientation
    // and casts visible shadows on most loaded scenes.
    if (!scene_.hasDirectionalLight())
    {
        auto sunNode = std::make_unique<Node>("Sun");
        auto& sun = sunNode->component().emplace<Light>();
        sun.type(Light::Type::Directional);
        sun.colour(Colour3{1.0f, 1.0f, 1.0f});
        sun.intensity(kDirectionalLightIntensity);
        const Vec3 sunForward = Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f});
        sunNode->transform().rotation(Quaternion::fromVectors(Vec3{0.0f, 0.0f, -1.0f}, sunForward));
        scene_.addNode(std::move(sunNode));
    }

    std::print("{}\n", scene_);
}

void FireEngine::mainLoop()
{
    constexpr float fixedDt = 1.0f / 60.0f;
    constexpr float maxFrameTime = 0.25f;
    double lastTime = System::getTime();
    float accumulator = 0.0f;
    while (!window_->shouldClose())
    {
        double now = System::getTime();
        float dt = std::min(static_cast<float>(now - lastTime), maxFrameTime);
        lastTime = now;

        auto input_state = input_.update(*window_, dt);
        input_state.time(now);
        scene_.update(input_state);
        scene_.submitPhysics(physics_);

        accumulator += dt;
        while (accumulator >= fixedDt)
        {
            physics_.step(fixedDt);
            accumulator -= fixedDt;
        }

        scene_.applyPhysics(physics_);

        renderer_->drawFrame(*window_, scene_, camera_->worldPosition(), camera_->worldTarget());
    }
    renderer_->waitIdle();
}

} // namespace fire_engine
