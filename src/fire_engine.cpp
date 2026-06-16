#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <print>

#include <fire_engine/fire_engine.hpp>

#include <fire_engine/core/gltf_loader.hpp>
#include <fire_engine/core/system.hpp>
#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/object.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/scene/mesh.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/particle_emitter.hpp>
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
                     bool addParticles, bool addCloth, RendererDebug debug)
{
    window_ = std::make_unique<Window>(width, height, app_name);

    renderer_ = std::make_unique<Renderer>(*window_, std::string(skybox_path), debug);

    loadScene(scene_path);
    if (addFloor)
    {
        addFloorPlane();
    }
    if (addParticles)
    {
        addParticleFountain();
    }
    if (addCloth)
    {
        addClothDemo();
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

void FireEngine::addParticleFountain()
{
    // Demo GPU particle fountain (Roadmap Milestone B), gated behind the -p flag:
    // a warm upward emitter just above the floor. Component defaults give the
    // fountain look; only the position is set here. The renderer's ParticleSystem
    // gathers and simulates it each frame.
    auto fountainNode = std::make_unique<Node>("ParticleFountain");
    fountainNode->component().emplace<ParticleEmitter>();
    fountainNode->transform().position(Vec3{0.0f, 0.1f, 0.0f});
    scene_.addNode(std::move(fountainNode));
}

namespace
{

// Procedural UV sphere (positions + outward normals + uv). Used as the visible
// prop the demo cloth drapes over.
void buildUvSphere(Geometry& geo, float radius, uint32_t stacks, uint32_t slices, Colour3 colour)
{
    constexpr float pi = std::numbers::pi_v<float>;
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i <= stacks; ++i)
    {
        const float v = static_cast<float>(i) / static_cast<float>(stacks);
        const float phi = v * pi; // 0 (north pole) .. pi
        for (uint32_t j = 0; j <= slices; ++j)
        {
            const float u = static_cast<float>(j) / static_cast<float>(slices);
            const float theta = u * 2.0f * pi;
            const Vec3 n{std::sin(phi) * std::cos(theta), std::cos(phi),
                         std::sin(phi) * std::sin(theta)};
            verts.emplace_back(n * radius, colour, n, Vec2{u, v});
        }
    }
    const uint32_t cols = slices + 1;
    for (uint32_t i = 0; i < stacks; ++i)
    {
        for (uint32_t j = 0; j < slices; ++j)
        {
            const uint32_t a = i * cols + j;
            const uint32_t b = a + cols;
            indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    }
    geo.vertices(std::move(verts));
    geo.indices(std::move(indices));
}

} // namespace

void FireEngine::addClothDemo()
{
    // Roadmap #2. A world-space unpinned cloth sheet falls and drapes over a
    // sphere (a Static physics body whose collider the solver gathers), pooling on
    // the ground plane (added in mainLoop). The solver writes the cloth's storage
    // vertex buffer each frame; it renders through the normal forward/shadow path.
    constexpr Vec3 sphereCenter{0.0f, 1.0f, 0.0f};
    constexpr float sphereRadius = 0.7f;

    // Visible sphere prop.
    auto& sphereMat = assets_.addMaterial(Material{});
    sphereMat.name("ClothSphere");
    sphereMat.baseColor(Colour3{0.25f, 0.3f, 0.4f});
    sphereMat.roughness(0.6f);
    sphereMat.metallic(0.0f);
    sphereGeometry_ = std::make_unique<Geometry>();
    buildUvSphere(*sphereGeometry_, sphereRadius, 24, 32, Colour3{1.0f, 1.0f, 1.0f});
    sphereGeometry_->material(&sphereMat);
    sphereGeometry_->load(renderer_->resources());
    Object sphereObject;
    sphereObject.addGeometry(*sphereGeometry_);
    sphereObject.load(renderer_->resources());
    auto sphereNode = std::make_unique<Node>("ClothSphere");
    sphereNode->component().emplace<Mesh>(std::move(sphereObject));
    sphereNode->transform().position(sphereCenter);
    scene_.addNode(std::move(sphereNode));

    // Static physics body + sphere collider the cloth solver gathers each frame.
    const PhysicsBodyHandle body = physics_.createBody(
        PhysicsBodyDesc{.type = PhysicsBodyType::Static, .position = sphereCenter});
    (void)physics_.createCollider(body, ColliderDesc{.shape = SphereShape{.radius = sphereRadius}});

    // World-space cloth sheet above the sphere, unpinned.
    ClothGridParams params;
    params.resX = 40;
    params.resZ = 40;
    params.spacing = 0.05f;
    params.pinTopCorners = false;
    params.origin = Vec3{0.0f, 2.5f, 0.0f};
    ClothMesh cloth = makeGridCloth(params);

    auto& mat = assets_.addMaterial(Material{});
    mat.name("ClothDemo");
    mat.baseColor(Colour3{0.85f, 0.2f, 0.25f});
    mat.alpha(1.0f);
    mat.roughness(0.9f);
    mat.metallic(0.0f);
    mat.doubleSided(true); // cloth is visible from both sides

    clothGeometry_ = std::make_unique<Geometry>();
    clothGeometry_->vertices(std::move(cloth.vertices));
    clothGeometry_->indices(std::move(cloth.indices));
    clothGeometry_->material(&mat);
    clothGeometry_->storageVertices(true);
    clothGeometry_->load(renderer_->resources());

    Object clothObject;
    clothObject.addGeometry(*clothGeometry_);
    clothObject.load(renderer_->resources());

    // The solver writes the storage vertex buffer; cloth simulates in world space,
    // so the node transform stays identity.
    renderer_->addCloth(cloth, clothGeometry_->vertexBuffer());

    auto clothNode = std::make_unique<Node>("ClothDemo");
    clothNode->component().emplace<Mesh>(std::move(clothObject));
    scene_.addNode(std::move(clothNode));
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
    bool f1Down = false;
    while (!window_->shouldClose())
    {
        double now = System::getTime();
        float dt = std::min(static_cast<float>(now - lastTime), maxFrameTime);
        lastTime = now;

        // F1 toggles the debug overlay (edge-detected so a held key fires once).
        const bool f1 = glfwGetKey(window_->handle(), GLFW_KEY_F1) == GLFW_PRESS;
        if (f1 && !f1Down)
        {
            renderer_->toggleOverlay();
        }
        f1Down = f1;

        // Suppress camera/keyboard movement while the overlay is capturing input
        // so dragging a widget doesn't also fly the camera.
        auto input_state = input_.update(*window_, dt, renderer_->overlayWantsMouse(),
                                         renderer_->overlayWantsKeyboard());
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

        // World colliders for the cloth solver: physics bodies + the ground plane.
        auto clothColliders = physics_.gatherColliders();
        clothColliders.push_back(makePlaneCollider(Vec3{0.0f, 1.0f, 0.0f}, 0.0f));
        renderer_->setClothColliders(clothColliders);

        renderer_->drawFrame(*window_, scene_, camera_->worldPosition(), camera_->worldTarget(),
                             dt);
    }
    renderer_->waitIdle();
}

} // namespace fire_engine
