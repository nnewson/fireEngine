#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <print>
#include <string>

#include <fire_engine/fire_engine.hpp>

#include <fire_engine/collision/ray.hpp>
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
    renderer_.reset();
    window_.reset();
    System::destroy();
}

void FireEngine::run(size_t width, size_t height, std::string_view app_name,
                     std::string_view scene_path, std::string_view skybox_path, bool addFloor,
                     bool addParticles, bool addCloth, bool addCharacter, bool addQueryProbe,
                     RendererDebug debug)
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
    if (addCharacter)
    {
        addCharacterDemo();
    }
    if (addQueryProbe)
    {
        addQueryProbeDemo();
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
    // Demo GPU particle fountain, gated behind the -p flag: a warm upward emitter
    // just above the floor. Component defaults give the
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

// Axis-aligned box mesh (per-face normals) centred at the origin. Used to build the
// visible obstacle course for the character demo.
void buildBox(Geometry& geo, Vec3 half, Colour3 colour)
{
    const std::array<Vec3, 6> normals{Vec3{1, 0, 0},  Vec3{-1, 0, 0}, Vec3{0, 1, 0},
                                      Vec3{0, -1, 0}, Vec3{0, 0, 1},  Vec3{0, 0, -1}};
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    for (const Vec3& n : normals)
    {
        // Two in-plane axes for this face.
        const Vec3 u{std::abs(n.y()) + std::abs(n.z()), std::abs(n.x()), 0.0f};
        const Vec3 w = Vec3::crossProduct(n, u);
        const Vec3 c{n.x() * half.x(), n.y() * half.y(), n.z() * half.z()};
        const auto base = static_cast<uint32_t>(verts.size());
        for (const auto& s : {std::pair{-1.0f, -1.0f}, std::pair{1.0f, -1.0f},
                              std::pair{1.0f, 1.0f}, std::pair{-1.0f, 1.0f}})
        {
            const Vec3 p{c.x() + (u.x() * s.first + w.x() * s.second) * half.x(),
                         c.y() + (u.y() * s.first + w.y() * s.second) * half.y(),
                         c.z() + (u.z() * s.first + w.z() * s.second) * half.z()};
            verts.emplace_back(p, colour, n, Vec2{0.0f, 0.0f});
        }
        indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    geo.vertices(std::move(verts));
    geo.indices(std::move(indices));
}

} // namespace

void FireEngine::addClothDemo()
{
    // A world-space unpinned cloth sheet falls and drapes over a
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
    clothGeometry_->indices(cloth.indices); // copy: addCloth still needs the indices
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

void FireEngine::addCharacterDemo()
{
    Resources& res = renderer_->resources();

    // A visible static box that is also a static physics collider (one course piece).
    const auto addBox = [&](const char* name, Vec3 center, Vec3 half, Colour3 colour,
                            Quaternion rotation = Quaternion::identity())
    {
        auto& mat = assets_.addMaterial(Material{});
        mat.name(name);
        mat.baseColor(colour);
        mat.alpha(1.0f);
        mat.roughness(0.9f);
        mat.metallic(0.0f);

        auto geo = std::make_unique<Geometry>();
        buildBox(*geo, half, Colour3{1.0f, 1.0f, 1.0f});
        geo->material(&mat);
        geo->load(res);

        Object obj;
        obj.addGeometry(*geo);
        obj.load(res);
        courseGeometries_.push_back(std::move(geo));

        auto node = std::make_unique<Node>(name);
        node->component().emplace<Mesh>(std::move(obj));
        node->transform().position(center);
        node->transform().rotation(rotation);
        scene_.addNode(std::move(node));

        PhysicsBodyDesc body;
        body.type = PhysicsBodyType::Static;
        body.position = center;
        body.rotation = rotation;
        const PhysicsBodyHandle handle = physics_.createBody(body);
        (void)physics_.createCollider(handle, ColliderDesc{.shape = BoxShape{half, Vec3{}}});
    };

    // Patrol course: a step pyramid the character climbs up and down, walled at both ends on
    // the floor. Each step rises 0.3 (≤ the controller's 0.35 step offset); the hardened
    // step-up (sweep-and-rest onto the step edge) mounts them reliably at walk speed, and the
    // timed direction flip turns the patrol around. Gated by the headless
    // PatrolsStepPyramidWithoutWedging controller test.
    addBox("CharFloor", {5.0f, -0.5f, 0.0f}, {10.0f, 0.5f, 5.0f}, Colour3{0.5f, 0.5f, 0.55f});
    addBox("CharStepA", {3.0f, 0.15f, 0.0f}, {0.5f, 0.15f, 3.0f}, Colour3{0.45f, 0.5f, 0.45f});
    addBox("CharStepB", {4.0f, 0.30f, 0.0f}, {0.5f, 0.30f, 3.0f}, Colour3{0.45f, 0.55f, 0.45f});
    addBox("CharPeak", {5.5f, 0.45f, 0.0f}, {1.0f, 0.45f, 3.0f}, Colour3{0.5f, 0.6f, 0.5f});
    addBox("CharStepC", {7.0f, 0.30f, 0.0f}, {0.5f, 0.30f, 3.0f}, Colour3{0.45f, 0.55f, 0.45f});
    addBox("CharStepD", {8.0f, 0.15f, 0.0f}, {0.5f, 0.15f, 3.0f}, Colour3{0.45f, 0.5f, 0.45f});

    // Visible character marker — a sphere; the controller's capsule is virtual (the
    // character has no physics body, so its own queries never self-hit).
    auto& charMat = assets_.addMaterial(Material{});
    charMat.name("Character");
    charMat.baseColor(Colour3{0.9f, 0.7f, 0.2f});
    charMat.alpha(1.0f);
    charMat.roughness(0.4f);
    charMat.metallic(0.0f);
    characterGeometry_ = std::make_unique<Geometry>();
    buildUvSphere(*characterGeometry_, 0.5f, 16, 24, Colour3{1.0f, 1.0f, 1.0f});
    characterGeometry_->material(&charMat);
    characterGeometry_->load(res);

    Object charObject;
    charObject.addGeometry(*characterGeometry_);
    charObject.load(res);

    // Start grounded on the floor (top y = 0; capsule centre rests at height/2 = 0.9).
    const Vec3 start{2.0f, 0.9f, 0.0f};
    auto charNode = std::make_unique<Node>("Character");
    charNode->component().emplace<Mesh>(std::move(charObject));
    charNode->transform().position(start);
    characterNode_ = &scene_.addNode(std::move(charNode));

    character_.emplace(CharacterControllerConfig{}, start);
}

void FireEngine::addQueryProbeDemo()
{
    Resources& res = renderer_->resources();

    // A ring of static sphere bodies (visible meshes + physics colliders) for the
    // raycasts to hit.
    const auto addSphere = [&](const char* name, Vec3 center, float radius, Colour3 colour)
    {
        auto& mat = assets_.addMaterial(Material{});
        mat.name(name);
        mat.baseColor(colour);
        mat.alpha(1.0f);
        mat.roughness(0.5f);
        mat.metallic(0.0f);

        auto geo = std::make_unique<Geometry>();
        buildUvSphere(*geo, radius, 16, 24, Colour3{1.0f, 1.0f, 1.0f});
        geo->material(&mat);
        geo->load(res);

        Object obj;
        obj.addGeometry(*geo);
        obj.load(res);
        queryProbeGeometries_.push_back(std::move(geo));

        auto node = std::make_unique<Node>(name);
        node->component().emplace<Mesh>(std::move(obj));
        node->transform().position(center);
        scene_.addNode(std::move(node));

        PhysicsBodyDesc body;
        body.type = PhysicsBodyType::Static;
        body.position = center;
        const PhysicsBodyHandle handle = physics_.createBody(body);
        (void)physics_.createCollider(handle, ColliderDesc{.shape = SphereShape{radius, Vec3{}}});
    };

    // Eight spheres at varied radii/positions around the origin (some gaps, so some
    // rays hit and some miss). Heights all at y = 0.7 so the planar ray fan reaches them.
    constexpr int kCount = 8;
    for (int i = 0; i < kCount; ++i)
    {
        const float ang = static_cast<float>(i) * (2.0f * std::numbers::pi_v<float> / kCount);
        const float dist = 3.5f + 0.6f * static_cast<float>(i % 3); // 3.5 / 4.1 / 4.7
        const float r = 0.45f + 0.15f * static_cast<float>(i % 2);
        const Vec3 c{dist * std::cos(ang), 0.7f, dist * std::sin(ang)};
        const Colour3 col{0.35f + 0.07f * static_cast<float>(i % 3), 0.45f, 0.6f};
        addSphere(("Probe" + std::to_string(i)).c_str(), c, r, col);
    }
    queryProbeActive_ = true;
}

std::vector<DebugLine> FireEngine::queryProbeLines(double time) const
{
    // A rotating fan of raycasts sweeping the XZ plane from a central origin: each ray
    // draws green to its hit (with a small marker) or faint to its max range on a miss.
    std::vector<DebugLine> lines;
    if (!queryProbeActive_)
    {
        return lines;
    }
    constexpr int kRays = 64;
    constexpr float kRange = 8.0f;
    const Vec3 origin{0.0f, 0.7f, 0.0f};
    const float sweep = static_cast<float>(time) * 0.4f; // slow rotation
    for (int i = 0; i < kRays; ++i)
    {
        const float ang =
            sweep + static_cast<float>(i) * (2.0f * std::numbers::pi_v<float> / kRays);
        const Vec3 dir{std::cos(ang), 0.0f, std::sin(ang)};
        const auto hit = physics_.raycast(Ray{origin, dir, kRange});
        if (hit.has_value())
        {
            lines.push_back({origin, hit->point, Colour3{0.2f, 0.9f, 0.35f}});
            // A small cross marking the hit point.
            const Colour3 mark{1.0f, 0.85f, 0.1f};
            lines.push_back(
                {hit->point - Vec3{0.12f, 0.0f, 0.0f}, hit->point + Vec3{0.12f, 0.0f, 0.0f}, mark});
            lines.push_back(
                {hit->point - Vec3{0.0f, 0.0f, 0.12f}, hit->point + Vec3{0.0f, 0.0f, 0.12f}, mark});
        }
        else
        {
            lines.push_back({origin, origin + dir * kRange, Colour3{0.22f, 0.25f, 0.32f}});
        }
    }
    return lines;
}

void FireEngine::updateCharacter(float dt)
{
    if (!character_.has_value() || characterNode_ == nullptr)
    {
        return;
    }
    constexpr float kSpeed = 3.0f;
    constexpr float kGravity = -9.8f;
    constexpr float kNearBound = 1.5f;
    constexpr float kFarBound = 9.5f;

    // Advance once per render frame by the *real* frame time, so the visible motion is smooth at
    // any refresh rate. (A fixed-timestep accumulator was tried, to make the climb deterministic;
    // but above 60 fps it does 0 sim steps on some frames and 2 on others, so the ball advances
    // in uneven 0/0.1/0.2 m chunks — a visible stutter, worst during the slow climb. The
    // controller is robust to any per-frame distance, so stepping at the frame rate is both
    // smoother and simpler.)

    // Bounds-based patrol: turn around only at the flat ends of the course, *past* the step
    // pyramid (which spans x≈2.5–8.5). Driving the turn from position rather than a fixed timer
    // means the character never reverses partway up a stair. Setting the direction (rather than
    // toggling) is idempotent, so it can't flip-flop while sitting past a bound.
    const float patrolX = character_->position().x();
    if (patrolX > kFarBound)
    {
        characterWalkDir_ = Vec3{-1.0f, 0.0f, 0.0f};
    }
    else if (patrolX < kNearBound)
    {
        characterWalkDir_ = Vec3{1.0f, 0.0f, 0.0f};
    }

    // Gravity only while airborne. A grounded character must not push *down* into the floor every
    // frame — that fights the resting contact and (with the rounded capsule grazing the ground)
    // can wedge it. Vertical rest is owned by the controller's ground snap; gravity kicks in only
    // once the snap reports the character has left the ground, and resets on landing.
    if (characterGrounded_)
    {
        characterVerticalVelocity_ = 0.0f;
    }
    else
    {
        characterVerticalVelocity_ += kGravity * dt;
    }
    const Vec3 horizontal = characterWalkDir_ * (kSpeed * dt);
    const float verticalStep = characterGrounded_ ? 0.0f : characterVerticalVelocity_ * dt;

    const CharacterMoveResult result =
        character_->move(physics_, horizontal + Vec3{0.0f, verticalStep, 0.0f});
    characterGrounded_ = result.grounded;

    characterNode_->transform().position(character_->position());
    characterNode_->transform().update(scene_.rootTransform());
}

void FireEngine::loadScene(std::string_view scene_path)
{
    // Load glTF scene (CLI arg overrides default)
    constexpr std::string_view default_scene = "RiggedSimple/RiggedSimple.gltf";
    std::string_view path = scene_path.empty() ? default_scene : scene_path;
    std::vector<GltfLoader::ClothRegistration> clothRegistrations;
    Node* activeCamera = GltfLoader::loadScene(std::string(path), scene_, renderer_->resources(),
                                               assets_, physics_, &clothRegistrations, &ragdolls_);

    // Register any glTF `extras.Cloth` meshes with the soft-body solver. The
    // geometry (Assets-owned) keeps its storage vertex buffer; the solver writes it.
    for (auto& reg : clothRegistrations)
    {
        renderer_->addCloth(reg.mesh, reg.geometry->vertexBuffer());
    }

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
        updateCharacter(dt);
        scene_.submitPhysics(physics_);

        accumulator += dt;
        while (accumulator >= fixedDt)
        {
            physics_.step(fixedDt);
            accumulator -= fixedDt;
        }

        scene_.applyPhysics(physics_);

        // World colliders for the cloth solver: physics bodies + the ground plane.
        auto colliders = physics_.gatherColliders();
        // Physics debug draw uses the authored shapes (pre-plane); only gather the
        // rest of the debug data when a debug-draw category is enabled. The query-probe
        // rays draw independently of the --debug-physics categories.
        if (renderer_->physicsDebugWanted() || queryProbeActive_)
        {
            PhysicsDebugData debugData;
            if (renderer_->physicsDebugWanted())
            {
                debugData.aabbs = physics_.debugColliderBounds();
                debugData.shapes = colliders;
                debugData.contacts = physics_.debugContacts();
                debugData.shapesAsleep = physics_.debugColliderSleeping();
            }
            debugData.queryLines = queryProbeLines(now);
            renderer_->setPhysicsDebug(std::move(debugData));
        }
        colliders.push_back(makePlaneCollider(Vec3{0.0f, 1.0f, 0.0f}, 0.0f));
        renderer_->setClothColliders(colliders);

        renderer_->drawFrame(*window_, scene_, camera_->worldPosition(), camera_->worldTarget(),
                             dt);
    }
    renderer_->waitIdle();
}

} // namespace fire_engine
