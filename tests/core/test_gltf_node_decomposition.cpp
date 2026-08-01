#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include <fire_engine/core/gltf_loader.hpp>
#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/scene/components.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_graph.hpp>

using fire_engine::Assets;
using fire_engine::Device;
using fire_engine::GltfLoader;
using fire_engine::InputState;
using fire_engine::Lighting;
using fire_engine::Node;
using fire_engine::PhysicsWorld;
using fire_engine::Resources;
using fire_engine::SceneGraph;

// ============================================================================================
// glTF node decomposition, through the PRODUCTION loader ([.][gpu], local only — `loadScene`
// needs a real `Resources`, so this cannot run on a CI container with no ICD; the rule itself is
// pinned headlessly in `test_node_component_layout.cpp`).
//
// The fixture (`tests/assets/node_component_decomposition.gltf`) authors one node per case a
// single-component variant has to resolve. THREE of them were broken: animated + light, static
// mesh + light, and animated mesh + light, where the light was attached while the node was Empty
// and then overwritten by `emplace<Animator>()` / `emplace<Mesh>()` with no warning, because the
// guard had already passed. The other two — camera + light, and animated camera — already worked;
// they are here as regression guards, since all five now share one placement rule and a change to
// it would break them together.
// ============================================================================================

namespace
{

// Depth-first search for a named node, so a test can assert on the SHAPE of the decomposition
// rather than on load order.
[[nodiscard]] const Node* findNode(const Node& node, std::string_view name)
{
    if (node.name() == name)
    {
        return &node;
    }
    for (const auto& child : node.children())
    {
        if (const Node* found = findNode(*child, name))
        {
            return found;
        }
    }
    return nullptr;
}

[[nodiscard]] const Node* findNode(const SceneGraph& scene, std::string_view name)
{
    for (const auto& root : scene.nodes())
    {
        if (const Node* found = findNode(*root, name))
        {
            return found;
        }
    }
    return nullptr;
}

[[nodiscard]] std::size_t countLights(const Node& node)
{
    std::size_t count = node.componentAs<fire_engine::Light>() != nullptr ? 1u : 0u;
    for (const auto& child : node.children())
    {
        count += countLights(*child);
    }
    return count;
}

} // namespace

TEST_CASE("GltfNodeDecomposition.EveryAuthoredComponentSurvivesTheLoad", "[.][gpu][GltfLoader]")
{
    Device device = Device::headlessCompute();
    Resources resources(device);
    SceneGraph scene;
    Assets assets;
    PhysicsWorld physics;
    GltfLoader::loadScene(std::string(FIRE_ENGINE_BUILD_ASSET_DIR) +
                              "/test_assets/node_component_decomposition.gltf",
                          scene, resources, assets, physics);

    SECTION("an animated light survives as a child of its animator")
    {
        const Node* animated = findNode(scene, "AnimatedLight");
        REQUIRE(animated != nullptr);
        CHECK(animated->componentAs<fire_engine::Animator>() != nullptr);
        const Node* light = findNode(scene, "AnimatedLight_Light");
        REQUIRE(light != nullptr);
        REQUIRE(light->componentAs<fire_engine::Light>() != nullptr);
        CHECK(light->componentAs<fire_engine::Light>()->type() ==
              fire_engine::Light::Type::Directional);
        // Exactly one, on the child: a rule that also left one on the parent would double the sun.
        CHECK(countLights(*animated) == 1u);
    }
    SECTION("a static mesh and light both survive")
    {
        const Node* node = findNode(scene, "StaticMeshLight");
        REQUIRE(node != nullptr);
        CHECK(node->componentAs<fire_engine::Mesh>() != nullptr);
        const Node* light = findNode(scene, "StaticMeshLight_Light");
        REQUIRE(light != nullptr);
        REQUIRE(light->componentAs<fire_engine::Light>() != nullptr);
        CHECK(light->componentAs<fire_engine::Light>()->type() == fire_engine::Light::Type::Point);
    }
    SECTION("a camera and light both survive")
    {
        const Node* node = findNode(scene, "CameraLight");
        REQUIRE(node != nullptr);
        // The light claims the node, the camera takes the child — the camera path already behaved
        // this way, and the rule now states it.
        CHECK(node->componentAs<fire_engine::Light>() != nullptr);
        const Node* camera = findNode(scene, "CameraLight_Camera");
        REQUIRE(camera != nullptr);
        CHECK(camera->componentAs<fire_engine::Camera>() != nullptr);
    }
    SECTION("an animated mesh and light become siblings under one animator")
    {
        const Node* node = findNode(scene, "AnimatedMeshLight");
        REQUIRE(node != nullptr);
        CHECK(node->componentAs<fire_engine::Animator>() != nullptr);
        CHECK(countLights(*node) == 1u);
        const Node* light = findNode(scene, "AnimatedMeshLight_Light");
        REQUIRE(light != nullptr);
        CHECK(light->componentAs<fire_engine::Light>() != nullptr);
        // The mesh is somewhere under the same animator, not displaced by the light.
        bool meshFound = false;
        for (const auto& child : node->children())
        {
            meshFound = meshFound || child->componentAs<fire_engine::Mesh>() != nullptr;
        }
        CHECK(meshFound);
    }
    SECTION("an animated camera still works, and still moves")
    {
        // Not a fix — a regression guard. This path already created a child; it must keep doing so
        // now that the light shares the same rule.
        const Node* node = findNode(scene, "AnimatedCamera");
        REQUIRE(node != nullptr);
        CHECK(node->componentAs<fire_engine::Animator>() != nullptr);
        const Node* camera = findNode(scene, "AnimatedCamera_Camera");
        REQUIRE(camera != nullptr);
        CHECK(camera->componentAs<fire_engine::Camera>() != nullptr);

        // Existing is not enough: a camera parked on an identity child of an animator would satisfy
        // every structural check above while sitting perfectly still. Advance the clock and require
        // the child's WORLD pose to follow its parent's animation.
        InputState input;
        input.time(0.0);
        scene.update(input);
        const fire_engine::Mat4 atRest = camera->composedWorld();
        input.time(0.5);
        scene.update(input);
        const fire_engine::Mat4 later = camera->composedWorld();

        // Compare the forward axis: the fixture rotates, so orientation is where the motion shows.
        const fire_engine::Vec3 restForward{-atRest[0, 2], -atRest[1, 2], -atRest[2, 2]};
        const fire_engine::Vec3 laterForward{-later[0, 2], -later[1, 2], -later[2, 2]};
        CHECK(fire_engine::Vec3::dotProduct(restForward, laterForward) < 0.99f);
    }
    SECTION("every authored light is gathered exactly once, with a stable identity")
    {
        InputState input;
        scene.update(input);
        const std::vector<Lighting> first = scene.gatherLights();
        // Four authored lights, four gathered: none lost, none duplicated.
        CHECK(first.size() == 4u);
        CHECK(scene.hasDirectionalLight());

        scene.update(input);
        const std::vector<Lighting> second = scene.gatherLights();
        REQUIRE(second.size() == first.size());
        for (std::size_t i = 0; i < first.size(); ++i)
        {
            // Shadow state is keyed on this: an identity that moved between frames would reset a
            // caster's hysteresis every frame.
            CHECK(second[i].nodeId == first[i].nodeId);
        }
    }
    SECTION("the animated light's direction actually follows its animation")
    {
        InputState input;
        input.time(0.0);
        scene.update(input);
        const auto atRest = scene.gatherLights();
        // The animator samples an ABSOLUTE clock (`InputState::time`), not a per-frame delta, so
        // advancing means moving that clock. Half a second lands mid-way through the fixture's
        // 0->1 s rotation keyframes, where the sampled orientation is unambiguously different.
        input.time(0.5);
        scene.update(input);
        const auto later = scene.gatherLights();
        REQUIRE(atRest.size() == later.size());

        const Lighting* restSun = nullptr;
        const Lighting* laterSun = nullptr;
        for (std::size_t i = 0; i < atRest.size(); ++i)
        {
            if (atRest[i].type == 0)
            {
                restSun = &atRest[i];
                laterSun = &later[i];
            }
        }
        REQUIRE(restSun != nullptr);
        // The whole point of keeping the light under the animator: the animation reaches it.
        const float dot =
            fire_engine::Vec3::dotProduct(restSun->worldDirection, laterSun->worldDirection);
        CHECK(dot < 0.99f);
    }
}
