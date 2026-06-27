#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/render_context.hpp>
#include <fire_engine/scene/components.hpp>
#include <fire_engine/scene/light.hpp>

using fire_engine::Colour3;
using fire_engine::componentName;
using fire_engine::Components;
using fire_engine::Light;
using fire_engine::Lighting;
using fire_engine::Mat4;
using fire_engine::RenderContext;
using fire_engine::Vec3;

TEST_CASE("Light.DefaultsAreDirectionalWhiteUnit", "[Light]")
{
    Light l;
    CHECK(l.type() == Light::Type::Directional);
    CHECK(l.colour() == Colour3(1.0f, 1.0f, 1.0f));
    CHECK(l.intensity() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(l.range() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Light.TypeRoundTrip", "[Light]")
{
    Light l;
    l.type(Light::Type::Point);
    CHECK(l.type() == Light::Type::Point);
    l.type(Light::Type::Spot);
    CHECK(l.type() == Light::Type::Spot);
    l.type(Light::Type::Directional);
    CHECK(l.type() == Light::Type::Directional);
}

TEST_CASE("Light.ColourRoundTrip", "[Light]")
{
    Light l;
    l.colour(Colour3(0.2f, 0.5f, 0.8f));
    CHECK(l.colour() == Colour3(0.2f, 0.5f, 0.8f));
}

TEST_CASE("Light.IntensityRoundTrip", "[Light]")
{
    Light l;
    l.intensity(12.5f);
    CHECK(l.intensity() == Catch::Approx(12.5f).margin(1e-5f));
}

TEST_CASE("Light.RangeRoundTrip", "[Light]")
{
    Light l;
    l.range(7.5f);
    CHECK(l.range() == Catch::Approx(7.5f).margin(1e-5f));
}

TEST_CASE("Light.ConeAnglesDefaultMatchKhrPunctualSensible", "[Light]")
{
    Light l;
    CHECK(l.innerConeRad() == Catch::Approx(fire_engine::pi / 8.0f).margin(1e-5f));
    CHECK(l.outerConeRad() == Catch::Approx(fire_engine::pi / 4.0f).margin(1e-5f));
}

TEST_CASE("Light.OuterConeStaysGreaterOrEqualToInnerWhenInnerIncreases", "[Light]")
{
    Light l;
    l.outerConeRad(fire_engine::pi / 6.0f);
    l.innerConeRad(fire_engine::pi / 3.0f);
    CHECK(l.innerConeRad() == Catch::Approx(fire_engine::pi / 3.0f).margin(1e-5f));
    CHECK(l.outerConeRad() == Catch::Approx(fire_engine::pi / 3.0f).margin(1e-5f));
}

TEST_CASE("Light.InnerConeStaysLessOrEqualToOuterWhenOuterDecreases", "[Light]")
{
    Light l;
    l.innerConeRad(fire_engine::pi / 3.0f);
    l.outerConeRad(fire_engine::pi / 6.0f);
    CHECK(l.innerConeRad() == Catch::Approx(fire_engine::pi / 6.0f).margin(1e-5f));
    CHECK(l.outerConeRad() == Catch::Approx(fire_engine::pi / 6.0f).margin(1e-5f));
}

TEST_CASE("Components.LightVariantNameIsLight", "[Components]")
{
    Components c = Light{};
    CHECK(componentName(c) == "Light");
}

// ---------------------------------------------------------------------------
// E2: gather pass — Light::toLighting resolves world-space data
// ---------------------------------------------------------------------------

TEST_CASE("LightGather.IdentityWorldDirectionalForwardIsNegativeZ", "[LightGather]")
{
    Light l;
    l.type(Light::Type::Directional);
    auto inst = Light::toLighting(l, Mat4::identity());
    CHECK(inst.type == 0);
    CHECK(inst.worldPosition.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(inst.worldPosition.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(inst.worldPosition.z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(inst.worldDirection.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(inst.worldDirection.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(inst.worldDirection.z() == Catch::Approx(-1.0f).margin(1e-5f));
}

TEST_CASE("LightGather.TranslatedPointKeepsForwardSetsPosition", "[LightGather]")
{
    Light l;
    l.type(Light::Type::Point);
    l.range(8.0f);
    auto world = Mat4::translate(Vec3{3.0f, 4.0f, 5.0f});
    auto inst = Light::toLighting(l, world);
    CHECK(inst.type == 1);
    CHECK(inst.worldPosition.x() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(inst.worldPosition.y() == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(inst.worldPosition.z() == Catch::Approx(5.0f).margin(1e-5f));
    CHECK(inst.range == Catch::Approx(8.0f).margin(1e-5f));
}

TEST_CASE("LightGather.RotatedNodeRotatesForward", "[LightGather]")
{
    // Yaw 90° → forward rotates from -Z to -X (matches glTF camera convention).
    Light l;
    l.type(Light::Type::Spot);
    auto world = Mat4::rotateY(fire_engine::pi * 0.5f);
    auto inst = Light::toLighting(l, world);
    CHECK(inst.type == 2);
    CHECK(inst.worldDirection.x() == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK(inst.worldDirection.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(inst.worldDirection.z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("LightGather.ConeAnglesPackAsCosines", "[LightGather]")
{
    Light l;
    l.type(Light::Type::Spot);
    l.outerConeRad(fire_engine::pi / 3.0f);
    l.innerConeRad(fire_engine::pi / 6.0f);
    auto inst = Light::toLighting(l, Mat4::identity());
    CHECK(inst.innerConeCos == Catch::Approx(std::cos(fire_engine::pi / 6.0f)).margin(1e-6f));
    CHECK(inst.outerConeCos == Catch::Approx(std::cos(fire_engine::pi / 3.0f)).margin(1e-6f));
}

TEST_CASE("LightGather.ColourAndIntensityRoundTrip", "[LightGather]")
{
    Light l;
    l.colour(Colour3(0.2f, 0.4f, 0.6f));
    l.intensity(7.5f);
    auto inst = Light::toLighting(l, Mat4::identity());
    CHECK(inst.colour == Colour3(0.2f, 0.4f, 0.6f));
    CHECK(inst.intensity == Catch::Approx(7.5f).margin(1e-5f));
}

TEST_CASE("LightGather.ScaledNodeStillEmitsUnitForward", "[LightGather]")
{
    Light l;
    auto world = Mat4::scale(Vec3{4.0f, 4.0f, 4.0f});
    auto inst = Light::toLighting(l, world);
    CHECK(inst.worldDirection.magnitude() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(inst.worldDirection.z() == Catch::Approx(-1.0f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// E3: SceneGraph::gatherLights walks the tree and resolves world-space light
// data from each Light component's node.composedWorld().
// ---------------------------------------------------------------------------

#include <fire_engine/input/input_state.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_graph.hpp>

using fire_engine::InputState;
using fire_engine::Node;
using fire_engine::SceneGraph;

TEST_CASE("GatherLights.EmptySceneReturnsNoLights", "[GatherLights]")
{
    SceneGraph scene;
    auto lights = scene.gatherLights();
    CHECK(lights.empty());
}

TEST_CASE("GatherLights.SingleDirectionalAtRoot", "[GatherLights]")
{
    SceneGraph scene;
    auto node = std::make_unique<Node>("Sun");
    node->component().emplace<Light>().intensity(2.5f);
    scene.addNode(std::move(node));

    InputState input;
    scene.update(input);

    auto lights = scene.gatherLights();
    REQUIRE(lights.size() == 1u);
    CHECK(lights[0].type == 0);
    CHECK(lights[0].intensity == Catch::Approx(2.5f).margin(1e-5f));
}

TEST_CASE("GatherLights.OutputVectorIsClearedAndReused", "[GatherLights]")
{
    SceneGraph scene;
    auto node = std::make_unique<Node>("Sun");
    node->component().emplace<Light>().intensity(4.0f);
    scene.addNode(std::move(node));

    InputState input;
    scene.update(input);

    std::vector<Lighting> lights(3);
    const auto previousCapacity = lights.capacity();
    scene.gatherLights(lights);

    REQUIRE(lights.size() == 1u);
    CHECK(lights.capacity() >= previousCapacity);
    CHECK(lights[0].intensity == Catch::Approx(4.0f).margin(1e-5f));
}

TEST_CASE("GatherLights.NestedLightUsesComposedWorldPosition", "[GatherLights]")
{
    SceneGraph scene;
    auto root = std::make_unique<Node>("Root");
    root->transform().position(Vec3{2.0f, 0.0f, 0.0f});
    auto child = std::make_unique<Node>("Lamp");
    child->transform().position(Vec3{0.0f, 3.0f, 0.0f});
    auto& childLight = child->component().emplace<Light>();
    childLight.type(Light::Type::Point);
    childLight.range(5.0f);
    root->addChild(std::move(child));
    scene.addNode(std::move(root));

    InputState input;
    scene.update(input);

    auto lights = scene.gatherLights();
    REQUIRE(lights.size() == 1u);
    CHECK(lights[0].type == 1);
    CHECK(lights[0].worldPosition.x() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(lights[0].worldPosition.y() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(lights[0].worldPosition.z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(lights[0].range == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("GatherLights.MultipleLightsPreserveTraversalOrder", "[GatherLights]")
{
    SceneGraph scene;
    auto a = std::make_unique<Node>("A");
    a->component().emplace<Light>().intensity(1.0f);
    auto b = std::make_unique<Node>("B");
    auto& bl = b->component().emplace<Light>();
    bl.type(Light::Type::Spot);
    bl.intensity(2.0f);
    scene.addNode(std::move(a));
    scene.addNode(std::move(b));

    InputState input;
    scene.update(input);

    auto lights = scene.gatherLights();
    REQUIRE(lights.size() == 2u);
    CHECK(lights[0].intensity == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(lights[1].intensity == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(lights[1].type == 2);
}

TEST_CASE("GatherLights.NonLightNodesAreSkipped", "[GatherLights]")
{
    SceneGraph scene;
    auto a = std::make_unique<Node>("Empty"); // default Empty component
    scene.addNode(std::move(a));
    auto b = std::make_unique<Node>("Light");
    b->component().emplace<Light>();
    scene.addNode(std::move(b));

    InputState input;
    scene.update(input);

    auto lights = scene.gatherLights();
    REQUIRE(lights.size() == 1u);
}

// ---------------------------------------------------------------------------
// D1 / E3: hasDirectionalLight — used by FireEngine to decide whether to seed
// the default Sun when KHR_lights_punctual already authored a directional.
// ---------------------------------------------------------------------------

TEST_CASE("HasDirectionalLight.EmptySceneReturnsFalse", "[HasDirectionalLight]")
{
    SceneGraph scene;
    CHECK_FALSE(scene.hasDirectionalLight());
}

TEST_CASE("HasDirectionalLight.SceneWithOnlyPointLightReturnsFalse", "[HasDirectionalLight]")
{
    SceneGraph scene;
    auto node = std::make_unique<Node>("Lamp");
    auto& l = node->component().emplace<Light>();
    l.type(Light::Type::Point);
    scene.addNode(std::move(node));
    CHECK_FALSE(scene.hasDirectionalLight());
}

TEST_CASE("HasDirectionalLight.SceneWithDirectionalReturnsTrue", "[HasDirectionalLight]")
{
    SceneGraph scene;
    auto node = std::make_unique<Node>("Sun");
    node->component().emplace<Light>(); // default type = Directional
    scene.addNode(std::move(node));
    CHECK(scene.hasDirectionalLight());
}

TEST_CASE("HasDirectionalLight.FindsDirectionalNestedAsChild", "[HasDirectionalLight]")
{
    SceneGraph scene;
    auto root = std::make_unique<Node>("Root");
    auto child = std::make_unique<Node>("Sun");
    child->component().emplace<Light>();
    root->addChild(std::move(child));
    scene.addNode(std::move(root));
    CHECK(scene.hasDirectionalLight());
}
