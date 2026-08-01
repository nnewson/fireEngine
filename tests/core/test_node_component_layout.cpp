#include <catch2/catch_test_macros.hpp>

#include <fire_engine/core/node_component_layout.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/scene/components.hpp>
#include <fire_engine/scene/node.hpp>

using fire_engine::materializeNodeComponentLayout;
using fire_engine::Node;
using fire_engine::NodeComponentLayout;
using fire_engine::NodeComponentTargets;
using fire_engine::planNodeComponents;
using Primary = fire_engine::NodeComponentLayout::Primary;

// The rule that replaces "whatever the attach order happened to be". Exhaustive over all sixteen
// combinations, because the defect it fixes was precisely a combination nobody had enumerated: a
// glTF node carrying BOTH a light and something that claims the single component slot silently lost
// the light, and `ShadowLodMotionDemo` then rendered every measurement under a fallback sun.
TEST_CASE("NodeComponentLayout.EveryCombinationPlacesEveryPieceOfContent", "[NodeComponentLayout]")
{
    for (int mask = 0; mask < 16; ++mask)
    {
        const bool anim = (mask & 1) != 0;
        const bool mesh = (mask & 2) != 0;
        const bool light = (mask & 4) != 0;
        const bool camera = (mask & 8) != 0;
        INFO("anim=" << anim << " mesh=" << mesh << " light=" << light << " camera=" << camera);

        const NodeComponentLayout layout = planNodeComponents(anim, mesh, light, camera);

        // NOTHING IS LOST. Each piece of content the glTF node declared ends up either on the node
        // or on a child of it — the property whose absence was the bug.
        const bool meshPlaced = layout.primary == Primary::Mesh || layout.meshOnChild;
        const bool lightPlaced = layout.primary == Primary::Light || layout.lightOnChild;
        const bool cameraPlaced = layout.primary == Primary::Camera || layout.cameraOnChild;
        CHECK(meshPlaced == mesh);
        CHECK(lightPlaced == light);
        CHECK(cameraPlaced == camera);
        // And an animated node always owns its Animator, since a child cannot animate its parent.
        CHECK((layout.primary == Primary::Animator) == anim);

        // Nothing is placed twice, and nothing is invented: the primary slot is never also a child.
        const bool meshTwice = layout.primary == Primary::Mesh && layout.meshOnChild;
        const bool lightTwice = layout.primary == Primary::Light && layout.lightOnChild;
        const bool cameraTwice = layout.primary == Primary::Camera && layout.cameraOnChild;
        CHECK_FALSE(meshTwice);
        CHECK_FALSE(lightTwice);
        CHECK_FALSE(cameraTwice);
    }
}

TEST_CASE("NodeComponentLayout.PrecedenceIsByWhatCannotMove", "[NodeComponentLayout]")
{
    SECTION("an animated light rides a child of its animator")
    {
        // The ShadowLodMotionDemo case exactly: a node with a rotation channel and a directional
        // light. Before the rule, the light was attached and then overwritten by the Animator.
        const auto layout = planNodeComponents(true, false, true, false);
        CHECK(layout.primary == Primary::Animator);
        CHECK(layout.lightOnChild);
        CHECK_FALSE(layout.meshOnChild);
    }
    SECTION("a static mesh and light both survive")
    {
        // The other half of the same defect, with no animation involved: `emplace<Mesh>` used to
        // overwrite the light attached moments earlier.
        const auto layout = planNodeComponents(false, true, true, false);
        CHECK(layout.primary == Primary::Mesh);
        CHECK(layout.lightOnChild);
    }
    SECTION("an animated mesh and light become two children of one animator")
    {
        const auto layout = planNodeComponents(true, true, true, false);
        CHECK(layout.primary == Primary::Animator);
        CHECK(layout.meshOnChild);
        CHECK(layout.lightOnChild);
    }
    SECTION("a camera yields the node to anything else")
    {
        CHECK(planNodeComponents(false, false, false, true).primary == Primary::Camera);
        CHECK_FALSE(planNodeComponents(false, false, false, true).cameraOnChild);
        // Light + camera: the light takes the node, the camera takes a child — the behaviour the
        // camera path already had, now stated rather than implied by call order.
        const auto withLight = planNodeComponents(false, false, true, true);
        CHECK(withLight.primary == Primary::Light);
        CHECK(withLight.cameraOnChild);
    }
    SECTION("weight-only animation is not transform animation")
    {
        // Morph weights are driven through the Mesh component, so a weight-animated node needs no
        // Animator and lays out as a plain mesh node. Passing `hasTransformAnim = false` is the
        // caller's job; this pins what the rule does with it.
        const auto layout = planNodeComponents(false, true, false, false);
        CHECK(layout.primary == Primary::Mesh);
        CHECK_FALSE(layout.meshOnChild);
    }
    SECTION("a node carrying nothing stays empty")
    {
        const auto layout = planNodeComponents(false, false, false, false);
        CHECK(layout.primary == Primary::Empty);
        CHECK_FALSE(layout.meshOnChild);
        CHECK_FALSE(layout.lightOnChild);
        CHECK_FALSE(layout.cameraOnChild);
    }
}

// The plan is only worth anything if PRODUCTION applies it, so the materialisation that the loader
// calls is exercised here on real nodes — headlessly, which is the point: the `[.][gpu]` loader
// fixture confirms the same shapes end to end, but CI can only run this one, and this one covers
// the whole topology rather than a single flag.
TEST_CASE("NodeComponentLayout.MaterialisationPlacesEveryPayloadOnAnEmptyTarget",
          "[NodeComponentLayout]")
{
    for (int mask = 0; mask < 16; ++mask)
    {
        const bool anim = (mask & 1) != 0;
        const bool mesh = (mask & 2) != 0;
        const bool light = (mask & 4) != 0;
        const bool camera = (mask & 8) != 0;
        INFO("anim=" << anim << " mesh=" << mesh << " light=" << light << " camera=" << camera);

        Node node("Thing");
        const NodeComponentLayout layout = planNodeComponents(anim, mesh, light, camera);
        const NodeComponentTargets targets = materializeNodeComponentLayout(node, layout);

        // A target exists for exactly the payloads the glTF node declared — never a fallback.
        CHECK((targets.mesh != nullptr) == mesh);
        CHECK((targets.light != nullptr) == light);
        CHECK((targets.camera != nullptr) == camera);

        // Every target is EMPTY and therefore safe to emplace into. This is the property that
        // makes attach order irrelevant, which is the whole fix: two payloads can never be handed
        // the same node, so neither can overwrite the other.
        const Node* seen[3] = {targets.mesh, targets.light, targets.camera};
        for (int i = 0; i < 3; ++i)
        {
            if (seen[i] == nullptr)
            {
                continue;
            }
            CHECK(seen[i]->componentAs<fire_engine::Empty>() != nullptr);
            for (int j = i + 1; j < 3; ++j)
            {
                const bool sameTarget = seen[j] != nullptr && seen[i] == seen[j];
                CHECK_FALSE(sameTarget);
            }
        }

        // Children are created only where the layout called for one, and each is IDENTITY: the
        // glTF transform stays on the parent, so an animated parent drives all of them together.
        const std::size_t expectedChildren = static_cast<std::size_t>(layout.meshOnChild) +
                                             static_cast<std::size_t>(layout.lightOnChild) +
                                             static_cast<std::size_t>(layout.cameraOnChild);
        CHECK(node.children().size() == expectedChildren);
        for (const auto& child : node.children())
        {
            CHECK(child->transform().position() == fire_engine::Vec3{});
            CHECK(child->transform().rotation() == fire_engine::Quaternion{});
        }
    }
}

TEST_CASE("NodeComponentLayout.MaterialisationNamesAndParentsTheChildren", "[NodeComponentLayout]")
{
    Node node("Lamp");
    const NodeComponentLayout layout = planNodeComponents(true, true, true, true);
    const NodeComponentTargets targets = materializeNodeComponentLayout(node, layout);

    // The animated node keeps itself for the Animator; everything else hangs off it.
    CHECK(targets.mesh != &node);
    CHECK(targets.light != &node);
    CHECK(targets.camera != &node);
    REQUIRE(node.children().size() == 3u);
    CHECK(targets.mesh->name() == "Lamp_Mesh");
    CHECK(targets.light->name() == "Lamp_Light");
    CHECK(targets.camera->name() == "Lamp_Camera");

    // glTF meshes carry their own names and the loader prefers them, so the mesh child can be named
    // by the caller; the light and camera children never are.
    Node named("Lamp");
    const NodeComponentTargets namedTargets =
        materializeNodeComponentLayout(named, layout, "TeapotMesh");
    CHECK(namedTargets.mesh->name() == "TeapotMesh");
    CHECK(namedTargets.light->name() == "Lamp_Light");
}

TEST_CASE("NodeComponentLayout.MaterialisationKeepsSoleContentOnTheNode", "[NodeComponentLayout]")
{
    // No children when nothing competes: a static mesh node stays one node, which is what every
    // existing scene expects.
    Node meshOnly("Rock");
    const NodeComponentTargets mesh =
        materializeNodeComponentLayout(meshOnly, planNodeComponents(false, true, false, false));
    CHECK(mesh.mesh == &meshOnly);
    CHECK(meshOnly.children().empty());

    Node lightOnly("Sun");
    const NodeComponentTargets sun =
        materializeNodeComponentLayout(lightOnly, planNodeComponents(false, false, true, false));
    CHECK(sun.light == &lightOnly);
    CHECK(lightOnly.children().empty());

    Node cameraOnly("Eye");
    const NodeComponentTargets eye =
        materializeNodeComponentLayout(cameraOnly, planNodeComponents(false, false, false, true));
    CHECK(eye.camera == &cameraOnly);
    CHECK(cameraOnly.children().empty());
}
