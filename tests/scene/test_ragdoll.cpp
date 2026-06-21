#include <fire_engine/scene/ragdoll.hpp>

#include <memory>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_graph.hpp>

using fire_engine::InputState;
using fire_engine::Mat4;
using fire_engine::Node;
using fire_engine::PhysicsWorld;
using fire_engine::Ragdoll;
using fire_engine::SceneGraph;
using fire_engine::Vec3;

namespace
{

constexpr float kDt = 1.0f / 120.0f;

Vec3 bodyPos(const PhysicsWorld& physics, const Ragdoll& rag, std::size_t i)
{
    return physics.bodyTransform(rag.body(i))->position();
}

// Translation column of a composed-world matrix.
Vec3 nodeWorldPos(const Node& node)
{
    const Mat4& w = node.composedWorld();
    return {w[0, 3], w[1, 3], w[2, 3]};
}

} // namespace

TEST_CASE("Ragdoll.BuildsBodyAndJointPerBone", "[Ragdoll]")
{
    SceneGraph sg;
    Node& b0 = sg.addNode(std::make_unique<Node>("bone0"));
    b0.transform().position({0.0f, 2.0f, 0.0f});
    Node& b1 = b0.addChild(std::make_unique<Node>("bone1"));
    b1.transform().position({0.0f, -0.3f, 0.0f});
    Node& b2 = b1.addChild(std::make_unique<Node>("bone2"));
    b2.transform().position({0.0f, -0.3f, 0.0f});

    sg.update(InputState{});

    PhysicsWorld physics;
    const std::vector<Node*> bones{&b0, &b1, &b2};
    const Ragdoll rag = Ragdoll::make(physics, bones);

    CHECK(rag.boneCount() == 3U);
    CHECK(physics.bodyCount() == 3U);
    // A joint links each non-root bone to its parent: 2 joints for a 3-bone chain.
    CHECK(physics.jointCount() == 2U);
    CHECK_FALSE(rag.joint(0).valid()); // root has no parent joint
    CHECK(rag.joint(1).valid());
    CHECK(rag.joint(2).valid());

    // Bodies seed at the bones' world positions.
    CHECK(bodyPos(physics, rag, 0).y() == Catch::Approx(2.0f));
    CHECK(bodyPos(physics, rag, 1).y() == Catch::Approx(1.7f));
    CHECK(bodyPos(physics, rag, 2).y() == Catch::Approx(1.4f));
}

TEST_CASE("Ragdoll.ActivateSetsWorldOverrideOnBones", "[Ragdoll]")
{
    SceneGraph sg;
    Node& b0 = sg.addNode(std::make_unique<Node>("bone0"));
    b0.transform().position({0.0f, 2.0f, 0.0f});
    Node& b1 = b0.addChild(std::make_unique<Node>("bone1"));
    b1.transform().position({0.0f, -0.3f, 0.0f});
    sg.update(InputState{});

    PhysicsWorld physics;
    const std::vector<Node*> bones{&b0, &b1};
    Ragdoll rag = Ragdoll::make(physics, bones);

    CHECK_FALSE(b0.hasWorldOverride());
    rag.activate();
    CHECK(rag.active());
    CHECK(b0.hasWorldOverride());
    CHECK(b1.hasWorldOverride());

    rag.deactivate();
    CHECK_FALSE(rag.active());
    CHECK_FALSE(b0.hasWorldOverride());
}

TEST_CASE("Ragdoll.ChainFallsAndStaysConnected", "[Ragdoll]")
{
    SceneGraph sg;
    Node& b0 = sg.addNode(std::make_unique<Node>("bone0"));
    b0.transform().position({0.0f, 2.0f, 0.0f});
    Node& b1 = b0.addChild(std::make_unique<Node>("bone1"));
    b1.transform().position({0.0f, -0.3f, 0.0f});
    Node& b2 = b1.addChild(std::make_unique<Node>("bone2"));
    b2.transform().position({0.0f, -0.3f, 0.0f});

    sg.update(InputState{});

    PhysicsWorld physics;
    const std::vector<Node*> bones{&b0, &b1, &b2};
    Ragdoll rag = Ragdoll::make(physics, bones);
    rag.activate();

    const float startY = bodyPos(physics, rag, 0).y();

    for (int i = 0; i < 120; ++i)
    {
        physics.step(kDt);
        sg.applyPhysics(physics);
    }

    // The whole chain fell under gravity.
    CHECK(bodyPos(physics, rag, 0).y() < startY - 0.2f);

    // The ball-socket joints kept the bones connected: adjacent bodies stay roughly a
    // bone length (0.3) apart rather than flying apart or collapsing.
    const float d01 = (bodyPos(physics, rag, 0) - bodyPos(physics, rag, 1)).magnitude();
    const float d12 = (bodyPos(physics, rag, 1) - bodyPos(physics, rag, 2)).magnitude();
    CHECK(d01 == Catch::Approx(0.3f).margin(0.05f));
    CHECK(d12 == Catch::Approx(0.3f).margin(0.05f));

    // The world-override drove the bone nodes: each node's composed-world translation
    // tracks its body (so the skinning path renders the simulated pose).
    for (std::size_t i = 0; i < rag.boneCount(); ++i)
    {
        const Vec3 nodePos = nodeWorldPos(*rag.node(i));
        const Vec3 physicsPos = bodyPos(physics, rag, i);
        CHECK(nodePos.x() == Catch::Approx(physicsPos.x()).margin(1e-4f));
        CHECK(nodePos.y() == Catch::Approx(physicsPos.y()).margin(1e-4f));
        CHECK(nodePos.z() == Catch::Approx(physicsPos.z()).margin(1e-4f));
    }
}
