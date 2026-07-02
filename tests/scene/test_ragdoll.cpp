#include <fire_engine/scene/ragdoll.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <simdjson.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/core/gltf_loader.hpp>
#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/physics/articulation.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_graph.hpp>

using fire_engine::GltfLoader;
using fire_engine::InputState;
using fire_engine::Mat4;
using fire_engine::Node;
using fire_engine::PhysicsWorld;
using fire_engine::Ragdoll;
using fire_engine::RagdollParams;
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

Vec3 colliderPoint(const fire_engine::ClothCollider& collider, bool first)
{
    const float* p = first ? collider.a : collider.b;
    return {p[0], p[1], p[2]};
}

Vec3 colliderCenter(const fire_engine::ClothCollider& collider)
{
    return (colliderPoint(collider, true) + colliderPoint(collider, false)) * 0.5f;
}

Vec3 colliderAxis(const fire_engine::ClothCollider& collider)
{
    return Vec3::normalise(colliderPoint(collider, false) - colliderPoint(collider, true));
}

std::vector<int> parentIndices(std::span<Node* const> bones)
{
    std::unordered_map<const Node*, int> indexOf;
    for (std::size_t i = 0; i < bones.size(); ++i)
    {
        indexOf[bones[i]] = static_cast<int>(i);
    }

    std::vector<int> parents(bones.size(), -1);
    for (std::size_t i = 0; i < bones.size(); ++i)
    {
        for (const Node* p = bones[i]->parent(); p != nullptr; p = p->parent())
        {
            const auto it = indexOf.find(p);
            if (it != indexOf.end())
            {
                parents[i] = it->second;
                break;
            }
        }
    }
    return parents;
}

std::vector<int> firstChildIndices(std::span<Node* const> bones, std::span<const int> parents)
{
    std::vector<int> children(bones.size(), -1);
    for (std::size_t i = 0; i < bones.size(); ++i)
    {
        if (parents[i] < 0)
        {
            continue;
        }
        const auto p = static_cast<std::size_t>(parents[i]);
        if (children[p] < 0)
        {
            children[p] = static_cast<int>(i);
        }
    }
    return children;
}

fastgltf::Asset parseGltfAsset(const std::filesystem::path& gltfPath)
{
    fastgltf::Parser parser;
    auto dataResult = fastgltf::GltfDataBuffer::FromPath(gltfPath);
    REQUIRE(dataResult.error() == fastgltf::Error::None);

    auto result = parser.loadGltf(dataResult.get(), gltfPath.parent_path(),
                                  fastgltf::Options::LoadExternalBuffers);
    REQUIRE(result.error() == fastgltf::Error::None);
    return std::move(result.get());
}

RagdollParams parseGeneratedRagdollParams(const std::filesystem::path& gltfPath)
{
    simdjson::dom::parser parser;
    auto json = simdjson::padded_string::load(gltfPath.string());
    REQUIRE(json.error() == simdjson::SUCCESS);
    auto doc = parser.parse(json.value());
    REQUIRE(doc.error() == simdjson::SUCCESS);

    simdjson::dom::array nodes;
    REQUIRE(doc["nodes"].get_array().get(nodes) == simdjson::SUCCESS);
    for (simdjson::dom::element nodeElement : nodes)
    {
        simdjson::dom::object node;
        REQUIRE(nodeElement.get_object().get(node) == simdjson::SUCCESS);
        if (node.at_key("skin").error() == simdjson::NO_SUCH_FIELD)
        {
            continue;
        }

        simdjson::dom::object extras;
        REQUIRE(node.at_key("extras").get_object().get(extras) == simdjson::SUCCESS);
        auto params = GltfLoader::nodeExtrasRagdoll(&extras);
        REQUIRE(params.has_value());
        return *params;
    }

    FAIL("RagdollDemo.gltf has no skinned node with extras.Ragdoll");
    return {};
}

void applyGltfTransform(const fastgltf::Node& gltfNode, Node& node)
{
    if (const auto* trs = std::get_if<fastgltf::TRS>(&gltfNode.transform))
    {
        node.transform().position(
            {trs->translation.x(), trs->translation.y(), trs->translation.z()});
        node.transform().rotation(
            {trs->rotation.x(), trs->rotation.y(), trs->rotation.z(), trs->rotation.w()});
        node.transform().scale({trs->scale.x(), trs->scale.y(), trs->scale.z()});
    }
    else if (const auto* mat = std::get_if<fastgltf::math::fmat4x4>(&gltfNode.transform))
    {
        fastgltf::math::fvec3 scale;
        fastgltf::math::fquat rotation;
        fastgltf::math::fvec3 translation;
        fastgltf::math::decomposeTransformMatrix(*mat, scale, rotation, translation);

        node.transform().position({translation.x(), translation.y(), translation.z()});
        node.transform().rotation({rotation.x(), rotation.y(), rotation.z(), rotation.w()});
        node.transform().scale({scale.x(), scale.y(), scale.z()});
    }
}

Node& addGltfNode(SceneGraph& scene, const fastgltf::Asset& asset, std::size_t nodeIndex,
                  std::unordered_map<std::size_t, Node*>& nodeMap)
{
    const fastgltf::Node& gltfNode = asset.nodes[nodeIndex];
    auto node = std::make_unique<Node>(std::string{gltfNode.name});
    applyGltfTransform(gltfNode, *node);
    Node& ref = scene.addNode(std::move(node));
    nodeMap[nodeIndex] = &ref;
    return ref;
}

void addGltfChildren(Node& parent, const fastgltf::Asset& asset, std::size_t parentIndex,
                     std::unordered_map<std::size_t, Node*>& nodeMap)
{
    const fastgltf::Node& gltfParent = asset.nodes[parentIndex];
    for (const std::size_t childIndex : gltfParent.children)
    {
        const fastgltf::Node& gltfChild = asset.nodes[childIndex];
        auto child = std::make_unique<Node>(std::string{gltfChild.name});
        applyGltfTransform(gltfChild, *child);
        Node& childRef = parent.addChild(std::move(child));
        nodeMap[childIndex] = &childRef;
        addGltfChildren(childRef, asset, childIndex, nodeMap);
    }
}

std::vector<Node*> buildRagdollDemoBones(SceneGraph& scene, const fastgltf::Asset& asset)
{
    std::unordered_map<std::size_t, Node*> nodeMap;
    const auto sceneIndex = asset.defaultScene.value_or(0);
    for (const std::size_t nodeIndex : asset.scenes[sceneIndex].nodeIndices)
    {
        Node& root = addGltfNode(scene, asset, nodeIndex, nodeMap);
        addGltfChildren(root, asset, nodeIndex, nodeMap);
    }
    scene.resolve();

    const std::size_t skinnedNode = static_cast<std::size_t>(std::distance(
        asset.nodes.begin(), std::ranges::find_if(asset.nodes, [](const auto& node)
                                                  { return node.skinIndex.has_value(); })));
    REQUIRE(skinnedNode < asset.nodes.size());
    const auto skinIndex = asset.nodes[skinnedNode].skinIndex.value();
    REQUIRE(skinIndex < asset.skins.size());

    std::vector<Node*> bones;
    bones.reserve(asset.skins[skinIndex].joints.size());
    for (const std::size_t jointNodeIndex : asset.skins[skinIndex].joints)
    {
        const auto it = nodeMap.find(jointNodeIndex);
        REQUIRE(it != nodeMap.end());
        bones.push_back(it->second);
    }
    return bones;
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

TEST_CASE("Ragdoll.HumanoidSettlesOnFloor", "[Ragdoll]")
{
    // P9.1 gate: a complex articulated ragdoll (17 bones / 16 joints, branching) dropped
    // onto a floor must come to REST. With the old hard-Baumgarte joints this never
    // happened — the joint correction pumped energy, leaving a ~0.4–1.5 m/s limit cycle
    // forever. Soft/compliant joints dissipate instead of pumping, so the island settles.
    SceneGraph sg;
    std::vector<Node*> bones;
    const auto add = [&](Node* parent, const char* name, Vec3 local) -> Node*
    {
        auto node = std::make_unique<Node>(name);
        Node* p = parent ? &parent->addChild(std::move(node)) : &sg.addNode(std::move(node));
        p->transform().position(local);
        bones.push_back(p);
        return p;
    };
    Node* pelvis = add(nullptr, "pelvis", {0.0f, 1.6f, 0.0f});
    Node* spine = add(pelvis, "spine", {0.0f, 0.3f, 0.0f});
    Node* chest = add(spine, "chest", {0.0f, 0.3f, 0.0f});
    Node* neck = add(chest, "neck", {0.0f, 0.25f, 0.0f});
    add(neck, "head", {0.0f, 0.2f, 0.0f});
    Node* shL = add(chest, "shL", {0.2f, 0.05f, 0.0f});
    Node* uaL = add(shL, "uaL", {0.25f, 0.0f, 0.0f});
    add(uaL, "faL", {0.25f, 0.0f, 0.0f});
    Node* shR = add(chest, "shR", {-0.2f, 0.05f, 0.0f});
    Node* uaR = add(shR, "uaR", {-0.25f, 0.0f, 0.0f});
    add(uaR, "faR", {-0.25f, 0.0f, 0.0f});
    Node* hipL = add(pelvis, "hipL", {0.12f, -0.05f, 0.0f});
    Node* thL = add(hipL, "thL", {0.0f, -0.35f, 0.0f});
    add(thL, "shinL", {0.0f, -0.35f, 0.0f});
    Node* hipR = add(pelvis, "hipR", {-0.12f, -0.05f, 0.0f});
    Node* thR = add(hipR, "thR", {0.0f, -0.35f, 0.0f});
    add(thR, "shinR", {0.0f, -0.35f, 0.0f});
    sg.update(InputState{});

    PhysicsWorld physics;
    fire_engine::PhysicsBodyDesc floor;
    floor.type = fire_engine::PhysicsBodyType::Static;
    floor.position = {0.0f, -0.25f, 0.0f};
    floor.material = fire_engine::PhysicsMaterial{.restitution = 0.0f, .friction = 0.6f};
    const auto floorBody = physics.createBody(floor);
    static_cast<void>(physics.createCollider(
        floorBody,
        fire_engine::ColliderDesc{.shape = fire_engine::BoxShape{Vec3{6.0f, 0.25f, 6.0f}, {}}}));

    Ragdoll rag = Ragdoll::make(physics, bones);
    rag.activate();

    constexpr float dt = 1.0f / 60.0f;
    for (int i = 0; i < 900; ++i) // 15 s — fall, flop, and (now) settle
    {
        physics.step(dt);
        sg.applyPhysics(physics);
    }

    float vmax = 0.0f;
    for (std::size_t i = 0; i < rag.boneCount(); ++i)
    {
        const Vec3 v = physics.body(rag.body(i))->linearVelocity();
        vmax = std::max(vmax, v.magnitude());
        CHECK(std::isfinite(bodyPos(physics, rag, i).y()));
    }
    CHECK(vmax < 0.05f); // came to rest — no joint-driven limit cycle
}

TEST_CASE("Ragdoll.GeneratedSingleJointDemoAssetRestsOnFloor", "[Ragdoll][Demos]")
{
    const std::filesystem::path gltfPath = "../assets/physics_demos/SingleJointRagdollDemo.gltf";
    const fastgltf::Asset asset = parseGltfAsset(gltfPath);
    const RagdollParams params = parseGeneratedRagdollParams(gltfPath);

    SceneGraph sg;
    std::vector<Node*> bones = buildRagdollDemoBones(sg, asset);
    REQUIRE(bones.size() == 1U);

    PhysicsWorld physics;
    fire_engine::PhysicsBodyDesc floor;
    floor.type = fire_engine::PhysicsBodyType::Static;
    floor.position = {0.0f, -0.25f, 0.0f};
    floor.material = fire_engine::PhysicsMaterial{.restitution = 0.0f, .friction = 0.6f};
    const auto floorBody = physics.createBody(floor);
    static_cast<void>(physics.createCollider(
        floorBody,
        fire_engine::ColliderDesc{.shape = fire_engine::BoxShape{Vec3{4.0f, 0.25f, 4.0f}, {}}}));

    Ragdoll rag = Ragdoll::make(physics, bones, params);
    REQUIRE(rag.boneCount() == 1U);
    CHECK(physics.bodyCount() == 2U);
    CHECK(physics.jointCount() == 0U);
    CHECK(physics.debugJointAnchors().empty());

    const auto initialColliders = physics.gatherColliders();
    const auto capsuleCount = std::ranges::count_if(
        initialColliders, [](const auto& collider)
        { return collider.type == std::to_underlying(fire_engine::ClothColliderType::Capsule); });
    CHECK(capsuleCount == 1);

    rag.activate();

    constexpr float dt = 1.0f / 60.0f;
    for (int i = 0; i < 900; ++i)
    {
        physics.step(dt);
        sg.applyPhysics(physics);
    }

    const Vec3 finalPos = bodyPos(physics, rag, 0);
    const float finalSpeed = physics.body(rag.body(0))->linearVelocity().magnitude();
    const float finalSpin = physics.body(rag.body(0))->angularVelocity().magnitude();
    CHECK(std::isfinite(finalPos.y()));
    CHECK(finalPos.y() > 0.04f);
    CHECK(finalPos.y() < 0.4f);
    CHECK(finalSpeed < 0.05f);
    CHECK(finalSpin < 0.05f);
    CHECK(physics.sleeping(rag.body(0)));
}

TEST_CASE("Ragdoll.GeneratedTwoJointDemoAssetStaysConnected", "[Ragdoll][Demos]")
{
    const std::filesystem::path gltfPath = "../assets/physics_demos/TwoJointRagdollDemo.gltf";
    const fastgltf::Asset asset = parseGltfAsset(gltfPath);
    const RagdollParams params = parseGeneratedRagdollParams(gltfPath);

    SceneGraph sg;
    std::vector<Node*> bones = buildRagdollDemoBones(sg, asset);
    REQUIRE(bones.size() == 2U);

    PhysicsWorld physics;
    fire_engine::PhysicsBodyDesc floor;
    floor.type = fire_engine::PhysicsBodyType::Static;
    floor.position = {0.0f, -0.25f, 0.0f};
    floor.material = fire_engine::PhysicsMaterial{.restitution = 0.0f, .friction = 0.6f};
    const auto floorBody = physics.createBody(floor);
    static_cast<void>(physics.createCollider(
        floorBody,
        fire_engine::ColliderDesc{.shape = fire_engine::BoxShape{Vec3{4.0f, 0.25f, 4.0f}, {}}}));

    Ragdoll rag = Ragdoll::make(physics, bones, params);
    REQUIRE(rag.boneCount() == 2U);
    CHECK(physics.bodyCount() == 3U);
    CHECK(physics.jointCount() == 1U);

    auto jointAnchors = physics.debugJointAnchors();
    REQUIRE(jointAnchors.size() == 1U);
    CHECK((jointAnchors[0].anchorA - jointAnchors[0].anchorB).magnitude() < 1.0e-4f);
    CHECK((jointAnchors[0].originA - jointAnchors[0].originB).magnitude() ==
          Catch::Approx(0.4f).margin(1.0e-4f));

    rag.activate();

    constexpr float dt = 1.0f / 60.0f;
    float maxJointStretch = 0.0f;
    for (int i = 0; i < 240; ++i)
    {
        physics.step(dt);
        sg.applyPhysics(physics);

        jointAnchors = physics.debugJointAnchors();
        REQUIRE(jointAnchors.size() == 1U);
        maxJointStretch = std::max(maxJointStretch,
                                   (jointAnchors[0].anchorA - jointAnchors[0].anchorB).magnitude());
    }

    jointAnchors = physics.debugJointAnchors();
    REQUIRE(jointAnchors.size() == 1U);
    const float finalJointStretch = (jointAnchors[0].anchorA - jointAnchors[0].anchorB).magnitude();
    const float finalOriginDistance =
        (jointAnchors[0].originA - jointAnchors[0].originB).magnitude();

    for (std::size_t i = 0; i < rag.boneCount(); ++i)
    {
        CHECK(std::isfinite(bodyPos(physics, rag, i).y()));
    }

    INFO("max joint stretch: " << maxJointStretch);
    INFO("final joint stretch: " << finalJointStretch);
    INFO("final origin distance: " << finalOriginDistance);
    CHECK(finalJointStretch < 0.05f);
    CHECK(maxJointStretch < 0.5f);
    CHECK(finalOriginDistance == Catch::Approx(0.4f).margin(0.08f));
}

TEST_CASE("Ragdoll.GeneratedDemoAssetSettlesOnFloor", "[Ragdoll][Demos]")
{
    const std::filesystem::path gltfPath = "../assets/physics_demos/RagdollDemo.gltf";
    const fastgltf::Asset asset = parseGltfAsset(gltfPath);
    const RagdollParams params = parseGeneratedRagdollParams(gltfPath);

    SceneGraph sg;
    std::vector<Node*> bones = buildRagdollDemoBones(sg, asset);
    REQUIRE(bones.size() == 17U);

    PhysicsWorld physics;
    fire_engine::PhysicsBodyDesc floor;
    floor.type = fire_engine::PhysicsBodyType::Static;
    floor.position = {0.0f, -0.25f, 0.0f};
    floor.material = fire_engine::PhysicsMaterial{.restitution = 0.0f, .friction = 0.6f};
    const auto floorBody = physics.createBody(floor);
    static_cast<void>(physics.createCollider(
        floorBody,
        fire_engine::ColliderDesc{.shape = fire_engine::BoxShape{Vec3{6.0f, 0.25f, 6.0f}, {}}}));

    Ragdoll rag = Ragdoll::make(physics, bones, params);
    REQUIRE(rag.boneCount() == 17U);
    rag.activate();

    const std::vector<int> parents = parentIndices(bones);
    const std::vector<int> firstChildren = firstChildIndices(bones, parents);
    const auto initialColliders = physics.gatherColliders();
    const auto capsuleCount = std::ranges::count_if(
        initialColliders, [](const auto& collider)
        { return collider.type == std::to_underlying(fire_engine::ClothColliderType::Capsule); });
    CHECK(capsuleCount == static_cast<std::ptrdiff_t>(rag.boneCount()));
    for (std::size_t i = 0; i < rag.boneCount(); ++i)
    {
        Vec3 expectedDir{0.0f, 1.0f, 0.0f};
        if (parents[i] >= 0)
        {
            expectedDir = nodeWorldPos(*bones[i]) -
                          nodeWorldPos(*bones[static_cast<std::size_t>(parents[i])]);
        }
        if (expectedDir.magnitudeSquared() < 1.0e-8f && firstChildren[i] >= 0)
        {
            expectedDir = nodeWorldPos(*bones[static_cast<std::size_t>(firstChildren[i])]) -
                          nodeWorldPos(*bones[i]);
        }
        if (expectedDir.magnitudeSquared() < 1.0e-8f)
        {
            expectedDir = {0.0f, 1.0f, 0.0f};
        }
        expectedDir = Vec3::normalise(expectedDir);

        bool found = false;
        for (const auto& collider : initialColliders)
        {
            if (collider.type != std::to_underlying(fire_engine::ClothColliderType::Capsule))
            {
                continue;
            }
            found = (colliderCenter(collider) - nodeWorldPos(*bones[i])).magnitude() < 0.01f &&
                    std::abs(Vec3::dotProduct(colliderAxis(collider), expectedDir)) > 0.98f;
            if (found)
            {
                break;
            }
        }
        INFO("missing aligned capsule for " << bones[i]->name());
        CHECK(found);
    }

    std::vector<Vec3> parentAnchors(rag.boneCount());
    for (std::size_t i = 0; i < rag.boneCount(); ++i)
    {
        if (parents[i] >= 0)
        {
            const auto p = static_cast<std::size_t>(parents[i]);
            const auto parentTransform = physics.bodyTransform(rag.body(p));
            REQUIRE(parentTransform.has_value());
            parentAnchors[i] = parentTransform->rotation().conjugate().rotate(
                bodyPos(physics, rag, i) - bodyPos(physics, rag, p));
        }
    }

    constexpr float dt = 1.0f / 60.0f;
    float maxJointStretch = 0.0f;
    std::size_t maxStretchIndex = 0;
    for (int i = 0; i < 900; ++i)
    {
        physics.step(dt);
        sg.applyPhysics(physics);
        for (std::size_t j = 0; j < rag.boneCount(); ++j)
        {
            if (parents[j] < 0)
            {
                continue;
            }
            const auto p = static_cast<std::size_t>(parents[j]);
            const auto parentTransform = physics.bodyTransform(rag.body(p));
            REQUIRE(parentTransform.has_value());
            const Vec3 parentAnchor =
                bodyPos(physics, rag, p) + parentTransform->rotation().rotate(parentAnchors[j]);
            const float stretch = (parentAnchor - bodyPos(physics, rag, j)).magnitude();
            if (stretch > maxJointStretch)
            {
                maxJointStretch = stretch;
                maxStretchIndex = j;
            }
        }
    }

    float vmax = 0.0f;
    std::size_t vmaxIndex = 0;
    float wmax = 0.0f;
    std::size_t wmaxIndex = 0;
    float finalJointStretch = 0.0f;
    std::size_t finalStretchIndex = 0;
    for (std::size_t i = 0; i < rag.boneCount(); ++i)
    {
        const Vec3 v = physics.body(rag.body(i))->linearVelocity();
        if (const float speed = v.magnitude(); speed > vmax)
        {
            vmax = speed;
            vmaxIndex = i;
        }
        const Vec3 w = physics.body(rag.body(i))->angularVelocity();
        if (const float speed = w.magnitude(); speed > wmax)
        {
            wmax = speed;
            wmaxIndex = i;
        }
        CHECK(std::isfinite(bodyPos(physics, rag, i).y()));
        if (parents[i] >= 0)
        {
            const auto p = static_cast<std::size_t>(parents[i]);
            const auto parentTransform = physics.bodyTransform(rag.body(p));
            REQUIRE(parentTransform.has_value());
            const Vec3 parentAnchor =
                bodyPos(physics, rag, p) + parentTransform->rotation().rotate(parentAnchors[i]);
            const float stretch = (parentAnchor - bodyPos(physics, rag, i)).magnitude();
            if (stretch > finalJointStretch)
            {
                finalJointStretch = stretch;
                finalStretchIndex = i;
            }
        }
    }
    INFO("max residual velocity: " << vmax << " on " << rag.node(vmaxIndex)->name());
    INFO("max residual angular velocity: " << wmax << " on " << rag.node(wmaxIndex)->name());
    INFO("max joint stretch: " << maxJointStretch << " on " << rag.node(maxStretchIndex)->name());
    INFO("final joint stretch: " << finalJointStretch << " on "
                                 << rag.node(finalStretchIndex)->name());
    CHECK(finalJointStretch < 0.15f);
    CHECK(maxJointStretch < 0.75f);
    CHECK(vmax < 0.05f);
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

namespace
{

// Drop an articulated ragdoll (built from `bones`) onto a floor and return the resting state:
// the lowest link Y ever reached, the final lowest/highest link Y, and the base speed.
struct ArticulatedRest
{
    float minEver{0.0f};
    float lowest{0.0f};
    float highest{0.0f};
    float baseSpeed{0.0f};
    bool finite{false};
};

ArticulatedRest dropArticulated(std::span<Node* const> bones, const RagdollParams& params,
                                float floorHalfXz, int steps)
{
    PhysicsWorld physics;
    fire_engine::PhysicsBodyDesc floor;
    floor.type = fire_engine::PhysicsBodyType::Static;
    floor.position = {0.0f, -0.5f, 0.0f}; // top face at y = 0
    floor.material = fire_engine::PhysicsMaterial{.restitution = 0.0f, .friction = 0.6f};
    const auto floorBody = physics.createBody(floor);
    static_cast<void>(physics.createCollider(
        floorBody,
        fire_engine::ColliderDesc{
            .shape = fire_engine::BoxShape{Vec3{floorHalfXz, 0.5f, floorHalfXz}},
            .material = fire_engine::PhysicsMaterial{.restitution = 0.0f, .friction = 0.6f}}));

    Ragdoll rag = Ragdoll::makeArticulated(physics, bones, params);
    REQUIRE(rag.articulated());
    const fire_engine::Articulation* art = physics.articulation(rag.articulation());
    REQUIRE(art != nullptr);

    ArticulatedRest r;
    r.minEver = 1e9f;
    for (int i = 0; i < steps; ++i)
    {
        physics.step(1.0f / 60.0f);
        for (std::size_t link = 0; link < art->linkCount(); ++link)
        {
            r.minEver = std::min(r.minEver, art->linkWorld(link).translation.y());
        }
    }
    r.lowest = 1e9f;
    r.highest = -1e9f;
    for (std::size_t link = 0; link < art->linkCount(); ++link)
    {
        const float y = art->linkWorld(link).translation.y();
        r.lowest = std::min(r.lowest, y);
        r.highest = std::max(r.highest, y);
    }
    r.baseSpeed = art->baseVelocity().linear.magnitude();
    r.finite = std::isfinite(r.lowest) && std::isfinite(r.highest);
    return r;
}

} // namespace

TEST_CASE("Ragdoll.ArticulatedTwoJointDemoSettlesOnFloor", "[Ragdoll][Demos]")
{
    // Phase F2: the reduced-coordinate path binds the generated two-joint ragdoll (a floating
    // pelvis + one spherical child) and settles it on the floor — joint error is zero by
    // construction, so it rests rather than limit-cycling.
    const std::filesystem::path gltfPath = "../assets/physics_demos/TwoJointRagdollDemo.gltf";
    const fastgltf::Asset asset = parseGltfAsset(gltfPath);
    const RagdollParams params = parseGeneratedRagdollParams(gltfPath);

    SceneGraph sg;
    std::vector<Node*> bones = buildRagdollDemoBones(sg, asset);
    REQUIRE(bones.size() == 2U);

    const ArticulatedRest r = dropArticulated(bones, params, 4.0f, 900);
    CHECK(r.finite);
    CHECK(r.minEver > -0.3f);  // never tunnelled through the floor
    CHECK(r.lowest > 0.0f);    // rests on top of it
    CHECK(r.baseSpeed < 0.3f); // came (near) to rest
}

TEST_CASE("Ragdoll.ArticulatedHumanoidSettlesOnFloor", "[Ragdoll][Demos]")
{
    // The Phase F2 gate: the full 17-bone humanoid (branching, multi-child) bound to a single
    // reduced-coordinate articulation — floating pelvis, spherical joints, cone-twist limits —
    // falls and settles on the floor without tunnelling or blowing up. The CesiumMan mechanism
    // at demo scale.
    const std::filesystem::path gltfPath = "../assets/physics_demos/RagdollDemo.gltf";
    const fastgltf::Asset asset = parseGltfAsset(gltfPath);
    const RagdollParams params = parseGeneratedRagdollParams(gltfPath);

    SceneGraph sg;
    std::vector<Node*> bones = buildRagdollDemoBones(sg, asset);
    REQUIRE(bones.size() == 17U);

    const ArticulatedRest r = dropArticulated(bones, params, 6.0f, 1200);
    CHECK(r.finite);
    CHECK(r.minEver > -0.3f);  // never tunnelled through the floor
    CHECK(r.lowest > 0.0f);    // the whole skeleton rests above the floor
    CHECK(r.baseSpeed < 0.4f); // came (near) to rest
}
