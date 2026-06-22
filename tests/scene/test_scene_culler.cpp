#include <fire_engine/scene/scene_culler.hpp>

#include <array>
#include <deque>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/object.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/scene/mesh.hpp>
#include <fire_engine/scene/node.hpp>

using fire_engine::Colour3;
using fire_engine::Frustum;
using fire_engine::Geometry;
using fire_engine::Mat4;
using fire_engine::Material;
using fire_engine::Mesh;
using fire_engine::Node;
using fire_engine::Object;
using fire_engine::pi;
using fire_engine::SceneCuller;
using fire_engine::Vec2;
using fire_engine::Vec3;
using fire_engine::Vertex;

namespace
{

// Owns the geometry/material backing the scene's objects so their addresses stay stable
// for the lifetime of a test (Object stores raw pointers into these).
struct Assets
{
    std::deque<Material> materials;
    std::deque<Geometry> geometries;
};

// A camera at the origin looking down -z, 90° fov, near 0.1, far 100 — same convention as
// test_frustum. Frustum half-extent at depth d is d.
Frustum cameraFrustum()
{
    return Frustum::fromViewProj(Mat4::perspective(0.5f * pi, 1.0f, 0.1f, 100.0f));
}

std::unique_ptr<Node> makeCubeNode(Assets& assets, Vec3 position, bool deformable = false)
{
    Material& material = assets.materials.emplace_back();
    Geometry& geometry = assets.geometries.emplace_back();
    geometry.material(&material);
    geometry.vertices({Vertex{Vec3{-0.5f, -0.5f, -0.5f}, Colour3{}, Vec3{}, Vec2{}},
                       Vertex{Vec3{0.5f, 0.5f, 0.5f}, Colour3{}, Vec3{}, Vec2{}}});

    Object object;
    object.addGeometry(geometry);
    if (deformable)
    {
        object.morphWeights({0.0f}); // marks the object deformable → never tracked/culled
    }

    auto node = std::make_unique<Node>("cube");
    node->transform().position(position);
    node->component() = Mesh(std::move(object));
    node->resolve(Mat4::identity());
    return node;
}

} // namespace

TEST_CASE("SceneCuller.InViewNodeIsNotCulled", "[SceneCuller]")
{
    Assets assets;
    std::vector<std::unique_ptr<Node>> nodes;
    Node* inView = nodes.emplace_back(makeCubeNode(assets, {0.0f, 0.0f, -5.0f})).get();

    SceneCuller culler;
    culler.sync(nodes);
    const std::array<Frustum, 1> frustums{cameraFrustum()};
    const auto& culled = culler.cull(frustums);

    CHECK(culler.trackedCount() == 1);
    CHECK_FALSE(culled.contains(inView));
}

TEST_CASE("SceneCuller.BehindAndOffscreenNodesAreCulled", "[SceneCuller]")
{
    Assets assets;
    std::vector<std::unique_ptr<Node>> nodes;
    Node* behind = nodes.emplace_back(makeCubeNode(assets, {0.0f, 0.0f, 50.0f})).get();
    Node* offToSide = nodes.emplace_back(makeCubeNode(assets, {100.0f, 0.0f, -5.0f})).get();
    Node* inView = nodes.emplace_back(makeCubeNode(assets, {0.0f, 0.0f, -5.0f})).get();

    SceneCuller culler;
    culler.sync(nodes);
    const std::array<Frustum, 1> frustums{cameraFrustum()};
    const auto& culled = culler.cull(frustums);

    CHECK(culler.trackedCount() == 3);
    CHECK(culled.contains(behind));
    CHECK(culled.contains(offToSide));
    CHECK_FALSE(culled.contains(inView));
}

TEST_CASE("SceneCuller.NodeVisibleInAnyFrustumSurvives", "[SceneCuller]")
{
    Assets assets;
    std::vector<std::unique_ptr<Node>> nodes;
    // Behind the main camera, so the camera frustum alone would cull it.
    Node* behind = nodes.emplace_back(makeCubeNode(assets, {0.0f, 0.0f, 50.0f})).get();

    SceneCuller culler;
    culler.sync(nodes);

    // A second frustum (e.g. a shadow caster) at z=100 looking -z does see it.
    const Frustum second = Frustum::fromViewProj(
        Mat4::perspective(0.5f * pi, 1.0f, 0.1f, 100.0f) *
        Mat4::lookAt({0.0f, 0.0f, 100.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}));
    const std::array<Frustum, 2> frustums{cameraFrustum(), second};
    const auto& culled = culler.cull(frustums);

    CHECK_FALSE(culled.contains(behind));
}

TEST_CASE("SceneCuller.DeformableNodesAreNeverTracked", "[SceneCuller]")
{
    Assets assets;
    std::vector<std::unique_ptr<Node>> nodes;
    // Deformable and behind the camera — a tracked rigid node here would be culled.
    Node* skinned = nodes.emplace_back(makeCubeNode(assets, {0.0f, 0.0f, 50.0f}, true)).get();

    SceneCuller culler;
    culler.sync(nodes);
    const std::array<Frustum, 1> frustums{cameraFrustum()};
    const auto& culled = culler.cull(frustums);

    CHECK(culler.trackedCount() == 0);
    CHECK_FALSE(culled.contains(skinned));
}

TEST_CASE("SceneCuller.MovingANodeUpdatesItsCullState", "[SceneCuller]")
{
    Assets assets;
    std::vector<std::unique_ptr<Node>> nodes;
    Node* node = nodes.emplace_back(makeCubeNode(assets, {0.0f, 0.0f, -5.0f})).get();

    SceneCuller culler;
    const std::array<Frustum, 1> frustums{cameraFrustum()};

    culler.sync(nodes);
    CHECK_FALSE(culler.cull(frustums).contains(node));

    // Move it behind the camera and re-sync — moveProxy must refit the BVH bound.
    node->transform().position({0.0f, 0.0f, 50.0f});
    node->resolve(Mat4::identity());
    culler.sync(nodes);
    CHECK(culler.cull(frustums).contains(node));
}

TEST_CASE("SceneCuller.RemovingANodeDestroysItsProxy", "[SceneCuller]")
{
    Assets assets;
    std::vector<std::unique_ptr<Node>> nodes;
    nodes.emplace_back(makeCubeNode(assets, {0.0f, 0.0f, -5.0f}));
    nodes.emplace_back(makeCubeNode(assets, {2.0f, 0.0f, -5.0f}));

    SceneCuller culler;
    culler.sync(nodes);
    CHECK(culler.trackedCount() == 2);

    nodes.pop_back();
    culler.sync(nodes);
    CHECK(culler.trackedCount() == 1);
}
