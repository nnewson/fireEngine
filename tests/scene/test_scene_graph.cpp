#include <fire_engine/scene/scene_graph.hpp>

#include <array>
#include <deque>
#include <memory>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/object.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/scene/camera.hpp>
#include <fire_engine/scene/mesh.hpp>

using fire_engine::Camera;
using fire_engine::CameraView;
using fire_engine::Colour3;
using fire_engine::CullStats;
using fire_engine::DrawCommand;
using fire_engine::FrameInfo;
using fire_engine::Frustum;
using fire_engine::Geometry;
using fire_engine::InputState;
using fire_engine::Mat4;
using fire_engine::Material;
using fire_engine::Mesh;
using fire_engine::Node;
using fire_engine::Object;
using fire_engine::pi;
using fire_engine::SceneGraph;
using fire_engine::Vec2;
using fire_engine::Vec3;
using fire_engine::Vertex;

namespace
{

// Owns the geometry/material backing the scene's objects so their addresses stay stable for the
// test (Object stores raw pointers into these). Mirrors tests/scene/test_scene_culler.cpp.
struct DrawAssets
{
    std::deque<Material> materials;
    std::deque<Geometry> geometries;
};

// A renderable (culler-trackable) cube node at `position`, built entirely on the CPU — no GPU /
// Resources. Only ever exercised through buildDrawCommands' *culled* path, which skips the
// GPU-bound Object::render emission.
std::unique_ptr<Node> makeCubeNode(DrawAssets& assets, Vec3 position)
{
    Material& material = assets.materials.emplace_back();
    Geometry& geometry = assets.geometries.emplace_back();
    geometry.material(&material);
    geometry.vertices({Vertex{Vec3{-0.5f, -0.5f, -0.5f}, Colour3{}, Vec3{}, Vec2{}},
                       Vertex{Vec3{0.5f, 0.5f, 0.5f}, Colour3{}, Vec3{}, Vec2{}}});

    Object object;
    object.addGeometry(geometry);

    auto node = std::make_unique<Node>("cube");
    node->transform().position(position);
    node->component() = Mesh(std::move(object));
    return node;
}

// Camera at the origin looking down -z, 90° fov — nodes at +z are behind it (culled).
Frustum forwardFrustum()
{
    return Frustum::fromViewProj(Mat4::perspective(0.5f * pi, 1.0f, 0.1f, 100.0f));
}

} // namespace

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("SceneGraphConstruction.DefaultHasNoNodes", "[SceneGraphConstruction]")
{
    SceneGraph sg;
    CHECK(sg.nodes().empty());
}

TEST_CASE("SceneGraphConstruction.DefaultRootTransformIsIdentity", "[SceneGraphConstruction]")
{
    SceneGraph sg;
    CHECK(sg.rootTransform() == Mat4::identity());
}

// ==========================================================================
// Adding Nodes
// ==========================================================================

TEST_CASE("SceneGraphNodes.AddNodeReturnsReference", "[SceneGraphNodes]")
{
    SceneGraph sg;
    auto node = std::make_unique<Node>("Test");
    Node& ref = sg.addNode(std::move(node));
    CHECK(ref.name() == "Test");
}

TEST_CASE("SceneGraphNodes.AddNodeIncreasesCount", "[SceneGraphNodes]")
{
    SceneGraph sg;
    sg.addNode(std::make_unique<Node>("A"));
    sg.addNode(std::make_unique<Node>("B"));
    CHECK(sg.nodes().size() == 2u);
}

TEST_CASE("SceneGraphNodes.NodesPreserveOrder", "[SceneGraphNodes]")
{
    SceneGraph sg;
    sg.addNode(std::make_unique<Node>("First"));
    sg.addNode(std::make_unique<Node>("Second"));
    sg.addNode(std::make_unique<Node>("Third"));

    REQUIRE(sg.nodes().size() == 3u);
    CHECK(sg.nodes()[0]->name() == "First");
    CHECK(sg.nodes()[1]->name() == "Second");
    CHECK(sg.nodes()[2]->name() == "Third");
}

// ==========================================================================
// Root Transform
// ==========================================================================

TEST_CASE("SceneGraphRootTransform.SetRootTransform", "[SceneGraphRootTransform]")
{
    SceneGraph sg;
    Mat4 t = Mat4::translate({10.0f, 20.0f, 30.0f});
    sg.rootTransform(t);
    CHECK(sg.rootTransform() == t);
}

// ==========================================================================
// Move Semantics
// ==========================================================================

TEST_CASE("SceneGraphMove.MoveConstructTransfersNodes", "[SceneGraphMove]")
{
    SceneGraph a;
    a.addNode(std::make_unique<Node>("N1"));
    a.addNode(std::make_unique<Node>("N2"));

    SceneGraph b{std::move(a)};
    REQUIRE(b.nodes().size() == 2u);
    CHECK(b.nodes()[0]->name() == "N1");
}

TEST_CASE("SceneGraphMove.MoveConstructTransfersRootTransform", "[SceneGraphMove]")
{
    SceneGraph a;
    Mat4 t = Mat4::scale({2.0f, 2.0f, 2.0f});
    a.rootTransform(t);

    SceneGraph b{std::move(a)};
    CHECK(b.rootTransform() == t);
}

// ==========================================================================
// buildDrawCommands — the RenderableScene seam (CR-09). Only the *culled* path is exercised
// headlessly: culled nodes skip the GPU-bound Object::render emission, so no Device is needed.
// The isolated culling logic itself is covered by tests/scene/test_scene_culler.cpp; these tests
// cover the integration glue buildDrawCommands adds — frustum→culler wiring, CullStats plumbing,
// the empty-span "render all" branch, and skipping culled nodes.
// ==========================================================================

TEST_CASE("SceneGraphDraw.FullyCulledMeshNodesEmitNoDrawsAndReportCullStats", "[SceneGraphDraw]")
{
    DrawAssets assets;
    SceneGraph sg;
    // Two cubes behind the forward-looking camera → outside the frustum → culled.
    sg.addNode(makeCubeNode(assets, {0.0f, 0.0f, 50.0f}));
    sg.addNode(makeCubeNode(assets, {0.0f, 0.0f, 60.0f}));
    sg.resolve(); // populate composedWorld the culler reads

    const FrameInfo frame{}; // never consumed — culled nodes emit nothing
    const std::array<Frustum, 1> frustums{forwardFrustum()};
    std::vector<DrawCommand> out;
    // The prepass authority the draw walk requires. These cases exercise the CULLED path, where no
    // object emits a draw, so an empty record is the correct input rather than a shortcut.
    fire_engine::ShadowCasterBoundsFrame casterBounds;
    const CullStats stats = sg.buildDrawCommands(frame, frustums, casterBounds, out);

    CHECK(out.empty()); // culled → no draws (and no GPU touched)
    CHECK(stats.tracked == 2u);
    CHECK(stats.culled == 2u);
}

TEST_CASE("SceneGraphDraw.EmptyFrustumSpanDisablesCulling", "[SceneGraphDraw]")
{
    // An empty frustum span means "render all": the culler is not invoked, so CullStats is zero
    // even though a node sits where a real frustum would have culled it. (Non-mesh nodes only, so
    // the render-all path emits nothing and stays headless.)
    SceneGraph sg;
    sg.addNode(std::make_unique<Node>("empty"));
    sg.resolve();

    const FrameInfo frame{};
    std::vector<DrawCommand> out;
    fire_engine::ShadowCasterBoundsFrame casterBounds;
    const CullStats stats = sg.buildDrawCommands(frame, {}, casterBounds, out);

    CHECK(out.empty());
    CHECK(stats.tracked == 0u);
    CHECK(stats.culled == 0u);
}

TEST_CASE("SceneGraphDraw.NonRenderableNodesAreNotTracked", "[SceneGraphDraw]")
{
    // Only mesh nodes are culler-tracked; an Empty node contributes nothing to the counts.
    DrawAssets assets;
    SceneGraph sg;
    sg.addNode(std::make_unique<Node>("empty"));
    sg.addNode(makeCubeNode(assets, {0.0f, 0.0f, 40.0f})); // behind camera → culled
    sg.resolve();

    const FrameInfo frame{};
    const std::array<Frustum, 1> frustums{forwardFrustum()};
    std::vector<DrawCommand> out;
    // The prepass authority the draw walk requires. These cases exercise the CULLED path, where no
    // object emits a draw, so an empty record is the correct input rather than a shortcut.
    fire_engine::ShadowCasterBoundsFrame casterBounds;
    const CullStats stats = sg.buildDrawCommands(frame, frustums, casterBounds, out);

    CHECK(out.empty());
    CHECK(stats.tracked == 1u); // only the cube
    CHECK(stats.culled == 1u);
}

// activeCamera — the RenderableScene seam the renderer views the scene from.

TEST_CASE("SceneGraph.ActiveCameraFallbackWhenUnset", "[SceneGraph]")
{
    // No camera authored: a fixed debug pose (not a scene node). Position {2,2,2} looking at
    // origin.
    SceneGraph sg;
    const CameraView cv = sg.activeCamera();
    CHECK(cv.position.x() == Catch::Approx(2.0f));
    CHECK(cv.position.y() == Catch::Approx(2.0f));
    CHECK(cv.position.z() == Catch::Approx(2.0f));
    CHECK(cv.target.x() == Catch::Approx(0.0f));
    CHECK(cv.target.y() == Catch::Approx(0.0f));
    CHECK(cv.target.z() == Catch::Approx(0.0f));
}

TEST_CASE("SceneGraph.ActiveCameraReflectsSetNode", "[SceneGraph]")
{
    // A set camera node drives the view; activeCamera() reads its live world pose (so a moved node
    // moves the view). Update first so the Camera computes its world position from the node
    // transform.
    SceneGraph sg;
    auto node = std::make_unique<Node>("Camera");
    auto& camera = node->component().emplace<Camera>();
    camera.localPosition({5.0f, 1.0f, -3.0f});
    Node& added = sg.addNode(std::move(node));
    sg.activeCamera(&added);
    sg.update(InputState{});

    const CameraView cv = sg.activeCamera();
    CHECK(cv.position.x() == Catch::Approx(5.0f));
    CHECK(cv.position.y() == Catch::Approx(1.0f));
    CHECK(cv.position.z() == Catch::Approx(-3.0f));
}
