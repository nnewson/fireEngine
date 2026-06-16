#include <fire_engine/scene/scene_graph.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Mat4;
using fire_engine::Node;
using fire_engine::SceneGraph;

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
