#include <fire_engine/scene/scene_graph_format.hpp>

#include <format>

#include <gtest/gtest.h>

using fire_engine::Node;
using fire_engine::SceneGraph;
using fire_engine::Transform;

// ==========================================================================
// Transform Formatter
// ==========================================================================

TEST(TransformFormatter, DefaultTransform)
{
    Transform t;
    auto result = std::format("{}", t);
    EXPECT_EQ(result, "pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) scale(1.00, 1.00, 1.00)");
}

TEST(TransformFormatter, CustomValues)
{
    Transform t;
    t.position({1.0f, 2.0f, 3.0f});
    t.rotation(fire_engine::Quaternion{0.5f, 1.0f, 1.5f, 2.0f});
    t.scale({2.0f, 2.0f, 2.0f});
    auto result = std::format("{}", t);
    EXPECT_EQ(result, "pos(1.00, 2.00, 3.00) rot(0.50, 1.00, 1.50, 2.00) scale(2.00, 2.00, 2.00)");
}

// ==========================================================================
// Node Formatter
// ==========================================================================

TEST(NodeFormatter, SingleNode)
{
    Node node("TestNode");
    auto result = std::format("{}", node);
    EXPECT_EQ(result, "TestNode [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) "
                      "scale(1.00, 1.00, 1.00)");
}

TEST(NodeFormatter, NodeWithChildren)
{
    Node parent("Parent");
    auto child = std::make_unique<Node>("Child");
    parent.addChild(std::move(child));

    auto result = std::format("{}", parent);
    std::string expected = "Parent [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) "
                           "scale(1.00, 1.00, 1.00)\n"
                           "  Child [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) "
                           "scale(1.00, 1.00, 1.00)";
    EXPECT_EQ(result, expected);
}

TEST(NodeFormatter, NestedChildren)
{
    Node root("Root");
    auto child = std::make_unique<Node>("Child");
    auto grandchild = std::make_unique<Node>("Grandchild");
    child->addChild(std::move(grandchild));
    root.addChild(std::move(child));

    auto result = std::format("{}", root);
    std::string expected =
        "Root [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) scale(1.00, 1.00, 1.00)\n"
        "  Child [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) scale(1.00, 1.00, "
        "1.00)\n"
        "    Grandchild [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) "
        "scale(1.00, 1.00, 1.00)";
    EXPECT_EQ(result, expected);
}

// ==========================================================================
// SceneGraph Formatter
// ==========================================================================

TEST(SceneGraphFormatter, EmptyGraph)
{
    SceneGraph scene;
    auto result = std::format("{}", scene);
    EXPECT_EQ(result, "SceneGraph:");
}

TEST(SceneGraphFormatter, SingleRootNode)
{
    SceneGraph scene;
    scene.addNode(std::make_unique<Node>("Root"));

    auto result = std::format("{}", scene);
    std::string expected =
        "SceneGraph:\n"
        "  Root [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) scale(1.00, 1.00, 1.00)";
    EXPECT_EQ(result, expected);
}

TEST(SceneGraphFormatter, MultipleRootsWithChildren)
{
    SceneGraph scene;

    auto root1 = std::make_unique<Node>("Root1");
    root1->addChild(std::make_unique<Node>("Child1"));
    scene.addNode(std::move(root1));

    scene.addNode(std::make_unique<Node>("Root2"));

    auto result = std::format("{}", scene);
    std::string expected =
        "SceneGraph:\n"
        "  Root1 [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) scale(1.00, 1.00, "
        "1.00)\n"
        "    Child1 [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) scale(1.00, 1.00, "
        "1.00)\n"
        "  Root2 [Empty] pos(0.00, 0.00, 0.00) rot(0.00, 0.00, 0.00, 1.00) scale(1.00, 1.00, 1.00)";
    EXPECT_EQ(result, expected);
}
