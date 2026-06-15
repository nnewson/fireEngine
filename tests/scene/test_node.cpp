#include <fire_engine/scene/node.hpp>

#include <gtest/gtest.h>

using fire_engine::Camera;
using fire_engine::InputState;
using fire_engine::Mat4;
using fire_engine::Node;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsColliderHandle;
using fire_engine::Vec3;

// ==========================================================================
// Construction
// ==========================================================================

TEST(NodeConstruction, DefaultNameIsEmpty)
{
    Node n;
    EXPECT_TRUE(n.name().empty());
}

TEST(NodeConstruction, NamedConstruction)
{
    Node n("TestNode");
    EXPECT_EQ(n.name(), "TestNode");
}

TEST(NodeConstruction, DefaultHasNoParent)
{
    Node n("Orphan");
    EXPECT_EQ(n.parent(), nullptr);
}

TEST(NodeConstruction, DefaultHasNoChildren)
{
    Node n("Leaf");
    EXPECT_TRUE(n.children().empty());
}

TEST(NodeConstruction, DefaultTransformIsIdentity)
{
    Node n;
    EXPECT_EQ(n.transform().local(), fire_engine::Mat4::identity());
    EXPECT_EQ(n.transform().world(), fire_engine::Mat4::identity());
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST(NodeAccessors, SetName)
{
    Node n;
    n.name("NewName");
    EXPECT_EQ(n.name(), "NewName");
}

TEST(NodeAccessors, TransformIsMutable)
{
    Node n;
    n.transform().position({1.0f, 2.0f, 3.0f});
    EXPECT_FLOAT_EQ(n.transform().position().x(), 1.0f);
}

TEST(NodeAccessors, ConstTransform)
{
    Node n("Test");
    n.transform().position({5.0f, 0.0f, 0.0f});
    const Node& cn = n;
    EXPECT_FLOAT_EQ(cn.transform().position().x(), 5.0f);
}

TEST(NodeAccessors, DefaultHasNoControllable)
{
    Node n("ControllableNode");
    EXPECT_FALSE(n.hasControllable());
    EXPECT_EQ(n.controllable(), nullptr);
}

TEST(NodeAccessors, EmplaceControllable)
{
    Node n("ControllableNode");
    auto& controllable = n.emplaceControllable();

    EXPECT_TRUE(n.hasControllable());
    EXPECT_EQ(n.controllable(), &controllable);
}

TEST(NodeAccessors, DefaultHasNoPhysicsHandles)
{
    Node n("DynamicNode");
    EXPECT_FALSE(n.hasPhysicsBodyHandle());
    EXPECT_FALSE(n.hasPhysicsColliderHandle());
    EXPECT_FALSE(n.physicsBodyHandle().valid());
    EXPECT_FALSE(n.physicsColliderHandle().valid());
}

TEST(NodeAccessors, PhysicsHandlesAreMutable)
{
    Node n("DynamicNode");
    n.physicsBodyHandle(PhysicsBodyHandle{12U});
    n.physicsColliderHandle(PhysicsColliderHandle{34U});

    EXPECT_TRUE(n.hasPhysicsBodyHandle());
    EXPECT_TRUE(n.hasPhysicsColliderHandle());
    EXPECT_EQ(n.physicsBodyHandle(), PhysicsBodyHandle{12U});
    EXPECT_EQ(n.physicsColliderHandle(), PhysicsColliderHandle{34U});
}

TEST(NodeUpdate, UpdatesTransform)
{
    Node n("Node");
    n.transform().position({10.0f, 20.0f, 30.0f});

    n.update(InputState{}, Mat4::identity());

    EXPECT_FLOAT_EQ((n.transform().world()[0, 3]), 10.0f);
    EXPECT_FLOAT_EQ((n.transform().world()[1, 3]), 20.0f);
    EXPECT_FLOAT_EQ((n.transform().world()[2, 3]), 30.0f);
}

TEST(NodeMotion, FirstUpdatePreviousEqualsCurrent)
{
    Node n("Node");
    n.transform().position({5.0f, 0.0f, 0.0f});

    n.update(InputState{}, Mat4::identity());

    // First frame: no prior world, so previous == current (zero motion vector).
    EXPECT_FLOAT_EQ((n.previousComposedWorld()[0, 3]), 5.0f);
    EXPECT_FLOAT_EQ((n.composedWorld()[0, 3]), 5.0f);
}

TEST(NodeMotion, PreviousComposedWorldLagsOneFrame)
{
    Node n("Node");

    n.transform().position({1.0f, 0.0f, 0.0f});
    n.update(InputState{}, Mat4::identity());

    n.transform().position({4.0f, 0.0f, 0.0f});
    n.update(InputState{}, Mat4::identity());

    // Current is this frame's position; previous is last frame's.
    EXPECT_FLOAT_EQ((n.composedWorld()[0, 3]), 4.0f);
    EXPECT_FLOAT_EQ((n.previousComposedWorld()[0, 3]), 1.0f);
}

TEST(NodeUpdate, ControllableMovesTransformFromControllerState)
{
    Node n("ControllableNode");
    n.emplaceControllable();

    InputState state;
    state.controllerState().deltaPosition({0.5f, 0.0f, 0.0f});
    n.update(state, Mat4::identity());

    EXPECT_FLOAT_EQ(n.transform().position().x(), 5.0f);
    EXPECT_FLOAT_EQ((n.transform().world()[0, 3]), 5.0f);
}

TEST(NodeUpdate, NonControllableIgnoresControllerState)
{
    Node n("StaticNode");
    n.transform().position({1.0f, 2.0f, 3.0f});

    InputState state;
    state.controllerState().deltaPosition({0.5f, 0.0f, 0.0f});
    n.update(state, Mat4::identity());

    EXPECT_FLOAT_EQ(n.transform().position().x(), 1.0f);
    EXPECT_FLOAT_EQ((n.transform().world()[0, 3]), 1.0f);
}

// ==========================================================================
// Parent-child relationships
// ==========================================================================

TEST(NodeHierarchy, AddChildSetsParent)
{
    Node parent("Parent");
    auto child = std::make_unique<Node>("Child");
    Node* childRaw = child.get();

    parent.addChild(std::move(child));

    EXPECT_EQ(childRaw->parent(), &parent);
}

TEST(NodeHierarchy, AddChildReturnsReference)
{
    Node parent("Parent");
    auto child = std::make_unique<Node>("Child");

    Node& ref = parent.addChild(std::move(child));
    EXPECT_EQ(ref.name(), "Child");
}

TEST(NodeHierarchy, ParentHasChild)
{
    Node parent("Parent");
    parent.addChild(std::make_unique<Node>("Child"));

    EXPECT_EQ(parent.children().size(), 1u);
    EXPECT_EQ(parent.children()[0]->name(), "Child");
}

TEST(NodeHierarchy, MultipleChildren)
{
    Node parent("Parent");
    parent.addChild(std::make_unique<Node>("A"));
    parent.addChild(std::make_unique<Node>("B"));
    parent.addChild(std::make_unique<Node>("C"));

    EXPECT_EQ(parent.children().size(), 3u);
    EXPECT_EQ(parent.children()[0]->name(), "A");
    EXPECT_EQ(parent.children()[1]->name(), "B");
    EXPECT_EQ(parent.children()[2]->name(), "C");
}

TEST(NodeHierarchy, ThreeLevelNesting)
{
    Node root("Root");
    auto& mid = root.addChild(std::make_unique<Node>("Mid"));
    auto& leaf = mid.addChild(std::make_unique<Node>("Leaf"));

    EXPECT_EQ(leaf.parent(), &mid);
    EXPECT_EQ(mid.parent(), &root);
    EXPECT_EQ(root.parent(), nullptr);
}

TEST(NodeHierarchy, ChildOwnership)
{
    Node parent("Parent");
    parent.addChild(std::make_unique<Node>("Child"));

    // Children are owned by the parent — verify they survive parent scope
    EXPECT_EQ(parent.children().size(), 1u);
    EXPECT_NE(parent.children()[0], nullptr);
}

// ==========================================================================
// Component access
// ==========================================================================

TEST(NodeComponent, DefaultComponentIsFirstVariantAlternative)
{
    Node n;
    // Default-constructed variant holds the first alternative (Empty)
    EXPECT_TRUE(std::holds_alternative<fire_engine::Empty>(n.component()));
}

TEST(NodeComponent, EmplaceCamera)
{
    Node n("Camera");
    n.component().emplace<Camera>();
    EXPECT_TRUE(std::holds_alternative<Camera>(n.component()));
}

TEST(NodeComponent, ConstComponentAccess)
{
    Node n("Camera");
    n.component().emplace<Camera>();
    const Node& cn = n;
    EXPECT_TRUE(std::holds_alternative<Camera>(cn.component()));
}

// ==========================================================================
// Move Semantics
// ==========================================================================

TEST(NodeMove, MoveConstructTransfersState)
{
    Node a("Original");
    a.transform().position({1.0f, 2.0f, 3.0f});

    Node b{std::move(a)};
    EXPECT_EQ(b.name(), "Original");
    EXPECT_FLOAT_EQ(b.transform().position().x(), 1.0f);
}

TEST(NodeMove, MoveConstructTransfersChildren)
{
    Node a("Parent");
    a.addChild(std::make_unique<Node>("Child"));

    Node b{std::move(a)};
    EXPECT_EQ(b.children().size(), 1u);
    EXPECT_EQ(b.children()[0]->name(), "Child");
}
