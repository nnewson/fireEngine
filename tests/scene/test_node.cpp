#include <fire_engine/scene/node.hpp>

#include <fire_engine/animation/animation.hpp>
#include <fire_engine/math/constants.hpp>

#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("NodeConstruction.DefaultNameIsEmpty", "[NodeConstruction]")
{
    Node n;
    CHECK(n.name().empty());
}

TEST_CASE("NodeConstruction.NamedConstruction", "[NodeConstruction]")
{
    Node n("TestNode");
    CHECK(n.name() == "TestNode");
}

TEST_CASE("NodeConstruction.DefaultHasNoParent", "[NodeConstruction]")
{
    Node n("Orphan");
    CHECK(n.parent() == nullptr);
}

TEST_CASE("NodeConstruction.DefaultHasNoChildren", "[NodeConstruction]")
{
    Node n("Leaf");
    CHECK(n.children().empty());
}

TEST_CASE("NodeConstruction.DefaultTransformIsIdentity", "[NodeConstruction]")
{
    Node n;
    CHECK(n.transform().local() == fire_engine::Mat4::identity());
    CHECK(n.transform().world() == fire_engine::Mat4::identity());
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("NodeAccessors.SetName", "[NodeAccessors]")
{
    Node n;
    n.name("NewName");
    CHECK(n.name() == "NewName");
}

TEST_CASE("NodeAccessors.TransformIsMutable", "[NodeAccessors]")
{
    Node n;
    n.transform().position({1.0f, 2.0f, 3.0f});
    CHECK(n.transform().position().x() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("NodeAccessors.ConstTransform", "[NodeAccessors]")
{
    Node n("Test");
    n.transform().position({5.0f, 0.0f, 0.0f});
    const Node& cn = n;
    CHECK(cn.transform().position().x() == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("NodeAccessors.DefaultHasNoControllable", "[NodeAccessors]")
{
    Node n("ControllableNode");
    CHECK_FALSE(n.hasControllable());
    CHECK(n.controllable() == nullptr);
}

TEST_CASE("NodeAccessors.EmplaceControllable", "[NodeAccessors]")
{
    Node n("ControllableNode");
    auto& controllable = n.emplaceControllable();

    CHECK(n.hasControllable());
    CHECK(n.controllable() == &controllable);
}

TEST_CASE("NodeAccessors.DefaultHasNoPhysicsHandles", "[NodeAccessors]")
{
    Node n("DynamicNode");
    CHECK_FALSE(n.hasPhysicsBodyHandle());
    CHECK_FALSE(n.hasPhysicsColliderHandle());
    CHECK_FALSE(n.physicsBodyHandle().valid());
    CHECK_FALSE(n.physicsColliderHandle().valid());
}

TEST_CASE("NodeAccessors.PhysicsHandlesAreMutable", "[NodeAccessors]")
{
    Node n("DynamicNode");
    n.physicsBodyHandle(PhysicsBodyHandle{12U});
    n.physicsColliderHandle(PhysicsColliderHandle{34U});

    CHECK(n.hasPhysicsBodyHandle());
    CHECK(n.hasPhysicsColliderHandle());
    CHECK(n.physicsBodyHandle() == PhysicsBodyHandle{12U});
    CHECK(n.physicsColliderHandle() == PhysicsColliderHandle{34U});
}

TEST_CASE("NodeUpdate.UpdatesTransform", "[NodeUpdate]")
{
    Node n("Node");
    n.transform().position({10.0f, 20.0f, 30.0f});

    n.update(InputState{}, Mat4::identity());

    CHECK((n.transform().world()[0, 3]) == Catch::Approx(10.0f).margin(1e-5f));
    CHECK((n.transform().world()[1, 3]) == Catch::Approx(20.0f).margin(1e-5f));
    CHECK((n.transform().world()[2, 3]) == Catch::Approx(30.0f).margin(1e-5f));
}

TEST_CASE("NodeMotion.FirstUpdatePreviousEqualsCurrent", "[NodeMotion]")
{
    Node n("Node");
    n.transform().position({5.0f, 0.0f, 0.0f});

    n.update(InputState{}, Mat4::identity());

    // First frame: no prior world, so previous == current (zero motion vector).
    CHECK((n.previousComposedWorld()[0, 3]) == Catch::Approx(5.0f).margin(1e-5f));
    CHECK((n.composedWorld()[0, 3]) == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("NodeMotion.PreviousComposedWorldLagsOneFrame", "[NodeMotion]")
{
    Node n("Node");

    n.transform().position({1.0f, 0.0f, 0.0f});
    n.update(InputState{}, Mat4::identity());

    n.transform().position({4.0f, 0.0f, 0.0f});
    n.update(InputState{}, Mat4::identity());

    // Current is this frame's position; previous is last frame's.
    CHECK((n.composedWorld()[0, 3]) == Catch::Approx(4.0f).margin(1e-5f));
    CHECK((n.previousComposedWorld()[0, 3]) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("NodeMotion.WorldRevisionTracksComposedWorldChanges", "[NodeMotion]")
{
    Node n("Node");
    CHECK(n.worldRevision() == 0U);

    n.transform().position({1.0f, 0.0f, 0.0f});
    n.resolve(Mat4::identity());
    const std::uint64_t initialRevision = n.worldRevision();
    CHECK(initialRevision == 1U);

    n.resolve(Mat4::identity());
    CHECK(n.worldRevision() == initialRevision);

    n.transform().position({4.0f, 0.0f, 0.0f});
    n.resolve(Mat4::identity());
    CHECK(n.worldRevision() == initialRevision + 1U);
}

TEST_CASE("NodeUpdate.ControllableMovesTransformFromControllerState", "[NodeUpdate]")
{
    Node n("ControllableNode");
    n.emplaceControllable();

    InputState state;
    state.controllerState().deltaPosition({0.5f, 0.0f, 0.0f});
    n.update(state, Mat4::identity());

    CHECK(n.transform().position().x() == Catch::Approx(5.0f).margin(1e-5f));
    CHECK((n.transform().world()[0, 3]) == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("NodeUpdate.NonControllableIgnoresControllerState", "[NodeUpdate]")
{
    Node n("StaticNode");
    n.transform().position({1.0f, 2.0f, 3.0f});

    InputState state;
    state.controllerState().deltaPosition({0.5f, 0.0f, 0.0f});
    n.update(state, Mat4::identity());

    CHECK(n.transform().position().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((n.transform().world()[0, 3]) == Catch::Approx(1.0f).margin(1e-5f));
}

// ==========================================================================
// Parent-child relationships
// ==========================================================================

TEST_CASE("NodeHierarchy.AddChildSetsParent", "[NodeHierarchy]")
{
    Node parent("Parent");
    auto child = std::make_unique<Node>("Child");
    Node* childRaw = child.get();

    parent.addChild(std::move(child));

    CHECK(childRaw->parent() == &parent);
}

TEST_CASE("NodeHierarchy.AddChildReturnsReference", "[NodeHierarchy]")
{
    Node parent("Parent");
    auto child = std::make_unique<Node>("Child");

    Node& ref = parent.addChild(std::move(child));
    CHECK(ref.name() == "Child");
}

TEST_CASE("NodeHierarchy.ParentHasChild", "[NodeHierarchy]")
{
    Node parent("Parent");
    parent.addChild(std::make_unique<Node>("Child"));

    REQUIRE(parent.children().size() == 1u);
    CHECK(parent.children()[0]->name() == "Child");
}

TEST_CASE("NodeHierarchy.MultipleChildren", "[NodeHierarchy]")
{
    Node parent("Parent");
    parent.addChild(std::make_unique<Node>("A"));
    parent.addChild(std::make_unique<Node>("B"));
    parent.addChild(std::make_unique<Node>("C"));

    REQUIRE(parent.children().size() == 3u);
    CHECK(parent.children()[0]->name() == "A");
    CHECK(parent.children()[1]->name() == "B");
    CHECK(parent.children()[2]->name() == "C");
}

TEST_CASE("NodeHierarchy.ThreeLevelNesting", "[NodeHierarchy]")
{
    Node root("Root");
    auto& mid = root.addChild(std::make_unique<Node>("Mid"));
    auto& leaf = mid.addChild(std::make_unique<Node>("Leaf"));

    CHECK(leaf.parent() == &mid);
    CHECK(mid.parent() == &root);
    CHECK(root.parent() == nullptr);
}

TEST_CASE("NodeHierarchy.ChildOwnership", "[NodeHierarchy]")
{
    Node parent("Parent");
    parent.addChild(std::make_unique<Node>("Child"));

    // Children are owned by the parent — verify they survive parent scope
    REQUIRE(parent.children().size() == 1u);
    CHECK(parent.children()[0] != nullptr);
}

// ==========================================================================
// Component access
// ==========================================================================

TEST_CASE("NodeComponent.DefaultComponentIsFirstVariantAlternative", "[NodeComponent]")
{
    Node n;
    // Default-constructed variant holds the first alternative (Empty)
    CHECK(std::holds_alternative<fire_engine::Empty>(n.component()));
}

TEST_CASE("NodeComponent.EmplaceCamera", "[NodeComponent]")
{
    Node n("Camera");
    n.component().emplace<Camera>();
    CHECK(std::holds_alternative<Camera>(n.component()));
}

TEST_CASE("NodeComponent.ConstComponentAccess", "[NodeComponent]")
{
    Node n("Camera");
    n.component().emplace<Camera>();
    const Node& cn = n;
    CHECK(std::holds_alternative<Camera>(cn.component()));
}

// ==========================================================================
// Move Semantics
// ==========================================================================

TEST_CASE("NodeMove.MoveConstructTransfersState", "[NodeMove]")
{
    Node a("Original");
    a.transform().position({1.0f, 2.0f, 3.0f});

    Node b{std::move(a)};
    CHECK(b.name() == "Original");
    CHECK(b.transform().position().x() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("NodeMove.MoveConstructTransfersChildren", "[NodeMove]")
{
    Node a("Parent");
    a.addChild(std::make_unique<Node>("Child"));

    Node b{std::move(a)};
    REQUIRE(b.children().size() == 1u);
    CHECK(b.children()[0]->name() == "Child");
}

// ---------------------------------------------------------------------------
// One transform source for the draw walk and the transform walks (SH-06).
//
// `update` / `resolve` treat a world-override — a ragdoll-driven body, whose pose the physics
// solver owns — as authoritative, bypassing the parent chain and the local transform. The draw walk
// used to recompute `parentWorld * local` and ignore it, so an overridden node drew at a pose that
// was not the one its shadow-caster bounds were measured at.
// ---------------------------------------------------------------------------

TEST_CASE("NodeDrawWorld.MatchesComposedWorldForAnOrdinaryNode", "[Node]")
{
    Node node("Crate");
    node.transform().position({3.0f, 0.0f, 0.0f});

    const Mat4 parent = Mat4::translate({0.0f, 5.0f, 0.0f});
    node.resolve(parent);

    const Mat4 drawWorld = node.drawWorld(parent);
    // A node with no component matrix draws exactly where its composed world puts it, which is what
    // makes the prepass's bounds and the draw's geometry the same thing.
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            CHECK(drawWorld[row, col] ==
                  Catch::Approx(node.composedWorld()[row, col]).margin(1e-5f));
        }
    }
}

TEST_CASE("NodeDrawWorld.HonoursAWorldOverrideLikeTheTransformWalksDo", "[Node]")
{
    Node node("RagdollLimb");
    // A local transform that would place it somewhere else entirely, to make the two answers
    // unmistakably different if the override were ignored.
    node.transform().position({100.0f, 0.0f, 0.0f});
    const Mat4 physicsPose = Mat4::translate({-7.0f, 2.0f, 1.0f});
    node.worldOverride(physicsPose);

    const Mat4 parent = Mat4::translate({0.0f, 50.0f, 0.0f});
    node.resolve(parent);

    // Both walks agree, and both agree with the physics pose rather than the parent chain.
    CHECK(node.composedWorld()[0, 3] == Catch::Approx(-7.0f).margin(1e-5f));
    CHECK(node.drawWorld(parent)[0, 3] == Catch::Approx(-7.0f).margin(1e-5f));
    CHECK(node.drawWorld(parent)[1, 3] == Catch::Approx(2.0f).margin(1e-5f));

    // And clearing it returns both to the parent chain.
    node.clearWorldOverride();
    node.resolve(parent);
    CHECK(node.drawWorld(parent)[0, 3] == Catch::Approx(100.0f).margin(1e-5f));
    CHECK(node.composedWorld()[0, 3] == Catch::Approx(100.0f).margin(1e-5f));
}

TEST_CASE("NodeDrawWorld.AnOverriddenAnimatorDoesNotMoveItsChildren", "[Node]")
{
    // The case `drawWorld` alone does not cover. `update` / `resolve` return early on a
    // world-override, so they skip the component matrix as well as the parent chain: an overridden
    // Animator's children inherit the override ITSELF. The draw walk applies the component matrix
    // on the way down, so without the shared rule it would hand children `override * animation`
    // while their cached shadow bounds were measured at `override`.
    //
    // SCOPE, stated plainly: this pins the ALGEBRA of the rule by calling the helpers directly. It
    // does NOT prove that `Node::render` calls them, and it would still pass if the visitor stopped
    // using `childWorld`. Proving the call site needs a component that records the world it was
    // rendered with, and the only one that does is `Mesh`, whose draw path needs a GPU-backed
    // `Object` — so that half is covered by the render smoke rather than here. The call site is
    // instead constrained structurally: in `Node::render` the component's world is an unnamed
    // argument to `childWorld`, so it cannot reach the children by any other route.
    Node rig("RagdollRig");
    auto& animator = rig.component().emplace<fire_engine::Animator>();
    fire_engine::Animation spin;
    // 90 degrees about Y over 2 s, so the animation matrix is unmistakably not identity.
    const float s45 = std::sin(fire_engine::pi / 4.0f);
    const float c45 = std::cos(fire_engine::pi / 4.0f);
    spin.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 0.0f, s45, 0.0f, c45},
    });
    animator.addAnimation(&spin);

    fire_engine::InputState state;
    state.time(0.0);
    animator.update(state, rig.transform());
    state.time(1.0);
    animator.update(state, rig.transform());
    // The animator really is rotating — otherwise this test could not tell the two rules apart.
    const Mat4 animated = animator.render(Mat4::identity());
    REQUIRE(animated[0, 0] == Catch::Approx(c45).margin(1e-4f));

    const Mat4 physicsPose = Mat4::translate({4.0f, 0.0f, 0.0f});
    rig.worldOverride(physicsPose);
    auto& child = rig.addChild(std::make_unique<Node>("Limb_Mesh"));
    rig.resolve(Mat4::identity());

    // What the transform walk gave the child, and what the draw walk must give it: the override,
    // with no animation folded in.
    const Mat4 drawWorld = rig.drawWorld(Mat4::identity());
    const Mat4 componentWorld = animator.render(drawWorld);
    const Mat4 handedToChildren = rig.childWorld(drawWorld, componentWorld);

    CHECK(child.composedWorld()[0, 3] == Catch::Approx(4.0f).margin(1e-5f));
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            CHECK(handedToChildren[row, col] ==
                  Catch::Approx(child.composedWorld()[row, col]).margin(1e-5f));
        }
    }
    // And the animated world genuinely differs, so the check above is not vacuous.
    CHECK(componentWorld[0, 0] != Catch::Approx(handedToChildren[0, 0]).margin(1e-4f));

    // Without the override, the animation is exactly what children inherit.
    rig.clearWorldOverride();
    const Mat4 unOverridden = rig.drawWorld(Mat4::identity());
    CHECK(rig.childWorld(unOverridden, animator.render(unOverridden))[0, 0] ==
          Catch::Approx(c45).margin(1e-4f));
}
