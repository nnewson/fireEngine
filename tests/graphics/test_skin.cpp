#include <fire_engine/graphics/skin.hpp>
#include <fire_engine/input/input_state.hpp>
#include <fire_engine/scene/node.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::InputState;
using fire_engine::Mat4;
using fire_engine::Node;
using fire_engine::Skin;

// ==========================================================================
// Default Construction
// ==========================================================================

TEST_CASE("SkinConstruction.DefaultEmpty", "[SkinConstruction]")
{
    Skin skin;
    CHECK(skin.empty());
    CHECK(skin.jointCount() == 0u);
}

TEST_CASE("SkinConstruction.DefaultName", "[SkinConstruction]")
{
    Skin skin;
    CHECK(skin.name().empty());
}

// ==========================================================================
// Name
// ==========================================================================

TEST_CASE("SkinName.SetAndGet", "[SkinName]")
{
    Skin skin;
    skin.name("TestSkin");
    CHECK(skin.name() == "TestSkin");
}

// ==========================================================================
// Add Joints
// ==========================================================================

TEST_CASE("SkinAddJoint.SingleJoint", "[SkinAddJoint]")
{
    Skin skin;
    Node node("joint0");
    skin.addJoint(&node, Mat4::identity());
    CHECK(skin.jointCount() == 1u);
    CHECK_FALSE(skin.empty());
}

TEST_CASE("SkinAddJoint.MultipleJoints", "[SkinAddJoint]")
{
    Skin skin;
    Node n0("joint0");
    Node n1("joint1");
    Node n2("joint2");
    skin.addJoint(&n0, Mat4::identity());
    skin.addJoint(&n1, Mat4::identity());
    skin.addJoint(&n2, Mat4::identity());
    CHECK(skin.jointCount() == 3u);
}

// ==========================================================================
// Compute Joint Matrices
// ==========================================================================

TEST_CASE("SkinComputeJointMatrices.EmptySkin", "[SkinComputeJointMatrices]")
{
    Skin skin;
    auto matrices = skin.computeJointMatrices();
    CHECK(matrices.empty());
}

TEST_CASE("SkinComputeJointMatrices.IdentityInverseBind", "[SkinComputeJointMatrices]")
{
    Skin skin;
    Node node("joint0");
    InputState cs;
    node.update(cs, Mat4::identity());

    skin.addJoint(&node, Mat4::identity());
    auto matrices = skin.computeJointMatrices();

    REQUIRE(matrices.size() == 1u);
    // world (identity) * inverseBind (identity) = identity
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            float expected = (row == col) ? 1.0f : 0.0f;
            float actual = matrices[0][row, col];
            CHECK(actual == Catch::Approx(expected).margin(1e-5f));
        }
    }
}

TEST_CASE("SkinComputeJointMatrices.WithTranslation", "[SkinComputeJointMatrices]")
{
    Skin skin;
    Node node("joint0");
    node.transform().position({5.0f, 0.0f, 0.0f});

    InputState cs;
    node.update(cs, Mat4::identity());

    skin.addJoint(&node, Mat4::identity());
    auto matrices = skin.computeJointMatrices();

    REQUIRE(matrices.size() == 1u);
    float tx = matrices[0][0, 3];
    CHECK(tx == Catch::Approx(5.0f).margin(1e-5f));
}

// ==========================================================================
// Move Semantics
// ==========================================================================

TEST_CASE("SkinMove.MoveConstruction", "[SkinMove]")
{
    Skin skin;
    Node n0("joint0");
    skin.addJoint(&n0, Mat4::identity());
    skin.name("TestSkin");

    Skin moved(std::move(skin));
    CHECK(moved.jointCount() == 1u);
    CHECK(moved.name() == "TestSkin");
}

TEST_CASE("SkinMove.MoveAssignment", "[SkinMove]")
{
    Skin skin;
    Node n0("joint0");
    skin.addJoint(&n0, Mat4::identity());

    Skin other;
    other = std::move(skin);
    CHECK(other.jointCount() == 1u);
}
