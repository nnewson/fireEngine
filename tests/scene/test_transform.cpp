#include <fire_engine/scene/transform.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/math/quaternion.hpp>

using fire_engine::Mat4;
using fire_engine::Quaternion;
using fire_engine::Transform;

// ==========================================================================
// Default Construction
// ==========================================================================

TEST_CASE("TransformConstruction.DefaultPosition", "[TransformConstruction]")
{
    Transform t;
    CHECK(t.position().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.position().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.position().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("TransformConstruction.DefaultRotation", "[TransformConstruction]")
{
    Transform t;
    CHECK(t.rotation().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.rotation().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.rotation().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.rotation().w() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("TransformConstruction.DefaultScale", "[TransformConstruction]")
{
    Transform t;
    CHECK(t.scale().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(t.scale().y() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(t.scale().z() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("TransformConstruction.DefaultLocalIsIdentity", "[TransformConstruction]")
{
    Transform t;
    CHECK(t.local() == Mat4::identity());
}

TEST_CASE("TransformConstruction.DefaultWorldIsIdentity", "[TransformConstruction]")
{
    Transform t;
    CHECK(t.world() == Mat4::identity());
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("TransformAccessors.SetPosition", "[TransformAccessors]")
{
    Transform t;
    t.position({1.0f, 2.0f, 3.0f});
    CHECK(t.position().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(t.position().y() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(t.position().z() == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("TransformAccessors.SetRotation", "[TransformAccessors]")
{
    Transform t;
    t.rotation(Quaternion{0.1f, 0.2f, 0.3f, 0.4f});
    CHECK(t.rotation().x() == Catch::Approx(0.1f).margin(1e-5f));
    CHECK(t.rotation().y() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(t.rotation().z() == Catch::Approx(0.3f).margin(1e-5f));
    CHECK(t.rotation().w() == Catch::Approx(0.4f).margin(1e-5f));
}

TEST_CASE("TransformAccessors.SetScale", "[TransformAccessors]")
{
    Transform t;
    t.scale({2.0f, 3.0f, 4.0f});
    CHECK(t.scale().x() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(t.scale().y() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(t.scale().z() == Catch::Approx(4.0f).margin(1e-5f));
}

// ==========================================================================
// Update with identity parent
// ==========================================================================

TEST_CASE("TransformUpdate.DefaultWithIdentityParentProducesIdentity", "[TransformUpdate]")
{
    Transform t;
    t.update(Mat4::identity());
    CHECK(t.local() == Mat4::identity());
    CHECK(t.world() == Mat4::identity());
}

TEST_CASE("TransformUpdate.TranslationAffectsLocalMatrix", "[TransformUpdate]")
{
    Transform t;
    t.position({5.0f, 10.0f, 15.0f});
    t.update(Mat4::identity());

    // Column-major: translation is in column 3
    CHECK((t.local()[0, 3]) == Catch::Approx(5.0f).margin(1e-5f));
    CHECK((t.local()[1, 3]) == Catch::Approx(10.0f).margin(1e-5f));
    CHECK((t.local()[2, 3]) == Catch::Approx(15.0f).margin(1e-5f));
}

TEST_CASE("TransformUpdate.ScaleAffectsLocalMatrix", "[TransformUpdate]")
{
    Transform t;
    t.scale({2.0f, 3.0f, 4.0f});
    t.update(Mat4::identity());

    CHECK((t.local()[0, 0]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((t.local()[1, 1]) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK((t.local()[2, 2]) == Catch::Approx(4.0f).margin(1e-5f));
}

TEST_CASE("TransformUpdate.WorldEqualsLocalWithIdentityParent", "[TransformUpdate]")
{
    Transform t;
    t.position({1.0f, 2.0f, 3.0f});
    t.scale({2.0f, 2.0f, 2.0f});
    t.update(Mat4::identity());

    CHECK(t.world() == t.local());
}

// ==========================================================================
// Update with non-identity parent
// ==========================================================================

TEST_CASE("TransformUpdate.ParentTranslationAddsToChild", "[TransformUpdate]")
{
    Mat4 parent = Mat4::translate({10.0f, 20.0f, 30.0f});

    Transform t;
    t.position({1.0f, 2.0f, 3.0f});
    t.update(parent);

    // World translation should be parent + child
    CHECK((t.world()[0, 3]) == Catch::Approx(11.0f).margin(1e-5f));
    CHECK((t.world()[1, 3]) == Catch::Approx(22.0f).margin(1e-5f));
    CHECK((t.world()[2, 3]) == Catch::Approx(33.0f).margin(1e-5f));
}

TEST_CASE("TransformUpdate.ParentScaleScalesChildTranslation", "[TransformUpdate]")
{
    Mat4 parent = Mat4::scale({2.0f, 2.0f, 2.0f});

    Transform t;
    t.position({5.0f, 5.0f, 5.0f});
    t.update(parent);

    // Child position should be doubled by parent scale
    CHECK((t.world()[0, 3]) == Catch::Approx(10.0f).margin(1e-5f));
    CHECK((t.world()[1, 3]) == Catch::Approx(10.0f).margin(1e-5f));
    CHECK((t.world()[2, 3]) == Catch::Approx(10.0f).margin(1e-5f));
}

TEST_CASE("TransformUpdate.WorldIsParentTimesLocal", "[TransformUpdate]")
{
    Mat4 parent = Mat4::translate({1.0f, 0.0f, 0.0f}) * Mat4::scale({2.0f, 2.0f, 2.0f});

    Transform t;
    t.position({3.0f, 4.0f, 5.0f});
    t.update(parent);

    Mat4 expected = parent * t.local();
    CHECK(t.world() == expected);
}

// ==========================================================================
// Hierarchy simulation
// ==========================================================================

TEST_CASE("TransformHierarchy.ThreeLevelHierarchy", "[TransformHierarchy]")
{
    // Simulate grandparent → parent → child
    Transform grandparent;
    grandparent.position({10.0f, 0.0f, 0.0f});
    grandparent.update(Mat4::identity());

    Transform parent;
    parent.position({5.0f, 0.0f, 0.0f});
    parent.update(grandparent.world());

    Transform child;
    child.position({1.0f, 0.0f, 0.0f});
    child.update(parent.world());

    // Child's world X should be 10 + 5 + 1 = 16
    CHECK((child.world()[0, 3]) == Catch::Approx(16.0f).margin(1e-5f));
}

// ==========================================================================
// Copy and Move Semantics
// ==========================================================================

TEST_CASE("TransformCopy.CopyConstructCreatesIndependentCopy", "[TransformCopy]")
{
    Transform a;
    a.position({1.0f, 2.0f, 3.0f});
    a.update(Mat4::identity());

    Transform b{a};
    CHECK(b.position().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(b.local() == a.local());

    b.position({9.0f, 9.0f, 9.0f});
    CHECK(a.position().x() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("TransformMove.MoveConstructTransfersState", "[TransformMove]")
{
    Transform a;
    a.position({4.0f, 5.0f, 6.0f});
    a.update(Mat4::identity());
    Mat4 expectedLocal = a.local();

    Transform b{std::move(a)};
    CHECK(b.position().x() == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(b.local() == expectedLocal);
}
