#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/object.hpp>

using fire_engine::Geometry;
using fire_engine::Material;
using fire_engine::Object;

// ---------------------------------------------------------------------------
// Per-instance shadow eligibility.
//
// `extras.Shadow.Casts` is authored per NODE, but a Geometry is shared by every node that
// instances that mesh — the glTF loader builds one per mesh, not one per node. Holding the
// flag on the Geometry therefore cannot represent a scene that instances a mesh as both a
// caster and a receive-only surface: whichever node loaded last would silently rewrite the
// other. These tests pin the flag to the per-instance binding.
//
// Headless: adding a geometry only records pointers, so no GPU device is involved.
// ---------------------------------------------------------------------------

namespace
{
// A Geometry needs a material to be bound (Object caches it per binding), but nothing here
// reads it — no GPU resources are created.
struct SharedMesh
{
    Material material{};
    Geometry geometry{};

    SharedMesh()
    {
        geometry.material(&material);
    }
};
} // namespace

TEST_CASE("Object.ShadowEligibilityDefaultsToCasting", "[Object]")
{
    SharedMesh mesh;
    Object object;
    object.addGeometry(mesh.geometry);

    CHECK(object.castsShadow(0));
}

TEST_CASE("Object.TwoInstancesOfOneMeshHoldOppositeShadowFlags", "[Object]")
{
    // The case the previous design could not express: ONE mesh, two instances, opposite
    // authoring. Loading order must not matter, so both orders are checked.
    SharedMesh mesh;

    Object caster;
    caster.addGeometry(mesh.geometry, true);
    Object receiver;
    receiver.addGeometry(mesh.geometry, false);

    CHECK(caster.castsShadow(0));
    CHECK_FALSE(receiver.castsShadow(0));

    Object receiverFirst;
    receiverFirst.addGeometry(mesh.geometry, false);
    Object casterSecond;
    casterSecond.addGeometry(mesh.geometry, true);

    CHECK_FALSE(receiverFirst.castsShadow(0));
    CHECK(casterSecond.castsShadow(0));
    // And the first instance is untouched by the second's arrival — the defect this replaces
    // was exactly a later load reaching back and rewriting an earlier one.
    CHECK_FALSE(receiver.castsShadow(0));
    CHECK(caster.castsShadow(0));
}

TEST_CASE("Object.ShadowEligibilityIsPerBindingNotPerObject", "[Object]")
{
    // A multi-primitive mesh becomes several bindings on one Object; they are independent.
    SharedMesh first;
    SharedMesh second;
    Object object;
    object.addGeometry(first.geometry, false);
    object.addGeometry(second.geometry, true);

    CHECK_FALSE(object.castsShadow(0));
    CHECK(object.castsShadow(1));
}

TEST_CASE("Object.ShadowEligibilityOfAnAbsentBindingIsFalse", "[Object]")
{
    // Out-of-range answers "no shadow" rather than reading past the end: a caller asking
    // about a binding that doesn't exist must not be told it casts.
    Object object;
    CHECK_FALSE(object.castsShadow(0));
}
