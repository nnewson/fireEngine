#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/shadow_caster_deformation.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// SH-04: classifying a shadow caster.
//
// The resolver's response to a Deformable classification is pinned in test_shadow_lod_resolver.cpp.
// These cases pin the OTHER half — that the classification itself is reached for the carriers that
// actually deform a mesh after the simplifier measured it. Testing only the resolver would prove
// that a Deformable request falls back while leaving the question nobody asked: does anything ever
// classify a skinned mesh as Deformable in the first place?
// ---------------------------------------------------------------------------

TEST_CASE("ShadowCasterDeformation.RigidGeometryOnARigidInstanceSelects", "[ShadowLodResolver]")
{
    const Geometry geo;
    CHECK(shadowCasterDeformation(geo, false) == ShadowCasterDeformation::Rigid);
}

TEST_CASE("ShadowCasterDeformation.AnInstanceThatDeformsMakesItsGeometryDeformable",
          "[ShadowLodResolver]")
{
    // Skin and morph WEIGHTS live on the instance, not the geometry: two nodes can share one mesh
    // with only one of them skinned, so the instance's answer has to be able to force the
    // classification on its own.
    const Geometry geo;
    CHECK(shadowCasterDeformation(geo, true) == ShadowCasterDeformation::Deformable);
}

TEST_CASE("ShadowCasterDeformation.MorphTargetsCountEvenWithNoWeightsSet", "[ShadowLodResolver]")
{
    // The case the naive predicate misses. `Object::deformable()` asks whether morph WEIGHTS are
    // set; a morph-capable mesh sitting at its base pose has none, and would classify Rigid — until
    // an animation drove the weights, at which point the caster would silently swap error models
    // mid-animation. Capability is the stable property, so capability is what is classified.
    SECTION("positions")
    {
        Geometry geo;
        geo.morphPositions({{Vec3{0.0f, 1.0f, 0.0f}}});
        CHECK(shadowCasterDeformation(geo, false) == ShadowCasterDeformation::Deformable);
    }
    SECTION("normals")
    {
        Geometry geo;
        geo.morphNormals({{Vec3{0.0f, 1.0f, 0.0f}}});
        CHECK(shadowCasterDeformation(geo, false) == ShadowCasterDeformation::Deformable);
    }
    SECTION("tangents")
    {
        Geometry geo;
        geo.morphTangents({{Vec3{0.0f, 1.0f, 0.0f}}});
        CHECK(shadowCasterDeformation(geo, false) == ShadowCasterDeformation::Deformable);
    }
}

TEST_CASE("ShadowCasterDeformation.StorageVertexGeometryIsDeformable", "[ShadowLodResolver]")
{
    // Cloth: a compute pass rewrites the vertex buffer every frame, so the measured mesh is not the
    // rasterised one. It is single-level today and would draw whole regardless — which is exactly
    // why classifying it explicitly matters. The safety must come from the classification, not from
    // a property of today's cloth assets that nothing enforces.
    Geometry geo;
    geo.storageVertices(true);
    CHECK(shadowCasterDeformation(geo, false) == ShadowCasterDeformation::Deformable);
}
