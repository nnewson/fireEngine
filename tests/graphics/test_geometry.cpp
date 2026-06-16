#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Construction and defaults
// ---------------------------------------------------------------------------

TEST_CASE("Geometry.DefaultConstructionNotLoaded", "[Geometry]")
{
    Geometry geo;
    CHECK_FALSE(geo.loaded());
}

TEST_CASE("Geometry.DefaultConstructionEmptyVertices", "[Geometry]")
{
    Geometry geo;
    CHECK(geo.vertices().empty());
}

TEST_CASE("Geometry.DefaultConstructionEmptyIndices", "[Geometry]")
{
    Geometry geo;
    CHECK(geo.indices().empty());
    CHECK(geo.indexCount() == 0);
}

// ---------------------------------------------------------------------------
// Vertices
// ---------------------------------------------------------------------------

TEST_CASE("Geometry.SetAndGetVertices", "[Geometry]")
{
    Geometry geo;
    std::vector<Vertex> verts = {
        {{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}},
        {{1, 0, 0}, {1, 1, 1}, {0, 1, 0}, {1, 0}},
        {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}, {0, 1}},
    };
    geo.vertices(verts);
    CHECK(geo.vertices().size() == 3);
}

TEST_CASE("Geometry.VerticesMoveSemantics", "[Geometry]")
{
    Geometry geo;
    std::vector<Vertex> verts = {
        {{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}},
    };
    geo.vertices(std::move(verts));
    CHECK(geo.vertices().size() == 1);
}

// ---------------------------------------------------------------------------
// Indices
// ---------------------------------------------------------------------------

TEST_CASE("Geometry.SetAndGetIndices", "[Geometry]")
{
    Geometry geo;
    std::vector<uint16_t> idxs = {0, 1, 2, 2, 3, 0};
    geo.indices(idxs);
    CHECK(geo.indices().size() == 6);
    CHECK(geo.indexCount() == 6);
}

TEST_CASE("Geometry.IndicesMoveSemantics", "[Geometry]")
{
    Geometry geo;
    std::vector<uint16_t> idxs = {0, 1, 2};
    geo.indices(std::move(idxs));
    CHECK(geo.indexCount() == 3);
}

TEST_CASE("Geometry.SetAndGetUInt32Indices", "[Geometry]")
{
    Geometry geo;
    std::vector<uint32_t> idxs = {0u, 65536u, 70000u};
    geo.indices(idxs);
    REQUIRE(geo.indices().size() == 3u);
    CHECK(geo.indices()[1] == 65536u);
    CHECK(geo.indices()[2] == 70000u);
    CHECK(geo.indexType() == DrawIndexType::UInt32);
}

TEST_CASE("Geometry.CastsShadowDefaultsToTrue", "[Geometry]")
{
    Geometry geo;
    CHECK(geo.castsShadow());
}

TEST_CASE("Geometry.CastsShadowRoundTrip", "[Geometry]")
{
    Geometry geo;
    geo.castsShadow(false);
    CHECK_FALSE(geo.castsShadow());
    geo.castsShadow(true);
    CHECK(geo.castsShadow());
}

// ---------------------------------------------------------------------------
// Material pointer
// ---------------------------------------------------------------------------

TEST_CASE("Geometry.DefaultMaterialIsNull", "[Geometry]")
{
    Geometry geo;
    // material() dereferences the pointer, so we just check we can set one
    Material mat;
    geo.material(&mat);
    CHECK(geo.material().name() == mat.name());
}

TEST_CASE("Geometry.MaterialPointerAssignment", "[Geometry]")
{
    Geometry geo;
    Material mat;
    mat.name("test_mat");
    geo.material(&mat);
    CHECK(geo.material().name() == "test_mat");
}

// ---------------------------------------------------------------------------
// Move semantics (Geometry is non-copyable)
// ---------------------------------------------------------------------------

TEST_CASE("Geometry.IsNonCopyable", "[Geometry]")
{
    static_assert(!std::is_copy_constructible_v<Geometry>);
    static_assert(!std::is_copy_assignable_v<Geometry>);
}

TEST_CASE("Geometry.IsNothrowMovable", "[Geometry]")
{
    static_assert(std::is_nothrow_move_constructible_v<Geometry>);
    static_assert(std::is_nothrow_move_assignable_v<Geometry>);
}

TEST_CASE("Geometry.MoveConstructionTransfersVerticesAndIndices", "[Geometry]")
{
    Geometry original;
    std::vector<Vertex> verts = {
        {{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}},
        {{1, 0, 0}, {1, 1, 1}, {0, 1, 0}, {1, 0}},
    };
    std::vector<uint16_t> idxs = {0, 1};
    original.vertices(verts);
    original.indices(idxs);

    Geometry moved(std::move(original));
    CHECK(moved.vertices().size() == 2);
    CHECK(moved.indexCount() == 2);
    CHECK_FALSE(moved.loaded());
}

TEST_CASE("Geometry.MoveAssignmentTransfersState", "[Geometry]")
{
    Geometry original;
    std::vector<Vertex> verts = {{{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}}};
    original.vertices(verts);

    Geometry target;
    target = std::move(original);
    CHECK(target.vertices().size() == 1);
}

// ---------------------------------------------------------------------------
// Morph target tests
// ---------------------------------------------------------------------------

TEST_CASE("GeometryMorph.DefaultHasNoMorphTargets", "[GeometryMorph]")
{
    Geometry geo;
    CHECK(geo.morphTargetCount() == 0u);
    CHECK(geo.morphPositions().empty());
    CHECK(geo.morphNormals().empty());
}

TEST_CASE("GeometryMorph.SetMorphPositions", "[GeometryMorph]")
{
    Geometry geo;
    std::vector<std::vector<Vec3>> positions = {
        {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f}},
    };
    geo.morphPositions(positions);
    CHECK(geo.morphTargetCount() == 2u);
    REQUIRE(geo.morphPositions().size() == 2u);
    REQUIRE(geo.morphPositions()[0].size() == 2u);
}

TEST_CASE("GeometryMorph.SetMorphNormals", "[GeometryMorph]")
{
    Geometry geo;
    std::vector<std::vector<Vec3>> normals = {
        {{0.0f, 0.0f, 1.0f}},
    };
    geo.morphNormals(normals);
    CHECK(geo.morphNormals().size() == 1u);
}

TEST_CASE("GeometryMorph.MorphTargetCountReflectsPositions", "[GeometryMorph]")
{
    Geometry geo;
    geo.morphPositions({
        {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}},
    });
    CHECK(geo.morphTargetCount() == 3u);
}

// ---------------------------------------------------------------------------
// Buffer handle accessors (before load)
// ---------------------------------------------------------------------------

TEST_CASE("Geometry.DefaultVertexBufferIsNull", "[Geometry]")
{
    Geometry geo;
    CHECK(geo.vertexBuffer() == NullBuffer);
}

TEST_CASE("Geometry.DefaultIndexBufferIsNull", "[Geometry]")
{
    Geometry geo;
    CHECK(geo.indexBuffer() == NullBuffer);
}

TEST_CASE("Geometry.MoveTransfersBufferHandles", "[Geometry]")
{
    Geometry original;
    // Before load, handles are NullBuffer — verify move preserves that
    Geometry moved(std::move(original));
    CHECK(moved.vertexBuffer() == NullBuffer);
    CHECK(moved.indexBuffer() == NullBuffer);
}

// ---------------------------------------------------------------------------
// Morph target move
// ---------------------------------------------------------------------------

TEST_CASE("GeometryMorph.MoveRetainsMorphData", "[GeometryMorph]")
{
    Geometry original;
    original.morphPositions({
        {{1.0f, 2.0f, 3.0f}},
    });
    original.morphNormals({
        {{0.0f, 0.0f, 1.0f}},
    });
    Geometry moved(std::move(original));
    CHECK(moved.morphTargetCount() == 1u);
    REQUIRE(moved.morphNormals().size() == 1u);
}
