#include <gtest/gtest.h>

#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Construction and defaults
// ---------------------------------------------------------------------------

TEST(Geometry, DefaultConstructionNotLoaded)
{
    Geometry geo;
    EXPECT_FALSE(geo.loaded());
}

TEST(Geometry, DefaultConstructionEmptyVertices)
{
    Geometry geo;
    EXPECT_TRUE(geo.vertices().empty());
}

TEST(Geometry, DefaultConstructionEmptyIndices)
{
    Geometry geo;
    EXPECT_TRUE(geo.indices().empty());
    EXPECT_EQ(geo.indexCount(), 0);
}

// ---------------------------------------------------------------------------
// Vertices
// ---------------------------------------------------------------------------

TEST(Geometry, SetAndGetVertices)
{
    Geometry geo;
    std::vector<Vertex> verts = {
        {{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}},
        {{1, 0, 0}, {1, 1, 1}, {0, 1, 0}, {1, 0}},
        {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}, {0, 1}},
    };
    geo.vertices(verts);
    EXPECT_EQ(geo.vertices().size(), 3);
}

TEST(Geometry, VerticesMoveSemantics)
{
    Geometry geo;
    std::vector<Vertex> verts = {
        {{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}},
    };
    geo.vertices(std::move(verts));
    EXPECT_EQ(geo.vertices().size(), 1);
}

// ---------------------------------------------------------------------------
// Indices
// ---------------------------------------------------------------------------

TEST(Geometry, SetAndGetIndices)
{
    Geometry geo;
    std::vector<uint16_t> idxs = {0, 1, 2, 2, 3, 0};
    geo.indices(idxs);
    EXPECT_EQ(geo.indices().size(), 6);
    EXPECT_EQ(geo.indexCount(), 6);
}

TEST(Geometry, IndicesMoveSemantics)
{
    Geometry geo;
    std::vector<uint16_t> idxs = {0, 1, 2};
    geo.indices(std::move(idxs));
    EXPECT_EQ(geo.indexCount(), 3);
}

TEST(Geometry, SetAndGetUInt32Indices)
{
    Geometry geo;
    std::vector<uint32_t> idxs = {0u, 65536u, 70000u};
    geo.indices(idxs);
    EXPECT_EQ(geo.indices().size(), 3u);
    EXPECT_EQ(geo.indices()[1], 65536u);
    EXPECT_EQ(geo.indices()[2], 70000u);
    EXPECT_EQ(geo.indexType(), DrawIndexType::UInt32);
}

TEST(Geometry, CastsShadowDefaultsToTrue)
{
    Geometry geo;
    EXPECT_TRUE(geo.castsShadow());
}

TEST(Geometry, CastsShadowRoundTrip)
{
    Geometry geo;
    geo.castsShadow(false);
    EXPECT_FALSE(geo.castsShadow());
    geo.castsShadow(true);
    EXPECT_TRUE(geo.castsShadow());
}

// ---------------------------------------------------------------------------
// Material pointer
// ---------------------------------------------------------------------------

TEST(Geometry, DefaultMaterialIsNull)
{
    Geometry geo;
    // material() dereferences the pointer, so we just check we can set one
    Material mat;
    geo.material(&mat);
    EXPECT_EQ(geo.material().name(), mat.name());
}

TEST(Geometry, MaterialPointerAssignment)
{
    Geometry geo;
    Material mat;
    mat.name("test_mat");
    geo.material(&mat);
    EXPECT_EQ(geo.material().name(), "test_mat");
}

// ---------------------------------------------------------------------------
// Move semantics (Geometry is non-copyable)
// ---------------------------------------------------------------------------

TEST(Geometry, IsNonCopyable)
{
    EXPECT_FALSE(std::is_copy_constructible_v<Geometry>);
    EXPECT_FALSE(std::is_copy_assignable_v<Geometry>);
}

TEST(Geometry, IsNothrowMovable)
{
    EXPECT_TRUE(std::is_nothrow_move_constructible_v<Geometry>);
    EXPECT_TRUE(std::is_nothrow_move_assignable_v<Geometry>);
}

TEST(Geometry, MoveConstructionTransfersVerticesAndIndices)
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
    EXPECT_EQ(moved.vertices().size(), 2);
    EXPECT_EQ(moved.indexCount(), 2);
    EXPECT_FALSE(moved.loaded());
}

TEST(Geometry, MoveAssignmentTransfersState)
{
    Geometry original;
    std::vector<Vertex> verts = {{{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}}};
    original.vertices(verts);

    Geometry target;
    target = std::move(original);
    EXPECT_EQ(target.vertices().size(), 1);
}

// ---------------------------------------------------------------------------
// Morph target tests
// ---------------------------------------------------------------------------

TEST(GeometryMorph, DefaultHasNoMorphTargets)
{
    Geometry geo;
    EXPECT_EQ(geo.morphTargetCount(), 0u);
    EXPECT_TRUE(geo.morphPositions().empty());
    EXPECT_TRUE(geo.morphNormals().empty());
}

TEST(GeometryMorph, SetMorphPositions)
{
    Geometry geo;
    std::vector<std::vector<Vec3>> positions = {
        {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f}},
    };
    geo.morphPositions(positions);
    EXPECT_EQ(geo.morphTargetCount(), 2u);
    EXPECT_EQ(geo.morphPositions().size(), 2u);
    EXPECT_EQ(geo.morphPositions()[0].size(), 2u);
}

TEST(GeometryMorph, SetMorphNormals)
{
    Geometry geo;
    std::vector<std::vector<Vec3>> normals = {
        {{0.0f, 0.0f, 1.0f}},
    };
    geo.morphNormals(normals);
    EXPECT_EQ(geo.morphNormals().size(), 1u);
}

TEST(GeometryMorph, MorphTargetCountReflectsPositions)
{
    Geometry geo;
    geo.morphPositions({
        {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}},
    });
    EXPECT_EQ(geo.morphTargetCount(), 3u);
}

// ---------------------------------------------------------------------------
// Buffer handle accessors (before load)
// ---------------------------------------------------------------------------

TEST(Geometry, DefaultVertexBufferIsNull)
{
    Geometry geo;
    EXPECT_EQ(geo.vertexBuffer(), NullBuffer);
}

TEST(Geometry, DefaultIndexBufferIsNull)
{
    Geometry geo;
    EXPECT_EQ(geo.indexBuffer(), NullBuffer);
}

TEST(Geometry, MoveTransfersBufferHandles)
{
    Geometry original;
    // Before load, handles are NullBuffer — verify move preserves that
    Geometry moved(std::move(original));
    EXPECT_EQ(moved.vertexBuffer(), NullBuffer);
    EXPECT_EQ(moved.indexBuffer(), NullBuffer);
}

// ---------------------------------------------------------------------------
// Morph target move
// ---------------------------------------------------------------------------

TEST(GeometryMorph, MoveRetainsMorphData)
{
    Geometry original;
    original.morphPositions({
        {{1.0f, 2.0f, 3.0f}},
    });
    original.morphNormals({
        {{0.0f, 0.0f, 1.0f}},
    });
    Geometry moved(std::move(original));
    EXPECT_EQ(moved.morphTargetCount(), 1u);
    EXPECT_EQ(moved.morphNormals().size(), 1u);
}
