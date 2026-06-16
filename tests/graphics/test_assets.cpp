#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include <fire_engine/graphics/assets.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST_CASE("Assets.DefaultConstructionHasZeroCounts", "[Assets]")
{
    Assets assets;
    CHECK(assets.textureCount() == 0);
    CHECK(assets.materialCount() == 0);
    CHECK(assets.geometryCount() == 0);
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

TEST_CASE("Assets.ResizeTexturesUpdatesCounts", "[Assets]")
{
    Assets assets;
    assets.resizeTextures(5);
    CHECK(assets.textureCount() == 5);
}

TEST_CASE("Assets.ResizeMaterialsUpdatesCounts", "[Assets]")
{
    Assets assets;
    assets.resizeMaterials(3);
    CHECK(assets.materialCount() == 3);
}

TEST_CASE("Assets.ResizeGeometriesUpdatesCounts", "[Assets]")
{
    Assets assets;
    assets.resizeGeometries(7);
    CHECK(assets.geometryCount() == 7);
}

// ---------------------------------------------------------------------------
// Material access and modification
// ---------------------------------------------------------------------------

TEST_CASE("Assets.MaterialAccessByIndex", "[Assets]")
{
    Assets assets;
    assets.resizeMaterials(2);
    assets.material(0).name("mat_a");
    assets.material(1).name("mat_b");

    CHECK(assets.material(0).name() == "mat_a");
    CHECK(assets.material(1).name() == "mat_b");
}

TEST_CASE("Assets.ConstMaterialAccessByIndex", "[Assets]")
{
    Assets assets;
    assets.resizeMaterials(1);
    assets.material(0).name("const_test");

    const auto& constAssets = assets;
    CHECK(constAssets.material(0).name() == "const_test");
}

// ---------------------------------------------------------------------------
// Geometry access (CPU-side, no Renderer needed)
// ---------------------------------------------------------------------------

TEST_CASE("Assets.GeometryAccessSetVerticesAndIndices", "[Assets]")
{
    Assets assets;
    assets.resizeGeometries(1);

    std::vector<Vertex> verts = {
        {{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}},
        {{1, 0, 0}, {1, 1, 1}, {0, 1, 0}, {1, 0}},
        {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}, {0, 1}},
    };
    std::vector<uint16_t> idxs = {0, 1, 2};

    assets.geometry(0).vertices(verts);
    assets.geometry(0).indices(idxs);

    CHECK(assets.geometry(0).vertices().size() == 3);
    CHECK(assets.geometry(0).indices().size() == 3);
    CHECK(assets.geometry(0).indexCount() == 3);
}

TEST_CASE("Assets.GeometryDefaultNotLoaded", "[Assets]")
{
    Assets assets;
    assets.resizeGeometries(1);
    CHECK_FALSE(assets.geometry(0).loaded());
}

// ---------------------------------------------------------------------------
// Pointer stability after resize
// ---------------------------------------------------------------------------

TEST_CASE("Assets.MaterialPointerStableAfterModification", "[Assets]")
{
    Assets assets;
    assets.resizeMaterials(3);

    Material* ptr = &assets.material(1);
    ptr->name("stable_test");

    CHECK(assets.material(1).name() == "stable_test");
    CHECK(ptr == &assets.material(1));
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

TEST_CASE("Assets.MoveConstructionTransfersOwnership", "[Assets]")
{
    Assets original;
    original.resizeMaterials(2);
    original.resizeGeometries(3);
    original.material(0).name("moved");

    Assets moved(std::move(original));

    CHECK(moved.materialCount() == 2);
    CHECK(moved.geometryCount() == 3);
    CHECK(moved.material(0).name() == "moved");
}

TEST_CASE("Assets.MoveAssignmentTransfersOwnership", "[Assets]")
{
    Assets original;
    original.resizeMaterials(1);
    original.material(0).name("assigned");

    Assets target;
    target = std::move(original);

    CHECK(target.materialCount() == 1);
    CHECK(target.material(0).name() == "assigned");
}

// ---------------------------------------------------------------------------
// Non-copyable
// ---------------------------------------------------------------------------

TEST_CASE("Assets.IsNonCopyable", "[Assets]")
{
    static_assert(!std::is_copy_constructible_v<Assets>);
    static_assert(!std::is_copy_assignable_v<Assets>);
}
