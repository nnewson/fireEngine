#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/object.hpp>
#include <fire_engine/graphics/shadow_caster_bounds_frame.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/render/cascade_fit.hpp>
#include <fire_engine/render/constants.hpp>

using Catch::Approx;
using fire_engine::Geometry;
using fire_engine::Mat4;
using fire_engine::Material;
using fire_engine::Object;
using fire_engine::ShadowCasterBoundsFrame;
using fire_engine::ShadowCasterBoundsKind;
using fire_engine::Vec3;
using fire_engine::Vertex;

// The SH-06 prepass, at the level that actually decides what the depth fit sees. Headless: the walk
// reads CPU vertex data and world matrices only, so no GPU device is involved.
namespace
{

// A unit box of vertices centred on `centre`, so a binding's extent is known exactly.
[[nodiscard]] std::vector<Vertex> boxVertices(Vec3 centre)
{
    std::vector<Vertex> vertices;
    for (int corner = 0; corner < 8; ++corner)
    {
        Vertex v{};
        v.position(centre + Vec3{(corner & 1) != 0 ? 1.0f : -1.0f, (corner & 2) != 0 ? 1.0f : -1.0f,
                                 (corner & 4) != 0 ? 1.0f : -1.0f});
        vertices.push_back(v);
    }
    return vertices;
}

struct Mesh
{
    Material material{};
    Geometry geometry{};

    explicit Mesh(Vec3 centre)
    {
        geometry.material(&material);
        geometry.vertices(boxVertices(centre));
    }
};

} // namespace

TEST_CASE("ObjectShadowCasters.DisjointBindingsStaySeparate", "[Object][ShadowCasterBounds]")
{
    // The defect this replaces: the draw path computed an object-WIDE union and stamped it onto
    // every binding's command. Two bindings ten metres apart would each claim a box spanning both.
    Mesh left{Vec3{-10.0f, 0.0f, 0.0f}};
    Mesh right{Vec3{10.0f, 0.0f, 0.0f}};
    Object object;
    object.addGeometry(left.geometry);
    object.addGeometry(right.geometry);

    ShadowCasterBoundsFrame frame;
    frame.reset();
    object.gatherShadowCasterBounds(Mat4::identity(), frame);

    REQUIRE(frame.size() == 2u);
    const auto entries = frame.entries();
    // Each entry spans its own box (2 units wide), not the 22-unit union of both.
    for (const auto& entry : entries)
    {
        CHECK(entry.world.valid);
        CHECK(entry.world.max.x() - entry.world.min.x() == Approx(2.0f));
        CHECK(entry.kind == ShadowCasterBoundsKind::Exact);
    }
    CHECK(entries[0].casterId != entries[1].casterId);
}

TEST_CASE("ObjectShadowCasters.NonCastingBindingsAreAbsent", "[Object][ShadowCasterBounds]")
{
    // A binding the shadow pass will never rasterise must not widen the range the fit computes.
    Mesh caster{Vec3{0.0f, 0.0f, 0.0f}};
    Mesh receiverOnly{Vec3{100.0f, 0.0f, 0.0f}};
    Object object;
    object.addGeometry(caster.geometry, true);
    object.addGeometry(receiverOnly.geometry, false);

    ShadowCasterBoundsFrame frame;
    frame.reset();
    object.gatherShadowCasterBounds(Mat4::identity(), frame);

    REQUIRE(frame.size() == 1u);
    // The far-away receive-only box would have dragged the union out to x = 101.
    CHECK(frame.entries()[0].world.max.x() == Approx(1.0f));
}

TEST_CASE("ObjectShadowCasters.BoundsFollowTheWorldTransform", "[Object][ShadowCasterBounds]")
{
    Mesh mesh{Vec3{0.0f, 0.0f, 0.0f}};
    Object object;
    object.addGeometry(mesh.geometry);

    ShadowCasterBoundsFrame frame;
    frame.reset();
    object.gatherShadowCasterBounds(Mat4::translate({5.0f, 0.0f, 0.0f}), frame);

    REQUIRE(frame.size() == 1u);
    CHECK(frame.entries()[0].world.min.x() == Approx(4.0f));
    CHECK(frame.entries()[0].world.max.x() == Approx(6.0f));
}

TEST_CASE("ObjectShadowCasters.MorphedGeometryIsExactAtItsCurrentWeights",
          "[Object][ShadowCasterBounds]")
{
    // Morph deltas are applied by the walk, so the bounds describe the pose that will DRAW — which
    // is what makes them `Exact` even though the mesh is not in its bind pose.
    Mesh mesh{Vec3{0.0f, 0.0f, 0.0f}};
    std::vector<Vec3> deltas(mesh.geometry.vertices().size(), Vec3{0.0f, 3.0f, 0.0f});
    mesh.geometry.morphPositions({deltas});
    Object object;
    object.addGeometry(mesh.geometry);
    object.morphWeights(std::vector<float>{1.0f});

    ShadowCasterBoundsFrame frame;
    frame.reset();
    object.gatherShadowCasterBounds(Mat4::identity(), frame);

    REQUIRE(frame.size() == 1u);
    CHECK(frame.entries()[0].kind == ShadowCasterBoundsKind::Exact);
    // Fully weighted, so the whole box moved up by the delta.
    CHECK(frame.entries()[0].world.min.y() == Approx(2.0f));
    CHECK(frame.entries()[0].world.max.y() == Approx(4.0f));
}

TEST_CASE("ObjectShadowCasters.StorageVertexGeometryIsStale", "[Object][ShadowCasterBounds]")
{
    // Cloth. Nothing about the INSTANCE deforms — no skin, no morph weights — but a compute pass
    // rewrites the vertex buffer every frame, so the CPU copy this walk reads is the bind pose.
    // The bounds are still reported (they are useful evidence) and marked as what they are.
    Mesh cloth{Vec3{0.0f, 0.0f, 0.0f}};
    cloth.geometry.storageVertices(true);
    Object object;
    object.addGeometry(cloth.geometry);

    ShadowCasterBoundsFrame frame;
    frame.reset();
    object.gatherShadowCasterBounds(Mat4::identity(), frame);

    REQUIRE(frame.size() == 1u);
    CHECK(frame.entries()[0].kind == ShadowCasterBoundsKind::Stale);
    CHECK(frame.entries()[0].world.valid);
}

TEST_CASE("ObjectShadowCasters.ClothIsNotCulledByItsBindPoseBound", "[Object][ShadowCasterBounds]")
{
    // `deformable()` answers "does this instance carry a skin or morph weights", and cloth answers
    // no — which is correct for classifying a sibling binding's shadow deformation, and wrong for
    // deciding whether the coarse cull may reject the node. The separate predicate keeps both
    // answers available instead of broadening one of them.
    Mesh cloth{Vec3{0.0f, 0.0f, 0.0f}};
    cloth.geometry.storageVertices(true);
    Object clothObject;
    clothObject.addGeometry(cloth.geometry);

    CHECK_FALSE(clothObject.deformable());
    CHECK_FALSE(clothObject.localBoundsCoverDrawnGeometry());

    Mesh rigid{Vec3{0.0f, 0.0f, 0.0f}};
    Object rigidObject;
    rigidObject.addGeometry(rigid.geometry);
    CHECK(rigidObject.localBoundsCoverDrawnGeometry());
}

TEST_CASE("ObjectShadowCasters.TwoObjectsSharingOneMeshGetTheirOwnEntries",
          "[Object][ShadowCasterBounds]")
{
    // One Geometry, two instances at different places: the prepass must key on the BINDING, not the
    // geometry, or the second instance would collide with the first and the frame would reject it.
    Mesh shared{Vec3{0.0f, 0.0f, 0.0f}};
    Object first;
    first.addGeometry(shared.geometry);
    Object second;
    second.addGeometry(shared.geometry);

    ShadowCasterBoundsFrame frame;
    frame.reset();
    first.gatherShadowCasterBounds(Mat4::identity(), frame);
    second.gatherShadowCasterBounds(Mat4::translate({20.0f, 0.0f, 0.0f}), frame);

    REQUIRE(frame.size() == 2u);
    CHECK(frame.entries()[0].world.max.x() == Approx(1.0f));
    CHECK(frame.entries()[1].world.max.x() == Approx(21.0f));
}

TEST_CASE("ObjectShadowCasters.ACorruptVertexIsReportedNotSwallowed",
          "[Object][ShadowCasterBounds]")
{
    // The prepass walks vertices with `Bounds3::expandChecked`, not `expand`, because std::min /
    // std::max return the OTHER operand when one side is NaN: a corrupt vertex would otherwise
    // leave a perfectly finite box that does not contain the geometry it claims to, and SH-06's
    // depth range would be fitted tight around a caster it never accounted for — and clip it.
    Mesh mesh{Vec3{0.0f, 0.0f, 0.0f}};
    auto vertices = mesh.geometry.vertices();
    REQUIRE(vertices.size() > 1u);
    vertices[1].position(Vec3{std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f});
    mesh.geometry.vertices(vertices);

    Object object;
    object.addGeometry(mesh.geometry);

    ShadowCasterBoundsFrame frame;
    frame.reset();
    object.gatherShadowCasterBounds(Mat4::identity(), frame);

    REQUIRE(frame.size() == 1u);
    const auto& recorded = frame.entries()[0];
    // VALID but non-finite: this caster has an extent and the engine cannot state it. Distinct from
    // a binding with no vertices, which is invalid and contributes nothing.
    CHECK(recorded.world.valid);
    CHECK_FALSE(std::isfinite(recorded.world.min.x()));

    // And that is terminal for the depth policy rather than a caster it quietly skips.
    const fire_engine::CascadeReceiverInput input{.cameraPosition = Vec3{0.0f, 2.0f, 8.0f},
                                                  .cameraTarget = Vec3{0.0f, 1.0f, 0.0f},
                                                  .lightDirection =
                                                      Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f}),
                                                  .fovRadians = fire_engine::kCameraFovRadians,
                                                  .aspect = 4.0f / 3.0f,
                                                  .sliceNear = 1.0f,
                                                  .sliceFar = 12.0f,
                                                  .shadowMapExtent = fire_engine::kShadowMapExtent};
    const auto receiver = fire_engine::CascadeReceiverFit::fit(input);
    REQUIRE(receiver);
    CHECK_FALSE(fire_engine::fitCasterAwareCascadeDepth(*receiver, frame.entries(),
                                                        fire_engine::kShadowDepthBackExtend)
                    .has_value());
}

TEST_CASE("ObjectShadowCasters.EmptyGeometryIsDistinctFromCorrupt", "[Object][ShadowCasterBounds]")
{
    // No vertices: no extent to state, invalid bounds, and the policy simply skips it — the
    // difference the checked build exists to preserve.
    Material material{};
    Geometry empty{};
    empty.material(&material);
    Object object;
    object.addGeometry(empty);

    ShadowCasterBoundsFrame frame;
    frame.reset();
    object.gatherShadowCasterBounds(Mat4::identity(), frame);

    REQUIRE(frame.size() == 1u);
    CHECK_FALSE(frame.entries()[0].world.valid);
}
