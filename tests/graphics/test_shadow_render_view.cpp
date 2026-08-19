#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <limits>

#include <fire_engine/graphics/shadow_render_view.hpp>

using namespace fire_engine;

namespace
{

// SH-07: metrics for a fixture view. Values are arbitrary but WELL-FORMED, and each family gets its
// own kind — the writers check the metrics kind against the slot, so a fixture cannot pin a pairing
// the production code is forbidden to make.
[[nodiscard]] ShadowViewMetrics someOrthoMetrics()
{
    return ShadowViewMetrics::orthographic(0.05f, 100.0f);
}
[[nodiscard]] ShadowViewMetrics someSpotMetrics()
{
    return ShadowViewMetrics::spot(0.002f, 0.1f, 50.0f);
}
[[nodiscard]] ShadowViewMetrics somePointMetrics()
{
    return ShadowViewMetrics::pointLight(0.004f, 25.0f);
}

// The effective range a point cube's faces store depth against. Any positive finite value will do
// for the fixtures that only care that the cube was accepted; the tests that care about the value
// itself pass their own.
constexpr float kPointRange = 25.0f;

Mat4 markedMatrix(float mark)
{
    Mat4 m = Mat4::identity();
    m[0, 3] = mark; // a translation, so each view's matrix is trivially identifiable
    return m;
}

ShadowView someOrtho()
{
    return ShadowView::orthographic(0.05f);
}

ShadowView somePerspective(const Vec3& forward = Vec3{0.0f, 0.0f, -1.0f})
{
    return ShadowView::perspective(Vec3{1.0f, 2.0f, 3.0f}, forward, 1.5708f, 512, 0.05f);
}

// A whole point cube for a fixture. Faces differ by matrix and forward — which is the only thing
// that legitimately varies per face — while the identity and metrics belong to the LIGHT and are
// passed once, because `setPointLight` is atomic.
[[nodiscard]] std::array<ShadowPointFace, kCubeFaceCount> someCube(float matrixSeed)
{
    const std::array<Vec3, kCubeFaceCount> forwards{Vec3{1, 0, 0},  Vec3{-1, 0, 0}, Vec3{0, 1, 0},
                                                    Vec3{0, -1, 0}, Vec3{0, 0, 1},  Vec3{0, 0, -1}};
    const auto at = [&](std::uint8_t face)
    {
        return ShadowPointFace{markedMatrix(matrixSeed + static_cast<float>(face)),
                               somePerspective(forwards[face])};
    };
    return {at(0), at(1), at(2), at(3), at(4), at(5)};
}

[[nodiscard]] std::span<const ShadowPointFace, kCubeFaceCount>
cubeSpan(const std::array<ShadowPointFace, kCubeFaceCount>& faces)
{
    return std::span<const ShadowPointFace, kCubeFaceCount>{faces};
}

} // namespace

// ---------------------------------------------------------------------------
// The per-frame shadow view set (SH-03 slice 2).
//
// Each entry holds the matrix, the projection descriptor and the logical identity together and
// read-only, written only through a family-specific writer. That makes the contradictions
// unrepresentable rather than merely detectable: there is no way to put a cascade identity in a
// spot slot, or an orthographic descriptor on a point face. Everything is Vulkan-free, so the
// mapping, the reset and the stale-slot behaviour are unit-testable rather than only visible in a
// frame capture.
// ---------------------------------------------------------------------------

TEST_CASE("ShadowRenderViewSet.StartsEmptyAndResetsEveryEntry", "[ShadowRenderView]")
{
    // The stale-slot failure: a view that stops being active this frame must not linger from the
    // last one and be read as current.
    ShadowRenderViewSet views;
    CHECK_FALSE(views.active(ShadowViewGroup::Cascade, 0));

    REQUIRE(views.setCascade(0, markedMatrix(1.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.setSpot(2, static_cast<NodeId>(4), markedMatrix(50.0f), somePerspective(),
                          someSpotMetrics()));
    CHECK(views.active(ShadowViewGroup::Cascade, 0));
    CHECK(views.active(ShadowViewGroup::Spot, 2));

    views.reset();
    for (std::size_t group = 0; group < kShadowViewGroupCount; ++group)
    {
        CAPTURE(group);
        CHECK(views.activeCount(static_cast<ShadowViewGroup>(group)) == 0);
    }
}

TEST_CASE("ShadowRenderViewSet.EachWriterStampsItsOwnSlotsIdentity", "[ShadowRenderView]")
{
    // The identity is derived from the slot being written, never supplied — so an entry cannot sit
    // in one slot while claiming to be another view.
    ShadowRenderViewSet views;
    const auto light = static_cast<NodeId>(31);
    REQUIRE(views.setCascade(2, markedMatrix(3.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.setSelf(1, 77, markedMatrix(20.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.setSpot(3, light, markedMatrix(30.0f), somePerspective(), someSpotMetrics()));
    REQUIRE(
        views.setPointLight(1, light, somePointMetrics(), kPointRange, cubeSpan(someCube(40.0f))));

    REQUIRE(views.find(ShadowViewGroup::Cascade, 2) != nullptr);
    CHECK(views.find(ShadowViewGroup::Cascade, 2)->logicalId() == ShadowLogicalViewId::cascade(2));
    CHECK(views.find(ShadowViewGroup::Self, 1)->logicalId() == ShadowLogicalViewId::self(77));
    CHECK(views.find(ShadowViewGroup::Spot, 3)->logicalId() == ShadowLogicalViewId::spot(light));
    // The point entry lands at the flat slot AND carries that same face.
    const ShadowRenderView* face = views.find(ShadowViewGroup::Point, shadowPointViewSlot(1, 4));
    REQUIRE(face != nullptr);
    CHECK(face->logicalId() == ShadowLogicalViewId::point(light, 4));
}

TEST_CASE("ShadowRenderViewSet.WritersRejectTheWrongProjectionKind", "[ShadowRenderView]")
{
    // A perspective descriptor in a cascade slot (or an orthographic one on a point face) is not
    // merely wrong, it is meaningless — and would be read back as authoritative. Dev asserts; these
    // expectations describe the release behaviour.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    const auto light = static_cast<NodeId>(9);

    CHECK_FALSE(views.setCascade(0, markedMatrix(1.0f), somePerspective(), someOrthoMetrics()));
    CHECK_FALSE(views.setSelf(0, 5, markedMatrix(2.0f), somePerspective(), someOrthoMetrics()));
    CHECK_FALSE(views.setSpot(0, light, markedMatrix(3.0f), someOrtho(), someSpotMetrics()));
    const auto orthoFace = ShadowPointFace{markedMatrix(4.0f), someOrtho()};
    const std::array<ShadowPointFace, kCubeFaceCount> orthoCube{orthoFace, orthoFace, orthoFace,
                                                                orthoFace, orthoFace, orthoFace};
    CHECK_FALSE(
        views.setPointLight(0, light, somePointMetrics(), kPointRange, cubeSpan(orthoCube)));

    CHECK(views.activeCount(ShadowViewGroup::Cascade) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Self) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Spot) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Point) == 0);
#endif
}

TEST_CASE("ShadowRenderViewSet.PointFacesCarryTheLightTheirDepthIsMeasuredAgainst",
          "[ShadowRenderView]")
{
    // A point face stores `distance / range` rather than projected depth, so the light's position
    // and its effective range are shader inputs — raster content, not consequences of the matrix.
    // They live on the view because a cached cube is only reusable while they are unchanged, and
    // the position comes from the face's own projection descriptor, which is also what LOD
    // selection measures depth from.
    ShadowRenderViewSet views;
    const auto light = static_cast<NodeId>(12);
    REQUIRE(views.setPointLight(0, light, somePointMetrics(), 42.0f, cubeSpan(someCube(1.0f))));

    const ShadowRenderView* face = views.find(ShadowViewGroup::Point, shadowPointViewSlot(0, 2));
    REQUIRE(face != nullptr);
    const std::optional<ShadowPointLightDepth> depth = face->pointLightDepth();
    REQUIRE(depth.has_value());
    CHECK(depth->range == 42.0f);
    CHECK(depth->position.x() == face->projection().lightPosition().x());
    CHECK(depth->position.y() == face->projection().lightPosition().y());
    CHECK(depth->position.z() == face->projection().lightPosition().z());

    // Nothing else has one. A cascade carries no light at all, and a zero position with a zero
    // range is a value a caller could read and push — so the answer is absence, not zeroes.
    REQUIRE(views.setCascade(0, markedMatrix(1.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.setSpot(0, light, markedMatrix(2.0f), somePerspective(), someSpotMetrics()));
    CHECK_FALSE(views.find(ShadowViewGroup::Cascade, 0)->pointLightDepth().has_value());
    CHECK_FALSE(views.find(ShadowViewGroup::Spot, 0)->pointLightDepth().has_value());
}

TEST_CASE("ShadowRenderViewSet.APointCubeNeedsOneLightAndAUsableRange", "[ShadowRenderView]")
{
    // Both halves of the stored ratio are checked as strictly as the matrices. A zero or non-finite
    // range makes every texel of all six faces meaningless; six faces about DIFFERENT positions
    // mean the caller assembled the cube from more than one light, and half of it would then be
    // measured from the wrong origin while every matrix still looked fine. Dev asserts; these
    // expectations describe the release behaviour.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    const auto light = static_cast<NodeId>(13);

    CHECK_FALSE(views.setPointLight(0, light, somePointMetrics(), 0.0f, cubeSpan(someCube(1.0f))));
    CHECK_FALSE(views.setPointLight(0, light, somePointMetrics(),
                                    std::numeric_limits<float>::infinity(),
                                    cubeSpan(someCube(1.0f))));
    CHECK(views.activeCount(ShadowViewGroup::Point) == 0);

    // Five faces about one light and a sixth about another.
    std::array<ShadowPointFace, kCubeFaceCount> mixed = someCube(1.0f);
    mixed[4] = ShadowPointFace{markedMatrix(9.0f), ShadowView::perspective(Vec3{40.0f, 0.0f, 0.0f},
                                                                           Vec3{0.0f, 0.0f, 1.0f},
                                                                           1.5708f, 512, 0.05f)};
    CHECK_FALSE(views.setPointLight(0, light, somePointMetrics(), kPointRange, cubeSpan(mixed)));
    // ALL SIX cleared, not five installed: a cube is accepted or refused whole.
    CHECK(views.activeCount(ShadowViewGroup::Point) == 0);
#endif
}

TEST_CASE("ShadowRenderViewSet.WritersRejectAnUnkeyableIdentity", "[ShadowRenderView]")
{
    // An engaged entry must always be keyable: hysteresis keys on the identity, so an invalid one
    // would leave the view rendering but its history unreachable.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    CHECK_FALSE(
        views.setSelf(0, 0, markedMatrix(1.0f), someOrtho(), someOrthoMetrics())); // objectId 0
    CHECK_FALSE(views.setSpot(0, NodeId::Invalid, markedMatrix(2.0f), somePerspective(),
                              someSpotMetrics()));
    // A face index out of range is no longer expressible — `setPointLight` takes a fixed span of
    // six — so what is left to reject here is the light's own identity.
    CHECK_FALSE(views.setPointLight(0, NodeId::Invalid, somePointMetrics(), kPointRange,
                                    cubeSpan(someCube(3.0f))));

    CHECK(views.activeCount(ShadowViewGroup::Self) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Spot) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Point) == 0);
#endif
}

TEST_CASE("ShadowRenderViewSet.AbsentMeansInactiveAndEngagedInvalidIsDifferent",
          "[ShadowRenderView]")
{
    // Two distinct states. An engaged entry with an unusable projection is a view that IS
    // rendering but whose descriptor could not be built — it must stay visible as an InvalidView
    // selection, not be demoted to "this view isn't running". Note the invalid descriptor still
    // carries its KIND, which is why the writer accepts it.
    ShadowRenderViewSet views;
    const ShadowView degenerate = ShadowView::orthographic(0.0f);
    REQUIRE_FALSE(degenerate.valid());
    REQUIRE(degenerate.kind() == ShadowViewKind::Orthographic);
    REQUIRE(views.setCascade(1, markedMatrix(7.0f), degenerate, someOrthoMetrics()));

    const ShadowRenderView* engaged = views.find(ShadowViewGroup::Cascade, 1);
    REQUIRE(engaged != nullptr);                // active: it will rasterise
    CHECK_FALSE(engaged->projection().valid()); // but selection must report InvalidView
    CHECK(engaged->logicalId().valid());        // and it is still keyable

    CHECK(views.find(ShadowViewGroup::Cascade, 2) == nullptr); // genuinely inactive
}

TEST_CASE("ShadowRenderViewSet.OutOfRangeAccessIsNullAndOutOfRangeWritesAreDropped",
          "[ShadowRenderView]")
{
    // Never clamped into a neighbouring valid slot: that would bill one view's matrix to another
    // and render wrongly instead of failing.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    const std::size_t cascades = shadowViewSlotCount(ShadowViewGroup::Cascade);

    CHECK_FALSE(views.setCascade(static_cast<std::uint32_t>(cascades), markedMatrix(1.0f),
                                 someOrtho(), someOrthoMetrics()));
    CHECK(views.activeCount(ShadowViewGroup::Cascade) == 0);
    CHECK(views.find(ShadowViewGroup::Cascade, cascades) == nullptr);
    CHECK(views.find(ShadowViewGroup::Count, 0) == nullptr);
#endif
}

TEST_CASE("ShadowRenderViewSet.ActiveCountIsNotADensePrefix", "[ShadowRenderView]")
{
    // Active slots have GAPS — a self-shadow slot can be freed while a later one stays occupied —
    // so a caller looping 0..activeCount() would render the wrong views. Every consumer must
    // iterate all physical slots and skip the inactive ones.
    ShadowRenderViewSet views;
    REQUIRE(views.setSelf(0, 11, markedMatrix(1.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.setSelf(2, 22, markedMatrix(2.0f), someOrtho(), someOrthoMetrics()));

    CHECK(views.activeCount(ShadowViewGroup::Self) == 2);
    CHECK(views.active(ShadowViewGroup::Self, 0));
    CHECK_FALSE(views.active(ShadowViewGroup::Self, 1)); // the gap
    CHECK(views.active(ShadowViewGroup::Self, 2));       // beyond activeCount()
}

TEST_CASE("ShadowRenderViewSet.WorldOnlyAliasesItsCascade", "[ShadowRenderView]")
{
    // Structural, not asserted: there is no way to give world-only a view of its own, so the two
    // passes cannot make different choices for a rigid caster.
    ShadowRenderViewSet views;
    for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        REQUIRE(views.setCascade(cascade, markedMatrix(static_cast<float>(cascade) + 1.0f),
                                 someOrtho(), someOrthoMetrics()));
        CHECK(views.enableWorldOnly(cascade));
    }

    for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        CAPTURE(cascade);
        const ShadowRenderView* full = views.find(ShadowViewGroup::Cascade, cascade);
        const ShadowRenderView* worldOnly = views.find(ShadowViewGroup::WorldOnly, cascade);
        REQUIRE(full != nullptr);
        REQUIRE(worldOnly != nullptr);
        CHECK(worldOnly->logicalId() == full->logicalId());
        CHECK(worldOnly->viewProj()[0, 3] == full->viewProj()[0, 3]);
        // The projection descriptor too — the earlier assert-based version compared matrix and
        // identity but never this, so a mismatched descriptor would have passed.
        CHECK(worldOnly->projection().worldUnitsPerTexel() ==
              full->projection().worldUnitsPerTexel());
        CHECK(worldOnly->projection().kind() == full->projection().kind());
    }
}

TEST_CASE("ShadowRenderViewSet.WorldOnlyFollowsACascadeRefitAfterEnabling", "[ShadowRenderView]")
{
    // The case a COPY would have got wrong: enable world-only, then re-fit the cascade. A snapshot
    // taken at enable time would leave the world-only pass rasterising the previous fit — the two
    // CSMs disagreeing for a rigid caster, which is exactly the invariant this type exists to hold.
    // Aliasing makes it hold under any later re-fit, in either order. Activation itself stays
    // ordered: enabling before the cascade exists is intentionally rejected (see the test below).
    ShadowRenderViewSet views;
    REQUIRE(views.setCascade(0, markedMatrix(1.0f), ShadowView::orthographic(0.25f),
                             someOrthoMetrics()));
    REQUIRE(views.enableWorldOnly(0));
    REQUIRE(views.setCascade(0, markedMatrix(2.0f), ShadowView::orthographic(0.5f),
                             someOrthoMetrics()));

    const ShadowRenderView* worldOnly = views.find(ShadowViewGroup::WorldOnly, 0);
    REQUIRE(worldOnly != nullptr);
    CHECK(worldOnly->viewProj()[0, 3] == 2.0f);
    CHECK(worldOnly->projection().worldUnitsPerTexel() == 0.5f);
}

TEST_CASE("ShadowRenderViewSet.WorldOnlyCannotOutliveOrPrecedeItsCascade", "[ShadowRenderView]")
{
#ifdef NDEBUG
    ShadowRenderViewSet views;
    CHECK_FALSE(views.enableWorldOnly(0)); // no cascade populated yet
    CHECK(views.activeCount(ShadowViewGroup::WorldOnly) == 0);
#endif
}

TEST_CASE("ShadowRenderViewSet.PointFacesOccupyDistinctFlatSlotsWithDistinctForwards",
          "[ShadowRenderView]")
{
    // Each cube face is its own view: same light position, different forward, so depth differs per
    // face. Sharing one descriptor across faces would give five of the six the wrong depth.
    const auto light = static_cast<NodeId>(31);
    const std::array<Vec3, kCubeFaceCount> forwards{Vec3{1, 0, 0},  Vec3{-1, 0, 0}, Vec3{0, 1, 0},
                                                    Vec3{0, -1, 0}, Vec3{0, 0, 1},  Vec3{0, 0, -1}};

    ShadowRenderViewSet views;
    constexpr std::size_t lightSlot = 2;
    const auto at = [&](std::uint8_t face)
    {
        return ShadowPointFace{markedMatrix(100.0f + static_cast<float>(face)),
                               somePerspective(forwards[face])};
    };
    const std::array<ShadowPointFace, kCubeFaceCount> cube{at(0), at(1), at(2),
                                                           at(3), at(4), at(5)};
    REQUIRE(views.setPointLight(lightSlot, light, somePointMetrics(), kPointRange, cubeSpan(cube)));

    CHECK(views.activeCount(ShadowViewGroup::Point) == kCubeFaceCount);
    for (std::uint8_t face = 0; face < kCubeFaceCount; ++face)
    {
        CAPTURE(face);
        const ShadowRenderView* view =
            views.find(ShadowViewGroup::Point, shadowPointViewSlot(lightSlot, face));
        REQUIRE(view != nullptr);
        CHECK(view->logicalId() == ShadowLogicalViewId::point(light, face));
        CHECK(view->projection().forward().x() == forwards[face].x());
        CHECK(view->projection().forward().y() == forwards[face].y());
        CHECK(view->projection().forward().z() == forwards[face].z());
        CHECK(view->viewProj()[0, 3] == 100.0f + static_cast<float>(face));
    }
    // A different light's faces do not overlap this one's.
    CHECK_FALSE(views.active(ShadowViewGroup::Point, shadowPointViewSlot(lightSlot + 1, 0)));
}

TEST_CASE("ShadowRenderViewSet.PointLightSlotIsValidatedBeforeFlattening", "[ShadowRenderView]")
{
    // `lightSlot * kCubeFaceCount + face` is unsigned arithmetic, so a large enough slot WRAPS back
    // into the valid range: with an even face count, the top bit times 6 is 0 modulo the word.
    // Validating only the flat result would accept it and overwrite a real light's face — a
    // rejection that turns into an aliased write.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    const auto real = static_cast<NodeId>(7);
    const auto impostor = static_cast<NodeId>(8);
    // The top bit of whatever width size_t is, rather than a hard-coded 63.
    constexpr std::size_t wrapping = std::size_t{1}
                                     << (std::numeric_limits<std::size_t>::digits - 1);
    STATIC_REQUIRE(kCubeFaceCount % 2 == 0);        // what makes the product wrap to zero
    REQUIRE(shadowPointViewSlot(wrapping, 1) == 1); // it really does land on light 0, face 1

    REQUIRE(
        views.setPointLight(0, real, somePointMetrics(), kPointRange, cubeSpan(someCube(11.0f))));
    CHECK_FALSE(views.setPointLight(wrapping, impostor, somePointMetrics(), kPointRange,
                                    cubeSpan(someCube(99.0f))));

    const ShadowRenderView* view = views.find(ShadowViewGroup::Point, shadowPointViewSlot(0, 1));
    REQUIRE(view != nullptr);
    CHECK(view->logicalId() == ShadowLogicalViewId::point(real, 1));
    // someCube marks face f with seed + f, so the real light's face 1 is 12 — unchanged by the
    // rejected write, which is the point.
    CHECK(view->viewProj()[0, 3] == 12.0f);
    // The whole cube, not one face: installation is atomic now, so the real light's six faces are
    // all that is active and the impostor added none.
    CHECK(views.activeCount(ShadowViewGroup::Point) == kCubeFaceCount);
#endif
}

TEST_CASE("ShadowRenderViewSet.ANonFiniteRenderMatrixIsNotAView", "[ShadowRenderView]")
{
    // The asymmetry that matters. An invalid PROJECTION is a reportable state: the view rasterises
    // and selection says InvalidView. A non-finite MATRIX is not a view at all — it is what the
    // pass rasterises with and what Frustum::fromViewProj culls with. A NaN makes the cull
    // PERMISSIVE (every comparison against a NaN plane is false, so nothing is rejected) while the
    // rasteriser draws nothing from NaN clip coordinates — wasted work and an empty map, with
    // selection reporting something reassuring about the descriptor beside it.
    ShadowRenderViewSet views;
    Mat4 poisoned = Mat4::identity();
    poisoned[2, 3] = std::numeric_limits<float>::quiet_NaN();

    // Finite matrix + invalid descriptor: ACCEPTED, and stays visible as an InvalidView selection.
    REQUIRE(views.setCascade(0, markedMatrix(1.0f), ShadowView::orthographic(0.0f),
                             someOrthoMetrics()));
    const ShadowRenderView* engaged = views.find(ShadowViewGroup::Cascade, 0);
    REQUIRE(engaged != nullptr);
    CHECK_FALSE(engaged->projection().valid());

#ifdef NDEBUG
    // Non-finite matrix + perfectly good descriptor: REJECTED. Dev asserts; this is the release
    // behaviour.
    CHECK_FALSE(views.setCascade(1, poisoned, someOrtho(), someOrthoMetrics()));
    CHECK_FALSE(
        views.setSpot(0, static_cast<NodeId>(3), poisoned, somePerspective(), someSpotMetrics()));
    CHECK_FALSE(views.active(ShadowViewGroup::Cascade, 1));
    CHECK(views.activeCount(ShadowViewGroup::Spot) == 0);

    // An infinity is the same failure with a different bit pattern.
    Mat4 unbounded = Mat4::identity();
    unbounded[0, 0] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(views.setSelf(0, 5, unbounded, someOrtho(), someOrthoMetrics()));
    CHECK_FALSE(views.active(ShadowViewGroup::Self, 0));
#endif
}

TEST_CASE("ShadowRenderViewSet.ARejectedReplacementClearsTheSlotItAddressed", "[ShadowRenderView]")
{
    // Rejecting a write is only half the job. A producer that ATTEMPTED to describe this view and
    // failed has invalidated whatever was there: keeping the previous entry would rasterise the
    // earlier fit under the new frame's expectations — stale geometry, reported as a healthy view.
    // "Absent" is the honest answer, and it is the one extraction asserts on for a cascade.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    Mat4 poisoned = Mat4::identity();
    poisoned[1, 2] = std::numeric_limits<float>::quiet_NaN();

    REQUIRE(views.setCascade(0, markedMatrix(1.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.active(ShadowViewGroup::Cascade, 0));
    CHECK_FALSE(views.setCascade(0, poisoned, someOrtho(), someOrthoMetrics()));
    CHECK_FALSE(views.active(ShadowViewGroup::Cascade, 0));

    // The same for an optional family, and via the other content check (wrong projection kind).
    REQUIRE(views.setSpot(1, static_cast<NodeId>(4), markedMatrix(2.0f), somePerspective(),
                          someSpotMetrics()));
    REQUIRE(views.active(ShadowViewGroup::Spot, 1));
    CHECK_FALSE(views.setSpot(1, static_cast<NodeId>(4), markedMatrix(3.0f), someOrtho(),
                              someSpotMetrics()));
    CHECK_FALSE(views.active(ShadowViewGroup::Spot, 1));

    // World-only inherits it through the alias: a cleared cascade leaves nothing to point at, so
    // the pass reads as inactive rather than quietly retaining the previous fit.
    REQUIRE(views.setCascade(2, markedMatrix(4.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.enableWorldOnly(2));
    CHECK_FALSE(views.setCascade(2, poisoned, someOrtho(), someOrthoMetrics()));
    CHECK_FALSE(views.active(ShadowViewGroup::WorldOnly, 2));

    // But an OUT-OF-RANGE address names no slot, so it must leave every real one alone — including
    // the wrapping point slot, which is rejected before it can be flattened onto a live face. Both
    // families are seeded first, so each rejection has a live entry it could have damaged.
    REQUIRE(views.setCascade(1, markedMatrix(8.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.setPointLight(0, static_cast<NodeId>(5), somePointMetrics(), kPointRange,
                                cubeSpan(someCube(5.0f))));
    constexpr std::size_t wrapping = std::size_t{1}
                                     << (std::numeric_limits<std::size_t>::digits - 1);
    CHECK_FALSE(views.setPointLight(wrapping, static_cast<NodeId>(6), somePointMetrics(),
                                    kPointRange, cubeSpan(someCube(6.0f))));
    CHECK_FALSE(
        views.setCascade(kShadowCascadeCount, markedMatrix(7.0f), someOrtho(), someOrthoMetrics()));

    REQUIRE(views.active(ShadowViewGroup::Point, shadowPointViewSlot(0, 0)));
    CHECK(views.find(ShadowViewGroup::Point, shadowPointViewSlot(0, 0))->viewProj()[0, 3] == 5.0f);
    REQUIRE(views.active(ShadowViewGroup::Cascade, 1));
    CHECK(views.find(ShadowViewGroup::Cascade, 1)->viewProj()[0, 3] == 8.0f);
#endif
}

TEST_CASE("ShadowRenderViewSet.BothSelfPassesReadOneEntry", "[ShadowRenderView]")
{
    // "Both depth passes of one slot use one choice" is structural: there is only one entry to
    // read, so they cannot diverge.
    ShadowRenderViewSet views;
    REQUIRE(views.setSelf(0, 5, markedMatrix(11.0f), someOrtho(), someOrthoMetrics()));

    const ShadowRenderView* first = views.find(ShadowViewGroup::Self, 0);
    const ShadowRenderView* second = views.find(ShadowViewGroup::Self, 0);
    REQUIRE(first != nullptr);
    CHECK(first == second);
    CHECK(first->logicalId() == ShadowLogicalViewId::self(5));
}

// --- Shader-array extraction -------------------------------------------------------------------

namespace
{
// Cascades are mandatory for extraction, so every extraction test starts from a populated set.
ShadowRenderViewSet withAllCascades()
{
    ShadowRenderViewSet views;
    for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        REQUIRE(views.setCascade(cascade, markedMatrix(static_cast<float>(cascade) + 1.0f),
                                 someOrtho(), someOrthoMetrics()));
    }
    return views;
}
} // namespace

// The combined matrix ARRAY is gone (arc 2 #4 step 1). It existed to fill
// `ShadowUBO::lightViewProj[32]`, which every shadow draw carried so a push constant could index
// one row; every path now rasterises with `pc.lightViewProj`, taken from the view being recorded.
// The per-family LightUBO extractors below are unaffected — those feed the RECEIVER, which still
// needs each family's matrices to project a fragment into light space.

TEST_CASE("ShadowRenderView.LightUboArraysExtractTheirOwnFamily", "[ShadowRenderView]")
{
    ShadowRenderViewSet views = withAllCascades();
    REQUIRE(views.setSpot(1, static_cast<NodeId>(8), markedMatrix(60.0f), somePerspective(),
                          someSpotMetrics()));
    REQUIRE(views.setSelf(2, 9, markedMatrix(70.0f), someOrtho(), someOrthoMetrics()));

    const auto cascades = cascadeViewProjArray(views);
    const auto spots = spotViewProjArray(views);
    const auto selves = selfShadowViewProjArray(views);

    // Sizes are compile-time facts, so an undersized destination is not a representable state.
    STATIC_REQUIRE(cascades.size() == kShadowCascadeCount);
    STATIC_REQUIRE(spots.size() == static_cast<std::size_t>(kMaxSpotShadowCasters));
    STATIC_REQUIRE(selves.size() == static_cast<std::size_t>(kMaxSkinnedSelfShadowCasters));

    CHECK(cascades[0][0, 3] == 1.0f);
    CHECK(cascades[3][0, 3] == 4.0f);
    CHECK(spots[1][0, 3] == 60.0f);
    CHECK(spots[0][0, 3] == 0.0f); // inactive ⇒ identity
    CHECK(selves[2][0, 3] == 70.0f);
    CHECK(selves[0][0, 3] == 0.0f);
    // No family leaks into another's array.
    CHECK(spots[2][0, 3] == 0.0f);
}

// ---------------------------------------------------------------------------
// SH-07: the bias metrics carried by the set, and the arrays extracted from it.
//
// This is the CPU half of a contract the GLSL depends on — `shaders/shadow_bias.glsl` reads these
// packings positionally, so a change here that nobody notices is a shader reading the wrong
// quantities with no error anywhere. The exact packings are pinned deliberately, not just their
// presence.
// ---------------------------------------------------------------------------

TEST_CASE("ShadowViewMetrics.EachKindPacksItsOwnQuantities", "[ShadowRenderView]")
{
    // Orthographic resolves BOTH quantities up front — they are constants of the layer.
    const auto ortho = ShadowViewMetrics::orthographic(0.25f, 80.0f);
    CHECK(ortho.kind() == ShadowViewMetricsKind::Orthographic);
    CHECK(ortho.packed()[0] == Catch::Approx(0.25f));
    CHECK(ortho.packed()[1] == Catch::Approx(1.0f / 80.0f)); // 1 / depthSpan
    CHECK(ortho.packed()[2] == Catch::Approx(0.0f));
    CHECK(ortho.packed()[3] == Catch::Approx(0.0f));

    // Spot resolves NEITHER: the receiver derives both from its own depth, so what travels is the
    // constants it needs to do that.
    const auto spot = ShadowViewMetrics::spot(0.002f, 0.1f, 50.0f);
    CHECK(spot.kind() == ShadowViewMetricsKind::Spot);
    CHECK(spot.packed()[0] == Catch::Approx(0.002f)); // texel angle scale
    CHECK(spot.packed()[1] == Catch::Approx(0.1f));   // near
    CHECK(spot.packed()[2] == Catch::Approx(50.0f));  // far
    CHECK(spot.packed()[3] == Catch::Approx(0.0f));

    // Point resolves the depth conversion (linear in range) but not the footprint (major axis, per
    // fragment).
    const auto point = ShadowViewMetrics::pointLight(0.004f, 25.0f);
    CHECK(point.kind() == ShadowViewMetricsKind::PointLight);
    CHECK(point.packed()[0] == Catch::Approx(0.004f));
    CHECK(point.packed()[1] == Catch::Approx(1.0f / 25.0f)); // 1 / range
    CHECK(point.packed()[2] == Catch::Approx(0.0f));
    CHECK(point.packed()[3] == Catch::Approx(0.0f));
}

TEST_CASE("ShadowViewMetrics.ADegenerateFitPacksNoConversionRatherThanAnInfinity",
          "[ShadowRenderView]")
{
    // A collapsed depth span or a zero range would divide by zero. Zero propagates into "no bias",
    // which shows as acne — visible and locatable; an infinity would detach every shadow the view
    // casts.
    CHECK(ShadowViewMetrics::orthographic(0.25f, 0.0f).packed()[1] == Catch::Approx(0.0f));
    CHECK(ShadowViewMetrics::pointLight(0.004f, 0.0f).packed()[1] == Catch::Approx(0.0f));
}

TEST_CASE("ShadowRenderViewSet.WritersRejectMetricsOfTheWrongKind", "[ShadowRenderView]")
{
    // The three packings share a shape, so a mismatch is not caught by anything downstream: the
    // receiver would simply read a spot's near plane as a cascade's depth conversion and produce a
    // bias. Rejected here for the same reason a perspective descriptor is rejected from a cascade
    // slot.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    const auto light = static_cast<NodeId>(12);

    CHECK_FALSE(views.setCascade(0, markedMatrix(1.0f), someOrtho(), someSpotMetrics()));
    CHECK_FALSE(views.setSelf(0, 5, markedMatrix(2.0f), someOrtho(), somePointMetrics()));
    CHECK_FALSE(views.setSpot(0, light, markedMatrix(3.0f), somePerspective(), someOrthoMetrics()));
    CHECK_FALSE(
        views.setPointLight(0, light, someSpotMetrics(), kPointRange, cubeSpan(someCube(4.0f))));

    CHECK(views.activeCount(ShadowViewGroup::Cascade) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Self) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Spot) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Point) == 0);
#endif
}

TEST_CASE("ShadowRenderViewSet.APointCubeCarriesOneMetricForTheWholeLight", "[ShadowRenderView]")
{
    // The invariant the atomic writer exists to make structural: every face reports the LIGHT's
    // metrics, because there is only one value and it is copied into all six. A per-face writer let
    // two faces disagree about the range while extraction trusted face 0.
    ShadowRenderViewSet views;
    const auto light = static_cast<NodeId>(21);
    const auto metrics = ShadowViewMetrics::pointLight(0.008f, 40.0f);
    REQUIRE(views.setPointLight(2, light, metrics, kPointRange, cubeSpan(someCube(7.0f))));

    for (std::uint8_t face = 0; face < kCubeFaceCount; ++face)
    {
        CAPTURE(face);
        const ShadowRenderView* view =
            views.find(ShadowViewGroup::Point, shadowPointViewSlot(2, face));
        REQUIRE(view != nullptr);
        CHECK(view->biasMetrics().packed()[0] == Catch::Approx(metrics.packed()[0]));
        CHECK(view->biasMetrics().packed()[1] == Catch::Approx(metrics.packed()[1]));
    }
}

TEST_CASE("ShadowRenderViewSet.ARejectedCubeLeavesNoFaceBehind", "[ShadowRenderView]")
{
    // "All six faces or none" is now the writer's contract rather than a comment in the renderer. A
    // five-face cube is not a usable caster: the missing face would sample whatever the previous
    // frame left, which is a shadow that is wrong from exactly one direction.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    const auto light = static_cast<NodeId>(33);
    REQUIRE(
        views.setPointLight(0, light, somePointMetrics(), kPointRange, cubeSpan(someCube(10.0f))));
    REQUIRE(views.activeCount(ShadowViewGroup::Point) == kCubeFaceCount);

    // One bad face rejects the whole cube AND clears the previously good one.
    auto cube = someCube(20.0f);
    cube[4] =
        ShadowPointFace{markedMatrix(std::numeric_limits<float>::quiet_NaN()), somePerspective()};
    CHECK_FALSE(views.setPointLight(0, light, somePointMetrics(), kPointRange, cubeSpan(cube)));
    CHECK(views.activeCount(ShadowViewGroup::Point) == 0);
#endif
}

TEST_CASE("ShadowRenderView.BiasMetricArraysFollowTheirFamilies", "[ShadowRenderView]")
{
    // Family placement and inactive zero-fill, per extractor. Zeros are the "no metrics" the law
    // answers with no bias, so an extractor writing a plausible default instead would be inventing
    // a scale for a view that does not exist.
    ShadowRenderViewSet views;
    const auto light = static_cast<NodeId>(9);
    REQUIRE(views.setCascade(1, markedMatrix(1.0f), someOrtho(),
                             ShadowViewMetrics::orthographic(0.5f, 20.0f)));
    REQUIRE(views.setSelf(2, 44, markedMatrix(2.0f), someOrtho(),
                          ShadowViewMetrics::orthographic(0.125f, 4.0f)));
    REQUIRE(views.setSpot(3, light, markedMatrix(3.0f), somePerspective(),
                          ShadowViewMetrics::spot(0.01f, 0.2f, 30.0f)));
    REQUIRE(views.setPointLight(1, light, ShadowViewMetrics::pointLight(0.02f, 10.0f), kPointRange,
                                cubeSpan(someCube(4.0f))));

    const auto cascades = cascadeBiasMetricsArray(views);
    CHECK(cascades[1][0] == Catch::Approx(0.5f));
    CHECK(cascades[1][1] == Catch::Approx(1.0f / 20.0f));
    CHECK(cascades[0][0] == Catch::Approx(0.0f)); // inactive
    CHECK(cascades[0][1] == Catch::Approx(0.0f));

    const auto selves = selfBiasMetricsArray(views);
    CHECK(selves[2][0] == Catch::Approx(0.125f));
    CHECK(selves[2][1] == Catch::Approx(1.0f / 4.0f));
    CHECK(selves[0][0] == Catch::Approx(0.0f));

    const auto spots = spotBiasMetricsArray(views);
    CHECK(spots[3][0] == Catch::Approx(0.01f));
    CHECK(spots[3][1] == Catch::Approx(0.2f));
    CHECK(spots[3][2] == Catch::Approx(30.0f));
    CHECK(spots[0][0] == Catch::Approx(0.0f));

    // Point metrics are indexed by LIGHT, not by flat face slot.
    const auto points = pointBiasMetricsArray(views);
    CHECK(points[1][0] == Catch::Approx(0.02f));
    CHECK(points[1][1] == Catch::Approx(1.0f / 10.0f));
    CHECK(points[0][0] == Catch::Approx(0.0f));
}

TEST_CASE("ShadowRenderView.WorldOnlyReadsItsCascadesRefittedMetrics", "[ShadowRenderView]")
{
    // World-only stores nothing of its own, so a cascade refitted AFTER it was enabled must move
    // both passes' metrics together — the same aliasing the matrices rely on. A copy taken at
    // enable time would bias the world-only map against the previous frame's fit.
    ShadowRenderViewSet views;
    REQUIRE(views.setCascade(0, markedMatrix(1.0f), someOrtho(),
                             ShadowViewMetrics::orthographic(0.5f, 20.0f)));
    REQUIRE(views.enableWorldOnly(0));
    REQUIRE(views.setCascade(0, markedMatrix(2.0f), someOrtho(),
                             ShadowViewMetrics::orthographic(4.0f, 200.0f)));

    const ShadowRenderView* worldOnly = views.find(ShadowViewGroup::WorldOnly, 0);
    REQUIRE(worldOnly != nullptr);
    CHECK(worldOnly->biasMetrics().packed()[0] == Catch::Approx(4.0f));
    CHECK(worldOnly->biasMetrics().packed()[1] == Catch::Approx(1.0f / 200.0f));
}
