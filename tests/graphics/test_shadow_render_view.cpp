#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <limits>

#include <fire_engine/graphics/shadow_render_view.hpp>

using namespace fire_engine;

namespace
{

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

    REQUIRE(views.setCascade(0, markedMatrix(1.0f), someOrtho()));
    REQUIRE(views.setSpot(2, static_cast<NodeId>(4), markedMatrix(50.0f), somePerspective()));
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
    REQUIRE(views.setCascade(2, markedMatrix(3.0f), someOrtho()));
    REQUIRE(views.setSelf(1, 77, markedMatrix(20.0f), someOrtho()));
    REQUIRE(views.setSpot(3, light, markedMatrix(30.0f), somePerspective()));
    REQUIRE(views.setPoint(1, 4, light, markedMatrix(40.0f), somePerspective()));

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

    CHECK_FALSE(views.setCascade(0, markedMatrix(1.0f), somePerspective()));
    CHECK_FALSE(views.setSelf(0, 5, markedMatrix(2.0f), somePerspective()));
    CHECK_FALSE(views.setSpot(0, light, markedMatrix(3.0f), someOrtho()));
    CHECK_FALSE(views.setPoint(0, 0, light, markedMatrix(4.0f), someOrtho()));

    CHECK(views.activeCount(ShadowViewGroup::Cascade) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Self) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Spot) == 0);
    CHECK(views.activeCount(ShadowViewGroup::Point) == 0);
#endif
}

TEST_CASE("ShadowRenderViewSet.WritersRejectAnUnkeyableIdentity", "[ShadowRenderView]")
{
    // An engaged entry must always be keyable: hysteresis keys on the identity, so an invalid one
    // would leave the view rendering but its history unreachable.
#ifdef NDEBUG
    ShadowRenderViewSet views;
    CHECK_FALSE(views.setSelf(0, 0, markedMatrix(1.0f), someOrtho())); // objectId 0
    CHECK_FALSE(views.setSpot(0, NodeId::Invalid, markedMatrix(2.0f), somePerspective()));
    CHECK_FALSE(views.setPoint(0, kCubeFaceCount, static_cast<NodeId>(3), markedMatrix(3.0f),
                               somePerspective())); // face out of range

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
    REQUIRE(views.setCascade(1, markedMatrix(7.0f), degenerate));

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

    CHECK_FALSE(
        views.setCascade(static_cast<std::uint32_t>(cascades), markedMatrix(1.0f), someOrtho()));
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
    REQUIRE(views.setSelf(0, 11, markedMatrix(1.0f), someOrtho()));
    REQUIRE(views.setSelf(2, 22, markedMatrix(2.0f), someOrtho()));

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
                                 someOrtho()));
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
    REQUIRE(views.setCascade(0, markedMatrix(1.0f), ShadowView::orthographic(0.25f)));
    REQUIRE(views.enableWorldOnly(0));
    REQUIRE(views.setCascade(0, markedMatrix(2.0f), ShadowView::orthographic(0.5f)));

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
    for (std::uint8_t face = 0; face < kCubeFaceCount; ++face)
    {
        REQUIRE(views.setPoint(lightSlot, face, light,
                               markedMatrix(100.0f + static_cast<float>(face)),
                               somePerspective(forwards[face])));
    }

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

    REQUIRE(views.setPoint(0, 1, real, markedMatrix(11.0f), somePerspective()));
    CHECK_FALSE(views.setPoint(wrapping, 1, impostor, markedMatrix(99.0f), somePerspective()));

    const ShadowRenderView* view = views.find(ShadowViewGroup::Point, shadowPointViewSlot(0, 1));
    REQUIRE(view != nullptr);
    CHECK(view->logicalId() == ShadowLogicalViewId::point(real, 1));
    CHECK(view->viewProj()[0, 3] == 11.0f);
    CHECK(views.activeCount(ShadowViewGroup::Point) == 1);
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
    REQUIRE(views.setCascade(0, markedMatrix(1.0f), ShadowView::orthographic(0.0f)));
    const ShadowRenderView* engaged = views.find(ShadowViewGroup::Cascade, 0);
    REQUIRE(engaged != nullptr);
    CHECK_FALSE(engaged->projection().valid());

#ifdef NDEBUG
    // Non-finite matrix + perfectly good descriptor: REJECTED. Dev asserts; this is the release
    // behaviour.
    CHECK_FALSE(views.setCascade(1, poisoned, someOrtho()));
    CHECK_FALSE(views.setSpot(0, static_cast<NodeId>(3), poisoned, somePerspective()));
    CHECK_FALSE(views.active(ShadowViewGroup::Cascade, 1));
    CHECK(views.activeCount(ShadowViewGroup::Spot) == 0);

    // An infinity is the same failure with a different bit pattern.
    Mat4 unbounded = Mat4::identity();
    unbounded[0, 0] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(views.setSelf(0, 5, unbounded, someOrtho()));
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

    REQUIRE(views.setCascade(0, markedMatrix(1.0f), someOrtho()));
    REQUIRE(views.active(ShadowViewGroup::Cascade, 0));
    CHECK_FALSE(views.setCascade(0, poisoned, someOrtho()));
    CHECK_FALSE(views.active(ShadowViewGroup::Cascade, 0));

    // The same for an optional family, and via the other content check (wrong projection kind).
    REQUIRE(views.setSpot(1, static_cast<NodeId>(4), markedMatrix(2.0f), somePerspective()));
    REQUIRE(views.active(ShadowViewGroup::Spot, 1));
    CHECK_FALSE(views.setSpot(1, static_cast<NodeId>(4), markedMatrix(3.0f), someOrtho()));
    CHECK_FALSE(views.active(ShadowViewGroup::Spot, 1));

    // World-only inherits it through the alias: a cleared cascade leaves nothing to point at, so
    // the pass reads as inactive rather than quietly retaining the previous fit.
    REQUIRE(views.setCascade(2, markedMatrix(4.0f), someOrtho()));
    REQUIRE(views.enableWorldOnly(2));
    CHECK_FALSE(views.setCascade(2, poisoned, someOrtho()));
    CHECK_FALSE(views.active(ShadowViewGroup::WorldOnly, 2));

    // But an OUT-OF-RANGE address names no slot, so it must leave every real one alone — including
    // the wrapping point slot, which is rejected before it can be flattened onto a live face. Both
    // families are seeded first, so each rejection has a live entry it could have damaged.
    REQUIRE(views.setCascade(1, markedMatrix(8.0f), someOrtho()));
    REQUIRE(views.setPoint(0, 0, static_cast<NodeId>(5), markedMatrix(5.0f), somePerspective()));
    constexpr std::size_t wrapping = std::size_t{1}
                                     << (std::numeric_limits<std::size_t>::digits - 1);
    CHECK_FALSE(
        views.setPoint(wrapping, 0, static_cast<NodeId>(6), markedMatrix(6.0f), somePerspective()));
    CHECK_FALSE(views.setCascade(kShadowCascadeCount, markedMatrix(7.0f), someOrtho()));

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
    REQUIRE(views.setSelf(0, 5, markedMatrix(11.0f), someOrtho()));

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
                                 someOrtho()));
    }
    return views;
}
} // namespace

TEST_CASE("ShadowRenderView.MatrixArrayPlacesEachFamilyAtItsOwnBase", "[ShadowRenderView]")
{
    ShadowRenderViewSet views = withAllCascades();
    const auto light = static_cast<NodeId>(4);
    REQUIRE(views.setSpot(2, light, markedMatrix(50.0f), somePerspective()));
    REQUIRE(views.setPoint(1, 3, light, markedMatrix(103.0f), somePerspective()));

    const auto matrices = shadowMatrixArray(views);

    CHECK(matrices[static_cast<std::size_t>(kShadowCascadeMatrixBase) + 1][0, 3] == 2.0f);
    CHECK(matrices[static_cast<std::size_t>(kShadowSpotMatrixBase) + 2][0, 3] == 50.0f);
    CHECK(matrices[static_cast<std::size_t>(kShadowPointMatrixBase) + shadowPointViewSlot(1, 3)]
                  [0, 3] == 103.0f);
    // Inactive slots are identity, not stale content from another family.
    CHECK(matrices[static_cast<std::size_t>(kShadowSpotMatrixBase)][0, 3] == 0.0f);
    CHECK(matrices[static_cast<std::size_t>(kShadowPointMatrixBase)][0, 3] == 0.0f);
}

TEST_CASE("ShadowRenderView.WorldOnlyWritesNoShaderSlot", "[ShadowRenderView]")
{
    // World-only rasterises with its cascade's matrix, so enabling it must change nothing in the
    // shader array.
    ShadowRenderViewSet withWorld = withAllCascades();
    for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        REQUIRE(withWorld.enableWorldOnly(cascade));
    }
    const auto withWorldOnly = shadowMatrixArray(withWorld);
    const auto withoutWorldOnly = shadowMatrixArray(withAllCascades());

    for (std::size_t i = 0; i < withWorldOnly.size(); ++i)
    {
        CAPTURE(i);
        CHECK(withWorldOnly[i][0, 3] == withoutWorldOnly[i][0, 3]);
    }
}

TEST_CASE("ShadowRenderView.LightUboArraysExtractTheirOwnFamily", "[ShadowRenderView]")
{
    ShadowRenderViewSet views = withAllCascades();
    REQUIRE(views.setSpot(1, static_cast<NodeId>(8), markedMatrix(60.0f), somePerspective()));
    REQUIRE(views.setSelf(2, 9, markedMatrix(70.0f), someOrtho()));

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
