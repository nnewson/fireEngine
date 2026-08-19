#include <fire_engine/graphics/shadow_diagnostics.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <set>
#include <string>

using namespace fire_engine;

namespace
{

// What the old single `beginRasterPass(identity)` did, now that engagement and raster-pass
// accounting are separate calls: claim the row for a logical view AND count one rasterised layer.
// Most cases below are about the identity rules, which are `claimView`'s; the ones that care about
// the split assert on `claimed()` / `touched()` / `rasterPasses` directly.
[[nodiscard]] bool engageRow(ShadowViewStats& row, ShadowLogicalViewId view) noexcept
{
    return row.claimView(view) && row.beginRasterPass(view);
}

} // namespace

TEST_CASE("shadow view slots are a dense, collision-free flattening", "[ShadowDiagnostics]")
{
    // Every (group, slot) must map to its own index, and the indices must exactly fill
    // kShadowViewCount — a collision would silently merge two views' counters, and a gap would
    // waste a row and hide a view.
    std::set<std::size_t> seen;
    for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
    {
        const auto group = static_cast<ShadowViewGroup>(g);
        for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
        {
            const std::size_t index = shadowViewIndex(group, slot);
            CHECK(index < kShadowViewCount);
            CHECK(seen.insert(index).second); // no collision
        }
    }
    CHECK(seen.size() == kShadowViewCount);
}

TEST_CASE("a cube has six faces", "[ShadowDiagnostics]")
{
    // Pinned ONCE, on its own, because it is a Vulkan topology fact rather than a tunable: a cube
    // map has six faces and always will. Every other expectation below is then expressed through
    // the constant, so changing the constant cannot leave a test agreeing with a stale literal.
    STATIC_REQUIRE(kCubeFaceCount == 6);
}

TEST_CASE("point views flatten as lightSlot * kCubeFaceCount + face", "[ShadowDiagnostics]")
{
    constexpr std::size_t lastFace = kCubeFaceCount - 1;

    CHECK(shadowPointViewSlot(0, 0) == 0);
    CHECK(shadowPointViewSlot(0, lastFace) == lastFace);
    // The first face of the next light starts exactly one full cube on.
    CHECK(shadowPointViewSlot(1, 0) == kCubeFaceCount);
    CHECK(shadowPointViewSlot(3, lastFace) == (std::size_t{3} * kCubeFaceCount) + lastFace);
    // The last point face must still be inside the group's capacity.
    CHECK(shadowPointViewSlot(static_cast<std::size_t>(kMaxPointShadowCasters) - 1, lastFace) <
          shadowViewSlotCount(ShadowViewGroup::Point));
}

TEST_CASE("LOD histogram bins saturate at 3+", "[ShadowDiagnostics]")
{
    CHECK(shadowLodBin(0) == 0);
    CHECK(shadowLodBin(1) == 1);
    CHECK(shadowLodBin(2) == 2);
    CHECK(shadowLodBin(3) == 3);
    CHECK(shadowLodBin(4) == 3);
    CHECK(shadowLodBin(99) == 3);
}

TEST_CASE("candidate and drawn are counted independently", "[ShadowDiagnostics]")
{
    ShadowFrameStats stats;
    ShadowViewStats& cascade0 = stats.view(ShadowViewGroup::Cascade, 0);

    // Three casters offered to cascade 0; the middle one is frustum-rejected. SH-03 split the two
    // triangle counts: the first is FULL DETAIL (what the view was offered, known without
    // resolving), the second is what this view's resolution actually draws.
    REQUIRE(engageRow(cascade0, ShadowLogicalViewId::cascade(0)));
    cascade0.observe(100, true, 40, 0, ShadowLodReason::Selected, true);
    // Rejected before resolution, so it has no resolved count, level or reason to contribute.
    cascade0.observe(50, false, 0, 0, ShadowLodReason::Count, true);
    cascade0.observe(20, true, 5, 2, ShadowLodReason::Selected, true);

    CHECK(cascade0.rasterPasses == 1);
    CHECK(cascade0.candidateDraws == 3);
    CHECK(cascade0.drawnDraws == 2);
    CHECK(cascade0.candidateTriangles == 170); // full detail, including the rejected caster
    CHECK(cascade0.drawnTriangles == 45);      // the resolved cost, not the offered one
    CHECK(cascade0.lodHistogram[0] == 1);
    CHECK(cascade0.lodHistogram[1] == 0); // the rejected draw contributes no selection
    CHECK(cascade0.lodHistogram[2] == 1);
    CHECK(cascade0.lodReasons[static_cast<std::size_t>(ShadowLodReason::Selected)] == 2);
    CHECK(cascade0.touched());
    CHECK_FALSE(stats.view(ShadowViewGroup::Cascade, 1).touched());
}

TEST_CASE("a twice-rasterised self-shadow view doubles cost but not selection",
          "[ShadowDiagnostics]")
{
    // The self-shadow families rasterise one logical view as two depth layers. Real GPU cost is
    // both; the LOD *distribution* is a property of the single logical view, so doubling it would
    // misreport it.
    ShadowFrameStats stats;
    ShadowViewStats& self = stats.view(ShadowViewGroup::Self, 2);

    REQUIRE(engageRow(self, ShadowLogicalViewId::self(9)));
    self.observe(64, true, 30, 1, ShadowLodReason::Selected, true); // first layer: counts selection
    REQUIRE(engageRow(self, ShadowLogicalViewId::self(9)));
    self.observe(64, true, 30, 1, ShadowLodReason::Selected, false); // second layer: cost only

    CHECK(self.rasterPasses == 2);
    CHECK(self.drawnDraws == 2);
    CHECK(self.drawnTriangles == 60);
    CHECK(self.lodHistogram[1] == 1);
    // The reason follows the histogram's rule for the same reason: one decision, rasterised twice.
    CHECK(self.lodReasons[static_cast<std::size_t>(ShadowLodReason::Selected)] == 1);
}

TEST_CASE("a rasterised view with no candidates stays visible", "[ShadowDiagnostics]")
{
    // The case that motivated rasterPasses: a cascade or punctual map is rendered and cleared even
    // when the draw span offers it nothing. Keying "active" off candidates made exactly this view —
    // a real per-frame cost, and evidence that a map is being rendered for no reason — vanish from
    // the panel.
    ShadowFrameStats stats;
    ShadowViewStats& cascade2 = stats.view(ShadowViewGroup::Cascade, 2);
    REQUIRE(engageRow(cascade2, ShadowLogicalViewId::cascade(2)));

    CHECK(cascade2.rasterPasses == 1);
    CHECK(cascade2.candidateDraws == 0);
    CHECK(cascade2.drawnDraws == 0);
    CHECK(cascade2.touched());
    CHECK(stats.activeViewCount(ShadowViewGroup::Cascade) == 1);
}

TEST_CASE("drawn can never exceed candidate", "[ShadowDiagnostics]")
{
    // Guaranteed by construction: one `observe` per walked draw carries the filter verdict, so
    // there is no API shape in which an accepted draw was never offered. `candidate − drawn` is a
    // promised metric, and this is what makes the promise structural rather than conventional.
    ShadowFrameStats stats;
    ShadowViewStats& spot = stats.view(ShadowViewGroup::Spot, 1);
    REQUIRE(engageRow(spot, ShadowLogicalViewId::spot(static_cast<NodeId>(4))));
    for (std::uint32_t i = 0; i < 5; ++i)
    {
        spot.observe(10, i % 2 == 0, 4, i, ShadowLodReason::Selected, true);
    }
    CHECK(spot.candidateDraws == 5);
    CHECK(spot.drawnDraws == 3);
    CHECK(spot.drawnDraws <= spot.candidateDraws);
    CHECK(spot.drawnTriangles <= spot.candidateTriangles);
}

TEST_CASE("group and scene rollups sum their slots", "[ShadowDiagnostics]")
{
    ShadowFrameStats stats;
    for (const std::size_t slot : {std::size_t{0}, std::size_t{3}})
    {
        REQUIRE(engageRow(stats.view(ShadowViewGroup::Cascade, slot),
                          ShadowLogicalViewId::cascade(static_cast<std::uint32_t>(slot))));
    }
    REQUIRE(engageRow(stats.view(ShadowViewGroup::Point, shadowPointViewSlot(1, 4)),
                      ShadowLogicalViewId::point(static_cast<NodeId>(6), 4)));

    stats.view(ShadowViewGroup::Cascade, 0)
        .observe(10, true, 10, 0, ShadowLodReason::Selected, true);
    stats.view(ShadowViewGroup::Cascade, 3)
        .observe(30, true, 30, 3, ShadowLodReason::SingleLevel, true);
    stats.view(ShadowViewGroup::Point, shadowPointViewSlot(1, 4))
        .observe(7, true, 7, 1, ShadowLodReason::Selected, true);

    const ShadowViewStats cascades = stats.groupTotal(ShadowViewGroup::Cascade);
    CHECK(cascades.drawnDraws == 2);
    CHECK(cascades.drawnTriangles == 40);
    CHECK(cascades.lodHistogram[0] == 1);
    CHECK(cascades.lodHistogram[3] == 1);

    // A point-group write must not leak into the cascade rollup (the flattening's whole point).
    CHECK(stats.groupTotal(ShadowViewGroup::Point).drawnTriangles == 7);
    CHECK(stats.groupTotal(ShadowViewGroup::Spot).drawnDraws == 0);
    CHECK(stats.sceneTotal().drawnTriangles == 47);
    // Reasons roll up the same way — they are per-view counters now (SH-03), so the scene total is
    // the only place a frame-wide reason distribution exists.
    CHECK(cascades.lodReasons[static_cast<std::size_t>(ShadowLodReason::Selected)] == 1);
    CHECK(cascades.lodReasons[static_cast<std::size_t>(ShadowLodReason::SingleLevel)] == 1);
    CHECK(stats.sceneTotal().lodReasons[static_cast<std::size_t>(ShadowLodReason::Selected)] == 2);
}

TEST_CASE("activeViewCount reports rasterised slots", "[ShadowDiagnostics]")
{
    ShadowFrameStats stats;
    CHECK(stats.activeViewCount(ShadowViewGroup::Spot) == 0);

    REQUIRE(engageRow(stats.view(ShadowViewGroup::Spot, 0),
                      ShadowLogicalViewId::spot(static_cast<NodeId>(1))));
    REQUIRE(engageRow(stats.view(ShadowViewGroup::Spot, 2),
                      ShadowLogicalViewId::spot(static_cast<NodeId>(2))));
    CHECK(stats.activeViewCount(ShadowViewGroup::Spot) == 2);

    // A view whose every candidate was culled is still ACTIVE — it was rasterised (cleared), and
    // hiding it would hide "this map ran and drew nothing", which is the interesting case.
    REQUIRE(engageRow(stats.view(ShadowViewGroup::Spot, 3),
                      ShadowLogicalViewId::spot(static_cast<NodeId>(3))));
    stats.view(ShadowViewGroup::Spot, 3).observe(5, false, 0, 0, ShadowLodReason::Count, true);
    CHECK(stats.activeViewCount(ShadowViewGroup::Spot) == 3);
    CHECK(stats.view(ShadowViewGroup::Spot, 3).drawnDraws == 0);
}

TEST_CASE("LOD reasons are recorded per view, level 0 distinguishable from forced",
          "[ShadowDiagnostics]")
{
    // SH-03: a reason belongs to the VIEW that resolved it. One caster drawn into a cascade and a
    // spot can be Selected in one and forced in the other — a single frame-wide tally could not
    // express that, and after per-view selection it would be the sum of unrelated decisions.
    ShadowFrameStats stats;
    ShadowViewStats& cascade = stats.view(ShadowViewGroup::Cascade, 0);
    ShadowViewStats& spot = stats.view(ShadowViewGroup::Spot, 0);
    REQUIRE(engageRow(cascade, ShadowLogicalViewId::cascade(0)));
    REQUIRE(engageRow(spot, ShadowLogicalViewId::spot(static_cast<NodeId>(5))));

    cascade.observe(10, true, 10, 0, ShadowLodReason::Selected, true);    // level 0, within budget
    cascade.observe(10, true, 2, 2, ShadowLodReason::Selected, true);     // level 2
    cascade.observe(10, true, 10, 0, ShadowLodReason::SingleLevel, true); // cloth
    spot.observe(10, true, 10, 0, ShadowLodReason::NearPlane, true); // forced: at the light plane

    CHECK(cascade.lodReasons[static_cast<std::size_t>(ShadowLodReason::Selected)] == 2);
    CHECK(cascade.lodReasons[static_cast<std::size_t>(ShadowLodReason::SingleLevel)] == 1);
    CHECK(cascade.lodReasons[static_cast<std::size_t>(ShadowLodReason::NearPlane)] == 0);
    CHECK(spot.lodReasons[static_cast<std::size_t>(ShadowLodReason::NearPlane)] == 1);
    // Both level-0 draws land in the same histogram bin, so the reason is the ONLY thing that
    // distinguishes a deliberate LOD0 from a forced one.
    CHECK(cascade.lodHistogram[0] == 2);
}

TEST_CASE("a focused view distinguishes 'ran and drew nothing' from 'never ran'",
          "[ShadowDiagnostics]")
{
    // The distinction the whole panel rests on (SH-03 slice 4). A view that rasterised and drew
    // nothing is a FINDING — a map cleared for no reason — and reports zeros. A view that never ran
    // has nothing measured at all, and reporting zeros for it would state that finding falsely.
    ShadowFrameStats stats;
    const auto lit = ShadowLogicalViewId::spot(static_cast<NodeId>(11));
    REQUIRE(engageRow(stats.view(ShadowViewGroup::Spot, 1), lit));

    const FocusedShadowView ran = stats.focused(
        ShadowViewFocus{.perView = true, .group = ShadowViewGroup::Spot, .view = lit});
    REQUIRE(ran.found());
    CHECK(ran.slot == 1);
    CHECK(ran.stats->rasterPasses == 1);
    CHECK(ran.stats->candidateDraws == 0);

    // A real identity that simply did not rasterise: addressable, but not found.
    const ShadowViewFocus absent{.perView = true,
                                 .group = ShadowViewGroup::Spot,
                                 .view = ShadowLogicalViewId::spot(static_cast<NodeId>(12))};
    CHECK(absent.addressable());
    CHECK_FALSE(stats.focused(absent).found());
}

TEST_CASE("one diagnostic row belongs to one logical view", "[ShadowDiagnostics]")
{
    // Slots are reused across frames, but WITHIN a frame a row is one view's counters. Two
    // identities landing on one slot would sum two views' draws, triangles and level distributions
    // and then label the total with whichever came second — a row that reads like a measurement of
    // something that never existed.
    ShadowFrameStats stats;
    ShadowViewStats& slot = stats.view(ShadowViewGroup::Spot, 0);
    const auto first = ShadowLogicalViewId::spot(static_cast<NodeId>(31));

    REQUIRE(engageRow(slot, first));
    slot.observe(10, true, 10, 0, ShadowLodReason::Selected, true);

    // The SAME identity again is the normal case — a self-shadow slot's two depth layers — and is
    // accepted in every build.
    CHECK(engageRow(slot, first));
    CHECK(slot.rasterPasses == 2);

#ifdef NDEBUG
    // Dev asserts at the contradiction (and the renderer's call site throws either way); this is
    // the release behaviour, which must leave the row exactly as it was.
    const auto second = ShadowLogicalViewId::spot(static_cast<NodeId>(32));
    CHECK_FALSE(engageRow(slot, second));
    CHECK(slot.rasterPasses == 2);
    CHECK(slot.logicalId == first);
    CHECK(slot.drawnDraws == 1);

    // An invalid identity is refused before it can even count the pass.
    CHECK_FALSE(engageRow(slot, ShadowLogicalViewId{}));
    CHECK(slot.rasterPasses == 2);
    CHECK(slot.logicalId == first);

    // On a fresh row an invalid identity leaves it untouched, rather than "rasterised but unnamed"
    // — a row nothing could ever select.
    ShadowViewStats& fresh = stats.view(ShadowViewGroup::Spot, 1);
    CHECK_FALSE(engageRow(fresh, ShadowLogicalViewId{}));
    CHECK_FALSE(fresh.touched());
#endif
}

TEST_CASE("a focus must pair its group with a compatible identity kind", "[ShadowDiagnostics]")
{
    // Independent checks miss this: both halves are well-formed, but a cascade identity can never
    // appear in the Spot group, so the focus names nothing. Left unchecked it looks addressable and
    // is then never found — reading as "this view keeps not rendering" instead of "this selection
    // is malformed", which sends a reader hunting a rendering bug that does not exist.
    STATIC_REQUIRE(shadowViewKindFor(ShadowViewGroup::Cascade) == ShadowLogicalViewKind::Cascade);
    // World-only shares the cascade's identity deliberately — the two passes are one logical view.
    STATIC_REQUIRE(shadowViewKindFor(ShadowViewGroup::WorldOnly) == ShadowLogicalViewKind::Cascade);
    STATIC_REQUIRE(shadowViewKindFor(ShadowViewGroup::Self) == ShadowLogicalViewKind::Self);
    STATIC_REQUIRE(shadowViewKindFor(ShadowViewGroup::Spot) == ShadowLogicalViewKind::Spot);
    STATIC_REQUIRE(shadowViewKindFor(ShadowViewGroup::Point) == ShadowLogicalViewKind::Point);
    STATIC_REQUIRE(shadowViewKindFor(ShadowViewGroup::Count) == ShadowLogicalViewKind::Invalid);

    ShadowFrameStats stats;
    REQUIRE(engageRow(stats.view(ShadowViewGroup::Spot, 0),
                      ShadowLogicalViewId::spot(static_cast<NodeId>(41))));

    const ShadowViewFocus mismatched{
        .perView = true, .group = ShadowViewGroup::Spot, .view = ShadowLogicalViewId::cascade(0)};
    CHECK_FALSE(mismatched.addressable());
    CHECK_FALSE(stats.focused(mismatched).found());

    // A spot identity in the Point group is equally impossible, even though both are punctual.
    const ShadowViewFocus wrongPunctual{.perView = true,
                                        .group = ShadowViewGroup::Point,
                                        .view = ShadowLogicalViewId::spot(static_cast<NodeId>(41))};
    CHECK_FALSE(wrongPunctual.addressable());

    const ShadowViewFocus matched{.perView = true,
                                  .group = ShadowViewGroup::Spot,
                                  .view = ShadowLogicalViewId::spot(static_cast<NodeId>(41))};
    CHECK(matched.addressable());
    CHECK(stats.focused(matched).found());
}

TEST_CASE("an unaddressable focus is a different state from an inactive view",
          "[ShadowDiagnostics]")
{
    // The panel says these differently, so they must BE different: "selection is not a valid view"
    // (structurally malformed — no frame can satisfy it, so re-select) versus a well-formed focus
    // simply "not present in this frame" (which says nothing about whether it returns).
    ShadowFrameStats stats;
    REQUIRE(engageRow(stats.view(ShadowViewGroup::Cascade, 0), ShadowLogicalViewId::cascade(0)));

    // The scene rollup names no view at all — and is not addressable, so the panel takes its own
    // branch rather than being handed one view's numbers.
    CHECK_FALSE(ShadowViewFocus{}.addressable());
    CHECK_FALSE(stats.focused(ShadowViewFocus{}).found());

    // perView with a default (invalid) identity: the state a focus lands in if something forgets to
    // fill it. Not addressable — it names nothing, rather than naming view 0.
    const ShadowViewFocus unset{.perView = true, .group = ShadowViewGroup::Cascade, .view = {}};
    CHECK_FALSE(unset.addressable());
    CHECK_FALSE(stats.focused(unset).found());

    const ShadowViewFocus noGroup{
        .perView = true, .group = ShadowViewGroup::Count, .view = ShadowLogicalViewId::cascade(0)};
    CHECK_FALSE(noGroup.addressable());
    CHECK_FALSE(stats.focused(noGroup).found());
}

TEST_CASE("focusing follows the view, not the slot's occupant", "[ShadowDiagnostics]")
{
    // THE reason the focus is keyed by identity. Punctual and self slots are compacted in
    // scene-gather order, so when a light leaves, the lights after it move DOWN a slot. A
    // slot-keyed focus would silently start reporting the replacement's numbers under the original
    // selection — and once the tint reads the same focus, the panel (a completed ring frame) and
    // the tint (the current frame) could disagree about which view they mean.
    const auto first = ShadowLogicalViewId::spot(static_cast<NodeId>(21));
    const auto second = ShadowLogicalViewId::spot(static_cast<NodeId>(22));

    ShadowFrameStats before;
    REQUIRE(engageRow(before.view(ShadowViewGroup::Spot, 0), first));
    REQUIRE(engageRow(before.view(ShadowViewGroup::Spot, 1), second));
    before.view(ShadowViewGroup::Spot, 1).observe(30, true, 12, 1, ShadowLodReason::Selected, true);

    const ShadowViewFocus focus{.perView = true, .group = ShadowViewGroup::Spot, .view = second};
    const FocusedShadowView atFirst = before.focused(focus);
    REQUIRE(atFirst.found());
    CHECK(atFirst.slot == 1);
    CHECK(atFirst.stats->drawnTriangles == 12);

    // Next frame the first light is gone, so `second` compacts down into slot 0 and draws
    // something different. The focus must follow the LIGHT.
    ShadowFrameStats after;
    REQUIRE(engageRow(after.view(ShadowViewGroup::Spot, 0), second));
    after.view(ShadowViewGroup::Spot, 0).observe(30, true, 7, 2, ShadowLodReason::Selected, true);

    const FocusedShadowView moved = after.focused(focus);
    REQUIRE(moved.found());
    CHECK(moved.slot == 0);                  // a different slot ...
    CHECK(moved.stats->drawnTriangles == 7); // ... and that light's own numbers

    // And the departed light is simply not found — never silently answered by its replacement.
    CHECK_FALSE(after
                    .focused(ShadowViewFocus{
                        .perView = true, .group = ShadowViewGroup::Spot, .view = first})
                    .found());
}

TEST_CASE("a cascade and its world-only twin share an identity but not a row",
          "[ShadowDiagnostics]")
{
    // worldOnly(i) IS cascade(i) — deliberately, so they share one LOD decision. That is exactly
    // why the focus carries the GROUP as well: the two are different maps with different counters,
    // and an identity alone could not tell them apart.
    ShadowFrameStats stats;
    const auto shared = ShadowLogicalViewId::cascade(2);
    REQUIRE(engageRow(stats.view(ShadowViewGroup::Cascade, 2), shared));
    stats.view(ShadowViewGroup::Cascade, 2)
        .observe(50, true, 50, 0, ShadowLodReason::Selected, true);
    REQUIRE(
        engageRow(stats.view(ShadowViewGroup::WorldOnly, 2), ShadowLogicalViewId::worldOnly(2)));
    stats.view(ShadowViewGroup::WorldOnly, 2)
        .observe(20, true, 20, 0, ShadowLodReason::Selected, true);

    const FocusedShadowView full = stats.focused(
        ShadowViewFocus{.perView = true, .group = ShadowViewGroup::Cascade, .view = shared});
    const FocusedShadowView worldOnly = stats.focused(
        ShadowViewFocus{.perView = true, .group = ShadowViewGroup::WorldOnly, .view = shared});
    REQUIRE(full.found());
    REQUIRE(worldOnly.found());
    CHECK(full.stats->drawnTriangles == 50);
    CHECK(worldOnly.stats->drawnTriangles == 20);
    CHECK(full.stats != worldOnly.stats);
}

TEST_CASE("--shadow-focus parses a group and slot, or nothing", "[ShadowDiagnostics]")
{
    // A slot is the only handle a person has before the engine runs, so the command line speaks in
    // slots — but every rejection here matters more than the acceptances: a request that silently
    // became something else would produce a capture of the wrong view that looks perfectly correct.
    const auto request = parseShadowViewSlotRequest("cascade:2");
    REQUIRE(request.has_value());
    CHECK(request->group == ShadowViewGroup::Cascade);
    CHECK(request->slot == 2);

    CHECK(parseShadowViewSlotRequest("world-only:1")->group == ShadowViewGroup::WorldOnly);
    CHECK(parseShadowViewSlotRequest("worldonly:1")->group == ShadowViewGroup::WorldOnly);
    CHECK(parseShadowViewSlotRequest("self:0")->group == ShadowViewGroup::Self);
    CHECK(parseShadowViewSlotRequest("spot:1")->slot == 1);

    // Point takes light + face, because a flat point slot is an implementation detail no one should
    // have to compute — and it must flatten to exactly the slot the renderer rasterises.
    const auto face = parseShadowViewSlotRequest("point:1:4");
    REQUIRE(face.has_value());
    CHECK(face->group == ShadowViewGroup::Point);
    CHECK(face->slot == shadowPointViewSlot(1, 4));

    // Rejections, each with a different way of being wrong.
    CHECK_FALSE(parseShadowViewSlotRequest("cascade").has_value());    // no slot
    CHECK_FALSE(parseShadowViewSlotRequest("bogus:0").has_value());    // unknown family
    CHECK_FALSE(parseShadowViewSlotRequest("cascade:").has_value());   // empty slot
    CHECK_FALSE(parseShadowViewSlotRequest("cascade:-1").has_value()); // not a whole number
    CHECK_FALSE(parseShadowViewSlotRequest("cascade:2x").has_value()); // a numeric PREFIX only
    CHECK_FALSE(parseShadowViewSlotRequest("spot:0:1").has_value());   // only point has faces
    CHECK_FALSE(parseShadowViewSlotRequest("point:0:6").has_value());  // face out of range
    CHECK_FALSE(parseShadowViewSlotRequest("").has_value());
    // Beyond the group's capacity: rejected here rather than at lookup, so the failure names the
    // request rather than reporting an inactive view.
    CHECK_FALSE(
        parseShadowViewSlotRequest("cascade:" + std::to_string(kShadowCascadeCount)).has_value());

    // The adversarial one: `light * kCubeFaceCount + face` is unsigned, so a large enough light
    // index WRAPS back into range — with an even face count, the top bit times 6 is 0 modulo the
    // word — and a nonsense light would silently focus a real view. Same defect and same fix as
    // ShadowRenderViewSet::setPointLight: the light is validated BEFORE it is flattened.
    constexpr std::size_t wrapping = std::size_t{1}
                                     << (std::numeric_limits<std::size_t>::digits - 1);
    STATIC_REQUIRE(kCubeFaceCount % 2 == 0);
    REQUIRE(shadowPointViewSlot(wrapping, 1) == 1); // it really does land on light 0, face 1
    CHECK_FALSE(parseShadowViewSlotRequest("point:" + std::to_string(wrapping) + ":1").has_value());
    // An ordinary over-capacity light is rejected the same way.
    CHECK_FALSE(
        parseShadowViewSlotRequest(
            "point:" + std::to_string(static_cast<std::size_t>(kMaxPointShadowCasters)) + ":0")
            .has_value());
}

TEST_CASE("a calibration override is validated, not coerced", "[ShadowDiagnostics]")
{
    // The numeric half of --shadow-budget / --shadow-ratio. Every rejection matters more than the
    // acceptances: the renderer turns "nothing" into a refusal to start, and the alternative —
    // falling back to the constant — would produce a sweep row that reads exactly like a
    // measurement of the value that was asked for.
    constexpr float kNoMax = std::numeric_limits<float>::infinity();
    REQUIRE(parseShadowCalibrationValue("4", kNoMax).value() == 4.0f);
    REQUIRE(parseShadowCalibrationValue("0.5", 1.0f).value() == 0.5f);
    REQUIRE(parseShadowCalibrationValue("1", 1.0f).value() == 1.0f); // inclusive maximum

    CHECK_FALSE(parseShadowCalibrationValue("", kNoMax).has_value());
    CHECK_FALSE(parseShadowCalibrationValue("0", kNoMax).has_value());   // non-positive
    CHECK_FALSE(parseShadowCalibrationValue("-2", kNoMax).has_value());  // negative
    CHECK_FALSE(parseShadowCalibrationValue("abc", kNoMax).has_value()); // not a number
    CHECK_FALSE(parseShadowCalibrationValue("4x", kNoMax).has_value());  // a numeric PREFIX only
    CHECK_FALSE(parseShadowCalibrationValue(" 4", kNoMax).has_value());  // leading space
    CHECK_FALSE(parseShadowCalibrationValue("nan", kNoMax).has_value()); // non-finite
    CHECK_FALSE(parseShadowCalibrationValue("inf", kNoMax).has_value()); // non-finite
    // Above the maximum: a ratio outside (0, 1] makes the selector report InvalidCaster for every
    // caster, so a whole sweep would silently measure forced LOD0 while still drawing a picture.
    CHECK_FALSE(parseShadowCalibrationValue("1.5", 1.0f).has_value());
    // ... but the same text is fine where there is no ceiling.
    CHECK(parseShadowCalibrationValue("1.5", kNoMax).has_value());
}

TEST_CASE("every reason and group has a name", "[ShadowDiagnostics]")
{
    for (std::size_t r = 0; r < kShadowLodReasonCount; ++r)
    {
        CHECK(toString(static_cast<ShadowLodReason>(r)) != "unknown");
    }
    for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
    {
        CHECK(toString(static_cast<ShadowViewGroup>(g)) != "unknown");
    }
}

TEST_CASE("reset clears every counter", "[ShadowDiagnostics]")
{
    ShadowFrameStats stats;
    REQUIRE(engageRow(stats.view(ShadowViewGroup::Cascade, 1), ShadowLogicalViewId::cascade(1)));
    stats.view(ShadowViewGroup::Cascade, 1)
        .observe(99, true, 40, 2, ShadowLodReason::Selected, true);

    stats.reset();

    CHECK(stats.sceneTotal().drawnTriangles == 0);
    CHECK(stats.sceneTotal().candidateDraws == 0);
    CHECK(stats.sceneTotal().candidateTriangles == 0);
    CHECK(stats.sceneTotal().lodReasons[static_cast<std::size_t>(ShadowLodReason::Selected)] == 0);
    CHECK(stats.sceneTotal().rasterPasses == 0);
    CHECK(stats.activeViewCount(ShadowViewGroup::Cascade) == 0);
}

TEST_CASE("claiming a view and rasterising a layer are separate facts", "[ShadowDiagnostics]")
{
    // The split the shadow cache needs. A view whose map is REUSED is claimed and observed while
    // rasterising nothing, so a row forced to count a raster pass in order to be observable would
    // report intended work as performed work — and reuse would be unobservable.
    ShadowFrameStats stats;
    ShadowViewStats& cascade = stats.view(ShadowViewGroup::Cascade, 1);

    CHECK_FALSE(cascade.claimed());
    CHECK_FALSE(cascade.touched());

    const ShadowLogicalViewId view = ShadowLogicalViewId::cascade(1);
    REQUIRE(cascade.claimView(view));
    CHECK(cascade.claimed());
    // Claimed, no raster pass recorded. That is all this state says on its own — it also describes
    // a view that will be recorded later this frame, and (once caching lands) a recorder bug that
    // omitted its work. Proving REUSE needs the plan's disposition, which the cache will record.
    CHECK_FALSE(cascade.touched());
    CHECK(cascade.rasterPasses == 0);

    // Observation belongs to the CLAIM, not to a raster pass: the counters describe what the map
    // holds, which is knowable before (and without) any recording.
    cascade.observe(100, true, 40, 0, ShadowLodReason::Selected, true);
    CHECK(cascade.candidateDraws == 1);
    CHECK(cascade.drawnDraws == 1);
    CHECK(cascade.rasterPasses == 0);

    // And the recorder's call is what turns it into GPU work.
    REQUIRE(cascade.beginRasterPass(view));
    CHECK(cascade.touched());
    CHECK(cascade.rasterPasses == 1);
    REQUIRE(cascade.beginRasterPass(view)); // a second layer of the same view
    CHECK(cascade.rasterPasses == 2);
}

TEST_CASE("a raster pass cannot be counted for an unclaimed or mismatched row",
          "[ShadowDiagnostics]")
{
#ifdef NDEBUG
    // Two refusals. GPU work attributed to NO view carries a cost with nothing to name it; work
    // attributed to the WRONG view is worse, because the row stays plausible under the identity
    // that claimed it. Dev builds assert at the source; this is the release contract.
    ShadowFrameStats stats;
    ShadowViewStats& unclaimed = stats.view(ShadowViewGroup::Spot, 0);
    CHECK_FALSE(unclaimed.beginRasterPass(ShadowLogicalViewId::spot(static_cast<NodeId>(1))));
    CHECK(unclaimed.rasterPasses == 0);
    CHECK_FALSE(unclaimed.claimed());

    // A claimed, B rasterises: refused, and the pass count stays at zero.
    ShadowViewStats& row = stats.view(ShadowViewGroup::Spot, 1);
    const ShadowLogicalViewId a = ShadowLogicalViewId::spot(static_cast<NodeId>(7));
    const ShadowLogicalViewId b = ShadowLogicalViewId::spot(static_cast<NodeId>(8));
    REQUIRE(row.claimView(a));
    CHECK_FALSE(row.beginRasterPass(b));
    CHECK(row.rasterPasses == 0);
    // The claim is untouched — a rejected raster pass must not re-point the row at B.
    CHECK(row.logicalId == a);
    // And A can still record.
    CHECK(row.beginRasterPass(a));
    CHECK(row.rasterPasses == 1);
#else
    SUCCEED(
        "Dev builds assert inside beginRasterPass; the release contract is tested under NDEBUG");
#endif
}

TEST_CASE("a claimed view can be focused even with no raster pass", "[ShadowDiagnostics]")
{
    // PRESENCE IS THE CLAIM, not the work. A view whose map is reused records nothing, and keying
    // focus off rasterisation would make the panel's selection — and the ShadowLod tint that
    // follows it — vanish the moment a view became free. That is the case the cache exists to
    // produce, so it must be the case focus handles.
    ShadowFrameStats stats;
    const ShadowLogicalViewId spot = ShadowLogicalViewId::spot(static_cast<NodeId>(11));
    ShadowViewStats& row = stats.view(ShadowViewGroup::Spot, 2);
    REQUIRE(row.claimView(spot));
    row.observe(90, true, 30, 1, ShadowLodReason::Selected, true);
    REQUIRE_FALSE(row.touched()); // claimed, no raster pass recorded

    const FocusedShadowView found = stats.focused(
        ShadowViewFocus{.perView = true, .group = ShadowViewGroup::Spot, .view = spot});
    REQUIRE(found.stats != nullptr);
    CHECK(found.slot == 2);
    CHECK(found.stats->candidateDraws == 1);
    CHECK(found.stats->rasterPasses == 0);

    // An UNCLAIMED row is still absent: its identity is whatever the slot last held, possibly
    // frames ago, so matching against it would resurrect a stale view.
    const FocusedShadowView stale =
        stats.focused(ShadowViewFocus{.perView = true,
                                      .group = ShadowViewGroup::Spot,
                                      .view = ShadowLogicalViewId::spot(static_cast<NodeId>(12))});
    CHECK(stale.stats == nullptr);
}
