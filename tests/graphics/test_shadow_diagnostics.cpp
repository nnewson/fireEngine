#include <fire_engine/graphics/shadow_diagnostics.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <set>

using namespace fire_engine;

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

TEST_CASE("point views flatten as lightSlot * 6 + face", "[ShadowDiagnostics]")
{
    CHECK(shadowPointViewSlot(0, 0) == 0);
    CHECK(shadowPointViewSlot(0, 5) == 5);
    CHECK(shadowPointViewSlot(1, 0) == 6);
    CHECK(shadowPointViewSlot(3, 5) == 23);
    // The last point face must still be inside the group's capacity.
    CHECK(shadowPointViewSlot(static_cast<std::size_t>(kMaxPointShadowCasters) - 1, 5) <
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

    // Three casters offered to cascade 0; the middle one is frustum-rejected.
    cascade0.beginRasterPass();
    cascade0.observe(100, true, 0, true);
    cascade0.observe(50, false, 1, true); // rejected: candidate only, no histogram entry
    cascade0.observe(20, true, 2, true);

    CHECK(cascade0.rasterPasses == 1);
    CHECK(cascade0.candidateDraws == 3);
    CHECK(cascade0.drawnDraws == 2);
    CHECK(cascade0.candidateTriangles == 170);
    CHECK(cascade0.drawnTriangles == 120); // the rejected 50 must NOT be billed as GPU cost
    CHECK(cascade0.lodHistogram[0] == 1);
    CHECK(cascade0.lodHistogram[1] == 0); // the rejected draw contributes no selection
    CHECK(cascade0.lodHistogram[2] == 1);
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

    self.beginRasterPass();
    self.observe(64, true, 1, true); // first layer: counts selection
    self.beginRasterPass();
    self.observe(64, true, 1, false); // second layer: cost only

    CHECK(self.rasterPasses == 2);
    CHECK(self.drawnDraws == 2);
    CHECK(self.drawnTriangles == 128);
    CHECK(self.lodHistogram[1] == 1);
}

TEST_CASE("a rasterised view with no candidates stays visible", "[ShadowDiagnostics]")
{
    // The case that motivated rasterPasses: a cascade or punctual map is rendered and cleared even
    // when the draw span offers it nothing. Keying "active" off candidates made exactly this view —
    // a real per-frame cost, and evidence that a map is being rendered for no reason — vanish from
    // the panel.
    ShadowFrameStats stats;
    ShadowViewStats& cascade2 = stats.view(ShadowViewGroup::Cascade, 2);
    cascade2.beginRasterPass();

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
    spot.beginRasterPass();
    for (std::uint32_t i = 0; i < 5; ++i)
    {
        spot.observe(10, i % 2 == 0, i, true);
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
        stats.view(ShadowViewGroup::Cascade, slot).beginRasterPass();
    }
    stats.view(ShadowViewGroup::Point, shadowPointViewSlot(1, 4)).beginRasterPass();

    stats.view(ShadowViewGroup::Cascade, 0).observe(10, true, 0, true);
    stats.view(ShadowViewGroup::Cascade, 3).observe(30, true, 3, true);
    stats.view(ShadowViewGroup::Point, shadowPointViewSlot(1, 4)).observe(7, true, 1, true);

    const ShadowViewStats cascades = stats.groupTotal(ShadowViewGroup::Cascade);
    CHECK(cascades.drawnDraws == 2);
    CHECK(cascades.drawnTriangles == 40);
    CHECK(cascades.lodHistogram[0] == 1);
    CHECK(cascades.lodHistogram[3] == 1);

    // A point-group write must not leak into the cascade rollup (the flattening's whole point).
    CHECK(stats.groupTotal(ShadowViewGroup::Point).drawnTriangles == 7);
    CHECK(stats.groupTotal(ShadowViewGroup::Spot).drawnDraws == 0);
    CHECK(stats.sceneTotal().drawnTriangles == 47);
}

TEST_CASE("activeViewCount reports rasterised slots", "[ShadowDiagnostics]")
{
    ShadowFrameStats stats;
    CHECK(stats.activeViewCount(ShadowViewGroup::Spot) == 0);

    stats.view(ShadowViewGroup::Spot, 0).beginRasterPass();
    stats.view(ShadowViewGroup::Spot, 2).beginRasterPass();
    CHECK(stats.activeViewCount(ShadowViewGroup::Spot) == 2);

    // A view whose every candidate was culled is still ACTIVE — it was rasterised (cleared), and
    // hiding it would hide "this map ran and drew nothing", which is the interesting case.
    stats.view(ShadowViewGroup::Spot, 3).beginRasterPass();
    stats.view(ShadowViewGroup::Spot, 3).observe(5, false, 0, true);
    CHECK(stats.activeViewCount(ShadowViewGroup::Spot) == 3);
    CHECK(stats.view(ShadowViewGroup::Spot, 3).drawnDraws == 0);
}

TEST_CASE("LOD reasons are recorded per decision, level 0 distinguishable from forced",
          "[ShadowDiagnostics]")
{
    ShadowFrameStats stats;
    stats.addLodReason(ShadowLodReason::Selected);    // chose level 0 within budget
    stats.addLodReason(ShadowLodReason::Selected);    // chose level 2
    stats.addLodReason(ShadowLodReason::SingleLevel); // cloth: nothing to select
    stats.addLodReason(ShadowLodReason::LodDisabled);

    CHECK(stats.lodReasons[static_cast<std::size_t>(ShadowLodReason::Selected)] == 2);
    CHECK(stats.lodReasons[static_cast<std::size_t>(ShadowLodReason::SingleLevel)] == 1);
    CHECK(stats.lodReasons[static_cast<std::size_t>(ShadowLodReason::LodDisabled)] == 1);
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
    stats.view(ShadowViewGroup::Cascade, 1).beginRasterPass();
    stats.view(ShadowViewGroup::Cascade, 1).observe(99, true, 2, true);
    stats.addLodReason(ShadowLodReason::Selected);

    stats.reset();

    CHECK(stats.sceneTotal().drawnTriangles == 0);
    CHECK(stats.sceneTotal().candidateDraws == 0);
    CHECK(stats.lodReasons[static_cast<std::size_t>(ShadowLodReason::Selected)] == 0);
    CHECK(stats.sceneTotal().rasterPasses == 0);
    CHECK(stats.activeViewCount(ShadowViewGroup::Cascade) == 0);
}
