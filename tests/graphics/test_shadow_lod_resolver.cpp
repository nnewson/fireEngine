#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/shadow_lod_resolver.hpp>

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

// ---------------------------------------------------------------------------
// SH-03 slice 3: resolving one unresolved caster per shadow view.
//
// The defect being closed: a caster used to arrive with one index buffer already chosen from the
// camera, and every shadow view rasterised that. These cases pin the two stores that replace it —
// a per-frame cache keyed on the LOGICAL view (so passes that must agree cannot disagree) and a
// staged history that only a justified decision may write to.
// ---------------------------------------------------------------------------

constexpr float kBudget = 4.0f;
constexpr ShadowLodHysteresis kNoHysteresis{.coarsenRatio = 1.0f};

BufferHandle buffer(std::uint32_t index)
{
    return makeHandle<BufferHandle>(index, 0);
}

// A three-level chain whose deviations are far apart, so a level change is unambiguous rather than
// a boundary case: at the texel sizes below, level 1 and level 2 sit either side of the budget
// depending only on the view.
std::vector<GeometryLod> chain()
{
    return {GeometryLod{.indexBuffer = buffer(10), .indexCount = 900, .shadowDeviation = 0.0f},
            GeometryLod{.indexBuffer = buffer(11), .indexCount = 450, .shadowDeviation = 0.02f},
            GeometryLod{.indexBuffer = buffer(12), .indexCount = 120, .shadowDeviation = 0.20f}};
}

Bounds3 someBounds()
{
    Bounds3 b{};
    b.valid = true;
    b.min = Vec3{-1.0f, -1.0f, -1.0f};
    b.max = Vec3{1.0f, 1.0f, 1.0f};
    return b;
}

ShadowGeometryRequest request(const std::vector<GeometryLod>& lods, ShadowCasterId caster,
                              ShadowCasterGeneration generation = ShadowCasterGeneration::First)
{
    return ShadowGeometryRequest{
        .lods = lods,
        .baseIndexBuffer = buffer(10),
        .baseIndexCount = 900,
        .worldScale = 1.0f,
        .casterId = caster,
        .generation = generation,
        .lodEnabled = true,
        // Explicit: these fixtures are rigid casters. The field defaults to
        // Deformable (the safe answer), so stating it here is what keeps
        // every selection case below testing SELECTION rather than SH-04's
        // fallback. A deformable request is built by deformableRequest().
        .deformation = ShadowCasterDeformation::Rigid,
        // Explicit for the same reason (SH-05): `alpha` defaults to Masked, which pins a caster to
        // full detail, so a fixture that left it out would test the cutout fallback everywhere
        // instead of selection. A masked request is built by maskedRequest().
        .alpha = ShadowCasterAlpha::Opaque};
}

// A set with every cascade populated (and world-only enabled), one spot and one point light, so a
// test can ask any family for a real view.
ShadowRenderViewSet populatedViews()
{
    ShadowRenderViewSet views;
    for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        // A coarse texel: 0.05 world units per texel puts level 2's 0.20 deviation at 4 texels.
        REQUIRE(views.setCascade(cascade, Mat4::identity(), ShadowView::orthographic(0.05f),
                                 someOrthoMetrics()));
        REQUIRE(views.enableWorldOnly(cascade));
    }
    REQUIRE(
        views.setSelf(0, 7, Mat4::identity(), ShadowView::orthographic(0.05f), someOrthoMetrics()));
    const auto light = static_cast<NodeId>(21);
    REQUIRE(views.setSpot(
        0, light, Mat4::identity(),
        ShadowView::perspective(Vec3{0.0f, 0.0f, 20.0f}, Vec3{0.0f, 0.0f, -1.0f}, 1.0f, 1024, 0.1f),
        someSpotMetrics()));
    const std::array<Vec3, kCubeFaceCount> forwards{Vec3{1, 0, 0},  Vec3{-1, 0, 0}, Vec3{0, 1, 0},
                                                    Vec3{0, -1, 0}, Vec3{0, 0, 1},  Vec3{0, 0, -1}};
    const auto pointFace = [&](std::uint8_t face)
    {
        return ShadowPointFace{
            Mat4::identity(),
            ShadowView::perspective(Vec3{0.0f, 0.0f, 30.0f}, forwards[face], 1.5708f, 512, 0.1f)};
    };
    // One atomic cube: the six faces differ only by forward, and the light's identity and metrics
    // are passed once.
    const std::array<ShadowPointFace, kCubeFaceCount> cube{
        pointFace(0), pointFace(1), pointFace(2), pointFace(3), pointFace(4), pointFace(5)};
    REQUIRE(views.setPointLight(0, light, somePointMetrics(),
                                std::span<const ShadowPointFace, kCubeFaceCount>{cube}));
    return views;
}

const ShadowRenderView& view(const ShadowRenderViewSet& views, ShadowViewGroup group,
                             std::size_t slot)
{
    const ShadowRenderView* found = views.find(group, slot);
    REQUIRE(found != nullptr);
    return *found;
}

} // namespace

TEST_CASE("ShadowLodResolver.ResolutionBindsTheLevelItReports", "[ShadowLodResolver]")
{
    // Level, buffers and reason come out of one assignment, so a diagnostic can never describe a
    // different mesh than the one bound.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const ResolvedShadowDraw drawn = resolver.resolve(request(lods, static_cast<ShadowCasterId>(1)),
                                                      view(views, ShadowViewGroup::Cascade, 0),
                                                      someBounds(), kBudget, kNoHysteresis);
    REQUIRE(drawn.reason == ShadowLodReason::Selected);
    REQUIRE(drawn.level < lods.size());
    CHECK(drawn.indexBuffer == lods[drawn.level].indexBuffer);
    CHECK(drawn.indexCount == lods[drawn.level].indexCount);
    CHECK(drawn.drawable());
}

TEST_CASE("ShadowLodResolver.CascadeAndWorldOnlyShareOneResolution", "[ShadowLodResolver]")
{
    // The two CSMs must make the same choice for a rigid caster. They do so because they are ONE
    // logical view: the second lookup is a cache hit, not a second selection that happens to match.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const auto req = request(lods, static_cast<ShadowCasterId>(2));
    const ResolvedShadowDraw full = resolver.resolve(req, view(views, ShadowViewGroup::Cascade, 1),
                                                     someBounds(), kBudget, kNoHysteresis);
    CHECK(resolver.frameCacheSize() == 1);
    const ResolvedShadowDraw worldOnly = resolver.resolve(
        req, view(views, ShadowViewGroup::WorldOnly, 1), someBounds(), kBudget, kNoHysteresis);

    CHECK(resolver.frameCacheSize() == 1); // no second entry: same logical view
    CHECK(worldOnly.level == full.level);
    CHECK(worldOnly.indexBuffer == full.indexBuffer);
    CHECK(worldOnly.indexCount == full.indexCount);
}

TEST_CASE("ShadowLodResolver.SelfShadowSecondPassReusesTheFirstResolution", "[ShadowLodResolver]")
{
    // A self-shadow slot rasterises the same view twice. Resolving twice would risk two levels for
    // one map's two depth layers — and would double-count the decision in diagnostics.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const auto req = request(lods, static_cast<ShadowCasterId>(3));
    const ResolvedShadowDraw first = resolver.resolve(req, view(views, ShadowViewGroup::Self, 0),
                                                      someBounds(), kBudget, kNoHysteresis);
    const ResolvedShadowDraw second = resolver.resolve(req, view(views, ShadowViewGroup::Self, 0),
                                                       someBounds(), kBudget, kNoHysteresis);

    CHECK(resolver.frameCacheSize() == 1);
    CHECK(second.level == first.level);
    CHECK(second.indexBuffer == first.indexBuffer);
}

TEST_CASE("ShadowLodResolver.PointFacesResolveIndependently", "[ShadowLodResolver]")
{
    // Each cube face is its own view with its own forward, so each gets its own resolution — and
    // its own history slot. Collapsing them would apply one face's dead band to the other five.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const auto req = request(lods, static_cast<ShadowCasterId>(4));
    for (std::uint8_t face = 0; face < kCubeFaceCount; ++face)
    {
        const ResolvedShadowDraw drawn =
            resolver.resolve(req, view(views, ShadowViewGroup::Point, shadowPointViewSlot(0, face)),
                             someBounds(), kBudget, kNoHysteresis);
        CHECK(drawn.drawable());
    }
    CHECK(resolver.frameCacheSize() == kCubeFaceCount);
}

TEST_CASE("ShadowLodResolver.ACasterNeverResolvedGetsNoHistory", "[ShadowLodResolver]")
{
    // The filtering order this depends on lives in the shadow pass: a draw the view's frustum
    // rejects is never resolved, so it acquires no dead band against a view it does not appear in.
    // Here that is the whole statement — resolve is the only thing that can create history.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;

    resolver.beginFrame();
    const auto seen = request(lods, static_cast<ShadowCasterId>(5));
    const auto unseen = request(lods, static_cast<ShadowCasterId>(6));
    (void)resolver.resolve(seen, view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget,
                           kNoHysteresis);
    resolver.commitFrame();

    CHECK(resolver.historySize() == 1);
    CHECK(resolver.historyLevel(ShadowLodStateKey{unseen.casterId, unseen.generation,
                                                  ShadowLogicalViewId::cascade(0)}) ==
          kNoPreviousShadowLod);
}

TEST_CASE("ShadowLodResolver.HistoryIsStagedUntilTheFrameCommits", "[ShadowLodResolver]")
{
    // A level decided for a frame that was then abandoned describes geometry the GPU never saw.
    // Committing it would hand the next frame a dead band it never earned.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    const auto req = request(lods, static_cast<ShadowCasterId>(7));
    const ShadowLodStateKey key{req.casterId, req.generation, ShadowLogicalViewId::cascade(0)};

    resolver.beginFrame();
    (void)resolver.resolve(req, view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget,
                           kNoHysteresis);
    CHECK(resolver.historyLevel(key) == kNoPreviousShadowLod); // staged, not yet true
    resolver.discardFrame();
    resolver.beginFrame();
    CHECK(resolver.historyLevel(key) == kNoPreviousShadowLod);

    const ResolvedShadowDraw drawn = resolver.resolve(req, view(views, ShadowViewGroup::Cascade, 0),
                                                      someBounds(), kBudget, kNoHysteresis);
    resolver.commitFrame();
    CHECK(resolver.historyLevel(key) == drawn.level);

    // And beginFrame must not undo a committed level.
    resolver.beginFrame();
    CHECK(resolver.historyLevel(key) == drawn.level);
}

TEST_CASE("ShadowLodResolver.AGenerationChangeMissesTheOldHistory", "[ShadowLodResolver]")
{
    // The chain the dead band was measured against is gone (a reloaded or replaced shadow
    // geometry). The stored level must simply not be found — applying it to different geometry is
    // exactly the aliasing the generation exists to prevent.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    const auto caster = static_cast<ShadowCasterId>(8);

    resolver.beginFrame();
    (void)resolver.resolve(request(lods, caster, ShadowCasterGeneration::First),
                           view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget,
                           kNoHysteresis);
    resolver.commitFrame();

    const auto next = static_cast<ShadowCasterGeneration>(
        static_cast<std::uint32_t>(ShadowCasterGeneration::First) + 1);
    CHECK(resolver.historyLevel(ShadowLodStateKey{caster, ShadowCasterGeneration::First,
                                                  ShadowLogicalViewId::cascade(0)}) !=
          kNoPreviousShadowLod);
    CHECK(resolver.historyLevel(ShadowLodStateKey{caster, next, ShadowLogicalViewId::cascade(0)}) ==
          kNoPreviousShadowLod);
}

TEST_CASE("ShadowLodResolver.AnInvalidKeyNeverEntersEitherStore", "[ShadowLodResolver]")
{
    // A caster with no identity still has to draw — leaving a hole in a shadow map would be worse
    // than the producer bug — but it must not be cached (it would collide with every other
    // unkeyable caster) or remembered (it could never be looked up again). Dev asserts; this is
    // the release behaviour.
#ifdef NDEBUG
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    auto req = request(lods, ShadowCasterId::Invalid);
    const ResolvedShadowDraw drawn = resolver.resolve(req, view(views, ShadowViewGroup::Cascade, 0),
                                                      someBounds(), kBudget, kNoHysteresis);
    resolver.commitFrame();

    CHECK(drawn.drawable()); // it still rasterises
    CHECK(resolver.frameCacheSize() == 0);
    CHECK(resolver.historySize() == 0);
#endif
}

TEST_CASE("ShadowLodResolver.OnlyASelectedLevelReachesTheHistory", "[ShadowLodResolver]")
{
    // A forced fallback, a disabled LOD system and a single-level mesh all say nothing about where
    // the caster sits relative to its budget. Letting one overwrite a justified level would
    // discard the very evidence the dead band is built on.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    const auto caster = static_cast<ShadowCasterId>(9);
    const ShadowLodStateKey key{caster, ShadowCasterGeneration::First,
                                ShadowLogicalViewId::cascade(0)};

    resolver.beginFrame();
    const ResolvedShadowDraw selected =
        resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 0),
                         someBounds(), kBudget, kNoHysteresis);
    resolver.commitFrame();
    REQUIRE(selected.reason == ShadowLodReason::Selected);
    REQUIRE(resolver.historyLevel(key) == selected.level);

    // LOD switched off for a frame.
    resolver.beginFrame();
    auto disabled = request(lods, caster);
    disabled.lodEnabled = false;
    const ResolvedShadowDraw off = resolver.resolve(
        disabled, view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget, kNoHysteresis);
    resolver.commitFrame();
    CHECK(off.reason == ShadowLodReason::LodDisabled);
    CHECK(off.level == 0);
    CHECK(resolver.historyLevel(key) == selected.level); // untouched

    // A forced fallback (an unusable budget).
    resolver.beginFrame();
    const ResolvedShadowDraw forced =
        resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 0),
                         someBounds(), -1.0f, kNoHysteresis);
    resolver.commitFrame();
    CHECK(forced.reason == ShadowLodReason::InvalidCaster);
    CHECK(resolver.historyLevel(key) == selected.level); // still untouched
}

TEST_CASE("ShadowLodResolver.SingleLevelAndDisabledResolveToTheWholeMesh", "[ShadowLodResolver]")
{
    // Both bind the base geometry, and each says WHY — a mesh with nothing to choose between is a
    // property of the asset, a disabled LOD system is a property of the frame, and conflating them
    // would misattribute every cloth mesh to the debug toggle.
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const std::vector<GeometryLod> single{
        GeometryLod{.indexBuffer = buffer(10), .indexCount = 900, .shadowDeviation = 0.0f}};
    const ResolvedShadowDraw one = resolver.resolve(
        request(single, static_cast<ShadowCasterId>(10)), view(views, ShadowViewGroup::Cascade, 0),
        someBounds(), kBudget, kNoHysteresis);
    CHECK(one.reason == ShadowLodReason::SingleLevel);
    CHECK(one.indexBuffer == buffer(10));
    CHECK(one.indexCount == 900);

    const auto lods = chain();
    auto disabled = request(lods, static_cast<ShadowCasterId>(11));
    disabled.lodEnabled = false;
    const ResolvedShadowDraw off = resolver.resolve(
        disabled, view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget, kNoHysteresis);
    CHECK(off.reason == ShadowLodReason::LodDisabled);
    CHECK(off.indexBuffer == buffer(10));
    CHECK(off.indexCount == 900);
}

TEST_CASE("ShadowLodResolver.AnInvalidViewStillDraws", "[ShadowLodResolver]")
{
    // An engaged view whose descriptor could not be built is rendering — it just cannot be
    // projected through. The caster falls back to the whole mesh and the reason says so, rather
    // than the map being left with a hole.
    const auto lods = chain();
    ShadowRenderViewSet views;
    REQUIRE(
        views.setCascade(0, Mat4::identity(), ShadowView::orthographic(0.0f), someOrthoMetrics()));
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const ResolvedShadowDraw drawn = resolver.resolve(
        request(lods, static_cast<ShadowCasterId>(12)), view(views, ShadowViewGroup::Cascade, 0),
        someBounds(), kBudget, kNoHysteresis);
    CHECK(drawn.reason == ShadowLodReason::InvalidView);
    CHECK(drawn.drawable());
    CHECK(drawn.indexCount == lods[0].indexCount);
}

TEST_CASE("ShadowLodResolver.AMalformedCoarserLevelFallsBackInsteadOfBindingNothing",
          "[ShadowLodResolver]")
{
    // The selector reasons about ERRORS; nothing upstream promises the level's CARRIERS. A half-
    // failed build can leave a coarser level with a null buffer or a zero count, and selecting it
    // would bind nothing — the caster silently missing from that view's shadow map, while the
    // counters happily reported it drawn. It falls back to the whole mesh, and the reason must stop
    // saying `Selected`, because that level was not selectable however good the maths looked.
#ifdef NDEBUG
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;

    // Deviations that make level 2 comfortably the selected level, but its buffers are missing.
    const std::vector<GeometryLod> nullBuffer{
        GeometryLod{.indexBuffer = buffer(10), .indexCount = 900, .shadowDeviation = 0.0f},
        GeometryLod{.indexBuffer = buffer(11), .indexCount = 450, .shadowDeviation = 0.01f},
        GeometryLod{.indexBuffer = NullBuffer, .indexCount = 120, .shadowDeviation = 0.02f}};
    const std::vector<GeometryLod> zeroCount{
        GeometryLod{.indexBuffer = buffer(10), .indexCount = 900, .shadowDeviation = 0.0f},
        GeometryLod{.indexBuffer = buffer(11), .indexCount = 450, .shadowDeviation = 0.01f},
        GeometryLod{.indexBuffer = buffer(12), .indexCount = 0, .shadowDeviation = 0.02f}};

    resolver.beginFrame();
    for (const auto* broken : {&nullBuffer, &zeroCount})
    {
        const ResolvedShadowDraw drawn = resolver.resolve(
            request(*broken, static_cast<ShadowCasterId>(14)),
            view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget, kNoHysteresis);
        CHECK(drawn.reason == ShadowLodReason::InvalidCaster);
        CHECK(drawn.level == 0);
        CHECK(drawn.indexBuffer == buffer(10)); // the whole mesh
        CHECK(drawn.indexCount == 900);
        CHECK(drawn.drawable()); // and it really is drawable, which is the point
        resolver.beginFrame();   // a fresh frame, so the second case is not a cache hit
    }
#endif
}

TEST_CASE("ShadowLodResolver.AnUnsetWorldScaleForcesAFallback", "[ShadowLodResolver]")
{
    // The silent-worst-case: a producer that fills every field but the scale. Zero is a LEGITIMATE
    // scale (a singular transform genuinely flattens every deviation), so defaulting to zero would
    // make the omission look like a caster with no error at all — the coarsest level, in every
    // view, with a confident `Selected` beside it. The NaN default forces a fallback instead.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    auto unset = request(lods, static_cast<ShadowCasterId>(15));
    unset.worldScale = ShadowGeometryRequest{}.worldScale; // i.e. never assigned
    const ResolvedShadowDraw forced = resolver.resolve(
        unset, view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget, kNoHysteresis);
    CHECK(forced.reason == ShadowLodReason::InvalidCaster);
    CHECK(forced.level == 0);

    // An explicitly computed zero still selects — it is a real answer about a real transform.
    resolver.beginFrame();
    auto singular = request(lods, static_cast<ShadowCasterId>(16));
    singular.worldScale = 0.0f;
    const ResolvedShadowDraw flattened = resolver.resolve(
        singular, view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget, kNoHysteresis);
    CHECK(flattened.reason == ShadowLodReason::Selected);
    CHECK(flattened.level == lods.size() - 1); // zero error: the coarsest level fits
}

TEST_CASE("ShadowLodResolver.FrameResolutionExposesTheSharedDecision", "[ShadowLodResolver]")
{
    // The DECISION, independent of which families acted on it — no `noteDrawn` here on purpose.
    // This is the entry every view with that identity is handed, and the one a consumer reasoning
    // about the decision itself wants. Attribution is a separate question with a separate query
    // (`drawnResolution`, covered below), because the shared decision alone cannot say which pass
    // drew what.
    //
    // What both share: it is the entry the pass drew from, never a fresh selection — one would see
    // a different history state, and anything derived from it would contradict the geometry.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const auto caster = static_cast<ShadowCasterId>(17);
    const ResolvedShadowDraw drawn =
        resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 0),
                         someBounds(), kBudget, kNoHysteresis);

    const ShadowLodStateKey key{caster, ShadowCasterGeneration::First,
                                ShadowLogicalViewId::cascade(0)};
    const ResolvedShadowDraw* readBack = resolver.frameResolution(key);
    REQUIRE(readBack != nullptr);
    CHECK(readBack->level == drawn.level);
    CHECK(readBack->indexBuffer == drawn.indexBuffer);
    CHECK(readBack->reason == drawn.reason);

    // A view that never resolved this caster has NO decision for it — null, not level 0. Anything
    // that read level 0 here would claim "full detail chosen" about a view that never considered
    // it.
    CHECK(resolver.frameResolution(ShadowLodStateKey{caster, ShadowCasterGeneration::First,
                                                     ShadowLogicalViewId::cascade(3)}) == nullptr);
    // Same for a caster this frame never saw, and for an unkeyable one.
    CHECK(resolver.frameResolution(ShadowLodStateKey{static_cast<ShadowCasterId>(18),
                                                     ShadowCasterGeneration::First,
                                                     ShadowLogicalViewId::cascade(0)}) == nullptr);
    CHECK(resolver.frameResolution(ShadowLodStateKey{ShadowCasterId::Invalid,
                                                     ShadowCasterGeneration::First,
                                                     ShadowLogicalViewId::cascade(0)}) == nullptr);

    // And it is a FRAME cache: the next frame starts with no decisions, so a stale level cannot be
    // read back for a view that has stopped resolving that caster.
    resolver.commitFrame();
    resolver.beginFrame();
    CHECK(resolver.frameResolution(key) == nullptr);
}

TEST_CASE("ShadowLodResolver.ProvenanceIsPerFamilyEvenWhenTheResolutionIsShared",
          "[ShadowLodResolver]")
{
    // The gap a shared resolution opens. A cascade and its world-only twin are ONE logical view, so
    // they share a decision by design — that is what makes them agree. But they do not draw the
    // same casters: world-only exists to exclude skinned ones, and cascades record first. A
    // consumer reading the level alone would therefore report one for a caster the world-only pass
    // never offered: one view's decision presented as another's.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const auto skinned = static_cast<ShadowCasterId>(20);
    const ShadowLodStateKey key{skinned, ShadowCasterGeneration::First,
                                ShadowLogicalViewId::cascade(0)};

    // The cascade resolves and draws it; the world-only pass never does.
    const ResolvedShadowDraw drawn =
        resolver.resolve(request(lods, skinned), view(views, ShadowViewGroup::Cascade, 0),
                         someBounds(), kBudget, kNoHysteresis);
    resolver.noteDrawn(ShadowViewGroup::Cascade, key);

    // The DECISION is shared — asking for it plainly still finds the level ...
    REQUIRE(resolver.frameResolution(key) != nullptr);
    CHECK(resolver.frameResolution(key)->level == drawn.level);
    // ... but attributing it to a pass is a different question, and the one consumers must ask.
    REQUIRE(resolver.drawnResolution(ShadowViewGroup::Cascade, key) != nullptr);
    CHECK(resolver.drawnResolution(ShadowViewGroup::Cascade, key)->level == drawn.level);
    CHECK(resolver.drawnResolution(ShadowViewGroup::WorldOnly, key) == nullptr);

    // Once world-only does draw it, both attribute the same shared decision.
    resolver.noteDrawn(ShadowViewGroup::WorldOnly, key);
    REQUIRE(resolver.drawnResolution(ShadowViewGroup::WorldOnly, key) != nullptr);
    CHECK(resolver.drawnResolution(ShadowViewGroup::WorldOnly, key)->level == drawn.level);
    CHECK(resolver.drawnResolution(ShadowViewGroup::Cascade, key) != nullptr);

    // A caster nobody drew has no attribution anywhere.
    const ShadowLodStateKey unseen{static_cast<ShadowCasterId>(21), ShadowCasterGeneration::First,
                                   ShadowLogicalViewId::cascade(0)};
    CHECK(resolver.drawnResolution(ShadowViewGroup::Cascade, unseen) == nullptr);

    // Provenance is per FRAME: a view that stops drawing a caster must stop attributing it.
    resolver.commitFrame();
    resolver.beginFrame();
    CHECK(resolver.drawnResolution(ShadowViewGroup::Cascade, key) == nullptr);
}

TEST_CASE("ShadowLodResolver.MarkingAnUnresolvedCasterDrawnChangesNothing", "[ShadowLodResolver]")
{
    // Provenance is a FIELD of a decision, not a record of its own: marking a caster the resolver
    // never resolved would manufacture attribution for a level nobody chose. (The shadow pass
    // treats drawing an unresolved caster as terminal; this is the store refusing to record it.)
#ifdef NDEBUG
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const ShadowLodStateKey neverResolved{static_cast<ShadowCasterId>(22),
                                          ShadowCasterGeneration::First,
                                          ShadowLogicalViewId::cascade(0)};
    resolver.noteDrawn(ShadowViewGroup::Cascade, neverResolved);
    CHECK(resolver.frameCacheSize() == 0);
    CHECK(resolver.frameResolution(neverResolved) == nullptr);
    CHECK(resolver.drawnResolution(ShadowViewGroup::Cascade, neverResolved) == nullptr);

    // An unkeyable caster is in no store at all, so it likewise gains nothing.
    const ShadowLodStateKey unkeyable{ShadowCasterId::Invalid, ShadowCasterGeneration::First,
                                      ShadowLogicalViewId::cascade(0)};
    resolver.noteDrawn(ShadowViewGroup::Cascade, unkeyable);
    CHECK(resolver.frameCacheSize() == 0);
    CHECK(resolver.drawnResolution(ShadowViewGroup::Cascade, unkeyable) == nullptr);
#endif
}

TEST_CASE("ShadowLodResolver.OneCasterTintsDifferentlyPerFocusedView", "[ShadowLodResolver]")
{
    // What the tint exists to show. The same caster resolves against two views with very different
    // texel sizes, so reading back per view gives different levels — the SH-03 fix, in the one
    // place a person can see it.
    const auto lods = chain();
    ShadowRenderViewSet views;
    // A coarse cascade (0.5 world units per texel) and a fine one (0.005): level 2's 0.20 deviation
    // is 0.4 texels in the first and 40 in the second.
    REQUIRE(
        views.setCascade(0, Mat4::identity(), ShadowView::orthographic(0.5f), someOrthoMetrics()));
    REQUIRE(views.setCascade(1, Mat4::identity(), ShadowView::orthographic(0.005f),
                             someOrthoMetrics()));

    ShadowLodResolver resolver;
    resolver.beginFrame();
    const auto caster = static_cast<ShadowCasterId>(19);
    const ResolvedShadowDraw coarse =
        resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 0),
                         someBounds(), kBudget, kNoHysteresis);
    const ResolvedShadowDraw fine =
        resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 1),
                         someBounds(), kBudget, kNoHysteresis);

    CHECK(coarse.reason == ShadowLodReason::Selected);
    CHECK(fine.reason == ShadowLodReason::Selected);
    CHECK(coarse.level > fine.level); // the coarse map tolerates far more object-space error

    // Both cascades draw it, which is what makes the level attributable to each.
    const auto keyFor = [&](std::uint32_t cascade)
    {
        return ShadowLodStateKey{caster, ShadowCasterGeneration::First,
                                 ShadowLogicalViewId::cascade(cascade)};
    };
    resolver.noteDrawn(ShadowViewGroup::Cascade, keyFor(0));
    resolver.noteDrawn(ShadowViewGroup::Cascade, keyFor(1));

    // Read back through the TINT's query, per view: focusing one must not change what the other
    // reports. This is the SH-03 fix in the one place a person can see it.
    const auto tintLevelFor = [&](std::uint32_t cascade)
    {
        const ResolvedShadowDraw* r =
            resolver.drawnResolution(ShadowViewGroup::Cascade, keyFor(cascade));
        REQUIRE(r != nullptr);
        return r->level;
    };
    CHECK(tintLevelFor(0) == coarse.level);
    CHECK(tintLevelFor(1) == fine.level);
}

TEST_CASE("ShadowLodResolver.MovementSeparatesFirstSeenHeldTransitionedAndReversed",
          "[ShadowLodResolver]")
{
    // The dead band's calibration instrument, and the distinction it exists to make: a caster
    // receding legitimately steps L0 -> L1 -> L2 (transitions no hysteresis can or should remove),
    // while CHATTER is a reversal — L1 -> L2 -> L1 — a caster oscillating across a threshold.
    // Judging a dead band by transitions alone would blame it for ordinary motion.
    const auto lods = chain();
    ShadowLodResolver resolver;
    const auto caster = static_cast<ShadowCasterId>(30);
    const ShadowLodStateKey key{caster, ShadowCasterGeneration::First,
                                ShadowLogicalViewId::cascade(0)};

    // Drives one caster to a chosen level by picking a view whose texel size forces it.
    const auto commitLevelUsing = [&](float worldUnitsPerTexel)
    {
        ShadowRenderViewSet frameViews;
        REQUIRE(frameViews.setCascade(
            0, Mat4::identity(), ShadowView::orthographic(worldUnitsPerTexel), someOrthoMetrics()));
        resolver.beginFrame();
        const ResolvedShadowDraw drawn =
            resolver.resolve(request(lods, caster), view(frameViews, ShadowViewGroup::Cascade, 0),
                             someBounds(), kBudget, kNoHysteresis);
        resolver.commitFrame();
        return drawn.level;
    };

    // Frame 1: first sight of this caster in this view.
    const std::size_t fine = commitLevelUsing(0.005f);
    CHECK(resolver.lastCommitMovement().firstSeen == 1);
    CHECK(resolver.lastCommitMovement().transitions == 0);
    CHECK(resolver.lastCommitMovement().reversed == 0);

    // Frame 2: same view, same answer — held.
    CHECK(commitLevelUsing(0.005f) == fine);
    CHECK(resolver.lastCommitMovement().held == 1);
    CHECK(resolver.lastCommitMovement().transitions == 0);

    // Frame 3: a much coarser map, so the caster moves to a coarser level. Movement, not chatter.
    const std::size_t coarse = commitLevelUsing(0.5f);
    REQUIRE(coarse != fine);
    CHECK(resolver.lastCommitMovement().transitions == 1);
    CHECK(resolver.lastCommitMovement().reversed == 0);

    // Frame 4: back to the fine map — the caster returns to the level it held two commits ago.
    // THIS is chatter, and it is what a dead band could suppress.
    CHECK(commitLevelUsing(0.005f) == fine);
    CHECK(resolver.lastCommitMovement().transitions == 1);
    CHECK(resolver.lastCommitMovement().reversed == 1);
    CHECK(resolver.historyLevel(key) == fine);
}

TEST_CASE("ShadowLodResolver.AReversalAfterALongHoldIsNotChatter", "[ShadowLodResolver]")
{
    // The distinction that decides whether a dead band is justified. A periodic animation walks a
    // caster L1 -> L2 and back every few seconds; counting that return as chatter would report the
    // ANIMATION as instability and invite widening a band that cannot fix it. Chatter is a reversal
    // that undoes a RECENT transition — a caster sitting on a threshold, flipping.
    const auto lods = chain();
    ShadowLodResolver resolver;
    const auto caster = static_cast<ShadowCasterId>(33);

    const auto commitAt = [&](float worldUnitsPerTexel)
    {
        ShadowRenderViewSet frameViews;
        REQUIRE(frameViews.setCascade(
            0, Mat4::identity(), ShadowView::orthographic(worldUnitsPerTexel), someOrthoMetrics()));
        resolver.beginFrame();
        (void)resolver.resolve(request(lods, caster), view(frameViews, ShadowViewGroup::Cascade, 0),
                               someBounds(), kBudget, kNoHysteresis);
        resolver.commitFrame();
    };

    commitAt(0.005f); // first sight: fine level
    commitAt(0.5f);   // transition to coarse
    REQUIRE(resolver.lastCommitMovement().transitions == 1);

    // Hold the coarse level well beyond the window, as a slow animation would.
    for (std::uint64_t i = 0; i < 40; ++i)
    {
        commitAt(0.5f);
    }
    REQUIRE(resolver.lastCommitMovement().held == 1);

    // Now return. It IS a transition — the level really did change — but it is not chatter, and
    // must not be counted as evidence for a dead band.
    commitAt(0.005f);
    CHECK(resolver.lastCommitMovement().transitions == 1);
    CHECK(resolver.lastCommitMovement().reversed == 0);

    // Whereas an immediate flip back is chatter.
    commitAt(0.5f);
    CHECK(resolver.lastCommitMovement().reversed == 1);
}

TEST_CASE("ShadowLodResolver.MovementIsPerCommitAndIgnoresAbandonedFrames", "[ShadowLodResolver]")
{
    // Two properties the aggregate depends on. It describes ONE commit — otherwise a per-frame log
    // would report a running total and every sweep row would be meaningless — and a frame that was
    // never submitted contributes nothing, because its levels were never true.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    const auto caster = static_cast<ShadowCasterId>(31);

    resolver.beginFrame();
    (void)resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 0),
                           someBounds(), kBudget, kNoHysteresis);
    resolver.commitFrame();
    CHECK(resolver.lastCommitMovement().firstSeen == 1);

    // An abandoned frame: resolved, then discarded. The next commit must not report its decision,
    // and the history must not have moved.
    resolver.beginFrame();
    (void)resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 0),
                           someBounds(), kBudget, kNoHysteresis);
    resolver.discardFrame();
    resolver.beginFrame();
    resolver.commitFrame();
    const ShadowLodTransitions afterEmpty = resolver.lastCommitMovement();
    CHECK(afterEmpty.firstSeen == 0);
    CHECK(afterEmpty.held == 0);
    CHECK(afterEmpty.transitions == 0);
    CHECK(afterEmpty.reversed == 0);
}

TEST_CASE("ShadowLodResolver.ForcedDecisionsDoNotCountAsMovement", "[ShadowLodResolver]")
{
    // Only a genuinely Selected level is staged, so a forced fallback cannot register as a
    // transition. Otherwise a frame where the budget went invalid would read as a burst of chatter
    // and send someone tuning the dead band to fix a plumbing error.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    const auto caster = static_cast<ShadowCasterId>(32);

    resolver.beginFrame();
    (void)resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 0),
                           someBounds(), kBudget, kNoHysteresis);
    resolver.commitFrame();
    REQUIRE(resolver.lastCommitMovement().firstSeen == 1);

    resolver.beginFrame();
    const ResolvedShadowDraw forced =
        resolver.resolve(request(lods, caster), view(views, ShadowViewGroup::Cascade, 0),
                         someBounds(), -1.0f, kNoHysteresis);
    resolver.commitFrame();
    REQUIRE(forced.reason == ShadowLodReason::InvalidCaster);
    const ShadowLodTransitions movement = resolver.lastCommitMovement();
    CHECK(movement.transitions == 0);
    CHECK(movement.reversed == 0);
    CHECK(movement.held == 0);
    CHECK(movement.firstSeen == 0);
}

TEST_CASE("ShadowLodResolver.DifferentViewsOfOneCasterAreSeparateHistory", "[ShadowLodResolver]")
{
    // The point of the whole slice: one caster, several views, several answers — and each view's
    // dead band belongs to it alone.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;
    resolver.beginFrame();

    const auto req = request(lods, static_cast<ShadowCasterId>(13));
    (void)resolver.resolve(req, view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget,
                           kNoHysteresis);
    (void)resolver.resolve(req, view(views, ShadowViewGroup::Spot, 0), someBounds(), kBudget,
                           kNoHysteresis);
    (void)resolver.resolve(req, view(views, ShadowViewGroup::Self, 0), someBounds(), kBudget,
                           kNoHysteresis);
    resolver.commitFrame();

    CHECK(resolver.frameCacheSize() == 3);
    CHECK(resolver.historySize() == 3);
}

// ---------------------------------------------------------------------------
// SH-04: a caster that deforms after its error was measured may not select.
//
// The defect being closed: shadow LOD selected levels for skinned and morphed casters using a
// deviation the simplifier measured on the BIND POSE. Skinning can carry a vertex arbitrarily far
// from where that number was taken, so the estimate is not loose there — it describes a mesh that
// is never drawn. These cases pin the classification's consequences: full detail, a reason that
// says why, an error that claims nothing, and no entry in a history built from justified decisions.
// ---------------------------------------------------------------------------

namespace
{

ShadowGeometryRequest deformableRequest(const std::vector<GeometryLod>& lods, ShadowCasterId caster)
{
    ShadowGeometryRequest req = request(lods, caster);
    req.deformation = ShadowCasterDeformation::Deformable;
    return req;
}

} // namespace

TEST_CASE("ShadowLodResolver.ADeformableCasterCannotSelectBelowFullDetail", "[ShadowLodResolver]")
{
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();

    // The control matters more than the assertion: at this texel size a RIGID caster with the same
    // chain and the same view does select a coarser level. Without it, "level 0" would be evidence
    // of nothing — it is only meaningful against a case that would have chosen otherwise.
    const ResolvedShadowDraw rigid =
        resolveShadowDraw(request(lods, static_cast<ShadowCasterId>(1)),
                          view(views, ShadowViewGroup::Cascade, 0).projection(), someBounds(),
                          kBudget, kNoHysteresis, kNoPreviousShadowLod);
    REQUIRE(rigid.reason == ShadowLodReason::Selected);
    REQUIRE(rigid.level > 0);

    const ResolvedShadowDraw deformable =
        resolveShadowDraw(deformableRequest(lods, static_cast<ShadowCasterId>(2)),
                          view(views, ShadowViewGroup::Cascade, 0).projection(), someBounds(),
                          kBudget, kNoHysteresis, kNoPreviousShadowLod);
    CHECK(deformable.reason == ShadowLodReason::DeformableFallback);
    CHECK(deformable.level == 0);
    // The WHOLE mesh, not lods[0] — the two coincide here, but the fallback must bind the base
    // buffers so a chain whose level 0 is itself simplified cannot smuggle in a reduced mesh.
    CHECK(deformable.indexBuffer == buffer(10));
    CHECK(deformable.indexCount == 900);
    // Infinity, not zero: the error is unbounded, and 0 would rank this caster as the most accurate
    // in the frame — the exact inversion of what is known about it.
    CHECK(std::isinf(deformable.projectedTexels));
}

TEST_CASE("ShadowLodResolver.DeformationOutranksTheChainButNotTheUserOrAProducerBug",
          "[ShadowLodResolver]")
{
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    const ShadowView projection = view(views, ShadowViewGroup::Cascade, 0).projection();

    SECTION("a malformed request is a producer bug and outranks deformation")
    {
        // Dev asserts on an unfilled request, so this pins the RELEASE precedence — same convention
        // as the unkeyable-caster case earlier in this file.
#ifdef NDEBUG
        ShadowGeometryRequest req = deformableRequest(lods, static_cast<ShadowCasterId>(3));
        req.baseIndexCount = 0;
        const ResolvedShadowDraw resolved = resolveShadowDraw(
            req, projection, someBounds(), kBudget, kNoHysteresis, kNoPreviousShadowLod);
        CHECK(resolved.reason == ShadowLodReason::InvalidCaster);
#endif
    }

    SECTION("the user's toggle is the operative fact when shadow LOD is off")
    {
        ShadowGeometryRequest req = deformableRequest(lods, static_cast<ShadowCasterId>(4));
        req.lodEnabled = false;
        const ResolvedShadowDraw resolved = resolveShadowDraw(
            req, projection, someBounds(), kBudget, kNoHysteresis, kNoPreviousShadowLod);
        // A deformable caster is not MORE disabled than the rest of the frame; reporting the
        // fallback here would attribute a global switch to a property of this one caster.
        CHECK(resolved.reason == ShadowLodReason::LodDisabled);
    }

    SECTION("a single-level deformable caster still reports why it may not select")
    {
        // Cloth today: storage-vertex geometry with one level. It would draw the whole mesh either
        // way, so the LEVEL proves nothing — the reason is the whole point. SingleLevel would read
        // as "safe because there was nothing to choose", and that accident stops being true the day
        // storage-vertex geometry gains a chain.
        const std::vector<GeometryLod> single{
            GeometryLod{.indexBuffer = buffer(10), .indexCount = 900, .shadowDeviation = 0.0f}};
        const ResolvedShadowDraw resolved =
            resolveShadowDraw(deformableRequest(single, static_cast<ShadowCasterId>(5)), projection,
                              someBounds(), kBudget, kNoHysteresis, kNoPreviousShadowLod);
        CHECK(resolved.reason == ShadowLodReason::DeformableFallback);
    }
}

TEST_CASE("ShadowLodResolver.ADeformableCasterLeavesNoHysteresisHistory", "[ShadowLodResolver]")
{
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;

    resolver.beginFrame();
    (void)resolver.resolve(deformableRequest(lods, static_cast<ShadowCasterId>(6)),
                           view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget,
                           kNoHysteresis);
    resolver.commitFrame();

    // It IS in the frame cache — a consumer still has to be able to ask what this caster drew — but
    // never in the history, which exists to carry justified levels across frames. A dead band
    // derived from a forced fallback would be a dead band around nothing.
    CHECK(resolver.frameCacheSize() == 1);
    CHECK(resolver.historySize() == 0);
    CHECK(resolver.lastCommitMovement().firstSeen == 0);
    CHECK(resolver.lastCommitMovement().transitions == 0);
}

TEST_CASE("ShadowLodResolver.ARequestThatOmitsDeformationDoesNotSelect", "[ShadowLodResolver]")
{
    // The default is the SAFE answer, on the same principle as worldScale's NaN. A producer that
    // forgets this field loses triangles and says so loudly in the panel; the opposite default
    // would select levels on an error claim nobody established, silently.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();

    ShadowGeometryRequest req{};
    req.lods = lods;
    req.baseIndexBuffer = buffer(10);
    req.baseIndexCount = 900;
    req.worldScale = 1.0f;
    req.casterId = static_cast<ShadowCasterId>(7);

    const ResolvedShadowDraw resolved =
        resolveShadowDraw(req, view(views, ShadowViewGroup::Cascade, 0).projection(), someBounds(),
                          kBudget, kNoHysteresis, kNoPreviousShadowLod);
    CHECK(resolved.reason == ShadowLodReason::DeformableFallback);
}

// ---------------------------------------------------------------------------
// SH-05: an alpha-masked caster's shadow LOD begins at level 0.
//
// The defect being closed is upstream of the selector: a cutout's silhouette is decided by
// base-colour alpha sampled through the level's UVs, and NO channel the simplifier records measures
// that. A collapse can hold the surface inside the shadow budget while moving the cutout boundary
// anywhere, so a coarser level's shadow is unbounded in exactly the dimension that defines the
// shape. Until a silhouette-error policy exists (SH-05 leaves it as future work), a masked caster
// draws its whole mesh and the panel says why.
// ---------------------------------------------------------------------------

namespace
{

ShadowGeometryRequest maskedRequest(const std::vector<GeometryLod>& lods, ShadowCasterId caster)
{
    ShadowGeometryRequest req = request(lods, caster);
    req.alpha = ShadowCasterAlpha::Masked;
    return req;
}

} // namespace

TEST_CASE("ShadowLodResolver.AnAlphaMaskedCasterCannotSelectBelowFullDetail", "[ShadowLodResolver]")
{
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();

    // The control, as in the deformable case: at this texel size the SAME chain in the SAME view
    // selects a coarser level for an opaque caster, so "level 0" is evidence rather than a
    // coincidence of the fixture.
    const ResolvedShadowDraw opaque =
        resolveShadowDraw(request(lods, static_cast<ShadowCasterId>(31)),
                          view(views, ShadowViewGroup::Cascade, 0).projection(), someBounds(),
                          kBudget, kNoHysteresis, kNoPreviousShadowLod);
    REQUIRE(opaque.reason == ShadowLodReason::Selected);
    REQUIRE(opaque.level > 0);

    const ResolvedShadowDraw masked =
        resolveShadowDraw(maskedRequest(lods, static_cast<ShadowCasterId>(32)),
                          view(views, ShadowViewGroup::Cascade, 0).projection(), someBounds(),
                          kBudget, kNoHysteresis, kNoPreviousShadowLod);
    CHECK(masked.reason == ShadowLodReason::AlphaMaskedFallback);
    CHECK(masked.level == 0);
    // The WHOLE mesh, not lods[0] — same reasoning as the deformable fallback: a chain whose level
    // 0 is itself simplified must not smuggle a reduced mesh in through a fallback.
    CHECK(masked.indexBuffer == buffer(10));
    CHECK(masked.indexCount == 900);
    // Infinity, not zero: the silhouette error is unbounded, not measured-as-zero, and 0 would rank
    // a cutout as the most accurate caster in the frame.
    CHECK(std::isinf(masked.projectedTexels));
}

TEST_CASE("ShadowLodResolver.MaskingRanksBelowDeformationAndAboveTheChain", "[ShadowLodResolver]")
{
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    const ShadowView projection = view(views, ShadowViewGroup::Cascade, 0).projection();

    SECTION("a caster that both deforms and masks reports the deformation")
    {
        // Both produce the same whole-mesh draw, so only the REASON distinguishes them, and the
        // panel must name the stronger statement: a mesh that moves after measurement has no valid
        // error model at all, while a masked one has a valid surface error and an unmeasured
        // silhouette.
        ShadowGeometryRequest req = maskedRequest(lods, static_cast<ShadowCasterId>(33));
        req.deformation = ShadowCasterDeformation::Deformable;
        const ResolvedShadowDraw resolved = resolveShadowDraw(
            req, projection, someBounds(), kBudget, kNoHysteresis, kNoPreviousShadowLod);
        CHECK(resolved.reason == ShadowLodReason::DeformableFallback);
        CHECK(resolved.level == 0);
    }

    SECTION("the user's toggle is still the operative fact when shadow LOD is off")
    {
        ShadowGeometryRequest req = maskedRequest(lods, static_cast<ShadowCasterId>(34));
        req.lodEnabled = false;
        const ResolvedShadowDraw resolved = resolveShadowDraw(
            req, projection, someBounds(), kBudget, kNoHysteresis, kNoPreviousShadowLod);
        CHECK(resolved.reason == ShadowLodReason::LodDisabled);
    }

    SECTION("a single-level masked caster still reports why it may not select")
    {
        // Same argument SH-04 made for cloth: the level proves nothing here (there is only one), so
        // the reason is the whole point — SingleLevel would read as "safe because there was nothing
        // to choose", and that accident stops being true the day a cutout mesh grows a chain.
        const std::vector<GeometryLod> single{
            GeometryLod{.indexBuffer = buffer(10), .indexCount = 900, .shadowDeviation = 0.0f}};
        const ResolvedShadowDraw resolved =
            resolveShadowDraw(maskedRequest(single, static_cast<ShadowCasterId>(35)), projection,
                              someBounds(), kBudget, kNoHysteresis, kNoPreviousShadowLod);
        CHECK(resolved.reason == ShadowLodReason::AlphaMaskedFallback);
    }
}

TEST_CASE("ShadowLodResolver.AnAlphaMaskedCasterLeavesNoHysteresisHistory", "[ShadowLodResolver]")
{
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();
    ShadowLodResolver resolver;

    resolver.beginFrame();
    (void)resolver.resolve(maskedRequest(lods, static_cast<ShadowCasterId>(36)),
                           view(views, ShadowViewGroup::Cascade, 0), someBounds(), kBudget,
                           kNoHysteresis);
    resolver.commitFrame();

    // In the frame cache (a consumer must still be able to ask what this caster drew), never in the
    // history: a dead band derived from a forced fallback would be a dead band around nothing.
    CHECK(resolver.frameCacheSize() == 1);
    CHECK(resolver.historySize() == 0);
    CHECK(resolver.lastCommitMovement().firstSeen == 0);
    CHECK(resolver.lastCommitMovement().transitions == 0);
}

TEST_CASE("ShadowLodResolver.ARequestThatOmitsTheAlphaModeDoesNotSelect", "[ShadowLodResolver]")
{
    // The default is the safe answer, like `deformation`'s. A producer that forgets this field pays
    // triangles and a texture fetch and says so in the panel; the opposite default would silently
    // restore the bug SH-05 exists to fix — a cutout casting its quad.
    const auto lods = chain();
    ShadowRenderViewSet views = populatedViews();

    ShadowGeometryRequest req{};
    req.lods = lods;
    req.baseIndexBuffer = buffer(10);
    req.baseIndexCount = 900;
    req.worldScale = 1.0f;
    req.casterId = static_cast<ShadowCasterId>(37);
    // Deformation stated, so this case isolates the ALPHA default rather than tripping SH-04's.
    req.deformation = ShadowCasterDeformation::Rigid;

    const ResolvedShadowDraw resolved =
        resolveShadowDraw(req, view(views, ShadowViewGroup::Cascade, 0).projection(), someBounds(),
                          kBudget, kNoHysteresis, kNoPreviousShadowLod);
    CHECK(resolved.reason == ShadowLodReason::AlphaMaskedFallback);
}
