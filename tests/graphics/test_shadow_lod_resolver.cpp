#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

#include <fire_engine/graphics/shadow_lod_resolver.hpp>

using namespace fire_engine;

namespace
{

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
    return ShadowGeometryRequest{.lods = lods,
                                 .baseIndexBuffer = buffer(10),
                                 .baseIndexCount = 900,
                                 .worldScale = 1.0f,
                                 .casterId = caster,
                                 .generation = generation,
                                 .lodEnabled = true};
}

// A set with every cascade populated (and world-only enabled), one spot and one point light, so a
// test can ask any family for a real view.
ShadowRenderViewSet populatedViews()
{
    ShadowRenderViewSet views;
    for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        // A coarse texel: 0.05 world units per texel puts level 2's 0.20 deviation at 4 texels.
        REQUIRE(views.setCascade(cascade, Mat4::identity(), ShadowView::orthographic(0.05f)));
        REQUIRE(views.enableWorldOnly(cascade));
    }
    REQUIRE(views.setSelf(0, 7, Mat4::identity(), ShadowView::orthographic(0.05f)));
    const auto light = static_cast<NodeId>(21);
    REQUIRE(views.setSpot(0, light, Mat4::identity(),
                          ShadowView::perspective(Vec3{0.0f, 0.0f, 20.0f}, Vec3{0.0f, 0.0f, -1.0f},
                                                  1.0f, 1024, 0.1f)));
    for (std::uint8_t face = 0; face < kCubeFaceCount; ++face)
    {
        const std::array<Vec3, kCubeFaceCount> forwards{Vec3{1, 0, 0}, Vec3{-1, 0, 0},
                                                        Vec3{0, 1, 0}, Vec3{0, -1, 0},
                                                        Vec3{0, 0, 1}, Vec3{0, 0, -1}};
        REQUIRE(views.setPoint(
            0, face, light, Mat4::identity(),
            ShadowView::perspective(Vec3{0.0f, 0.0f, 30.0f}, forwards[face], 1.5708f, 512, 0.1f)));
    }
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
    REQUIRE(views.setCascade(0, Mat4::identity(), ShadowView::orthographic(0.0f)));
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
