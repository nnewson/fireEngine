#include <fire_engine/graphics/shadow_pass_prepare.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace fire_engine;

namespace
{

constexpr auto kLight = static_cast<NodeId>(21);
constexpr float kBudget = 4.0f;
constexpr ShadowLodHysteresis kNoHysteresis{.coarsenRatio = 1.0f};
constexpr float kPointRange = 30.0f;
const Vec3 kPointPosition{2.0f, 3.0f, 4.0f};

// --- view-set fixtures -------------------------------------------------------------------------

[[nodiscard]] ShadowView someOrtho()
{
    return ShadowView::orthographic(0.05f);
}
[[nodiscard]] ShadowViewMetrics someOrthoMetrics()
{
    return ShadowViewMetrics::orthographic(0.05f, 100.0f);
}
[[nodiscard]] ShadowView somePerspective(const Vec3& forward, const Vec3& position)
{
    return ShadowView::perspective(position, forward, 1.5708f, 512, 0.1f);
}

// An identity-ish matrix with a mark, so each view's transform is trivially identifiable and the
// frustum it produces admits geometry near the origin.
[[nodiscard]] Mat4 markedMatrix(float mark)
{
    Mat4 m = Mat4::identity();
    m[0, 3] = mark;
    return m;
}

// Every family populated at once, so a test can assert what a family did AND what its neighbours
// did — the accounting rules are per family, and most of the ways to break them show up as one
// family's work landing on another's row.
[[nodiscard]] ShadowRenderViewSet populatedViews()
{
    ShadowRenderViewSet views;
    REQUIRE(views.setCascade(0, markedMatrix(0.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.enableWorldOnly(0));
    REQUIRE(views.setSelf(0, 55, markedMatrix(0.0f), someOrtho(), someOrthoMetrics()));
    REQUIRE(views.setSpot(0, kLight, markedMatrix(0.0f),
                          somePerspective(Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 5.0f}),
                          ShadowViewMetrics::spot(0.002f, 0.1f, 50.0f)));

    const std::array<Vec3, kCubeFaceCount> forwards{Vec3{1, 0, 0},  Vec3{-1, 0, 0}, Vec3{0, 1, 0},
                                                    Vec3{0, -1, 0}, Vec3{0, 0, 1},  Vec3{0, 0, -1}};
    const auto face = [&](std::uint8_t f)
    { return ShadowPointFace{markedMatrix(0.0f), somePerspective(forwards[f], kPointPosition)}; };
    const std::array<ShadowPointFace, kCubeFaceCount> cube{face(0), face(1), face(2),
                                                           face(3), face(4), face(5)};
    REQUIRE(views.setPointLight(0, kLight, ShadowViewMetrics::pointLight(0.004f, kPointRange),
                                kPointRange,
                                std::span<const ShadowPointFace, kCubeFaceCount>{cube}));
    return views;
}

// --- caster fixtures ---------------------------------------------------------------------------

// A single-level rigid caster. Single-level so no test depends on the selector's arithmetic: what
// these tests pin is the plumbing around it, and a fixture whose level moved with an unrelated
// tuning change would fail for the wrong reason.
const std::vector<GeometryLod> kSingleLevel{GeometryLod{
    .indexBuffer = static_cast<BufferHandle>(3), .indexCount = 900, .shadowDeviation = 0.0f}};

[[nodiscard]] DrawCommand
caster(std::uint32_t objectId, Bounds3 bounds,
       ShadowCasterDeformation deformation = ShadowCasterDeformation::Rigid)
{
    DrawCommand dc{};
    dc.objectId = objectId;
    dc.vertexBuffer = static_cast<BufferHandle>(objectId + 100);
    dc.indexType = DrawIndexType::UInt16;
    dc.materialIndex = objectId;
    dc.shadowUbo = static_cast<BufferHandle>(objectId + 200);
    dc.skinUbo = static_cast<BufferHandle>(objectId + 300);
    dc.morphUbo = static_cast<BufferHandle>(objectId + 400);
    dc.morphSsbo = static_cast<BufferHandle>(objectId + 500);
    dc.shadowBounds = bounds;
    // EXACT, so the frustum filter is allowed to reject it — `Stale` bounds are admitted whatever
    // the frustum says, which would make the filter tests vacuous.
    dc.shadowBoundsKind = ShadowCasterBoundsKind::Exact;
    dc.shadowRequest = ShadowGeometryRequest{
        .lods = kSingleLevel,
        .baseIndexBuffer = static_cast<BufferHandle>(3),
        .baseIndexCount = 900,
        .pose =
            ShadowCasterPose::fromModel(Mat4::translate(Vec3{static_cast<float>(objectId), 0, 0})),
        .casterId = static_cast<ShadowCasterId>(objectId),
        .generation = ShadowCasterGeneration::First,
        .lodEnabled = true,
        .deformation = deformation,
        .alpha = ShadowCasterAlpha::Opaque,
    };
    return dc;
}

[[nodiscard]] Bounds3 boundsAt(Vec3 centre, float halfSize = 0.25f)
{
    Bounds3 b{};
    b.expand(centre - Vec3{halfSize, halfSize, halfSize});
    b.expand(centre + Vec3{halfSize, halfSize, halfSize});
    return b;
}

// Inside every fixture view's frustum (the marked matrices are identity-like, so the unit cube
// about the origin is in view).
[[nodiscard]] DrawCommand nearCaster(std::uint32_t objectId = 1)
{
    return caster(objectId, boundsAt(Vec3{0.0f, 0.0f, 0.0f}));
}

// Far outside every fixture view's frustum, so the filter rejects it.
[[nodiscard]] DrawCommand distantCaster(std::uint32_t objectId = 2)
{
    return caster(objectId, boundsAt(Vec3{5000.0f, 5000.0f, 5000.0f}));
}

[[nodiscard]] ShadowMapValidity allFamilies()
{
    return ShadowMapValidity{
        .cascades = true, .worldOnly = true, .self = true, .spot = true, .point = true};
}

struct Prepared
{
    ShadowFramePlan plan{};
    ShadowFrameStats stats{};
    ShadowLodResolver resolver{};
};

// Runs a preparation over the standard view set. Returned by value so each test owns its own
// resolver and stats — the hysteresis history is cross-frame state, and a shared one would make
// tests order-dependent.
void prepare(Prepared& out, const ShadowPreparationInputs& inputs, const ShadowRenderViewSet& views,
             ShadowMapValidity eligible)
{
    out.resolver.beginFrame();
    prepareShadowFrame(inputs, views, eligible, out.resolver, out.stats, out.plan);
}

[[nodiscard]] ShadowPreparationInputs inputsFor(std::span<const DrawCommand> shadowDraws,
                                                std::span<const DrawCommand> worldOnlyDraws = {},
                                                std::span<const DrawCommand> selfDraws = {})
{
    ShadowPreparationInputs inputs{
        .shadowDraws = shadowDraws,
        .worldOnlyShadowDraws = worldOnlyDraws,
        .selfShadowDraws = selfDraws,
        .lodBudgetTexels = kBudget,
        .hysteresis = kNoHysteresis,
        .cullingEnabled = true,
    };
    for (ShadowFamilyRaster& raster : inputs.raster)
    {
        raster =
            ShadowFamilyRaster{.extent = 1024, .depthBiasConstant = 1.0f, .depthBiasSlope = 2.0f};
    }
    return inputs;
}

} // namespace

TEST_CASE("every active view is recorded while nothing is resident", "[ShadowPassPrepare]")
{
    // The reuse half of the item has no residency store yet, so every view is a first use — and a
    // first use RECORDS whatever its prepared content looks like, because image creation
    // transitions the layout but leaves the depth undefined.
    const std::vector<DrawCommand> draws{nearCaster()};
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor(draws, draws, draws), views, allFamilies());

    CHECK(out.plan.disposition(ShadowViewGroup::Cascade, 0) == ShadowViewDisposition::Recorded);
    CHECK(out.plan.disposition(ShadowViewGroup::WorldOnly, 0) == ShadowViewDisposition::Recorded);
    CHECK(out.plan.disposition(ShadowViewGroup::Spot, 0) == ShadowViewDisposition::Recorded);
    for (std::size_t f = 0; f < static_cast<std::size_t>(kCubeFaceCount); ++f)
    {
        CHECK(out.plan.disposition(ShadowViewGroup::Point, shadowPointViewSlot(0, f)) ==
              ShadowViewDisposition::Recorded);
    }
    // An inactive slot is not claimed at all, which is a different answer from "recorded nothing".
    CHECK(out.plan.disposition(ShadowViewGroup::Cascade, 1) == ShadowViewDisposition::Invalid);
    CHECK(out.plan.view(ShadowViewGroup::Cascade, 1) == nullptr);
    CHECK_FALSE(out.stats.view(ShadowViewGroup::Cascade, 1).claimed());
}

TEST_CASE("a suppressed family is never resolved", "[ShadowPassPrepare]")
{
    // Not merely "not recorded". Resolution STAGES hysteresis, so a family that will be neither
    // recorded nor sampled must not leave a dead band behind for the commit to adopt — which means
    // the resolver may not be asked about its casters at all.
    const std::vector<DrawCommand> draws{nearCaster()};
    const ShadowRenderViewSet views = populatedViews();

    Prepared out{};
    prepare(
        out, inputsFor(draws, draws, draws), views,
        ShadowMapValidity{
            .cascades = false, .worldOnly = false, .self = false, .spot = true, .point = false});

    // The spot family was eligible and resolved; nothing else reached the resolver, so the only
    // frame entry is the spot view's.
    CHECK(out.resolver.frameCacheSize() == 1);
    CHECK(out.resolver.frameResolution(
              ShadowLodStateKey{static_cast<ShadowCasterId>(1), ShadowCasterGeneration::First,
                                ShadowLogicalViewId::spot(kLight)}) != nullptr);
    CHECK(out.resolver.frameResolution(
              ShadowLodStateKey{static_cast<ShadowCasterId>(1), ShadowCasterGeneration::First,
                                ShadowLogicalViewId::cascade(0)}) == nullptr);

    // And a suppressed family leaves no diagnostic row: it was not claimed, so nothing reports a
    // view that did not happen.
    CHECK_FALSE(out.stats.view(ShadowViewGroup::Cascade, 0).claimed());
    CHECK(out.stats.view(ShadowViewGroup::Spot, 0).claimed());
    CHECK(out.plan.sampleableCount(ShadowViewGroup::Cascade) == 0);
    CHECK(out.plan.sampleableCount(ShadowViewGroup::Spot) == 1);
}

TEST_CASE("the filter runs before resolution", "[ShadowPassPrepare]")
{
    // A caster this view drops must acquire NO history against it: a skinned caster would otherwise
    // accumulate a dead band against every other object's self-shadow map, and a wholly-rejected
    // perspective caster would be evaluated outside the domain the projection model is good for.
    const std::vector<DrawCommand> draws{distantCaster()};
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor(draws), views, allFamilies());

    CHECK(out.resolver.frameCacheSize() == 0);
    // The row still reports the caster as a CANDIDATE — it was offered and rejected, which is
    // exactly what `candidateDraws - drawnDraws` is supposed to measure.
    const ShadowViewStats& row = out.stats.view(ShadowViewGroup::Cascade, 0);
    CHECK(row.claimed());
    CHECK(row.candidateDraws == 1);
    CHECK(row.drawnDraws == 0);
    CHECK(row.drawnTriangles == 0);
    // Rejected means the view prepares no draw for it, not that the view is absent.
    const PreparedShadowView* view = out.plan.view(ShadowViewGroup::Cascade, 0);
    REQUIRE(view != nullptr);
    CHECK(view->draws().empty());
}

TEST_CASE("a self view prepares two layers and counts one selection", "[ShadowPassPrepare]")
{
    // The self family rasterises ONE logical view into TWO depth images. Both layers' costs are
    // real (each walks the caster set, each is a raster pass), but they share one LOD decision —
    // counting it twice would double a distribution that describes one choice.
    std::vector<DrawCommand> draws{nearCaster()};
    draws[0].selfShadowSlot = 0;
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor({}, {}, draws), views, allFamilies());

    const PreparedShadowView* view = out.plan.view(ShadowViewGroup::Self, 0);
    REQUIRE(view != nullptr);
    REQUIRE(view->layers().size() == 2);
    CHECK(view->layers()[0].kind == ShadowLayerKind::Depth);
    CHECK(view->layers()[1].kind == ShadowLayerKind::SelfSecondDepth);
    CHECK(view->layers()[0].draws.size() == 1);
    CHECK(view->layers()[1].draws.size() == 1);
    // SH-05: the first layer keeps every face (whatever the light sees first), the second keeps
    // only back faces so the dual-depth rejection is well-founded.
    CHECK(view->layers()[0].draws.front().cull == ShadowEffectiveCull::None);
    CHECK(view->layers()[1].draws.front().cull == ShadowEffectiveCull::FrontFaces);

    const ShadowViewStats& row = out.stats.view(ShadowViewGroup::Self, 0);
    CHECK(row.candidateDraws == 2); // one per layer walked
    CHECK(row.drawnDraws == 2);
    std::uint64_t selections = 0;
    for (const std::uint64_t bin : row.lodHistogram)
    {
        selections += bin;
    }
    CHECK(selections == 1); // one decision, however many layers rasterise it
}

TEST_CASE("a self view keeps only the caster holding its slot", "[ShadowPassPrepare]")
{
    std::vector<DrawCommand> draws{nearCaster(1), nearCaster(2)};
    draws[0].selfShadowSlot = 0;
    draws[1].selfShadowSlot = 1;
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor({}, {}, draws), views, allFamilies());

    const PreparedShadowView* view = out.plan.view(ShadowViewGroup::Self, 0);
    REQUIRE(view != nullptr);
    REQUIRE(view->draws().size() == 1);
    CHECK(view->draws().front().casterId == static_cast<ShadowCasterId>(1));
}

TEST_CASE("a point face carries the light its depth is measured against", "[ShadowPassPrepare]")
{
    // The stored depth is `distance / range`, so the light is a shader input rather than a
    // consequence of the face matrix — and it has to arrive in the prepared view EXACTLY, or a
    // moved light would keep a cube whose every texel is wrong.
    const std::vector<DrawCommand> draws{nearCaster()};
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor(draws), views, allFamilies());

    const PreparedShadowView* face =
        out.plan.view(ShadowViewGroup::Point, shadowPointViewSlot(0, 3));
    REQUIRE(face != nullptr);
    CHECK(face->depthMode() == ShadowDepthMode::RadialRatio);
    CHECK(face->lightPosition().x() == kPointPosition.x());
    CHECK(face->lightPosition().y() == kPointPosition.y());
    CHECK(face->lightPosition().z() == kPointPosition.z());
    CHECK(face->lightRange() == kPointRange);

    // Every other family stores projected depth and carries no light at all.
    const PreparedShadowView* cascade = out.plan.view(ShadowViewGroup::Cascade, 0);
    REQUIRE(cascade != nullptr);
    CHECK(cascade->depthMode() == ShadowDepthMode::Projected);
    CHECK(cascade->lightRange() == 0.0f);
}

TEST_CASE("the raster parameters a family is prepared with reach its view", "[ShadowPassPrepare]")
{
    // Extent and depth bias are raster CONTENT: a map rendered at another extent, or with another
    // bias, holds different depth. They travel in the prepared view rather than being re-read from
    // constants at record time, so the comparison sees what the rasteriser will.
    const std::vector<DrawCommand> draws{nearCaster()};
    const ShadowRenderViewSet views = populatedViews();
    ShadowPreparationInputs inputs = inputsFor(draws);
    inputs.raster[static_cast<std::size_t>(ShadowViewGroup::Cascade)] =
        ShadowFamilyRaster{.extent = 2048, .depthBiasConstant = 3.0f, .depthBiasSlope = 4.0f};
    Prepared out{};
    prepare(out, inputs, views, allFamilies());

    const PreparedShadowView* view = out.plan.view(ShadowViewGroup::Cascade, 0);
    REQUIRE(view != nullptr);
    CHECK(view->extent() == 2048);
    CHECK(view->depthBiasConstant() == 3.0f);
    CHECK(view->depthBiasSlope() == 4.0f);
}

TEST_CASE("a prepared draw carries what the rasteriser reads", "[ShadowPassPrepare]")
{
    const std::vector<DrawCommand> draws{nearCaster(7)};
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor(draws), views, allFamilies());

    const PreparedShadowView* view = out.plan.view(ShadowViewGroup::Cascade, 0);
    REQUIRE(view != nullptr);
    REQUIRE(view->draws().size() == 1);
    const PreparedShadowDraw& draw = view->draws().front();
    // The POSE's matrix — the same value the caster's ShadowUBO holds, not a re-derivation.
    CHECK(draw.model == draws.front().shadowRequest.pose.model());
    CHECK(draw.vertexBuffer == draws.front().vertexBuffer);
    CHECK(draw.materialIndex == draws.front().materialIndex);
    // The RESOLVED carrier, not the command's (a shadow command carries none).
    CHECK(draw.indexBuffer == static_cast<BufferHandle>(3));
    CHECK(draw.indexCount == 900);
    // The recording payload: per-frame ring handles, deliberately excluded from the comparison but
    // needed to record at all.
    CHECK(draw.shadowUbo == draws.front().shadowUbo);
    CHECK(draw.skinUbo == draws.front().skinUbo);
    CHECK(draw.morphUbo == draws.front().morphUbo);
    CHECK(draw.morphSsbo == draws.front().morphSsbo);
}

TEST_CASE("a deformable caster poisons its view's reuse", "[ShadowPassPrepare]")
{
    // SH-04's classification IS the cacheability question: skinned, morph-capable and
    // storage-vertex casters rewrite their vertices with no revision the comparison can see.
    const std::vector<DrawCommand> draws{
        caster(1, boundsAt(Vec3{0.0f, 0.0f, 0.0f}), ShadowCasterDeformation::Deformable)};
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor(draws), views, allFamilies());

    const PreparedShadowView* view = out.plan.view(ShadowViewGroup::Cascade, 0);
    REQUIRE(view != nullptr);
    REQUIRE(view->draws().size() == 1);
    CHECK(view->draws().front().deformable);
    CHECK_FALSE(view->cacheable());
}

TEST_CASE("the cascade and its world-only twin share one decision", "[ShadowPassPrepare]")
{
    // They deliberately carry ONE logical identity, so the resolver's frame cache hands the second
    // the first's answer — which is what makes the two maps agree for a rigid caster rather than
    // agreeing by coincidence. They do NOT contain the same casters, which is why content is
    // attributed per family.
    const std::vector<DrawCommand> shadowDraws{nearCaster(1), nearCaster(2)};
    const std::vector<DrawCommand> worldOnlyDraws{shadowDraws.front()};
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor(shadowDraws, worldOnlyDraws), views, allFamilies());

    const ShadowLodStateKey shared{static_cast<ShadowCasterId>(1), ShadowCasterGeneration::First,
                                   ShadowLogicalViewId::cascade(0)};
    const ShadowLodStateKey cascadeOnly{static_cast<ShadowCasterId>(2),
                                        ShadowCasterGeneration::First,
                                        ShadowLogicalViewId::cascade(0)};
    CHECK(out.resolver.contentResolution(ShadowViewGroup::Cascade, shared) != nullptr);
    CHECK(out.resolver.contentResolution(ShadowViewGroup::WorldOnly, shared) != nullptr);
    CHECK(out.resolver.contentResolution(ShadowViewGroup::Cascade, cascadeOnly) != nullptr);
    // The world-only map never held caster 2, so it reports nothing for it — a level from the
    // cascade would be one view's answer presented as another's.
    CHECK(out.resolver.contentResolution(ShadowViewGroup::WorldOnly, cascadeOnly) == nullptr);
}

TEST_CASE("preparing twice into one plan is refused", "[ShadowPassPrepare]")
{
    // `reset()` at the top of preparation is what makes this a fresh frame rather than two frames
    // accumulated into one plan — the second run must land the same claims, not collide with the
    // first's. The STATS are the other half: they are reset per frame by the renderer, so claiming
    // a row twice with the same identity is legal and claiming it with another is not.
    const std::vector<DrawCommand> draws{nearCaster()};
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    prepare(out, inputsFor(draws), views, allFamilies());
    const std::size_t firstCount = out.plan.sampleableCount(ShadowViewGroup::Cascade);

    out.stats.reset();
    prepare(out, inputsFor(draws), views, allFamilies());
    CHECK(out.plan.sampleableCount(ShadowViewGroup::Cascade) == firstCount);
}

TEST_CASE("a caster with no stated pose stops the frame", "[ShadowPassPrepare]")
{
    // The selector survives an unusable transform (InvalidCaster, full detail). The CACHE cannot:
    // an unstated pose is a default matrix, identical every frame, so the comparison would find the
    // view unchanged while the GPU rasterised the caster's real transform — a map reused forever
    // for something that is moving, with no symptom anywhere.
    std::vector<DrawCommand> draws{nearCaster()};
    draws[0].shadowRequest.pose = ShadowCasterPose{};
    const ShadowRenderViewSet views = populatedViews();
    Prepared out{};
    out.resolver.beginFrame();
    CHECK_THROWS(prepareShadowFrame(inputsFor(draws), views, allFamilies(), out.resolver, out.stats,
                                    out.plan));
}
