#include <fire_engine/graphics/shadow_pass_plan.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace fire_engine;

namespace
{

constexpr auto kCaster = static_cast<ShadowCasterId>(7);
constexpr auto kOtherCaster = static_cast<ShadowCasterId>(8);

Mat4 translation(float x)
{
    return Mat4::translate(Vec3{x, 0.0f, 0.0f});
}

PreparedShadowDraw rigidDraw()
{
    return PreparedShadowDraw{
        .casterId = kCaster,
        .generation = ShadowCasterGeneration::First,
        .model = translation(1.0f),
        .vertexBuffer = static_cast<BufferHandle>(11),
        .indexBuffer = static_cast<BufferHandle>(12),
        .indexCount = 300,
        .indexType = DrawIndexType::UInt16,
        .alpha = ShadowCasterAlpha::Opaque,
        .materialIndex = 3,
        .cull = ShadowEffectiveCull::FrontFaces,
        .deformable = false,
        .level = 1,
        .reason = ShadowLodReason::Selected,
    };
}

PreparedShadowView cascadeView()
{
    return PreparedShadowView::projected(ShadowLogicalViewId::cascade(1), translation(4.0f), 2048,
                                         0.0f, 2.0f);
}

PreparedShadowView viewWith(const PreparedShadowDraw& draw)
{
    PreparedShadowView view = cascadeView();
    REQUIRE(view.addDraw(draw));
    return view;
}

ShadowViewResidency residentFrom(const PreparedShadowView& view)
{
    ShadowViewResidency residency{};
    residency.commit(view);
    return residency;
}

// A point face: the depth it stores is a distance/range ratio taken against the light, so the light
// itself is part of the content.
PreparedShadowView pointFace(Vec3 lightPosition = Vec3{2.0f, 3.0f, 4.0f}, float range = 25.0f)
{
    return PreparedShadowView::pointFace(ShadowLogicalViewId::point(static_cast<NodeId>(1), 3),
                                         translation(4.0f), 1024, 0.0f, 2.0f, lightPosition, range);
}

PreparedShadowView pointFaceWith(const PreparedShadowDraw& draw,
                                 Vec3 lightPosition = Vec3{2.0f, 3.0f, 4.0f}, float range = 25.0f)
{
    PreparedShadowView view = pointFace(lightPosition, range);
    REQUIRE(view.addDraw(draw));
    return view;
}

} // namespace

TEST_CASE("identical prepared content reuses the resident map", "[ShadowPassPlan]")
{
    const PreparedShadowView prepared = viewWith(rigidDraw());
    const ShadowViewResidency resident = residentFrom(prepared);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, prepared, resident) ==
          ShadowViewDisposition::Reused);
    CHECK(shadowViewSampleable(ShadowViewDisposition::Reused));
    CHECK_FALSE(shadowViewRecords(ShadowViewDisposition::Reused));
}

TEST_CASE("an inactive view is invalid whatever its image holds", "[ShadowPassPlan]")
{
    const PreparedShadowView prepared = viewWith(rigidDraw());
    const ShadowViewResidency resident = residentFrom(prepared);
    // The view set is the authority on engagement (SH-03). Content that matches is irrelevant if
    // nothing this frame vouches for the matrix behind it.
    const ShadowViewDisposition disposition =
        shadowViewDisposition(false, ShadowReusePolicy::Enabled, prepared, resident);
    CHECK(disposition == ShadowViewDisposition::Invalid);
    CHECK_FALSE(shadowViewSampleable(disposition));
    CHECK_FALSE(shadowViewRecords(disposition));
}

TEST_CASE("first use must record even with matching content", "[ShadowPassPlan]")
{
    // Creation transitions the image to the read-only layout but leaves its depth undefined, so an
    // uncommitted slot has no answer to reuse — the one case where the right layout is not enough.
    const PreparedShadowView prepared = viewWith(rigidDraw());
    const ShadowViewResidency empty{};
    CHECK_FALSE(empty.hasContent());
    CHECK(empty.content() == nullptr);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, prepared, empty) ==
          ShadowViewDisposition::Recorded);

    // And once it IS committed, the same prepared work reuses.
    const ShadowViewResidency resident = residentFrom(prepared);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, prepared, resident) ==
          ShadowViewDisposition::Reused);

    // RECREATION is modelled the only way the engine can express it: a fresh store, because the
    // store lives with the images and is rebuilt when they are. There is no invalidate() to call —
    // a hook would be a second way for the record and the images to disagree, and the reason this
    // type has none is that the disagreement is what produces a wrong picture.
    const ShadowResidencyStore recreated{};
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, prepared,
                                recreated.at(ShadowViewGroup::Cascade, 1)) ==
          ShadowViewDisposition::Recorded);
}

TEST_CASE("a point face's light position and range are part of its content", "[ShadowPassPlan]")
{
    // The depth a point face stores is `length(worldPos - lightPos) / range`, written to
    // gl_FragDepth from the push constants — NOT a consequence of the face matrix. A light that
    // moves or re-ranges with an unchanged matrix changes every texel, so an equal viewProj is not
    // enough to reuse.
    const PreparedShadowView resident = pointFaceWith(rigidDraw());
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, pointFaceWith(rigidDraw()),
                                residentFrom(resident)) == ShadowViewDisposition::Reused);

    const PreparedShadowView moved = pointFaceWith(rigidDraw(), Vec3{2.0f, 3.0f, 4.01f});
    CHECK(moved.viewProj() == resident.viewProj()); // the hole this closes: matrices agree
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, moved, residentFrom(resident)) ==
          ShadowViewDisposition::Recorded);

    const PreparedShadowView reranged = pointFaceWith(rigidDraw(), Vec3{2.0f, 3.0f, 4.0f}, 26.0f);
    CHECK(reranged.viewProj() == resident.viewProj());
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, reranged,
                                residentFrom(resident)) == ShadowViewDisposition::Recorded);
}

TEST_CASE("the depth mode follows the identity and cannot be set against it", "[ShadowPassPlan]")
{
    // Projected hardware depth and a radial ratio are different numbers in the same texels, so a
    // point face prepared as projected would compare without its light while the shader still took
    // the radial branch. That state is unrepresentable: the mode is derived from the identity, and
    // each factory refuses the identity it cannot serve.
    CHECK(cascadeView().depthMode() == ShadowDepthMode::Projected);
    CHECK(pointFace().depthMode() == ShadowDepthMode::RadialRatio);

    const PreparedShadowView pointAsProjected = PreparedShadowView::projected(
        ShadowLogicalViewId::point(static_cast<NodeId>(1), 3), Mat4::identity(), 1024, 0.0f, 0.0f);
    CHECK_FALSE(pointAsProjected.valid());

    const PreparedShadowView cascadeAsPoint = PreparedShadowView::pointFace(
        ShadowLogicalViewId::cascade(0), Mat4::identity(), 2048, 0.0f, 0.0f, Vec3{}, 10.0f);
    CHECK_FALSE(cascadeAsPoint.valid());

    // And a default-constructed view is invalid rather than "cascade 0 with no draws".
    CHECK_FALSE(PreparedShadowView{}.valid());
    CHECK(cascadeView().valid());
}

TEST_CASE("a projected view carries no light to compare", "[ShadowPassPlan]")
{
    // Cascade, spot and self views write fixed-function depth and carry no light position in their
    // push constants — there is no setter that could give one a stray light, and the comparison
    // would ignore it anyway.
    const PreparedShadowView view = cascadeView();
    CHECK(view.lightRange() == 0.0f);
    CHECK(view.lightPosition().x() == 0.0f);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(rigidDraw()),
                                residentFrom(viewWith(rigidDraw()))) ==
          ShadowViewDisposition::Reused);
}

TEST_CASE("a moved caster records even though its buffers and bounds are unchanged",
          "[ShadowPassPlan]")
{
    // THE case the review flagged: bounds and geometry identity say nothing about the matrix, and
    // two transforms can share an AABB while rasterising different pixels.
    const PreparedShadowView resident = viewWith(rigidDraw());
    PreparedShadowDraw moved = rigidDraw();
    moved.model = translation(1.5f);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(moved),
                                residentFrom(resident)) == ShadowViewDisposition::Recorded);
}

TEST_CASE("a re-fitted view records — the matrix is compared, not the fit that explains it",
          "[ShadowPassPlan]")
{
    const PreparedShadowView resident = viewWith(rigidDraw());
    PreparedShadowView refitted = PreparedShadowView::projected(
        ShadowLogicalViewId::cascade(1), translation(4.5f), 2048, 0.0f, 2.0f);
    REQUIRE(refitted.addDraw(rigidDraw()));
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, refitted,
                                residentFrom(resident)) == ShadowViewDisposition::Recorded);
}

TEST_CASE("a swapped LOD carrier records even at the same level", "[ShadowPassPlan]")
{
    // The resolved carrier is the geometry; the level only names the decision. A chain rebuilt
    // behind the same level number is different pixels.
    const PreparedShadowView resident = viewWith(rigidDraw());
    PreparedShadowDraw swapped = rigidDraw();
    swapped.indexBuffer = static_cast<BufferHandle>(99);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(swapped),
                                residentFrom(resident)) == ShadowViewDisposition::Recorded);

    PreparedShadowDraw coarser = rigidDraw();
    coarser.indexCount = 150;
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(coarser),
                                residentFrom(resident)) == ShadowViewDisposition::Recorded);
}

TEST_CASE("the level and reason are diagnostics and do not force a re-record", "[ShadowPassPlan]")
{
    // Two levels resolving to one carrier is genuinely the same image. Comparing the level would
    // cost a re-record for no pixel difference — correctness-neutral, but a real loss.
    const PreparedShadowView resident = viewWith(rigidDraw());
    PreparedShadowDraw relabelled = rigidDraw();
    relabelled.level = 2;
    relabelled.reason = ShadowLodReason::SingleLevel;
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(relabelled),
                                residentFrom(resident)) == ShadowViewDisposition::Reused);
}

TEST_CASE("every pixel-producing field forces a re-record when it changes", "[ShadowPassPlan]")
{
    const PreparedShadowView resident = viewWith(rigidDraw());
    const auto records = [&](const PreparedShadowDraw& draw)
    {
        return shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(draw),
                                     residentFrom(resident)) == ShadowViewDisposition::Recorded;
    };

    PreparedShadowDraw d = rigidDraw();
    d.casterId = kOtherCaster;
    CHECK(records(d));

    d = rigidDraw();
    d.generation = nextShadowCasterGeneration(ShadowCasterGeneration::First);
    CHECK(records(d));

    d = rigidDraw();
    d.vertexBuffer = static_cast<BufferHandle>(77);
    CHECK(records(d));

    d = rigidDraw();
    d.indexType = DrawIndexType::UInt32;
    CHECK(records(d));

    d = rigidDraw();
    d.alpha = ShadowCasterAlpha::Masked;
    CHECK(records(d));

    d = rigidDraw();
    d.cull = ShadowEffectiveCull::None;
    CHECK(records(d));
}

TEST_CASE("the material index is content for a masked caster only", "[ShadowPassPlan]")
{
    // The masked fragment path samples the material to decide what it discards, so its index is a
    // raster input there. The opaque path reads no material data at all — two opaque variants of
    // one mesh differing only in material store identical depth and must reuse.
    PreparedShadowDraw masked = rigidDraw();
    masked.alpha = ShadowCasterAlpha::Masked;
    PreparedShadowDraw maskedOther = masked;
    maskedOther.materialIndex = masked.materialIndex + 1;
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(maskedOther),
                                residentFrom(viewWith(masked))) == ShadowViewDisposition::Recorded);

    PreparedShadowDraw opaqueOther = rigidDraw();
    opaqueOther.materialIndex = rigidDraw().materialIndex + 1;
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(opaqueOther),
                                residentFrom(viewWith(rigidDraw()))) ==
          ShadowViewDisposition::Reused);
}

TEST_CASE("per-view raster state changes force a re-record", "[ShadowPassPlan]")
{
    const PreparedShadowView resident = viewWith(rigidDraw());
    const auto records = [&](const PreparedShadowView& view)
    {
        return shadowViewDisposition(true, ShadowReusePolicy::Enabled, view,
                                     residentFrom(resident)) == ShadowViewDisposition::Recorded;
    };

    const auto cascadeVariant =
        [](std::uint32_t extent, float biasConstant, float biasSlope, std::uint32_t cascadeIndex)
    {
        PreparedShadowView view =
            PreparedShadowView::projected(ShadowLogicalViewId::cascade(cascadeIndex),
                                          translation(4.0f), extent, biasConstant, biasSlope);
        REQUIRE(view.addDraw(rigidDraw()));
        return view;
    };

    CHECK(records(cascadeVariant(1024, 0.0f, 2.0f, 1)));
    CHECK(records(cascadeVariant(2048, 1.0f, 2.0f, 1)));
    CHECK(records(cascadeVariant(2048, 0.0f, 3.0f, 1)));
    // A physical slot is reassigned between frames, so identity is part of the content: without it,
    // one light's resident depth could be matched against another light's prepared work.
    CHECK(records(cascadeVariant(2048, 0.0f, 2.0f, 2)));
    // The unchanged variant is the control: the four above differ in exactly one field each.
    CHECK_FALSE(records(cascadeVariant(2048, 0.0f, 2.0f, 1)));
}

TEST_CASE("a deformable caster poisons the whole view, in both directions", "[ShadowPassPlan]")
{
    // A skinned caster rewrites its vertices with the same buffers and the same matrix, so nothing
    // in the descriptor can see the change. Until arc 2 #5 supplies a deformation revision, the
    // honest answer is that such a view is never cacheable — and neither is content recorded while
    // it held one, since that content describes geometry that has since moved.
    PreparedShadowDraw deforming = rigidDraw();
    deforming.deformable = true;

    const PreparedShadowView withDeformable = viewWith(deforming);
    CHECK_FALSE(withDeformable.cacheable());
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, withDeformable,
                                residentFrom(withDeformable)) == ShadowViewDisposition::Recorded);

    // Prepared is rigid now, but the resident content was captured with a deformable in the set.
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, viewWith(rigidDraw()),
                                residentFrom(withDeformable)) == ShadowViewDisposition::Recorded);

    CHECK(viewWith(rigidDraw()).cacheable());
}

TEST_CASE("a changed draw set forces a re-record", "[ShadowPassPlan]")
{
    const PreparedShadowView resident = viewWith(rigidDraw());

    PreparedShadowView extra = viewWith(rigidDraw());
    PreparedShadowDraw second = rigidDraw();
    second.casterId = kOtherCaster;
    REQUIRE(extra.addDraw(second));
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, extra, residentFrom(resident)) ==
          ShadowViewDisposition::Recorded);

    // Same view, no draws: a caster that left the frame.
    const PreparedShadowView empty = cascadeView();
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, empty, residentFrom(resident)) ==
          ShadowViewDisposition::Recorded);

    // An empty view is cacheable and reusable against empty resident content: a cleared map that
    // stays cleared is a legitimate answer, and it is the case that makes an unchanged empty
    // cascade free rather than a clear plus two barriers every frame.
    CHECK(empty.cacheable());
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, empty, residentFrom(empty)) ==
          ShadowViewDisposition::Reused);
}

TEST_CASE("draw order is compared, conservatively", "[ShadowPassPlan]")
{
    // Depth-only output is order-independent, so a reorder is a FALSE miss — one wasted re-record,
    // never a wrong image. That trade is deliberate: order-insensitive comparison would mean
    // sorting or hashing every frame to save a case the stable gather order does not produce.
    PreparedShadowDraw first = rigidDraw();
    PreparedShadowDraw second = rigidDraw();
    second.casterId = kOtherCaster;

    PreparedShadowView forwards = viewWith(first);
    REQUIRE(forwards.addDraw(second));
    PreparedShadowView backwards = viewWith(second);
    REQUIRE(backwards.addDraw(first));

    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, backwards,
                                residentFrom(forwards)) == ShadowViewDisposition::Recorded);
}

TEST_CASE("disposition names are distinct and non-empty", "[ShadowPassPlan]")
{
    CHECK(toString(ShadowViewDisposition::Invalid) == "invalid");
    CHECK(toString(ShadowViewDisposition::Reused) == "reused");
    CHECK(toString(ShadowViewDisposition::Recorded) == "recorded");
}

TEST_CASE("the plan refuses an invalid prepared view before it can be dispositioned",
          "[ShadowPassPlan]")
{
    // A factory handed a mismatched identity yields an invalid view. Admitting one would record
    // from a matrix nobody vouched for and cache content that describes nothing, so the plan
    // rejects it and the caller must treat that as terminal — the same discipline as a rejected
    // view-set write.
    ShadowFramePlan plan{};
    const PreparedShadowView bad = PreparedShadowView::projected(
        ShadowLogicalViewId::point(static_cast<NodeId>(1), 0), Mat4::identity(), 1024, 0.0f, 0.0f);
    CHECK_FALSE(plan.add(ShadowViewGroup::Cascade, 0, bad, ShadowViewDisposition::Recorded));
    CHECK(plan.view(ShadowViewGroup::Cascade, 0) == nullptr);
    CHECK(plan.disposition(ShadowViewGroup::Cascade, 0) == ShadowViewDisposition::Invalid);

    // Out-of-range slots are refused the same way, rather than wrapping into another family's row.
    CHECK_FALSE(plan.add(ShadowViewGroup::Cascade, shadowViewSlotCount(ShadowViewGroup::Cascade),
                         cascadeView(), ShadowViewDisposition::Recorded));
}

TEST_CASE("a mixed CSM stays family-valid while timing only the recorded subset",
          "[ShadowPassPlan]")
{
    // THE case caching exists to produce: a camera that moved slightly re-fits the near cascades
    // while the far ones are untouched. All four are sampleable, so the family is valid and the
    // shader keeps sampling every layer; only the two that changed cost anything.
    ShadowFramePlan plan{};
    const std::size_t cascades = shadowViewSlotCount(ShadowViewGroup::Cascade);
    for (std::size_t slot = 0; slot < cascades; ++slot)
    {
        PreparedShadowView view = PreparedShadowView::projected(
            ShadowLogicalViewId::cascade(static_cast<std::uint32_t>(slot)), Mat4::identity(), 2048,
            0.0f, 2.0f);
        REQUIRE(view.addDraw(rigidDraw()));
        const ShadowViewDisposition disposition =
            slot < 2 ? ShadowViewDisposition::Recorded : ShadowViewDisposition::Reused;
        REQUIRE(plan.add(ShadowViewGroup::Cascade, slot, view, disposition));
    }

    CHECK(plan.sampleableCount(ShadowViewGroup::Cascade) == cascades);
    CHECK(plan.records(ShadowViewGroup::Cascade));
    CHECK_FALSE(plan.recordsNothing());
    // The families nobody prepared are neither sampleable nor timed.
    CHECK(plan.sampleableCount(ShadowViewGroup::Spot) == 0);
    CHECK_FALSE(plan.records(ShadowViewGroup::Spot));
}

TEST_CASE("a fully reused frame is sampleable and records nothing", "[ShadowPassPlan]")
{
    ShadowFramePlan plan{};
    for (std::size_t slot = 0; slot < shadowViewSlotCount(ShadowViewGroup::Cascade); ++slot)
    {
        PreparedShadowView view = PreparedShadowView::projected(
            ShadowLogicalViewId::cascade(static_cast<std::uint32_t>(slot)), Mat4::identity(), 2048,
            0.0f, 2.0f);
        REQUIRE(plan.add(ShadowViewGroup::Cascade, slot, view, ShadowViewDisposition::Reused));
    }
    CHECK(plan.sampleableCount(ShadowViewGroup::Cascade) ==
          shadowViewSlotCount(ShadowViewGroup::Cascade));
    CHECK_FALSE(plan.records(ShadowViewGroup::Cascade));
    CHECK(plan.recordsNothing()); // the pass returns without a single bracket

    // An Invalid entry keeps no content: "engaged but unusable" must not read as a description of
    // the image.
    ShadowFramePlan invalidated{};
    REQUIRE(invalidated.add(ShadowViewGroup::Cascade, 1, viewWith(rigidDraw()),
                            ShadowViewDisposition::Invalid));
    CHECK(invalidated.view(ShadowViewGroup::Cascade, 1) == nullptr);
    CHECK(invalidated.sampleableCount(ShadowViewGroup::Cascade) == 0);
    CHECK(invalidated.recordsNothing());
}

TEST_CASE("both self-shadow layers are content", "[ShadowPassPlan]")
{
    // A self view rasterises two depth images with different fragment paths and different cull. A
    // comparison that stopped at the first layer would reuse a view whose second had changed.
    const auto selfView = [](ShadowEffectiveCull secondCull)
    {
        PreparedShadowView view = PreparedShadowView::projected(
            ShadowLogicalViewId::self(42), translation(2.0f), 1024, 0.0f, 0.0f);
        // Both layers exist already — a self identity is born with its dual-depth pair.
        PreparedShadowDraw firstDraw = rigidDraw();
        firstDraw.cull = ShadowEffectiveCull::None;
        REQUIRE(view.addDraw(ShadowLayerKind::Depth, firstDraw));

        PreparedShadowDraw secondDraw = rigidDraw();
        secondDraw.cull = secondCull;
        REQUIRE(view.addDraw(ShadowLayerKind::SelfSecondDepth, secondDraw));
        return view;
    };

    const PreparedShadowView resident = selfView(ShadowEffectiveCull::FrontFaces);
    CHECK(resident.layers().size() == 2);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled,
                                selfView(ShadowEffectiveCull::FrontFaces),
                                residentFrom(resident)) == ShadowViewDisposition::Reused);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled,
                                selfView(ShadowEffectiveCull::None),
                                residentFrom(resident)) == ShadowViewDisposition::Recorded);
}

TEST_CASE("a view is born with its layers, so an empty one still clears", "[ShadowPassPlan]")
{
    // A first-use empty cascade is Recorded and sampleable. If its layers only appeared when a draw
    // did, the recorder would walk nothing, clear nothing, and leave undefined depth behind a map
    // the plan called usable — so topology comes from the identity, not from what happened to be
    // appended.
    const PreparedShadowView emptyCascade = cascadeView();
    REQUIRE(emptyCascade.layers().size() == 1);
    CHECK(emptyCascade.layers().front().kind == ShadowLayerKind::Depth);
    CHECK(emptyCascade.layers().front().draws.empty());

    const PreparedShadowView emptySelf = PreparedShadowView::projected(
        ShadowLogicalViewId::self(7), Mat4::identity(), 1024, 0.0f, 0.0f);
    REQUIRE(emptySelf.layers().size() == 2);
    CHECK(emptySelf.layers()[0].kind == ShadowLayerKind::Depth);
    CHECK(emptySelf.layers()[1].kind == ShadowLayerKind::SelfSecondDepth);

    // And a layer a view does not have cannot be drawn into: asking a cascade for a second
    // self-shadow depth is a producer bug, not a draw to drop.
    PreparedShadowView cascade = cascadeView();
    CHECK_FALSE(cascade.addDraw(ShadowLayerKind::SelfSecondDepth, rigidDraw()));
    CHECK(cascade.addDraw(ShadowLayerKind::Depth, rigidDraw()));
}

TEST_CASE("a slot may be claimed once per frame", "[ShadowPassPlan]")
{
    // Two producers preparing one physical view means two views are being prepared as one: the plan
    // would keep the last writer's work under the other's identity. Refused, like a duplicate
    // view-set write or a duplicate caster-bounds key.
    ShadowFramePlan plan{};
    REQUIRE(plan.add(ShadowViewGroup::Cascade, 1, viewWith(rigidDraw()),
                     ShadowViewDisposition::Recorded));
    CHECK_FALSE(plan.add(ShadowViewGroup::Cascade, 1, viewWith(rigidDraw()),
                         ShadowViewDisposition::Reused));
    CHECK(plan.disposition(ShadowViewGroup::Cascade, 1) == ShadowViewDisposition::Recorded);

    // An INVALID claim is still a claim — otherwise a second producer could overwrite it and the
    // duplicate would go unreported.
    ShadowFramePlan invalidFirst{};
    const PreparedShadowView spot = PreparedShadowView::projected(
        ShadowLogicalViewId::spot(static_cast<NodeId>(61)), translation(1.0f), 1024, 0.0f, 1.0f);
    REQUIRE(invalidFirst.add(ShadowViewGroup::Spot, 1, spot, ShadowViewDisposition::Invalid));
    CHECK_FALSE(invalidFirst.add(ShadowViewGroup::Spot, 1, spot, ShadowViewDisposition::Recorded));

    // reset() releases every claim.
    plan.reset();
    CHECK(plan.add(ShadowViewGroup::Cascade, 1, viewWith(rigidDraw()),
                   ShadowViewDisposition::Recorded));
}

namespace
{

ShadowFamilyEligibility eligibilityFor(ShadowViewGroup group, std::size_t activeViews)
{
    ShadowFamilyEligibility eligibility{};
    eligibility.primaryDirectionalLight = true;
    eligibility.activeViews[static_cast<std::size_t>(group)] = activeViews;
    return eligibility;
}

PreparedShadowView spotView(std::size_t slot, std::uint64_t lightId)
{
    return PreparedShadowView::projected(ShadowLogicalViewId::spot(static_cast<NodeId>(lightId)),
                                         translation(static_cast<float>(slot)), 1024, 0.0f, 1.0f);
}

PreparedShadowView pointFaceView(std::uint8_t face, std::uint64_t lightId)
{
    return PreparedShadowView::pointFace(
        ShadowLogicalViewId::point(static_cast<NodeId>(lightId), face), translation(1.0f), 1024,
        0.0f, 1.0f, Vec3{1.0f, 2.0f, 3.0f}, 20.0f);
}

} // namespace

TEST_CASE("confirmation requires every eligible view, not a plausible count", "[ShadowPassPlan]")
{
    // TWO SPOTS, ONE PREPARES. The eligibility law alone is satisfied by "any active slot", so the
    // family would stay valid and the light that failed to prepare would sample whatever its map
    // held last. Completeness against the EXPECTED count is what catches it.
    ShadowFramePlan partial{};
    REQUIRE(
        partial.add(ShadowViewGroup::Spot, 0, spotView(0, 11), ShadowViewDisposition::Recorded));
    const ShadowFamilyEligibility twoSpots = eligibilityFor(ShadowViewGroup::Spot, 2);
    CHECK(twoSpots.eligible().spot); // eligible: two active spots
    CHECK_FALSE(shadowMapValidityFromPlan(partial, twoSpots).spot);

    // Both prepared — including one REUSED, which counts as arrived because it is sampleable.
    ShadowFramePlan both{};
    REQUIRE(both.add(ShadowViewGroup::Spot, 0, spotView(0, 11), ShadowViewDisposition::Recorded));
    REQUIRE(both.add(ShadowViewGroup::Spot, 1, spotView(1, 12), ShadowViewDisposition::Reused));
    CHECK(shadowMapValidityFromPlan(both, twoSpots).spot);
}

TEST_CASE("twelve eligible point faces are not confirmed by six", "[ShadowPassPlan]")
{
    // Six of twelve is a whole number of cubes by arithmetic, so the modulo rule passes; the second
    // light is simply missing. Completeness catches the count, and pointCubesWhole catches the
    // shape — a half-prepared cube beside a whole one, which no count can see.
    const auto faces = static_cast<std::size_t>(kCubeFaceCount);
    const ShadowFamilyEligibility twoCubes = eligibilityFor(ShadowViewGroup::Point, 2 * faces);
    CHECK(twoCubes.eligible().point);

    ShadowFramePlan oneCube{};
    for (std::size_t face = 0; face < faces; ++face)
    {
        REQUIRE(oneCube.add(ShadowViewGroup::Point, face,
                            pointFaceView(static_cast<std::uint8_t>(face), 21),
                            ShadowViewDisposition::Recorded));
    }
    CHECK(oneCube.sampleableCount(ShadowViewGroup::Point) == faces); // a "whole number of cubes"
    CHECK(oneCube.pointCubesWhole());
    CHECK_FALSE(shadowMapValidityFromPlan(oneCube, twoCubes).point);

    // A half-prepared cube fails the shape check even when the total count matches.
    ShadowFramePlan ragged{};
    for (std::size_t face = 0; face < faces; ++face)
    {
        const bool firstCube = face < faces / 2;
        const std::size_t slot = firstCube ? face : faces + face;
        const std::uint8_t logicalFace = static_cast<std::uint8_t>(slot % faces);
        REQUIRE(ragged.add(ShadowViewGroup::Point, slot,
                           pointFaceView(logicalFace, firstCube ? 21 : 22),
                           ShadowViewDisposition::Recorded));
    }
    CHECK(ragged.sampleableCount(ShadowViewGroup::Point) == faces);
    CHECK_FALSE(ragged.pointCubesWhole());
    CHECK_FALSE(shadowMapValidityFromPlan(ragged, twoCubes).point);
}

TEST_CASE("the plan refuses a view in the wrong physical slot", "[ShadowPassPlan]")
{
    // The recorder trusts this object exclusively — there is no view set left to cross-check
    // against — so a misplaced identity has to be refused here. Each of these renders plausibly and
    // reports plausibly while drawing the wrong thing into the wrong image.
    ShadowFramePlan plan{};

    // A cascade identity in a spot slot.
    CHECK_FALSE(plan.add(ShadowViewGroup::Spot, 0, cascadeView(), ShadowViewDisposition::Recorded));
    // A cascade whose index is not its slot: it would rasterise cascade 1's matrix into layer 2.
    CHECK_FALSE(
        plan.add(ShadowViewGroup::Cascade, 2, cascadeView(), ShadowViewDisposition::Recorded));
    // World-only shares the cascade identity, and the same index rule.
    CHECK(plan.add(ShadowViewGroup::WorldOnly, 1, cascadeView(), ShadowViewDisposition::Recorded));
    // A point face in the wrong face slot of the right cube.
    CHECK_FALSE(
        plan.add(ShadowViewGroup::Point, 0, pointFaceView(3, 31), ShadowViewDisposition::Recorded));
    // A spot identity in a self slot.
    CHECK_FALSE(
        plan.add(ShadowViewGroup::Self, 0, spotView(0, 41), ShadowViewDisposition::Recorded));
}

TEST_CASE("one physical point cube belongs to one light", "[ShadowPassPlan]")
{
    // Two lights sharing a cube would each render half of it and both sample all of it. The view
    // set's atomic installation prevents it upstream; the plan is a second producer of the same
    // arrangement, so it checks rather than assuming.
    ShadowFramePlan plan{};
    REQUIRE(
        plan.add(ShadowViewGroup::Point, 0, pointFaceView(0, 51), ShadowViewDisposition::Recorded));
    CHECK(
        plan.add(ShadowViewGroup::Point, 1, pointFaceView(1, 51), ShadowViewDisposition::Recorded));
    CHECK_FALSE(
        plan.add(ShadowViewGroup::Point, 2, pointFaceView(2, 52), ShadowViewDisposition::Recorded));
    // A different cube may of course hold a different light.
    CHECK(plan.add(ShadowViewGroup::Point, kCubeFaceCount, pointFaceView(0, 52),
                   ShadowViewDisposition::Recorded));
}

TEST_CASE("a suppressed frame confirms nothing", "[ShadowPassPlan]")
{
    // --no-shadows: nothing eligible, so preparation produces no rows, and confirmation agrees.
    ShadowFamilyEligibility suppressed{};
    suppressed.shadowsDisabled = true;
    suppressed.primaryDirectionalLight = true;
    suppressed.activeViews[static_cast<std::size_t>(ShadowViewGroup::Cascade)] =
        shadowViewSlotCount(ShadowViewGroup::Cascade);
    CHECK(suppressed.eligible().none());

    const ShadowFramePlan empty{};
    CHECK(shadowMapValidityFromPlan(empty, suppressed).none());
    CHECK(empty.recordsNothing());
}

// --- The residency store: what the images HOLD, between frames -------------------------------

namespace
{

// A cascade view whose identity MATCHES the slot it will be added at — the plan refuses any other
// pairing, and residency is addressed by that same physical slot.
PreparedShadowView cascadeViewAt(std::size_t slot, const PreparedShadowDraw& draw)
{
    PreparedShadowView view = PreparedShadowView::projected(
        ShadowLogicalViewId::cascade(static_cast<std::uint32_t>(slot)), translation(4.0f), 2048,
        0.0f, 2.0f);
    REQUIRE(view.addDraw(draw));
    return view;
}

PreparedShadowView spotViewWith(std::size_t slot, std::uint64_t lightId,
                                const PreparedShadowDraw& draw)
{
    PreparedShadowView view = spotView(slot, lightId);
    REQUIRE(view.addDraw(draw));
    return view;
}

} // namespace

TEST_CASE("an empty store records everything, because an image holds no answer yet",
          "[ShadowPassPlan]")
{
    const ShadowResidencyStore store{};
    // Creation transitions a depth image's layout but writes no depth, so "nothing resident" is the
    // honest starting state and the law's answer to it is to draw.
    CHECK_FALSE(store.at(ShadowViewGroup::Cascade, 0).hasContent());
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, cascadeViewAt(0, rigidDraw()),
                                store.at(ShadowViewGroup::Cascade, 0)) ==
          ShadowViewDisposition::Recorded);
}

TEST_CASE("committing a recorded frame is what makes the next identical one reuse",
          "[ShadowPassPlan]")
{
    ShadowResidencyStore store{};
    ShadowFramePlan plan{};
    REQUIRE(plan.add(ShadowViewGroup::Cascade, 2, cascadeViewAt(2, rigidDraw()),
                     ShadowViewDisposition::Recorded));
    store.commit(plan);

    REQUIRE(store.at(ShadowViewGroup::Cascade, 2).hasContent());
    // The SECOND frame's identical preparation, judged against what the first one left behind.
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, cascadeViewAt(2, rigidDraw()),
                                store.at(ShadowViewGroup::Cascade, 2)) ==
          ShadowViewDisposition::Reused);
    // Per slot, not per family: nothing was committed for any other cascade, so each still draws.
    CHECK_FALSE(store.at(ShadowViewGroup::Cascade, 1).hasContent());
}

TEST_CASE("a moved caster is not the content that was committed", "[ShadowPassPlan]")
{
    ShadowResidencyStore store{};
    ShadowFramePlan plan{};
    REQUIRE(plan.add(ShadowViewGroup::Cascade, 0, cascadeViewAt(0, rigidDraw()),
                     ShadowViewDisposition::Recorded));
    store.commit(plan);

    PreparedShadowDraw moved = rigidDraw();
    moved.model = translation(9.0f);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, cascadeViewAt(0, moved),
                                store.at(ShadowViewGroup::Cascade, 0)) ==
          ShadowViewDisposition::Recorded);
}

TEST_CASE("only a recorded view commits, so residency keeps describing the recording",
          "[ShadowPassPlan]")
{
    ShadowResidencyStore store{};

    // Frame 1 RECORDS a caster resolved at level 1.
    ShadowFramePlan first{};
    REQUIRE(first.add(ShadowViewGroup::Cascade, 0, cascadeViewAt(0, rigidDraw()),
                      ShadowViewDisposition::Recorded));
    store.commit(first);
    REQUIRE(store.at(ShadowViewGroup::Cascade, 0).content() != nullptr);
    REQUIRE(store.at(ShadowViewGroup::Cascade, 0).content()->draws().front().level == 1);

    // Frame 2 REUSES it. The prepared view is identical in every compared field — that is why it
    // was reused — but its diagnostic fields describe a decision this image's pixels never saw.
    // Committing it would leave the record describing a frame that wrote nothing.
    PreparedShadowDraw reResolved = rigidDraw();
    reResolved.level = 3;
    reResolved.reason = ShadowLodReason::SingleLevel;
    ShadowFramePlan second{};
    REQUIRE(second.add(ShadowViewGroup::Cascade, 0, cascadeViewAt(0, reResolved),
                       ShadowViewDisposition::Reused));
    store.commit(second);

    REQUIRE(store.at(ShadowViewGroup::Cascade, 0).content() != nullptr);
    CHECK(store.at(ShadowViewGroup::Cascade, 0).content()->draws().front().level == 1);
}

TEST_CASE("an inactive slot keeps its residency, because nothing overwrote its image",
          "[ShadowPassPlan]")
{
    ShadowResidencyStore store{};
    ShadowFramePlan first{};
    REQUIRE(first.add(ShadowViewGroup::Spot, 0, spotViewWith(0, 61, rigidDraw()),
                      ShadowViewDisposition::Recorded));
    store.commit(first);
    REQUIRE(store.at(ShadowViewGroup::Spot, 0).hasContent());

    // The light goes away for a frame: the slot is claimed Invalid, nothing records, and the depth
    // image is not touched. Clearing the record here would force a re-render of content the image
    // demonstrably still holds.
    ShadowFramePlan second{};
    REQUIRE(second.add(ShadowViewGroup::Spot, 0, spotViewWith(0, 61, rigidDraw()),
                       ShadowViewDisposition::Invalid));
    store.commit(second);

    CHECK(store.at(ShadowViewGroup::Spot, 0).hasContent());
}

TEST_CASE("an out-of-range address holds nothing rather than a neighbour's content",
          "[ShadowPassPlan]")
{
    ShadowResidencyStore store{};
    ShadowFramePlan plan{};
    const std::size_t last = shadowViewSlotCount(ShadowViewGroup::Cascade) - 1;
    REQUIRE(plan.add(ShadowViewGroup::Cascade, last, cascadeViewAt(last, rigidDraw()),
                     ShadowViewDisposition::Recorded));
    store.commit(plan);

    // Never clamped into the last valid slot: that would answer one view's question with another
    // view's image, and the answer would be "reuse".
    CHECK_FALSE(store.at(ShadowViewGroup::Cascade, last + 1).hasContent());
    CHECK_FALSE(store.at(ShadowViewGroup::Count, 0).hasContent());
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, cascadeViewAt(last, rigidDraw()),
                                store.at(ShadowViewGroup::Cascade, last + 1)) ==
          ShadowViewDisposition::Recorded);
}

// --- the reuse toggle: scheduling, not pixels ---------------------------------------------------

TEST_CASE("reuse disabled records content that would otherwise have been reused",
          "[ShadowPassPlan]")
{
    const PreparedShadowView prepared = viewWith(rigidDraw());
    const ShadowViewResidency resident = residentFrom(prepared);

    // The SAME content and the SAME residency, differing only in the policy. That is the whole
    // claim: the toggle changes whether identical content is re-rasterised, never what it is.
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, prepared, resident) ==
          ShadowViewDisposition::Reused);
    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Disabled, prepared, resident) ==
          ShadowViewDisposition::Recorded);
}

TEST_CASE("reuse disabled does not make an inactive view render", "[ShadowPassPlan]")
{
    // Invalid outranks the toggle. A view nothing this frame vouches for has no work to schedule,
    // so "record everything" must not conjure a rasterisation of a matrix the set disowned — which
    // would also claim a row and stamp a timing for a view that does not exist.
    const PreparedShadowView prepared = viewWith(rigidDraw());
    const ShadowViewResidency resident = residentFrom(prepared);
    CHECK(shadowViewDisposition(false, ShadowReusePolicy::Disabled, prepared, resident) ==
          ShadowViewDisposition::Invalid);
    CHECK(shadowViewDisposition(false, ShadowReusePolicy::Disabled, prepared,
                                ShadowViewResidency{}) == ShadowViewDisposition::Invalid);
}

TEST_CASE("a frame recorded with reuse disabled still leaves usable residency", "[ShadowPassPlan]")
{
    // Recording commits either way, so flipping the toggle back on picks up from the NEWEST content
    // rather than from whatever was resident when it was switched off. Without this the A/B would
    // be misleading in the direction that matters: the first frame after re-enabling would reuse a
    // map some earlier frame recorded.
    ShadowResidencyStore store{};
    ShadowFramePlan plan{};
    PreparedShadowDraw moved = rigidDraw();
    moved.model = translation(5.0f);
    REQUIRE(plan.add(ShadowViewGroup::Cascade, 0, cascadeViewAt(0, moved),
                     ShadowViewDisposition::Recorded));
    store.commit(plan);

    CHECK(shadowViewDisposition(true, ShadowReusePolicy::Enabled, cascadeViewAt(0, moved),
                                store.at(ShadowViewGroup::Cascade, 0)) ==
          ShadowViewDisposition::Reused);
}

TEST_CASE("adoption consumes the recording it adopts", "[ShadowPassPlan]")
{
    // The plan hands its recorded content OVER rather than lending it: adoption runs after the
    // submit, where a copy could throw and there would be nothing useful to do about it. What the
    // entry must not become is a moved-from husk that still reads as `Recorded` content.
    ShadowResidencyStore store{};
    ShadowFramePlan plan{};
    REQUIRE(plan.add(ShadowViewGroup::Cascade, 0, cascadeViewAt(0, rigidDraw()),
                     ShadowViewDisposition::Recorded));
    store.commit(plan);

    CHECK(store.at(ShadowViewGroup::Cascade, 0).hasContent());
    CHECK(plan.view(ShadowViewGroup::Cascade, 0) == nullptr);
    CHECK(plan.disposition(ShadowViewGroup::Cascade, 0) == ShadowViewDisposition::Invalid);
    // The slot is free again, which is what "the entry is cleared" has to mean for a type whose
    // whole discipline is one claim per slot.
    CHECK(plan.add(ShadowViewGroup::Cascade, 0, cascadeViewAt(0, rigidDraw()),
                   ShadowViewDisposition::Recorded));
}

TEST_CASE("only a recorded slot can be taken", "[ShadowPassPlan]")
{
    ShadowFramePlan plan{};
    REQUIRE(plan.add(ShadowViewGroup::Cascade, 0, cascadeViewAt(0, rigidDraw()),
                     ShadowViewDisposition::Reused));
    REQUIRE(plan.add(ShadowViewGroup::Cascade, 1, cascadeViewAt(1, rigidDraw()),
                     ShadowViewDisposition::Invalid));
    CHECK_FALSE(plan.takeRecorded(ShadowViewGroup::Cascade, 0).has_value());
    CHECK_FALSE(plan.takeRecorded(ShadowViewGroup::Cascade, 1).has_value());
    // A reused entry is left INTACT: the frame is still describing it, and only the recorded ones
    // are being handed over.
    CHECK(plan.view(ShadowViewGroup::Cascade, 0) != nullptr);
    CHECK(plan.disposition(ShadowViewGroup::Cascade, 0) == ShadowViewDisposition::Reused);
    // And an address that names no slot answers "nothing", rather than reaching into a neighbour.
    CHECK_FALSE(
        plan.takeRecorded(ShadowViewGroup::Cascade, shadowViewSlotCount(ShadowViewGroup::Cascade))
            .has_value());
}
