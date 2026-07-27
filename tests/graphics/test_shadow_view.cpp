#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <fire_engine/graphics/shadow_view.hpp>

using namespace fire_engine;

namespace
{

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Two cuts with a known object-space deviation. LOD0 is always 0 by construction.
std::vector<GeometryLod> lodsWithDeviations(float level1, float level2)
{
    std::vector<GeometryLod> lods(3);
    lods[0].shadowDeviation = 0.0f;
    lods[1].shadowDeviation = level1;
    lods[2].shadowDeviation = level2;
    return lods;
}

Bounds3 boundsAt(Vec3 centre, float halfExtent)
{
    Bounds3 b;
    b.expand(centre - Vec3{halfExtent, halfExtent, halfExtent});
    b.expand(centre + Vec3{halfExtent, halfExtent, halfExtent});
    return b;
}

Mat4 scaled(float x, float y, float z)
{
    Mat4 m = Mat4::identity();
    m[0, 0] = x;
    m[1, 1] = y;
    m[2, 2] = z;
    return m;
}

constexpr ShadowLodHysteresis kNoHysteresis{.coarsenRatio = 1.0f};

} // namespace

// ---------------------------------------------------------------------------
// View construction
// ---------------------------------------------------------------------------

TEST_CASE("ShadowView.PerspectiveFactoryNormalisesItsOwnForward", "[ShadowView]")
{
    // The caller must not have to remember to normalise: an un-normalised direction would scale
    // every depth silently, and the resulting selection would look entirely plausible.
    const ShadowView unit =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.0f, 1024, 0.05f);
    const ShadowView long_ =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -50}, 1.0f, 1024, 0.05f);

    REQUIRE(unit.valid());
    REQUIRE(long_.valid());
    CHECK(long_.forward().z() == Catch::Approx(unit.forward().z()));
    CHECK(long_.forward().magnitude() == Catch::Approx(1.0f).epsilon(1e-5));

    const Bounds3 bounds = boundsAt(Vec3{0, 0, -10}, 1.0f);
    CHECK(nearestForwardDepth(long_, bounds) == Catch::Approx(nearestForwardDepth(unit, bounds)));
}

TEST_CASE("ShadowView.DegenerateDescriptorsAreInvalid", "[ShadowView]")
{
    CHECK_FALSE(ShadowView::orthographic(0.0f).valid());
    CHECK_FALSE(ShadowView::orthographic(-1.0f).valid());
    CHECK_FALSE(ShadowView::orthographic(kNaN).valid());
    CHECK(ShadowView::orthographic(0.01f).valid());

    // Zero-length and non-finite directions, a non-positive fov, a zero extent, a bad near plane.
    CHECK_FALSE(ShadowView::perspective({}, Vec3{0, 0, 0}, 1.0f, 512, 0.05f).valid());
    CHECK_FALSE(ShadowView::perspective({}, Vec3{kNaN, 0, -1}, 1.0f, 512, 0.05f).valid());
    CHECK_FALSE(ShadowView::perspective({}, Vec3{0, 0, -1}, 0.0f, 512, 0.05f).valid());
    CHECK_FALSE(
        ShadowView::perspective({}, Vec3{0, 0, -1}, std::numbers::pi_v<float>, 512, 0.05f).valid());
    CHECK_FALSE(ShadowView::perspective({}, Vec3{0, 0, -1}, 1.0f, 0, 0.05f).valid());
    CHECK_FALSE(ShadowView::perspective({}, Vec3{0, 0, -1}, 1.0f, 512, 0.0f).valid());
}

// ---------------------------------------------------------------------------
// Depth
// ---------------------------------------------------------------------------

TEST_CASE("ShadowView.NearestDepthUsesTheNEARESTCornerAlongTheForwardAxis", "[ShadowView]")
{
    const ShadowView view =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.0f, 512, 0.05f);
    // Centre sits 10 deep; the near face is at 9.
    CHECK(nearestForwardDepth(view, boundsAt(Vec3{0, 0, -10}, 1.0f)) == Catch::Approx(9.0f));
}

TEST_CASE("ShadowView.NearestDepthIsNotRadialDistance", "[ShadowView]")
{
    // A caster far off-axis is further away RADIALLY than it is DEEP, and depth is what sets
    // perspective texel size. Radial distance here would be sqrt(10^2 + 40^2) ~ 41.2, over-stating
    // depth four-fold and permitting a much coarser level than the view can afford.
    const ShadowView view =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.2f, 512, 0.05f);
    const float depth = nearestForwardDepth(view, boundsAt(Vec3{40, 0, -10}, 1.0f));

    CHECK(depth == Catch::Approx(9.0f));
    CHECK(depth < 41.0f);
}

TEST_CASE("ShadowView.BoundsStraddlingTheLightGiveNonPositiveDepth", "[ShadowView]")
{
    // A centre-distance test would report a comfortable positive depth here and hide the
    // intersection entirely.
    const ShadowView view =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.0f, 512, 0.05f);

    CHECK(nearestForwardDepth(view, boundsAt(Vec3{0, 0, -1}, 4.0f)) <= 0.0f);
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

TEST_CASE("ShadowView.OrthographicProjectionIsErrorOverTexelSize", "[ShadowView]")
{
    const ShadowView view = ShadowView::orthographic(0.05f);

    CHECK(projectShadowErrorTexels(0.10f, view, 0.0f) == Catch::Approx(2.0f));
    // Doubling world-units-per-texel HALVES the projected error — the coarser cascade tolerance.
    CHECK(projectShadowErrorTexels(0.10f, ShadowView::orthographic(0.10f), 0.0f) ==
          Catch::Approx(1.0f));
}

TEST_CASE("ShadowView.PerspectiveProjectionFallsWithDistance", "[ShadowView]")
{
    const ShadowView view =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.0f, 1024, 0.05f);

    const float near = projectShadowErrorTexels(0.1f, view, 5.0f);
    const float far = projectShadowErrorTexels(0.1f, view, 10.0f);

    CHECK(far < near);
    // Doubling the distance roughly halves the projection, but NOT exactly: the bound is evaluated
    // at `depth - worldError` (the closest the displaced surface can reach), and that correction is
    // proportionally larger at the nearer depth. 2% covers it for a 0.1 error at 5 vs 10 units.
    CHECK(far == Catch::Approx(near * 0.5f).epsilon(0.02));
}

TEST_CASE("ShadowView.PerspectiveProjectionExceedsTheOnAxisEstimate", "[ShadowView]")
{
    // The Lipschitz factor sqrt(1 + 2 tan^2) bounds a displacement with a depth component near a
    // frustum EDGE, which the on-axis formula understates. Being conservative here is the point:
    // under-stating the projection is what lets an over-coarse level through.
    const float fov = 1.0f;
    const ShadowView view =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, fov, 1024, 0.05f);
    const float tanHalf = std::tan(0.5f * fov);
    const float depth = 8.0f;

    const float worldError = 0.1f;
    // Both terms of the correction: the on-axis slope is itself taken at the DISPLACED depth, then
    // widened by the frustum-corner factor.
    const float displacedDepth = depth - worldError;
    const float onAxisAtDepth = worldError * 1024.0f / (2.0f * tanHalf * depth);
    const float onAxisDisplaced = worldError * 1024.0f / (2.0f * tanHalf * displacedDepth);
    const float bounded = projectShadowErrorTexels(worldError, view, depth);

    CHECK(bounded > onAxisAtDepth);
    CHECK(
        bounded ==
        Catch::Approx(onAxisDisplaced * std::sqrt(1.0f + 2.0f * tanHalf * tanHalf)).epsilon(1e-4));
}

TEST_CASE("ShadowView.InvalidProjectionInputsYieldInfinity", "[ShadowView]")
{
    const ShadowView ortho = ShadowView::orthographic(0.05f);
    const ShadowView persp =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.0f, 512, 0.5f);

    CHECK(std::isinf(projectShadowErrorTexels(0.1f, ShadowView::orthographic(0.0f), 0.0f)));
    CHECK(std::isinf(projectShadowErrorTexels(kNaN, ortho, 0.0f)));
    CHECK(std::isinf(projectShadowErrorTexels(-1.0f, ortho, 0.0f)));
    // At or behind the near plane the texel size diverges: no projected error is meaningful.
    CHECK(std::isinf(projectShadowErrorTexels(0.1f, persp, 0.5f)));
    CHECK(std::isinf(projectShadowErrorTexels(0.1f, persp, -3.0f)));
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

TEST_CASE("ShadowLod.CameraMovementCannotAlterAFixedLightViewSelection", "[ShadowView]")
{
    // The whole point of SH-02. The selector takes NO camera argument, so this is a structural
    // property — but it is the arc's central claim and deserves an explicit statement: the same
    // caster in the same light view selects the same level regardless of anything the camera does.
    const ShadowView view = ShadowView::orthographic(0.02f);
    const auto lods = lodsWithDeviations(0.03f, 0.20f);

    const auto first = selectShadowLod(lods, view, Mat4::identity(), boundsAt(Vec3{0, 0, 0}, 1.0f),
                                       2.0f, kNoHysteresis, kNoPreviousShadowLod);
    // Same caster, same view, "camera" now somewhere else entirely — nothing in the call changes.
    const auto second = selectShadowLod(lods, view, Mat4::identity(), boundsAt(Vec3{0, 0, 0}, 1.0f),
                                        2.0f, kNoHysteresis, kNoPreviousShadowLod);

    CHECK(first.level == second.level);
    CHECK(first.reason == ShadowLodReason::Selected);
}

TEST_CASE("ShadowLod.ACoarserCascadePermitsACoarserLevel", "[ShadowView]")
{
    // Near and far cascades differ only in world-units-per-texel; the far one tolerates more world
    // error per texel, so it may legitimately select a coarser cut for the same caster.
    const auto lods = lodsWithDeviations(0.03f, 0.20f);
    const Mat4 model = Mat4::identity();
    const Bounds3 bounds = boundsAt(Vec3{0, 0, 0}, 1.0f);

    const auto near = selectShadowLod(lods, ShadowView::orthographic(0.01f), model, bounds, 2.0f,
                                      kNoHysteresis, kNoPreviousShadowLod);
    const auto far = selectShadowLod(lods, ShadowView::orthographic(0.20f), model, bounds, 2.0f,
                                     kNoHysteresis, kNoPreviousShadowLod);

    CHECK(near.level < far.level);
    CHECK(far.level == 2);
}

TEST_CASE("ShadowLod.DoublingObjectScaleDoublesProjectedError", "[ShadowView]")
{
    const ShadowView view = ShadowView::orthographic(0.05f);
    const auto lods = lodsWithDeviations(0.10f, 0.40f);
    const Bounds3 bounds = boundsAt(Vec3{0, 0, 0}, 1.0f);

    // Budget deliberately clear of the exact boundary: sigma_max is outward-rounded one ULP up
    // (conservative by design), so a doubled error lands a hair ABOVE 2x and would fail a test
    // pinned exactly at the threshold — for the right reason.
    const auto unscaled = selectShadowLod(lods, view, Mat4::identity(), bounds, 5.0f, kNoHysteresis,
                                          kNoPreviousShadowLod);
    const auto doubled = selectShadowLod(lods, view, scaled(2.0f, 2.0f, 2.0f), bounds, 5.0f,
                                         kNoHysteresis, kNoPreviousShadowLod);

    REQUIRE(unscaled.reason == ShadowLodReason::Selected);
    REQUIRE(doubled.reason == ShadowLodReason::Selected);
    CHECK(doubled.projectedTexels == Catch::Approx(2.0f * unscaled.projectedTexels).epsilon(1e-4));
}

TEST_CASE("ShadowLod.NonUniformScaleUsesTheLargestStretch", "[ShadowView]")
{
    // A conservative bound must take the WORST axis: a caster stretched 3x on one axis has its
    // deviation stretched 3x in that direction, and selecting for the average would under-refine.
    const ShadowView view = ShadowView::orthographic(0.05f);
    const auto lods = lodsWithDeviations(0.05f, 0.30f);
    const Bounds3 bounds = boundsAt(Vec3{0, 0, 0}, 1.0f);

    const auto uniform = selectShadowLod(lods, view, scaled(3.0f, 3.0f, 3.0f), bounds, 3.0f,
                                         kNoHysteresis, kNoPreviousShadowLod);
    const auto nonUniform = selectShadowLod(lods, view, scaled(3.0f, 1.0f, 1.0f), bounds, 3.0f,
                                            kNoHysteresis, kNoPreviousShadowLod);

    CHECK(nonUniform.projectedTexels == Catch::Approx(uniform.projectedTexels).epsilon(1e-4));
}

TEST_CASE("ShadowLod.GreaterSpotDistancePermitsACoarserLevel", "[ShadowView]")
{
    const ShadowView view =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.0f, 1024, 0.05f);
    const auto lods = lodsWithDeviations(0.02f, 0.10f);
    const Mat4 model = Mat4::identity();

    const auto close = selectShadowLod(lods, view, model, boundsAt(Vec3{0, 0, -4}, 1.0f), 30.0f,
                                       kNoHysteresis, kNoPreviousShadowLod);
    const auto distant = selectShadowLod(lods, view, model, boundsAt(Vec3{0, 0, -40}, 1.0f), 30.0f,
                                         kNoHysteresis, kNoPreviousShadowLod);

    // The distant caster affords a coarser cut. (Comparing the two projectedTexels directly would
    // be meaningless — they describe DIFFERENT levels; the same-level distance falloff is pinned by
    // ShadowView.PerspectiveProjectionFallsWithDistance.)
    CHECK(close.level < distant.level);
    CHECK(close.reason == ShadowLodReason::Selected);
    CHECK(distant.reason == ShadowLodReason::Selected);
}

TEST_CASE("ShadowLod.ForcedFallbacksAreReportedDistinctly", "[ShadowView]")
{
    const auto lods = lodsWithDeviations(0.03f, 0.20f);
    const ShadowView ortho = ShadowView::orthographic(0.05f);
    const ShadowView persp =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.0f, 512, 1.0f);
    const Bounds3 bounds = boundsAt(Vec3{0, 0, -10}, 1.0f);
    Mat4 broken = Mat4::identity();
    broken[1, 1] = kNaN;

    // Every one of these is LOD0, and each must say WHY — reporting them as `Selected` would make
    // the SH-01 panel claim the selector chose full detail on purpose.
    const auto invalidView = selectShadowLod(lods, ShadowView::orthographic(0.0f), Mat4::identity(),
                                             bounds, 2.0f, kNoHysteresis, kNoPreviousShadowLod);
    const auto invalidModel =
        selectShadowLod(lods, ortho, broken, bounds, 2.0f, kNoHysteresis, kNoPreviousShadowLod);
    const auto invalidBudget = selectShadowLod(lods, ortho, Mat4::identity(), bounds, 0.0f,
                                               kNoHysteresis, kNoPreviousShadowLod);
    const auto nearPlane =
        selectShadowLod(lods, persp, Mat4::identity(), boundsAt(Vec3{0, 0, -1}, 4.0f), 2.0f,
                        kNoHysteresis, kNoPreviousShadowLod);
    const auto badPrevious =
        selectShadowLod(lods, ortho, Mat4::identity(), bounds, 2.0f, kNoHysteresis, 99);
    const auto badRatio =
        selectShadowLod(lods, ortho, Mat4::identity(), bounds, 2.0f,
                        ShadowLodHysteresis{.coarsenRatio = 0.0f}, kNoPreviousShadowLod);

    CHECK(invalidView.reason == ShadowLodReason::InvalidView);
    CHECK(invalidModel.reason == ShadowLodReason::InvalidCaster);
    CHECK(invalidBudget.reason == ShadowLodReason::InvalidCaster);
    CHECK(nearPlane.reason == ShadowLodReason::NearPlane);
    CHECK(badPrevious.reason == ShadowLodReason::InvalidPreviousLevel);
    CHECK(badRatio.reason == ShadowLodReason::InvalidCaster);
    for (const auto& selection :
         {invalidView, invalidModel, invalidBudget, nearPlane, badPrevious, badRatio})
    {
        CHECK(selection.level == 0);
        CHECK(std::isinf(selection.projectedTexels));
    }
}

TEST_CASE("ShadowLod.SingleLevelGeometryIsNotAFailure", "[ShadowView]")
{
    std::vector<GeometryLod> single(1);
    const auto selection =
        selectShadowLod(single, ShadowView::orthographic(0.05f), Mat4::identity(),
                        boundsAt(Vec3{0, 0, 0}, 1.0f), 2.0f, kNoHysteresis, kNoPreviousShadowLod);

    CHECK(selection.reason == ShadowLodReason::SingleLevel);
    CHECK(selection.level == 0);
}

// ---------------------------------------------------------------------------
// Hysteresis
// ---------------------------------------------------------------------------

TEST_CASE("ShadowLod.HysteresisHoldsALevelInsideTheDeadBand", "[ShadowView]")
{
    // A caster sitting just inside the coarsening threshold keeps the detail it holds. Without the
    // dead band it would alternate every frame and the silhouette would twitch in place.
    const ShadowView view = ShadowView::orthographic(0.05f);
    const auto lods = lodsWithDeviations(0.04f, 0.09f); // level 2 projects to 1.8 texels
    const Bounds3 bounds = boundsAt(Vec3{0, 0, 0}, 1.0f);
    constexpr ShadowLodHysteresis band{.coarsenRatio = 0.5f}; // coarsen only below 1.0 texels

    // Fresh: 1.8 <= budget 2.0, so the coarse level is taken.
    const auto fresh =
        selectShadowLod(lods, view, Mat4::identity(), bounds, 2.0f, band, kNoPreviousShadowLod);
    // Holding level 1 (0.8 texels, within budget): the coarse level's 1.8 exceeds the stricter
    // 1.0 threshold, so the finer level is KEPT.
    const auto held = selectShadowLod(lods, view, Mat4::identity(), bounds, 2.0f, band, 1);

    CHECK(fresh.level == 2);
    CHECK(held.level == 1);
    CHECK(held.reason == ShadowLodReason::Selected);
}

TEST_CASE("ShadowLod.RefinementMayJumpMultipleLevelsAtOnce", "[ShadowView]")
{
    // A caster holding a level that is now well over budget must be corrected NOW, not walked back
    // one level per frame while its shadow stays visibly wrong.
    const ShadowView view =
        ShadowView::orthographic(0.001f); // tiny texels ⇒ everything over budget
    const auto lods = lodsWithDeviations(0.05f, 0.30f);
    const auto selection =
        selectShadowLod(lods, view, Mat4::identity(), boundsAt(Vec3{0, 0, 0}, 1.0f), 2.0f,
                        ShadowLodHysteresis{.coarsenRatio = 0.5f}, 2);

    CHECK(selection.level == 0); // straight from 2 to 0, not 2 -> 1
    CHECK(selection.reason == ShadowLodReason::Selected);
}

TEST_CASE("ShadowLod.CoarseningReachesTheCoarsestLevelClearingTheThreshold", "[ShadowView]")
{
    // Symmetrically, a caster whose coarse level clears the stricter threshold goes all the way
    // there rather than trailing its correct level by a frame per step.
    const ShadowView view = ShadowView::orthographic(1.0f); // huge texels ⇒ everything is cheap
    const auto lods = lodsWithDeviations(0.05f, 0.30f);
    const auto selection =
        selectShadowLod(lods, view, Mat4::identity(), boundsAt(Vec3{0, 0, 0}, 1.0f), 2.0f,
                        ShadowLodHysteresis{.coarsenRatio = 0.5f}, 0);

    CHECK(selection.level == 2);
}

TEST_CASE("ShadowLod.HysteresisDoesNotChatterAcrossAThreshold", "[ShadowView]")
{
    // The property that matters in motion: sweep a caster across the boundary and back, feeding
    // each frame's decision into the next, and count how many times the level changes. With a dead
    // band it must settle rather than oscillate.
    const auto lods = lodsWithDeviations(0.04f, 0.09f);
    const Bounds3 bounds = boundsAt(Vec3{0, 0, 0}, 1.0f);
    constexpr ShadowLodHysteresis band{.coarsenRatio = 0.6f};

    std::size_t previous = kNoPreviousShadowLod;
    int changes = 0;
    // Texel size oscillates gently around the level-2 threshold.
    for (int frame = 0; frame < 40; ++frame)
    {
        const float wobble = 0.0455f + 0.0015f * std::sin(static_cast<float>(frame) * 0.7f);
        const auto selection = selectShadowLod(lods, ShadowView::orthographic(wobble),
                                               Mat4::identity(), bounds, 2.0f, band, previous);
        if (previous != kNoPreviousShadowLod && selection.level != previous)
        {
            ++changes;
        }
        previous = selection.level;
    }

    // Without hysteresis this alternates on most frames; with it, the level settles.
    CHECK(changes <= 1);
}

// ---------------------------------------------------------------------------
// Gaps closed after review
// ---------------------------------------------------------------------------

TEST_CASE("ShadowView.EdgeFactorBoundsAnActualFrustumCornerDisplacement", "[ShadowView]")
{
    // The earlier edge-factor test only reproduced the formula algebraically, which proves nothing
    // about the geometry it claims to bound. This one PROJECTS: take a point at a frustum corner,
    // displace it by a world error that includes a DEPTH component, and measure the resulting
    // radial displacement in texels. The reported projection must cover it.
    const float fov = 1.0f;
    const std::uint32_t extent = 1024;
    const ShadowView view =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, fov, extent, 0.05f);
    const float tanHalf = std::tan(0.5f * fov);
    const float depth = 8.0f;
    const float worldError = 0.05f;

    // Project a world point to texel coordinates in this view's square frustum.
    const auto toTexels = [&](const Vec3& p)
    {
        const float z = -p.z(); // forward is -Z, so depth is -z
        const float ndcX = p.x() / (tanHalf * z);
        const float ndcY = p.y() / (tanHalf * z);
        return Vec3{ndcX * 0.5f * static_cast<float>(extent),
                    ndcY * 0.5f * static_cast<float>(extent), 0.0f};
    };

    const float bound = projectShadowErrorTexels(worldError, view, depth);

    // The frustum CORNER — where a depth displacement slides a point furthest across the image.
    const Vec3 corner{tanHalf * depth, tanHalf * depth, -depth};
    float worstObserved = 0.0f;
    constexpr int steps = 64;
    for (int i = 0; i < steps; ++i)
    {
        for (int j = 0; j < steps; ++j)
        {
            // Every displacement direction of magnitude `worldError`, including depth-only ones.
            const float theta = static_cast<float>(i) / steps * 2.0f * std::numbers::pi_v<float>;
            const float phi = static_cast<float>(j) / steps * std::numbers::pi_v<float>;
            const Vec3 offset{worldError * std::sin(phi) * std::cos(theta),
                              worldError * std::cos(phi),
                              worldError * std::sin(phi) * std::sin(theta)};
            const Vec3 displaced = corner + offset;
            if (-displaced.z() <= view.nearPlane())
            {
                continue;
            }
            worstObserved =
                std::max(worstObserved, (toTexels(displaced) - toTexels(corner)).magnitude());
        }
    }

    CHECK(worstObserved > 0.0f);
    CHECK(worstObserved <= bound); // the bound actually bounds the geometry, not just the algebra
}

TEST_CASE("ShadowView.TwoPointFacesGiveDifferentDepthsForTheSameBounds", "[ShadowView]")
{
    // A point light's six faces share a position but not a forward, which is exactly why each face
    // is its own view. The same caster is 9 deep in the face looking at it and BEHIND the one
    // looking the other way — a shared descriptor would give both the same answer.
    const Bounds3 bounds = boundsAt(Vec3{0, 0, -10}, 1.0f);
    const ShadowView facingIt =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, -1}, 1.57f, 512, 0.05f);
    const ShadowView facingAway =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{0, 0, 1}, 1.57f, 512, 0.05f);
    const ShadowView facingSideways =
        ShadowView::perspective(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 1.57f, 512, 0.05f);

    CHECK(nearestForwardDepth(facingIt, bounds) == Catch::Approx(9.0f));
    CHECK(nearestForwardDepth(facingAway, bounds) == Catch::Approx(-11.0f));
    CHECK(nearestForwardDepth(facingSideways, bounds) == Catch::Approx(-1.0f));
}

TEST_CASE("ShadowLod.AnUnusableLodChainIsReportedAsInvalid", "[ShadowView]")
{
    // Every one of these used to fall through to a `Selected` level 0 — indistinguishable in the
    // diagnostics from a caster the selector genuinely judged to need full detail.
    const ShadowView view = ShadowView::orthographic(0.05f);
    const Bounds3 bounds = boundsAt(Vec3{0, 0, 0}, 1.0f);
    const auto select = [&](std::vector<GeometryLod> lods)
    {
        return selectShadowLod(lods, view, Mat4::identity(), bounds, 1000.0f, kNoHysteresis,
                               kNoPreviousShadowLod);
    };

    auto infinite = lodsWithDeviations(std::numeric_limits<float>::infinity(), 0.2f);
    auto notANumber = lodsWithDeviations(kNaN, 0.2f);
    auto negative = lodsWithDeviations(-0.1f, 0.2f);
    auto nonMonotonic = lodsWithDeviations(0.30f, 0.05f); // a coarser cut claiming LESS deviation
    auto nonZeroBase = lodsWithDeviations(0.10f, 0.20f);
    nonZeroBase[0].shadowDeviation = 0.01f; // LOD0 IS the original surface; it cannot deviate

    for (const auto& selection : {select(infinite), select(notANumber), select(negative),
                                  select(nonMonotonic), select(nonZeroBase)})
    {
        CHECK(selection.reason == ShadowLodReason::InvalidCaster);
        CHECK(selection.level == 0);
        CHECK(std::isinf(selection.projectedTexels));
    }
}

TEST_CASE("ShadowLod.TheHysteresisDefaultIsRejectedUntilStated", "[ShadowView]")
{
    // A plausible-looking default would be an unmeasured tuning decision inherited silently by
    // every caller. The selector refuses it, so a caller must state a ratio it chose.
    const auto lods = lodsWithDeviations(0.03f, 0.20f);
    const auto selection = selectShadowLod(lods, ShadowView::orthographic(0.05f), Mat4::identity(),
                                           boundsAt(Vec3{0, 0, 0}, 1.0f), 2.0f,
                                           ShadowLodHysteresis{}, kNoPreviousShadowLod);

    CHECK(selection.reason == ShadowLodReason::InvalidCaster);
}
