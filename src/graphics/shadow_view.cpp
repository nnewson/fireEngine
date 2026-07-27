#include <fire_engine/graphics/shadow_view.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

#include <fire_engine/math/singular_value.hpp>

namespace fire_engine
{

namespace
{

constexpr float kInfinity = std::numeric_limits<float>::infinity();

[[nodiscard]] bool finite(const Vec3& v) noexcept
{
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

[[nodiscard]] bool finite(const Mat4& m) noexcept
{
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            if (!std::isfinite(m[row, col]))
            {
                return false;
            }
        }
    }
    return true;
}

// The recorded LOD chain must be usable BEFORE anything is selected from it. Left to the
// projection, an invalid deviation merely projects to infinity, breaks the search loop, and returns
// a `Selected` level 0 — indistinguishable in the diagnostics from a caster the selector genuinely
// judged to need full detail. The distinction is the whole point of having reasons.
//
// Required: LOD0 is exactly zero (it IS the original surface), every coarser cut is finite and
// non-negative, and deviation is monotonically non-decreasing (a cut that removes more geometry
// cannot deviate less — if it claims to, the chain is not what the selector assumes and its
// early-out would silently skip qualifying levels).
[[nodiscard]] bool lodChainUsable(std::span<const GeometryLod> lods) noexcept
{
    if (lods[0].shadowDeviation != 0.0f)
    {
        return false;
    }
    float previous = 0.0f;
    for (std::size_t level = 1; level < lods.size(); ++level)
    {
        const float deviation = lods[level].shadowDeviation;
        if (!std::isfinite(deviation) || !(deviation >= previous))
        {
            return false;
        }
        previous = deviation;
    }
    return true;
}

[[nodiscard]] ShadowLodSelection forced(ShadowLodReason reason) noexcept
{
    // Every forced fallback is LOD0 — the only level whose deviation is known to be zero — carrying
    // an infinite projected error so the diagnostics show it was not a within-budget choice.
    return ShadowLodSelection{.level = 0, .reason = reason, .projectedTexels = kInfinity};
}

} // namespace

ShadowView ShadowView::orthographic(float worldUnitsPerTexel) noexcept
{
    ShadowView view;
    view.kind_ = ShadowViewKind::Orthographic;
    view.worldUnitsPerTexel_ = worldUnitsPerTexel;
    view.valid_ = std::isfinite(worldUnitsPerTexel) && worldUnitsPerTexel > 0.0f;
    return view;
}

ShadowView ShadowView::perspective(const Vec3& lightPosition, const Vec3& forward, float fovRadians,
                                   std::uint32_t extentTexels, float nearPlane) noexcept
{
    ShadowView view;
    view.kind_ = ShadowViewKind::Perspective;
    view.lightPosition_ = lightPosition;
    view.extentTexels_ = extentTexels;
    view.nearPlane_ = nearPlane;

    // Normalise here rather than trusting a caller convention: a un-normalised forward silently
    // scales every depth, and the resulting selection would look plausible while being wrong by the
    // direction's length.
    const float length = forward.magnitude();
    const bool directionUsable = finite(forward) && length > 1e-6f;
    if (directionUsable)
    {
        view.forward_ = forward / length;
    }

    // A half-fov at or beyond 90 degrees has no finite tangent, and a zero one has no extent.
    const bool fovUsable =
        std::isfinite(fovRadians) && fovRadians > 0.0f && fovRadians < std::numbers::pi_v<float>;
    if (fovUsable)
    {
        view.tanHalfFov_ = std::tan(0.5f * fovRadians);
    }

    view.valid_ = directionUsable && fovUsable && std::isfinite(view.tanHalfFov_) &&
                  view.tanHalfFov_ > 0.0f && finite(lightPosition) && extentTexels > 0 &&
                  std::isfinite(nearPlane) && nearPlane > 0.0f;
    return view;
}

float nearestForwardDepth(const ShadowView& view, const Bounds3& worldBounds) noexcept
{
    if (!view.valid() || !worldBounds.valid || !finite(worldBounds.min) || !finite(worldBounds.max))
    {
        return -kInfinity;
    }
    // The minimum signed projection of all EIGHT corners onto the light's forward axis. A centre
    // distance would under-estimate for a box straddling the light, and a radial distance would
    // over-estimate off-axis — a corner far to the side is further away radially than it is deep,
    // and it is DEPTH that sets perspective texel size.
    float nearest = kInfinity;
    for (int corner = 0; corner < 8; ++corner)
    {
        const Vec3 point{(corner & 1) != 0 ? worldBounds.max.x() : worldBounds.min.x(),
                         (corner & 2) != 0 ? worldBounds.max.y() : worldBounds.min.y(),
                         (corner & 4) != 0 ? worldBounds.max.z() : worldBounds.min.z()};
        nearest = std::min(nearest, Vec3::dotProduct(point - view.lightPosition(), view.forward()));
    }
    return nearest;
}

float projectShadowErrorTexels(float worldError, const ShadowView& view,
                               float nearestDepth) noexcept
{
    if (!view.valid() || !(worldError >= 0.0f) || !std::isfinite(worldError))
    {
        return kInfinity;
    }
    if (view.kind() == ShadowViewKind::Orthographic)
    {
        // Texel size is constant across the layer, so depth plays no part.
        return worldError / view.worldUnitsPerTexel();
    }
    if (!std::isfinite(nearestDepth) || nearestDepth <= view.nearPlane())
    {
        return kInfinity;
    }
    // The displacement is FINITE, and the projection steepens as a point approaches the light
    // (it scales with 1/depth). Evaluating at `nearestDepth` gives the slope where the surface
    // sits, which a displacement TOWARD the light then exceeds — measured at ~0.3% over on a
    // frustum corner, i.e. a real under-bound rather than a rounding artefact. Evaluate instead at
    // the closest depth the displaced surface can reach.
    const float displacedDepth = nearestDepth - worldError;
    if (displacedDepth <= view.nearPlane())
    {
        return kInfinity;
    }
    // Radial texel displacement, bounded over the whole square frustum rather than on-axis: a world
    // error with a depth component projects more strongly near a frustum edge. The projection's
    // Jacobian is (k/z)·[[1, 0, -X/z], [0, 1, -Y/z]], whose largest singular value is
    // (k/z)·sqrt(1 + (X² + Y²)/z²); at a corner of a square frustum X = Y = tanHalfFov·z, giving
    // the factor below.
    const float edgeFactor = std::sqrt(1.0f + 2.0f * view.tanHalfFov() * view.tanHalfFov());
    const float texelsPerWorldUnit = static_cast<float>(view.extentTexels()) * edgeFactor /
                                     (2.0f * view.tanHalfFov() * displacedDepth);
    return worldError * texelsPerWorldUnit;
}

ShadowLodSelection selectShadowLod(std::span<const GeometryLod> lods, const ShadowView& view,
                                   const Mat4& model, const Bounds3& worldBounds,
                                   float budgetTexels, ShadowLodHysteresis hysteresis,
                                   std::size_t previousLevel) noexcept
{
    if (lods.empty())
    {
        return forced(ShadowLodReason::InvalidCaster);
    }
    if (lods.size() == 1)
    {
        // Nothing to choose between — reported as the geometry's own property, not a failure.
        return ShadowLodSelection{
            .level = 0, .reason = ShadowLodReason::SingleLevel, .projectedTexels = 0.0f};
    }
    if (!view.valid())
    {
        return forced(ShadowLodReason::InvalidView);
    }
    if (!finite(model) || !worldBounds.valid || !std::isfinite(budgetTexels) ||
        budgetTexels <= 0.0f)
    {
        return forced(ShadowLodReason::InvalidCaster);
    }
    // Out of range is a plumbing error (state carried across a reassigned punctual slot, or a stale
    // record), NOT something to clamp: clamping would silently apply one caster's history to
    // another's geometry, which is the bug this reason exists to surface.
    if (previousLevel != kNoPreviousShadowLod && previousLevel >= lods.size())
    {
        return forced(ShadowLodReason::InvalidPreviousLevel);
    }
    if (!(hysteresis.coarsenRatio > 0.0f) || hysteresis.coarsenRatio > 1.0f)
    {
        return forced(ShadowLodReason::InvalidCaster);
    }
    if (!lodChainUsable(lods))
    {
        return forced(ShadowLodReason::InvalidCaster);
    }

    float depth = 0.0f;
    if (view.kind() == ShadowViewKind::Perspective)
    {
        depth = nearestForwardDepth(view, worldBounds);
        if (!std::isfinite(depth) || depth <= view.nearPlane())
        {
            return forced(ShadowLodReason::NearPlane);
        }
    }

    // The object-space deviation is bounded into world space by the conservative σ_max of the
    // transform's linear part — shared with VDPM, so a scaled instance can never under-refine.
    const float worldScale = largestSingularValue(linearPart(model));
    if (!std::isfinite(worldScale))
    {
        return forced(ShadowLodReason::InvalidCaster);
    }

    const auto projectedFor = [&](std::size_t level)
    { return projectShadowErrorTexels(lods[level].shadowDeviation * worldScale, view, depth); };

    // Refining may jump MULTIPLE levels at once: if the held level is over budget, the point is to
    // restore the budget now, not to walk back one level per frame while the shadow stays wrong.
    // Coarsening may likewise reach the coarsest level that clears the stricter threshold, so a
    // caster receding quickly doesn't trail its correct level by frames.
    const float coarsenBudget = budgetTexels * hysteresis.coarsenRatio;
    const bool holding = previousLevel != kNoPreviousShadowLod;
    const bool currentOverBudget = holding && projectedFor(previousLevel) > budgetTexels;
    // With no history, or when the held level is already over budget, select against the plain
    // budget. Otherwise the caster is comfortable where it is and only the stricter threshold can
    // move it coarser — that gap is the dead band.
    const float threshold = (!holding || currentOverBudget) ? budgetTexels : coarsenBudget;

    std::size_t chosen = 0;
    float chosenTexels = 0.0f;
    for (std::size_t level = 1; level < lods.size(); ++level)
    {
        const float texels = projectedFor(level);
        if (!(texels <= threshold))
        {
            break; // deviation is monotone across cuts, so no coarser level can qualify either
        }
        chosen = level;
        chosenTexels = texels;
    }
    if (holding && !currentOverBudget && chosen < previousLevel)
    {
        // The dead band: refuse to give back detail the caster already holds unless the coarser
        // level cleared the stricter threshold above. Without this a caster sitting on the boundary
        // alternates level every frame and its shadow silhouette twitches in place.
        return ShadowLodSelection{.level = previousLevel,
                                  .reason = ShadowLodReason::Selected,
                                  .projectedTexels = projectedFor(previousLevel)};
    }
    return ShadowLodSelection{
        .level = chosen, .reason = ShadowLodReason::Selected, .projectedTexels = chosenTexels};
}

} // namespace fire_engine
