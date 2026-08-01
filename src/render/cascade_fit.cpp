#include <fire_engine/render/cascade_fit.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/view_basis.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] bool finite(float v) noexcept
{
    return std::isfinite(v);
}

[[nodiscard]] bool finite(Vec3 v) noexcept
{
    return finite(v.x()) && finite(v.y()) && finite(v.z());
}

// A direction this short would send `normaliseOr` / `makeViewBasis` to their fallbacks, which is
// exactly the manufacturing this fit refuses to do.
[[nodiscard]] bool usableDirection(Vec3 v) noexcept
{
    return finite(v) && v.magnitudeSquared() >= float_epsilon * float_epsilon;
}

// Unit within the slack a float normalise actually leaves, and no more. `Vec3::normalise` routinely
// lands a squared length an ulp or two either side of one — the observed error on the engine's own
// sun directions is half an epsilon — so demanding exactness would reject every real caller. The
// tolerance is therefore sized in units of float rounding rather than as a round decimal: anything
// outside it is a SCALE the producer applied, and a scaled light direction scales every light-space
// depth this fit reports. A hand-picked 1e-3 would have admitted a 0.05% scale as "unit".
[[nodiscard]] bool unitDirection(Vec3 v) noexcept
{
    constexpr float kUnitTolerance = 8.0f * std::numeric_limits<float>::epsilon();
    return finite(v) && std::abs(v.magnitudeSquared() - 1.0f) <= kUnitTolerance;
}

} // namespace

std::optional<CascadeReceiverFit>
CascadeReceiverFit::fit(const CascadeReceiverInput& input) noexcept
{
    if (!finite(input.cameraPosition) || !finite(input.cameraTarget) || !finite(input.fovRadians) ||
        !finite(input.aspect) || !finite(input.sliceNear) || !finite(input.sliceFar))
    {
        return std::nullopt;
    }
    if (input.fovRadians <= 0.0f || input.fovRadians >= pi || input.aspect <= 0.0f)
    {
        return std::nullopt;
    }
    if (input.sliceNear <= 0.0f || input.sliceFar <= input.sliceNear)
    {
        return std::nullopt;
    }
    if (input.shadowMapExtent == 0)
    {
        return std::nullopt;
    }
    if (!unitDirection(input.lightDirection) ||
        !usableDirection(input.cameraTarget - input.cameraPosition))
    {
        return std::nullopt;
    }

    // Light basis. `lightUp` is a seed chosen to avoid degeneracy with the light direction, then
    // discarded once `lightRight` has been derived from it: the returned up is the
    // re-orthogonalised one, so the three axes are perpendicular whatever seed was picked.
    const Vec3 lightDirection = input.lightDirection;
    const Vec3 upSeed = stableUpForForward(lightDirection);
    const Vec3 lightRight =
        normaliseOr(Vec3::crossProduct(lightDirection, upSeed), {1.0f, 0.0f, 0.0f});
    const Vec3 lightUp = normaliseOr(Vec3::crossProduct(lightRight, lightDirection), upSeed);

    const ViewBasis basis = makeViewBasis(input.cameraPosition, input.cameraTarget);
    const float tanHalfFov = std::tan(input.fovRadians * 0.5f);

    const float nearH = tanHalfFov * input.sliceNear;
    const float nearW = nearH * input.aspect;
    const float farH = tanHalfFov * input.sliceFar;
    const float farW = farH * input.aspect;

    const Vec3 sliceNearCentre = input.cameraPosition + basis.forward * input.sliceNear;
    const Vec3 sliceFarCentre = input.cameraPosition + basis.forward * input.sliceFar;

    const std::array<Vec3, 8> corners{sliceNearCentre - basis.right * nearW - basis.up * nearH,
                                      sliceNearCentre + basis.right * nearW - basis.up * nearH,
                                      sliceNearCentre + basis.right * nearW + basis.up * nearH,
                                      sliceNearCentre - basis.right * nearW + basis.up * nearH,
                                      sliceFarCentre - basis.right * farW - basis.up * farH,
                                      sliceFarCentre + basis.right * farW - basis.up * farH,
                                      sliceFarCentre + basis.right * farW + basis.up * farH,
                                      sliceFarCentre - basis.right * farW + basis.up * farH};

    Vec3 frustumCentre{0.0f, 0.0f, 0.0f};
    for (const auto& c : corners)
    {
        frustumCentre += c;
    }
    frustumCentre /= 8.0f;

    float radius = 0.0f;
    for (const auto& c : corners)
    {
        radius = std::max(radius, (c - frustumCentre).magnitude());
    }
    radius = std::ceil(radius * 16.0f) / 16.0f;

    const float worldPerTexel = (2.0f * radius) / static_cast<float>(input.shadowMapExtent);
    const float centreU = Vec3::dotProduct(frustumCentre, lightRight);
    const float centreV = Vec3::dotProduct(frustumCentre, lightUp);
    const float centreW = Vec3::dotProduct(frustumCentre, lightDirection);
    const float snappedU = std::floor(centreU / worldPerTexel) * worldPerTexel;
    const float snappedV = std::floor(centreV / worldPerTexel) * worldPerTexel;
    const Vec3 snappedCentre =
        lightRight * snappedU + lightUp * snappedV + lightDirection * centreW;

    // Receiver depth read off the corners themselves. Every caster-aware depth policy needs to know
    // how deep the geometry it must cover actually reaches, and the bounding sphere does not say.
    float receiverMinW = Vec3::dotProduct(corners[0], lightDirection);
    float receiverMaxW = receiverMinW;
    for (const auto& c : corners)
    {
        const float w = Vec3::dotProduct(c, lightDirection);
        receiverMinW = std::min(receiverMinW, w);
        receiverMaxW = std::max(receiverMaxW, w);
    }

    // A non-finite result here means the arithmetic overflowed on finite inputs (a slice far enough
    // out that the corner sums leave float range). Report failure rather than hand a NaN matrix to
    // the shadow pass, where it becomes an empty or garbage map with no obvious cause.
    if (!finite(radius) || !finite(worldPerTexel) || worldPerTexel <= 0.0f ||
        !finite(frustumCentre) || !finite(snappedCentre) || !finite(receiverMinW) ||
        !finite(receiverMaxW))
    {
        return std::nullopt;
    }

    CascadeReceiverFit fit{};
    fit.sliceNear_ = input.sliceNear;
    fit.sliceFar_ = input.sliceFar;
    fit.aspect_ = input.aspect;
    fit.lightDirection_ = lightDirection;
    fit.lightRight_ = lightRight;
    fit.lightUp_ = lightUp;
    fit.frustumCentre_ = frustumCentre;
    fit.snappedCentre_ = snappedCentre;
    fit.radius_ = radius;
    fit.minU_ = snappedU - radius;
    fit.maxU_ = snappedU + radius;
    fit.minV_ = snappedV - radius;
    fit.maxV_ = snappedV + radius;
    fit.centreW_ = centreW;
    fit.receiverMinW_ = receiverMinW;
    fit.receiverMaxW_ = receiverMaxW;
    fit.worldPerTexel_ = worldPerTexel;
    return fit;
}

std::optional<CascadeDepthFit> fitLegacyCascadeDepth(const CascadeReceiverFit& receiver,
                                                     float backExtend) noexcept
{
    // `backExtend` is the only untrusted input: the receiver could only have come from
    // `CascadeReceiverFit::fit`, which guarantees an orthonormal basis, a positive radius and texel
    // size and ordered bounds. Re-checking those here would be a second authority on the same
    // question — the drift this extraction exists to remove.
    if (!finite(backExtend) || backExtend < 0.0f)
    {
        return std::nullopt;
    }

    // Deliberately the original expressions, in the original order: this function's whole job right
    // now is to be indistinguishable from what shipped, so the caster-aware policy that replaces it
    // has a verified baseline to differ from.
    const Vec3 lightPosition =
        receiver.snappedCentre() - receiver.lightDirection() * (receiver.radius() + backExtend);
    const Mat4 lightView =
        Mat4::lookAt(lightPosition, receiver.snappedCentre(), receiver.lightUp());
    const Mat4 lightProj =
        Mat4::ortho(-receiver.radius(), receiver.radius(), -receiver.radius(), receiver.radius(),
                    0.0f, 2.0f * receiver.radius() + 2.0f * backExtend);

    CascadeDepthFit depth{};
    depth.lightPosition = lightPosition;
    // The ortho near plane sits at the light position (near = 0), so the near plane's world-space W
    // is the light's own W; the far plane is the ortho far distance beyond it.
    depth.nearW = receiver.centreW() - receiver.radius() - backExtend;
    depth.farW = receiver.centreW() + receiver.radius() + backExtend;
    depth.viewDepthSpan = depth.farW - depth.nearW;
    depth.viewProj = lightProj * lightView;

    // An ORDERED, finite output — the check the view set cannot make for us. It rejects a
    // non-finite matrix, but a range that came out reversed produces a perfectly finite one whose
    // depth comparisons are all backwards, and nothing downstream would notice.
    if (!finite(depth.nearW) || !finite(depth.farW) || !(depth.farW > depth.nearW) ||
        !finite(depth.viewDepthSpan) || !finite(depth.lightPosition))
    {
        return std::nullopt;
    }
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            if (!finite(depth.viewProj[row, col]))
            {
                return std::nullopt;
            }
        }
    }
    return depth;
}

CascadeCasterPlacement placeCaster(const CascadeReceiverFit& receiver, const CascadeDepthFit& depth,
                                   const Bounds3& casterBounds) noexcept
{
    CascadeCasterPlacement placement{};
    if (!casterBounds.valid)
    {
        return placement;
    }

    // All eight corners, not the centre and a radius: an axis-aligned box is not a sphere, and the
    // whole question here is whether a FACE of it crosses a plane. A radius would round the box out
    // and report clipping that is not there.
    const Vec3& lo = casterBounds.min;
    const Vec3& hi = casterBounds.max;
    bool first = true;
    for (int corner = 0; corner < 8; ++corner)
    {
        const Vec3 p{(corner & 1) != 0 ? hi.x() : lo.x(), (corner & 2) != 0 ? hi.y() : lo.y(),
                     (corner & 4) != 0 ? hi.z() : lo.z()};
        const float u = Vec3::dotProduct(p, receiver.lightRight());
        const float v = Vec3::dotProduct(p, receiver.lightUp());
        const float w = Vec3::dotProduct(p, receiver.lightDirection());
        if (first)
        {
            placement.minU = placement.maxU = u;
            placement.minV = placement.maxV = v;
            placement.minW = placement.maxW = w;
            first = false;
            continue;
        }
        placement.minU = std::min(placement.minU, u);
        placement.maxU = std::max(placement.maxU, u);
        placement.minV = std::min(placement.minV, v);
        placement.maxV = std::max(placement.maxV, v);
        placement.minW = std::min(placement.minW, w);
        placement.maxW = std::max(placement.maxW, w);
    }

    // Conservative on both boundaries: touching an edge counts as straddling, never as Outside
    // (which a candidate query would reject) and never as Inside (which would claim the caster is
    // wholly covered when a texel of it may not be).
    const bool overlaps = placement.maxU >= receiver.minU() && placement.minU <= receiver.maxU() &&
                          placement.maxV >= receiver.minV() && placement.minV <= receiver.maxV();
    if (!overlaps)
    {
        placement.footprint = CascadeFootprintRelation::Outside;
    }
    else if (placement.minU > receiver.minU() && placement.maxU < receiver.maxU() &&
             placement.minV > receiver.minV() && placement.maxV < receiver.maxV())
    {
        placement.footprint = CascadeFootprintRelation::Inside;
    }
    else
    {
        placement.footprint = CascadeFootprintRelation::Straddles;
    }
    placement.clippedNear = placement.minW < depth.nearW;
    placement.clippedFar = placement.maxW > depth.farW;
    placement.insideDepth = !placement.clippedNear && !placement.clippedFar;
    // Wholly on one side of the range. Checked against the OPPOSITE bound of each pair, so a caster
    // straddling the range reports as clipped rather than outside.
    placement.outsideDepth = placement.maxW < depth.nearW || placement.minW > depth.farW;
    return placement;
}

} // namespace fire_engine
