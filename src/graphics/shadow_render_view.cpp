#include <fire_engine/graphics/shadow_render_view.hpp>

#include <fire_engine/graphics/shadow_bias.hpp>

#include <cassert>
#include <cmath>

namespace fire_engine
{

namespace
{

// The matrix an active view RASTERISES and culls with, so it has a stricter contract than the
// projection descriptor beside it. A NaN or infinity propagates two ways, and neither is a clean
// failure: into `Frustum::fromViewProj`, where every comparison against a NaN plane is false so the
// cull stops REJECTING anything (permissive, not empty — the pass then walks casters it should have
// dropped); and into the GPU transform, where NaN clip coordinates fail the rasteriser's tests so
// the view draws nothing. An unusable descriptor is a reportable state; an unusable render matrix
// is not a view at all.
[[nodiscard]] bool allFinite(const Mat4& m) noexcept
{
    for (std::size_t i = 0; i < 16; ++i)
    {
        if (!std::isfinite(m.data()[i]))
        {
            return false;
        }
    }
    return true;
}

} // namespace

bool ShadowRenderViewSet::inRange(ShadowViewGroup group, std::size_t slot) noexcept
{
    return static_cast<std::size_t>(group) < kShadowViewGroupCount &&
           slot < shadowViewSlotCount(group);
}

void ShadowRenderViewSet::reset() noexcept
{
    for (auto& view : views_)
    {
        view.reset();
    }
    worldOnlyActive_.fill(false);
}

ShadowViewMetrics ShadowViewMetrics::orthographic(float worldUnitsPerTexel,
                                                  float depthSpanWorld) noexcept
{
    // Both constants of the layer, so both are resolved here rather than per fragment.
    return ShadowViewMetrics{
        ShadowViewMetricsKind::Orthographic,
        {worldUnitsPerTexel, orthographicNormalizedDepthPerWorldUnit(depthSpanWorld), 0.0f, 0.0f}};
}

ShadowViewMetrics ShadowViewMetrics::spot(float texelAngleScale, float nearPlane,
                                          float farPlane) noexcept
{
    // Neither quantity is constant across a cone, so what travels is the per-view constants the
    // receiver derives both from — with its own forward and radial depth, which only it knows.
    return ShadowViewMetrics{ShadowViewMetricsKind::Spot,
                             {texelAngleScale, nearPlane, farPlane, 0.0f}};
}

ShadowViewMetrics ShadowViewMetrics::pointLight(float texelAxisScale, float rangeWorld) noexcept
{
    // Footprint is per-fragment (major axis), depth conversion is not: the point path stores radial
    // distance / range, which is linear in range.
    return ShadowViewMetrics{
        ShadowViewMetricsKind::PointLight,
        {texelAxisScale, cubeNormalizedDepthPerWorldUnit(rangeWorld), 0.0f, 0.0f}};
}

bool ShadowRenderViewSet::store(ShadowViewGroup group, std::size_t slot,
                                ShadowViewKind expectedKind,
                                ShadowViewMetricsKind expectedMetricsKind, const Mat4& viewProj,
                                const ShadowView& projection, const ShadowViewMetrics& biasMetrics,
                                const ShadowLogicalViewId& logicalId) noexcept
{
    // ADDRESSING is checked first and separately from CONTENT. An out-of-range address names no
    // slot, so nothing is touched — clamping into a neighbouring valid slot would bill one view's
    // matrix to another, which renders wrongly instead of failing.
    assert(inRange(group, slot) && "shadow view slot out of range");
    assert(group != ShadowViewGroup::WorldOnly &&
           "world-only stores nothing — it aliases its cascade (enableWorldOnly)");
    if (!inRange(group, slot) || group == ShadowViewGroup::WorldOnly)
    {
        return false;
    }

    // Kind is checked, not descriptor validity: an INVALID descriptor of the right kind is engaged
    // and reported (the view rasterises, selection says InvalidView), but a perspective descriptor
    // in a cascade slot is meaningless and would be read back as authoritative.
    assert(projection.kind() == expectedKind &&
           "shadow view projection kind does not fit its slot");
    // SH-07: the same argument for the metrics. The three packings share a shape, so a mismatched
    // one is not rejected by any later check — it is simply read as different quantities and yields
    // a plausible, wrong bias.
    assert(biasMetrics.kind() == expectedMetricsKind &&
           "shadow view bias metrics do not fit their slot's projection");
    // An engaged entry always carries a usable identity, even when its projection is unusable:
    // hysteresis keys on the identity, so an invalid one would make the entry unkeyable while
    // still rendering.
    assert(logicalId.valid() && "an engaged shadow view needs a valid logical identity");
    // The MATRIX has the stricter contract: it is what the pass rasterises and what culling tests
    // against, and a non-finite one cannot do either job — so it is not a view, however good the
    // descriptor beside it looks.
    assert(allFinite(viewProj) && "a shadow view's render matrix must be finite");
    if (projection.kind() != expectedKind || biasMetrics.kind() != expectedMetricsKind ||
        !logicalId.valid() || !allFinite(viewProj))
    {
        // The addressed slot is CLEARED, not merely left unwritten. A rejected write is a producer
        // that tried and failed to describe this view; keeping whatever was there before would
        // silently rasterise the previous attempt's fit — the failure has to surface as a missing
        // view (a cascade's absence asserts in extraction) rather than as stale geometry.
        views_[shadowViewIndex(group, slot)].reset();
        return false;
    }
    views_[shadowViewIndex(group, slot)] =
        ShadowRenderView{viewProj, projection, biasMetrics, logicalId};
    return true;
}

bool ShadowRenderViewSet::setCascade(std::uint32_t index, const Mat4& viewProj,
                                     const ShadowView& projection,
                                     const ShadowViewMetrics& biasMetrics) noexcept
{
    return store(ShadowViewGroup::Cascade, index, ShadowViewKind::Orthographic,
                 ShadowViewMetricsKind::Orthographic, viewProj, projection, biasMetrics,
                 ShadowLogicalViewId::cascade(index));
}

bool ShadowRenderViewSet::enableWorldOnly(std::uint32_t index) noexcept
{
    // Sets a BIT; `find` then reads the cascade's entry through it. Nothing is copied, so a later
    // setCascade on the same index moves both passes together — a copy taken here would only be
    // equal at the instant it was taken.
    assert(inRange(ShadowViewGroup::WorldOnly, index) && "world-only cascade index out of range");
    if (!inRange(ShadowViewGroup::WorldOnly, index))
    {
        return false;
    }
    const bool cascadePopulated = find(ShadowViewGroup::Cascade, index) != nullptr;
    assert(cascadePopulated && "world-only needs its cascade populated first");
    worldOnlyActive_[index] = cascadePopulated;
    return cascadePopulated;
}

bool ShadowRenderViewSet::setSelf(std::size_t slot, std::uint32_t objectId, const Mat4& viewProj,
                                  const ShadowView& projection,
                                  const ShadowViewMetrics& biasMetrics) noexcept
{
    return store(ShadowViewGroup::Self, slot, ShadowViewKind::Orthographic,
                 ShadowViewMetricsKind::Orthographic, viewProj, projection, biasMetrics,
                 ShadowLogicalViewId::self(objectId));
}

bool ShadowRenderViewSet::setSpot(std::size_t slot, NodeId light, const Mat4& viewProj,
                                  const ShadowView& projection,
                                  const ShadowViewMetrics& biasMetrics) noexcept
{
    return store(ShadowViewGroup::Spot, slot, ShadowViewKind::Perspective,
                 ShadowViewMetricsKind::Spot, viewProj, projection, biasMetrics,
                 ShadowLogicalViewId::spot(light));
}

bool ShadowRenderViewSet::setPointLight(
    std::size_t lightSlot, NodeId light, const ShadowViewMetrics& biasMetrics,
    std::span<const ShadowPointFace, kCubeFaceCount> faces) noexcept
{
    // ADDRESS first, before any flattening: `lightSlot * kCubeFaceCount + face` can wrap a huge
    // slot back into the valid range, aliasing a real light's faces instead of being rejected. Like
    // any out-of-range address this leaves every real slot untouched.
    assert(lightSlot < static_cast<std::size_t>(kMaxPointShadowCasters) &&
           "point light slot out of range");
    if (lightSlot >= static_cast<std::size_t>(kMaxPointShadowCasters))
    {
        return false;
    }

    // VALIDATE THE WHOLE CUBE before storing any of it. A partially-installed cube is the state
    // this writer exists to make unrepresentable: five faces rasterise and the sixth reads whatever
    // the previous frame left, which is a shadow that is wrong only when the light is looked at
    // from one direction.
    const ShadowLogicalViewId identity = ShadowLogicalViewId::point(light, 0);
    bool acceptable = identity.valid() && biasMetrics.kind() == ShadowViewMetricsKind::PointLight;
    for (const ShadowPointFace& face : faces)
    {
        acceptable = acceptable && face.projection.kind() == ShadowViewKind::Perspective &&
                     allFinite(face.viewProj);
    }
    assert(acceptable && "a point light's cube needs six perspective faces, finite matrices, a "
                         "valid identity and PointLight metrics");
    if (!acceptable)
    {
        // Cleared, not left unwritten — see store(): whatever was there is a previous attempt's
        // fit, and rasterising that silently is worse than the view going missing.
        for (std::uint8_t face = 0; face < kCubeFaceCount; ++face)
        {
            views_[shadowViewIndex(ShadowViewGroup::Point, shadowPointViewSlot(lightSlot, face))]
                .reset();
        }
        return false;
    }

    for (std::uint8_t face = 0; face < kCubeFaceCount; ++face)
    {
        // ONE derivation of the flat slot and the identity's face, from the same loop variable, and
        // ONE metrics value copied into all six — which is what makes "the faces cannot disagree
        // about the light" true of the type rather than of the caller.
        views_[shadowViewIndex(ShadowViewGroup::Point, shadowPointViewSlot(lightSlot, face))] =
            ShadowRenderView{faces[face].viewProj, faces[face].projection, biasMetrics,
                             ShadowLogicalViewId::point(light, face)};
    }
    return true;
}

const ShadowRenderView* ShadowRenderViewSet::find(ShadowViewGroup group,
                                                  std::size_t slot) const noexcept
{
    assert(inRange(group, slot) && "shadow view slot out of range");
    if (!inRange(group, slot))
    {
        return nullptr;
    }
    if (group == ShadowViewGroup::WorldOnly)
    {
        // THE alias. Resolved on every read rather than snapshotted, so the world-only pass sees
        // whatever its cascade currently holds — including a cascade re-fitted after it was
        // enabled — and the two can never rasterise different matrices.
        return worldOnlyActive_[slot] ? find(ShadowViewGroup::Cascade, slot) : nullptr;
    }
    const auto& entry = views_[shadowViewIndex(group, slot)];
    return entry.has_value() ? &*entry : nullptr;
}

bool ShadowRenderViewSet::active(ShadowViewGroup group, std::size_t slot) const noexcept
{
    return find(group, slot) != nullptr;
}

std::size_t ShadowRenderViewSet::activeCount(ShadowViewGroup group) const noexcept
{
    if (static_cast<std::size_t>(group) >= kShadowViewGroupCount)
    {
        assert(false && "shadow view group out of range");
        return 0;
    }
    std::size_t count = 0;
    for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
    {
        count += active(group, slot) ? 1 : 0;
    }
    return count;
}

std::array<Mat4, static_cast<std::size_t>(kShadowTotalMatrixCount)>
shadowMatrixArray(const ShadowRenderViewSet& views) noexcept
{
    std::array<Mat4, static_cast<std::size_t>(kShadowTotalMatrixCount)> matrices;
    matrices.fill(Mat4::identity());

    for (std::size_t cascade = 0; cascade < shadowViewSlotCount(ShadowViewGroup::Cascade);
         ++cascade)
    {
        const ShadowRenderView* view = views.find(ShadowViewGroup::Cascade, cascade);
        // Cascades are mandatory: the directional pass always runs, so a missing one means the fit
        // did not happen — unlike a punctual or self slot, which is legitimately inactive.
        assert(view != nullptr && "every cascade must be populated before extraction");
        if (view != nullptr)
        {
            matrices[static_cast<std::size_t>(kShadowCascadeMatrixBase) + cascade] =
                view->viewProj();
        }
    }
    for (std::size_t spot = 0; spot < shadowViewSlotCount(ShadowViewGroup::Spot); ++spot)
    {
        if (const ShadowRenderView* view = views.find(ShadowViewGroup::Spot, spot))
        {
            matrices[static_cast<std::size_t>(kShadowSpotMatrixBase) + spot] = view->viewProj();
        }
    }
    for (std::size_t flat = 0; flat < shadowViewSlotCount(ShadowViewGroup::Point); ++flat)
    {
        if (const ShadowRenderView* view = views.find(ShadowViewGroup::Point, flat))
        {
            matrices[static_cast<std::size_t>(kShadowPointMatrixBase) + flat] = view->viewProj();
        }
    }
    // World-only contributes nothing: its view IS the cascade's entry (an alias), so the cascade
    // slot written above IS its matrix. There is no second value to reconcile.
    return matrices;
}

std::array<Mat4, kShadowCascadeCount>
cascadeViewProjArray(const ShadowRenderViewSet& views) noexcept
{
    std::array<Mat4, kShadowCascadeCount> out;
    out.fill(Mat4::identity());
    for (std::size_t cascade = 0; cascade < out.size(); ++cascade)
    {
        const ShadowRenderView* view = views.find(ShadowViewGroup::Cascade, cascade);
        assert(view != nullptr && "every cascade must be populated before extraction");
        if (view != nullptr)
        {
            out[cascade] = view->viewProj();
        }
    }
    return out;
}

std::array<Mat4, static_cast<std::size_t>(kMaxSpotShadowCasters)>
spotViewProjArray(const ShadowRenderViewSet& views) noexcept
{
    std::array<Mat4, static_cast<std::size_t>(kMaxSpotShadowCasters)> out;
    out.fill(Mat4::identity());
    for (std::size_t spot = 0; spot < out.size(); ++spot)
    {
        if (const ShadowRenderView* view = views.find(ShadowViewGroup::Spot, spot))
        {
            out[spot] = view->viewProj();
        }
    }
    return out;
}

std::array<Mat4, static_cast<std::size_t>(kMaxSkinnedSelfShadowCasters)>
selfShadowViewProjArray(const ShadowRenderViewSet& views) noexcept
{
    std::array<Mat4, static_cast<std::size_t>(kMaxSkinnedSelfShadowCasters)> out;
    out.fill(Mat4::identity());
    for (std::size_t slot = 0; slot < out.size(); ++slot)
    {
        if (const ShadowRenderView* view = views.find(ShadowViewGroup::Self, slot))
        {
            out[slot] = view->viewProj();
        }
    }
    return out;
}

namespace
{

// One family's packed metrics, zero-filled where the slot is inactive. Zeros are not a neutral
// default here, they are the "no metrics" the bias law answers with no bias — visible acne rather
// than a silently invented scale.
template <std::size_t N>
[[nodiscard]] std::array<std::array<float, 4>, N> packedMetrics(const ShadowRenderViewSet& views,
                                                                ShadowViewGroup group) noexcept
{
    std::array<std::array<float, 4>, N> out{};
    for (std::size_t slot = 0; slot < N; ++slot)
    {
        if (const ShadowRenderView* view = views.find(group, slot); view != nullptr)
        {
            out[slot] = view->biasMetrics().packed();
        }
    }
    return out;
}

} // namespace

std::array<std::array<float, 4>, kShadowCascadeCount>
cascadeBiasMetricsArray(const ShadowRenderViewSet& views) noexcept
{
    return packedMetrics<kShadowCascadeCount>(views, ShadowViewGroup::Cascade);
}

std::array<std::array<float, 4>, static_cast<std::size_t>(kMaxSkinnedSelfShadowCasters)>
selfBiasMetricsArray(const ShadowRenderViewSet& views) noexcept
{
    return packedMetrics<static_cast<std::size_t>(kMaxSkinnedSelfShadowCasters)>(
        views, ShadowViewGroup::Self);
}

std::array<std::array<float, 4>, static_cast<std::size_t>(kMaxSpotShadowCasters)>
spotBiasMetricsArray(const ShadowRenderViewSet& views) noexcept
{
    return packedMetrics<static_cast<std::size_t>(kMaxSpotShadowCasters)>(views,
                                                                          ShadowViewGroup::Spot);
}

std::array<std::array<float, 4>, static_cast<std::size_t>(kMaxPointShadowCasters)>
pointBiasMetricsArray(const ShadowRenderViewSet& views) noexcept
{
    // FACE 0 of each light. Reading one face is sound because `setPointLight` installs the cube
    // atomically with ONE metrics value — the faces cannot disagree, and a cube cannot be missing
    // face 0 while holding another.
    std::array<std::array<float, 4>, static_cast<std::size_t>(kMaxPointShadowCasters)> out{};
    for (std::size_t lightSlot = 0; lightSlot < static_cast<std::size_t>(kMaxPointShadowCasters);
         ++lightSlot)
    {
        const ShadowRenderView* view =
            views.find(ShadowViewGroup::Point, shadowPointViewSlot(lightSlot, 0));
        if (view != nullptr)
        {
            out[lightSlot] = view->biasMetrics().packed();
        }
    }
    return out;
}

} // namespace fire_engine
