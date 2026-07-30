#include <fire_engine/graphics/shadow_render_view.hpp>

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

bool ShadowRenderViewSet::store(ShadowViewGroup group, std::size_t slot,
                                ShadowViewKind expectedKind, const Mat4& viewProj,
                                const ShadowView& projection,
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
    // An engaged entry always carries a usable identity, even when its projection is unusable:
    // hysteresis keys on the identity, so an invalid one would make the entry unkeyable while
    // still rendering.
    assert(logicalId.valid() && "an engaged shadow view needs a valid logical identity");
    // The MATRIX has the stricter contract: it is what the pass rasterises and what culling tests
    // against, and a non-finite one cannot do either job — so it is not a view, however good the
    // descriptor beside it looks.
    assert(allFinite(viewProj) && "a shadow view's render matrix must be finite");
    if (projection.kind() != expectedKind || !logicalId.valid() || !allFinite(viewProj))
    {
        // The addressed slot is CLEARED, not merely left unwritten. A rejected write is a producer
        // that tried and failed to describe this view; keeping whatever was there before would
        // silently rasterise the previous attempt's fit — the failure has to surface as a missing
        // view (a cascade's absence asserts in extraction) rather than as stale geometry.
        views_[shadowViewIndex(group, slot)].reset();
        return false;
    }
    views_[shadowViewIndex(group, slot)] = ShadowRenderView{viewProj, projection, logicalId};
    return true;
}

bool ShadowRenderViewSet::setCascade(std::uint32_t index, const Mat4& viewProj,
                                     const ShadowView& projection) noexcept
{
    return store(ShadowViewGroup::Cascade, index, ShadowViewKind::Orthographic, viewProj,
                 projection, ShadowLogicalViewId::cascade(index));
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
                                  const ShadowView& projection) noexcept
{
    return store(ShadowViewGroup::Self, slot, ShadowViewKind::Orthographic, viewProj, projection,
                 ShadowLogicalViewId::self(objectId));
}

bool ShadowRenderViewSet::setSpot(std::size_t slot, NodeId light, const Mat4& viewProj,
                                  const ShadowView& projection) noexcept
{
    return store(ShadowViewGroup::Spot, slot, ShadowViewKind::Perspective, viewProj, projection,
                 ShadowLogicalViewId::spot(light));
}

bool ShadowRenderViewSet::setPoint(std::size_t lightSlot, std::uint8_t face, NodeId light,
                                   const Mat4& viewProj, const ShadowView& projection) noexcept
{
    // ADDRESS first, and before flattening: `lightSlot * kCubeFaceCount + face` can wrap a huge
    // slot back into the valid range, which would alias a real light's face instead of being
    // rejected — and the flat index would then pass every check downstream. Like any out-of-range
    // address this leaves every real slot untouched.
    assert(face < kCubeFaceCount && "point face out of range");
    assert(lightSlot < static_cast<std::size_t>(kMaxPointShadowCasters) &&
           "point light slot out of range");
    if (face >= kCubeFaceCount || lightSlot >= static_cast<std::size_t>(kMaxPointShadowCasters))
    {
        return false;
    }
    // ONE derivation: the flat slot and the identity's face come from the same arguments, so the
    // entry cannot sit at one face while claiming another.
    return store(ShadowViewGroup::Point, shadowPointViewSlot(lightSlot, face),
                 ShadowViewKind::Perspective, viewProj, projection,
                 ShadowLogicalViewId::point(light, face));
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

} // namespace fire_engine
