#include "fire_engine/graphics/shadow_pass_plan.hpp"

#include <algorithm>
#include <type_traits>
#include <utility>

namespace fire_engine
{

namespace
{

// EXACT equality, component by component — `Vec3` offers no comparison and this is deliberately not
// an approximate one. A tolerance here would decide that a light which moved slightly still holds
// the same shadow map, which is a policy (and a wrong one: the texels differ), not a comparison.
[[nodiscard]] bool sameVec3(const Vec3& lhs, const Vec3& rhs) noexcept
{
    return lhs.x() == rhs.x() && lhs.y() == rhs.y() && lhs.z() == rhs.z();
}

} // namespace

bool PreparedShadowDraw::sameContent(const PreparedShadowDraw& other) const noexcept
{
    // A deformable draw never compares equal — not even to itself. Its vertices are rewritten
    // between frames by a skin, a morph or a compute pass, and every field here would still match.
    // Answering "equal" would be answering a question this struct cannot see the inputs to.
    if (deformable || other.deformable)
    {
        return false;
    }
    if (casterId != other.casterId || generation != other.generation || !(model == other.model) ||
        vertexBuffer != other.vertexBuffer || indexBuffer != other.indexBuffer ||
        indexCount != other.indexCount || indexType != other.indexType || alpha != other.alpha ||
        cull != other.cull)
    {
        return false;
    }
    // The material only reaches a MASKED caster's fragment shader. Comparing it for an opaque draw
    // would reject a reuse for a value that path never reads — two opaque variants of one mesh,
    // differing only in material, store identical depth.
    return alpha != ShadowCasterAlpha::Masked || materialIndex == other.materialIndex;
}

PreparedShadowView PreparedShadowView::projected(ShadowLogicalViewId logicalId,
                                                 const Mat4& viewProj, std::uint32_t extent,
                                                 float depthBiasConstant,
                                                 float depthBiasSlope) noexcept
{
    // A point face cannot store projected depth: its fragment stage overwrites gl_FragDepth with
    // the radial ratio whatever this says. Returning an invalid view makes the mismatch impossible
    // to build rather than merely wrong once built.
    if (!logicalId.valid() || logicalId.kind() == ShadowLogicalViewKind::Point)
    {
        return PreparedShadowView{};
    }
    PreparedShadowView view{};
    view.logicalId_ = logicalId;
    view.viewProj_ = viewProj;
    view.extent_ = extent;
    view.depthBiasConstant_ = depthBiasConstant;
    view.depthBiasSlope_ = depthBiasSlope;
    view.buildLayers();
    return view;
}

PreparedShadowView PreparedShadowView::pointFace(ShadowLogicalViewId logicalId,
                                                 const Mat4& viewProj, std::uint32_t extent,
                                                 float depthBiasConstant, float depthBiasSlope,
                                                 Vec3 lightPosition, float lightRange) noexcept
{
    if (!logicalId.valid() || logicalId.kind() != ShadowLogicalViewKind::Point)
    {
        return PreparedShadowView{};
    }
    PreparedShadowView view{};
    view.logicalId_ = logicalId;
    view.viewProj_ = viewProj;
    view.extent_ = extent;
    view.depthBiasConstant_ = depthBiasConstant;
    view.depthBiasSlope_ = depthBiasSlope;
    view.buildLayers();
    view.lightPosition_ = lightPosition;
    view.lightRange_ = lightRange;
    return view;
}

void PreparedShadowView::buildLayers() noexcept
{
    // TOPOLOGY FROM IDENTITY. Every view gets its layers at construction, empty or not, because the
    // recorder walks layers to know what to clear: a view whose layers appeared only when a draw
    // did would leave a first-use empty cascade with undefined depth while the plan reported it
    // sampleable. Self is the only family with two — its dual-depth pair — and it gets both here,
    // so "all layers or none" is a property of the type.
    //
    // Allocation-free: the layers are a fixed-capacity array, so this only names them and says how
    // many there are. That is what keeps the factories `noexcept` truthfully.
    layers_[0].kind = ShadowLayerKind::Depth;
    layers_[0].draws.clear();
    layers_[1].kind = ShadowLayerKind::SelfSecondDepth;
    layers_[1].draws.clear();
    layerCount_ = logicalId_.kind() == ShadowLogicalViewKind::Self ? 2 : 1;
}

bool PreparedShadowView::addDraw(ShadowLayerKind kind, const PreparedShadowDraw& draw)
{
    // Only the layers this view HAS — `layers_` always holds two, and `layerCount_` says how many
    // of them are real. Scanning the array instead would let a cascade accept a self-second draw
    // into storage nothing records.
    for (std::size_t i = 0; i < layerCount_; ++i)
    {
        if (layers_[i].kind == kind)
        {
            layers_[i].draws.push_back(draw);
            return true;
        }
    }
    return false;
}

bool PreparedShadowView::sameContent(const PreparedShadowView& other) const noexcept
{
    if (logicalId_ != other.logicalId_ || !(viewProj_ == other.viewProj_) ||
        extent_ != other.extent_ || depthBiasConstant_ != other.depthBiasConstant_ ||
        depthBiasSlope_ != other.depthBiasSlope_)
    {
        return false;
    }
    // The light the stored ratio is measured against — a shader input for point faces only, so it
    // is compared only for the mode that reads it. A point light moving with an unchanged face
    // matrix changes every texel; a cascade carries no light position at all. The MODE itself needs
    // no comparison: it is derived from the identity, which was compared above.
    if (depthMode() == ShadowDepthMode::RadialRatio &&
        (!sameVec3(lightPosition_, other.lightPosition_) || lightRange_ != other.lightRange_))
    {
        return false;
    }
    // No cull POLICY here: the raster input is the per-draw effective cull below, and the policy
    // only helps produce it — so the policy is not part of this type at all.
    //
    // EVERY layer, not just the first: a self-shadow view's two depth images are both its content,
    // and a comparison that stopped at one would reuse a first layer whose second had changed.
    return std::ranges::equal(layers(), other.layers(),
                              [](const PreparedShadowLayer& lhs, const PreparedShadowLayer& rhs)
                              { return lhs.sameContent(rhs); });
}

bool PreparedShadowView::cacheable() const noexcept
{
    return std::ranges::all_of(layers(),
                               [](const PreparedShadowLayer& layer)
                               {
                                   return std::ranges::none_of(layer.draws,
                                                               [](const PreparedShadowDraw& draw)
                                                               { return draw.deformable; });
                               });
}

bool PreparedShadowLayer::sameContent(const PreparedShadowLayer& other) const noexcept
{
    if (kind != other.kind)
    {
        return false;
    }
    return std::ranges::equal(draws, other.draws,
                              [](const PreparedShadowDraw& lhs, const PreparedShadowDraw& rhs)
                              { return lhs.sameContent(rhs); });
}

namespace
{

// Does this logical identity belong in this physical slot? Every family's answer, in one place.
[[nodiscard]] bool placementValid(ShadowViewGroup group, std::size_t slot,
                                  const ShadowLogicalViewId& id) noexcept
{
    switch (group)
    {
    case ShadowViewGroup::Cascade:
    case ShadowViewGroup::WorldOnly:
        // World-only deliberately SHARES the cascade identity (that is what makes the two passes
        // agree), so both groups want a Cascade kind — and the index must be the slot, or a cascade
        // would rasterise into another cascade's layer with its own matrix.
        return id.kind() == ShadowLogicalViewKind::Cascade && id.id() == slot;
    case ShadowViewGroup::Self:
        return id.kind() == ShadowLogicalViewKind::Self;
    case ShadowViewGroup::Spot:
        return id.kind() == ShadowLogicalViewKind::Spot;
    case ShadowViewGroup::Point:
        // The face is part of the physical address: slot = lightSlot * 6 + face. A face landing in
        // the wrong slot of the right cube points the same light's matrix at another face's image.
        return id.kind() == ShadowLogicalViewKind::Point &&
               id.face() == slot % static_cast<std::size_t>(kCubeFaceCount);
    case ShadowViewGroup::Count:
        break;
    }
    return false;
}

} // namespace

void ShadowFramePlan::reset() noexcept
{
    for (Entry& entry : entries_)
    {
        entry = Entry{};
        entry.claimed = false;
    }
}

bool ShadowFramePlan::add(ShadowViewGroup group, std::size_t slot, PreparedShadowView view,
                          ShadowViewDisposition disposition)
{
    if (static_cast<std::size_t>(group) >= kShadowViewGroupCount ||
        slot >= shadowViewSlotCount(group))
    {
        return false;
    }
    // An invalid prepared view means a factory was handed an identity it cannot serve — a point
    // face asked for projected depth, or a default-constructed identity. Recording it would
    // rasterise from a matrix nobody vouched for, and caching it would compare content that
    // describes nothing.
    if (!view.valid())
    {
        return false;
    }
    // PLACEMENT. The recorder is about to trust this object exclusively — it has no view set to
    // cross-check against — so an identity in the wrong physical slot has to be refused here.
    // Nothing downstream could notice: a cascade identity in a spot slot rasterises the cascade's
    // matrix into the spot map, and every counter and timing still reads plausibly.
    if (!placementValid(group, slot, view.logicalId()))
    {
        return false;
    }
    // ONE LIGHT PER PHYSICAL CUBE. The six faces of a cube are one light's map; two lights sharing
    // a cube would each render half of it and both sample all of it. `setPointLight` makes this
    // atomic upstream, and the plan is a second producer of the same arrangement, so it checks too
    // rather than assuming the caller preserved it.
    if (group == ShadowViewGroup::Point)
    {
        const auto faces = static_cast<std::size_t>(kCubeFaceCount);
        const std::size_t cubeBase = (slot / faces) * faces;
        for (std::size_t face = 0; face < faces; ++face)
        {
            const Entry& sibling = entries_[shadowViewIndex(group, cubeBase + face)];
            if (sibling.view.valid() && sibling.view.logicalId().id() != view.logicalId().id())
            {
                return false;
            }
        }
    }
    // ONE CLAIM PER SLOT. A second producer writing the same physical view means two views are
    // being prepared as one — the plan would record the last writer's work and cache it under the
    // other's identity — so it is refused rather than resolved, exactly as the view set and the
    // caster-bounds frame refuse a duplicate key. An earlier Invalid claim counts: it is still a
    // producer having spoken for the slot.
    Entry& entry = entries_[shadowViewIndex(group, slot)];
    if (entry.claimed)
    {
        return false;
    }
    entry.claimed = true;
    // An entry with nothing to sample carries no content: keeping the prepared work for an Invalid
    // disposition would let a later reader treat "engaged but unusable" as a description of the
    // image.
    entry.disposition = disposition;
    entry.view =
        disposition == ShadowViewDisposition::Invalid ? PreparedShadowView{} : std::move(view);
    return true;
}

const PreparedShadowView* ShadowFramePlan::view(ShadowViewGroup group,
                                                std::size_t slot) const noexcept
{
    if (static_cast<std::size_t>(group) >= kShadowViewGroupCount ||
        slot >= shadowViewSlotCount(group))
    {
        return nullptr;
    }
    const Entry& entry = entries_[shadowViewIndex(group, slot)];
    return entry.view.valid() ? &entry.view : nullptr;
}

std::optional<PreparedShadowView> ShadowFramePlan::takeRecorded(ShadowViewGroup group,
                                                                std::size_t slot) noexcept
{
    if (static_cast<std::size_t>(group) >= kShadowViewGroupCount ||
        slot >= shadowViewSlotCount(group))
    {
        return std::nullopt;
    }
    Entry& entry = entries_[shadowViewIndex(group, slot)];
    // RECORDED ONLY. A reused view did not touch its image, so its prepared work must not replace
    // the record of what that image holds; an invalid slot rasterised nothing at all.
    if (entry.disposition != ShadowViewDisposition::Recorded)
    {
        return std::nullopt;
    }
    std::optional<PreparedShadowView> taken{std::move(entry.view)};
    entry = Entry{};
    return taken;
}

ShadowViewDisposition ShadowFramePlan::disposition(ShadowViewGroup group,
                                                   std::size_t slot) const noexcept
{
    if (static_cast<std::size_t>(group) >= kShadowViewGroupCount ||
        slot >= shadowViewSlotCount(group))
    {
        return ShadowViewDisposition::Invalid;
    }
    return entries_[shadowViewIndex(group, slot)].disposition;
}

std::size_t ShadowFramePlan::sampleableCount(ShadowViewGroup group) const noexcept
{
    std::size_t count = 0;
    for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
    {
        if (shadowViewSampleable(disposition(group, slot)))
        {
            ++count;
        }
    }
    return count;
}

bool ShadowFramePlan::records(ShadowViewGroup group) const noexcept
{
    for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
    {
        if (shadowViewRecords(disposition(group, slot)))
        {
            return true;
        }
    }
    return false;
}

bool ShadowFramePlan::pointCubesWhole() const noexcept
{
    const auto faces = static_cast<std::size_t>(kCubeFaceCount);
    const std::size_t slots = shadowViewSlotCount(ShadowViewGroup::Point);
    for (std::size_t cube = 0; cube * faces < slots; ++cube)
    {
        std::size_t sampleable = 0;
        for (std::size_t face = 0; face < faces; ++face)
        {
            if (shadowViewSampleable(disposition(ShadowViewGroup::Point, cube * faces + face)))
            {
                ++sampleable;
            }
        }
        if (sampleable != 0 && sampleable != faces)
        {
            return false;
        }
    }
    return true;
}

bool ShadowFramePlan::recordsNothing() const noexcept
{
    for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
    {
        if (records(static_cast<ShadowViewGroup>(g)))
        {
            return false;
        }
    }
    return true;
}

ShadowMapValidity ShadowFamilyEligibility::eligible() const noexcept
{
    const auto count = [this](ShadowViewGroup group)
    { return activeViews[static_cast<std::size_t>(group)]; };
    return shadowMapValidity(ShadowMapValidityInputs{
        .shadowsDisabled = shadowsDisabled,
        .primaryDirectionalLight = primaryDirectionalLight,
        .activeCascadeViews = count(ShadowViewGroup::Cascade),
        .activeWorldOnlyViews = count(ShadowViewGroup::WorldOnly),
        .activeSelfViews = count(ShadowViewGroup::Self),
        .activeSpotViews = count(ShadowViewGroup::Spot),
        .activePointViews = count(ShadowViewGroup::Point),
    });
}

ShadowMapValidity shadowMapValidityFromPlan(const ShadowFramePlan& plan,
                                            const ShadowFamilyEligibility& eligibility) noexcept
{
    ShadowMapValidity validity = eligibility.eligible();

    // EVERY ELIGIBLE VIEW MUST HAVE MADE IT. Comparing against the EXPECTED count is what catches
    // the variable-size failures: one of two spots preparing leaves a sampleable count of 1, which
    // the eligibility law alone would call a valid family while the other light samples depth from
    // whenever its map was last written. A reused view counts as arrived — it is sampleable, which
    // is the point — so this measures completeness, not work.
    const auto complete = [&](ShadowViewGroup group)
    {
        return plan.sampleableCount(group) ==
               eligibility.activeViews[static_cast<std::size_t>(group)];
    };
    validity.cascades = validity.cascades && complete(ShadowViewGroup::Cascade);
    validity.worldOnly = validity.worldOnly && complete(ShadowViewGroup::WorldOnly);
    validity.self = validity.self && complete(ShadowViewGroup::Self);
    validity.spot = validity.spot && complete(ShadowViewGroup::Spot);
    // Point additionally needs its cubes WHOLE. Counting cannot see a half-prepared cube beside a
    // fully prepared one: six of twelve faces is a whole number of cubes by arithmetic and a light
    // missing three faces in fact.
    validity.point = validity.point && complete(ShadowViewGroup::Point) && plan.pointCubesWhole();
    return validity;
}

ShadowViewDisposition shadowViewDisposition(bool active, ShadowReusePolicy reuse,
                                            const PreparedShadowView& prepared,
                                            const ShadowViewResidency& resident) noexcept
{
    if (!active)
    {
        return ShadowViewDisposition::Invalid;
    }
    // The toggle is asked AFTER engagement and before anything about content: a view nothing
    // vouches for stays Invalid whatever the policy says (there is no work to schedule), while a
    // view that would have been reused simply records instead. Nothing else changes — the same
    // draws, the same order, the same commit afterwards.
    if (reuse == ShadowReusePolicy::Disabled)
    {
        return ShadowViewDisposition::Recorded;
    }
    // FIRST USE. Image creation transitions every layer to the sampler's read-only layout but
    // leaves the depth contents undefined, so "already in the right layout" is not "already holds
    // an answer". A slot with no committed content records, whatever its prepared work looks like.
    const PreparedShadowView* residentContent = resident.content();
    if (residentContent == nullptr)
    {
        return ShadowViewDisposition::Recorded;
    }
    if (!prepared.cacheable())
    {
        return ShadowViewDisposition::Recorded;
    }
    // The resident side is checked too, not just the prepared one: content recorded when the view
    // held a deformable caster describes geometry that has since moved, so it can never be matched
    // against — even if this frame's set happens to be static.
    if (!residentContent->cacheable())
    {
        return ShadowViewDisposition::Recorded;
    }
    return prepared.sameContent(*residentContent) ? ShadowViewDisposition::Reused
                                                  : ShadowViewDisposition::Recorded;
}

const ShadowViewResidency& ShadowResidencyStore::at(ShadowViewGroup group,
                                                    std::size_t slot) const noexcept
{
    // "Nothing resident" is a real state every entry starts in, so an out-of-range address is
    // answered with it rather than with a separate failure the law would have to learn about. The
    // consequence is a re-record, which is the safe direction.
    static const ShadowViewResidency kNothingResident{};
    if (static_cast<std::size_t>(group) >= kShadowViewGroupCount ||
        slot >= shadowViewSlotCount(group))
    {
        return kNothingResident;
    }
    return entries_[shadowViewIndex(group, slot)];
}

void ShadowResidencyStore::commit(ShadowFramePlan& plan) noexcept
{
    // The whole reason adoption is a move: this runs after `submitFrame`, so an allocation
    // failure here would throw out of a frame the GPU is already executing — and the throw would
    // leave residency describing the frame BEFORE this one while the images hold this one's depth,
    // which is precisely the "shadows from a frame that is gone" failure the type exists to
    // prevent. A moved prepared view allocates nothing, and these assertions are what keep that
    // true if someone gives it a member whose move can throw.
    static_assert(
        std::is_nothrow_move_constructible_v<PreparedShadowView>,
        "residency is adopted after the submit, so moving a prepared view must not throw");
    static_assert(
        std::is_nothrow_move_assignable_v<PreparedShadowView>,
        "residency is adopted after the submit, so moving a prepared view must not throw");

    for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
    {
        const auto group = static_cast<ShadowViewGroup>(g);
        for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
        {
            if (auto recorded = plan.takeRecorded(group, slot); recorded.has_value())
            {
                entries_[shadowViewIndex(group, slot)].commit(std::move(*recorded));
            }
        }
    }
}

} // namespace fire_engine
