#include <fire_engine/graphics/shadow_pass_prepare.hpp>

#include <format>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <fire_engine/graphics/frustum.hpp>

namespace fire_engine
{

namespace
{

// A diagnostic row that two logical views tried to claim in one frame, or a view prepared with no
// identity at all. Terminal because the alternative is a row whose counters are the sum of two
// unrelated views under one of their names — worse than no measurement, because it reads like one.
[[noreturn]] void contradictoryShadowViewRow(std::string_view group, std::size_t slot)
{
    throw std::runtime_error(
        std::format("shadow view row {} slot {} was claimed by two different logical views in one "
                    "frame (or by a view with no identity)",
                    group, slot));
}

// A shadow command that could not be resolved into geometry (SH-03). TERMINAL, on the same
// reasoning as the view set's rejections: the request is corrupt render input, and both ways of
// continuing are worse than stopping — dropping the draw leaves a caster missing from one shadow
// map with nothing to say so, and counting it as filtered corrupts the one metric the per-view
// diagnostics promise. In a Dev build the resolver's own assertion fires first, at the request that
// was malformed; under NDEBUG this throw carries the same refusal to main().
[[noreturn]] void unresolvableShadowCaster(std::uint32_t objectId)
{
    throw std::runtime_error(
        std::format("shadow caster (objectId {}) resolved to no geometry — its unresolved command "
                    "carries no drawable base mesh",
                    objectId));
}

// A caster whose request never stated where it is. TERMINAL rather than degraded, and not for the
// selector's sake — it already reports an unusable pose as InvalidCaster and draws full detail —
// but for the CACHE's: an unstated pose carries a default matrix, which is the same value every
// frame, so the comparison would find the view unchanged while the GPU rasterised the caster's
// actual transform. That is a shadow map reused forever for something that is moving, with no
// symptom anywhere. A producer that forgot the pose has to be stopped, not compensated for.
[[noreturn]] void unstatedShadowCasterPose(std::uint32_t objectId)
{
    throw std::runtime_error(
        std::format("shadow caster (objectId {}) has no stated pose — its request carries no model "
                    "matrix for the pass to rasterise or the cache to compare",
                    objectId));
}

// A view the plan refused, or one a factory could not build from its identity. Both mean the
// producer described a view that cannot exist in the slot it named — a point face asked for
// projected depth, an identity in the wrong family's slot, two producers claiming one slot. Nothing
// downstream could notice: every counter and timing would still read plausibly while one light's
// matrix rasterised into another's map.
[[noreturn]] void unpreparableShadowView(std::string_view group, std::size_t slot)
{
    throw std::runtime_error(std::format(
        "shadow view {} slot {} could not be prepared — its identity does not fit the slot, or the "
        "slot was already claimed this frame",
        group, slot));
}

struct ShadowDrawFilter
{
    const Frustum* frustum{nullptr};
    int selfShadowSlot{-1};

    [[nodiscard]] bool accepts(const DrawCommand& dc) const
    {
        if (selfShadowSlot >= 0 && dc.selfShadowSlot != selfShadowSlot)
        {
            return false;
        }
        if (frustum == nullptr)
        {
            return true;
        }
        // A caster whose bounds are STALE (cloth: a compute pass rewrites the vertices this box was
        // measured from) cannot be rejected by them. The box says roughly where the caster was in
        // its bind pose and nothing about where the drawn geometry is, so a frustum test against it
        // can only produce false rejections — a cloth that is genuinely in this view, dropped. It
        // is admitted until storage geometry carries a conservative envelope of its own.
        if (dc.shadowBoundsKind != ShadowCasterBoundsKind::Exact)
        {
            return true;
        }
        return frustum->intersects(dc.shadowBounds);
    }
};

// Which caster set a family draws from. One mapping, so a family cannot be prepared from the wrong
// span — the world-only CSM exists precisely to exclude skinned casters, and handing it the full
// set would restore the geometry it was built to leave out.
[[nodiscard]] std::span<const DrawCommand> familyDraws(const ShadowPreparationInputs& inputs,
                                                       ShadowViewGroup group) noexcept
{
    switch (group)
    {
    case ShadowViewGroup::WorldOnly:
        return inputs.worldOnlyShadowDraws;
    case ShadowViewGroup::Self:
        return inputs.selfShadowDraws;
    case ShadowViewGroup::Cascade:
    case ShadowViewGroup::Spot:
    case ShadowViewGroup::Point:
        return inputs.shadowDraws;
    case ShadowViewGroup::Count:
        break;
    }
    return {};
}

// SH-05: which faces one prepared LAYER keeps, before the caster's own sidedness is folded in.
//
// Keyed on the family AND the layer, in that order, because only the self family's layers
// disagree: its first depth image captures whatever the light sees first (so it keeps every face,
// whatever the winding), and its second keeps only back faces, which is what makes the dual-depth
// rejection well-founded rather than a coin-flip on marginal fragments. Everywhere else the CASTER
// decides — and a double-sided one culls nothing, since front-culling a sheet authored face-on to
// the light discards the only faces it has.
//
// One function rather than a family test wrapping a layer test: a layer-only mapping would answer
// "keep every face" for a cascade's depth layer, which is a policy no cascade has ever had.
[[nodiscard]] ShadowFaceCull layerCullPolicy(ShadowViewGroup group, ShadowLayerKind kind) noexcept
{
    if (group != ShadowViewGroup::Self)
    {
        return ShadowFaceCull::PerCaster;
    }
    return kind == ShadowLayerKind::SelfSecondDepth ? ShadowFaceCull::BackFacesOnly
                                                    : ShadowFaceCull::AllFaces;
}

// Whether this family may be prepared at all — the eligibility answer, read per family so the
// decision is made once, above, rather than re-derived here.
[[nodiscard]] bool familyEligible(ShadowMapValidity eligible, ShadowViewGroup group) noexcept
{
    switch (group)
    {
    case ShadowViewGroup::Cascade:
        return eligible.cascades;
    case ShadowViewGroup::WorldOnly:
        return eligible.worldOnly;
    case ShadowViewGroup::Self:
        return eligible.self;
    case ShadowViewGroup::Spot:
        return eligible.spot;
    case ShadowViewGroup::Point:
        return eligible.point;
    case ShadowViewGroup::Count:
        break;
    }
    return false;
}

// One view's transform and depth inputs, from the set entry and the family's raster parameters. The
// POINT factory is chosen by the identity's kind, not by the group, so a face can only ever be
// prepared with the light its own descriptor was built from.
[[nodiscard]] PreparedShadowView prepareView(const ShadowRenderView& view,
                                             const ShadowFamilyRaster& raster) noexcept
{
    if (const std::optional<ShadowPointLightDepth> light = view.pointLightDepth())
    {
        return PreparedShadowView::pointFace(view.logicalId(), view.viewProj(), raster.extent,
                                             raster.depthBiasConstant, raster.depthBiasSlope,
                                             light->position, light->range);
    }
    return PreparedShadowView::projected(view.logicalId(), view.viewProj(), raster.extent,
                                         raster.depthBiasConstant, raster.depthBiasSlope);
}

// One accepted caster as it will be recorded: the values that reach the rasteriser, the identity of
// the caster they belong to, and the per-frame ring handles the recorder pushes.
[[nodiscard]] PreparedShadowDraw prepareDraw(const DrawCommand& dc,
                                             const ResolvedShadowDraw& resolved,
                                             ShadowFaceCull cullPolicy) noexcept
{
    const ShadowGeometryRequest& request = dc.shadowRequest;
    return PreparedShadowDraw{
        .casterId = request.casterId,
        .generation = request.generation,
        // The pose's matrix, which is the same `world` that was written into this draw's
        // `ShadowUBO::model` — not a second derivation of where the caster is.
        .model = request.pose.model(),
        .vertexBuffer = dc.vertexBuffer,
        // The RESOLVED carrier, never the command's: a shadow command carries none.
        .indexBuffer = resolved.indexBuffer,
        .indexCount = resolved.indexCount,
        .indexType = dc.indexType,
        .alpha = request.alpha,
        .materialIndex = dc.materialIndex,
        // The EFFECTIVE answer, from the same pure function the recorder's Vulkan translation
        // consumes — so the compared value and the state that is set are one decision.
        .cull = shadowEffectiveCull(cullPolicy, dc.doubleSided),
        // SH-04's classification IS the cacheability question: a caster deformed after its geometry
        // was measured rewrites its vertices with no revision any field here could compare.
        .deformable = request.deformation == ShadowCasterDeformation::Deformable,
        .level = resolved.level,
        .reason = resolved.reason,
        .shadowUbo = dc.shadowUbo,
        .skinUbo = dc.skinUbo,
        .morphUbo = dc.morphUbo,
        .morphSsbo = dc.morphSsbo,
    };
}

} // namespace

void prepareShadowFrame(const ShadowPreparationInputs& inputs, const ShadowRenderViewSet& views,
                        ShadowMapValidity eligible, const ShadowResidencyStore& residency,
                        ShadowLodResolver& resolver, ShadowFrameStats& stats, ShadowFramePlan& plan)
{
    plan.reset();

    // Family order matches the order the pass records in. That is DETERMINISM, not correctness — a
    // reversal would not change a single resolved level. A cascade and its world-only twin share
    // one logical view, so whichever is prepared first is the one whose answer the resolver's frame
    // cache hands to the other; but the key carries the caster id and generation as well as the
    // view, the world-only span is a subset of the same commands with the same bounds, and both
    // resolve against the same aliased view entry, so the second call gets the answer it would have
    // computed anyway. What the stable order buys is comparability: the frame cache and the staged
    // history fill in one order, and per-view observations accumulate in one order, so two runs of
    // the same scene produce the same diagnostics.
    for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
    {
        const auto group = static_cast<ShadowViewGroup>(g);
        if (!familyEligible(eligible, group))
        {
            // Not filtered, not resolved, not claimed. Resolving stages hysteresis, and a family
            // that will neither record nor be sampled must not leave decisions behind for the
            // commit to adopt.
            continue;
        }
        const std::span<const DrawCommand> draws = familyDraws(inputs, group);
        const ShadowFamilyRaster& raster = inputs.raster[g];

        for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
        {
            // The SET decides which physical views exist. A slot it reports inactive is not
            // prepared — there is no second opinion to consult, which is what makes "absent means
            // inactive" true at the point of use.
            const ShadowRenderView* view = views.find(group, slot);
            if (view == nullptr)
            {
                continue;
            }

            // CLAIM FIRST, before the caster set is walked, so a view that ends up drawing nothing
            // is still a row in the report — an empty map that is rendered is a finding, and once
            // maps can be reused an untouched claimed row is the normal case.
            ShadowViewStats& viewStats = stats.view(group, slot);
            if (!viewStats.claimView(view->logicalId()))
            {
                contradictoryShadowViewRow(toString(group), slot);
            }

            PreparedShadowView prepared = prepareView(*view, raster);
            if (!prepared.valid())
            {
                unpreparableShadowView(toString(group), slot);
            }

            // Culling frustum from the view's OWN matrix. Self layers pass everything through: they
            // are already restricted to one caster by the slot filter, so a frustum test would only
            // repeat it. Disabled culling passes everything through too.
            const std::optional<Frustum> frustum =
                inputs.cullingEnabled && group != ShadowViewGroup::Self
                    ? std::optional<Frustum>{Frustum::fromViewProj(view->viewProj())}
                    : std::nullopt;
            const ShadowDrawFilter filter{
                .frustum = frustum ? &*frustum : nullptr,
                .selfShadowSlot = group == ShadowViewGroup::Self ? static_cast<int>(slot) : -1};

            // PER LAYER, because a layer is a depth image: a self-shadow view rasterises the same
            // caster set into two of them with different face policies, and each one's candidates
            // are its own. The LOD decision is not — it belongs to the logical view — which is why
            // only the first layer counts a selection.
            const std::size_t layerCount = prepared.layers().size();
            for (std::size_t layer = 0; layer < layerCount; ++layer)
            {
                const ShadowLayerKind kind = prepared.layers()[layer].kind;
                const ShadowFaceCull cullPolicy = layerCullPolicy(group, kind);
                const bool countSelection = layer == 0;

                for (const DrawCommand& dc : draws)
                {
                    // FILTER FIRST, resolve second. Selecting for a caster this view is about to
                    // drop would give it a dead band against a view it never appears in — a skinned
                    // caster would accumulate hysteresis against every other object's self-shadow
                    // map — and would evaluate wholly-rejected perspective casters outside the
                    // domain the projection model is good for.
                    const bool accepted = filter.accepts(dc);
                    const ResolvedShadowDraw resolved =
                        accepted ? resolver.resolve(dc.shadowRequest, *view, dc.shadowBounds,
                                                    inputs.lodBudgetTexels, inputs.hysteresis)
                                 : ResolvedShadowDraw{};
                    // TERMINAL, before anything is counted. The resolver returns drawable geometry
                    // for any request that carries base geometry — including every recoverable
                    // fallback — so a non-drawable result means the producer emitted a caster it
                    // could not describe. Skipping it would drop the caster from this shadow map
                    // silently, and folding it into the observed verdict would make it
                    // indistinguishable from a cull rejection, breaking the promise that
                    // `candidateDraws - drawnDraws` is exactly the filter's yield.
                    if (accepted && !resolved.drawable())
                    {
                        unresolvableShadowCaster(dc.objectId);
                    }
                    if (accepted && !dc.shadowRequest.pose.stated())
                    {
                        unstatedShadowCasterPose(dc.objectId);
                    }
                    // One observation per walked command, carrying the FILTER's verdict — nothing
                    // else. The full-detail count is what this view was OFFERED; the resolved count
                    // is what it will pay.
                    viewStats.observe(dc.shadowRequest.baseIndexCount / 3, accepted,
                                      resolved.indexCount / 3,
                                      static_cast<std::uint32_t>(resolved.level), resolved.reason,
                                      countSelection);
                    if (!accepted)
                    {
                        continue;
                    }
                    if (!prepared.addDraw(kind, prepareDraw(dc, resolved, cullPolicy)))
                    {
                        // A layer this view does not have. Unreachable while the kinds come from
                        // the view's own topology, and terminal rather than dropped because the
                        // draw would otherwise vanish from a map the plan still calls complete.
                        unpreparableShadowView(toString(group), slot);
                    }
                    // Recorded beside the draw it belongs to, so "this family's map holds this
                    // caster" cannot become true for a caster that was only considered. It is
                    // CONTENT, not "drew this frame": a reused map holds exactly this geometry
                    // without rasterising anything.
                    resolver.noteContent(group, ShadowLodStateKey{dc.shadowRequest.casterId,
                                                                  dc.shadowRequest.generation,
                                                                  view->logicalId()});
                }
            }

            // The view is ACTIVE by construction — the set said so above — so the law's remaining
            // questions are the cache's own: is there resident content, and does it match. Note
            // this is asked AFTER every draw has been resolved and noted: preparation costs the
            // same whether the answer is reuse or not, because the answer cannot be known without
            // the work that produces it.
            const ShadowViewDisposition disposition =
                shadowViewDisposition(true,
                                      inputs.residencyReuseEnabled ? ShadowReusePolicy::Enabled
                                                                   : ShadowReusePolicy::Disabled,
                                      prepared, residency.at(group, slot));
            // The ROW records what was decided, so the panel can tell "recorded an empty map" from
            // "reused the empty map it recorded earlier" — two rows that are otherwise identical in
            // every counter, one of which is GPU work and the other the cache doing its job.
            if (!viewStats.noteDisposition(view->logicalId(), disposition))
            {
                contradictoryShadowViewRow(toString(group), slot);
            }
            if (!plan.add(group, slot, std::move(prepared), disposition))
            {
                unpreparableShadowView(toString(group), slot);
            }
        }
    }
}

} // namespace fire_engine
