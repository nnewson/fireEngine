#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/shadow_caster_alpha.hpp>
#include <fire_engine/graphics/shadow_diagnostics.hpp>
#include <fire_engine/graphics/shadow_face_cull.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>
#include <fire_engine/graphics/shadow_lod_resolver.hpp>
#include <fire_engine/graphics/shadow_map_validity.hpp>
#include <fire_engine/graphics/shadow_view_disposition.hpp>
#include <fire_engine/math/mat4.hpp>

// What a shadow view will RASTERISE this frame, described exactly enough to decide whether last
// frame's depth image is still the right answer (arc 2 #4 / §2.1).
//
// The whole difficulty of caching a shadow map is the comparison, not the skipping. A map may be
// reused only when every input that produced its pixels is unchanged, so this describes the draws
// in terms of the values that reach the GPU — the model matrix written to ShadowUBO, the resolved
// index buffer, the effective cull mode — and not in terms of the higher-level quantities that
// explain them. Two different transforms can share an AABB; a snapped cascade origin plus a
// near/far pair explains a matrix without being one; a LOD level names a choice without being the
// geometry that choice selected. Each of those would compare equal while rasterising different
// pixels.
//
// STRUCTURAL COMPARISON, not a hash. A 64-bit digest of this content would be a probabilistic
// correctness argument for a decision that silently produces a wrong image, and "wrong image" here
// means shadows from a frame that no longer exists. A hash may be added later as an ACCELERATOR in
// front of the equality test; it may never replace it.
//
// Vulkan-free, so the comparison and the disposition law are testable without a device — which is
// the only practical way to exercise the cases that matter (a moved caster, a re-fitted cascade, a
// swapped LOD carrier, a deformable in the set).

namespace fire_engine
{

// One draw as it will be recorded. Every field here is either an input to the rasteriser or the
// identity of one; the diagnostic-only fields are called out below and excluded from equality.
struct PreparedShadowDraw
{
    // Identity, so a reloaded caster that happens to land the same matrix cannot pass as unchanged.
    ShadowCasterId casterId{ShadowCasterId::Invalid};
    ShadowCasterGeneration generation{ShadowCasterGeneration::First};
    // The EXACT matrix the recorder writes into ShadowUBO. Not the caster's bounds and not its node
    // transform: those are what produce this, and two of them can produce the same box.
    //
    // (The ShadowUBO buffer HANDLE is deliberately absent — it is a per-frame-ring handle, so
    // identical content alternates handles every frame and would defeat the cache entirely.)
    Mat4 model{};
    BufferHandle vertexBuffer{NullBuffer};
    // The RESOLVED carrier and count — what the LOD decision actually selected. Compared instead of
    // trusting the level, because the level explains the choice while this is the geometry.
    BufferHandle indexBuffer{NullBuffer};
    std::uint32_t indexCount{0};
    DrawIndexType indexType{DrawIndexType::UInt16};
    // SH-05: which fragment path rasterises this caster, and — for a MASKED one only — the material
    // that path samples. The opaque path reads no material data at all, so the index is compared
    // only once the alpha classes already agree and that class is `Masked`: two opaque variants of
    // one mesh differing solely in material produce identical depth and must reuse.
    //
    // The material's SLOT CONTENT is immutable once registered (`registerMaterial` writes the
    // packed block on first registration and dedups by identity), so the index fully describes it.
    // If materials ever become mutable, a revision joins this struct.
    ShadowCasterAlpha alpha{ShadowCasterAlpha::Opaque};
    std::uint32_t materialIndex{0};
    // The EFFECTIVE cull mode, already resolved against the family policy and the caster's
    // sidedness — the value the recorder sets, from the same pure function it uses.
    ShadowEffectiveCull cull{ShadowEffectiveCull::FrontFaces};
    // Poisons reuse for the whole view. A skinned, morph-capable or storage-vertex caster rewrites
    // its vertices with no revision anything here can compare: same buffers, same matrix, different
    // pixels. Arc 2 #5's deformation revision is what turns this exclusion into a comparison.
    bool deformable{false};

    // DIAGNOSTICS ONLY, excluded from equality (see `sameContent`). The level and reason describe
    // the decision; `indexBuffer`/`indexCount` above describe its result, and it is the result that
    // makes pixels. Comparing the level as well would reject a reuse that is genuinely identical
    // (two levels resolving to one carrier), which is a correctness-neutral loss but a real one.
    std::size_t level{0};
    ShadowLodReason reason{ShadowLodReason::Count};

    // RECORDING PAYLOAD, also excluded from equality: the per-frame-ring buffer handles the
    // recorder pushes as set 0. They alternate every frame for identical content — which is exactly
    // why the content descriptor above cannot contain them — but the recorder still needs them, and
    // carrying them here is what lets it consume prepared draws alone instead of reaching back to a
    // DrawCommand and re-deriving what to draw.
    BufferHandle shadowUbo{NullBuffer};
    BufferHandle skinUbo{NullBuffer};
    BufferHandle morphUbo{NullBuffer};
    BufferHandle morphSsbo{NullBuffer};

    // Pixel-producing equality. Deliberately a named function rather than `operator==`, so nobody
    // reaches for a defaulted comparison that would silently start including the diagnostic fields.
    [[nodiscard]] bool sameContent(const PreparedShadowDraw& other) const noexcept;
};

// How a view's fragments produce the depth they store. NOT derivable from `viewProj`: a point face
// overwrites `gl_FragDepth` with a linear distance/range ratio computed from the light position and
// range in the push constants (`shaders/shadow_depth.glsl`), so two faces with identical matrices
// store different depth when the light moves or its range changes.
enum class ShadowDepthMode : std::uint8_t
{
    // Cascade / world-only / spot / self: fixed-function hardware depth from the projection.
    Projected,
    // Point faces: `length(worldPos - lightPos) / range`, clamped. The comparison sampler tests
    // that same ratio, which is why the pass writes it rather than letting the face projection
    // decide.
    RadialRatio,
};

// The ONE derivation of a view's depth mode from its identity. `PreparedShadowView::depthMode()` is
// this, and so is the recorder's `radialDepth` push constant — a second mapping would let the
// comparison and the shader disagree about what a map holds.
[[nodiscard]] constexpr ShadowDepthMode shadowDepthModeFor(ShadowLogicalViewKind kind) noexcept
{
    return kind == ShadowLogicalViewKind::Point ? ShadowDepthMode::RadialRatio
                                                : ShadowDepthMode::Projected;
}

// The push-constant spelling of that mode (`ShadowPushConstants::radialDepth`).
[[nodiscard]] constexpr int shadowRadialDepthFlag(ShadowDepthMode mode) noexcept
{
    return mode == ShadowDepthMode::RadialRatio ? 1 : 0;
}

// WHICH depth image of a view a layer fills. ONE enum, not a pass plus a target: the physical image
// and the fragment path are both functions of the family and this kind, so carrying them separately
// would make `SelfSecondDepth` writing image 0, or two layers claiming one image, expressible
// states that mean nothing.
enum class ShadowLayerKind : std::uint8_t
{
    // The view's depth map. Every family has exactly one, including the self-shadow FIRST layer,
    // which captures the nearest light-facing surface.
    Depth,
    // Self-shadow only: the second depth layer, which samples the first and discards the surface it
    // already recorded so the forward pass can sample the next useful occluder. A different
    // fragment shader and a different cull, into a different image.
    SelfSecondDepth,
};

// One depth image's worth of draws. Most views have exactly one layer; a self-shadow view has TWO —
// the same logical view rasterised into two images — which is why a layer exists at all rather than
// a view being a flat draw list.
//
// Its KIND is fixed when the view is created and the draws are the only thing that varies, so the
// topology of a view is a property of its identity rather than of the order somebody appended in.
struct PreparedShadowLayer
{
    ShadowLayerKind kind{ShadowLayerKind::Depth};
    std::vector<PreparedShadowDraw> draws{};

    [[nodiscard]] bool sameContent(const PreparedShadowLayer& other) const noexcept;
};

// One view's prepared work: the transform and target it rasterises into, its per-view depth inputs,
// and the layers of draws in the order they will be recorded.
//
// ENCAPSULATED, like `ShadowView` and `ShadowLogicalViewId`, and for the same class of reason. As a
// public aggregate, a POINT face could be assembled with `Projected` depth: the comparison would
// then omit the light position and range while the shader still took its radial branch, so a moved
// light would keep a stale cube. The depth mode is therefore not a field at all — it is DERIVED
// from the logical view's kind, which the view set already guarantees matches the physical family
// it was written into. The factories enforce the other half: only a point identity may carry a
// light, and a point identity may not be prepared without one.
class PreparedShadowView
{
public:
    // Default is an INVALID view — never a usable one. Present so residency and containers can hold
    // one, not so a caller can fill it in field by field.
    PreparedShadowView() = default;

    // Fixed-function hardware depth: cascade, world-only, spot, self. Rejected (invalid result) for
    // a point identity, which cannot store projected depth.
    [[nodiscard]] static PreparedShadowView projected(ShadowLogicalViewId logicalId,
                                                      const Mat4& viewProj, std::uint32_t extent,
                                                      float depthBiasConstant,
                                                      float depthBiasSlope) noexcept;
    // One cube face, storing `length(worldPos - lightPosition) / lightRange`. Rejected (invalid
    // result) for any non-point identity: nothing else writes that ratio, so nothing else has a
    // light to be compared against.
    [[nodiscard]] static PreparedShadowView pointFace(ShadowLogicalViewId logicalId,
                                                      const Mat4& viewProj, std::uint32_t extent,
                                                      float depthBiasConstant, float depthBiasSlope,
                                                      Vec3 lightPosition,
                                                      float lightRange) noexcept;

    // False for a default-constructed value and for a factory given a mismatched identity. A plan
    // containing one is a producer bug, not a degraded frame — the caller must refuse it.
    [[nodiscard]] bool valid() const noexcept
    {
        return logicalId_.valid();
    }
    // The MATRIX, not the fit that produced it. A cascade's snapped origin, extent and SH-06
    // caster-aware near/far all explain this value; reconstructing equivalence from them is a
    // second derivation that can disagree with the one the GPU sees.
    [[nodiscard]] const Mat4& viewProj() const noexcept
    {
        return viewProj_;
    }
    [[nodiscard]] std::uint32_t extent() const noexcept
    {
        return extent_;
    }
    [[nodiscard]] float depthBiasConstant() const noexcept
    {
        return depthBiasConstant_;
    }
    [[nodiscard]] float depthBiasSlope() const noexcept
    {
        return depthBiasSlope_;
    }
    // DERIVED from the identity, so "a point face prepared as projected" cannot be expressed.
    [[nodiscard]] ShadowDepthMode depthMode() const noexcept
    {
        return shadowDepthModeFor(logicalId_.kind());
    }
    // The light the stored ratio is measured against — a SHADER INPUT
    // (`ShadowPushConstants::lightPosRange`), not a consequence of `viewProj`. Zero for every
    // projected view, which carries none.
    [[nodiscard]] const Vec3& lightPosition() const noexcept
    {
        return lightPosition_;
    }
    [[nodiscard]] float lightRange() const noexcept
    {
        return lightRange_;
    }
    // The logical view this content belongs to. A physical slot is reassigned between frames (dense
    // per-family assignment in gather order), so residency keyed only by slot could match one
    // light's content against another's.
    [[nodiscard]] const ShadowLogicalViewId& logicalId() const noexcept
    {
        return logicalId_;
    }
    [[nodiscard]] std::span<const PreparedShadowLayer> layers() const noexcept
    {
        return std::span<const PreparedShadowLayer>{layers_.data(), layerCount_};
    }
    // The FIRST layer's draws — the only layer every family has. A convenience for callers that are
    // not the recorder (tests, diagnostics); the recorder walks `layers()` so a second layer cannot
    // be silently skipped.
    [[nodiscard]] std::span<const PreparedShadowDraw> draws() const noexcept
    {
        return layerCount_ == 0 ? std::span<const PreparedShadowDraw>{}
                                : std::span<const PreparedShadowDraw>{layers_.front().draws};
    }

    // Appends to a layer this view HAS. Returns false if it has no layer of that kind — asking a
    // cascade for its second self-shadow depth is a producer bug, not a draw to drop silently.
    //
    // There is no way to add a LAYER: topology comes from the identity at construction, so an empty
    // view still has its layers and the recorder still clears them. A view whose layers appeared
    // only when a draw did would let a first-use empty cascade be recorded, walk nothing, and leave
    // the image's depth undefined while the plan called it sampleable.
    [[nodiscard]] bool addDraw(ShadowLayerKind kind, const PreparedShadowDraw& draw);
    // The common case: the view's own depth layer.
    [[nodiscard]] bool addDraw(const PreparedShadowDraw& draw)
    {
        return addDraw(ShadowLayerKind::Depth, draw);
    }

    // ORDER-SENSITIVE, and conservatively so. Depth-only rasterisation of these draws is
    // order-independent in the image it produces, so a reordered but otherwise identical set would
    // be a false miss — one wasted re-record, never a wrong image. Comparing order-insensitively
    // would mean sorting or hashing per frame to save a case that does not arise: the draw order
    // follows the scene's stable gather order.
    [[nodiscard]] bool sameContent(const PreparedShadowView& other) const noexcept;

    // No draw may deform. One is enough to poison the view: its pixels change every frame with
    // nothing here able to see it.
    [[nodiscard]] bool cacheable() const noexcept;

private:
    // Gives the view the layers its identity implies — one, or two for self. Called by the
    // factories, so there is no view in existence without them.
    void buildLayers() noexcept;

    Mat4 viewProj_{};
    std::uint32_t extent_{0};
    float depthBiasConstant_{0.0f};
    float depthBiasSlope_{0.0f};
    Vec3 lightPosition_{};
    float lightRange_{0.0f};
    ShadowLogicalViewId logicalId_{};
    // FIXED CAPACITY, because the topology is: one layer, or two for self. An array keeps
    // construction allocation-free, which is what lets the factories be honestly `noexcept` — a
    // `noexcept` function that grows a vector terminates on a bad allocation instead of propagating
    // it — and it drops a per-view allocation from every frame's preparation.
    static constexpr std::size_t kMaxLayers = 2;
    std::array<PreparedShadowLayer, kMaxLayers> layers_{};
    std::size_t layerCount_{0};
};

// One physical view slot's committed content — what its depth image actually holds.
//
// COMMITTED is the load-bearing word. A frame that prepared content and then never submitted (an
// out-of-date swapchain, a throw) must not leave a record claiming the image holds it, so content
// is staged during the frame and adopted only after the queue accepts the work — the same
// discipline the SH-03 hysteresis history already follows.
//
// Absence is STRUCTURAL: there is no content-plus-flag pair to get out of step, and no way to
// express "resident" with default-constructed content. An empty residency means the image's depth
// is undefined (creation transitions the layout but writes nothing), which is the one case where
// being in the right layout is not the same as holding an answer.
class ShadowViewResidency
{
public:
    [[nodiscard]] bool hasContent() const noexcept
    {
        return content_.has_value();
    }
    // Null when nothing is committed. A pointer rather than a reference so the empty case has to be
    // handled at the call site.
    [[nodiscard]] const PreparedShadowView* content() const noexcept
    {
        return content_.has_value() ? &*content_ : nullptr;
    }
    // Adopt content AFTER the frame that recorded it was submitted — and only for a view whose
    // disposition was `Recorded`. A `Reused` view did not touch its image, so committing its
    // prepared work would replace the record of what the image holds with a description of a frame
    // that never wrote to it. The two are equal in every compared field by construction (that is
    // why it was reused), but not in the diagnostic ones, and the resident record must keep
    // describing the recording that actually produced the depth.
    void commit(PreparedShadowView content) noexcept
    {
        content_ = std::move(content);
    }
    // There is deliberately NO `invalidate()`. An image that is recreated takes its record with it,
    // because `ShadowResidencyStore` lives inside `Shadows` alongside the images themselves —
    // reconstruction IS the invalidation, so there is nothing for a caller to remember to call and
    // no window in which a store can disagree with the images it describes. If in-place recreation
    // ever arrives, the targets and the store move into one private aggregate together rather than
    // this hook coming back.

private:
    std::optional<PreparedShadowView> content_{};
};

// The law.
//
// `active` is the view set's answer (SH-03): a slot the set reports inactive is Invalid regardless
// of what its image holds, because nothing this frame vouches for the matrix behind it.
[[nodiscard]] ShadowViewDisposition
shadowViewDisposition(bool active, ShadowReusePolicy reuse, const PreparedShadowView& prepared,
                      const ShadowViewResidency& resident) noexcept;

// One frame's prepared work for every physical shadow view, plus what each will do.
//
// The single object the recorder receives. It exists so the pass has NOTHING else to consult: no
// draw spans to re-filter, no resolver to re-resolve against, no view set to re-read a matrix from.
// Whatever the comparison decided was the content is exactly what gets recorded, because it is the
// only description of the work that survives preparation.
//
// Indexed by the same physical `(group, slot)` as the diagnostics and the view set, so a row, a
// timing and a plan entry all name one view.
// TWO USES OF ONE LAW, in a fixed order (arc 2 #4).
//
//  1. ELIGIBILITY, derived from the view SET before anything is prepared. A family that cannot be
//     sampled must not be prepared at all: preparation resolves casters and STAGES hysteresis, so a
//     frame that prepared a suppressed family and was then submitted would commit dead-band
//     decisions for views whose maps were neither recorded nor sampled. Deriving validity from the
//     finished plan is too late to prevent that — by then the resolver has already been asked.
//
//  2. CONFIRMATION, derived from the finished PLAN and uploaded to the shader. What the receiver is
//     told must describe the plan that was actually built, not the intention that preceded it: a
//     view that failed to prepare (a rejected identity) has to remove its family's bit even though
//     the family was eligible.
//
// CONFIRMATION NEEDS THE EXPECTED COUNTS, not just the achieved ones. Re-applying the eligibility
// law to the plan alone loses the information that matters for variable-size families: two active
// spots of which one prepared leaves `sampleableCount == 1`, which satisfies "any active slot", and
// the light that failed would sample stale depth. Twelve point faces of which six prepared still
// satisfy "a whole number of cubes". Validity is FAMILY-WIDE, so the question is not "is this a
// plausible family" but "did every view this family was eligible for actually make it".
//
// This snapshot is taken from the view SET before preparation, and is the thing preparation is
// judged against afterwards.
struct ShadowFamilyEligibility
{
    bool shadowsDisabled{false};
    bool primaryDirectionalLight{false};
    // Active views per family, indexed by ShadowViewGroup — what preparation is expected to
    // produce.
    std::array<std::size_t, kShadowViewGroupCount> activeViews{};

    // The eligibility answer itself: which families may be prepared at all. Preparation must
    // consult this BEFORE resolving anything, because resolving stages hysteresis for views that a
    // suppressed family will never record or sample.
    [[nodiscard]] ShadowMapValidity eligible() const noexcept;
};

// What the receiver is told, derived from the finished plan and judged against the eligibility that
// authorised it. A family is valid only when it was eligible AND every view it was eligible for is
// sampleable in the plan.
[[nodiscard]] ShadowMapValidity
shadowMapValidityFromPlan(const class ShadowFramePlan& plan,
                          const ShadowFamilyEligibility& eligibility) noexcept;

class ShadowFramePlan
{
public:
    // Start a frame: every slot unclaimed, every entry empty.
    //
    // This does NOT preserve the draw vectors' capacity — the entries are cleared, so each frame's
    // views allocate again. The cost is bounded by the number of physical views (a few dozen small
    // vectors per frame) and is not currently measured; if it ever shows up, the fix is for the
    // plan to own the storage and hand a slot out to be filled in place, not to cache capacity
    // behind a reset that claims more than it does.
    void reset() noexcept;

    // Record one view's prepared work and its disposition. Returns FALSE for an invalid prepared
    // view (a factory given a mismatched identity), an out-of-range slot, or a slot ALREADY CLAIMED
    // this frame — each a producer bug the caller must refuse, not a frame to degrade through,
    // exactly as a rejected view-set write is.
    [[nodiscard]] bool add(ShadowViewGroup group, std::size_t slot, PreparedShadowView view,
                           ShadowViewDisposition disposition);

    // Null when this slot has no entry this frame.
    [[nodiscard]] const PreparedShadowView* view(ShadowViewGroup group,
                                                 std::size_t slot) const noexcept;
    // `Invalid` for a slot with no entry — absent and inactive are the same answer to the pass.
    [[nodiscard]] ShadowViewDisposition disposition(ShadowViewGroup group,
                                                    std::size_t slot) const noexcept;

    // How many of a family's slots hold content that may be SAMPLED (recorded or reused). This is
    // what `ShadowMapValidity` consumes: a CSM with two cascades recorded and two reused is fully
    // valid, so counting only the recorded ones would blank the family in the shader.
    [[nodiscard]] std::size_t sampleableCount(ShadowViewGroup group) const noexcept;
    // Whether any of a family's slots RECORDS. Drives the family's timestamps: a family that reuses
    // everything does no GPU work and must not open a timing span around nothing.
    [[nodiscard]] bool records(ShadowViewGroup group) const noexcept;
    // Whether the whole frame records nothing — the pass can then return immediately.
    [[nodiscard]] bool recordsNothing() const noexcept;

    // Every physical point cube either has all six faces sampleable or none of them. A count alone
    // cannot see a half-prepared cube beside a whole one, and half a cube is a light whose shadow
    // depends on which way the receiver faces.
    [[nodiscard]] bool pointCubesWhole() const noexcept;

    // Hands this slot's RECORDED content over to residency, emptying the entry.
    //
    // A move, not a copy, and that is a correctness property rather than an optimisation: adoption
    // happens AFTER the queue has accepted the frame, where a throwing allocation would abandon
    // work the GPU is already executing. Moving a prepared view allocates nothing (the static
    // asserts in the .cpp pin that), so the post-submit path cannot fail.
    //
    // Empty for any other disposition, so the store's "Recorded only" rule is expressed once more
    // in the type that owns the content. The entry is CLEARED rather than left holding a moved-from
    // view: the plan is reset at the start of the next preparation, and until then it should say
    // the content is gone instead of describing a husk.
    [[nodiscard]] std::optional<PreparedShadowView> takeRecorded(ShadowViewGroup group,
                                                                 std::size_t slot) noexcept;

private:
    struct Entry
    {
        PreparedShadowView view{};
        ShadowViewDisposition disposition{ShadowViewDisposition::Invalid};
        // Separate from `disposition`, because an Invalid claim is still a claim: without this, a
        // producer that legitimately claimed a slot as Invalid could be silently overwritten by a
        // second one, which is the case a plain "is it still default?" check would miss.
        bool claimed{false};
    };
    std::array<Entry, kShadowViewCount> entries_{};
};

// What every physical shadow view's depth image HOLDS — the frame-to-frame half of the cache, and
// the other operand of `shadowViewDisposition`.
//
// OWNED BY THE IMAGES' OWNER (`render/shadows.cpp`), never by the frame. A plan describes one
// frame's intent and is reset at the start of the next; this describes durable GPU content, so it
// lives beside the images it is a record of. That is also the whole invalidation story: there is no
// `invalidate()` to forget to call, because recreating the images means reconstructing the object
// that holds both them and this.
//
// Indexed by the same physical `(group, slot)` as the plan, the view set and the diagnostics, so a
// row, a timing, a plan entry and a residency entry all name one view.
class ShadowResidencyStore
{
public:
    // What this slot's image holds. An out-of-range address answers "nothing resident", which the
    // law turns into `Recorded` — the conservative direction: a spurious re-render costs a frame's
    // raster, while a spurious reuse shows shadows from a frame that is gone.
    [[nodiscard]] const ShadowViewResidency& at(ShadowViewGroup group,
                                                std::size_t slot) const noexcept;

    // Adopt what the frame actually recorded, CONSUMING it. Call AFTER the queue has accepted the
    // work: content committed by a frame that was abandoned would claim an image holds pixels
    // nothing ever drew. `noexcept`, because this runs on the far side of the submit — see
    // `ShadowFramePlan::takeRecorded`.
    //
    // RECORDED ONLY, and the filter lives here rather than at the call site so there is one place
    // that decides. A `Reused` view did not touch its image; committing its prepared work would
    // replace the record of what the image holds with a description of a frame that never wrote to
    // it — equal in every compared field, by construction, but no longer the recording that made
    // the depth. An `Invalid` slot is left alone for the same reason read the other way: nothing
    // recorded, so nothing overwrote the image, so the existing record is still true — clearing it
    // would force a re-record of content the image still holds.
    void commit(ShadowFramePlan& plan) noexcept;

private:
    std::array<ShadowViewResidency, kShadowViewCount> entries_{};
};

} // namespace fire_engine
