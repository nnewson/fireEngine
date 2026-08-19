#pragma once

#include <cstdint>
#include <limits>
#include <span>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/singular_value.hpp>

namespace fire_engine
{

// SH-04: whether this caster's recorded error is a claim about the geometry that will actually be
// rasterised. It is a CLASSIFICATION, not a policy switch — the resolver decides what to do with
// it.
//
// The simplifier measures deviation on the mesh as authored: its bind pose, its base weights, the
// contents of its vertex buffer at build time. For anything deformed afterwards that measurement
// describes a mesh that is never drawn. Skinning does not merely perturb the error, it can amplify
// it without bound — a joint rotation carries a vertex arbitrarily far from where its rest-pose
// deviation was measured — so the number is not loose here, it is unfounded. SH-02 was careful to
// call the metric an estimate; this is the case where that word stops covering it.
enum class ShadowCasterDeformation : std::uint8_t
{
    // The rasterised geometry is the measured geometry, transformed rigidly. Error claims hold.
    Rigid,
    // Deformed after the measurement: skinned, morph-CAPABLE (independent of the current weights —
    // an all-zero weight set is a value, not a guarantee, and nothing stops it changing next
    // frame), or storage-vertex geometry whose vertices a compute pass rewrites (cloth).
    Deformable,
};

// SH-05: whether this caster's shadow silhouette is its geometry, or its geometry MINUS the
// fragments an alpha cutout discards.
//
// A MASK material's shadow is not the shadow of its triangles. A leaf card is a quad whose visible
// shape comes entirely from base-colour alpha, so a depth-only pass that samples no texture records
// the quad — not a slightly wrong leaf, a rectangle. Like ShadowCasterDeformation above this is a
// CLASSIFICATION and not a policy switch: the resolver decides what it means for LOD, and the
// shadow pass decides which fragment path rasterises it.
enum class ShadowCasterAlpha : std::uint8_t
{
    // Every rasterised fragment occludes. OPAQUE materials, and — deliberately — BLEND ones: what a
    // blended surface ought to cast (opaque / dithered / transmittance) is its own design decision,
    // and routing it through the cutout path would answer that question with "cutout" by accident.
    Opaque,
    // Fragments below the material's alphaCutoff must not occlude: the pass samples base-colour
    // alpha through this material's UV set, KHR_texture_transform, sampler and factor, and discards
    // on exactly the test the forward shader applies.
    Masked,
};

// The caster's world transform and the error scale DERIVED from it, as ONE constructed value.
//
// Two representations of one transform, so they are not two fields. The matrix is what the shadow
// pass rasterises with (it is written into `ShadowUBO::model`, and the cache compares it to decide
// whether a map still holds the right pixels); `worldScale` is its conservative sigma_max, which
// carries an object-space deviation into world space. Nothing may set one without the other: a
// pose with a stale scale would select levels for a transform the GPU is not using, and a pose with
// a stale matrix would compare a caster that has moved as unchanged.
//
// STATED-NESS IS EXPLICIT, and that is the point of the class. A defaulted `Mat4` is a real matrix
// — zero, or identity depending on the type's default — so a producer that filled every other field
// and forgot this one would hand the comparison a constant transform while the GPU rasterised the
// object's actual one, and every frame would compare equal: a shadow map reused forever for a
// caster that is moving. `stated()` distinguishes "no pose was supplied" (a producer bug, terminal
// where it is consumed) from "a pose was supplied and is degenerate" (a non-finite transform from a
// broken animation, which the selector already survives as InvalidCaster).
class ShadowCasterPose
{
public:
    // NOT STATED. Present so a request can be default-constructed at all; never a usable pose.
    ShadowCasterPose() = default;

    // The only way to state one. Derives the scale here rather than accepting it, so the two can
    // never describe different transforms.
    [[nodiscard]] static ShadowCasterPose fromModel(const Mat4& model) noexcept
    {
        ShadowCasterPose pose{};
        pose.model_ = model;
        pose.worldScale_ = largestSingularValue(linearPart(model));
        pose.stated_ = true;
        return pose;
    }

    [[nodiscard]] bool stated() const noexcept
    {
        return stated_;
    }
    // The matrix the shadow pass rasterises this caster with.
    [[nodiscard]] const Mat4& model() const noexcept
    {
        return model_;
    }
    // Conservative sigma_max of the model's linear part. NaN on an unstated pose, which the
    // selector reports as InvalidCaster rather than silently treating as "no error".
    [[nodiscard]] float worldScale() const noexcept
    {
        return worldScale_;
    }

private:
    Mat4 model_{};
    float worldScale_{std::numeric_limits<float>::quiet_NaN()};
    bool stated_{false};
};

// SH-03: a shadow caster described but NOT yet resolved to geometry.
//
// The whole point of the seam. A shadow command used to arrive with an index buffer already chosen
// from the camera, and every view then rasterised that one choice. Here the command carries only
// what the caster IS — its LOD chain, how its transform scales object-space error into world space,
// and who it is — and each view resolves that into its own draw. There is deliberately no index
// buffer on the command itself: an unresolved command that still carried one would be
// indistinguishable from a resolved one, and a view that forgot to resolve would silently draw the
// inherited level.
struct ShadowGeometryRequest
{
    // The caster's LOD chain, owned by the Geometry and stable for the frame. Empty or single-entry
    // chains are legitimate (cloth, storage-vertex meshes, meshes below the simplifier's threshold)
    // and resolve to the base buffers below.
    std::span<const GeometryLod> lods{};
    // Whole-mesh geometry: what a caster draws when there is nothing to select between, when LOD is
    // switched off, and whenever selection falls back. Not a "default level 0" — `lods[0]` may not
    // exist at all.
    BufferHandle baseIndexBuffer{NullBuffer};
    std::uint32_t baseIndexCount{0};
    // WHERE the caster is, and the scale that carries its object-space deviation into world space —
    // one value, because they are one transform (see `ShadowCasterPose`). Read by selection (the
    // scale) and by shadow-map caching (the matrix), which is why it must not be possible to supply
    // one without the other.
    //
    // Unstated by default, and an unstated pose carries a NaN scale, NOT 0 or 1. Zero is a
    // legitimate value — a singular transform really does flatten every deviation to nothing — so a
    // producer that forgot this field would otherwise claim its caster has zero error and take the
    // coarsest level in every view. NaN forces InvalidCaster instead, while an explicitly computed
    // zero still selects normally.
    ShadowCasterPose pose{};
    // Identity for hysteresis. The generation is part of the key so a reloaded or replaced shadow
    // geometry cannot inherit the previous chain's dead band.
    ShadowCasterId casterId{ShadowCasterId::Invalid};
    ShadowCasterGeneration generation{ShadowCasterGeneration::First};
    // Mirrors `FrameInfo::shadowLodEnabled` — the SHADOW switch, which SH-03 separated from the
    // forward `lodEnabled` so an A/B can isolate shadow selection. Carried rather than consulted
    // globally so one frame's commands cannot be resolved under two different answers.
    bool lodEnabled{true};
    // Deliberately defaults to Deformable, the SAFE answer, on the same principle as `worldScale`'s
    // NaN: a producer that forgot this field must not receive the optimistic one. Forgetting it
    // costs triangles and says so in the panel (every caster reporting DeformableFallback is a
    // loud, findable symptom); the opposite default would silently select levels on an error claim
    // nobody established, which is exactly the defect this field exists to close.
    ShadowCasterDeformation deformation{ShadowCasterDeformation::Deformable};
    // SH-05. Defaults to Masked, the safe answer, on the same principle as `deformation`'s
    // Deformable: the optimistic default is the one whose failure is silent. A rigid caster wrongly
    // classified Masked pins to full detail and pays a texture fetch per shadow fragment — visible
    // in the panel as AlphaMaskedFallback, and identical depth either way, because the material
    // authority publishes alphaCutoff 0 for everything that is not MASK. One wrongly classified
    // Opaque casts a solid rectangle for a leaf card and reports nothing at all.
    ShadowCasterAlpha alpha{ShadowCasterAlpha::Masked};

    // A request that can actually be resolved into a draw. False means the producer left it
    // unfilled — which must not silently become "full detail".
    //
    // The POSE is deliberately NOT part of this. A caster with no stated pose is still drawable —
    // the whole mesh, reported as InvalidCaster, which is the same degraded answer a non-finite
    // transform from a broken animation gets, and that path has to keep working. What an unstated
    // pose breaks is the shadow CACHE (a default matrix compares equal forever), and that is
    // checked where preparation builds the comparison, terminally.
    [[nodiscard]] bool valid() const noexcept
    {
        return baseIndexBuffer != NullBuffer && baseIndexCount > 0 &&
               casterId != ShadowCasterId::Invalid;
    }
};

} // namespace fire_engine
