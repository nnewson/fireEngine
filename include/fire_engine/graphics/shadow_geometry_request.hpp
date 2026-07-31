#pragma once

#include <cstdint>
#include <limits>
#include <span>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>

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
    // Conservative sigma_max of the model transform's linear part: the factor that carries an
    // object-space deviation into world space. Computed once per caster, not per view.
    //
    // Defaults to NaN, NOT to 0 or 1. Zero is a legitimate value — a singular transform really does
    // flatten every deviation to nothing — so a producer that filled every other field and forgot
    // this one would silently claim its caster has zero error and take the coarsest level in every
    // view. NaN forces InvalidCaster instead, while an explicitly computed zero still selects
    // normally.
    float worldScale{std::numeric_limits<float>::quiet_NaN()};
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

    // A request that can actually be resolved into a draw. False means the producer left it
    // unfilled — which must not silently become "full detail".
    [[nodiscard]] bool valid() const noexcept
    {
        return baseIndexBuffer != NullBuffer && baseIndexCount > 0 &&
               casterId != ShadowCasterId::Invalid;
    }
};

} // namespace fire_engine
