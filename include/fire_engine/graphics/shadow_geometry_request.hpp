#pragma once

#include <cstdint>
#include <limits>
#include <span>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>

namespace fire_engine
{

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
    // Mirrors RenderTunables::lodEnabled at the time the command was built. Carried rather than
    // consulted globally so one frame's commands cannot be resolved under two different answers.
    bool lodEnabled{true};

    // A request that can actually be resolved into a draw. False means the producer left it
    // unfilled — which must not silently become "full detail".
    [[nodiscard]] bool valid() const noexcept
    {
        return baseIndexBuffer != NullBuffer && baseIndexCount > 0 &&
               casterId != ShadowCasterId::Invalid;
    }
};

} // namespace fire_engine
