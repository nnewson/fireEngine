// The LightUBO block — the ONE declaration of it, included by every shader that binds the buffer.
//
// Layout must match C++ `LightUBO` in include/fire_engine/render/ubo.hpp, whose static_asserts pin
// each offset below. This file exists because the block was previously written out by hand in each
// shader: when `selfShadowViewProj` was added to the C++ struct and to shader.frag, skybox.frag's
// copy was not updated, so every field after `spotViewProj` was read 256 bytes early and the sky was
// multiplied by `selfShadowViewProj[1][0][0]` instead of `environmentParams.x`. That read the value
// 1.0 whenever fewer than two skinned self-shadow casters existed — unused slots are identity — so
// the bug stayed invisible until a scene supplied two (RecursiveSkeletons, the only such asset).
// Sharing the block makes that drift unrepresentable rather than merely fixed.
//
// The includer must define LIGHT_UBO_SET and LIGHT_UBO_BINDING before including this file, because
// the buffer sits at a different descriptor address in the forward set than in the skybox set.

#ifndef FIRE_ENGINE_LIGHT_UBO_GLSL
#define FIRE_ENGINE_LIGHT_UBO_GLSL

#if !defined(LIGHT_UBO_SET) || !defined(LIGHT_UBO_BINDING)
#error "define LIGHT_UBO_SET and LIGHT_UBO_BINDING before including light_ubo.glsl"
#endif

// MAX_LIGHTS, MAX_SPOT_SHADOW_CASTERS, MAX_SKINNED_SELF_SHADOW_CASTERS, MAX_POINT_SHADOW_CASTERS
// and SHADOW_CASCADE_COUNT — the same declarations graphics/gpu_limits.hpp re-exports, not a
// transcription of them.
#include "gpu_limits.glsl"

struct LightData {
    // .xyz = world position (point/spot), .w = type (0=dir, 1=point, 2=spot)
    vec4 position;
    // .xyz = world forward, .w = range (point/spot; 0 = infinite). For point
    // shadow casters, .w is the effective range used by the shadow pass.
    vec4 direction;
    // .rgb = colour, .a = intensity
    vec4 colour;
    // .x = cos(innerCone), .y = cos(outerCone), .z = shadow index (-1 = none)
    vec4 cone;
};

layout(set = LIGHT_UBO_SET, binding = LIGHT_UBO_BINDING) uniform LightUBO {
    mat4 cascadeViewProj[SHADOW_CASCADE_COUNT];
    mat4 spotViewProj[MAX_SPOT_SHADOW_CASTERS];
    // Per-skinned-object self-shadow matrices, indexed by ForwardPushConstants::selfShadowSlot.
    // Unused slots are identity, NOT zero — see selfShadowViewProjArray.
    mat4 selfShadowViewProj[MAX_SKINNED_SELF_SHADOW_CASTERS];
    vec4 cascadeSplits;
    vec4 iblParams;
    // SH-07 bias policy in TEXELS: x = slopeScale, y = constantTexels, z = normalOffsetTexels,
    // w = maxSlopeTangent. Passed straight to shadowBiasFor(); the per-view conversion into stored
    // depth lives in the metrics arrays at the end of this block.
    vec4 shadowParams;
    // x = kSkyboxIntensity, y = kEnvironmentShadowStrength,
    // z = debug view (0=off, 1=normals, 2=NdotL, 3=shadow visibility,
    // 4=directional raw depth, 5=velocity, 6=SSAO, 7=LOD tint, 8=shadow-LOD tint),
    // w = RESERVED (0) — the old "disable shadow lookups" flag, replaced by shadowMapValidMask
    // below, which suppresses the RECORDING as well and so cannot disagree with the pass.
    vec4 environmentParams;
    // x = cascade cross-fade fraction (kShadowCascadeBlendFraction). Uploaded, not a literal here:
    // the renderer expands each cascade's fitted slice to cover the previous cascade's blend band,
    // so the number that decides the band and the number that fits for it must be one value.
    // y = PCF kernel radius in TEXELS, read both by the kernel's offsets and by shadowBiasFor (the
    // slope term has to clear the whole disc the filter samples, not just the centre texel).
    // z/w reserved.
    vec4 cascadeParams;
    int  lightCount;
    // Which shadow-map families were RECORDED this frame — SHADOW_MAP_VALID_* bits from
    // gpu_limits.glsl. A cleared bit means that family's depth image was not rendered and is stale;
    // its sampling path must return fully lit rather than read it. `--no-shadows` clears every bit.
    int  shadowMapValidMask;
    int  _pad1;
    int  _pad2;
    LightData lights[MAX_LIGHTS];
    // SH-07 per-view bias metrics, appended after lights[] so no offset above moved. Read the array
    // matching the map being sampled — the three packings differ:
    //   cascade / self : (worldUnitsPerTexel, normalizedDepthPerWorldUnit, 0, 0)
    //   spot           : (texelAngleScale, nearPlane, farPlane, 0)
    //   point          : (texelAxisScale, 1 / range, 0, 0)   — per LIGHT, not per face
    // Zeros mean "no metrics" (an inactive slot); shadow_bias.glsl answers those with no bias.
    vec4 cascadeBiasMetrics[SHADOW_CASCADE_COUNT];
    vec4 selfBiasMetrics[MAX_SKINNED_SELF_SHADOW_CASTERS];
    vec4 spotBiasMetrics[MAX_SPOT_SHADOW_CASTERS];
    vec4 pointBiasMetrics[MAX_POINT_SHADOW_CASTERS];
} light;

#endif // FIRE_ENGINE_LIGHT_UBO_GLSL
