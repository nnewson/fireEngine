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

// Mirrors gpu_limits.hpp: kMaxLights, kMaxSpotShadowCasters, kMaxSkinnedSelfShadowCasters.
const int MAX_LIGHTS = 8;
const int MAX_SPOT_SHADOW_CASTERS = 4;
const int MAX_SKINNED_SELF_SHADOW_CASTERS = 4;

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
    mat4 cascadeViewProj[4];
    mat4 spotViewProj[MAX_SPOT_SHADOW_CASTERS];
    // Per-skinned-object self-shadow matrices, indexed by ForwardPushConstants::selfShadowSlot.
    // Unused slots are identity, NOT zero — see selfShadowViewProjArray.
    mat4 selfShadowViewProj[MAX_SKINNED_SELF_SHADOW_CASTERS];
    vec4 cascadeSplits;
    vec4 iblParams;
    vec4 shadowParams;
    vec4 pointSpotShadowParams;
    // x = kSkyboxIntensity, y = kEnvironmentShadowStrength,
    // z = debug view (0=off, 1=normals, 2=NdotL, 3=shadow visibility,
    // 4=directional raw depth, 5=velocity, 6=SSAO, 7=LOD tint, 8=shadow-LOD tint),
    // w = disable all shadow-map visibility lookups when > 0.5.
    vec4 environmentParams;
    // x = cascade cross-fade fraction (kShadowCascadeBlendFraction). Uploaded, not a literal here:
    // the renderer expands each cascade's fitted slice to cover the previous cascade's blend band,
    // so the number that decides the band and the number that fits for it must be one value.
    vec4 cascadeParams;
    int  lightCount;
    int  _pad0;
    int  _pad1;
    int  _pad2;
    LightData lights[MAX_LIGHTS];
} light;

#endif // FIRE_ENGINE_LIGHT_UBO_GLSL
