#pragma once

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/render/constants.hpp>

namespace fire_engine
{

struct UniformBufferObject
{
    Mat4 model;
    Mat4 view;
    Mat4 proj;
    alignas(16) float cameraPos[4];
    alignas(4) int hasSkin{0};
    int _pad1{0};
    int _pad2{0};
    int _pad3{0};
};

// KHR_texture_transform packed per material texture slot. `offsetScale.xy` is
// the UV offset; `offsetScale.zw` is the UV scale (identity = 0,0,1,1).
// `rotation` is radians CCW. Layout matches the std140 stride (16-byte vec4
// + float, padded to 32 bytes) of the matching GLSL struct in shader.frag.
struct UvXform
{
    alignas(16) float offsetScale[4]{0.0f, 0.0f, 1.0f, 1.0f};
    float rotation{0.0f};
    float _pad[3]{};
};

struct MaterialUBO
{
    alignas(16) float diffuseAlpha[4]{};
    alignas(16) float emissiveRoughness[4]{};
    alignas(16) float materialParams[4]{};
    alignas(16) int textureFlags[4]{};
    // .x = occlusion-texture present flag (legacy), .y = occlusion's UV-set
    // index (0 or 1). Other slots' UV-set indices live in texCoordIndices.
    alignas(16) int extraFlags[4]{};
    // glTF allows each material texture slot to point at TEXCOORD_0 or
    // TEXCOORD_1. Defaults are 0 everywhere — assets without the per-slot
    // override read TEXCOORD_0 as before. Layout: x=baseColor, y=emissive,
    // z=normal, w=metallicRoughness. Occlusion lives in extraFlags.y.
    alignas(16) int texCoordIndices[4]{};
    // KHR_materials_transmission + KHR_materials_ior. .x = transmissionFactor;
    // .y = transmission texture present (0 / 1); .z = transmission texCoord
    // index (0 / 1); .w = ior (KHR_materials_ior; default 1.5 per spec).
    alignas(16) float transmissionParams[4]{0.0f, 0.0f, 0.0f, 1.5f};
    // KHR_materials_clearcoat. .x = factor, .y = roughness, .z = normalScale,
    // .w reserved.
    alignas(16) float clearcoatParams[4]{0.0f, 0.0f, 1.0f, 0.0f};
    // .x = factor texture present, .y = roughness texture present,
    // .z = normal texture present, .w reserved (all 0 / 1 floats).
    alignas(16) float clearcoatFlags[4]{};
    // .x = factor texCoord, .y = roughness texCoord, .z = normal texCoord,
    // .w reserved (as floats — saves an alignas slot vs int4).
    alignas(16) float clearcoatTexCoords[4]{};
    // KHR_materials_volume.
    //   .x = thicknessFactor (world units, scaled in shader by node max scale)
    //   .y = thickness texture present (0/1)
    //   .z = thickness texCoord index (0/1)
    //   .w reserved (thickness rotation lives in uv[Thickness].rotation).
    alignas(16) float volumeParams[4]{};
    // .rgb = attenuationColor (default 1,1,1 — no absorption).
    // .a   = attenuationDistance in world units. Default packs the spec's
    //        +infinity as a very large finite number so the shader's
    //        exp(-coeff * d) collapses to 1 for thick surfaces without
    //        propagating inf through GLSL.
    alignas(16) float attenuation[4]{1.0f, 1.0f, 1.0f, 1.0e6f};
    // KHR_texture_transform per material texture slot. Indexed by
    // MaterialTextureSlot enum order (BaseColour..Thickness, 0..9).
    alignas(16) UvXform uv[10]{};
};

struct SkinUBO
{
    Mat4 joints[kMaxJoints];
};

struct MorphUBO
{
    alignas(4) int hasMorph{0};
    alignas(4) int morphTargetCount{0};
    alignas(4) int vertexCount{0};
    int _pad0{0};
    float weights[kMaxMorphTargets]{};
};

struct SkyboxUBO
{
    alignas(16) float cameraForward[4]{};
    alignas(16) float cameraRight[4]{};
    alignas(16) float cameraUp[4]{};
    alignas(16) float viewParams[4]{}; // x = tanHalfFov, y = aspect
};

struct EnvironmentCaptureUBO
{
    alignas(4) int faceIndex{0};
    alignas(4) int faceExtent{0};
    int _pad1{0};
    int _pad2{0};
};

// Per-light std140 entry packed into LightUBO::lights[]. Field semantics:
//   position.xyz  — world-space position (point/spot)
//   position.w    — type tag (0 = directional, 1 = point, 2 = spot)
//   direction.xyz — world-space forward (directional/spot)
//   direction.w   — range (0 = infinite; point/spot only)
//   colour.rgb    — RGB
//   colour.a      — intensity (scalar multiplier)
//   cone.x        — cos(innerCone)
//   cone.y        — cos(outerCone)
//   cone.z        — shadow index. For spot lights, layer in spot 2D-array
//                   shadow map (0..kMaxSpotShadowCasters-1). For point
//                   lights, cube layer in point cubemap-array shadow map
//                   (0..kMaxPointShadowCasters-1). -1 = no shadow caster.
//                   Stored as float; cast int() in shader.
struct LightData
{
    alignas(16) float position[4]{};
    alignas(16) float direction[4]{};
    alignas(16) float colour[4]{};
    alignas(16) float cone[4]{1.0f, 0.0f, -1.0f, 0.0f};
};

struct LightUBO
{
    // Per-cascade light-space view-projection matrices. Computed against the
    // first directional light in `lights[]` if any; otherwise against a
    // default direction so the matrices stay valid for the shadow pass.
    alignas(16) Mat4 cascadeViewProj[4]{};
    // Spot-light view-projection matrices for shadow sampling. Indexed by
    // LightData::cone.z (shadow index). Identity when the slot is unused.
    alignas(16) Mat4 spotViewProj[kMaxSpotShadowCasters]{};
    // Per-skinned-object self-shadow matrices. Indexed by ForwardPushConstants::selfShadowSlot.
    alignas(16) Mat4 selfShadowViewProj[kMaxSkinnedSelfShadowCasters]{};
    // View-space far-plane distances for each cascade (x..w = cascades 0..3).
    alignas(16) float cascadeSplits[4]{};
    alignas(16) float iblParams[4]{}; // x = maxReflectionLod, y/z = IBL strengths
    // x = csm minBias, y = csm slopeBias, z = filterRadius, w = normalOffset.
    alignas(16) float shadowParams[4]{};
    // x = punctual minBias, y = punctual slopeBias.
    alignas(16) float pointSpotShadowParams[4]{};
    // x = kSkyboxIntensity, y = kEnvironmentShadowStrength,
    // z = debug view (0=off, 1=normals, 2=NdotL, 3=shadow visibility,
    // 4=directional raw depth: red=receiver, green=stored, blue=cascade).
    // w = disable all shadow-map visibility lookups when > 0.5.
    alignas(16) float environmentParams[4]{};
    // Active light count and the packed light array. Convention: lights[0] is
    // the primary directional (CSM source) when one exists. The shader loops
    // 0..lightCount-1 and only applies CSM shadow at i==0 with type==0.
    alignas(16) int lightCount{0};
    int _pad0{0};
    int _pad1{0};
    int _pad2{0};
    LightData lights[kMaxLights]{};
};

struct EnvironmentPrefilterPushConstants
{
    alignas(4) int faceIndex{0};
    alignas(4) int faceExtent{0};
    float roughness{0.0f};
    // Extent of the *source* environment cubemap face (typically mip 0's size).
    // Used by the prefilter shader to compute the per-sample mip level for
    // Filament-style importance-sampled cubemap lookups.
    alignas(4) int sourceFaceExtent{0};
    // Max mip level available on the source environment cubemap.
    float sourceMaxMip{0.0f};
    float _pad0{0.0f};
    float _pad1{0.0f};
    float _pad2{0.0f};
};

// Shadow vertex shader projects each vertex into light-space using one of
// these matrices, picked via ShadowPushConstants::matrixIndex.
//   [0..3]   directional cascades 0..3
//   [4..]    spot lights, layout 4 + spotIndex
//   [4+S..]  point lights, layout (4 + S) + 6 * cubeIndex + face
// where S = kMaxSpotShadowCasters.
inline constexpr int kShadowCascadeMatrixBase = 0;
inline constexpr int kShadowSpotMatrixBase = 4;
inline constexpr int kShadowPointMatrixBase = kShadowSpotMatrixBase + kMaxSpotShadowCasters;
inline constexpr int kShadowTotalMatrixCount =
    kShadowPointMatrixBase + 6 * kMaxPointShadowCasters;

struct ShadowUBO
{
    alignas(16) Mat4 model;
    alignas(16) Mat4 lightViewProj[kShadowTotalMatrixCount];
    alignas(4) int hasSkin{0};
};

struct ShadowPushConstants
{
    // Selects which lightViewProj[] matrix the vertex shader uses.
    alignas(4) int matrixIndex{0};
    // Per-skinned-object self-shadow layer for the dual-depth self pass.
    alignas(4) int selfShadowSlot{-1};
    // Normalized-depth gap required before a fragment counts as the second
    // surface behind the first light-facing surface.
    float selfShadowDepthEpsilon{kSkinnedSelfShadowDepthEpsilon};
    float _pad0{0.0f};
    // Point shadow (matrixIndex >= kShadowPointMatrixBase): xyz = light
    // world position, w = effective range. shadow.frag writes linear distance
    // / range so the cube-array compare sampler agrees with the main shader.
    // Zero for cascade/spot shadow passes.
    alignas(16) float lightPosRange[4]{};
    // Used when matrixIndex < 0 for tightly-fit per-object self-shadow passes.
    alignas(16) Mat4 lightViewProj{Mat4::identity()};
};

struct ForwardPushConstants
{
    alignas(4) int selfShadowSlot{-1};
    int _pad0{0};
    int _pad1{0};
    int _pad2{0};
};

struct BloomPushConstants
{
    // Inverse of the input mip's pixel resolution — used by the down/up
    // filters to step in source-texel units across the kernel.
    alignas(8) float invInputResolution[2]{0.0f, 0.0f};
    // 1 = first downsample pass (reads HDR target). Triggers Karis-average
    // weighting in the downsample shader to suppress firefly halos.
    alignas(4) int isFirstPass{0};
    int _pad0{0};
};

struct PostProcessPushConstants
{
    // 0 = bloom off (output identical to pre-bloom). Typical 0.02–0.08.
    alignas(4) float kBloomStrength{0.0f};
    float _pad0{0.0f};
    float _pad1{0.0f};
    float _pad2{0.0f};
};

} // namespace fire_engine
