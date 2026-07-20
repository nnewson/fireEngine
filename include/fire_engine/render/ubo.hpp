#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/render/constants.hpp>

namespace fire_engine
{

// Per-object data for the forward pass (set 0, binding 0 — pushed per draw). Only what actually
// varies per object: the world transform, its previous-frame value (TAA motion vectors), and the
// skin flag. Re-uploaded only when it changes (Object caches + skips a byte-identical rewrite), so
// a static object costs no per-frame UBO write. The camera/view data that USED to live here moved
// to CameraUBO (per-frame, bound once) — see the note there. Read by the vertex stage (skinning +
// motion vectors) and the fragment stage (transmission thickness uses `model`; `hasSkin`).
struct ObjectUBO
{
    Mat4 model;
    alignas(4) int hasSkin{0};
    int _pad1{0};
    int _pad2{0};
    int _pad3{0};
    // previousModel = the node's composed world from last frame, for the jitter-free motion vector.
    Mat4 previousModel{Mat4::identity()};
};

static_assert(offsetof(ObjectUBO, previousModel) == 80, "ObjectUBO std140 layout");
static_assert(sizeof(ObjectUBO) == 144, "ObjectUBO std140 size");

// Per-frame camera data for the forward pass (set 1, the global per-frame set — bound once per
// frame, not per object). Shared by every forward draw, so it is written exactly once per frame by
// the Renderer instead of being duplicated into every object's UBO. `proj` is jittered (TAA) for
// rasterisation; the two view-projections are jitter-free so motion vectors are independent of the
// sub-pixel jitter. Read by the vertex stage (gl_Position + motion vectors) and the fragment stage
// (view vector from cameraPos; transmission F3 reprojection uses proj·view).
struct CameraUBO
{
    Mat4 view;
    Mat4 proj;
    alignas(16) float cameraPos[4];
    Mat4 currentViewProj{Mat4::identity()};
    Mat4 previousViewProj{Mat4::identity()};
};

static_assert(offsetof(CameraUBO, currentViewProj) == 144, "CameraUBO std140 layout");
static_assert(sizeof(CameraUBO) == 272, "CameraUBO std140 size");

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
    // Bindless texture index per material texture slot (= the texture's
    // TextureHandle value, i.e. its slot in the global set-2 textures[] array).
    // Packed as 3 ivec4s (10 slots used) to match the shader's ivec4[3]; read only
    // where the slot's present-flag is set, 0 otherwise.
    alignas(16) int32_t textureIndex[12]{};
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
    // VIPM geomorph (Continuous LOD): the vertex shader slides each drawn vertex whose removal
    // level equals vipmTargetLevel toward its target by morphFactor. Both 0 in Discrete mode / for
    // non-VIPM meshes (a no-op mix). Trailing pad keeps the struct a 16-byte multiple to match
    // std140.
    alignas(4) float morphFactor{0.0f};
    alignas(4) int vipmTargetLevel{0};
    float _pad1{0.0f};
    float _pad2{0.0f};
};
static_assert(sizeof(MorphUBO) == 64, "MorphUBO must match its std140 block size");
static_assert(offsetof(MorphUBO, morphFactor) == 48);
static_assert(offsetof(MorphUBO, vipmTargetLevel) == 52);

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

// Shadow matrix layout (kShadowCascadeMatrixBase / kShadowSpotMatrixBase /
// kShadowPointMatrixBase / kShadowTotalMatrixCount) lives in
// graphics/gpu_limits.hpp — the graphics-side FrameInfo sizes an array to match
// ShadowUBO::lightViewProj, so the count must be visible without including
// render/.
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
    // Index into the global materials[] SSBO (bindless) for this draw.
    uint32_t materialIndex{0};
    // Selected discrete LOD level (0 = full mesh); read by the shader only for the LOD debug tint.
    uint32_t lodLevel{0};
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

// Fragment push constant for the TAA resolve. Mirrors the push block in
// shaders/taa.frag.
struct TaaResolvePushConstants
{
    // 1 / render-target resolution — steps the 3x3 neighbourhood clamp in
    // texel units.
    alignas(8) float texelSize[2]{0.0f, 0.0f};
    // History weight in the resolve blend (kTaaHistoryBlend).
    alignas(4) float historyBlend{0.0f};
    // Post-resolve unsharp amount (0 = off). Claws back the slight softness TAA
    // trades for stability.
    alignas(4) float sharpen{0.0f};
    // 0 on the first frame after a (re)create — the history slot holds no valid
    // data yet, so the resolve falls back to the current frame.
    alignas(4) int historyValid{0};
};

// Per-emitter parameters consumed by the particle compute kernel. std140 layout:
// four vec4s = 64 bytes, 16-aligned. Mirrors EmitterGpu in
// shaders/particle_simulate.comp.
struct ParticleEmitterGpu
{
    alignas(16) float posCone[4]{};      // xyz world position, w cone half-angle (rad)
    alignas(16) float velLifetime[4]{};  // xyz base velocity, w lifetime (s)
    alignas(16) float colourSize[4]{};   // rgb colour * intensity, w billboard half-size
    alignas(16) float gravitySpawn[4]{}; // x gravity, y spawn budget (this frame), zw pad
};

// Per-frame particle UBO: camera matrices for billboard rendering, sim
// parameters, and the active emitter array. Bound by both the compute (sim) and
// graphics (render) particle passes. std140 layout — header is 16-aligned before
// the emitter array.
struct ParticleFrameUBO
{
    Mat4 view;
    Mat4 proj;
    alignas(16) float dt{0.0f};
    uint32_t frameCounter{0};
    uint32_t emitterCount{0};
    uint32_t particlesPerEmitter{0};
    ParticleEmitterGpu emitters[kMaxParticleEmitters]{};
};

// Fragment push constant for the particle soft-fade depth comparison.
struct ParticleSoftPushConstants
{
    alignas(4) float nearPlane{0.0f};
    float farPlane{0.0f};
    float softRange{0.0f};
    float _pad0{0.0f};
};

static_assert(sizeof(ParticleEmitterGpu) == 64, "ParticleEmitterGpu must be std140 4x vec4");
static_assert(sizeof(ParticleFrameUBO) % 16 == 0, "ParticleFrameUBO must be std140 16-aligned");
static_assert(offsetof(ParticleFrameUBO, emitters) == 144,
              "emitter array must follow the 16-aligned header (2x mat4 + 16-byte scalars)");

// Per-frame SSAO UBO (binding 1 of the SSAO pass). The fragment shader
// reconstructs view-space position + normal from depth + `proj` (no matrix
// inverse: the projection terms unproject analytically), then samples the
// hemisphere `kernel` scaled by `radius`. Mirrors SsaoUBO in shaders/ssao.frag.
struct SsaoUBO
{
    // The jittered projection the depth prepass rendered with — its elements
    // unproject depth → view space and reproject kernel samples → screen.
    Mat4 proj;
    // Hemisphere samples (xyz, w unused), tangent-space (+Z = surface normal).
    alignas(16) float kernel[kSsaoKernelSize][4]{};
    // x = radius, y = bias, z = intensity (0 disables AO), w = power.
    alignas(16) float params[4]{};
    // x = contact-shadow length (view units), y = contact step count,
    // z = sun-enabled (>0.5), w unused.
    alignas(16) float contact[4]{};
    // Sun direction in view space (xyz), for the contact-shadow ray-march.
    alignas(16) float sunViewDir[4]{};
    // x = width, y = height, z = 1/width, w = 1/height (full-res AO target).
    alignas(16) float screen[4]{};
};

static_assert(sizeof(SsaoUBO) % 16 == 0, "SsaoUBO must be std140 16-aligned");

// Push constant for the physics debug-line pipeline: the jitter-free
// view-projection that transforms world-space line endpoints to clip space.
struct DebugLinePushConstants
{
    Mat4 viewProj;
};

// Push constant for the bilateral AO blur. texelSize steps the taps; projC/projD
// (= proj[2][2] / proj[3][2]) linearise depth so the edge-stop weight uses
// view-space Z. Mirrors the Push block in shaders/ssao_blur.frag.
struct SsaoBlurPushConstants
{
    alignas(8) float texelSize[2]{0.0f, 0.0f};
    alignas(4) float projC{0.0f};
    alignas(4) float projD{0.0f};
};

// ---- VDPM GPU-front scoring ABI (rendering-spine #3, GPU-driven front Stage B1) --------------
// std430 SSBO images of the Vulkan-free scoring authority (graphics/vdpm.hpp). The shader
// `vdpm_score.comp` reads these as GL_EXT_buffer_reference blocks; the CPU pack helpers
// (render/vdpm_gpu.hpp) fill them from VdpmViewParams / VertexSplit. Every field's offset + each
// struct's size is asserted so a host↔GPU layout drift fails to compile — the only defence against
// a silent, expensive ABI mistake. RULES followed: flags are uint32 (never bool/uint8); a GLSL
// `mat3` under std430 occupies 48 bytes (three 16-byte columns), so worldLinear is three padded
// vec4 columns; positions + the back-face output write all channels (see the shader contract).

// One canonical vertex's object-space position (parent/child IDs in VdpmSplitGpu index this array).
// Padded to a std430 vec4 (16B) — shared with the later repair/emit stages, which need the same
// positions. GLSL: `struct Position { vec4 p; };` (xyz used, w unused).
struct VdpmPositionGpu
{
    alignas(16) float position[4]{0.0f, 0.0f, 0.0f, 0.0f};
};
static_assert(offsetof(VdpmPositionGpu, position) == 0);
static_assert(sizeof(VdpmPositionGpu) == 16, "VdpmPositionGpu std430 vec4");
static_assert(alignof(VdpmPositionGpu) == 16, "VdpmPositionGpu vec4 alignment");
static_assert(std::is_standard_layout_v<VdpmPositionGpu>);
static_assert(std::is_trivially_copyable_v<VdpmPositionGpu>);

// Per-split static metric input. `coneAxisCos` packs the normal cone (xyz = axis, w = cos). GLSL:
//   struct Split { vec4 coneAxisCos; float supportRadius, error, uvError, normalError,
//   tangentError;
//                  uint parentId, childId, _pad; };
struct VdpmSplitGpu
{
    alignas(16) float coneAxisCos[4]{0.0f, 0.0f, 0.0f,
                                     1.0f}; // 0: normalConeAxis.xyz, normalConeCos
    float supportRadius{0.0f};              // 16
    float error{0.0f};                      // 20
    float uvError{0.0f};                    // 24
    float normalError{0.0f};                // 28
    float tangentError{0.0f};               // 32
    std::uint32_t parentId{0};              // 36
    std::uint32_t childId{0};               // 40
    std::uint32_t _pad{0};                  // 44 -> 48
};
static_assert(offsetof(VdpmSplitGpu, coneAxisCos) == 0);
static_assert(offsetof(VdpmSplitGpu, supportRadius) == 16);
static_assert(offsetof(VdpmSplitGpu, error) == 20);
static_assert(offsetof(VdpmSplitGpu, uvError) == 24);
static_assert(offsetof(VdpmSplitGpu, normalError) == 28);
static_assert(offsetof(VdpmSplitGpu, tangentError) == 32);
static_assert(offsetof(VdpmSplitGpu, parentId) == 36);
static_assert(offsetof(VdpmSplitGpu, childId) == 40);
static_assert(offsetof(VdpmSplitGpu, _pad) == 44);
static_assert(sizeof(VdpmSplitGpu) == 48, "VdpmSplitGpu std430 size/stride");
static_assert(alignof(VdpmSplitGpu) == 16, "VdpmSplitGpu vec4 alignment");
static_assert(std::is_standard_layout_v<VdpmSplitGpu>);
static_assert(std::is_trivially_copyable_v<VdpmSplitGpu>);

// Per-split scoring output — ALL channels (not just the max), so a broken non-winning channel is
// visible to the harness. All scalars ⇒ std430 array stride 24. GLSL:
//   struct ScoreOut { float geometry, uv, normal, tangent, straddle; uint backface; };
struct VdpmScoreOut
{
    float geometry{0.0f}; // 0
    float uv{0.0f};       // 4
    float normal{0.0f};   // 8
    float tangent{0.0f};  // 12
    float straddle{0.0f}; // 16
    std::uint32_t backface{0};
};
static_assert(offsetof(VdpmScoreOut, geometry) == 0);
static_assert(offsetof(VdpmScoreOut, uv) == 4);
static_assert(offsetof(VdpmScoreOut, normal) == 8);
static_assert(offsetof(VdpmScoreOut, tangent) == 12);
static_assert(offsetof(VdpmScoreOut, straddle) == 16);
static_assert(offsetof(VdpmScoreOut, backface) == 20);
static_assert(sizeof(VdpmScoreOut) == 24, "VdpmScoreOut std430 size/stride");
static_assert(alignof(VdpmScoreOut) == 4, "VdpmScoreOut scalar alignment");
static_assert(std::is_standard_layout_v<VdpmScoreOut>);
static_assert(std::is_trivially_copyable_v<VdpmScoreOut>);

// Per-instance scoring params (the std430 image of VdpmViewParams) + the three buffer_reference
// device addresses + splitCount. Only THIS struct's address is pushed (8B, well under the 128B push
// guarantee); the shader dereferences the addresses. `worldLinear` is a GLSL mat3 (three padded
// vec4 columns, 48B). Flags are uint32. GLSL (buffer_reference, std430):
//   Params { mat3 worldLinear; vec4 worldTranslationMinusCamera; vec4 cameraObj;
//            float worldLengthScale, facingSign, projScaleY, halfViewport, silhouetteBoost,
//                  uvScale, normalScale, tangentScale;
//            uint coneUsable, coneCullEnabled, splitCount, _pad0;
//            Splits splits; Positions positions; ScoreOuts outputs; };  // last three are addresses
struct alignas(16) VdpmScoreParams
{
    float worldLinearCol0[4]{0.0f, 0.0f, 0.0f, 0.0f};             // 0  mat3 column 0 (.xyz)
    float worldLinearCol1[4]{0.0f, 0.0f, 0.0f, 0.0f};             // 16 column 1
    float worldLinearCol2[4]{0.0f, 0.0f, 0.0f, 0.0f};             // 32 column 2
    float worldTranslationMinusCamera[4]{0.0f, 0.0f, 0.0f, 0.0f}; // 48 vec4 (.xyz)
    float cameraObj[4]{0.0f, 0.0f, 0.0f, 0.0f};                   // 64 vec4 (.xyz)
    float worldLengthScale{1.0f};                                 // 80
    float facingSign{1.0f};                                       // 84
    float projScaleY{1.0f};                                       // 88
    float halfViewport{0.0f};                                     // 92
    float silhouetteBoost{0.0f};                                  // 96
    float uvScale{1.0f};                                          // 100
    float normalScale{1.0f};                                      // 104
    float tangentScale{1.0f};                                     // 108
    std::uint32_t coneUsable{1};                                  // 112
    std::uint32_t coneCullEnabled{1};                             // 116
    std::uint32_t splitCount{0};                                  // 120
    std::uint32_t _pad0{0};                                       // 124 (align the uint64s to 128)
    std::uint64_t splitsAddress{0};                               // 128
    std::uint64_t positionsAddress{0};                            // 136
    std::uint64_t outputsAddress{0};                              // 144
    std::uint64_t _pad1{0};                                       // 152 -> 160
};
static_assert(offsetof(VdpmScoreParams, worldLinearCol0) == 0);
static_assert(offsetof(VdpmScoreParams, worldLinearCol1) == 16);
static_assert(offsetof(VdpmScoreParams, worldLinearCol2) == 32);
static_assert(offsetof(VdpmScoreParams, worldTranslationMinusCamera) == 48);
static_assert(offsetof(VdpmScoreParams, cameraObj) == 64);
static_assert(offsetof(VdpmScoreParams, worldLengthScale) == 80);
static_assert(offsetof(VdpmScoreParams, facingSign) == 84);
static_assert(offsetof(VdpmScoreParams, projScaleY) == 88);
static_assert(offsetof(VdpmScoreParams, halfViewport) == 92);
static_assert(offsetof(VdpmScoreParams, silhouetteBoost) == 96);
static_assert(offsetof(VdpmScoreParams, uvScale) == 100);
static_assert(offsetof(VdpmScoreParams, normalScale) == 104);
static_assert(offsetof(VdpmScoreParams, tangentScale) == 108);
static_assert(offsetof(VdpmScoreParams, coneUsable) == 112);
static_assert(offsetof(VdpmScoreParams, coneCullEnabled) == 116);
static_assert(offsetof(VdpmScoreParams, splitCount) == 120);
static_assert(offsetof(VdpmScoreParams, _pad0) == 124);
static_assert(offsetof(VdpmScoreParams, splitsAddress) == 128);
static_assert(offsetof(VdpmScoreParams, positionsAddress) == 136);
static_assert(offsetof(VdpmScoreParams, outputsAddress) == 144);
static_assert(offsetof(VdpmScoreParams, _pad1) == 152);
static_assert(sizeof(VdpmScoreParams) == 160, "VdpmScoreParams std430 size");
static_assert(alignof(VdpmScoreParams) == 16, "VdpmScoreParams alignment");
static_assert(std::is_standard_layout_v<VdpmScoreParams>);
static_assert(std::is_trivially_copyable_v<VdpmScoreParams>);

// Push constant: only the params block's device address (8B, mirroring a GLSL buffer_reference
// pointer). The shader dereferences it, then the three typed references inside it.
struct VdpmScorePushConstants
{
    std::uint64_t paramsAddress{0};
};
static_assert(sizeof(VdpmScorePushConstants) == 8);
static_assert(std::is_standard_layout_v<VdpmScorePushConstants>);
static_assert(std::is_trivially_copyable_v<VdpmScorePushConstants>);

// ---- VDPM GPU exclusive prefix-sum (Stage B2 emit compaction) --------------------------------
// The recursive hierarchical scan's block size — the number of elements ONE workgroup scans, and
// MUST equal the `local_size_x` in vdpm_scan_block.comp / the block stride in vdpm_scan_add.comp.
// Named by MEANING (elements per block) so the B±1 boundary tests stay correct if a shader later
// processes more than one element per invocation.
inline constexpr std::uint32_t kScanElementsPerBlock = 256;
// The Blelloch up-/down-sweep requires a power-of-two block size.
static_assert(std::has_single_bit(kScanElementsPerBlock));

// Push for vdpm_scan_block.comp: exclusive-scan `input[count]` into `output` (may alias `input`)
// and write each block's total to `blockSums[blockIndex]`. buffer_reference addresses (uint
// arrays).
struct alignas(8) VdpmScanBlockPush
{
    std::uint64_t inputAddress{0};
    std::uint64_t outputAddress{0};
    std::uint64_t blockSumsAddress{0};
    std::uint32_t count{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmScanBlockPush, inputAddress) == 0);
static_assert(offsetof(VdpmScanBlockPush, outputAddress) == 8);
static_assert(offsetof(VdpmScanBlockPush, blockSumsAddress) == 16);
static_assert(offsetof(VdpmScanBlockPush, count) == 24);
static_assert(sizeof(VdpmScanBlockPush) == 32);
static_assert(alignof(VdpmScanBlockPush) == 8);
static_assert(std::is_standard_layout_v<VdpmScanBlockPush>);
static_assert(std::is_trivially_copyable_v<VdpmScanBlockPush>);

// Push for vdpm_scan_add.comp: output[i] += offsets[i / kScanElementsPerBlock] for i < count.
struct alignas(8) VdpmScanAddPush
{
    std::uint64_t outputAddress{0};
    std::uint64_t offsetsAddress{0};
    std::uint32_t count{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmScanAddPush, outputAddress) == 0);
static_assert(offsetof(VdpmScanAddPush, offsetsAddress) == 8);
static_assert(offsetof(VdpmScanAddPush, count) == 16);
static_assert(sizeof(VdpmScanAddPush) == 24);
static_assert(alignof(VdpmScanAddPush) == 8);
static_assert(std::is_standard_layout_v<VdpmScanAddPush>);
static_assert(std::is_trivially_copyable_v<VdpmScanAddPush>);

// Push for vdpm_ancestor.comp (VDPM GPU emit pass 1): resolve each canonical vertex's active
// ancestor
// + depth by walking the collapsed removalParent chain, bounded by maxDepth. counters[0] is the
// ancestor-failure atomic counter. Field order mirrors the shader's push_constant block.
struct alignas(8) VdpmAncestorPush
{
    std::uint64_t activeAddress{0};
    std::uint64_t removalParentAddress{0};
    std::uint64_t ancestorIdAddress{0};
    std::uint64_t ancestorDepthAddress{0};
    std::uint64_t countersAddress{0};
    std::uint32_t vertexCount{0};
    std::uint32_t maxDepth{0};
};
static_assert(offsetof(VdpmAncestorPush, activeAddress) == 0);
static_assert(offsetof(VdpmAncestorPush, removalParentAddress) == 8);
static_assert(offsetof(VdpmAncestorPush, ancestorIdAddress) == 16);
static_assert(offsetof(VdpmAncestorPush, ancestorDepthAddress) == 24);
static_assert(offsetof(VdpmAncestorPush, countersAddress) == 32);
static_assert(offsetof(VdpmAncestorPush, vertexCount) == 40);
static_assert(offsetof(VdpmAncestorPush, maxDepth) == 44);
static_assert(sizeof(VdpmAncestorPush) == 48);
static_assert(alignof(VdpmAncestorPush) == 8);
static_assert(std::is_standard_layout_v<VdpmAncestorPush>);
static_assert(std::is_trivially_copyable_v<VdpmAncestorPush>);

// Push for vdpm_survival.comp (VDPM GPU emit pass 2): per finest face, write a 0/1 survival flag
// (three distinct, non-failed ancestors). Field order mirrors the shader's push_constant block.
struct alignas(8) VdpmSurvivalPush
{
    std::uint64_t indicesAddress{0};
    std::uint64_t weldAddress{0};
    std::uint64_t ancestorIdAddress{0};
    std::uint64_t surviveAddress{0};
    std::uint32_t faceCount{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmSurvivalPush, indicesAddress) == 0);
static_assert(offsetof(VdpmSurvivalPush, weldAddress) == 8);
static_assert(offsetof(VdpmSurvivalPush, ancestorIdAddress) == 16);
static_assert(offsetof(VdpmSurvivalPush, surviveAddress) == 24);
static_assert(offsetof(VdpmSurvivalPush, faceCount) == 32);
static_assert(sizeof(VdpmSurvivalPush) == 40);
static_assert(alignof(VdpmSurvivalPush) == 8);
static_assert(std::is_standard_layout_v<VdpmSurvivalPush>);
static_assert(std::is_trivially_copyable_v<VdpmSurvivalPush>);

// Push for vdpm_scatter.comp (VDPM GPU emit pass 4): a surviving face writes its three
// restored-wedge indices at its stable scan slot. Field order mirrors the shader's push_constant
// block.
struct alignas(8) VdpmScatterPush
{
    std::uint64_t indicesAddress{0};
    std::uint64_t weldAddress{0};
    std::uint64_t ancestorDepthAddress{0};
    std::uint64_t surviveAddress{0};
    std::uint64_t outSlotAddress{0};
    std::uint64_t wedgeChoicesAddress{0};
    std::uint64_t wedgeOffsetsAddress{0};
    std::uint64_t emittedIndicesAddress{0};
    std::uint32_t faceCount{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmScatterPush, indicesAddress) == 0);
static_assert(offsetof(VdpmScatterPush, weldAddress) == 8);
static_assert(offsetof(VdpmScatterPush, ancestorDepthAddress) == 16);
static_assert(offsetof(VdpmScatterPush, surviveAddress) == 24);
static_assert(offsetof(VdpmScatterPush, outSlotAddress) == 32);
static_assert(offsetof(VdpmScatterPush, wedgeChoicesAddress) == 40);
static_assert(offsetof(VdpmScatterPush, wedgeOffsetsAddress) == 48);
static_assert(offsetof(VdpmScatterPush, emittedIndicesAddress) == 56);
static_assert(offsetof(VdpmScatterPush, faceCount) == 64);
static_assert(sizeof(VdpmScatterPush) == 72);
static_assert(alignof(VdpmScatterPush) == 8);
static_assert(std::is_standard_layout_v<VdpmScatterPush>);
static_assert(std::is_trivially_copyable_v<VdpmScatterPush>);

// The finalize pass writes the 5-word draw indirect command into a buffer laid out as the
// Vulkan-free graphics::DrawIndexedIndirectCommand (already asserted bit-for-bit against
// VkDrawIndexedIndirectCommand in render/resources.cpp) — the single authority; no VDPM-local
// mirror.

// Push for vdpm_emit_finalize.comp (VDPM GPU emit pass 5): a single invocation writes counters[2] =
// 3 * counters[1] (the emitted index count) AND the full 5-word draw indirect command.
struct alignas(8) VdpmEmitFinalizePush
{
    std::uint64_t countersAddress{0};
    std::uint64_t indirectAddress{0};
};
static_assert(offsetof(VdpmEmitFinalizePush, countersAddress) == 0);
static_assert(offsetof(VdpmEmitFinalizePush, indirectAddress) == 8);
static_assert(sizeof(VdpmEmitFinalizePush) == 16);
static_assert(alignof(VdpmEmitFinalizePush) == 8);
static_assert(std::is_standard_layout_v<VdpmEmitFinalizePush>);
static_assert(std::is_trivially_copyable_v<VdpmEmitFinalizePush>);

// ---- VDPM GPU refine/coarsen (Stage B3) ----

// Static per-split mutation topology for the GPU refine/coarsen passes (Stage B3). The VERTEX slots
// (parent/vl/vr — vr may be the boundary sentinel `kInvalidVertex`) drive the `dependents` atomic
// add/sub and the child activation; the DEPENDENCY-SPLIT slots (parentDep/vlDep/vrDep — `kNoSplit`
// for a root/boundary) are the closure edges, mirroring `DependencyDag::dependencies` 1:1 (the
// shared authority). Kept SEPARATE from the scoring ABI so neither bloats the other.
struct VdpmFrontSplitGpu
{
    std::uint32_t parent{0};
    std::uint32_t child{0};
    std::uint32_t vl{0};
    std::uint32_t vr{0};
    std::uint32_t parentDep{0};
    std::uint32_t vlDep{0};
    std::uint32_t vrDep{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmFrontSplitGpu, parent) == 0);
static_assert(offsetof(VdpmFrontSplitGpu, child) == 4);
static_assert(offsetof(VdpmFrontSplitGpu, vl) == 8);
static_assert(offsetof(VdpmFrontSplitGpu, vr) == 12);
static_assert(offsetof(VdpmFrontSplitGpu, parentDep) == 16);
static_assert(offsetof(VdpmFrontSplitGpu, vlDep) == 20);
static_assert(offsetof(VdpmFrontSplitGpu, vrDep) == 24);
static_assert(sizeof(VdpmFrontSplitGpu) == 32, "VdpmFrontSplitGpu std430 size/stride");
static_assert(alignof(VdpmFrontSplitGpu) == 4);
static_assert(std::is_standard_layout_v<VdpmFrontSplitGpu>);
static_assert(std::is_trivially_copyable_v<VdpmFrontSplitGpu>);

// One rank's contiguous range in `splitsByRank` (Stage B3 / repair-scheduler ABI). Uploaded
// device-local (VdpmGpuMeshBinding::rankRangesAddress) so the persistent repair kernel walks ranks
// in-workgroup; the CPU recorder issues one dispatch per rank over `[offset, offset + count)`.
// `render/vdpm_gpu.hpp` aliases RankRange to this — one authority for the CPU dispatch loop and the
// GPU buffer.
struct alignas(4) VdpmRankRangeGpu
{
    std::uint32_t offset{0};
    std::uint32_t count{0};
};
static_assert(offsetof(VdpmRankRangeGpu, offset) == 0);
static_assert(offsetof(VdpmRankRangeGpu, count) == 4);
static_assert(sizeof(VdpmRankRangeGpu) == 8, "VdpmRankRangeGpu std430 size/stride");
static_assert(alignof(VdpmRankRangeGpu) == 4);
static_assert(std::is_standard_layout_v<VdpmRankRangeGpu>);
static_assert(std::is_trivially_copyable_v<VdpmRankRangeGpu>);

// Push for vdpm_mark.comp: over ALL splits, full-overwrite `required[s] = (backface == 0 && s >
// pixelBudget)` where `s = max(geometry, uv, normal, tangent)` (straddle excluded, matching
// VdpmSplitScore::score). No separate clear — this writes every entry.
struct alignas(8) VdpmMarkPush
{
    std::uint64_t scoresAddress{0};
    std::uint64_t requiredAddress{0};
    std::uint32_t splitCount{0};
    float pixelBudget{0.0f};
};
static_assert(offsetof(VdpmMarkPush, scoresAddress) == 0);
static_assert(offsetof(VdpmMarkPush, requiredAddress) == 8);
static_assert(offsetof(VdpmMarkPush, splitCount) == 16);
static_assert(offsetof(VdpmMarkPush, pixelBudget) == 20);
static_assert(sizeof(VdpmMarkPush) == 24);
static_assert(alignof(VdpmMarkPush) == 8);
static_assert(std::is_standard_layout_v<VdpmMarkPush>);
static_assert(std::is_trivially_copyable_v<VdpmMarkPush>);

// Push for vdpm_close.comp: one rank's splits (`splitsByRank[rankOffset .. rankOffset +
// rankCount]`); each required split atomic-ORs its three dependency splits' `required`.
struct alignas(8) VdpmClosePush
{
    std::uint64_t splitsByRankAddress{0};
    std::uint64_t frontSplitsAddress{0};
    std::uint64_t requiredAddress{0};
    std::uint32_t rankOffset{0};
    std::uint32_t rankCount{0};
};
static_assert(offsetof(VdpmClosePush, splitsByRankAddress) == 0);
static_assert(offsetof(VdpmClosePush, frontSplitsAddress) == 8);
static_assert(offsetof(VdpmClosePush, requiredAddress) == 16);
static_assert(offsetof(VdpmClosePush, rankOffset) == 24);
static_assert(offsetof(VdpmClosePush, rankCount) == 28);
static_assert(sizeof(VdpmClosePush) == 32);
static_assert(alignof(VdpmClosePush) == 8);
static_assert(std::is_standard_layout_v<VdpmClosePush>);
static_assert(std::is_trivially_copyable_v<VdpmClosePush>);

// Push for vdpm_refine.comp: one rank's splits; a required, not-yet-refined split whose
// parent/vl/vr are active refines (activate child + atomic-add `dependents` per vertex-slot). A
// required refine with an inactive dependency sets `failFlags[0]` (refine failure) and does NOT
// mutate — the GPU analogue of the CPU's "required refine must succeed".
struct alignas(8) VdpmRefinePush
{
    std::uint64_t splitsByRankAddress{0};
    std::uint64_t frontSplitsAddress{0};
    std::uint64_t requiredAddress{0};
    std::uint64_t refinedAddress{0};
    std::uint64_t activeAddress{0};
    std::uint64_t dependentsAddress{0};
    std::uint64_t failFlagsAddress{0};
    std::uint32_t rankOffset{0};
    std::uint32_t rankCount{0};
};
static_assert(offsetof(VdpmRefinePush, splitsByRankAddress) == 0);
static_assert(offsetof(VdpmRefinePush, frontSplitsAddress) == 8);
static_assert(offsetof(VdpmRefinePush, requiredAddress) == 16);
static_assert(offsetof(VdpmRefinePush, refinedAddress) == 24);
static_assert(offsetof(VdpmRefinePush, activeAddress) == 32);
static_assert(offsetof(VdpmRefinePush, dependentsAddress) == 40);
static_assert(offsetof(VdpmRefinePush, failFlagsAddress) == 48);
static_assert(offsetof(VdpmRefinePush, rankOffset) == 56);
static_assert(offsetof(VdpmRefinePush, rankCount) == 60);
static_assert(sizeof(VdpmRefinePush) == 64);
static_assert(alignof(VdpmRefinePush) == 8);
static_assert(std::is_standard_layout_v<VdpmRefinePush>);
static_assert(std::is_trivially_copyable_v<VdpmRefinePush>);

// Push for vdpm_coarsen.comp: one rank's splits; a refined split with (backface || s <
// coarsenBudget) whose child is a leaf (`dependents[child] == 0`) coarsens (deactivate child +
// atomic-sub `dependents` per vertex-slot). A dependent decrement whose old value is 0 sets
// `failFlags[1]` (underflow).
struct alignas(8) VdpmCoarsenPush
{
    std::uint64_t splitsByRankAddress{0};
    std::uint64_t frontSplitsAddress{0};
    std::uint64_t scoresAddress{0};
    std::uint64_t refinedAddress{0};
    std::uint64_t activeAddress{0};
    std::uint64_t dependentsAddress{0};
    std::uint64_t failFlagsAddress{0};
    std::uint32_t rankOffset{0};
    std::uint32_t rankCount{0};
    float coarsenBudget{0.0f};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmCoarsenPush, splitsByRankAddress) == 0);
static_assert(offsetof(VdpmCoarsenPush, frontSplitsAddress) == 8);
static_assert(offsetof(VdpmCoarsenPush, scoresAddress) == 16);
static_assert(offsetof(VdpmCoarsenPush, refinedAddress) == 24);
static_assert(offsetof(VdpmCoarsenPush, activeAddress) == 32);
static_assert(offsetof(VdpmCoarsenPush, dependentsAddress) == 40);
static_assert(offsetof(VdpmCoarsenPush, failFlagsAddress) == 48);
static_assert(offsetof(VdpmCoarsenPush, rankOffset) == 56);
static_assert(offsetof(VdpmCoarsenPush, rankCount) == 60);
static_assert(offsetof(VdpmCoarsenPush, coarsenBudget) == 64);
static_assert(sizeof(VdpmCoarsenPush) == 72);
static_assert(alignof(VdpmCoarsenPush) == 8);
static_assert(std::is_standard_layout_v<VdpmCoarsenPush>);
static_assert(std::is_trivially_copyable_v<VdpmCoarsenPush>);

// ---- VDPM GPU repair fixpoint (Stage B4) ----

// Per-repair view params for the foldover/coverage detector (Stage B4): the world + jitter-free
// viewProj transforms, camera position, and viewport + cull policy the CPU `detail::` classifiers
// consume. Uploaded once per repair (host-visible). Mat4 maps directly to a GLSL mat4
// (column-major, 64 B), as in CameraUBO.
struct alignas(16) VdpmRepairParams
{
    Mat4 world;
    Mat4 viewProj;
    float cameraPos[4]{};                      // xyz + pad
    float viewport[4]{0.0f, 0.0f, 0.0f, 0.0f}; // x=width, y=height, z=rasterBackfaceCulling(0/1), w
};
static_assert(offsetof(VdpmRepairParams, world) == 0);
static_assert(offsetof(VdpmRepairParams, viewProj) == 64);
static_assert(offsetof(VdpmRepairParams, cameraPos) == 128);
static_assert(offsetof(VdpmRepairParams, viewport) == 144);
static_assert(sizeof(VdpmRepairParams) == 160);
static_assert(alignof(VdpmRepairParams) == 16);
static_assert(std::is_standard_layout_v<VdpmRepairParams>);
static_assert(std::is_trivially_copyable_v<VdpmRepairParams>);

// Push for vdpm_repair_detect.comp: per canonical finest face, classify foldover ∪ coverage against
// the settled front (ancestor cache from the shared ancestor pass) and atomic-OR each violation's
// inactive-corner removing split into `required`; atomic-OR `repairControl[0]` (anyMarked).
struct alignas(8) VdpmRepairDetectPush
{
    std::uint64_t finestFacesAddress{0};
    std::uint64_t positionsAddress{0};
    std::uint64_t activeAddress{0};
    std::uint64_t ancestorIdAddress{0};
    std::uint64_t removingSplitAddress{0};
    std::uint64_t requiredAddress{0};
    std::uint64_t repairControlAddress{0};
    std::uint64_t paramsAddress{0};
    std::uint64_t classificationAddress{0}; // per face: packed {foldover, coverageKind, worstLocal}
    std::uint32_t faceCount{0};
    std::uint32_t writeClassification{0}; // 1 = write `classification` (test only); 0 = skip
};
static_assert(offsetof(VdpmRepairDetectPush, finestFacesAddress) == 0);
static_assert(offsetof(VdpmRepairDetectPush, positionsAddress) == 8);
static_assert(offsetof(VdpmRepairDetectPush, activeAddress) == 16);
static_assert(offsetof(VdpmRepairDetectPush, ancestorIdAddress) == 24);
static_assert(offsetof(VdpmRepairDetectPush, removingSplitAddress) == 32);
static_assert(offsetof(VdpmRepairDetectPush, requiredAddress) == 40);
static_assert(offsetof(VdpmRepairDetectPush, repairControlAddress) == 48);
static_assert(offsetof(VdpmRepairDetectPush, paramsAddress) == 56);
static_assert(offsetof(VdpmRepairDetectPush, classificationAddress) == 64);
static_assert(offsetof(VdpmRepairDetectPush, faceCount) == 72);
static_assert(offsetof(VdpmRepairDetectPush, writeClassification) == 76);
static_assert(sizeof(VdpmRepairDetectPush) == 80);
static_assert(alignof(VdpmRepairDetectPush) == 8);
static_assert(std::is_standard_layout_v<VdpmRepairDetectPush>);
static_assert(std::is_trivially_copyable_v<VdpmRepairDetectPush>);

// Packing for VdpmRepairDetectPush::classification (the test-only per-face readback): bit 0 =
// foldover; bits 1-2 = coverage kind (0 None / 1 AllInactiveCorners / 2 WorstInactiveCorner); bits
// 4-5 = the worst LOCAL corner (0..2) when the kind is WorstInactiveCorner. Mirrors the CPU
// `detail::isFoldover` + `detail::CoverageRepair` so the harness can compare per face per branch.
inline constexpr std::uint32_t kVdpmDetectFoldoverBit = 0x1u;
inline constexpr std::uint32_t kVdpmDetectCoverageKindShift = 1u;
inline constexpr std::uint32_t kVdpmDetectCoverageKindMask = 0x3u;
inline constexpr std::uint32_t kVdpmDetectWorstCornerShift = 4u;
inline constexpr std::uint32_t kVdpmDetectWorstCornerMask = 0x3u;

// Push for vdpm_repair_fallback.comp: the correctness-preserving fallback. Per split,
// FULL-OVERWRITE `required[s] = (repairControl[0] != 0 && refined[s] == 0) ? 1 : 0` — so if the
// post-budget detect still found a violation (anyMarked), every unrefined split is seeded and the
// subsequent close+refine drives to FULL DETAIL (guaranteed hole-free); otherwise required stays 0
// (a no-op). A thread sets `repairControl[2]` (fallback fired) when triggered.
struct alignas(8) VdpmRepairFallbackPush
{
    std::uint64_t requiredAddress{0};
    std::uint64_t refinedAddress{0};
    std::uint64_t repairControlAddress{0};
    std::uint32_t splitCount{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmRepairFallbackPush, requiredAddress) == 0);
static_assert(offsetof(VdpmRepairFallbackPush, refinedAddress) == 8);
static_assert(offsetof(VdpmRepairFallbackPush, repairControlAddress) == 16);
static_assert(offsetof(VdpmRepairFallbackPush, splitCount) == 24);
static_assert(sizeof(VdpmRepairFallbackPush) == 32);
static_assert(alignof(VdpmRepairFallbackPush) == 8);
static_assert(std::is_standard_layout_v<VdpmRepairFallbackPush>);
static_assert(std::is_trivially_copyable_v<VdpmRepairFallbackPush>);

// One front's complete repair job (Stage 2 persistent-kernel ABI). An array of these is the batch
// vehicle: the kernel reads `jobs[gl_WorkGroupID.x]` and runs the whole GPU-resident repair
// fixpoint for that front in ONE workgroup. Stage 2 uploads a 1-element array + dispatches 1
// workgroup; Stage 4 fills N + dispatches N — the shader/ABI unchanged. All *Address fields are BDA
// device addresses; the counts drive the strided loops. `rankCount` (not maxRank) is the length of
// the rankRanges array (0 for a zero-split front); `roundHistoryCapacity` is the ALLOCATED history
// length
// (>= roundBudget), which the kernel clears IN FULL so a shorter budget leaves no stale tail.
struct alignas(8) VdpmRepairJobGpu
{
    std::uint64_t activeAddress{0};        // per canonical: 0/1
    std::uint64_t refinedAddress{0};       // per split
    std::uint64_t requiredAddress{0};      // per split
    std::uint64_t dependentsAddress{0};    // per canonical
    std::uint64_t failFlagsAddress{0};     // 2 uint (close/refine failure flags)
    std::uint64_t ancestorIdAddress{0};    // per canonical (repair ancestor cache)
    std::uint64_t ancestorDepthAddress{0}; // per canonical
    std::uint64_t repairControlAddress{
        0};                               // 4 uint [anyMarked, ancestorFailure, fallbackFired, pad]
    std::uint64_t roundHistoryAddress{0}; // roundHistoryCapacity uint
    std::uint64_t finestFacesAddress{0};  // 3 * finestFaceCount canonical corners
    std::uint64_t positionsAddress{0};    // per canonical vec4
    std::uint64_t removalParentAddress{0}; // per canonical (ancestor walk)
    std::uint64_t removingSplitAddress{0}; // per canonical (mark target)
    std::uint64_t frontSplitsAddress{0};   // VdpmFrontSplitGpu per split
    std::uint64_t splitsByRankAddress{0};  // uint per split, packed by ascending rank
    std::uint64_t rankRangesAddress{0};    // VdpmRankRangeGpu[rankCount]
    std::uint64_t paramsAddress{0};        // VdpmRepairParams
    std::uint32_t vertexCount{0};
    std::uint32_t finestFaceCount{0};
    std::uint32_t splitCount{0};
    std::uint32_t maxDepth{0};
    std::uint32_t rankCount{0};
    std::uint32_t roundBudget{0};
    std::uint32_t roundHistoryCapacity{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmRepairJobGpu, activeAddress) == 0);
static_assert(offsetof(VdpmRepairJobGpu, paramsAddress) == 128); // 17th uint64 (index 16)
static_assert(offsetof(VdpmRepairJobGpu, vertexCount) == 136);
static_assert(offsetof(VdpmRepairJobGpu, roundHistoryCapacity) == 160);
static_assert(sizeof(VdpmRepairJobGpu) == 168, "VdpmRepairJobGpu std430 size/stride");
static_assert(alignof(VdpmRepairJobGpu) == 8);
static_assert(std::is_standard_layout_v<VdpmRepairJobGpu>);
static_assert(std::is_trivially_copyable_v<VdpmRepairJobGpu>);

// Push constant for the persistent repair kernel: only the job-array address + count (pin 6 — never
// push the per-front addresses). The kernel dispatches `jobCount` workgroups, one per front.
struct alignas(8) VdpmRepairKernelPush
{
    std::uint64_t jobsAddress{0};
    std::uint32_t jobCount{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmRepairKernelPush, jobsAddress) == 0);
static_assert(offsetof(VdpmRepairKernelPush, jobCount) == 8);
static_assert(sizeof(VdpmRepairKernelPush) == 16);
static_assert(alignof(VdpmRepairKernelPush) == 8);
static_assert(std::is_standard_layout_v<VdpmRepairKernelPush>);
static_assert(std::is_trivially_copyable_v<VdpmRepairKernelPush>);

// One front's complete APPLY job (apply-kernel arc, persistent-kernel ABI). Same batch vehicle as
// VdpmRepairJobGpu: the apply kernel reads `jobs[gl_WorkGroupID.x]` and runs the whole
// refine/coarsen apply (mark → close → refine → coarsen) for that front in ONE workgroup. A strict
// SUBSET of the repair job's addresses — apply reads the front's own SCORES (repair doesn't) and
// needs no ancestor/positions/params/history state. Budgets live IN the job (no separate params
// block). All *Address fields are BDA device addresses.
struct alignas(8) VdpmApplyJobGpu
{
    std::uint64_t scoresAddress{0};       // VdpmScoreOut per split (the front's own score output)
    std::uint64_t activeAddress{0};       // per canonical: 0/1
    std::uint64_t refinedAddress{0};      // per split
    std::uint64_t requiredAddress{0};     // per split (mark seed → closure)
    std::uint64_t dependentsAddress{0};   // per canonical
    std::uint64_t failFlagsAddress{0};    // 2 uint [0]=refine failure, [1]=dependents underflow
    std::uint64_t frontSplitsAddress{0};  // VdpmFrontSplitGpu per split
    std::uint64_t splitsByRankAddress{0}; // uint per split, packed by ascending rank
    std::uint64_t rankRangesAddress{0};   // VdpmRankRangeGpu[rankCount]
    std::uint32_t vertexCount{0};
    std::uint32_t splitCount{0};
    std::uint32_t rankCount{0}; // = maxRank + 1 (0 for a zero-split front)
    std::uint32_t pad{0};
    float pixelBudget{0.0f};
    float coarsenBudget{0.0f};
};
static_assert(offsetof(VdpmApplyJobGpu, scoresAddress) == 0);
static_assert(offsetof(VdpmApplyJobGpu, rankRangesAddress) == 64); // 9th uint64 (index 8)
static_assert(offsetof(VdpmApplyJobGpu, vertexCount) == 72);
static_assert(offsetof(VdpmApplyJobGpu, pixelBudget) == 88);
static_assert(offsetof(VdpmApplyJobGpu, coarsenBudget) == 92);
static_assert(sizeof(VdpmApplyJobGpu) == 96, "VdpmApplyJobGpu std430 size/stride");
static_assert(alignof(VdpmApplyJobGpu) == 8);
static_assert(std::is_standard_layout_v<VdpmApplyJobGpu>);
static_assert(std::is_trivially_copyable_v<VdpmApplyJobGpu>);

// Push constant for the persistent apply kernel: only the job-array address + count (never push the
// per-front addresses). The kernel dispatches `jobCount` workgroups, one per front.
struct alignas(8) VdpmApplyKernelPush
{
    std::uint64_t jobsAddress{0};
    std::uint32_t jobCount{0};
    std::uint32_t pad{0};
};
static_assert(offsetof(VdpmApplyKernelPush, jobsAddress) == 0);
static_assert(offsetof(VdpmApplyKernelPush, jobCount) == 8);
static_assert(sizeof(VdpmApplyKernelPush) == 16);
static_assert(alignof(VdpmApplyKernelPush) == 8);
static_assert(std::is_standard_layout_v<VdpmApplyKernelPush>);
static_assert(std::is_trivially_copyable_v<VdpmApplyKernelPush>);

} // namespace fire_engine
