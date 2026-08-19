#pragma once

#include <cstddef>
#include <cstdint>

// GPU data-layout limits shared by the Vulkan-free graphics layer and the
// render layer. These define the sizes of UBO arrays and per-frame resource
// pools, so both the graphics structs (FrameInfo, Object bindings) and the
// render-side UBOs must agree on them. They live here, below render/, so
// graphics headers can size their arrays without including render/ headers.
//
// Pure scalar tunables (biases, strengths, FOV, IBL/shadow extents) stay in
// render/constants.hpp, which includes this header so existing render-side
// users keep seeing every constant through a single include.
//
// The values the SHADERS also need are not written here: they live in
// `shaders/gpu_limits.glsl`, a file in the common subset of GLSL and C++ that
// is included below and re-exported under the engine's `k`-names. Add a limit
// there — not here — whenever a shader must know it too, so the two sides
// cannot drift. The rest (frames in flight, joints, bindless capacities…) is
// C++-only and stays in this file.

namespace fire_engine
{

// The shared declarations, parked in their own namespace: they are GLSL-style
// SHOUTING_CASE and this header exists to give them engine names and engine
// types. Nothing outside this file should name them.
namespace shader_limits
{
#include "gpu_limits.glsl" // NOLINT(bugprone-suspicious-include): the shared GLSL/C++ limits
} // namespace shader_limits

// Frames-in-flight: how many copies of per-frame GPU resources exist.
inline constexpr int kMaxFramesInFlight = 2;

// Skinning joint matrices per SkinUBO.
inline constexpr std::size_t kMaxJoints = shader_limits::MAX_JOINTS;

// Morph target weights per MorphUBO. The GPU block packs them as vec4s, so the shader's array
// length is kMorphWeightVec4Count — derived, not written twice. The packing REQUIRES a multiple of
// four: a ninth weight would need a third vec4 the shader would not have declared, and the write
// would land outside the block. GLSL cannot say that, so it is asserted here, on the shared value
// both sides read.
inline constexpr int kMaxMorphTargets = shader_limits::MAX_MORPH_TARGETS;
inline constexpr int kMorphWeightVec4Count = shader_limits::MORPH_WEIGHT_VEC4_COUNT;
static_assert(kMaxMorphTargets % 4 == 0,
              "MorphUBO packs weights as vec4s; a non-multiple of four would not fit the block");
static_assert(kMorphWeightVec4Count * 4 == kMaxMorphTargets);

// Cap on lights consumed by the forward shader's main lighting loop. Sized so
// the LightUBO array fits comfortably under any sane Vulkan UBO limit. Bump
// when scenes routinely exceed this; or swap to an SSBO at that point.
inline constexpr int kMaxLights = shader_limits::MAX_LIGHTS;

// Per-skinned-object self-shadow slots (LightUBO::selfShadowViewProj).
inline constexpr int kMaxSkinnedSelfShadowCasters = shader_limits::MAX_SKINNED_SELF_SHADOW_CASTERS;

// Directional cascade layers in the 2D-array shadow map.
inline constexpr uint32_t kShadowCascadeCount = shader_limits::SHADOW_CASCADE_COUNT;

// Shadow casters for punctual lights. Caps are independent of kMaxLights;
// excess punctual lights remain unshadowed. First-N policy in gather order.
inline constexpr int kMaxSpotShadowCasters = shader_limits::MAX_SPOT_SHADOW_CASTERS;
inline constexpr int kMaxPointShadowCasters = shader_limits::MAX_POINT_SHADOW_CASTERS;

// Faces of a cube map — ONE authority, because this value participates in three separate things:
// logical-view key validation, image layer indexing, and the flat point-view slot arithmetic. Two
// definitions drifting apart would corrupt all of them at once, and quietly: every index would
// still be in range, just pointing at the wrong face.
inline constexpr std::uint32_t kCubeFaceCount = shader_limits::CUBE_FACE_COUNT;

// (The shadow MATRIX-TABLE layout — cascade/spot/point bases and a total count — is gone. It sized
// `ShadowUBO::lightViewProj[]`, a copy of every shadow matrix in the frame carried by every shadow
// draw so a push constant could select one row. Each path now rasterises with `pc.lightViewProj`
// from the view being recorded, so no slot arithmetic exists to keep in step.)

// Which shadow-map families a frame recorded, packed into `LightUBO::shadowMapValidMask`. The
// producer is `ShadowMapValidity::packedMask()` (graphics/shadow_map_validity.hpp); the consumer is
// every sampling path in shader.frag. Bit values shared with the shader for the same reason the
// sizes are: a bit that means one family on one side and another on the other is a silent misread.
inline constexpr std::int32_t kShadowMapValidCascades = shader_limits::SHADOW_MAP_VALID_CASCADES;
inline constexpr std::int32_t kShadowMapValidWorldOnly = shader_limits::SHADOW_MAP_VALID_WORLD_ONLY;
inline constexpr std::int32_t kShadowMapValidSelf = shader_limits::SHADOW_MAP_VALID_SELF;
inline constexpr std::int32_t kShadowMapValidSpot = shader_limits::SHADOW_MAP_VALID_SPOT;
inline constexpr std::int32_t kShadowMapValidPoint = shader_limits::SHADOW_MAP_VALID_POINT;

// Bindless material textures: capacity of the global combined-image-sampler
// array (forward set 2). Indexed directly by TextureHandle value, so it caps the
// total number of textures Resources can allocate. Partially-bound, so unused /
// non-2D slots (cubemaps, shadow/render targets) cost nothing.
//
// Sized under the device's maxPerStageDescriptorUpdateAfterBindSamplers (1024 on
// this MoltenVK), which counts *all* combined-image-samplers across every set in
// the pipeline layout — the array plus the set-1 IBL/sceneColor samplers must stay
// under it. 512 leaves ample headroom; bump toward ~1000 if a scene needs it.
inline constexpr uint32_t kMaxBindlessTextures = 512;

// Bindless materials: capacity of the global materials[] SSBO (forward set 2,
// binding 1), indexed by the per-draw ForwardPushConstants::materialIndex. Each
// distinct material registered with Resources takes one slot. Bump if a scene
// exceeds it.
inline constexpr uint32_t kMaxMaterials = 256;

// Particle system pool sizing. The GPU particle pool holds
// kMaxParticleEmitters * kMaxParticlesPerEmitter particles; each active emitter
// owns a contiguous slice (emitterIndex = particleIndex / kMaxParticlesPerEmitter).
inline constexpr int kMaxParticleEmitters = shader_limits::MAX_PARTICLE_EMITTERS;
inline constexpr int kMaxParticlesPerEmitter = 4096;

// SSAO hemisphere kernel size — the SsaoUBO kernel[] length and ssao.frag's loop bound, which read
// the same shared declaration rather than being "kept in lockstep". TAA denoises the per-pixel
// rotation noise, so a modest count suffices.
inline constexpr uint32_t kSsaoKernelSize = shader_limits::SSAO_KERNEL_SIZE;

} // namespace fire_engine
