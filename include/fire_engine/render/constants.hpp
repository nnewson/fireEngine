#pragma once

#include <cstdint>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/math/constants.hpp>

// Single source of truth for engine-wide rendering tunables. Every scalar value
// that might want to be tweaked (light intensity, shadow biases, cascade split
// λ, bloom strength, IBL extents, camera FOV, etc.) lives here so a one-knob
// adjustment never has to chase usages across the codebase.
//
// GPU data-layout limits (frames-in-flight, joint/morph/light caps, shadow
// caster counts, shadow matrix layout) live in graphics/gpu_limits.hpp because
// the Vulkan-free graphics layer also needs them to size its arrays. That
// header is included above, so every constant remains reachable through a
// single include of this file.

namespace fire_engine
{

// ---------------------------------------------------------------------------
// Camera projection — shared between Object::render (perspective matrix)
// and Renderer::updateLightData (cascade frustum fitting).
// ---------------------------------------------------------------------------

inline constexpr float kCameraFovRadians = 45.0f * deg_to_rad;
inline constexpr float kCameraNearPlane = 0.1f;
inline constexpr float kCameraFarPlane = 1000.0f;

// ---------------------------------------------------------------------------
// Directional light + IBL strengths. Keep diffuse IBL below direct sun so
// shadowed areas do not get filled back to near-white by bright environments.
// ---------------------------------------------------------------------------

inline constexpr float kDirectionalLightIntensity = 1.35f;
inline constexpr float kDiffuseIblStrength = 0.35f;
inline constexpr float kSpecularIblStrength = 0.7f;
inline constexpr float kSkyboxIntensity = 1.0f;
// Keep image-based lighting independent from CSM visibility by default. Direct
// light carries the shadow contrast; ambient light stays stable across assets.
inline constexpr float kEnvironmentShadowStrength = 0.0f;

// ---------------------------------------------------------------------------
// Shadow mapping — extent, cascading, bias, and filter radius.
// ---------------------------------------------------------------------------

inline constexpr uint32_t kShadowMapExtent = 2048;
// Past this distance, casters don't shadow — keeps the cascade ortho fits
// tight. Anything in shadow range stays inside [kCameraNearPlane, kShadowFarPlane].
inline constexpr float kShadowFarPlane = 50.0f;
// Pulls the light-space near plane back along lightDir so casters behind the
// fitted bounding sphere still write to the shadow map.
inline constexpr float kShadowDepthBackExtend = 20.0f;
// Practical Split Scheme blend between linear and log-uniform cascade splits.
// 0 = pure linear (cascades evenly spaced in view distance), 1 = pure log
// (each cascade covers a constant ratio of the previous). 0.5 keeps close
// cascades tight for near-camera detail while still covering kShadowFarPlane.
inline constexpr float kShadowCascadeSplitLambda = 0.5f;
inline constexpr float kShadowMinBias = 0.0008f;
inline constexpr float kShadowSlopeBias = 0.0035f;
inline constexpr float kShadowFilterRadius = 0.0f;
inline constexpr float kShadowNormalOffset = 0.0f;
// Keep directional caster bias conservative so contact shadows remain attached.
// Punctual lights use their own bias constants below.
inline constexpr float kDirectionalShadowRasterBiasConstant = 0.0f;
inline constexpr float kDirectionalShadowRasterBiasSlope = 0.0f;

// Punctual shadow caster caps (kMaxSpotShadowCasters / kMaxPointShadowCasters)
// and the derived ShadowUBO matrix layout live in graphics/gpu_limits.hpp.
inline constexpr uint32_t kSpotShadowMapExtent = 1024;
inline constexpr uint32_t kPointShadowMapExtent = 512;
inline constexpr uint32_t kSkinnedSelfShadowMapExtent = 1024;
inline constexpr float kSkinnedSelfShadowDepthEpsilon = 0.0005f;
inline constexpr float kPointSpotShadowMinBias = 0.005f;
inline constexpr float kPointSpotShadowSlopeBias = 0.01f;
inline constexpr float kPunctualShadowRasterBiasConstant = 1.25f;
inline constexpr float kPunctualShadowRasterBiasSlope = 1.75f;
inline constexpr float kPointShadowNearPlane = 0.1f;
// Substituted far plane for point lights with range==0 (glTF "infinite") so
// the cube projection stays finite. Used only for shadow-map projection.
inline constexpr float kPointShadowInfiniteRangeFallback = 100.0f;

// ---------------------------------------------------------------------------
// IBL precompute extents — chosen at engine start, baked into texture sizes.
// ---------------------------------------------------------------------------

// HDR environment cubemap. log2(extent)+1 mip levels — full chain so the
// prefilter pass can do Filament-style mip-weighted importance sampling.
inline constexpr uint32_t kSkyboxCubemapExtent = 1024;
inline constexpr uint32_t kSkyboxCubemapMipLevels = 11;

// Diffuse IBL — small cubemap is fine; convolution kernel is wide.
inline constexpr uint32_t kIrradianceCubemapExtent = 32;

// Specular IBL — per-mip roughness baking.
inline constexpr uint32_t kPrefilteredCubemapExtent = 128;
inline constexpr uint32_t kPrefilteredCubemapMipLevels = 8;

// Split-sum BRDF integration lookup table.
inline constexpr uint32_t kBrdfLutExtent = 256;

// ---------------------------------------------------------------------------
// Bloom — dual-filter chain inserted between forward HDR + post-process.
// ---------------------------------------------------------------------------

inline constexpr uint32_t kBloomMipCount = 6;
// 0 → bloom off (output bit-identical to pre-bloom). 0.04 is photographic.
inline constexpr float kBloomStrength = 0.04f;

} // namespace fire_engine
