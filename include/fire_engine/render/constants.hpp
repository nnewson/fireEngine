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
// Default IBL strengths when no skybox is drawn (no environment arg given). Calmer than the
// full-strength values above so the lighting sits between a bright skybox.hdr and a dark
// nightbox.hdr rather than blazing off an undrawn environment. Live-tunable via the overlay.
inline constexpr float kNoSkyboxDiffuseIblStrength = 0.2f;
inline constexpr float kNoSkyboxSpecularIblStrength = 0.4f;
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

// Shadow-LOD selection (SH-03). The budget is in SHADOW-MAP TEXELS of the view doing the
// rasterising — not camera pixels — so it means the same thing for a 2048-texel cascade and a
// 512-texel point face, which the retired camera-pixel heuristic could not.
//
// UNCALIBRATED starting values. SH-03's calibration step sweeps the budget at ratio 1.0 first (so
// the budget is measured without a dead band confusing the result), then widens the dead band until
// level chatter stops on the moving-light acceptance scene. Treat these as the baseline that
// exercise begins from, not as measured answers.
inline constexpr float kShadowLodPixelBudget = 2.0f;
// Coarsening must project within `budget * ratio`, while refining triggers at `budget` — the gap is
// the dead band. 1.0 disables hysteresis, which is deliberately where calibration starts.
inline constexpr float kShadowLodCoarsenRatio = 1.0f;

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

// Soft particles: eye-space distance (metres) over which a particle fades out as
// it approaches scene geometry, removing the hard clip edge at intersections.
inline constexpr float kParticleSoftFadeRange = 0.5f;

// ---------------------------------------------------------------------------
// SSAO + contact shadows — screen-space, reconstructed from the depth prepass.
// ---------------------------------------------------------------------------

// View-space hemisphere sampling radius (world units). Scene-scale dependent;
// 0.5 suits the sample assets (~1–3 unit models).
inline constexpr float kSsaoRadius = 0.5f;
// Depth bias (view units) added before the occlusion compare to avoid self-
// occlusion acne on near-flat surfaces.
inline constexpr float kSsaoBias = 0.025f;
// Occlusion strength multiplier and contrast power applied to the raw AO.
inline constexpr float kSsaoIntensity = 1.0f;
inline constexpr float kSsaoPower = 2.0f;
// Contact shadows (screen-space ray-march toward the sun): march length in view
// units and step count. Catches short-range contact the CSM misses.
inline constexpr float kContactShadowLength = 0.5f;
inline constexpr int kContactShadowSteps = 16;
// Depth-silhouette edge-guard threshold (view-space Z step, in world units) at
// which contact shadows fade to lit — suppresses the screen-space "hair" the
// ray-march leaves at object silhouettes. The smoothstep spans ±50% of this.
inline constexpr float kContactEdgeThreshold = 0.1f;

// ---------------------------------------------------------------------------
// Temporal anti-aliasing — sub-pixel jitter + velocity-reprojected history.
// ---------------------------------------------------------------------------

// Length of the Halton(2,3) jitter sequence cycled through the projection
// matrix. 8 spreads samples well without the history needing to remember too
// far back.
inline constexpr uint32_t kTaaJitterSamples = 8;
// History weight in the resolve blend: resolved = mix(current, history, this).
// 0.9 = heavy accumulation (smooth, slightly softer); 0 = TAA off (pure
// current frame).
inline constexpr float kTaaHistoryBlend = 0.9f;

} // namespace fire_engine
