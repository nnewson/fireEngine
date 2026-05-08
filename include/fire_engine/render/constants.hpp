#pragma once

#include <cstddef>
#include <cstdint>

#include <fire_engine/math/constants.hpp>

// Single source of truth for engine-wide rendering tunables. Values that
// previously lived in renderer.cpp's anon namespace, were inlined inside
// updateLightData / Object::render, or sat in the EnvironmentConfig struct
// have all been pulled here so a tweak only needs to touch one file.

namespace fire_engine
{

// ---------------------------------------------------------------------------
// Frame and resource limits
// ---------------------------------------------------------------------------

inline constexpr int MAX_FRAMES_IN_FLIGHT = 2;
inline constexpr std::size_t MAX_JOINTS = 64;
inline constexpr int MAX_MORPH_TARGETS = 8;
// Cap on lights consumed by the forward shader's main lighting loop. Sized so
// the LightUBO array fits comfortably under any sane Vulkan UBO limit. Bump
// when scenes routinely exceed this; or swap to an SSBO at that point.
inline constexpr int MAX_LIGHTS = 8;

// ---------------------------------------------------------------------------
// Camera projection — shared between Object::render (perspective matrix)
// and Renderer::updateLightData (cascade frustum fitting).
// ---------------------------------------------------------------------------

inline constexpr float cameraFovRadians = 45.0f * deg_to_rad;
inline constexpr float cameraNearPlane = 0.1f;
inline constexpr float cameraFarPlane = 1000.0f;

// ---------------------------------------------------------------------------
// Directional light + IBL strengths. Keep diffuse IBL below direct sun so
// shadowed areas do not get filled back to near-white by bright environments.
// ---------------------------------------------------------------------------

inline constexpr float directionalLightIntensity = 1.35f;
inline constexpr float diffuseIblStrength = 0.35f;
inline constexpr float specularIblStrength = 0.7f;
inline constexpr float skyboxIntensity = 1.0f;
// Keep image-based lighting independent from CSM visibility by default. Direct
// light carries the shadow contrast; ambient light stays stable across assets.
inline constexpr float environmentShadowStrength = 0.0f;

// ---------------------------------------------------------------------------
// Shadow mapping — extent, cascading, bias, and filter radius.
// ---------------------------------------------------------------------------

inline constexpr uint32_t shadowMapExtent = 2048;
// Per-cascade layer in the 2D-array shadow map. 4 cascades cover camera-near
// through shadowFarPlane via log-uniform splits.
inline constexpr uint32_t shadowCascadeCount = 4;
// Past this distance, casters don't shadow — keeps the cascade ortho fits
// tight. Anything in shadow range stays inside [cameraNearPlane, shadowFarPlane].
inline constexpr float shadowFarPlane = 50.0f;
// Pulls the light-space near plane back along lightDir so casters behind the
// fitted bounding sphere still write to the shadow map.
inline constexpr float shadowDepthBackExtend = 20.0f;
inline constexpr float shadowMinBias = 0.0008f;
inline constexpr float shadowSlopeBias = 0.0035f;
inline constexpr float shadowFilterRadius = 0.0f;
inline constexpr float shadowNormalOffset = 0.0f;
// Keep directional caster bias conservative so contact shadows remain attached.
// Punctual lights use their own bias constants below.
inline constexpr float directionalShadowRasterBiasConstant = 0.0f;
inline constexpr float directionalShadowRasterBiasSlope = 0.0f;

// Shadow casters for punctual lights. Caps are independent of MAX_LIGHTS;
// excess punctual lights remain unshadowed. First-N policy in gather order.
// Total ShadowUBO matrix slots: 4 cascades + spot + point*6.
inline constexpr int MAX_SPOT_SHADOW_CASTERS = 4;
inline constexpr int MAX_POINT_SHADOW_CASTERS = 4;
inline constexpr uint32_t spotShadowMapExtent = 1024;
inline constexpr uint32_t pointShadowMapExtent = 512;
inline constexpr float pointSpotShadowMinBias = 0.005f;
inline constexpr float pointSpotShadowSlopeBias = 0.01f;
inline constexpr float punctualShadowRasterBiasConstant = 1.25f;
inline constexpr float punctualShadowRasterBiasSlope = 1.75f;
inline constexpr float pointShadowNearPlane = 0.1f;
// Substituted far plane for point lights with range==0 (glTF "infinite") so
// the cube projection stays finite. Used only for shadow-map projection.
inline constexpr float pointShadowInfiniteRangeFallback = 100.0f;

// ---------------------------------------------------------------------------
// IBL precompute extents — chosen at engine start, baked into texture sizes.
// ---------------------------------------------------------------------------

// HDR environment cubemap. log2(extent)+1 mip levels — full chain so the
// prefilter pass can do Filament-style mip-weighted importance sampling.
inline constexpr uint32_t skyboxCubemapExtent = 1024;
inline constexpr uint32_t skyboxCubemapMipLevels = 11;

// Diffuse IBL — small cubemap is fine; convolution kernel is wide.
inline constexpr uint32_t irradianceCubemapExtent = 32;

// Specular IBL — per-mip roughness baking.
inline constexpr uint32_t prefilteredCubemapExtent = 128;
inline constexpr uint32_t prefilteredCubemapMipLevels = 8;

// Split-sum BRDF integration lookup table.
inline constexpr uint32_t brdfLutExtent = 256;

// ---------------------------------------------------------------------------
// Bloom — dual-filter chain inserted between forward HDR + post-process.
// ---------------------------------------------------------------------------

inline constexpr uint32_t bloomMipCount = 6;
// 0 → bloom off (output bit-identical to pre-bloom). 0.04 is photographic.
inline constexpr float bloomStrength = 0.04f;

} // namespace fire_engine
