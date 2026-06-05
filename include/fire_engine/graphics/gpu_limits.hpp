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

namespace fire_engine
{

// Frames-in-flight: how many copies of per-frame GPU resources exist.
inline constexpr int kMaxFramesInFlight = 2;

// Skinning joint matrices per SkinUBO.
inline constexpr std::size_t kMaxJoints = 64;

// Morph target weights per MorphUBO.
inline constexpr int kMaxMorphTargets = 8;

// Cap on lights consumed by the forward shader's main lighting loop. Sized so
// the LightUBO array fits comfortably under any sane Vulkan UBO limit. Bump
// when scenes routinely exceed this; or swap to an SSBO at that point.
inline constexpr int kMaxLights = 8;

// Per-skinned-object self-shadow slots (LightUBO::selfShadowViewProj).
inline constexpr int kMaxSkinnedSelfShadowCasters = 4;

// Directional cascade layers in the 2D-array shadow map.
inline constexpr uint32_t kShadowCascadeCount = 4;

// Shadow casters for punctual lights. Caps are independent of kMaxLights;
// excess punctual lights remain unshadowed. First-N policy in gather order.
inline constexpr int kMaxSpotShadowCasters = 4;
inline constexpr int kMaxPointShadowCasters = 4;

// Shadow vertex shader projects each vertex into light-space using one of the
// ShadowUBO::lightViewProj matrices, picked via ShadowPushConstants::matrixIndex.
//   [0..3]   directional cascades 0..3
//   [4..]    spot lights, layout 4 + spotIndex
//   [4+S..]  point lights, layout (4 + S) + 6 * cubeIndex + face
// where S = kMaxSpotShadowCasters.
inline constexpr int kShadowCascadeMatrixBase = 0;
inline constexpr int kShadowSpotMatrixBase = 4;
inline constexpr int kShadowPointMatrixBase = kShadowSpotMatrixBase + kMaxSpotShadowCasters;
inline constexpr int kShadowTotalMatrixCount = kShadowPointMatrixBase + 6 * kMaxPointShadowCasters;

} // namespace fire_engine
