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
// Pulls the light-space near plane back along lightDir so casters behind the fitted bounding sphere
// still write to the shadow map.
//
// SH-06 RETIRED THIS AS THE POLICY. The cascade depth range is now fitted to the frame's actual
// caster bounds (`fitCasterAwareCascadeDepth`), which is what a fixed distance could never do: too
// small clips a caster (measured — the probe's shadow came out 14% short linearly, 26% by area),
// too large spends depth precision on empty space, and no constant is right for an arbitrary scene.
// It survives for exactly two uses: `fitLegacyCascadeDepth`, which is what the caster-aware policy
// is measured against, and the fallback that policy takes when a frame contains a caster whose
// bounds cannot bound it (cloth). Retire it fully when storage geometry carries a conservative
// envelope.
inline constexpr float kShadowDepthBackExtend = 20.0f;
// Practical Split Scheme blend between linear and log-uniform cascade splits.
// 0 = pure linear (cascades evenly spaced in view distance), 1 = pure log
// (each cascade covers a constant ratio of the previous). 0.5 keeps close
// cascades tight for near-camera detail while still covering kShadowFarPlane.
inline constexpr float kShadowCascadeSplitLambda = 0.5f;
// The forward shader cross-fades into cascade i+1 over the last fraction of cascade i's view-depth
// range (`cascadeBlendFactor` in `shaders/shader.frag`), so receivers in that band sample BOTH
// maps.
//
// SH-06 made that a fitting constraint rather than a shader detail: cascade i+1 is now fitted from
// the start of the band, not from the hard split, or its tightly-fitted XY rectangle and depth
// range would not cover receivers that legitimately sample it — the fit would be exactly as wrong
// as the fixed extension was, in the other direction.
//
// THIS is the single authority. The value is uploaded in `LightUBO::cascadeParams.x` and the shader
// reads it from there (`cascadeBlendFactor` in `shaders/shader.frag`), so changing it here changes
// both the fitting and the blending together — there is no shader-side literal to keep in step.
inline constexpr float kShadowCascadeBlendFraction = 0.1f;
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
// CALIBRATED (SH-03 slice 6) on `shadow_lod/ShadowLodDemo.gltf` — reproduce with
// `tools/shadow_lod_sweep.sh`, which is the procedure, not a remembered one.
//
// The reference is `--no-shadow-lod`: forward LOD untouched, every caster at shadow LOD0. NOT
// `--no-lod` (it also changes the visible geometry, so the comparison stops being about shadows)
// and NOT a tiny budget (selection still runs, and a cut with zero estimated deviation stays
// eligible). The metric compares the SHADOW VISIBILITY image (`--debug-shadow`) and counts pixels
// whose shadow state differs by >8/255, because shadow error is localised to silhouette edges and
// an image-wide average dilutes it away.
//
// The shadowed area is MEASURED, not assumed: the pixels that differ between the reference and the
// same view with `--no-shadows`. It came out at 12.07% of the frame — 10.4% before SH-05, which is
// most of why this table was re-measured (see the notes below). (A first pass called "darker than
// half" shadowed, which counted the night skybox and every dark material — 39.8% — and flattered
// every percentage by ~3.8x.) The reference captured twice gives a noise floor of exactly zero, so
// every number below is signal — and a NON-ZERO noise floor invalidates the run (see the SH-06
// note).
//
// MEASURED ON MERGED `main` (SH-05 + SH-06 together), 2026-08-04:
//
//   budget   differing shadow px   worst px   cascade tris (of 43472 at full detail)
//   0.5      0.000%                 0/255     23272  (53.5%)  identical error to the reference
//   1        0.003%                60/255     22120  (50.9%)
//   2        0.249%               121/255     20968  (48.2%)
//   4        0.322%               121/255     19624  (45.1%)
//   8        1.426%               161/255     16200  (37.3%)
//   16       5.103%               172/255     13064  (30.1%)
//
// ACCEPTANCE THRESHOLD, registered before the numbers were corrected: at most 0.1% of the shadowed
// pixels may differ from full detail, AND the differences must sit on silhouette edges rather than
// in filled regions. Under it only 1 passes (0.003%); 2 and 4 fail. An earlier table with the
// inflated denominator put 4 at 0.093% and selected it — the threshold is deliberately NOT being
// widened to preserve that answer, because moving a stated criterion after seeing the data is how a
// calibration becomes a rationalisation. The cost of holding the line is 2.7 percentage points of
// geometry (50.9% of full detail rather than 48.2%); the gain is an eightyfold smaller error.
//
// There is no knee in the savings curve to appeal to — each doubling keeps buying triangles — so a
// threshold is the only honest basis for the choice.
//
// SCOPE: this calibrates the CSM. `--debug-shadow` visualises the primary directional visibility
// and the triangle column is the cascade row, so cascade and world-only are measured; spot, point
// and self share the constant but their QUALITY is not measured here. Extending the evidence to the
// punctual families needs a per-family visibility view, and is worth doing before this constant is
// treated as globally validated.
//
// RE-DERIVED AFTER SH-04 (deformable casters forced to LOD0), because that changes the caster mix
// and the tables above could not be assumed to survive it. What actually moved:
//
//   * the error column did NOT move — not one figure. WHY is not established. The honest statement
//     is that the composite visibility mask is unchanged and the available instrumentation cannot
//     attribute that. It is tempting to conclude the metric never saw the deformable casters — the
//     panel reports 13 deformable resolutions scene-wide against 0 in cascade 0 — but that does not
//     follow: `primaryDirectionalVisibility` (shader.frag) selects whichever cascade the RECEIVER
//     depth lands in, not cascade 0, and on skinned receivers it folds in
//     `min(worldShadow, selfShadow)`. So deformable casters can contribute to this mask through
//     cascades nobody focused and through the self term. Attributing the result needs the
//     per-family visibility view already listed as missing below; until it exists, "unchanged" is
//     the measurement and the cause is open;
//   * COST moved, which is where the change lives: the skinned self-shadow caster went from
//     184/1248 triangles to 1248/1248 — it had been drawing a level chosen from a bind-pose error
//     claim — and the cascade group went from 59.9% to 68.2% of full detail at this budget.
//
// The 0.1% threshold was re-applied unchanged, and still selects 1: budget 2 remains at 0.243% and
// 4 at 0.356%. Budget 2 becoming eligible once deformable error disappeared was a live possibility
// worth checking, and it did not happen.
//
// Re-run 2026-08-01 after the glTF animated-light fix and reproduced to every printed digit. That
// is the expected result rather than a lucky one: the budget half of the sweep runs on the STATIC
// `ShadowLodDemo`, whose sun sits on a node with no animation and was therefore never dropped. The
// dead-band half, which did run under a fallback sun, is re-measured below.
//
// Re-run 2026-08-03 after SH-06's caster-aware depth fit, which changes the shadow matrices (a
// tighter depth range, and each cascade's slice expanded to cover the previous one's blend band):
// budget 2 moved 0.243% -> 0.289% and budget 4 0.356% -> 0.374%, with 8 and 16 within a hundredth.
// The 0.1% threshold still selects 1 by a wide margin. Measured on an IDLE machine — a run that
// overlapped a Docker CI build reported a 0.12% noise floor where an idle one reports 0.0000%, and
// eight identical-flag capture pairs differed by zero pixels, so contention (not the renderer)
// produced that sample. Discard any sweep whose noise-floor line is not zero.
//
// RE-MEASURED AFTER SH-05 (material-aware casters), 2026-08-04 — the whole table above, in one run.
// SH-05 changed the caster mix in the one way this calibration cannot absorb: casters that
// contributed NOTHING to the shadow mask now contribute (the double-sided sheet was front-culled
// out of every view; the alpha-masked cutout cast a solid rectangle where it should cast a
// perforated one). What moved:
//
//   * the MEASURED SHADOWED AREA grew 10.4% -> 12.08% of the frame, exactly as predicted, and that
//     is a bigger denominator for every relative error: 2 went 0.243% -> 0.210%, 4 0.356% ->
//     0.306%, 8 1.551% -> 1.439%, 16 5.837% -> 5.129%. Budget 1 is unchanged at 0.003%. The
//     absolute differing-pixel counts moved very little; nearly all of the change is the
//     denominator;
//   * the TRIANGLE column is now measured post-SH-04 and post-SH-05 (it had been a pre-SH-04
//     measurement carried forward with a caveat). At budget 1 it reads 29658/43472 = 68.2%,
//     matching the 68.2% SH-04's re-derivation reported, and the whole column sits a CONSTANT +3612
//     triangles above the old pre-SH-04 figures at every budget — the signature of casters pinned
//     to level 0 regardless of budget, which is what a fallback is;
//   * SH-05's own contribution to that column is ~nil ON THIS SCENE, and the reason is worth
//     recording rather than inferring: the alpha-masked caster is a two-triangle quad
//     (`quad_geometry` in `tools/assetgen/geometry.py`), so it carries a single level and would
//     have drawn its whole mesh anyway. Before SH-05 it reported `SingleLevel`; now it reports
//     `AlphaMaskedFallback`. The PIN is therefore untested by cost here — the panel reason is the
//     only visible effect, and a scene with a cutout carrying a real LOD chain would be needed to
//     price it.
//
// The 0.1% threshold was re-applied unchanged and still selects 1. Budget 2 becoming eligible on
// the larger denominator was a live possibility — 0.210% is still twice the threshold, so it did
// not happen.
//
// RE-MEASURED ON MERGED `main` (2026-08-04) — and this was NOT optional. SH-05 and SH-06 were cut
// from the same base, so each note above was measured without the other, and they move the same
// figures in OPPOSITE directions: SH-06 took budget 2 to 0.289% (a tighter depth range), SH-05 to
// 0.210% (a larger area to divide by). Neither described the merged engine, and the merged value is
// not the mean of them. The table above IS the merged run. What it shows:
//
//   * the ERROR column lands between the two single-item runs, as the mechanism predicts: budget 2
//     at 0.249% (0.243% before either), 4 at 0.322%, 8 at 1.426%, 16 at 5.103%. Budget 1 is
//     unchanged at 0.003%, and 0.5 still matches the reference exactly. The shadowed area settled
//     at 12.07%, so SH-06's fit barely moved the area SH-05 grew;
//   * the TRIANGLE column dropped hard — 50.9% of full detail at budget 1, against 68.2% measured
//     on SH-05 alone — because SH-06's caster-aware depth range also CULLS more: the cascade group
//     draws 30 of 52 candidate draws per frame where it drew 39 before. A range that ends at the
//     receiver volume rejects casters the old fixed back-extension kept. That is a cost win the
//     shadow-LOD budget did not buy and must not be credited with;
//   * the DEAD-BAND half is unchanged again (3 / 1 / 1 transitions over ~675 frames, ZERO reversals
//     at every ratio), so ratio 1.0 still stands on the same evidence.
//
// The 0.1% threshold was re-applied unchanged and STILL selects budget 1. It has now survived four
// independent re-derivations — SH-04, SH-06, SH-05, and this merged run — without being moved,
// which is the point of registering a criterion before seeing the data.
inline constexpr float kShadowLodPixelBudget = 1.0f;
// Coarsening must project within `budget * ratio`, while refining triggers at `budget` — the gap is
// the dead band, and 1.0 disables it.
//
// Kept at 1.0 ON EVIDENCE. A smaller ratio WIDENS the dead band (coarsening must clear a stricter
// threshold), which costs triangles — a caster holds finer geometry longer — to buy stability.
//
// Measured AT THIS BUDGET (chatter depends on which thresholds casters sit near, so a ratio swept
// at a different budget would not reproduce this decision), over ~600-frame runs of the animated
// scene, per 100 frames:
//
//   ratio 1.0   0.50 transitions   0.00 REVERSALS
//   ratio 0.75  0.16 transitions   0.00 REVERSALS
//   ratio 0.5   0.16 transitions   0.00 REVERSALS
//
// Re-run again after SH-06's caster-aware depth fit (2026-08-03): 3, 1 and 1 raw events once more,
// over 1323 / 1250 / 642 frames, so the per-100 rates read 0.23 / 0.08 / 0.16 — the rate moves with
// the frame count, which is what "counting noise" means. Zero reversals at every ratio, unchanged.
//
// RE-MEASURED 2026-08-01, after the glTF loader stopped dropping lights on animated nodes. Until
// then `ShadowLodMotionDemo`'s authored sun was silently replaced by the engine's fallback
// directional, so the scene's sun-swing animation did nothing and every dead-band figure taken on
// it described a STATIC sun. The rates barely moved (0.46/0.16/0.15 before) because they are 3, 1
// and 1 raw events over ~600 frames — dominated by counting noise — and the REVERSAL column, the
// only one that can justify a ratio, is still zero at every ratio.
//
// Chatter is a reversal that undoes a RECENT transition (within kReversalWindowCommits) — a caster
// oscillating across a threshold. Counting every return would have scored the scene's periodic
// animation as instability: an earlier, time-blind reversal count reported 0.31 per 100 frames,
// which was a caster legitimately walking L1 -> L2 and back as the MOVING CASTER crossed a
// threshold. (That was first written up as the sun swinging, which it cannot have been: under the
// loader defect above the sun never moved. The correction does not change the finding.) With the
// window applied there is no chatter at all, at any ratio.
//
// Plain transitions likewise include ordinary motion that no dead band can or should remove, so
// only the reversal column can justify a ratio — and it is zero.
//
// RE-MEASURED after SH-05 and again on merged `main` (2026-08-04), both in the same sweep run as
// the budget table: 3 / 1 / 1 raw transitions over ~675 frames = 0.44 / 0.15 / 0.15 per 100 frames,
// and ZERO reversals at every ratio, in both runs. Statistically indistinguishable from the figures
// above, which is the expected result rather than a lucky one: the animated scene's masked caster
// is now pinned, so it cannot transition at all, and it was a single-level quad that never
// transitioned before either. Ratio 1.0 stands on the same evidence — no chatter to buy out.
//
// Revisit when the caster mix changes again. Instruments: the overlay's "LOD movement" line, the
// per-frame `FE_LOG=render:debug` record, and `tools/shadow_lod_sweep.sh`.
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
