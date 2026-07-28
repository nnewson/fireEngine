#pragma once

#include <array>
#include <cstddef>

#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/render/constants.hpp>

namespace fire_engine
{

// Selects which debug-view branch the forward fragment shader runs. Values 0..8 map
// 1:1 to LightUBO::environmentParams[2] — keep the enum and shader constants in lockstep.
// `Joints` (9) is the exception: it has no shader branch — instead it replaces the scene
// geometry with the ragdoll articulation gizmo + labels, and is forced to `None` when
// written to the shader (see Renderer::updateLightData). Keep it LAST, after any new
// shader-backed view, so the shader-backed range stays contiguous from 0.
enum class DebugView : int
{
    None = 0,
    Normals = 1,
    NdotL = 2,
    Shadow = 3,
    ShadowDepth = 4,
    // Visualises the TAA motion-vector attachment (abs screen-space velocity,
    // scaled). Sanity check for the motion vectors before the resolve consumes them.
    Velocity = 5,
    // Raw SSAO + contact term (grayscale), before it modulates ambient.
    Ssao = 6,
    // Tints each mesh by its selected discrete LOD level (0 green → 1 yellow → 2 red → …).
    Lod = 7,
    // Tints each mesh by the level its SHADOW draw selected — same palette as Lod, plus a neutral
    // grey for meshes that cast no shadow (they have no level to report; see kNoShadowLod). Reading
    // the two views side by side is how the SH-01 camera-derived-shadow-LOD defect shows up on
    // screen rather than only in the counters.
    ShadowLod = 8,
    // Overlay-only (no shader branch): suppresses the scene meshes and draws a per-joint RGB
    // axis gizmo + an "index: bone-name" label for each articulation link, so a skeleton's
    // joints can be identified (e.g. to author per-bone hinge limits).
    Joints = 9,
    Count
};

// Display names, kept BESIDE the enum so adding a view without naming it fails to compile (the
// table previously lived in the overlay, where it could — and did — drift out of step).
inline constexpr std::array kDebugViewNames{
    "None",     "Normals", "N·L",      "Shadow",          "Shadow depth",
    "Velocity", "SSAO",    "LOD tint", "Shadow LOD tint", "Joints"};
static_assert(kDebugViewNames.size() == static_cast<std::size_t>(DebugView::Count),
              "every DebugView needs a display name, in enum order");

// Live, runtime-editable render parameters surfaced by the debug overlay. Seeded
// from the compile-time constants (and the CLI debug flags) at startup; the
// Renderer reads these every frame instead of the constexprs so the overlay can
// tune them without a recompile.
struct RenderTunables
{
    // Temporal anti-aliasing.
    bool taaEnabled{true};
    float taaHistoryBlend{kTaaHistoryBlend};
    float taaSharpen{0.0f}; // 0 = off; post-resolve unsharp amount

    // Frustum culling: skip draws whose world bounds fall outside the camera / shadow
    // frustums. Off = submit everything (A/B comparison + a regression escape hatch).
    bool cullingEnabled{true};

    // Debug visualisation.
    DebugView debugView{DebugView::None};
    bool noShadows{false};

    // Physics debug draw (wireframes into the scene). debugDepthTest off = x-ray
    // (drawn over geometry); on = occluded by the scene.
    bool debugDrawAabbs{false};
    bool debugDrawColliders{false};
    bool debugDrawContacts{false};
    bool debugDepthTest{false};

    // Lighting / post.
    float bloomStrength{kBloomStrength};
    float diffuseIbl{kDiffuseIblStrength};
    float specularIbl{kSpecularIblStrength};
    float directionalIntensityScale{1.0f}; // multiplies the primary directional light

    // Discrete mesh LOD: pick a coarser index set for distant/small static meshes within a
    // screen-space pixel error budget. The toggle doubles as the A/B regression escape hatch.
    bool lodEnabled{true};
    float lodPixelErrorBudget{kLodPixelErrorBudget};
    // SH-03: the shadow-LOD budget, in SHADOW-MAP TEXELS of the view doing the rasterising. Kept
    // beside the camera budget but deliberately separate — the two are in different units and are
    // calibrated against different evidence, and sharing one number is what made every shadow view
    // rasterise the camera's choice.
    float shadowLodPixelBudget{kShadowLodPixelBudget};
    // Discrete = hard LOD swaps (Phase 1); Continuous = VIPM geomorph (Phase 2). Coexist —
    // selectable.
    LodMode lodMode{LodMode::Discrete};
    // Global CPU/GPU selector for the VDPM (ViewDependent) front (rendering-spine #3). When true
    // AND the device meets the compute/scan capability, the Renderer runs each visible front's
    // lifecycle on the GPU and draws from the GPU-emitted index/indirect buffers; unsupported
    // hardware or a per-mesh ineligibility falls back (logged once) to the CPU front. Seeded by the
    // Renderer from the tri-state CLI request resolved against device capability (B5c-4: defaults
    // ON where supported); toggled at runtime by the overlay "GPU-driven front" checkbox.
    bool vdpmGpuBackend{false};

    // SSAO + contact shadows (screen-space, from the depth prepass). When
    // ssaoEnabled is false the pass still runs but writes AO = 1 (no darkening).
    bool ssaoEnabled{true};
    float ssaoRadius{kSsaoRadius};
    float ssaoBias{kSsaoBias};
    float ssaoIntensity{kSsaoIntensity};
    float ssaoPower{kSsaoPower};

    // Contact shadows.
    bool contactShadowsEnabled{true};
    float contactShadowLength{kContactShadowLength};
    // View-space-Z step at which contact shadows fade out near silhouettes
    // (kills the screen-space "hair"). Higher = guard only the sharpest edges.
    float contactEdgeThreshold{kContactEdgeThreshold};

    // Particles — scales applied to every gathered emitter before the sim.
    float particleRateScale{1.0f};
    float particleLifetimeScale{1.0f};
    float particleSizeScale{1.0f};

    // Soft-body / cloth solver (defaults mirror the solver's former constants).
    int clothSubsteps{20};
    float clothComplianceScale{1.0f}; // global multiplier on authored per-type compliance
    float clothDamping{0.99f};
    float clothGravity{-9.8f}; // world-Y acceleration
    float clothWind[3]{0.0f, 0.0f, 0.0f};
};

} // namespace fire_engine
