#pragma once

#include <fire_engine/render/constants.hpp>

namespace fire_engine
{

// Selects which debug-view branch the forward fragment shader runs. Mapped
// 1:1 to LightUBO::environmentParams[2] (0..5) — keep the enum and shader
// constants in lockstep.
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
};

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
