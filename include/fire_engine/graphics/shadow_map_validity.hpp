#pragma once

#include "fire_engine/graphics/gpu_limits.hpp"

#include <cstddef>
#include <cstdint>

// Which shadow-map families a frame actually RECORDS — one decision, used twice.
//
// The shadow pass may legitimately skip a whole family: shadows switched off, no primary
// directional light for the cascades to be fitted to, no punctual caster. Skipping is only correct
// while the receiver knows it happened. A skipped family's depth image still holds whatever the
// last frame that did render it left behind, and "nothing samples it anyway" is an argument about
// the CURRENT arrangement of the forward shader, not a property of the data — the kind of accident
// that survives until someone adds a sampler and gets last-second-of-last-frame shadows.
//
// So the recording decision and the signal the shader reads are the SAME VALUE: the renderer
// computes this once per frame, gates the pass's families on it, and uploads its packed mask in
// LightUBO. Neither side can be right while the other is wrong, because there is only one of them.
//
// Vulkan-free and pure, so the policy is testable without a device — which matters here because the
// interesting cases (a partially fitted cascade set, a cube with a missing face) are exactly the
// ones that are awkward to stage against a GPU.

namespace fire_engine
{

// What the frame knows by the time the view set is complete. Counts are ACTIVE VIEWS from the
// authoritative set, not requests: what the pass will iterate over.
struct ShadowMapValidityInputs
{
    // The `--no-shadows` tunable. Suppresses recording as well as sampling, so nothing is drawn
    // into any map and nothing reads one.
    bool shadowsDisabled{false};
    // A primary directional light exists this frame. The cascade, world-only and self families are
    // all fitted to it; with no sun there is nothing for them to describe.
    bool primaryDirectionalLight{false};
    std::size_t activeCascadeViews{0};
    std::size_t activeWorldOnlyViews{0};
    std::size_t activeSelfViews{0};
    std::size_t activeSpotViews{0};
    // Flattened cube faces (lightSlot * kCubeFaceCount + face), as the set stores them.
    std::size_t activePointViews{0};
};

// One bit per family. Deliberately a struct of named bools rather than the mask itself: the callers
// that matter ask "is this family valid" (the pass, per group) and only the upload wants the packed
// form, so the packing is a projection instead of the currency.
struct ShadowMapValidity
{
    bool cascades{false};
    bool worldOnly{false};
    bool self{false};
    bool spot{false};
    bool point{false};

    [[nodiscard]] bool operator==(const ShadowMapValidity&) const noexcept = default;

    // The LightUBO field, using the bit values shared with the shader (shaders/gpu_limits.glsl).
    [[nodiscard]] std::int32_t packedMask() const noexcept;

    // True when the pass records nothing at all — the shadow groups can then skip their timestamps
    // as well as their draws.
    [[nodiscard]] bool none() const noexcept;
};

// The law.
//
// WHOLE-FAMILY, not "some slot is active", wherever the family is rendered as a unit:
//
//   * the cascades and their world-only twin are sampled by cascade INDEX, chosen per fragment from
//     the view depth. A fragment landing in cascade 2 samples layer 2 whether or not layer 2 was
//     fitted, so three fitted cascades out of four is not "three quarters valid" — it is a family
//     with a hole in it, and the honest answer is to record nothing and report invalid.
//   * a point light's six faces are installed atomically (`ShadowRenderViewSet::setPointLight`), so
//     a non-zero count that is not a whole number of cubes means that invariant has been broken
//     upstream; validity reports false rather than rendering part of a cube.
//   * self-shadow slots ARE independent — one caster per slot, sampled only by draws carrying that
//     slot — so the family is valid when any slot is active. Same for spot lights, which are
//     addressed by their own shadow index.
//
// The directional families additionally require a primary directional light: their fit has no
// meaning without one, and a fallback direction would produce a map that looks valid and describes
// a sun that is not in the scene.
[[nodiscard]] ShadowMapValidity shadowMapValidity(const ShadowMapValidityInputs& inputs) noexcept;

} // namespace fire_engine
