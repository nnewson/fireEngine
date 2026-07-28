#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <fire_engine/graphics/bounds.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/shadow_diagnostics.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// SH-02: the pure shadow-view projection model
// ([`docs/shadowplans.md`](../../../docs/shadowplans.md) § SH-02). Everything here is Vulkan-free
// and headless-testable — the renderer supplies the view descriptors (SH-03) but owns none of the
// maths.
//
// The defect this exists to fix: a caster's LOD is currently chosen from the CAMERA — its distance,
// projection and viewport — and the single resulting draw is replayed into every shadow view. A
// shadow's error budget is in SHADOW-MAP TEXELS of the view that rasterises it, which has nothing
// to do with where the camera happens to be standing.

// Which projection a shadow map layer uses.
enum class ShadowViewKind : std::uint8_t
{
    // Directional cascades, the world-only cascades, AND the per-caster self-shadow layers: all
    // orthographic, so texel size is constant across the layer and independent of depth.
    Orthographic,
    // Spot maps and the six faces of a point map. Texel size grows with distance from the light, so
    // the same caster projects a smaller error the further away it sits.
    Perspective,
};

// One shadow map layer, described in the terms selection actually needs.
//
// ENCAPSULATED deliberately: the only way to obtain one is through a factory that normalises and
// validates. Public fields would let a call site assemble a view that claims to be valid while
// carrying an un-normalised forward or a degenerate texel size — and the resulting selection would
// look entirely plausible, which is the worst kind of wrong for a value that decides how much
// geometry a shadow keeps.
class ShadowView
{
public:
    // An orthographic layer — a directional cascade, a world-only cascade, or a self-shadow layer.
    // `worldUnitsPerTexel` must be finite and > 0.
    [[nodiscard]] static ShadowView orthographic(float worldUnitsPerTexel) noexcept;

    // A perspective layer — a spot map, or ONE face of a point map (each face is its own view,
    // since each has its own forward). NORMALISES `forward` itself: relying on the caller to pass a
    // unit vector would make every call site a place to get depth silently wrong. A zero-length or
    // non-finite direction yields an invalid view.
    [[nodiscard]] static ShadowView perspective(const Vec3& lightPosition, const Vec3& forward,
                                                float fovRadians, std::uint32_t extentTexels,
                                                float nearPlane) noexcept;

    [[nodiscard]] ShadowViewKind kind() const noexcept
    {
        return kind_;
    }
    // Orthographic: the world-space size of one shadow-map texel. Constant over the layer.
    [[nodiscard]] float worldUnitsPerTexel() const noexcept
    {
        return worldUnitsPerTexel_;
    }
    [[nodiscard]] const Vec3& lightPosition() const noexcept
    {
        return lightPosition_;
    }
    // Always unit length on a valid perspective view — the factory guarantees it.
    [[nodiscard]] const Vec3& forward() const noexcept
    {
        return forward_;
    }
    [[nodiscard]] float tanHalfFov() const noexcept
    {
        return tanHalfFov_;
    }
    [[nodiscard]] float nearPlane() const noexcept
    {
        return nearPlane_;
    }
    [[nodiscard]] std::uint32_t extentTexels() const noexcept
    {
        return extentTexels_;
    }
    // False when the descriptor could not be built (a degenerate texel size, a zero-length or
    // non-finite direction, a non-positive fov/extent/near plane). Selection returns LOD0 with an
    // explicit reason rather than projecting through nonsense.
    [[nodiscard]] bool valid() const noexcept
    {
        return valid_;
    }

private:
    // Private: a ShadowView is only ever produced by a factory that normalises and validates. An
    // INVALID view is still produced that way — `perspective()` given a zero-length forward returns
    // one that keeps its Perspective kind — so "invalid" never means "default-constructed", and
    // nothing needs a default to represent it.
    ShadowView() = default;

    ShadowViewKind kind_{ShadowViewKind::Orthographic};
    float worldUnitsPerTexel_{0.0f};
    Vec3 lightPosition_{};
    Vec3 forward_{0.0f, 0.0f, -1.0f};
    float tanHalfFov_{0.0f};
    float nearPlane_{0.0f};
    std::uint32_t extentTexels_{0};
    bool valid_{false};
};

// Nearest distance from the light along its FORWARD axis to `worldBounds` — light-view z, not
// radial distance, because the perspective texel size varies with depth along the view axis.
//
// The minimum over all EIGHT corners: a bounding sphere's centre distance would under-estimate for
// a box straddling the light, and a radial distance would over-estimate off-axis (a corner far to
// the side is further away radially than it is deep). Returns a non-positive value when the bounds
// reach the light plane or behind it, which the caller must treat as a near-plane intersection.
[[nodiscard]] float nearestForwardDepth(const ShadowView& view,
                                        const Bounds3& worldBounds) noexcept;

// A world-space error, projected into shadow-map texels of `view`.
//
// METRIC: RADIAL texel displacement — the magnitude of the worst-case displacement in the map's
// texel grid, not a per-axis component. That choice matters for the perspective case: a world error
// with a depth component projects MORE strongly near a frustum edge than on the axis, because the
// same depth change slides a point further across the image there. The on-axis estimate
// `extent / (2 · tanHalfFov · depth)` understates that, so the factor `sqrt(1 + 2 · tanHalfFov²)`
// bounds the projection's Lipschitz constant over the whole square frustum:
//
//     texels = worldError · extent · sqrt(1 + 2 · tanHalfFov²) / (2 · tanHalfFov · (depth −
//     worldError))
//
// The `depth − worldError` is not a fudge: the displacement is FINITE and the projection steepens
// toward the light, so evaluating at the undisplaced depth under-bounds a displacement moving that
// way — measured at ~0.3% over on a frustum corner. The subtraction evaluates at the closest depth
// the displaced surface can reach. A result at or behind the near plane returns infinity.
//
// Orthographic needs no such factor — texel size is constant everywhere in the layer:
//
//     texels = worldError / worldUnitsPerTexel
//
// `nearestDepth` is ignored for orthographic views. Returns infinity for an invalid view, a
// non-finite error, or a perspective depth at or behind the near plane, so a caller cannot mistake
// a broken projection for a small error.
[[nodiscard]] float projectShadowErrorTexels(float worldError, const ShadowView& view,
                                             float nearestDepth) noexcept;

// Hysteresis for a caster's level in ONE logical shadow view across frames. Without it a caster
// drifting across a threshold flips level every other frame, which reads as a shadow silhouette
// twitching in place.
struct ShadowLodHysteresis
{
    // Coarsening requires the candidate to project within `budget * coarsenRatio`, while refining
    // triggers as soon as the current level exceeds `budget`. The gap between the two is the dead
    // band. Must satisfy 0 < coarsenRatio <= 1; 1 disables hysteresis (both directions at the same
    // threshold), and smaller values demand more margin before giving up detail.
    //
    // The default is DELIBERATELY INVALID. Any plausible-looking number here would be an unmeasured
    // tuning decision inherited silently by every caller; the selector rejects this one with
    // InvalidCaster, so a caller must state a ratio it chose. SH-03 supplies it from
    // render/constants.hpp alongside the texel budget.
    float coarsenRatio{-1.0f};
};

// "No previous level" — a caster's first frame in a view. Distinct from level 0, which is a real
// previous decision that hysteresis must respect.
inline constexpr std::size_t kNoPreviousShadowLod = static_cast<std::size_t>(-1);

// The outcome of one selection: which level, why, and what it projected to.
struct ShadowLodSelection
{
    std::size_t level{0};
    ShadowLodReason reason{ShadowLodReason::Count};
    // Projected texel error of the CHOSEN level; 0 for LOD0, infinity for a forced fallback. The
    // SH-01 panel reports this, so a forced LOD0 is visibly different from a deliberate one.
    float projectedTexels{0.0f};
};

// Select the coarsest cut whose conservatively accumulated shadow-deviation estimate projects
// within `budgetTexels` for THIS view — the SH-02 acceptance statement. The estimate's adequacy is
// judged empirically against full-detail shadow masks; it is not a certified bound.
//
// `model` is the caster's world transform (its linear part scales the object-space deviation into
// world space via the shared conservative σ_max). `worldBounds` positions it for perspective depth.
// `previousLevel` is the level this caster held in this view last frame, or kNoPreviousShadowLod.
//
// CONTRACT for `previousLevel`: it is valid only for the SAME draw and the SAME LOGICAL view. A
// physical spot/point slot reassigned to a different light is a different logical view, and feeding
// the old level across that boundary applies one caster's hysteresis to another's geometry. An
// out-of-range value is not clamped — it produces an explicit fallback, because clamping would hide
// exactly that plumbing bug.
[[nodiscard]] ShadowLodSelection selectShadowLod(std::span<const GeometryLod> lods,
                                                 const ShadowView& view, const Mat4& model,
                                                 const Bounds3& worldBounds, float budgetTexels,
                                                 ShadowLodHysteresis hysteresis,
                                                 std::size_t previousLevel) noexcept;

} // namespace fire_engine
