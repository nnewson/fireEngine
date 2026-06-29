#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <fire_engine/collision/contact_manifold.hpp>
#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/solver_math.hpp>

namespace fire_engine
{

// TGS soft-step contact solver (P2 + P3, modernised P9.2). Resolves a set of contact
// manifolds into corrected linear + angular velocities (normal + friction impulses with
// lever-arm torque). Effective mass per contact is
//   invMassA + invMassB + (rA×d)·IA⁻¹(rA×d) + (rB×d)·IB⁻¹(rB×d),
// the angular terms vanishing for a centred contact (reproducing the P2 linear case).
//
// Penetration is corrected by a **soft constraint** (Box2D-v3 `b2MakeSoft`) — a damped
// spring whose bias velocity closes the gap and whose impulse-decay term dissipates
// energy — instead of the old split-impulse position pass. The caller drives a TGS
// substep loop: prepare() once per step, then per substep warmStart → solveVelocity(true)
// (bias) → integrate positions → solveVelocity(false) (relax, no bias); applyRestitution()
// once at the end. Each substep recomputes the contact separation analytically from the
// moving bodies (the anchor stored body-local at prepare, rotated by the current pose).
//
// The solver is deliberately decoupled from PhysicsWorld: it operates on a flat
// `SolverBody` array (see solver_math.hpp) and a list of SolverContactInput, so it
// can be unit-tested in isolation. The per-body math (world inverse inertia, relative
// velocity, effective mass, apply-impulse) is shared with the joint solver via
// solver_math.hpp.

// A contact pair handed to the solver. `bodyA`/`bodyB` index the SolverBody
// array; `normal` points B -> A (target -> moving), i.e. the direction body A is
// pushed to separate. `key` identifies the pair for warm starting.
struct SolverContactInput
{
    int bodyA{-1}; // "moving"
    int bodyB{-1}; // "target"
    Vec3 normal{};
    std::array<Vec3, kMaxManifoldPoints> points{};
    std::array<float, kMaxManifoldPoints> penetration{};
    int pointCount{0};
    float restitution{0.0f};
    float friction{0.0f};
    std::uint64_t key{0};
};

class ContactSolver
{
public:
    ContactSolver() = default;
    ~ContactSolver() = default;

    ContactSolver(const ContactSolver&) = default;
    ContactSolver& operator=(const ContactSolver&) = default;
    ContactSolver(ContactSolver&&) noexcept = default;
    ContactSolver& operator=(ContactSolver&&) noexcept = default;

    // Build constraint points from the contacts: anchors stored body-local (for
    // per-substep separation tracking), effective masses, soft-constraint coefficients
    // (at the substep `h`), and the initial relative-normal velocity for restitution.
    // Clears any state from the previous step.
    void prepare(std::span<const SolverBody> bodies, std::span<const SolverContactInput> contacts,
                 float h);

    // Apply the (warm-started) accumulated impulses. Called once per substep, after
    // the velocity integration, before solving.
    void warmStart(std::vector<SolverBody>& bodies) const;

    // One velocity-constraint Gauss-Seidel sweep (friction then normal). `useBias`
    // adds the soft penetration bias (the spring push-out) and its mass/impulse
    // scaling; the no-bias *relax* pass (useBias = false) projects out only the
    // residual velocity, removing the bias velocity so it cannot pump energy.
    // Separation is recomputed from the current body poses each call.
    void solveVelocity(std::vector<SolverBody>& bodies, bool useBias);

    // End-of-step restitution: for points that were approaching faster than the
    // threshold and actually engaged, drive the relative-normal velocity to the
    // restitution rebound speed. Applied once, after the substep loop.
    void applyRestitution(std::vector<SolverBody>& bodies);

    // Kinematic-only split-impulse position correction. Dynamic bodies resolve
    // penetration through the soft bias + substep position integration (positionWeight
    // 0 for them), so this pass only nudges scene-driven Kinematic bodies (positionWeight
    // > 0) out of penetration — a pseudo-velocity push-out that never alters real
    // velocity. Run once, after the substep loop. (No-op when no body carries weight.)
    void solvePosition(std::vector<SolverBody>& bodies);

    // Warm-start persistence is split into three calls so a single step can run
    // many independent per-island solves through one solver instance: `beginStore`
    // once before the islands clears the pending cache, `store` after each island's
    // solve appends that island's impulses, and `commitStore` once after all islands
    // swaps the pending cache in as next frame's source. (Each pair key belongs to
    // exactly one island, so the appends never collide.) A single global solve is
    // just begin → store → commit, identical to the old single `store()`.
    void beginStore() noexcept;
    void store();
    void commitStore() noexcept;

    [[nodiscard]]
    bool empty() const noexcept
    {
        return points_.empty();
    }

private:
    struct ConstraintPoint
    {
        int a{-1};
        int b{-1};
        float invMassA{0.0f};
        float invMassB{0.0f};
        // Position-pass weights (Kinematic 1, everything else 0): only scene-driven
        // bodies are pushed out of penetration by the pseudo-velocity position pass.
        float posWeightA{0.0f};
        float posWeightB{0.0f};
        Vec3 normal{};
        Vec3 tangent1{};
        Vec3 tangent2{};
        Vec3 point{}; // prepare-time world contact point (proximity-matched warm start)
        // Contact lever rotated into each body's frame at prepare. The current world
        // lever is orientation·anchorLocal; the current separation is recovered from
        // the moving anchors + `adjustedSeparation` (TGS analytic separation tracking).
        Vec3 anchorLocalA{};
        Vec3 anchorLocalB{};
        // Separation at prepare (= −penetration); current separation is
        //   dot((posA + rA) − (posB + rB), normal) + adjustedSeparation.
        float adjustedSeparation{0.0f};
        float normalMass{0.0f};
        float posNormalMass{0.0f}; // effective mass along the normal using posWeights
        float tangentMass1{0.0f};
        float tangentMass2{0.0f};
        float pseudoImpulse{0.0f}; // accumulated position-pass pseudo-impulse
        // Relative-normal velocity at prepare (negative = approaching); the impact
        // speed restitution rebounds against at end-of-step.
        float relVelN0{0.0f};
        float restitution{0.0f};
        float friction{0.0f};
        float normalImpulse{0.0f};
        float tangentImpulse1{0.0f};
        float tangentImpulse2{0.0f};
        // Largest normal impulse reached over the substeps; restitution is only
        // applied to points that actually engaged (maxNormalImpulse > 0).
        float maxNormalImpulse{0.0f};
        std::uint64_t key{0};
    };

    // One persisted contact point: its world position (for proximity matching next
    // frame) and the impulses accumulated this frame.
    struct CachedPoint
    {
        Vec3 point{};
        float normalImpulse{0.0f};
        float tangentImpulse1{0.0f};
        float tangentImpulse2{0.0f};
    };

    std::vector<ConstraintPoint> points_;
    // Pseudo-velocity scratch for the kinematic position pass (indexed by body).
    std::vector<Vec3> pseudoVelocity_;
    std::vector<Vec3> pseudoAngularVelocity_;
    // World-space inverse inertia per body, R·diag(invI_local)·Rᵀ, built in prepare.
    std::vector<Mat3> invInertiaWorld_;
    // Substep dt (h = dt / kSubstepCount); the soft coefficients and the speculative
    // 1/h gap bias are computed against it.
    float h_{0.0f};

    // Soft-constraint coefficients (Box2D-v3 `b2MakeSoft`), computed once per step in
    // prepare() from kContactHertz / kContactDampingRatio / h. `impulseScale_` decays
    // the accumulated impulse during the bias solve — the dissipative term.
    float biasRate_{0.0f};
    float massScale_{1.0f};
    float impulseScale_{0.0f};

    // Warm-start cache, keyed by collider pair. `cache_` holds the previous step's
    // impulses (read in prepare); `next_` accumulates this step's (written in
    // store, then swapped in). Only ever looked up by key — never iterated to
    // produce results — so map ordering cannot leak into the simulation.
    std::unordered_map<std::uint64_t, std::vector<CachedPoint>> cache_;
    std::unordered_map<std::uint64_t, std::vector<CachedPoint>> next_;
};

} // namespace fire_engine
