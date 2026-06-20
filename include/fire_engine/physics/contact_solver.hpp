#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <fire_engine/collision/contact_manifold.hpp>
#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Sequential-impulse contact solver (P2 + P3). Resolves a set of contact manifolds
// into corrected linear + angular velocities (normal + friction impulses with
// lever-arm torque) and a split-impulse positional/orientation correction for
// penetration. Effective mass per contact is
//   invMassA + invMassB + (rA×d)·IA⁻¹(rA×d) + (rB×d)·IB⁻¹(rB×d),
// the angular terms vanishing for a centred contact (reproducing the P2 linear case).
//
// The solver is deliberately decoupled from PhysicsWorld: it operates on a flat
// SolverBody array (indexed by the caller) and a list of SolverContactInput, so
// it can be unit-tested in isolation.

// One body as the solver sees it. `invMass` / `inverseInertiaLocal` drive the
// velocity (impulse) pass: both are 0 for Static and Kinematic bodies, so contact
// impulses never shove or spin a scene-driven body. `positionWeight` drives the
// split-impulse position pass: Kinematic gets a nominal weight there so it still
// slides out of penetration (mirroring the pre-P2 pushWeight), Static stays 0.
// `position` is the centre of mass (contact anchors are measured from it).
struct SolverBody
{
    Vec3 velocity{};
    Vec3 angularVelocity{};
    Vec3 position{};
    Quaternion orientation{};
    float invMass{0.0f};
    Vec3 inverseInertiaLocal{}; // diagonal, principal frame (0 ⇒ infinite inertia)
    float positionWeight{0.0f};
};

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

    // Build constraint rows from the contacts, compute effective masses and the
    // restitution bias from the current velocities. Clears any state from the
    // previous step.
    void prepare(const std::vector<SolverBody>& bodies,
                 const std::vector<SolverContactInput>& contacts, float dt);

    // Apply the (warm-started) accumulated impulses once before iterating.
    void warmStart(std::vector<SolverBody>& bodies) const;

    // One velocity-constraint Gauss-Seidel sweep (normal then friction). Call
    // kVelocityIterations times.
    void solveVelocity(std::vector<SolverBody>& bodies);

    // Split-impulse penetration correction: runs kPositionIterations internal
    // sweeps over pseudo-velocities and writes the corrected positions back.
    void solvePosition(std::vector<SolverBody>& bodies);

    // Persist accumulated impulses for next frame's warm start.
    void store();

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
        float posWeightA{0.0f};
        float posWeightB{0.0f};
        Vec3 normal{};
        Vec3 tangent1{};
        Vec3 tangent2{};
        Vec3 point{};
        Vec3 rA{}; // contact point − body A centre of mass (lever arm)
        Vec3 rB{}; // contact point − body B centre of mass
        float normalMass{0.0f};
        float posNormalMass{0.0f};
        float tangentMass1{0.0f};
        float tangentMass2{0.0f};
        // Target relative-normal velocity for the normal constraint. For a
        // penetrating/touching contact this is the restitution bounce velocity; for
        // a speculative gap (penetration < 0) it is -separation/dt, which lets the
        // body close the gap but not overshoot through the surface in one step.
        float normalBias{0.0f};
        float penetration{0.0f};
        float friction{0.0f};
        float normalImpulse{0.0f};
        float tangentImpulse1{0.0f};
        float tangentImpulse2{0.0f};
        float pseudoImpulse{0.0f};
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
    std::vector<Vec3> pseudoVelocity_;
    std::vector<Vec3> pseudoAngularVelocity_;
    // World-space inverse inertia per body, R·diag(invI_local)·Rᵀ, built in prepare.
    std::vector<Mat3> invInertiaWorld_;
    float dt_{0.0f};

    // Warm-start cache, keyed by collider pair. `cache_` holds the previous step's
    // impulses (read in prepare); `next_` accumulates this step's (written in
    // store, then swapped in). Only ever looked up by key — never iterated to
    // produce results — so map ordering cannot leak into the simulation.
    std::unordered_map<std::uint64_t, std::vector<CachedPoint>> cache_;
    std::unordered_map<std::uint64_t, std::vector<CachedPoint>> next_;
};

} // namespace fire_engine
