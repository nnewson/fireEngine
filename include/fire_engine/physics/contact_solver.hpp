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

// Sequential-impulse contact solver (P2 + P3). Resolves a set of contact manifolds
// into corrected linear + angular velocities (normal + friction impulses with
// lever-arm torque) and a split-impulse positional/orientation correction for
// penetration. Effective mass per contact is
//   invMassA + invMassB + (rA×d)·IA⁻¹(rA×d) + (rB×d)·IB⁻¹(rB×d),
// the angular terms vanishing for a centred contact (reproducing the P2 linear case).
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

    // Build constraint rows from the contacts, compute effective masses and the
    // restitution bias from the current velocities. Clears any state from the
    // previous step.
    void prepare(std::span<const SolverBody> bodies, std::span<const SolverContactInput> contacts,
                 float dt);

    // Apply the (warm-started) accumulated impulses once before iterating.
    void warmStart(std::vector<SolverBody>& bodies) const;

    // One velocity-constraint Gauss-Seidel sweep (normal then friction). Call
    // kVelocityIterations times.
    void solveVelocity(std::vector<SolverBody>& bodies);

    // Split-impulse penetration correction: runs kPositionIterations internal
    // sweeps over pseudo-velocities and writes the corrected positions back.
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
