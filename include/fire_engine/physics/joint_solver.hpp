#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/joint.hpp>
#include <fire_engine/physics/solver_math.hpp>

namespace fire_engine
{

// Sequential-impulse joint solver (P4). Mirrors ContactSolver but operates on
// generic bilateral/limit constraint rows rather than contact points: each joint
// expands into one or more ConstraintRows (a full Jacobian linearA/angularA/
// linearB/angularB), solved Gauss-Seidel alongside the contacts over the same
// SolverBody array. Position error is folded into a Baumgarte velocity bias
// (kJointBaumgarte) instead of a separate split-impulse pass — joints are stiff and
// the bias is recomputed each step from the live anchor/axis error.
//
// Like ContactSolver it is decoupled from PhysicsWorld (operates on JointInput,
// world-space anchors composed by the caller) so it can be unit-tested in isolation,
// and warm-starts across frames (impulses cached per joint key, lookup-only).
class JointSolver
{
public:
    JointSolver() = default;
    ~JointSolver() = default;

    JointSolver(const JointSolver&) = default;
    JointSolver& operator=(const JointSolver&) = default;
    JointSolver(JointSolver&&) noexcept = default;
    JointSolver& operator=(JointSolver&&) noexcept = default;

    // Build constraint rows from the joints, compute effective masses + the
    // Baumgarte position bias from the current anchor/axis error. Clears prior state.
    void prepare(std::span<const SolverBody> bodies, std::span<const JointInput> joints, float dt);

    // Apply the (warm-started) accumulated impulses once before iterating.
    void warmStart(std::vector<SolverBody>& bodies) const;

    // One velocity-constraint Gauss-Seidel sweep. Call kVelocityIterations times.
    void solveVelocity(std::vector<SolverBody>& bodies);

    // Warm-start persistence split for the per-island solve (see ContactSolver):
    // `beginStore` once before the islands, `store` per island (appends), `commitStore`
    // once after. A single global solve is begin → store → commit.
    void beginStore() noexcept;
    void store();
    void commitStore() noexcept;

    [[nodiscard]]
    bool empty() const noexcept
    {
        return rows_.empty();
    }

private:
    // One generic constraint row. The Jacobian rows act on (vA, ωA, vB, ωB); the
    // impulse is accumulated and clamped to [lower, upper] ([-∞,∞] bilateral, a
    // one-sided range for a limit). `key`/`slot` identify the row for warm starting.
    struct ConstraintRow
    {
        int a{-1};
        int b{-1};
        float invMassA{0.0f};
        float invMassB{0.0f};
        Vec3 linearA{};
        Vec3 angularA{};
        Vec3 linearB{};
        Vec3 angularB{};
        float effectiveMass{0.0f};
        float bias{0.0f};
        float lower{0.0f};
        float upper{0.0f};
        float impulse{0.0f};
        std::uint64_t key{0};
        int slot{0};
    };

    // Append one constraint row: full Jacobian (linA/angA act on body A, linB/angB
    // on body B), with effective mass + Baumgarte bias derived from `positionError`.
    void pushRow(int a, int b, float invMassA, float invMassB, const Mat3& iA, const Mat3& iB,
                 const Vec3& linearA, const Vec3& angularA, const Vec3& linearB,
                 const Vec3& angularB, float positionError, float lower, float upper,
                 std::uint64_t key, int slot);

    // Three rows holding the two world anchors coincident (ball-socket); shared by
    // the BallSocket and Hinge joints.
    void addPointRows(const SolverBody& a, const SolverBody& b, const JointInput& joint,
                      const Mat3& iA, const Mat3& iB, const Vec3& rA, const Vec3& rB, int& slot);
    // Two angular rows aligning the two world hinge axes (Hinge).
    void addAxisRows(const SolverBody& a, const SolverBody& b, const JointInput& joint,
                     const Mat3& iA, const Mat3& iB, int& slot);

    // One unilateral angular row clamping rotation about `worldAxis` when `angle`
    // leaves [lower, upper]; nothing while inside the range. Shared by the hinge
    // angle limit and the cone-twist's twist limit.
    void addAngleLimitRow(const SolverBody& a, const SolverBody& b, const JointInput& joint,
                          const Mat3& iA, const Mat3& iB, const Vec3& worldAxis, float angle,
                          float lower, float upper, int slot);

    // Hinge [lowerAngle, upperAngle] limit about the hinge axis.
    void addHingeLimit(const SolverBody& a, const SolverBody& b, const JointInput& joint,
                       const Mat3& iA, const Mat3& iB, int& slot);

    // Ball-socket cone-twist: swing-cone + twist-angle unilateral rows.
    void addConeTwistLimit(const SolverBody& a, const SolverBody& b, const JointInput& joint,
                           const Mat3& iA, const Mat3& iB, int& slot);

    std::vector<ConstraintRow> rows_;
    // World-space inverse inertia per body, R·diag(invI_local)·Rᵀ, built in prepare.
    std::vector<Mat3> invInertiaWorld_;
    float dt_{0.0f};

    // Warm-start cache, keyed by joint id → accumulated impulse per row slot. Only
    // ever looked up by key, never iterated to produce results, so map ordering can
    // not leak into the simulation.
    std::unordered_map<std::uint64_t, std::vector<float>> cache_;
    std::unordered_map<std::uint64_t, std::vector<float>> next_;
};

} // namespace fire_engine
