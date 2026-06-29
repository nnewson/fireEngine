#pragma once

namespace fire_engine
{

// Tunables for the linear sequential-impulse contact solver (P2). Kept in one
// place so the solver math is not littered with magic numbers; the values follow
// the well-trodden Box2D/Catto defaults scaled for this engine's metre units.

// TGS soft-step solver (P9.2). The fixed step is split into kSubstepCount equal
// sub-steps of h = dt / kSubstepCount; collision detection runs ONCE per fixed step
// (outside the loop), but the solve, position integration, and a no-bias *relax* pass
// run per substep. The relax pass removes the soft-constraint bias velocity each
// substep (the dissipation that whole-step substepping lacked) so the correction can
// not pump energy. A higher substep count both lets stiffer contacts stay stable (the
// stability cap is ~0.25/h) and gives the joint+contact coupling of an articulated
// ragdoll the temporal resolution it needs to settle — 8 was the threshold at which the
// 17-bone humanoid gate comes fully to rest. See roadmap P9 (C).
inline constexpr int kSubstepCount = 8;

// Velocity-constraint (impulse) Gauss-Seidel sweeps per substep, for each of the bias
// solve and the relax solve. TGS gets most of its iteration from the substeps themselves,
// but the extra sweeps balance multi-point manifold torque (so box stacks don't micro-rock)
// and resolve resting friction/penetration on multi-triangle mesh contacts — 4 is the
// lowest that keeps every contact/mesh resting test green alongside the ragdoll gate.
inline constexpr int kVelocityIterations = 4;

// Soft contact constraint (Box2D-v3 `b2MakeSoft`), the contact analogue of the soft
// joint (P9.1): a damped spring at frequency kContactHertz / damping kContactDampingRatio
// whose impulse-decay term dissipates energy, replacing the old split-impulse Baumgarte
// position pass. Stiff (near-rigid) contacts matter for resting articulations: an overly
// compliant floor contact lets a ragdoll's limbs bob and pumps that motion back through
// the joints. 90 Hz stays under the ~0.25/h stability cap at kSubstepCount = 8
// (h = dt/8 ⇒ cap ≈ 120 Hz) and settles the humanoid gate.
inline constexpr float kContactHertz = 90.0f;
inline constexpr float kContactDampingRatio = 10.0f;

// Cap (m/s) on the soft-constraint push-out (bias) speed, so a deep initial
// penetration can't launch a body. Only bites on large overlaps.
inline constexpr float kMaxBiasVelocity = 4.0f;

// Kinematic-only split-impulse position pass (P9.2): Dynamic bodies resolve penetration
// through the soft bias, but scene-driven Kinematic bodies (which carry no inverse mass)
// are pushed out of penetration by a pseudo-velocity pass. Iterations + Baumgarte fraction
// as in the classic split-impulse correction.
inline constexpr int kPositionIterations = 3;
inline constexpr float kBaumgarte = 0.2f;

// Penetration the contact bias leaves uncorrected, so resting contacts do not
// jitter around exact touching (metres).
inline constexpr float kLinearSlop = 0.005f;

// Below this closing speed (m/s) restitution is suppressed, so bodies settling
// under gravity come to rest instead of buzzing with micro-bounces.
inline constexpr float kRestitutionThreshold = 1.0f;

// Radius (metres) within which a new contact point inherits the previous frame's
// accumulated impulse for warm starting (proximity match).
inline constexpr float kWarmStartMatchRadius = 0.02f;

// Base speculative-contact margin (metres) for CCD: shapes separated by up to this
// distance still generate a (negative-penetration) contact, on top of a
// motion-dependent term (relative speed × dt). ~4× the linear slop catches slow
// near-touching pairs; the motion term scales it so fast movers can't tunnel.
inline constexpr float kSpeculativeDistance = 0.02f;

// Joints (P4) reuse the same sequential-impulse machinery as contacts, but use a
// **soft / compliant** constraint (Erin Catto / Box2D-v3 `b2MakeSoft`) instead of a hard
// Baumgarte velocity bias (P9.1). A soft constraint is a damped spring: it has finite
// stiffness (a target frequency `kJointHertz`) and a damping ratio (`kJointDampingRatio`),
// and its impulse-decay term *dissipates* energy rather than re-injecting unresolved
// position error as velocity each step — which a hard Baumgarte bias does, pumping energy
// into many-joint graphs (ragdolls) so they never settle. See roadmap P9 (B2)/(C).
//
// Frequency is a fraction of the step rate (stiff but stable at a single step); damping
// ratio > 1 is overdamped (no overshoot). Tune so joints hold tightly *and* ragdolls
// settle (gated by the [Joint] tests + the ragdoll-settles test).
inline constexpr float kJointHertz = 8.0f;
inline constexpr float kJointDampingRatio = 5.0f;

// Anchor/axis error (metres / radians) a joint leaves uncorrected, so a satisfied
// joint contributes no bias and never buzzes.
inline constexpr float kJointSlop = 0.0005f;

// Sleeping (P5): a Dynamic body whose linear AND angular speed stay below these
// thresholds for kSleepTime becomes eligible to sleep; a whole island sleeps once
// every dynamic member is eligible, stops integrating + solving, and has its
// velocities zeroed until disturbed. Squared magnitudes are compared, so these are
// the linear (m/s) and angular (rad/s) speeds.
inline constexpr float kLinearSleepThreshold = 0.05f;
inline constexpr float kAngularSleepThreshold = 0.05f;
inline constexpr float kSleepTime = 0.5f;

// Static-mesh contacts (P6): a triangle contact point deeper than this below the
// triangle plane is treated as a degenerate EPA result (a garbage witness point on a
// flat triangle) and dropped, so it can't inject a huge correction. Generous — the
// speculative margin brakes fast movers before they penetrate this far.
inline constexpr float kMaxMeshPenetration = 1.0f;

} // namespace fire_engine
