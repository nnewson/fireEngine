#pragma once

namespace fire_engine
{

// Tunables for the linear sequential-impulse contact solver (P2). Kept in one
// place so the solver math is not littered with magic numbers; the values follow
// the well-trodden Box2D/Catto defaults scaled for this engine's metre units.

// Velocity-constraint (impulse) iterations per fixed step.
inline constexpr int kVelocityIterations = 8;

// Position-correction (split-impulse) iterations per fixed step.
inline constexpr int kPositionIterations = 3;

// Penetration the position pass leaves uncorrected, so resting contacts do not
// jitter around exact touching (metres).
inline constexpr float kLinearSlop = 0.005f;

// Fraction of the remaining penetration the split-impulse position pass removes
// per step (Baumgarte factor). < 1 keeps correction stable / non-explosive.
inline constexpr float kBaumgarte = 0.2f;

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
