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

// Coulomb friction (P9.3 model). Friction is solved as a coupled 2-vector over the contact
// tangent plane against a symmetric 2x2 effective mass (which captures the two tangents'
// angular cross-coupling) and clamped to the friction *disk* |λ| ≤ μ·N (a circle, not an
// independent-axis box). That coupling + circular clamp is what keeps a tipping/edge contact
// from pumping spurious torque; an earlier per-axis box clamp + scalar masses over-budgeted
// diagonally and mis-distributed torque. (A positional static-friction *anchor* scheme was
// tried and removed: per-point anchors on a 2-point edge contact formed a torsional couple
// that pumped energy and flipped settled convex bodies; the 2x2-mass + disk clamp alone is
// both simpler and strictly better — see roadmap P9.3.)

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

// Mid-step manifold refresh (P9.6 stage 1). Contact manifolds are detected once per fixed
// step at the step-start pose; a body sweeping more than this rotation in one step (|ω|·dt,
// radians) has its non-mesh manifolds re-collided once at substep kSubstepCount/2 and the
// solver rows re-prepared at the current mid-step pose — a stale manifold under fast rotation
// concentrates the impact impulses on 1–2 wrongly-placed points, producing intra-step yaw
// bursts (a settling tetra visibly "snaps") and rotational tunnelling. 0.03 rad triggers at
// ≥ ~1.8 rad/s; the motivating ConvexHullDemo tetra face-slam tumbles at ~2.9 rad/s
// (0.048 rad/step). Slow bodies never trip the gate and pay nothing.
inline constexpr float kSubstepRefreshRotation = 0.03f;

// Sequential-impulse velocity iterations per substep for a reduced-coordinate articulation's
// contact + joint-limit solve. Unlike the rigid solver (whose bodies each carry their own
// velocity), an articulation's contacts are strongly coupled through the shared kinematic
// chain, so a single Gauss-Seidel sweep under-converges a stiff many-contact collapse. The
// unified velocity model (applyImpulse folds its per-link velocity change into the cached link
// velocities) lets these sweeps iterate to convergence within one substep.
inline constexpr int kArticulationVelocityIterations = 4;

// Uniform passive joint damping for the reduced-coordinate articulation solve. Modest but
// non-zero: a chaotic chain needs dissipation to settle under the explicit integrator, while
// per-joint drives and limits add pose-specific behaviour. This is a physics tunable kept here
// until it graduates into RagdollParams.
inline constexpr float kArticulationDamping = 0.2f;

// Settle assist for a reduced-coordinate ragdoll's floating base: the joints damp via an explicit
// −c·q̇ torque, but the free 6-DOF root has none, so a settled ragdoll keeps its residual
// horizontal drift and slides. A *linear* base-velocity decay fixes it (angular damping
// destabilises a violent impact and is not applied). The rate ramps up once the base is slow
// (below kBaseSettleSpeed) so a resting ragdoll stops quickly, while a fast (falling) base is left
// almost untouched — no floaty fall.
inline constexpr float kBaseSettleSpeed = 0.3f;   // m/s — below this the base is "settling"
inline constexpr float kBaseSettleDamping = 8.0f; // 1/s — strong linear decay when settling

// The same settle assist for the joint velocities (Articulation::integrateVelocities). A landed
// ragdoll's limbs otherwise crawl toward the articulation sleep threshold over seconds under the
// modest passive kArticulationDamping, which reads as an arm curling unnaturally after the body has
// come to rest. Any DOF already moving slowly (below kJointSettleSpeed) is decayed strongly so it
// crosses into sleep quickly; fast, dramatic collapse motion (well above it) is left untouched.
inline constexpr float kJointSettleSpeed = 0.5f;   // rad/s — below this a joint is "settling"
inline constexpr float kJointSettleDamping = 8.0f; // 1/s — strong decay of residual joint velocity

// Once the base has *landed* (its linear speed has fallen below kBaseSettleSpeed), the joint settle
// gate widens to this. The residual hip/limb wiggle after a collapse rings in a band (~1–4 rad/s)
// that is too fast for kJointSettleSpeed yet well below the dramatic collapse (6–13 rad/s); gating
// the wider band on "landed" bleeds the wiggle while the airborne collapse — whose joints are
// faster than this — is left untouched (the base can momentarily read slow mid-collapse, so the
// speed gate still guards the fast joints even then).
inline constexpr float kJointSettleSpeedLanded = 4.0f; // rad/s

// Settle decay for the base's spin about the vertical (yaw) only, once the base is settling.
// Contact friction doesn't resist rotation about the vertical, so a rested ragdoll keeps a slow
// residual yaw — the last visible motion once the limbs stop. Damping the *full* base angular
// destabilises a near-planar chain, so only the world-up component is decayed
// (Articulation::integrateVelocities).
inline constexpr float kBaseYawSettleDamping = 8.0f; // 1/s

// Settle decay for the base's roll/pitch (spin about a *horizontal* axis) once the base is
// settling. A ragdoll that has collapsed onto its side/back rocks left-right about a horizontal
// axis — the base linear + yaw dampers don't touch it, so it's the last mode to ring out (only
// contact friction + passive joint damping bleed it, slowly). Damping the full base angular during
// a *violent* impact destabilises a near-planar chain (why it was left out), so this is
// double-gated: applied only while the base is settling (linear-speed gate below kBaseSettleSpeed)
// AND only to a residual roll/pitch already slower than kBaseRockSettleSpeed. A fast, dramatic
// collapse is left untouched.
inline constexpr float kBaseRockSettleSpeed = 1.2f; // rad/s — below this the roll/pitch is residual
inline constexpr float kBaseRockSettleDamping = 6.0f; // 1/s — decay of that residual roll/pitch

// Velocity-level cone-twist joint-limit push-out (solveJointLimits): close a fraction
// kJointLimitErp of the angular over-limit per step, capped at kJointLimitMaxPush rad/s so a
// deep violation can't fling the joint back. A projection, not a spring — stable at any inertia.
inline constexpr float kJointLimitErp = 0.2f;
inline constexpr float kJointLimitMaxPush = 3.0f;

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
// Articulations are jointed chains, so their angular and qDot sleep threshold mirrors the looser
// jointed-island rigid-body threshold instead of the singleton rigid-body angular threshold.
inline constexpr float kArticulationAngularSleepThreshold = 0.15f;
inline constexpr float kSleepTime = 0.5f;

// Near-rest snap for a reduced-coordinate articulation. A lone straggler DOF (e.g. a shoulder
// creeping in under gravity after the body has landed) can hover right at
// kArticulationAngularSleepThreshold, ticking above it often enough to keep resetting the sleep
// dwell — so the whole chain stays awake for ~a second while one limb crawls to rest. Once
// EVERYTHING (base linear via kLinearSleepThreshold, base angular, and every joint rate) sits
// within this slightly wider band for kArticulationRestSnapTime, the residual is zeroed so the body
// sleeps as a unit instead of trailing a limb. Wider than the sleep threshold (to catch the
// straggler) with a shorter dwell (it is a "basically done" signal); the dwell keeps it from firing
// mid-motion, and by fire time the residual has decayed well under the band so the zero is not a
// visible pop.
inline constexpr float kArticulationRestSnapThreshold = 0.30f; // rad/s
inline constexpr float kArticulationRestSnapTime = 0.25f;      // s held below the band before snap

// Static-mesh contacts (P6): a triangle contact point deeper than this below the
// triangle plane is treated as a degenerate EPA result (a garbage witness point on a
// flat triangle) and dropped, so it can't inject a huge correction. Generous — the
// speculative margin brakes fast movers before they penetrate this far.
inline constexpr float kMaxMeshPenetration = 1.0f;

} // namespace fire_engine
