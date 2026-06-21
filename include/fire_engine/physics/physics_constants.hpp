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

// Joints (P4) reuse the same sequential-impulse machinery as contacts. Position
// error is fed back through a Baumgarte velocity bias rather than a separate
// split-impulse pass: bias = -(kJointBaumgarte/dt)·C, removing that fraction of the
// anchor/axis error per step. Stiffer than contacts (joints should hold tightly)
// but < 1 to stay stable.
inline constexpr float kJointBaumgarte = 0.2f;

// Anchor/axis error (metres / radians) a joint leaves uncorrected, so a satisfied
// joint contributes no bias and never buzzes.
inline constexpr float kJointSlop = 0.0005f;

} // namespace fire_engine
