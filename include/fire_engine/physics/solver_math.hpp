#pragma once

#include <fire_engine/math/mat3.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// One body as the constraint solvers (contact + joint) see it. `invMass` /
// `inverseInertiaLocal` drive the velocity (impulse) pass: both are 0 for Static
// and Kinematic bodies, so impulses never shove or spin a scene-driven body.
// `positionWeight` drives the split-impulse position pass: Kinematic gets a nominal
// weight there so it still slides out of penetration, Static stays 0. `position`
// is the centre of mass (contact anchors / joint anchors are measured from it).
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

// World-space inverse inertia tensor: R · diag(invI_local) · Rᵀ (zero matrix for a
// body with infinite inertia, so it picks up no angular response).
[[nodiscard]] inline Mat3 worldInverseInertia(const Quaternion& orientation,
                                              const Vec3& inverseInertiaLocal) noexcept
{
    const Mat3 r = Mat3::fromQuaternion(orientation);
    return r * Mat3::diagonal(inverseInertiaLocal) * r.transpose();
}

// Relative velocity at the two anchor points, including each body's angular term ω×r.
[[nodiscard]] inline Vec3 relativeVelocity(const SolverBody& a, const SolverBody& b, const Vec3& rA,
                                           const Vec3& rB) noexcept
{
    return (a.velocity + Vec3::crossProduct(a.angularVelocity, rA)) -
           (b.velocity + Vec3::crossProduct(b.angularVelocity, rB));
}

// Effective mass for a constraint along unit `dir` with lever arms rA/rB and world
// inverse inertias: k = invMassA + invMassB + (rA×d)·IA⁻¹(rA×d) + (rB×d)·IB⁻¹(rB×d).
// The angular terms vanish for a centred (rA=rB=0) constraint, giving the linear case.
[[nodiscard]] inline float effectiveMassAlong(float invMassA, float invMassB, const Vec3& rA,
                                              const Vec3& rB, const Vec3& dir, const Mat3& iA,
                                              const Mat3& iB) noexcept
{
    const Vec3 raxd = Vec3::crossProduct(rA, dir);
    const Vec3 rbxd = Vec3::crossProduct(rB, dir);
    const float k =
        invMassA + invMassB + Vec3::dotProduct(raxd, iA * raxd) + Vec3::dotProduct(rbxd, iB * rbxd);
    return k > 0.0f ? 1.0f / k : 0.0f;
}

// Apply impulse `P` at the anchors to both bodies — linear (v += invMass·P) and
// angular (ω += I⁻¹(r×P)); body B takes the equal-and-opposite impulse.
inline void applyImpulse(SolverBody& a, SolverBody& b, float invMassA, float invMassB,
                         const Mat3& iA, const Mat3& iB, const Vec3& rA, const Vec3& rB,
                         const Vec3& impulse) noexcept
{
    a.velocity += impulse * invMassA;
    a.angularVelocity += iA * Vec3::crossProduct(rA, impulse);
    b.velocity -= impulse * invMassB;
    b.angularVelocity -= iB * Vec3::crossProduct(rB, impulse);
}

} // namespace fire_engine
