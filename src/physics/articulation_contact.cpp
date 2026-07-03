#include <fire_engine/physics/articulation_contact.hpp>

#include <algorithm>
#include <numbers>
#include <vector>

#include <fire_engine/physics/articulation.hpp>
#include <fire_engine/physics/constraint_body.hpp>
#include <fire_engine/physics/physics_constants.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] float dot(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::dotProduct(a, b);
}

// Box2D-v3 soft-contact coefficients (b2MakeSoft) at substep h — the same damped-spring
// non-penetration the rigid ContactSolver uses. Returned as (biasRate, massScale, impulseScale).
struct SoftCoeffs
{
    float biasRate{0.0f};
    float massScale{1.0f};
    float impulseScale{0.0f};
};

[[nodiscard]] SoftCoeffs softContact(float h) noexcept
{
    const float omega = 2.0f * std::numbers::pi_v<float> * kContactHertz;
    const float a1 = 2.0f * kContactDampingRatio + h * omega;
    const float a2 = h * omega * a1;
    const float a3 = 1.0f / (1.0f + a2);
    return SoftCoeffs{omega / a1, a2 * a3, a3};
}

} // namespace

void stepArticulationOnPlanes(Articulation& articulation,
                              std::span<const ArticulationPlaneContact> contacts,
                              const Vec3& gravity, float dt, float jointDamping)
{
    const float h = dt / static_cast<float>(kSubstepCount);
    const float invH = h > 0.0f ? 1.0f / h : 0.0f;
    const SoftCoeffs soft = softContact(h);
    // Normal impulses persist across substeps (warm start within the step); friction is
    // re-derived from zero each substep and clamped to the friction cone (no cross-frame
    // memory), matching the rigid solver's anti-pump choice.
    std::vector<float> normalImpulse(contacts.size(), 0.0f);
    const Mat3 unused{}; // ConstraintBody link path ignores the inertia argument

    // Per-contact geometry precomputed once per substep — positions are fixed after integrate(),
    // so only velocities change across the iteration. `active` gates points beyond the
    // speculative band (free flight). invEffN is the normal inverse effective mass through the
    // articulated response.
    struct Prepared
    {
        Vec3 point{};
        float invEffN{0.0f};
        float bias{0.0f};
        float massScale{1.0f};
        float impulseScale{0.0f};
        bool active{false};
    };
    std::vector<Prepared> prep(contacts.size());

    // One Gauss-Seidel sweep over the contacts. `useBias` on the first sweep applies the soft
    // push-out; the relax sweeps (bias 0, no impulse decay) only converge the coupled velocities
    // to non-penetration without re-pumping the push-out. Reads/writes the cached link velocities
    // (unified model), so every contact sees the impulses already applied this sweep.
    const auto sweep = [&](bool useBias)
    {
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            const Prepared& p = prep[i];
            if (!p.active)
            {
                continue;
            }
            const ArticulationPlaneContact& c = contacts[i];
            ConstraintBody link = ConstraintBody::link(articulation, c.link);

            const float bias = useBias ? p.bias : 0.0f;
            const float massScale = useBias ? p.massScale : 1.0f;
            const float impulseScale = useBias ? p.impulseScale : 0.0f;
            const float vn = dot(link.velocityAt(p.point), c.normal);
            const float lambda =
                -(massScale * (vn + bias)) / p.invEffN - impulseScale * normalImpulse[i];
            const float old = normalImpulse[i];
            normalImpulse[i] = std::max(0.0f, old + lambda);
            link.applyImpulse(p.point, c.normal * (normalImpulse[i] - old), unused);

            // Coulomb friction over the current slip direction, clamped to μ·Nₙ.
            if (normalImpulse[i] <= 0.0f)
            {
                continue;
            }
            const Vec3 vel = link.velocityAt(p.point);
            const Vec3 vt = vel - c.normal * dot(vel, c.normal);
            const float vtMag = vt.magnitude();
            if (vtMag < 1e-6f)
            {
                continue;
            }
            const Vec3 tangent = vt * (1.0f / vtMag);
            const float invEffT = link.inverseEffectiveMassAlong(p.point, tangent, unused);
            if (invEffT <= 0.0f)
            {
                continue;
            }
            const float budget = c.friction * normalImpulse[i];
            const float lambdaT = std::clamp(-vtMag / invEffT, -budget, budget);
            link.applyImpulse(p.point, tangent * lambdaT, unused);
        }
        // Cone-twist joint limits share the sweep so limits and contacts converge together
        // through the same articulated response (both velocity-level, unified velocity model).
        articulation.solveJointLimits(invH, useBias, kJointLimitErp, kJointLimitMaxPush);
    };

    for (int s = 0; s < kSubstepCount; ++s)
    {
        // TGS substep: advance velocities, solve contacts with the soft bias, integrate the bias
        // velocity into position (the push-out), then relax it away so it doesn't pump — the same
        // shape as the rigid ContactSolver. The factorization + cached link poses (linkWorld_)
        // are taken at the substep-start pose; integratePositions advances the *state* (base/q)
        // but not linkWorld_, so the relax sweeps stay consistent with that factorization.
        articulation.computeAccelerations(gravity, jointDamping); // FK + factorize @ start pose
        articulation.integrateVelocities(h);
        articulation.computeLinkVelocities();

        // Prepare each contact against the substep-start pose (positions fixed for this solve).
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            const ArticulationPlaneContact& c = contacts[i];
            ConstraintBody link = ConstraintBody::link(articulation, c.link);
            const Vec3 point = articulation.linkWorld(c.link).transformPoint(c.localPoint);
            const float separation = dot(c.normal, point) - c.offset;
            const float invEffN = link.inverseEffectiveMassAlong(point, c.normal, unused);
            Prepared& p = prep[i];
            p.point = point;
            p.invEffN = invEffN;
            p.active = separation <= kSpeculativeDistance && invEffN > 0.0f;
            if (!p.active)
            {
                normalImpulse[i] = 0.0f;
                continue;
            }
            // Soft non-penetration (b2MakeSoft): a speculative gap closes at most this substep
            // (bias = separation·invH); a real penetration is pushed out by a damped spring capped
            // at kMaxBiasVelocity. massScale/impulseScale decay the impulse so it doesn't pump.
            p.bias = 0.0f;
            p.massScale = 1.0f;
            p.impulseScale = 0.0f;
            if (separation > 0.0f)
            {
                p.bias = separation * invH;
            }
            else
            {
                p.bias = std::max(soft.biasRate * separation, -kMaxBiasVelocity);
                p.massScale = soft.massScale;
                p.impulseScale = soft.impulseScale;
            }
        }

        sweep(true);                        // biased: establish the soft push-out velocity
        articulation.integratePositions(h); // push-out moves into position here
        for (int r = 1; r < kArticulationVelocityIterations; ++r)
        {
            sweep(false); // relax: converge coupling, remove bias velocity
        }
    }
}

} // namespace fire_engine
