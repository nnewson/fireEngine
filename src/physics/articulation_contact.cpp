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

    for (int s = 0; s < kSubstepCount; ++s)
    {
        articulation.computeAccelerations(gravity, jointDamping);
        articulation.integrate(h);
        articulation.factorizeArticulatedInertia();

        // One sequential-impulse (Gauss-Seidel) contact sweep per substep. Link velocities are
        // refreshed before each contact so every impulse sees the ones already applied this
        // sweep — multiple coupled contacts solved against a single stale snapshot over-correct
        // and diverge. The substep cadence (kSubstepCount passes) converges the rest.
        {
            for (std::size_t i = 0; i < contacts.size(); ++i)
            {
                articulation.computeLinkVelocities();
                const ArticulationPlaneContact& c = contacts[i];
                ConstraintBody link = ConstraintBody::link(articulation, c.link);
                const Vec3 point = articulation.linkWorld(c.link).transformPoint(c.localPoint);
                const float separation = dot(c.normal, point) - c.offset;
                // A point well beyond the speculative band is in free flight — no impulse.
                if (separation > kSpeculativeDistance)
                {
                    normalImpulse[i] = 0.0f;
                    continue;
                }

                const float invEffN = link.inverseEffectiveMassAlong(point, c.normal, unused);
                if (invEffN <= 0.0f)
                {
                    continue;
                }
                // Soft non-penetration (b2MakeSoft), matching the rigid solver: a speculative gap
                // is allowed to close at most this substep (bias = separation·invH); a real
                // penetration is pushed out by a damped spring capped at kMaxBiasVelocity — never
                // the raw Baumgarte/h, which at the substep rate launches the body. massScale /
                // impulseScale decay the impulse so the push-out doesn't pump energy.
                float bias = 0.0f;
                float massScale = 1.0f;
                float impulseScale = 0.0f;
                if (separation > 0.0f)
                {
                    bias = separation * invH;
                }
                else
                {
                    bias = std::max(soft.biasRate * separation, -kMaxBiasVelocity);
                    massScale = soft.massScale;
                    impulseScale = soft.impulseScale;
                }
                const float vn = dot(link.velocityAt(point), c.normal);
                const float lambda =
                    -(massScale * (vn + bias)) / invEffN - impulseScale * normalImpulse[i];
                const float old = normalImpulse[i];
                normalImpulse[i] = std::max(0.0f, old + lambda);
                const float applied = normalImpulse[i] - old;
                link.applyImpulse(point, c.normal * applied, unused);

                // Coulomb friction over the current slip direction, clamped to μ·Nₙ. Only when
                // the point is actually loaded (normalImpulse > 0).
                if (normalImpulse[i] <= 0.0f)
                {
                    continue;
                }
                const Vec3 vel = link.velocityAt(point);
                const Vec3 vt = vel - c.normal * dot(vel, c.normal);
                const float vtMag = vt.magnitude();
                if (vtMag < 1e-6f)
                {
                    continue;
                }
                const Vec3 tangent = vt * (1.0f / vtMag);
                const float invEffT = link.inverseEffectiveMassAlong(point, tangent, unused);
                if (invEffT <= 0.0f)
                {
                    continue;
                }
                float lambdaT = -vtMag / invEffT;
                const float budget = c.friction * normalImpulse[i];
                lambdaT = std::clamp(lambdaT, -budget, budget);
                link.applyImpulse(point, tangent * lambdaT, unused);
            }
        }
    }
}

} // namespace fire_engine
