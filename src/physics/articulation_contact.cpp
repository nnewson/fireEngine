#include <fire_engine/physics/articulation_contact.hpp>

#include <algorithm>
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

} // namespace

void stepArticulationOnPlanes(Articulation& articulation,
                              std::span<const ArticulationPlaneContact> contacts,
                              const Vec3& gravity, float dt, float jointDamping)
{
    const float h = dt / static_cast<float>(kSubstepCount);
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
        articulation.computeLinkVelocities();

        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            const ArticulationPlaneContact& c = contacts[i];
            ConstraintBody link = ConstraintBody::link(articulation, c.link);
            const Vec3 point = articulation.linkWorld(c.link).transformPoint(c.localPoint);
            const float separation = dot(c.normal, point) - c.offset;
            // Only engage within the speculative band: a point well above the plane is in
            // free flight and must not feel a braking impulse.
            if (separation > kSpeculativeDistance)
            {
                normalImpulse[i] = 0.0f;
                continue;
            }

            // Normal: drive the point to a small push-out speed when penetrating, never
            // pulling it back (accumulated impulse clamped to ≥ 0).
            const float invEffN = link.inverseEffectiveMassAlong(point, c.normal, unused);
            if (invEffN <= 0.0f)
            {
                continue;
            }
            const float biasVel = separation < 0.0f ? -separation * kBaumgarte / h : 0.0f;
            const float vn = dot(link.velocityAt(point), c.normal);
            const float lambda = (biasVel - vn) / invEffN;
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

} // namespace fire_engine
