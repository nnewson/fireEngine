#include <fire_engine/physics/contact_solver.hpp>

#include <algorithm>
#include <cmath>

#include <fire_engine/physics/physics_constants.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] float dot(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::dotProduct(a, b);
}

[[nodiscard]] Vec3 cross(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::crossProduct(a, b);
}

// Effective mass for a constraint along unit `dir` with lever arms rA/rB and world
// inverse inertias: k = invMassA + invMassB + (rA×d)·IA⁻¹(rA×d) + (rB×d)·IB⁻¹(rB×d).
// The angular terms vanish for a centred contact, reproducing the linear case.
[[nodiscard]] float angularEffectiveMass(float invMassA, float invMassB, const Vec3& rA,
                                         const Vec3& rB, const Vec3& dir, const Mat3& iA,
                                         const Mat3& iB) noexcept
{
    const Vec3 raxd = cross(rA, dir);
    const Vec3 rbxd = cross(rB, dir);
    const float k = invMassA + invMassB + dot(raxd, iA * raxd) + dot(rbxd, iB * rbxd);
    return k > 0.0f ? 1.0f / k : 0.0f;
}

// Relative velocity at the contact point, including each body's angular term ω×r.
[[nodiscard]] Vec3 relativeVelocity(const SolverBody& a, const SolverBody& b, const Vec3& rA,
                                    const Vec3& rB) noexcept
{
    return (a.velocity + cross(a.angularVelocity, rA)) -
           (b.velocity + cross(b.angularVelocity, rB));
}

// Orthonormal tangent basis spanning the contact plane of `n` (assumed unit).
// Picks the more stable seed axis to avoid a near-zero cross product.
void buildTangents(const Vec3& n, Vec3& t1, Vec3& t2) noexcept
{
    if (std::abs(n.x()) >= 0.57735f)
    {
        t1 = Vec3::normalise(Vec3{n.y(), -n.x(), 0.0f});
    }
    else
    {
        t1 = Vec3::normalise(Vec3{0.0f, n.z(), -n.y()});
    }
    t2 = Vec3::crossProduct(n, t1);
}

} // namespace

void ContactSolver::prepare(const std::vector<SolverBody>& bodies,
                            const std::vector<SolverContactInput>& contacts, float dt)
{
    dt_ = dt;
    points_.clear();
    pseudoVelocity_.assign(bodies.size(), Vec3{});
    pseudoAngularVelocity_.assign(bodies.size(), Vec3{});

    // World inverse inertia per body: R · diag(invI_local) · Rᵀ (zero matrix for
    // bodies with infinite inertia, so they pick up no angular response).
    invInertiaWorld_.assign(bodies.size(), Mat3{});
    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        const Mat3 r = Mat3::fromQuaternion(bodies[i].orientation);
        invInertiaWorld_[i] = r * Mat3::diagonal(bodies[i].inverseInertiaLocal) * r.transpose();
    }

    for (const SolverContactInput& contact : contacts)
    {
        const SolverBody& bodyA = bodies[static_cast<std::size_t>(contact.bodyA)];
        const SolverBody& bodyB = bodies[static_cast<std::size_t>(contact.bodyB)];
        const Mat3& iA = invInertiaWorld_[static_cast<std::size_t>(contact.bodyA)];
        const Mat3& iB = invInertiaWorld_[static_cast<std::size_t>(contact.bodyB)];

        Vec3 tangent1;
        Vec3 tangent2;
        buildTangents(contact.normal, tangent1, tangent2);

        for (int i = 0; i < contact.pointCount; ++i)
        {
            ConstraintPoint cp;
            cp.a = contact.bodyA;
            cp.b = contact.bodyB;
            cp.invMassA = bodyA.invMass;
            cp.invMassB = bodyB.invMass;
            cp.posWeightA = bodyA.positionWeight;
            cp.posWeightB = bodyB.positionWeight;
            cp.normal = contact.normal;
            cp.tangent1 = tangent1;
            cp.tangent2 = tangent2;
            cp.point = contact.points[static_cast<std::size_t>(i)];
            cp.rA = cp.point - bodyA.position;
            cp.rB = cp.point - bodyB.position;
            cp.penetration = contact.penetration[static_cast<std::size_t>(i)];
            cp.friction = contact.friction;

            cp.normalMass =
                angularEffectiveMass(cp.invMassA, cp.invMassB, cp.rA, cp.rB, cp.normal, iA, iB);
            cp.tangentMass1 =
                angularEffectiveMass(cp.invMassA, cp.invMassB, cp.rA, cp.rB, tangent1, iA, iB);
            cp.tangentMass2 =
                angularEffectiveMass(cp.invMassA, cp.invMassB, cp.rA, cp.rB, tangent2, iA, iB);
            cp.posNormalMass =
                angularEffectiveMass(cp.posWeightA, cp.posWeightB, cp.rA, cp.rB, cp.normal, iA, iB);
            cp.key = contact.key;

            if (cp.penetration < 0.0f)
            {
                // Speculative gap contact: target relative-normal velocity of
                // -separation/dt (= penetration/dt, penetration < 0). The body may
                // close the gap this step but the normal-impulse clamp stops it
                // overshooting through the surface — anti-tunnelling, no restitution.
                cp.normalBias = cp.penetration / dt;
            }
            else
            {
                // Restitution as a target separating speed, suppressed below the
                // resting threshold so settling stacks do not buzz. vRelNormal < 0
                // means the bodies are approaching (angular term included).
                const float vRelNormal =
                    dot(relativeVelocity(bodyA, bodyB, cp.rA, cp.rB), cp.normal);
                cp.normalBias =
                    vRelNormal < -kRestitutionThreshold ? -contact.restitution * vRelNormal : 0.0f;
            }

            // Warm start: inherit the previous step's impulses from the nearest
            // cached point of this pair (resting bodies barely move, so proximity
            // matching is robust). Unmatched points start from zero.
            const auto cached = cache_.find(contact.key);
            if (cached != cache_.end())
            {
                float bestDistSq = kWarmStartMatchRadius * kWarmStartMatchRadius;
                const CachedPoint* match = nullptr;
                for (const CachedPoint& prev : cached->second)
                {
                    const float distSq = (prev.point - cp.point).magnitudeSquared();
                    if (distSq <= bestDistSq)
                    {
                        bestDistSq = distSq;
                        match = &prev;
                    }
                }
                if (match != nullptr)
                {
                    cp.normalImpulse = match->normalImpulse;
                    cp.tangentImpulse1 = match->tangentImpulse1;
                    cp.tangentImpulse2 = match->tangentImpulse2;
                }
            }

            points_.push_back(cp);
        }
    }
}

void ContactSolver::warmStart(std::vector<SolverBody>& bodies) const
{
    for (const ConstraintPoint& cp : points_)
    {
        SolverBody& a = bodies[static_cast<std::size_t>(cp.a)];
        SolverBody& b = bodies[static_cast<std::size_t>(cp.b)];
        const Vec3 impulse = cp.normal * cp.normalImpulse + cp.tangent1 * cp.tangentImpulse1 +
                             cp.tangent2 * cp.tangentImpulse2;
        a.velocity += impulse * cp.invMassA;
        a.angularVelocity +=
            invInertiaWorld_[static_cast<std::size_t>(cp.a)] * cross(cp.rA, impulse);
        b.velocity -= impulse * cp.invMassB;
        b.angularVelocity -=
            invInertiaWorld_[static_cast<std::size_t>(cp.b)] * cross(cp.rB, impulse);
    }
}

void ContactSolver::solveVelocity(std::vector<SolverBody>& bodies)
{
    // Friction first (clamped against the accumulated normal impulse from the
    // previous iteration), then the normal constraint — the standard ordering
    // that keeps the friction cone consistent. Each impulse updates both linear
    // and angular velocity (lever arm rA/rB through the world inverse inertia).
    for (ConstraintPoint& cp : points_)
    {
        SolverBody& a = bodies[static_cast<std::size_t>(cp.a)];
        SolverBody& b = bodies[static_cast<std::size_t>(cp.b)];
        const Mat3& iA = invInertiaWorld_[static_cast<std::size_t>(cp.a)];
        const Mat3& iB = invInertiaWorld_[static_cast<std::size_t>(cp.b)];

        auto applyImpulse = [&](const Vec3& impulse)
        {
            a.velocity += impulse * cp.invMassA;
            a.angularVelocity += iA * cross(cp.rA, impulse);
            b.velocity -= impulse * cp.invMassB;
            b.angularVelocity -= iB * cross(cp.rB, impulse);
        };

        if (cp.friction > 0.0f && cp.normalImpulse > 0.0f)
        {
            const float maxFriction = cp.friction * cp.normalImpulse;

            const float vt1 = dot(relativeVelocity(a, b, cp.rA, cp.rB), cp.tangent1);
            float lambda1 = -cp.tangentMass1 * vt1;
            const float old1 = cp.tangentImpulse1;
            cp.tangentImpulse1 = std::clamp(old1 + lambda1, -maxFriction, maxFriction);
            lambda1 = cp.tangentImpulse1 - old1;
            applyImpulse(cp.tangent1 * lambda1);

            const float vt2 = dot(relativeVelocity(a, b, cp.rA, cp.rB), cp.tangent2);
            float lambda2 = -cp.tangentMass2 * vt2;
            const float old2 = cp.tangentImpulse2;
            cp.tangentImpulse2 = std::clamp(old2 + lambda2, -maxFriction, maxFriction);
            lambda2 = cp.tangentImpulse2 - old2;
            applyImpulse(cp.tangent2 * lambda2);
        }

        const float vn = dot(relativeVelocity(a, b, cp.rA, cp.rB), cp.normal);
        float lambda = -cp.normalMass * (vn - cp.normalBias);
        const float oldImpulse = cp.normalImpulse;
        cp.normalImpulse = std::max(oldImpulse + lambda, 0.0f);
        lambda = cp.normalImpulse - oldImpulse;
        applyImpulse(cp.normal * lambda);
    }
}

void ContactSolver::solvePosition(std::vector<SolverBody>& bodies)
{
    if (points_.empty())
    {
        return;
    }

    std::ranges::fill(pseudoVelocity_, Vec3{});
    std::ranges::fill(pseudoAngularVelocity_, Vec3{});

    for (int iteration = 0; iteration < kPositionIterations; ++iteration)
    {
        for (ConstraintPoint& cp : points_)
        {
            const auto a = static_cast<std::size_t>(cp.a);
            const auto b = static_cast<std::size_t>(cp.b);

            // Desired separation speed: remove a fraction of the penetration that
            // exceeds the slop. dt cancels against the position update below.
            const float correction =
                kBaumgarte * std::max(cp.penetration - kLinearSlop, 0.0f) / dt_;

            // Pseudo relative-normal velocity includes a pseudo-angular term, so the
            // correction can rotate a body out of penetration as well as translate it.
            const Vec3 prvVec = (pseudoVelocity_[a] + cross(pseudoAngularVelocity_[a], cp.rA)) -
                                (pseudoVelocity_[b] + cross(pseudoAngularVelocity_[b], cp.rB));
            float lambda = -cp.posNormalMass * (dot(prvVec, cp.normal) - correction);
            const float oldImpulse = cp.pseudoImpulse;
            cp.pseudoImpulse = std::max(oldImpulse + lambda, 0.0f);
            lambda = cp.pseudoImpulse - oldImpulse;

            const Vec3 p = cp.normal * lambda;
            pseudoVelocity_[a] += p * cp.posWeightA;
            pseudoAngularVelocity_[a] += invInertiaWorld_[a] * cross(cp.rA, p);
            pseudoVelocity_[b] -= p * cp.posWeightB;
            pseudoAngularVelocity_[b] -= invInertiaWorld_[b] * cross(cp.rB, p);
        }
    }

    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        bodies[i].position += pseudoVelocity_[i] * dt_;
        if (pseudoAngularVelocity_[i].magnitudeSquared() > 0.0f)
        {
            bodies[i].orientation = bodies[i].orientation.integrate(pseudoAngularVelocity_[i], dt_);
        }
    }
}

void ContactSolver::store()
{
    // Collect this step's accumulated impulses per pair (in deterministic points_
    // order), then swap them in as next step's warm-start source.
    next_.clear();
    for (const ConstraintPoint& cp : points_)
    {
        next_[cp.key].push_back(
            CachedPoint{cp.point, cp.normalImpulse, cp.tangentImpulse1, cp.tangentImpulse2});
    }
    cache_.swap(next_);
}

} // namespace fire_engine
