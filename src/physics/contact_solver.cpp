#include <fire_engine/physics/contact_solver.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

#include <fire_engine/physics/physics_constants.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] float dot(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::dotProduct(a, b);
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

void ContactSolver::prepare(std::span<const SolverBody> bodies,
                            std::span<const SolverContactInput> contacts, float h)
{
    h_ = h;
    points_.clear();
    pseudoVelocity_.assign(bodies.size(), Vec3{});
    pseudoAngularVelocity_.assign(bodies.size(), Vec3{});

    // Soft contact coefficients (Box2D-v3 `b2MakeSoft`): a damped spring at frequency
    // kContactHertz with damping ratio kContactDampingRatio, evaluated at the substep h.
    {
        const float omega = 2.0f * std::numbers::pi_v<float> * kContactHertz;
        const float a1 = 2.0f * kContactDampingRatio + h_ * omega;
        const float a2 = h_ * omega * a1;
        const float a3 = 1.0f / (1.0f + a2);
        biasRate_ = omega / a1;
        massScale_ = a2 * a3;
        impulseScale_ = a3;
    }

    // World inverse inertia per body (shared helper; zero for infinite-inertia bodies).
    invInertiaWorld_.assign(bodies.size(), Mat3{});
    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        invInertiaWorld_[i] =
            worldInverseInertia(bodies[i].orientation, bodies[i].inverseInertiaLocal);
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

            // Prepare-time levers (contact point − COM). Stored body-local so the
            // current lever can be recovered as orientation·anchorLocal each substep.
            const Vec3 rA0 = cp.point - bodyA.position;
            const Vec3 rB0 = cp.point - bodyB.position;
            cp.anchorLocalA = bodyA.orientation.conjugate().rotate(rA0);
            cp.anchorLocalB = bodyB.orientation.conjugate().rotate(rB0);
            // Both anchors coincide at the single manifold point at prepare, so the
            // adjusted separation is just the prepare separation (= −penetration).
            cp.adjustedSeparation = -contact.penetration[static_cast<std::size_t>(i)];

            cp.restitution = contact.restitution;
            cp.friction = contact.friction;

            cp.normalMass =
                effectiveMassAlong(cp.invMassA, cp.invMassB, rA0, rB0, cp.normal, iA, iB);
            cp.posNormalMass =
                effectiveMassAlong(cp.posWeightA, cp.posWeightB, rA0, rB0, cp.normal, iA, iB);
            cp.tangentMass1 =
                effectiveMassAlong(cp.invMassA, cp.invMassB, rA0, rB0, tangent1, iA, iB);
            cp.tangentMass2 =
                effectiveMassAlong(cp.invMassA, cp.invMassB, rA0, rB0, tangent2, iA, iB);
            cp.key = contact.key;

            // Approach velocity at prepare (negative = closing), for the end-of-step
            // restitution pass: rebound to −restitution·relVelN0.
            cp.relVelN0 = dot(relativeVelocity(bodyA, bodyB, rA0, rB0), cp.normal);

            // Warm start: inherit the previous step's impulses from the nearest cached
            // point of this pair (resting bodies barely move, so proximity matching is
            // robust). Unmatched points start from zero.
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
        const Vec3 rA = a.orientation.rotate(cp.anchorLocalA);
        const Vec3 rB = b.orientation.rotate(cp.anchorLocalB);
        const Vec3 impulse = cp.normal * cp.normalImpulse + cp.tangent1 * cp.tangentImpulse1 +
                             cp.tangent2 * cp.tangentImpulse2;
        applyImpulse(a, b, cp.invMassA, cp.invMassB,
                     invInertiaWorld_[static_cast<std::size_t>(cp.a)],
                     invInertiaWorld_[static_cast<std::size_t>(cp.b)], rA, rB, impulse);
    }
}

void ContactSolver::solveVelocity(std::vector<SolverBody>& bodies, bool useBias)
{
    // Normal first, then friction clamped against the fresh normal impulse (Box2D-v3
    // ordering) — solving the normal first keeps the friction cone consistent with the
    // soft constraint's per-pass impulse, which the friction-first ordering would lag.
    // The levers rA/rB and the penetration separation are recomputed from the current
    // pose each call, so corrections track the bodies as they move through the substeps.
    const float invH = h_ > 0.0f ? 1.0f / h_ : 0.0f;

    for (ConstraintPoint& cp : points_)
    {
        SolverBody& a = bodies[static_cast<std::size_t>(cp.a)];
        SolverBody& b = bodies[static_cast<std::size_t>(cp.b)];
        const Mat3& iA = invInertiaWorld_[static_cast<std::size_t>(cp.a)];
        const Mat3& iB = invInertiaWorld_[static_cast<std::size_t>(cp.b)];

        const Vec3 rA = a.orientation.rotate(cp.anchorLocalA);
        const Vec3 rB = b.orientation.rotate(cp.anchorLocalB);

        auto apply = [&](const Vec3& impulse)
        { applyImpulse(a, b, cp.invMassA, cp.invMassB, iA, iB, rA, rB, impulse); };

        // Current separation along the normal (negative = penetrating).
        const float separation =
            dot((a.position + rA) - (b.position + rB), cp.normal) + cp.adjustedSeparation;

        float bias = 0.0f;
        float massScale = 1.0f;
        float impulseScale = 0.0f;
        if (separation > 0.0f)
        {
            // Speculative gap: allow the bodies to close it this substep but no more
            // (anti-tunnelling), regardless of the bias/relax phase.
            bias = separation * invH;
        }
        else if (useBias)
        {
            // Soft penetration push-out, capped so a deep initial overlap can't launch.
            bias = std::max(biasRate_ * separation, -kMaxBiasVelocity);
            massScale = massScale_;
            impulseScale = impulseScale_;
        }

        const float vn = dot(relativeVelocity(a, b, rA, rB), cp.normal);
        float lambda = -cp.normalMass * massScale * (vn + bias) - impulseScale * cp.normalImpulse;
        const float oldImpulse = cp.normalImpulse;
        cp.normalImpulse = std::max(oldImpulse + lambda, 0.0f);
        cp.maxNormalImpulse = std::max(cp.maxNormalImpulse, cp.normalImpulse);
        lambda = cp.normalImpulse - oldImpulse;
        apply(cp.normal * lambda);

        if (cp.friction > 0.0f && cp.normalImpulse > 0.0f)
        {
            const float maxFriction = cp.friction * cp.normalImpulse;

            const float vt1 = dot(relativeVelocity(a, b, rA, rB), cp.tangent1);
            float lambda1 = -cp.tangentMass1 * vt1;
            const float old1 = cp.tangentImpulse1;
            cp.tangentImpulse1 = std::clamp(old1 + lambda1, -maxFriction, maxFriction);
            lambda1 = cp.tangentImpulse1 - old1;
            apply(cp.tangent1 * lambda1);

            const float vt2 = dot(relativeVelocity(a, b, rA, rB), cp.tangent2);
            float lambda2 = -cp.tangentMass2 * vt2;
            const float old2 = cp.tangentImpulse2;
            cp.tangentImpulse2 = std::clamp(old2 + lambda2, -maxFriction, maxFriction);
            lambda2 = cp.tangentImpulse2 - old2;
            apply(cp.tangent2 * lambda2);
        }
    }
}

void ContactSolver::applyRestitution(std::vector<SolverBody>& bodies)
{
    for (ConstraintPoint& cp : points_)
    {
        // Only points that were approaching faster than the resting threshold and
        // actually carried a normal impulse rebound — settling stacks do not buzz.
        if (cp.restitution <= 0.0f || cp.relVelN0 > -kRestitutionThreshold ||
            cp.maxNormalImpulse <= 0.0f)
        {
            continue;
        }

        SolverBody& a = bodies[static_cast<std::size_t>(cp.a)];
        SolverBody& b = bodies[static_cast<std::size_t>(cp.b)];
        const Mat3& iA = invInertiaWorld_[static_cast<std::size_t>(cp.a)];
        const Mat3& iB = invInertiaWorld_[static_cast<std::size_t>(cp.b)];
        const Vec3 rA = a.orientation.rotate(cp.anchorLocalA);
        const Vec3 rB = b.orientation.rotate(cp.anchorLocalB);

        const float vn = dot(relativeVelocity(a, b, rA, rB), cp.normal);
        // Drive the separating speed to −restitution·(approach speed).
        float lambda = -cp.normalMass * (vn + cp.restitution * cp.relVelN0);
        const float oldImpulse = cp.normalImpulse;
        cp.normalImpulse = std::max(oldImpulse + lambda, 0.0f);
        lambda = cp.normalImpulse - oldImpulse;
        applyImpulse(a, b, cp.invMassA, cp.invMassB, iA, iB, rA, rB, cp.normal * lambda);
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

    const float invH = h_ > 0.0f ? 1.0f / h_ : 0.0f;

    for (int iteration = 0; iteration < kPositionIterations; ++iteration)
    {
        for (ConstraintPoint& cp : points_)
        {
            const auto a = static_cast<std::size_t>(cp.a);
            const auto b = static_cast<std::size_t>(cp.b);

            // Current lever arms + separation (recomputed from the moving poses, so the
            // dynamic side's substep motion is reflected for kinematic-vs-dynamic pairs).
            const Vec3 rA = bodies[a].orientation.rotate(cp.anchorLocalA);
            const Vec3 rB = bodies[b].orientation.rotate(cp.anchorLocalB);
            const float separation =
                dot((bodies[a].position + rA) - (bodies[b].position + rB), cp.normal) +
                cp.adjustedSeparation;
            const float penetration = -separation;

            // Desired separation speed: remove a fraction of the penetration beyond the
            // slop. The 1/h here cancels against the *h in the position write-back below.
            const float correction = kBaumgarte * std::max(penetration - kLinearSlop, 0.0f) * invH;

            const Vec3 prvVec =
                (pseudoVelocity_[a] + Vec3::crossProduct(pseudoAngularVelocity_[a], rA)) -
                (pseudoVelocity_[b] + Vec3::crossProduct(pseudoAngularVelocity_[b], rB));
            float lambda = -cp.posNormalMass * (dot(prvVec, cp.normal) - correction);
            const float oldImpulse = cp.pseudoImpulse;
            cp.pseudoImpulse = std::max(oldImpulse + lambda, 0.0f);
            lambda = cp.pseudoImpulse - oldImpulse;

            const Vec3 p = cp.normal * lambda;
            pseudoVelocity_[a] += p * cp.posWeightA;
            pseudoAngularVelocity_[a] += invInertiaWorld_[a] * Vec3::crossProduct(rA, p);
            pseudoVelocity_[b] -= p * cp.posWeightB;
            pseudoAngularVelocity_[b] -= invInertiaWorld_[b] * Vec3::crossProduct(rB, p);
        }
    }

    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        if (pseudoVelocity_[i].magnitudeSquared() > 0.0f)
        {
            bodies[i].position += pseudoVelocity_[i] * h_;
        }
        if (pseudoAngularVelocity_[i].magnitudeSquared() > 0.0f)
        {
            bodies[i].orientation = bodies[i].orientation.integrate(pseudoAngularVelocity_[i], h_);
        }
    }
}

void ContactSolver::beginStore() noexcept
{
    next_.clear();
}

void ContactSolver::store()
{
    // Append this island's accumulated impulses per pair (in deterministic points_
    // order) to the pending cache. Keys are island-local, so appends never collide.
    for (const ConstraintPoint& cp : points_)
    {
        next_[cp.key].push_back(
            CachedPoint{cp.point, cp.normalImpulse, cp.tangentImpulse1, cp.tangentImpulse2});
    }
}

void ContactSolver::commitStore() noexcept
{
    // Swap the pending cache in as next step's warm-start source.
    cache_.swap(next_);
}

} // namespace fire_engine
