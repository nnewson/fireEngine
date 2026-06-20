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

// Effective inverse mass along a unit direction for a two-body constraint.
// Linear only: the contribution is just invMassA + invMassB. Returns 0 (an
// inert row) when both bodies are immovable, so the caller can skip it safely.
[[nodiscard]] float effectiveMass(float invMassA, float invMassB) noexcept
{
    const float sum = invMassA + invMassB;
    return sum > 0.0f ? 1.0f / sum : 0.0f;
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

    for (const SolverContactInput& contact : contacts)
    {
        const SolverBody& bodyA = bodies[static_cast<std::size_t>(contact.bodyA)];
        const SolverBody& bodyB = bodies[static_cast<std::size_t>(contact.bodyB)];

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
            cp.penetration = contact.penetration[static_cast<std::size_t>(i)];
            cp.friction = contact.friction;

            cp.normalMass = effectiveMass(cp.invMassA, cp.invMassB);
            cp.posNormalMass = effectiveMass(cp.posWeightA, cp.posWeightB);
            cp.tangentMass1 = cp.normalMass;
            cp.tangentMass2 = cp.normalMass;
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
                // means the bodies are approaching.
                const float vRelNormal = dot(bodyA.velocity - bodyB.velocity, cp.normal);
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
        const Vec3 impulse = cp.normal * cp.normalImpulse + cp.tangent1 * cp.tangentImpulse1 +
                             cp.tangent2 * cp.tangentImpulse2;
        bodies[static_cast<std::size_t>(cp.a)].velocity += impulse * cp.invMassA;
        bodies[static_cast<std::size_t>(cp.b)].velocity -= impulse * cp.invMassB;
    }
}

void ContactSolver::solveVelocity(std::vector<SolverBody>& bodies)
{
    // Friction first (clamped against the accumulated normal impulse from the
    // previous iteration), then the normal constraint — the standard ordering
    // that keeps the friction cone consistent.
    for (ConstraintPoint& cp : points_)
    {
        SolverBody& a = bodies[static_cast<std::size_t>(cp.a)];
        SolverBody& b = bodies[static_cast<std::size_t>(cp.b)];

        if (cp.friction > 0.0f && cp.normalImpulse > 0.0f)
        {
            const float maxFriction = cp.friction * cp.normalImpulse;

            const float vt1 = dot(a.velocity - b.velocity, cp.tangent1);
            float lambda1 = -cp.tangentMass1 * vt1;
            const float old1 = cp.tangentImpulse1;
            cp.tangentImpulse1 = std::clamp(old1 + lambda1, -maxFriction, maxFriction);
            lambda1 = cp.tangentImpulse1 - old1;
            a.velocity += cp.tangent1 * (lambda1 * cp.invMassA);
            b.velocity -= cp.tangent1 * (lambda1 * cp.invMassB);

            const float vt2 = dot(a.velocity - b.velocity, cp.tangent2);
            float lambda2 = -cp.tangentMass2 * vt2;
            const float old2 = cp.tangentImpulse2;
            cp.tangentImpulse2 = std::clamp(old2 + lambda2, -maxFriction, maxFriction);
            lambda2 = cp.tangentImpulse2 - old2;
            a.velocity += cp.tangent2 * (lambda2 * cp.invMassA);
            b.velocity -= cp.tangent2 * (lambda2 * cp.invMassB);
        }

        const float vn = dot(a.velocity - b.velocity, cp.normal);
        float lambda = -cp.normalMass * (vn - cp.normalBias);
        const float oldImpulse = cp.normalImpulse;
        cp.normalImpulse = std::max(oldImpulse + lambda, 0.0f);
        lambda = cp.normalImpulse - oldImpulse;
        a.velocity += cp.normal * (lambda * cp.invMassA);
        b.velocity -= cp.normal * (lambda * cp.invMassB);
    }
}

void ContactSolver::solvePosition(std::vector<SolverBody>& bodies)
{
    if (points_.empty())
    {
        return;
    }

    std::ranges::fill(pseudoVelocity_, Vec3{});

    for (int iteration = 0; iteration < kPositionIterations; ++iteration)
    {
        for (ConstraintPoint& cp : points_)
        {
            // Desired separation speed: remove a fraction of the penetration that
            // exceeds the slop. dt cancels against the position update below.
            const float correction =
                kBaumgarte * std::max(cp.penetration - kLinearSlop, 0.0f) / dt_;

            const float prv = dot(pseudoVelocity_[static_cast<std::size_t>(cp.a)] -
                                      pseudoVelocity_[static_cast<std::size_t>(cp.b)],
                                  cp.normal);
            float lambda = -cp.posNormalMass * (prv - correction);
            const float oldImpulse = cp.pseudoImpulse;
            cp.pseudoImpulse = std::max(oldImpulse + lambda, 0.0f);
            lambda = cp.pseudoImpulse - oldImpulse;

            pseudoVelocity_[static_cast<std::size_t>(cp.a)] += cp.normal * (lambda * cp.posWeightA);
            pseudoVelocity_[static_cast<std::size_t>(cp.b)] -= cp.normal * (lambda * cp.posWeightB);
        }
    }

    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        bodies[i].position += pseudoVelocity_[i] * dt_;
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
