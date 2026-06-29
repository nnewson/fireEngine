#include <fire_engine/physics/joint_solver.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include <fire_engine/physics/physics_constants.hpp>

namespace fire_engine
{

namespace
{

constexpr float kInf = std::numeric_limits<float>::infinity();

[[nodiscard]] float dot(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::dotProduct(a, b);
}

// Orthonormal pair spanning the plane perpendicular to unit `n` (same seed logic
// as the contact solver's tangent basis — avoids a near-zero cross product).
void perpendicularBasis(const Vec3& n, Vec3& t1, Vec3& t2) noexcept
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

// Signed rotation angle of `q` about unit `axis` (radians, in [-π, π]). `q` is
// canonicalised to w ≥ 0 first so the half-angle stays in range. This is the twist
// component of a swing-twist decomposition when `axis` is the twist axis.
[[nodiscard]] float twistAngleAbout(const Quaternion& q, const Vec3& axis) noexcept
{
    Quaternion c = q;
    if (c.w() < 0.0f)
    {
        c = -c;
    }
    const float d = c.x() * axis.x() + c.y() * axis.y() + c.z() * axis.z();
    return 2.0f * std::atan2(d, c.w());
}

} // namespace

void JointSolver::pushRow(int a, int b, float invMassA, float invMassB, const Mat3& iA,
                          const Mat3& iB, const Vec3& linearA, const Vec3& angularA,
                          const Vec3& linearB, const Vec3& angularB, float positionError,
                          float lower, float upper, std::uint64_t key, int slot)
{
    ConstraintRow row;
    row.a = a;
    row.b = b;
    row.invMassA = invMassA;
    row.invMassB = invMassB;
    row.linearA = linearA;
    row.angularA = angularA;
    row.linearB = linearB;
    row.angularB = angularB;

    // Effective mass = 1 / (J M⁻¹ Jᵀ) for the full Jacobian.
    const float k = invMassA * dot(linearA, linearA) + dot(angularA, iA * angularA) +
                    invMassB * dot(linearB, linearB) + dot(angularB, iB * angularB);
    row.effectiveMass = k > 0.0f ? 1.0f / k : 0.0f;

    // Store the slop-corrected error C, leaving a small deadzone so a satisfied joint
    // contributes no spring bias. Sign-preserving reduction keeps bilateral errors
    // stable. solveVelocity() turns this into a soft-constraint (damped-spring) bias.
    row.positionError = positionError > 0.0f ? std::max(positionError - kJointSlop, 0.0f)
                                             : std::min(positionError + kJointSlop, 0.0f);

    row.lower = lower;
    row.upper = upper;
    row.key = key;
    row.slot = slot;
    rows_.push_back(row);
}

void JointSolver::addPointRows(const SolverBody& a, const SolverBody& b, const JointInput& joint,
                               const Mat3& iA, const Mat3& iB, const Vec3& rA, const Vec3& rB,
                               int& slot)
{
    // Keep the two world anchors coincident: C = anchorA - anchorB = 0, one row per
    // world axis. linearA = e, angularA = rA×e; body B takes the opposite Jacobian.
    const Vec3 separation = joint.anchorA - joint.anchorB;
    const Vec3 axes[3] = {Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f}};
    for (const Vec3& e : axes)
    {
        pushRow(joint.bodyA, joint.bodyB, a.invMass, b.invMass, iA, iB, e,
                Vec3::crossProduct(rA, e), e * -1.0f, Vec3::crossProduct(rB, e) * -1.0f,
                dot(separation, e), -kInf, kInf, joint.key, slot++);
    }
}

void JointSolver::addAxisRows(const SolverBody& a, const SolverBody& b, const JointInput& joint,
                              const Mat3& iA, const Mat3& iB, int& slot)
{
    // Align the two world hinge axes (remove the 2 DOF perpendicular to the axis).
    // For a perpendicular t (fixed to A), C = t·axisB; Ċ = (ωB - ωA)·(axisB×t), so the
    // angular Jacobian is angularA = -(axisB×t), angularB = +(axisB×t), no linear part.
    const Vec3 axisAWorld = Vec3::normalise(joint.axisA);
    const Vec3 axisBWorld = Vec3::normalise(joint.axisB);
    Vec3 t1;
    Vec3 t2;
    perpendicularBasis(axisAWorld, t1, t2);

    const Vec3 perps[2] = {t1, t2};
    for (const Vec3& t : perps)
    {
        const Vec3 axis = Vec3::crossProduct(axisBWorld, t);
        pushRow(joint.bodyA, joint.bodyB, a.invMass, b.invMass, iA, iB, Vec3{}, axis * -1.0f,
                Vec3{}, axis, dot(t, axisBWorld), -kInf, kInf, joint.key, slot++);
    }
}

void JointSolver::addAngleLimitRow(const SolverBody& a, const SolverBody& b,
                                   const JointInput& joint, const Mat3& iA, const Mat3& iB,
                                   const Vec3& worldAxis, float angle, float lower, float upper,
                                   int slot)
{
    // Rotation about worldAxis: Ċ = (ωB - ωA)·axis, so angularA = -axis, angularB =
    // +axis (no linear part). Only add a row while a limit is violated, with a
    // one-sided clamp so it pushes the angle back into range and never the other way.
    float positionError;
    float lo;
    float hi;
    if (angle > upper)
    {
        positionError = angle - upper;
        lo = -kInf;
        hi = 0.0f;
    }
    else if (angle < lower)
    {
        positionError = angle - lower;
        lo = 0.0f;
        hi = kInf;
    }
    else
    {
        return;
    }

    pushRow(joint.bodyA, joint.bodyB, a.invMass, b.invMass, iA, iB, Vec3{}, worldAxis * -1.0f,
            Vec3{}, worldAxis, positionError, lo, hi, joint.key, slot);
}

void JointSolver::addHingeLimit(const SolverBody& a, const SolverBody& b, const JointInput& joint,
                                const Mat3& iA, const Mat3& iB, int& slot)
{
    const Vec3 axisWorld = Vec3::normalise(joint.axisA);
    const float angle = twistAngleAbout(joint.relative, Vec3::normalise(joint.twistAxisLocal));
    addAngleLimitRow(a, b, joint, iA, iB, axisWorld, angle, joint.limits.lowerAngle,
                     joint.limits.upperAngle, slot);
    ++slot;
}

void JointSolver::addConeTwistLimit(const SolverBody& a, const SolverBody& b,
                                    const JointInput& joint, const Mat3& iA, const Mat3& iB,
                                    int& slot)
{
    const Vec3 twistAxis = Vec3::normalise(joint.twistAxisLocal);

    // Swing-twist decomposition of the relative orientation about the twist axis:
    // q = swing · twist, twist a rotation about twistAxis, swing the remainder.
    Quaternion q = joint.relative;
    if (q.w() < 0.0f)
    {
        q = -q;
    }
    const float d = q.x() * twistAxis.x() + q.y() * twistAxis.y() + q.z() * twistAxis.z();
    const Quaternion twist = Quaternion::normalise(
        Quaternion{twistAxis.x() * d, twistAxis.y() * d, twistAxis.z() * d, q.w()});
    const Quaternion swing = q * twist.conjugate();

    // Swing cone: a single unilateral row about the (world) swing axis once the cone
    // half-angle is exceeded.
    const Vec3 swingVec{swing.x(), swing.y(), swing.z()};
    const float swingSin = swingVec.magnitude();
    const float swingAngle = 2.0f * std::atan2(swingSin, swing.w());
    if (swingSin > 1.0e-5f)
    {
        const Vec3 swingAxisWorld = a.orientation.rotate(swingVec / swingSin);
        addAngleLimitRow(a, b, joint, iA, iB, swingAxisWorld, swingAngle, -kInf,
                         joint.limits.swingLimit, slot);
    }
    ++slot;

    // Twist: a ±twistLimit clamp about the (world) twist axis.
    const Vec3 twistAxisWorld = a.orientation.rotate(twistAxis);
    const float twistAngle = 2.0f * std::atan2(d, q.w());
    addAngleLimitRow(a, b, joint, iA, iB, twistAxisWorld, twistAngle, -joint.limits.twistLimit,
                     joint.limits.twistLimit, slot);
    ++slot;
}

void JointSolver::prepare(std::span<const SolverBody> bodies, std::span<const JointInput> joints,
                          float dt)
{
    dt_ = dt;
    rows_.clear();

    // Soft-constraint coefficients (Box2D-v3 `b2MakeSoft`): a damped spring at frequency
    // kJointHertz with damping ratio kJointDampingRatio. `impulseScale_` decays the
    // accumulated impulse each sweep (dissipative — the anti-energy-pump term).
    {
        const float omega = 2.0f * std::numbers::pi_v<float> * kJointHertz;
        const float a1 = 2.0f * kJointDampingRatio + dt_ * omega;
        const float a2 = dt_ * omega * a1;
        const float a3 = 1.0f / (1.0f + a2);
        biasRate_ = omega / a1;
        massScale_ = a2 * a3;
        impulseScale_ = a3;
    }

    invInertiaWorld_.assign(bodies.size(), Mat3{});
    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        invInertiaWorld_[i] =
            worldInverseInertia(bodies[i].orientation, bodies[i].inverseInertiaLocal);
    }

    for (const JointInput& joint : joints)
    {
        const SolverBody& a = bodies[static_cast<std::size_t>(joint.bodyA)];
        const SolverBody& b = bodies[static_cast<std::size_t>(joint.bodyB)];
        const Mat3& iA = invInertiaWorld_[static_cast<std::size_t>(joint.bodyA)];
        const Mat3& iB = invInertiaWorld_[static_cast<std::size_t>(joint.bodyB)];
        const Vec3 rA = joint.anchorA - a.position;
        const Vec3 rB = joint.anchorB - b.position;

        int slot = 0;
        switch (joint.type)
        {
        case JointType::Distance:
        {
            // Hold the anchor separation at restLength: 1 row along the anchor axis.
            const Vec3 delta = joint.anchorA - joint.anchorB;
            const float length = delta.magnitude();
            const Vec3 dir = length > 1.0e-6f ? delta / length : Vec3{0.0f, 1.0f, 0.0f};
            pushRow(joint.bodyA, joint.bodyB, a.invMass, b.invMass, iA, iB, dir,
                    Vec3::crossProduct(rA, dir), dir * -1.0f, Vec3::crossProduct(rB, dir) * -1.0f,
                    length - joint.restLength, -kInf, kInf, joint.key, slot++);
            break;
        }
        case JointType::BallSocket:
            addPointRows(a, b, joint, iA, iB, rA, rB, slot);
            if (joint.limits.coneTwist)
            {
                addConeTwistLimit(a, b, joint, iA, iB, slot);
            }
            break;
        case JointType::Hinge:
            addPointRows(a, b, joint, iA, iB, rA, rB, slot);
            addAxisRows(a, b, joint, iA, iB, slot);
            if (joint.limits.hinge)
            {
                addHingeLimit(a, b, joint, iA, iB, slot);
            }
            break;
        }
    }

    // Warm start: seed each row's accumulated impulse from the matching cached slot.
    for (ConstraintRow& row : rows_)
    {
        const auto cached = cache_.find(row.key);
        if (cached != cache_.end() && row.slot < static_cast<int>(cached->second.size()))
        {
            row.impulse = cached->second[static_cast<std::size_t>(row.slot)];
        }
    }
}

void JointSolver::warmStart(std::vector<SolverBody>& bodies) const
{
    for (const ConstraintRow& row : rows_)
    {
        SolverBody& a = bodies[static_cast<std::size_t>(row.a)];
        SolverBody& b = bodies[static_cast<std::size_t>(row.b)];
        const Mat3& iA = invInertiaWorld_[static_cast<std::size_t>(row.a)];
        const Mat3& iB = invInertiaWorld_[static_cast<std::size_t>(row.b)];

        a.velocity += row.linearA * (row.invMassA * row.impulse);
        a.angularVelocity += iA * (row.angularA * row.impulse);
        b.velocity += row.linearB * (row.invMassB * row.impulse);
        b.angularVelocity += iB * (row.angularB * row.impulse);
    }
}

void JointSolver::solveVelocity(std::vector<SolverBody>& bodies)
{
    for (ConstraintRow& row : rows_)
    {
        SolverBody& a = bodies[static_cast<std::size_t>(row.a)];
        SolverBody& b = bodies[static_cast<std::size_t>(row.b)];
        const Mat3& iA = invInertiaWorld_[static_cast<std::size_t>(row.a)];
        const Mat3& iB = invInertiaWorld_[static_cast<std::size_t>(row.b)];

        const float jv = dot(row.linearA, a.velocity) + dot(row.angularA, a.angularVelocity) +
                         dot(row.linearB, b.velocity) + dot(row.angularB, b.angularVelocity);
        // Soft constraint: a damped-spring bias toward zero error, scaled mass, and an
        // impulse-decay term that dissipates energy (vs the old hard `-(kBaumgarte/dt)·C`
        // bias, which fed unresolved error back as velocity = an energy pump).
        const float bias = biasRate_ * row.positionError;
        float lambda = -row.effectiveMass * massScale_ * (jv + bias) - impulseScale_ * row.impulse;

        const float oldImpulse = row.impulse;
        row.impulse = std::clamp(oldImpulse + lambda, row.lower, row.upper);
        lambda = row.impulse - oldImpulse;

        a.velocity += row.linearA * (row.invMassA * lambda);
        a.angularVelocity += iA * (row.angularA * lambda);
        b.velocity += row.linearB * (row.invMassB * lambda);
        b.angularVelocity += iB * (row.angularB * lambda);
    }
}

void JointSolver::beginStore() noexcept
{
    next_.clear();
}

void JointSolver::store()
{
    // Append this island's accumulated row impulses (per joint key, by slot) to the
    // pending cache. Each joint key belongs to one island, so appends never collide.
    for (const ConstraintRow& row : rows_)
    {
        std::vector<float>& impulses = next_[row.key];
        if (row.slot >= static_cast<int>(impulses.size()))
        {
            impulses.resize(static_cast<std::size_t>(row.slot) + 1, 0.0f);
        }
        impulses[static_cast<std::size_t>(row.slot)] = row.impulse;
    }
}

void JointSolver::commitStore() noexcept
{
    cache_.swap(next_);
}

} // namespace fire_engine
