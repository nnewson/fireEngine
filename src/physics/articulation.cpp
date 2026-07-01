#include <fire_engine/physics/articulation.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

#include <fire_engine/math/quaternion.hpp>

namespace fire_engine
{

namespace
{

// Invert the top-left `nd`×`nd` block of a joint's D = SᵀU matrix: a scalar reciprocal for
// a 1-DOF revolute, a full 3×3 inverse for a 3-DOF spherical, zero otherwise.
[[nodiscard]] Mat3 invertDof(const Mat3& d, int nd) noexcept
{
    Mat3 r{};
    if (nd == 1)
    {
        if (d[0, 0] > 1e-9f)
        {
            r[0, 0] = 1.0f / d[0, 0];
        }
    }
    else if (nd == 3)
    {
        r = d.inverse();
    }
    return r;
}

// Passive cone-twist restoring torque (joint frame) for a spherical joint at rotation `q`
// with joint-frame angular velocity `omega`. Zero inside the swing cone (`swingLimit` about
// `twistAxis`) and the ±`twistLimit` twist; past either, a spring (`k`·excess) + damping
// pushes back. Returns the generalized torque on the 3 spherical DOFs. (Assumes the child
// frame coincides with the joint rotation frame — jointToChild a pure rotation about it.)
[[nodiscard]] Vec3 coneTwistTorque(const Quaternion& q, const Vec3& omega, const Vec3& twistAxis,
                                   float swingLimit, float twistLimit, float k,
                                   float damping) noexcept
{
    // Swing-twist decomposition about twistAxis: q = swing · twist.
    const float d = q.x() * twistAxis.x() + q.y() * twistAxis.y() + q.z() * twistAxis.z();
    const Quaternion twist = Quaternion::normalise(
        Quaternion{twistAxis.x() * d, twistAxis.y() * d, twistAxis.z() * d, q.w()});
    const Quaternion swing = q * twist.conjugate();

    Vec3 torque{};

    // Swing cone: push back once the twist axis leaves the cone half-angle.
    const Vec3 swingVec{swing.x(), swing.y(), swing.z()};
    const float swingSin = swingVec.magnitude();
    if (swingSin > 1.0e-5f)
    {
        const float swingAngle = 2.0f * std::atan2(swingSin, swing.w());
        if (swingAngle > swingLimit)
        {
            const Vec3 axis = swingVec * (1.0f / swingSin);
            torque = torque + axis * (-k * (swingAngle - swingLimit) -
                                      damping * Vec3::dotProduct(omega, axis));
        }
    }

    // Twist: clamp to ±twistLimit about the twist axis.
    const float twistAngle = 2.0f * std::atan2(d, q.w());
    if (twistAngle > twistLimit || twistAngle < -twistLimit)
    {
        const float excess =
            twistAngle > twistLimit ? twistAngle - twistLimit : twistAngle + twistLimit;
        torque = torque + twistAxis * (-k * excess - damping * Vec3::dotProduct(omega, twistAxis));
    }

    return torque;
}

// Body-frame rotation-vector error current→target (2·log of q⁻¹·q_target, small-angle exact),
// the axis·angle a spherical drive spring pulls along. `q` is the joint rotation (parent→child),
// so q⁻¹·q_target is the residual rotation expressed in the child (body) frame — the frame the
// generalized velocities q̇ live in.
[[nodiscard]] Vec3 orientationError(const Quaternion& q, const Quaternion& target) noexcept
{
    Quaternion e = q.conjugate() * target;
    if (e.w() < 0.0f) // shortest arc
    {
        e = Quaternion{-e.x(), -e.y(), -e.z(), -e.w()};
    }
    const Vec3 v{e.x(), e.y(), e.z()};
    const float s = v.magnitude();
    if (s < 1.0e-6f)
    {
        return Vec3{};
    }
    return v * (2.0f * std::atan2(s, e.w()) / s);
}

} // namespace

int Articulation::addRootLink(const ArticulationLinkDesc& desc)
{
    assert(links_.empty() && "addRootLink must be called exactly once, before any addLink");

    Link link;
    link.parent = -1;
    link.joint = ArticulationJointType::Fixed; // the root is the free-floating base
    link.mass = desc.mass;
    link.inertiaLocal = desc.inertiaLocal;
    link.comLocal = desc.comLocal;
    link.dofOffset = -1;
    link.dofCount = 0;

    links_.push_back(link);
    linkWorld_.emplace_back();
    return 0;
}

int Articulation::addLink(const ArticulationLinkDesc& desc)
{
    assert(!links_.empty() && "addRootLink must be added before any child link");
    assert(desc.parent >= 0 && static_cast<std::size_t>(desc.parent) < links_.size() &&
           "child link parent must reference an already-added link");

    Link link;
    link.parent = desc.parent;
    link.joint = desc.joint;
    link.jointAxis = desc.jointAxis;
    link.parentToJoint = desc.parentToJoint;
    link.jointToChild = desc.jointToChild;
    link.mass = desc.mass;
    link.inertiaLocal = desc.inertiaLocal;
    link.comLocal = desc.comLocal;
    link.swingLimit = desc.swingLimit;
    link.twistLimit = desc.twistLimit;
    link.limitStiffness = desc.limitStiffness;
    link.limitDamping = desc.limitDamping;
    link.driveTarget = desc.driveTarget;
    link.driveTargetRotation = desc.driveTargetRotation;
    link.driveStiffness = desc.driveStiffness;
    link.driveDamping = desc.driveDamping;

    link.dofCount = desc.joint == ArticulationJointType::Revolute    ? 1
                    : desc.joint == ArticulationJointType::Spherical ? 3
                                                                     : 0;
    if (link.dofCount > 0)
    {
        link.dofOffset = dofCount_;
        dofCount_ += link.dofCount;
        q_.resize(static_cast<std::size_t>(dofCount_), 0.0f);
        qDot_.resize(static_cast<std::size_t>(dofCount_), 0.0f);
        qDDot_.resize(static_cast<std::size_t>(dofCount_), 0.0f);
    }

    const int index = static_cast<int>(links_.size());
    links_.push_back(link);
    linkWorld_.emplace_back();
    return index;
}

void Articulation::q(int dof, float value) noexcept
{
    assert(dof >= 0 && dof < dofCount_ && "q index out of range");
    q_[static_cast<std::size_t>(dof)] = value;
}

void Articulation::qDot(int dof, float value) noexcept
{
    assert(dof >= 0 && dof < dofCount_ && "qDot index out of range");
    qDot_[static_cast<std::size_t>(dof)] = value;
}

void Articulation::jointRotation(std::size_t link, const Quaternion& rotation) noexcept
{
    links_[link].jointRotation = Quaternion::normalise(rotation);
}

void Articulation::forwardKinematics()
{
    if (links_.empty())
    {
        return;
    }

    // The root link's body frame is the floating base. Every other link is reached from
    // its (already-resolved) parent — links are topologically ordered, so one sweep suffices:
    //   world_i = world_parent · parentToJoint · R(q) · jointToChild
    // where R(q) is the joint's own motion about jointAxis (identity for a Fixed joint).
    linkWorld_[0] = base_;

    for (std::size_t i = 1; i < links_.size(); ++i)
    {
        const Link& link = links_[i];
        const RigidTransform& parentWorld = linkWorld_[static_cast<std::size_t>(link.parent)];

        RigidTransform jointMotion; // identity for Fixed
        if (link.joint == ArticulationJointType::Revolute)
        {
            const float angle = q_[static_cast<std::size_t>(link.dofOffset)];
            jointMotion.rotation = Quaternion::fromAxisAngle(link.jointAxis, angle);
        }
        else if (link.joint == ArticulationJointType::Spherical)
        {
            jointMotion.rotation = link.jointRotation;
        }

        linkWorld_[i] = parentWorld * link.parentToJoint * jointMotion * link.jointToChild;
    }
}

int Articulation::jointSubspace(std::size_t i, std::array<SpatialVector, 3>& s) const
{
    s = {};
    switch (links_[i].joint)
    {
    case ArticulationJointType::Revolute:
        s[0] = SpatialVector{links_[i].jointAxis, Vec3{}};
        return 1;
    case ArticulationJointType::Spherical:
        // The three joint-frame axes: a free 3-DOF rotation.
        s[0] = SpatialVector{Vec3{1.0f, 0.0f, 0.0f}, Vec3{}};
        s[1] = SpatialVector{Vec3{0.0f, 1.0f, 0.0f}, Vec3{}};
        s[2] = SpatialVector{Vec3{0.0f, 0.0f, 1.0f}, Vec3{}};
        return 3;
    case ArticulationJointType::Fixed:
        break;
    }
    return 0;
}

void Articulation::factorizeArticulatedInertia()
{
    forwardKinematics();
    const std::size_t n = links_.size();
    xup_.assign(n, SpatialMatrix{});
    xforce_.assign(n, SpatialMatrix{});
    artInertia_.assign(n, SpatialMatrix{});
    ia_.assign(n, SpatialMatrix{});
    subspace_.assign(n, std::array<SpatialVector, 3>{});
    u_.assign(n, std::array<SpatialVector, 3>{});
    uDinv_.assign(n, std::array<SpatialVector, 3>{});
    dInv_.assign(n, Mat3{});

    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        RigidTransform jm; // joint's own rotation (identity for Fixed)
        if (link.joint == ArticulationJointType::Revolute)
        {
            jm.rotation = Quaternion::fromAxisAngle(link.jointAxis,
                                                    q_[static_cast<std::size_t>(link.dofOffset)]);
        }
        else if (link.joint == ArticulationJointType::Spherical)
        {
            jm.rotation = link.jointRotation;
        }
        const RigidTransform t = link.parentToJoint * jm * link.jointToChild;
        xup_[i] = motionTransform(t.inverse());
        xforce_[i] = forceTransform(t);
        jointSubspace(i, subspace_[i]);
        artInertia_[i] =
            spatialInertia(link.mass, link.comLocal, Mat3::diagonal(link.inertiaLocal));
    }

    // Inward: build each joint's DOF factorization (U = Iᴬ·S, D⁻¹, U·D⁻¹, ia = Iᴬ − U·D⁻¹·Uᵀ)
    // and fold the projected inertia onto the parent. Generalises rank-1 (revolute) to rank-3
    // (spherical) via the small D⁻¹.
    for (std::size_t i = n - 1; i >= 1; --i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const int nd = link.dofCount;

        for (int k = 0; k < nd; ++k)
        {
            u_[i][static_cast<std::size_t>(k)] =
                artInertia_[i] * subspace_[i][static_cast<std::size_t>(k)];
        }
        Mat3 dmat{};
        for (int j = 0; j < nd; ++j)
        {
            for (int k = 0; k < nd; ++k)
            {
                dmat[j, k] = subspace_[i][static_cast<std::size_t>(j)].dot(
                    u_[i][static_cast<std::size_t>(k)]);
            }
        }
        dInv_[i] = invertDof(dmat, nd);

        SpatialMatrix ia = artInertia_[i];
        for (int k = 0; k < nd; ++k)
        {
            SpatialVector udk{};
            for (int j = 0; j < nd; ++j)
            {
                udk = udk + u_[i][static_cast<std::size_t>(j)] * dInv_[i][j, k];
            }
            uDinv_[i][static_cast<std::size_t>(k)] = udk;
            ia = ia - spatialOuter(udk, u_[i][static_cast<std::size_t>(k)]);
        }
        ia_[i] = ia;

        if (!(p == 0 && baseFixed_))
        {
            artInertia_[p] = artInertia_[p] + xforce_[i] * ia_[i] * xup_[i];
        }
    }
}

void Articulation::computeAccelerations(const Vec3& gravity, float jointDamping)
{
    factorizeArticulatedInertia(); // FK + geometry/inertia factorization (xup/xforce/ia/dInv/…)
    const std::size_t n = links_.size();
    const bool rootFixed = baseFixed_;

    std::vector<SpatialVector> vel(n);
    std::vector<SpatialVector> velProd(n); // velocity-product accel c = v × (S q̇)
    std::vector<SpatialVector> bias(n);    // articulated bias pᴬ (gravity + velocity product)
    std::vector<SpatialVector> accel(n);
    std::vector<std::array<float, 3>> uForce(n);

    // Pass 1 (outward): spatial velocities + the gravity / velocity-product bias.
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const int nd = link.dofCount;
        const auto off = static_cast<std::size_t>(link.dofOffset);

        SpatialVector vJoint{};
        for (int k = 0; k < nd; ++k)
        {
            vJoint = vJoint + subspace_[i][static_cast<std::size_t>(k)] *
                                  qDot_[off + static_cast<std::size_t>(k)];
        }
        const SpatialVector vParent = (p == 0 && rootFixed) ? SpatialVector{} : vel[p];
        vel[i] = xup_[i] * vParent + vJoint;
        velProd[i] = crossMotion(vel[i], vJoint);

        const SpatialMatrix linkInertia =
            spatialInertia(link.mass, link.comLocal, Mat3::diagonal(link.inertiaLocal));
        const Vec3 gLink = linkWorld_[i].rotation.conjugate().rotate(gravity);
        const Vec3 gForce = gLink * link.mass;
        const SpatialVector gravityWrench{Vec3::crossProduct(link.comLocal, gForce), gForce};
        bias[i] = crossForce(vel[i], linkInertia * vel[i]) - gravityWrench;
    }

    // Pass 2 (inward): fold the bias to each parent; record uForce = τ − Sᵀpᴬ per DOF.
    for (std::size_t i = n - 1; i >= 1; --i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const int nd = link.dofCount;
        const auto off = static_cast<std::size_t>(link.dofOffset);

        // Passive joint torque per DOF: global damping + cone-twist limit + drive spring.
        float jointTorque[3]{0.0f, 0.0f, 0.0f};
        if (link.joint == ArticulationJointType::Spherical)
        {
            const Vec3 omega{qDot_[off], qDot_[off + 1], qDot_[off + 2]};
            Vec3 t{};
            if (link.limitStiffness > 0.0f)
            {
                t = coneTwistTorque(link.jointRotation, omega, link.jointAxis, link.swingLimit,
                                    link.twistLimit, link.limitStiffness, link.limitDamping);
            }
            if (link.driveStiffness > 0.0f)
            {
                // Spring toward the target orientation (body-frame error) − drive damping.
                t = t +
                    orientationError(link.jointRotation, link.driveTargetRotation) *
                        link.driveStiffness -
                    omega * link.driveDamping;
            }
            jointTorque[0] = t.x();
            jointTorque[1] = t.y();
            jointTorque[2] = t.z();
        }
        else if (link.joint == ArticulationJointType::Revolute && link.driveStiffness > 0.0f)
        {
            jointTorque[0] =
                link.driveStiffness * (link.driveTarget - q_[off]) - link.driveDamping * qDot_[off];
        }
        for (int k = 0; k < nd; ++k)
        {
            const float tau =
                -jointDamping * qDot_[off + static_cast<std::size_t>(k)] + jointTorque[k];
            uForce[i][static_cast<std::size_t>(k)] =
                tau - subspace_[i][static_cast<std::size_t>(k)].dot(bias[i]);
        }
        std::array<float, 3> g{};
        for (int k = 0; k < nd; ++k)
        {
            for (int j = 0; j < nd; ++j)
            {
                g[static_cast<std::size_t>(k)] +=
                    dInv_[i][k, j] * uForce[i][static_cast<std::size_t>(j)];
            }
        }
        SpatialVector pa = bias[i] + ia_[i] * velProd[i];
        for (int k = 0; k < nd; ++k)
        {
            pa = pa + u_[i][static_cast<std::size_t>(k)] * g[static_cast<std::size_t>(k)];
        }
        if (!(p == 0 && rootFixed))
        {
            bias[p] = bias[p] + xforce_[i] * pa;
        }
    }

    // Pass 3 (outward): base acceleration (0, fixed) down to joint accelerations q̈.
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const int nd = link.dofCount;
        const auto off = static_cast<std::size_t>(link.dofOffset);
        const SpatialVector aParent = (p == 0 && rootFixed) ? SpatialVector{} : accel[p];
        const SpatialVector aPrime = xup_[i] * aParent + velProd[i];

        std::array<float, 3> e{};
        for (int j = 0; j < nd; ++j)
        {
            e[static_cast<std::size_t>(j)] = uForce[i][static_cast<std::size_t>(j)] -
                                             u_[i][static_cast<std::size_t>(j)].dot(aPrime);
        }
        accel[i] = aPrime;
        for (int k = 0; k < nd; ++k)
        {
            float qdd = 0.0f;
            for (int j = 0; j < nd; ++j)
            {
                qdd += dInv_[i][k, j] * e[static_cast<std::size_t>(j)];
            }
            qDDot_[off + static_cast<std::size_t>(k)] = qdd;
            accel[i] = accel[i] + subspace_[i][static_cast<std::size_t>(k)] * qdd;
        }
    }
}

void Articulation::computeLinkVelocities()
{
    forwardKinematics();
    const std::size_t n = links_.size();
    linkVelWorld_.assign(n, SpatialVector{}); // fixed base: root velocity 0

    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const SpatialVector& parentVel = linkVelWorld_[p];
        // Transport the parent's spatial velocity to this link's origin (rigid link), then
        // add the joint's own angular rate about the world axis (a revolute joint through the
        // link origin adds no linear velocity there).
        const Vec3 offset = linkWorld_[i].translation - linkWorld_[p].translation;
        Vec3 angular = parentVel.angular;
        if (link.joint == ArticulationJointType::Revolute)
        {
            const Vec3 worldAxis = linkWorld_[i].rotation.rotate(link.jointAxis);
            angular = angular + worldAxis * qDot_[static_cast<std::size_t>(link.dofOffset)];
        }
        const Vec3 linear = parentVel.linear + Vec3::crossProduct(parentVel.angular, offset);
        linkVelWorld_[i] = SpatialVector{angular, linear};
    }
}

Vec3 Articulation::pointVelocity(std::size_t link, const Vec3& worldPoint) const
{
    const SpatialVector& lv = linkVelWorld_[link];
    const Vec3 r = worldPoint - linkWorld_[link].translation;
    return lv.linear + Vec3::crossProduct(lv.angular, r);
}

Vec3 Articulation::impulseResponse(std::size_t link, const Vec3& worldPoint,
                                   const Vec3& worldImpulse, bool commit)
{
    const std::size_t n = links_.size();

    // World impulse → spatial impulse at the link origin, in the link frame.
    const Quaternion& rot = linkWorld_[link].rotation;
    const Vec3 fLink = rot.conjugate().rotate(worldImpulse);
    const Vec3 rLink = rot.conjugate().rotate(worldPoint - linkWorld_[link].translation);
    // Bias force pᴬ = −(applied impulse), matching the acceleration ABA where an external
    // force enters the bias with a minus sign; uForce = −Sᵀpᴬ then carries the right sign.
    std::vector<SpatialVector> bias(n, SpatialVector{});
    bias[link] = SpatialVector{Vec3::crossProduct(rLink, fLink), fLink} * -1.0f;

    // Inward pass: fold the bias toward the root, recording each joint's uForce = −Sᵀpᴬ per DOF.
    std::vector<std::array<float, 3>> uForce(n);
    for (std::size_t i = n - 1; i >= 1; --i)
    {
        const Link& lk = links_[i];
        const auto p = static_cast<std::size_t>(lk.parent);
        const int nd = lk.dofCount;
        for (int k = 0; k < nd; ++k)
        {
            uForce[i][static_cast<std::size_t>(k)] =
                -subspace_[i][static_cast<std::size_t>(k)].dot(bias[i]);
        }
        std::array<float, 3> g{};
        for (int k = 0; k < nd; ++k)
        {
            for (int j = 0; j < nd; ++j)
            {
                g[static_cast<std::size_t>(k)] +=
                    dInv_[i][k, j] * uForce[i][static_cast<std::size_t>(j)];
            }
        }
        SpatialVector pa = bias[i];
        for (int k = 0; k < nd; ++k)
        {
            pa = pa + u_[i][static_cast<std::size_t>(k)] * g[static_cast<std::size_t>(k)];
        }
        if (!(p == 0 && baseFixed_))
        {
            bias[p] = bias[p] + xforce_[i] * pa;
        }
    }

    // Outward pass: base velocity delta (0, fixed) down to the joint velocity deltas Δq̇.
    std::vector<SpatialVector> dv(n, SpatialVector{});
    std::vector<float> dq(static_cast<std::size_t>(dofCount_), 0.0f);
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& lk = links_[i];
        const auto p = static_cast<std::size_t>(lk.parent);
        const int nd = lk.dofCount;
        const auto off = static_cast<std::size_t>(lk.dofOffset);
        const SpatialVector dvParent = (p == 0 && baseFixed_) ? SpatialVector{} : dv[p];
        const SpatialVector dvPrime = xup_[i] * dvParent;

        std::array<float, 3> e{};
        for (int j = 0; j < nd; ++j)
        {
            e[static_cast<std::size_t>(j)] = uForce[i][static_cast<std::size_t>(j)] -
                                             u_[i][static_cast<std::size_t>(j)].dot(dvPrime);
        }
        dv[i] = dvPrime;
        for (int k = 0; k < nd; ++k)
        {
            float delta = 0.0f;
            for (int j = 0; j < nd; ++j)
            {
                delta += dInv_[i][k, j] * e[static_cast<std::size_t>(j)];
            }
            dq[off + static_cast<std::size_t>(k)] = delta;
            dv[i] = dv[i] + subspace_[i][static_cast<std::size_t>(k)] * delta;
        }
    }

    // Resulting world velocity change of the contact point on `link`.
    const SpatialVector& dvL = dv[link];
    const Vec3 dvPointLink = dvL.linear + Vec3::crossProduct(dvL.angular, rLink);
    const Vec3 dvPointWorld = rot.rotate(dvPointLink);

    if (commit)
    {
        for (int k = 0; k < dofCount_; ++k)
        {
            qDot_[static_cast<std::size_t>(k)] += dq[static_cast<std::size_t>(k)];
        }
    }
    return dvPointWorld;
}

float Articulation::inverseEffectiveMass(std::size_t link, const Vec3& worldPoint,
                                         const Vec3& worldDir) const
{
    // A unit impulse along worldDir; the point-velocity response projected back onto worldDir
    // is dᵀ (J M⁻¹ Jᵀ) d = the inverse effective mass. const via a non-committing probe.
    const Vec3 dv =
        const_cast<Articulation*>(this)->impulseResponse(link, worldPoint, worldDir, false);
    return Vec3::dotProduct(dv, worldDir);
}

void Articulation::applyImpulse(std::size_t link, const Vec3& worldPoint, const Vec3& worldImpulse)
{
    impulseResponse(link, worldPoint, worldImpulse, true);
}

void Articulation::integrate(float dt)
{
    // Semi-implicit Euler (velocity first), matching the rigid-body integrator.
    for (int i = 0; i < dofCount_; ++i)
    {
        qDot_[static_cast<std::size_t>(i)] += qDDot_[static_cast<std::size_t>(i)] * dt;
    }
    // Advance each joint's position from its (updated) velocity. A revolute q integrates
    // linearly; a spherical joint's quaternion advances by its angular velocity via the
    // exponential map (stable, re-normalised) since its q̇ *is* the joint-frame angular rate.
    for (Link& link : links_)
    {
        if (link.joint == ArticulationJointType::Revolute)
        {
            const auto off = static_cast<std::size_t>(link.dofOffset);
            q_[off] += qDot_[off] * dt;
        }
        else if (link.joint == ArticulationJointType::Spherical)
        {
            // q̇ is the angular velocity in the *child* (body) frame, so the joint quaternion
            // advances by right multiplication R·Δ (Quaternion::integrate left-multiplies, a
            // world-frame convention — that would drift energy for out-of-plane motion).
            const auto off = static_cast<std::size_t>(link.dofOffset);
            const Vec3 omega{qDot_[off], qDot_[off + 1], qDot_[off + 2]};
            const float angle = omega.magnitude() * dt;
            if (angle > 1e-8f)
            {
                const Vec3 axis = omega * (1.0f / omega.magnitude());
                link.jointRotation = Quaternion::normalise(link.jointRotation *
                                                           Quaternion::fromAxisAngle(axis, angle));
            }
        }
    }
}

} // namespace fire_engine
