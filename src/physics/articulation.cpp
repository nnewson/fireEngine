#include <fire_engine/physics/articulation.hpp>

#include <cassert>
#include <cstddef>

#include <fire_engine/math/quaternion.hpp>

namespace fire_engine
{

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

    link.dofCount = (desc.joint == ArticulationJointType::Revolute) ? 1 : 0;
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

        linkWorld_[i] = parentWorld * link.parentToJoint * jointMotion * link.jointToChild;
    }
}

void Articulation::computeAccelerations(const Vec3& gravity, float jointDamping)
{
    forwardKinematics(); // link world orientations, needed for the per-link gravity wrench

    const std::size_t n = links_.size();

    // Featherstone ABA scratch, indexed by link (0 = root). Everything is expressed in each
    // link's own frame, moved between links by the Plücker transforms Xup (motion,
    // parent→child) and Xforce (force, child→parent).
    std::vector<SpatialMatrix> xup(n);
    std::vector<SpatialMatrix> xforce(n);
    std::vector<SpatialVector> subspace(n); // joint motion subspace S (child frame)
    std::vector<SpatialVector> vel(n);      // spatial velocity v
    std::vector<SpatialVector> velProd(n);  // velocity-product accel c = v ×  (S q̇)
    std::vector<SpatialMatrix> artInertia(n);
    std::vector<SpatialVector> artBias(n); // articulated bias force pᴬ
    std::vector<SpatialVector> u(n);       // U = Iᴬ S
    std::vector<float> d(n, 0.0f);         // D = Sᵀ U
    std::vector<float> uForce(n, 0.0f);    // u = τ − Sᵀ pᴬ
    std::vector<SpatialVector> accel(n);

    const bool rootFixed = baseFixed_;

    // --- Pass 1 (outward): transforms, velocities, link inertias, bias forces. ---
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);

        RigidTransform jointMotion; // rotation about the joint axis by q (identity for Fixed)
        if (link.joint == ArticulationJointType::Revolute)
        {
            jointMotion.rotation = Quaternion::fromAxisAngle(
                link.jointAxis, q_[static_cast<std::size_t>(link.dofOffset)]);
        }
        // Relative parent→child transform. jointToChild is assumed a pure rotation (the axis
        // passes through the child origin), so the subspace stays (axis; 0).
        const RigidTransform t = link.parentToJoint * jointMotion * link.jointToChild;
        xup[i] = motionTransform(t.inverse());
        xforce[i] = forceTransform(t);

        subspace[i] = (link.joint == ArticulationJointType::Revolute)
                          ? SpatialVector{link.jointAxis, Vec3{}}
                          : SpatialVector{};

        const float qDotI =
            link.dofCount > 0 ? qDot_[static_cast<std::size_t>(link.dofOffset)] : 0.0f;
        const SpatialVector vJoint = subspace[i] * qDotI;
        const SpatialVector vParent = (p == 0 && rootFixed) ? SpatialVector{} : vel[p];
        vel[i] = xup[i] * vParent + vJoint;
        velProd[i] = crossMotion(vel[i], vJoint);

        artInertia[i] = spatialInertia(link.mass, link.comLocal, Mat3::diagonal(link.inertiaLocal));

        // Gravity as an external wrench in the link frame: a force m·g at the COM.
        const Vec3 gLink = linkWorld_[i].rotation.conjugate().rotate(gravity);
        const Vec3 gForce = gLink * link.mass;
        const SpatialVector gravityWrench{Vec3::crossProduct(link.comLocal, gForce), gForce};
        artBias[i] = crossForce(vel[i], artInertia[i] * vel[i]) - gravityWrench;
    }

    // --- Pass 2 (inward): articulated inertia + bias, projected onto each parent. ---
    for (std::size_t i = n - 1; i >= 1; --i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const bool toParent = !(p == 0 && rootFixed);

        u[i] = artInertia[i] * subspace[i];
        d[i] = subspace[i].dot(u[i]);
        const float tau = link.dofCount > 0
                              ? -jointDamping * qDot_[static_cast<std::size_t>(link.dofOffset)]
                              : 0.0f;
        uForce[i] = tau - subspace[i].dot(artBias[i]);

        if (d[i] > 1e-9f) // a real (revolute) DOF: rank-1 articulated update
        {
            const SpatialMatrix ia = artInertia[i] - spatialOuter(u[i] * (1.0f / d[i]), u[i]);
            const SpatialVector pa = artBias[i] + ia * velProd[i] + u[i] * (uForce[i] / d[i]);
            if (toParent)
            {
                artInertia[p] = artInertia[p] + xforce[i] * ia * xup[i];
                artBias[p] = artBias[p] + xforce[i] * pa;
            }
        }
        else if (toParent) // a Fixed joint (no DOF): rigidly fold the link onto its parent
        {
            artInertia[p] = artInertia[p] + xforce[i] * artInertia[i] * xup[i];
            artBias[p] = artBias[p] + xforce[i] * (artBias[i] + artInertia[i] * velProd[i]);
        }
    }

    // --- Pass 3 (outward): base acceleration down to joint accelerations. ---
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const SpatialVector aParent = (p == 0 && rootFixed) ? SpatialVector{} : accel[p];
        const SpatialVector aPrime = xup[i] * aParent + velProd[i];
        if (link.dofCount > 0 && d[i] > 1e-9f)
        {
            const float qdd = (uForce[i] - u[i].dot(aPrime)) / d[i];
            qDDot_[static_cast<std::size_t>(link.dofOffset)] = qdd;
            accel[i] = aPrime + subspace[i] * qdd;
        }
        else
        {
            accel[i] = aPrime;
        }
    }
}

void Articulation::factorizeArticulatedInertia()
{
    forwardKinematics();
    const std::size_t n = links_.size();
    xup_.assign(n, SpatialMatrix{});
    xforce_.assign(n, SpatialMatrix{});
    subspace_.assign(n, SpatialVector{});
    artInertia_.assign(n, SpatialMatrix{});
    u_.assign(n, SpatialVector{});
    d_.assign(n, 0.0f);

    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        RigidTransform jointMotion;
        if (link.joint == ArticulationJointType::Revolute)
        {
            jointMotion.rotation = Quaternion::fromAxisAngle(
                link.jointAxis, q_[static_cast<std::size_t>(link.dofOffset)]);
        }
        const RigidTransform t = link.parentToJoint * jointMotion * link.jointToChild;
        xup_[i] = motionTransform(t.inverse());
        xforce_[i] = forceTransform(t);
        subspace_[i] = (link.joint == ArticulationJointType::Revolute)
                           ? SpatialVector{link.jointAxis, Vec3{}}
                           : SpatialVector{};
        artInertia_[i] =
            spatialInertia(link.mass, link.comLocal, Mat3::diagonal(link.inertiaLocal));
    }

    for (std::size_t i = n - 1; i >= 1; --i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const bool toParent = !(p == 0 && baseFixed_);
        u_[i] = artInertia_[i] * subspace_[i];
        d_[i] = subspace_[i].dot(u_[i]);
        if (d_[i] > 1e-9f)
        {
            const SpatialMatrix ia = artInertia_[i] - spatialOuter(u_[i] * (1.0f / d_[i]), u_[i]);
            if (toParent)
            {
                artInertia_[p] = artInertia_[p] + xforce_[i] * ia * xup_[i];
            }
        }
        else if (toParent)
        {
            artInertia_[p] = artInertia_[p] + xforce_[i] * artInertia_[i] * xup_[i];
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

    // Inward pass: fold the bias toward the root, recording each joint's uForce = −Sᵀpᴬ.
    std::vector<float> uForce(n, 0.0f);
    for (std::size_t i = n - 1; i >= 1; --i)
    {
        const auto p = static_cast<std::size_t>(links_[i].parent);
        const bool toParent = !(p == 0 && baseFixed_);
        if (d_[i] > 1e-9f)
        {
            uForce[i] = -subspace_[i].dot(bias[i]);
            const SpatialVector pa = bias[i] + u_[i] * (uForce[i] / d_[i]);
            if (toParent)
            {
                bias[p] = bias[p] + xforce_[i] * pa;
            }
        }
        else if (toParent)
        {
            bias[p] = bias[p] + xforce_[i] * bias[i];
        }
    }

    // Outward pass: base velocity delta (0, fixed) down to the joint velocity deltas Δq̇.
    std::vector<SpatialVector> dv(n, SpatialVector{});
    std::vector<float> dq(static_cast<std::size_t>(dofCount_), 0.0f);
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const SpatialVector dvParent = (p == 0 && baseFixed_) ? SpatialVector{} : dv[p];
        const SpatialVector dvPrime = xup_[i] * dvParent;
        if (link.dofCount > 0 && d_[i] > 1e-9f)
        {
            const float delta = (uForce[i] - u_[i].dot(dvPrime)) / d_[i];
            dq[static_cast<std::size_t>(link.dofOffset)] = delta;
            dv[i] = dvPrime + subspace_[i] * delta;
        }
        else
        {
            dv[i] = dvPrime;
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
        q_[static_cast<std::size_t>(i)] += qDot_[static_cast<std::size_t>(i)] * dt;
    }
}

} // namespace fire_engine
