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
