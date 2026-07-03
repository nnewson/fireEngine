#include <fire_engine/physics/articulation.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

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

namespace
{
[[nodiscard]] std::uint64_t selfCollisionKey(std::size_t a, std::size_t b) noexcept
{
    const std::uint64_t lo = a < b ? a : b;
    const std::uint64_t hi = a < b ? b : a;
    return (lo << 20) | hi;
}
} // namespace

void Articulation::excludeSelfCollision(std::size_t a, std::size_t b)
{
    selfCollisionExcluded_.insert(selfCollisionKey(a, b));
}

bool Articulation::selfCollisionExcluded(std::size_t a, std::size_t b) const
{
    return selfCollisionExcluded_.count(selfCollisionKey(a, b)) != 0;
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

    // The root's own spatial inertia seeds artInertia_[0]; the inward pass folds the children
    // onto it, so the floating-base solve inverts the full articulated inertia of the tree.
    artInertia_[0] =
        spatialInertia(links_[0].mass, links_[0].comLocal, Mat3::diagonal(links_[0].inertiaLocal));

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

    // Cache the floating-base articulated-inertia inverse: it is constant for this factorization
    // but the base solve (a₀ = −Iᴬ₀⁻¹·pᴬ₀) runs on every impulse response — inverting the 6×6 per
    // call dominated the solve. One inverse per substep instead of thousands.
    baseInertiaInv_ = baseFixed_ ? SpatialMatrix{} : artInertia_[0].inverse();
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
    std::vector<std::array<float, 3>>& uForce = scratchUForce_;
    uForce.assign(n, std::array<float, 3>{});

    // Floating base: seed the root with its own spatial velocity + gravity/velocity-product
    // bias (Pass 2 folds the children's bias in). Fixed base leaves everything zero.
    if (!rootFixed)
    {
        const Link& root = links_[0];
        vel[0] = baseVel_;
        const SpatialMatrix i0 =
            spatialInertia(root.mass, root.comLocal, Mat3::diagonal(root.inertiaLocal));
        const Vec3 gBase = base_.rotation.conjugate().rotate(gravity);
        const Vec3 gForce = gBase * root.mass;
        const SpatialVector gravityWrench{Vec3::crossProduct(root.comLocal, gForce), gForce};
        bias[0] = crossForce(vel[0], i0 * vel[0]) - gravityWrench;
    }

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
        vel[i] = xup_[i] * vel[p] + vJoint; // vel[0] is 0 (fixed) or baseVel_ (floating)
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

        // Passive joint torque per DOF: global damping + drive spring. Cone-twist LIMITS are no
        // longer a torque here — a stiff limit spring on a low-inertia (small-bone) joint blows
        // up under explicit Euler; limits are now a velocity-level constraint (solveJointLimits),
        // solved with the contacts.
        float jointTorque[3]{0.0f, 0.0f, 0.0f};
        if (link.joint == ArticulationJointType::Spherical)
        {
            const Vec3 omega{qDot_[off], qDot_[off + 1], qDot_[off + 2]};
            Vec3 t{};
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
        bias[p] = bias[p] + xforce_[i] * pa; // p may be the floating root (bias[0])
    }

    // Base acceleration: the free root carries no constraint force, so Iᴬ₀·a₀ + pᴬ₀ = 0 ⇒
    // a₀ = −Iᴬ₀⁻¹·pᴬ₀ (the 6×6 solve). A fixed base stays at rest.
    if (!rootFixed)
    {
        accel[0] = baseInertiaInv_ * (bias[0] * -1.0f);
    }
    baseAccel_ = accel[0];

    // Pass 3 (outward): base acceleration down to joint accelerations q̈.
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& link = links_[i];
        const auto p = static_cast<std::size_t>(link.parent);
        const int nd = link.dofCount;
        const auto off = static_cast<std::size_t>(link.dofOffset);
        const SpatialVector aPrime = xup_[i] * accel[p] + velProd[i]; // accel[0] set above

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
    if (!baseFixed_)
    {
        // baseVel_ is in the base (body) frame; rotate to world (linear is the base origin's
        // velocity). The child transport below then carries it down the chain.
        linkVelWorld_[0] = SpatialVector{base_.rotation.rotate(baseVel_.angular),
                                         base_.rotation.rotate(baseVel_.linear)};
    }

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
        const auto off = static_cast<std::size_t>(link.dofOffset);
        if (link.joint == ArticulationJointType::Revolute)
        {
            const Vec3 worldAxis = linkWorld_[i].rotation.rotate(link.jointAxis);
            angular = angular + worldAxis * qDot_[off];
        }
        else if (link.joint == ArticulationJointType::Spherical)
        {
            // The spherical joint's three velocities are the joint angular rate in the child
            // frame — rotate to world and add. (Omitting this dropped the whole spherical
            // joint's motion from pointVelocity, so the contact solve read stale velocities.)
            const Vec3 jointOmega{qDot_[off], qDot_[off + 1], qDot_[off + 2]};
            angular = angular + linkWorld_[i].rotation.rotate(jointOmega);
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
    std::vector<SpatialVector>& bias = scratchBias_;
    bias.assign(n, SpatialVector{});
    bias[link] = SpatialVector{Vec3::crossProduct(rLink, fLink), fLink} * -1.0f;

    // Inward pass: fold the bias toward the root, recording each joint's uForce = −Sᵀpᴬ per DOF.
    std::vector<std::array<float, 3>>& uForce = scratchUForce_;
    uForce.assign(n, std::array<float, 3>{});
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
        bias[p] = bias[p] + xforce_[i] * pa; // p may be the floating root (bias[0])
    }

    // Base velocity delta: a floating root responds Δv₀ = −Iᴬ₀⁻¹·Y₀ to the folded impulse.
    std::vector<SpatialVector>& dv = scratchDv_;
    dv.assign(n, SpatialVector{});
    if (!baseFixed_)
    {
        dv[0] = baseInertiaInv_ * (bias[0] * -1.0f);
    }

    // Outward pass: base velocity delta down to the joint velocity deltas Δq̇.
    std::vector<float>& dq = scratchDq_;
    dq.assign(static_cast<std::size_t>(dofCount_), 0.0f);
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& lk = links_[i];
        const auto p = static_cast<std::size_t>(lk.parent);
        const int nd = lk.dofCount;
        const auto off = static_cast<std::size_t>(lk.dofOffset);
        const SpatialVector dvPrime = xup_[i] * dv[p]; // dv[0] set above (0 when fixed)

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
        baseVel_ = baseVel_ + dv[0]; // floating base absorbs its share of the impulse

        // Fold this impulse's per-link velocity change into the cached world link velocities so
        // an *iterated* solve sees it immediately — without recomputing from qDot via
        // computeLinkVelocities(), whose classical transport differs from this Plücker response
        // and drifts under repeated intra-substep refresh. dv[i] is the link-frame spatial
        // change at the link origin; rotate to world (same frame linkVelWorld_ is stored in).
        if (linkVelWorld_.size() == n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                const Quaternion& r = linkWorld_[i].rotation;
                linkVelWorld_[i].angular = linkVelWorld_[i].angular + r.rotate(dv[i].angular);
                linkVelWorld_[i].linear = linkVelWorld_[i].linear + r.rotate(dv[i].linear);
            }
        }
    }
    baseDeltaVel_ = dv[0];
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

Vec3 Articulation::jointImpulseResponse(std::size_t link, const Vec3& genImpulse, bool commit)
{
    // Response to a *generalized* impulse on link `link`'s spherical DOFs (the joint-limit path),
    // as opposed to a point impulse on a link body. Same M⁻¹ articulated machinery as
    // impulseResponse, but the impulse enters as uForce at `link` (a generalized force) with no
    // point bias. Returns the resulting joint-rate change Δq̇ on `link`'s 3 DOFs; a floating base
    // recoils, exactly conserving momentum (so clamping a joint rate stays physical).
    const std::size_t n = links_.size();
    const Link& target = links_[link];
    if (target.joint != ArticulationJointType::Spherical)
    {
        return Vec3{};
    }
    const auto targetOff = static_cast<std::size_t>(target.dofOffset);

    std::vector<SpatialVector>& bias = scratchBias_;
    bias.assign(n, SpatialVector{});
    std::vector<std::array<float, 3>>& uForce = scratchUForce_;
    uForce.assign(n, std::array<float, 3>{});
    for (std::size_t i = n - 1; i >= 1; --i)
    {
        const Link& lk = links_[i];
        const auto p = static_cast<std::size_t>(lk.parent);
        const int nd = lk.dofCount;
        for (int k = 0; k < nd; ++k)
        {
            const float ext = (i == link) ? (k == 0   ? genImpulse.x()
                                             : k == 1 ? genImpulse.y()
                                                      : genImpulse.z())
                                          : 0.0f;
            uForce[i][static_cast<std::size_t>(k)] =
                ext - subspace_[i][static_cast<std::size_t>(k)].dot(bias[i]);
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
        bias[p] = bias[p] + xforce_[i] * pa;
    }

    std::vector<SpatialVector>& dv = scratchDv_;
    dv.assign(n, SpatialVector{});
    if (!baseFixed_)
    {
        dv[0] = baseInertiaInv_ * (bias[0] * -1.0f);
    }

    std::vector<float>& dq = scratchDq_;
    dq.assign(static_cast<std::size_t>(dofCount_), 0.0f);
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& lk = links_[i];
        const auto p = static_cast<std::size_t>(lk.parent);
        const int nd = lk.dofCount;
        const auto off = static_cast<std::size_t>(lk.dofOffset);
        const SpatialVector dvPrime = xup_[i] * dv[p];
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

    const Vec3 dOmega{dq[targetOff], dq[targetOff + 1], dq[targetOff + 2]};
    if (commit)
    {
        for (int k = 0; k < dofCount_; ++k)
        {
            qDot_[static_cast<std::size_t>(k)] += dq[static_cast<std::size_t>(k)];
        }
        baseVel_ = baseVel_ + dv[0];
        if (linkVelWorld_.size() == n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                const Quaternion& r = linkWorld_[i].rotation;
                linkVelWorld_[i].angular = linkVelWorld_[i].angular + r.rotate(dv[i].angular);
                linkVelWorld_[i].linear = linkVelWorld_[i].linear + r.rotate(dv[i].linear);
            }
        }
    }
    return dOmega;
}

Vec3 Articulation::pairImpulseResponse(std::size_t linkA, const Vec3& worldPointA,
                                       std::size_t linkB, const Vec3& worldPointB,
                                       const Vec3& worldImpulse, bool commit)
{
    // Response to an equal-and-opposite impulse pair — +worldImpulse at linkA's point,
    // −worldImpulse at linkB's point — through one M⁻¹ pass. The self-collision primitive:
    // a contact between two links of the *same* articulation. Returns the change in the RELATIVE
    // point velocity (Δ(vA − vB)); a floating base conserves momentum (the pair is internal).
    const std::size_t n = links_.size();

    // Contact offsets in each link's frame (used both to seed the bias and to read the response).
    const Quaternion& rotA = linkWorld_[linkA].rotation;
    const Quaternion& rotB = linkWorld_[linkB].rotation;
    const Vec3 rLinkA = rotA.conjugate().rotate(worldPointA - linkWorld_[linkA].translation);
    const Vec3 rLinkB = rotB.conjugate().rotate(worldPointB - linkWorld_[linkB].translation);

    std::vector<SpatialVector>& bias = scratchBias_;
    bias.assign(n, SpatialVector{});
    const auto seed =
        [&](std::size_t link, const Quaternion& rot, const Vec3& rLink, const Vec3& imp)
    {
        const Vec3 fLink = rot.conjugate().rotate(imp);
        bias[link] = bias[link] + SpatialVector{Vec3::crossProduct(rLink, fLink), fLink} * -1.0f;
    };
    seed(linkA, rotA, rLinkA, worldImpulse);
    seed(linkB, rotB, rLinkB, worldImpulse * -1.0f);

    std::vector<std::array<float, 3>>& uForce = scratchUForce_;
    uForce.assign(n, std::array<float, 3>{});
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
        bias[p] = bias[p] + xforce_[i] * pa;
    }

    std::vector<SpatialVector>& dv = scratchDv_;
    dv.assign(n, SpatialVector{});
    if (!baseFixed_)
    {
        dv[0] = baseInertiaInv_ * (bias[0] * -1.0f);
    }

    std::vector<float>& dq = scratchDq_;
    dq.assign(static_cast<std::size_t>(dofCount_), 0.0f);
    for (std::size_t i = 1; i < n; ++i)
    {
        const Link& lk = links_[i];
        const auto p = static_cast<std::size_t>(lk.parent);
        const int nd = lk.dofCount;
        const auto off = static_cast<std::size_t>(lk.dofOffset);
        const SpatialVector dvPrime = xup_[i] * dv[p];
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

    // Resulting world velocity change of each contact point, and their difference.
    const auto pointVel = [&](std::size_t link, const Quaternion& rot, const Vec3& rLink)
    {
        const SpatialVector& d = dv[link];
        return rot.rotate(d.linear + Vec3::crossProduct(d.angular, rLink));
    };
    const Vec3 dvRel = pointVel(linkA, rotA, rLinkA) - pointVel(linkB, rotB, rLinkB);

    if (commit)
    {
        for (int k = 0; k < dofCount_; ++k)
        {
            qDot_[static_cast<std::size_t>(k)] += dq[static_cast<std::size_t>(k)];
        }
        baseVel_ = baseVel_ + dv[0];
        if (linkVelWorld_.size() == n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                const Quaternion& r = linkWorld_[i].rotation;
                linkVelWorld_[i].angular = linkVelWorld_[i].angular + r.rotate(dv[i].angular);
                linkVelWorld_[i].linear = linkVelWorld_[i].linear + r.rotate(dv[i].linear);
            }
        }
    }
    return dvRel;
}

void Articulation::solveJointLimits(float invH, bool useBias, float erp, float maxPush)
{
    for (std::size_t i = 1; i < links_.size(); ++i)
    {
        const Link& link = links_[i];
        if (link.joint != ArticulationJointType::Spherical)
        {
            continue;
        }
        if (!(link.swingLimit < pi || link.twistLimit < pi)) // no limit configured
        {
            continue;
        }
        const auto off = static_cast<std::size_t>(link.dofOffset);
        const Quaternion q = link.jointRotation;
        const Vec3 twistAxis = link.jointAxis;

        // Unilateral velocity-level limit along `axis` (child frame) violated by `excess`: an
        // impulse through the articulated response drives the joint rate along `axis` down to a
        // bounded push-back target (−min(excess·erp/h, maxPush) when biased, 0 on relax), but
        // only if it is currently increasing the violation faster than that.
        const auto applyLimit = [&](const Vec3& axis, float excess)
        {
            const Vec3 omega{qDot_[off], qDot_[off + 1], qDot_[off + 2]};
            const float cRate = Vec3::dotProduct(omega, axis);
            const float target = useBias ? -std::min(excess * erp * invH, maxPush) : 0.0f;
            if (cRate <= target)
            {
                return;
            }
            const float invEff = Vec3::dotProduct(axis, jointImpulseResponse(i, axis, false));
            if (invEff <= 1.0e-9f)
            {
                return;
            }
            jointImpulseResponse(i, axis * ((target - cRate) / invEff), true);
        };

        // Swing-twist decomposition about the twist axis: q = swing · twist.
        const float d = q.x() * twistAxis.x() + q.y() * twistAxis.y() + q.z() * twistAxis.z();
        const Quaternion twist = Quaternion::normalise(
            Quaternion{twistAxis.x() * d, twistAxis.y() * d, twistAxis.z() * d, q.w()});
        const Quaternion swing = q * twist.conjugate();

        const Vec3 swingVec{swing.x(), swing.y(), swing.z()};
        const float swingSin = swingVec.magnitude();
        if (swingSin > 1.0e-5f)
        {
            const float swingAngle = 2.0f * std::atan2(swingSin, swing.w());
            if (swingAngle > link.swingLimit)
            {
                applyLimit(swingVec * (1.0f / swingSin), swingAngle - link.swingLimit);
            }
        }

        const float twistAngle = 2.0f * std::atan2(d, q.w());
        if (twistAngle > link.twistLimit)
        {
            applyLimit(twistAxis, twistAngle - link.twistLimit);
        }
        else if (twistAngle < -link.twistLimit)
        {
            applyLimit(twistAxis * -1.0f, -link.twistLimit - twistAngle);
        }
    }
}

void Articulation::integrateVelocities(float dt)
{
    // Semi-implicit Euler, velocity half: advance generalized + base velocities from the
    // accelerations. Split from the position half so a TGS contact solve can run *between* them
    // (bias velocity applied here → moved into position by integratePositions → relaxed away).
    for (int i = 0; i < dofCount_; ++i)
    {
        qDot_[static_cast<std::size_t>(i)] += qDDot_[static_cast<std::size_t>(i)] * dt;
    }
    if (!baseFixed_)
    {
        baseVel_ = baseVel_ + baseAccel_ * dt;
    }
}

void Articulation::integratePositions(float dt)
{
    // Position half: advance the base pose and each joint's position from the current velocity.
    if (!baseFixed_)
    {
        // baseVel_ is body-frame: the origin translates by R·v, the orientation right-multiplies.
        const Vec3 worldLinear = base_.rotation.rotate(baseVel_.linear);
        base_.translation = base_.translation + worldLinear * dt;
        const float angle = baseVel_.angular.magnitude() * dt;
        if (angle > 1e-8f)
        {
            const Vec3 axis = baseVel_.angular * (1.0f / baseVel_.angular.magnitude());
            base_.rotation =
                Quaternion::normalise(base_.rotation * Quaternion::fromAxisAngle(axis, angle));
        }
    }
    // A revolute q integrates linearly; a spherical joint's quaternion advances by its angular
    // velocity via the exponential map (stable, re-normalised) since its q̇ *is* the joint-frame
    // angular rate, right-multiplied (a world-frame left-multiply drifts energy out-of-plane).
    for (Link& link : links_)
    {
        if (link.joint == ArticulationJointType::Revolute)
        {
            const auto off = static_cast<std::size_t>(link.dofOffset);
            q_[off] += qDot_[off] * dt;
        }
        else if (link.joint == ArticulationJointType::Spherical)
        {
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

void Articulation::integrate(float dt)
{
    // Convenience for callers without a contact solve (free dynamics / tests): the two halves
    // back-to-back are the original semi-implicit Euler step.
    integrateVelocities(dt);
    integratePositions(dt);
}

} // namespace fire_engine
