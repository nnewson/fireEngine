#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/spatial.hpp>

namespace fire_engine
{

// Joint connecting a link to its parent in a reduced-coordinate articulation. Phase A
// supports the two that prove the architecture without quaternion/singularity handling:
//   - Fixed:    0 DOF — the link is rigidly welded to its parent (a multi-segment bone).
//   - Revolute: 1 DOF — rotation by q about a fixed axis (a knee/elbow). Spherical joints
//                (3 DOF, cone-twist) arrive in a later phase.
enum class ArticulationJointType
{
    Fixed,
    Revolute,
};

// Authoring description for one link. The link's body frame is reached from its parent's
// body frame by: parentToJoint (a fixed offset placing the joint anchor on the parent) →
// the joint's own motion (R(q) about jointAxis for Revolute, identity for Fixed) →
// jointToChild (a fixed offset from the joint frame to this link's body frame). `jointAxis`
// is expressed in the joint frame and assumed unit-length. Mass properties (mass /
// inertiaLocal / comLocal) are carried now but consumed by the Phase B dynamics.
struct ArticulationLinkDesc
{
    int parent{-1}; // index of the parent link; the root link uses -1
    ArticulationJointType joint{ArticulationJointType::Fixed};
    Vec3 jointAxis{0.0f, 0.0f, 1.0f};
    RigidTransform parentToJoint{};
    RigidTransform jointToChild{};
    float mass{1.0f};
    Vec3 inertiaLocal{1.0f, 1.0f, 1.0f}; // diagonal principal inertia, link frame
    Vec3 comLocal{};                     // centre of mass in the link frame
};

// A reduced-coordinate articulation: a tree of links whose joints are *parameterized*
// (generalized coordinates q) rather than glued together by constraints. Joint error is
// therefore zero by construction — the failure mode that makes maximal-coordinate ragdoll
// chains limit-cycle simply does not exist here. The root link is a 6-DOF floating base
// (its world pose is baseTransform); every other link adds its joint's DOFs below it.
//
// Phase A is the kinematic foundation: build the tree, set q + the base pose, and run
// forward kinematics to get each link's world transform. The articulated-body dynamics
// (ABA), the contact/limit coupling through a ConstraintBody, and the Ragdoll binding
// build on this in later phases.
class Articulation
{
public:
    Articulation() = default;

    // Add the floating-base root link (index 0). Its body frame *is* the base frame, so
    // parent / joint / parentToJoint on the desc are ignored; mass properties are kept.
    // Must be called exactly once, before any addLink.
    int addRootLink(const ArticulationLinkDesc& desc);

    // Add a non-root link. `desc.parent` must reference an already-added link (a strictly
    // lower index, so the tree is topologically ordered for a single forward sweep).
    // Returns the new link's index. A Revolute joint appends one generalized coordinate.
    int addLink(const ArticulationLinkDesc& desc);

    [[nodiscard]]
    std::size_t linkCount() const noexcept
    {
        return links_.size();
    }

    // Total generalized-coordinate count (sum of per-joint DOFs; Revolute = 1, Fixed = 0).
    [[nodiscard]]
    int dofCount() const noexcept
    {
        return dofCount_;
    }

    // Floating-base pose (world transform of the root link's body frame).
    void baseTransform(const RigidTransform& transform) noexcept
    {
        base_ = transform;
    }

    [[nodiscard]]
    RigidTransform baseTransform() const noexcept
    {
        return base_;
    }

    [[nodiscard]]
    std::span<const float> q() const noexcept
    {
        return q_;
    }

    [[nodiscard]]
    std::span<const float> qDot() const noexcept
    {
        return qDot_;
    }

    // Set generalized coordinate / velocity `dof` (0-based over dofCount()).
    void q(int dof, float value) noexcept;
    void qDot(int dof, float value) noexcept;

    // The generalized-coordinate offset of link `i`'s joint (−1 for a 0-DOF Fixed joint
    // or the root). Lets callers map a link's joint to its slice of q / qDot.
    [[nodiscard]]
    int linkDofOffset(std::size_t i) const noexcept
    {
        return links_[i].dofOffset;
    }

    [[nodiscard]]
    int parent(std::size_t i) const noexcept
    {
        return links_[i].parent;
    }

    // Recompute every link's world transform from the base pose and the current q. Cheap
    // single forward sweep (links are topologically ordered). Call after changing q / base.
    void forwardKinematics();

    // World transform of link `i`, valid after forwardKinematics().
    [[nodiscard]]
    RigidTransform linkWorld(std::size_t i) const noexcept
    {
        return linkWorld_[i];
    }

private:
    struct Link
    {
        int parent{-1};
        ArticulationJointType joint{ArticulationJointType::Fixed};
        Vec3 jointAxis{0.0f, 0.0f, 1.0f};
        RigidTransform parentToJoint{};
        RigidTransform jointToChild{};
        float mass{1.0f};
        Vec3 inertiaLocal{1.0f, 1.0f, 1.0f};
        Vec3 comLocal{};
        int dofOffset{-1}; // offset into q_/qDot_, or −1 for a 0-DOF joint
        int dofCount{0};
    };

    std::vector<Link> links_;
    std::vector<RigidTransform> linkWorld_; // FK output, 1:1 with links_
    RigidTransform base_{};
    std::vector<float> q_;
    std::vector<float> qDot_;
    int dofCount_{0};
};

} // namespace fire_engine
