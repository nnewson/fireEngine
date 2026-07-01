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

} // namespace fire_engine
