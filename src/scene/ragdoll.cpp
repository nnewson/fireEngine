#include <fire_engine/scene/ragdoll.hpp>

#include <unordered_map>

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/collider_shape.hpp>
#include <fire_engine/physics/joint.hpp>
#include <fire_engine/physics/physics_body.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/node.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] Vec3 translation(const Mat4& m) noexcept
{
    return {m[0, 3], m[1, 3], m[2, 3]};
}

} // namespace

Ragdoll Ragdoll::make(PhysicsWorld& physics, const std::vector<Node*>& boneNodes,
                      const RagdollParams& params)
{
    Ragdoll rag;
    rag.physics_ = &physics;

    const std::size_t count = boneNodes.size();

    // Node → bone index, for resolving each bone's parent bone (the nearest ancestor
    // node that is itself a bone).
    std::unordered_map<const Node*, int> indexOf;
    for (std::size_t i = 0; i < count; ++i)
    {
        indexOf[boneNodes[i]] = static_cast<int>(i);
    }

    // World pose of each bone from its current composed world (the bind/animated
    // pose the ragdoll is seeded from).
    std::vector<Vec3> pos(count);
    std::vector<Quaternion> rot(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const Mat4& w = boneNodes[i]->composedWorld();
        pos[i] = translation(w);
        rot[i] = Quaternion::fromMatrix(w);
    }

    std::vector<int> parent(count, -1);
    for (std::size_t i = 0; i < count; ++i)
    {
        for (const Node* p = boneNodes[i]->parent(); p != nullptr; p = p->parent())
        {
            const auto it = indexOf.find(p);
            if (it != indexOf.end())
            {
                parent[i] = it->second;
                break;
            }
        }
    }

    // Pass 1: a capsule body per bone. Capsule length spans the bone-to-parent gap
    // (falling back to the default for roots / coincident joints).
    for (std::size_t i = 0; i < count; ++i)
    {
        Bone bone;
        bone.node = boneNodes[i];
        bone.parent = parent[i];

        PhysicsBodyDesc bodyDesc;
        bodyDesc.type = PhysicsBodyType::Dynamic;
        bodyDesc.position = pos[i];
        bodyDesc.rotation = rot[i];
        bodyDesc.mass = params.mass;
        bodyDesc.gravityScale = 1.0f;
        bone.body = physics.createBody(bodyDesc);
        // Bind the node to its body so SceneGraph::applyPhysics keeps the bone's
        // world-override synced to the simulated pose each step.
        bone.node->physicsBodyHandle(bone.body);

        float length = params.defaultBoneLength;
        if (parent[i] >= 0)
        {
            const float span = (pos[i] - pos[static_cast<std::size_t>(parent[i])]).magnitude();
            if (span > 1.0e-4f)
            {
                length = span;
            }
        }

        ColliderDesc colliderDesc;
        colliderDesc.shape = CapsuleShape{params.radius, 0.5f * length, Vec3{}};
        colliderDesc.collisionLayer = params.collisionLayer;
        colliderDesc.collisionMask = params.collisionMask;
        [[maybe_unused]] const auto collider = physics.createCollider(bone.body, colliderDesc);

        rag.bones_.push_back(bone);
    }

    // Pass 2: a ball-socket (optionally cone-twist limited) joint pinning each bone's
    // origin to its parent bone. Bodies all exist now, so a parent that appears after
    // its child in the list still resolves.
    for (std::size_t i = 0; i < count; ++i)
    {
        const int pa = parent[i];
        if (pa < 0)
        {
            continue;
        }
        const auto pai = static_cast<std::size_t>(pa);

        const Quaternion parentRot = rot[pai];
        const Vec3 pivotInParent = parentRot.conjugate().rotate(pos[i] - pos[pai]);

        JointDesc jointDesc;
        jointDesc.type = JointType::BallSocket;
        jointDesc.bodyA = rag.bones_[pai].body;
        jointDesc.bodyB = rag.bones_[i].body;
        jointDesc.anchorA = pivotInParent; // joint pivot on the parent
        jointDesc.anchorB = Vec3{};        // child's centre of mass is the pivot

        // Twist axis = the bone direction, expressed in the parent's local frame.
        Vec3 dir = pos[i] - pos[pai];
        if (dir.magnitudeSquared() > 1.0e-8f)
        {
            jointDesc.axisA = parentRot.conjugate().rotate(Vec3::normalise(dir));
        }

        if (params.coneTwist)
        {
            jointDesc.limits.coneTwist = true;
            jointDesc.limits.swingLimit = params.swingLimit;
            jointDesc.limits.twistLimit = params.twistLimit;
        }

        rag.bones_[i].joint = physics.createJoint(jointDesc);
    }

    return rag;
}

void Ragdoll::activate()
{
    if (physics_ == nullptr)
    {
        return;
    }
    // Seed each bone node's world-override from its body's current world pose, so the
    // skinning path (Skin reads Node::composedWorld) renders the simulated skeleton.
    for (Bone& bone : bones_)
    {
        const auto transform = physics_->bodyTransform(bone.body);
        if (transform.has_value())
        {
            bone.node->worldOverride(transform->world());
        }
    }
    active_ = true;
}

void Ragdoll::deactivate()
{
    for (Bone& bone : bones_)
    {
        bone.node->clearWorldOverride();
    }
    active_ = false;
}

} // namespace fire_engine
