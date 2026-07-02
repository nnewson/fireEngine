#include <fire_engine/scene/ragdoll.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/articulation.hpp>
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

// Default enforcement of a ragdoll joint's cone-twist limit: a moderate spring (a rigid
// limit would ring under the explicit integrator) plus damping. Uniform for now.
constexpr float kRagdollLimitStiffness = 60.0f;
constexpr float kRagdollLimitDamping = 6.0f;

// Diagonal inertia of a capsule of mass `m`, radius `r`, cylinder half-height `h`, aligned
// with the link's local Y (the collider axis). Rod-plus-radius approximation, floored so a
// near-degenerate bone can't produce a singular inertia.
[[nodiscard]] Vec3 capsuleInertia(float m, float r, float h) noexcept
{
    const float axis = std::max(0.5f * m * r * r, 1.0e-4f);
    const float perp = std::max(m * (h * h / 3.0f + 0.25f * r * r), 1.0e-4f);
    return Vec3{perp, axis, perp};
}

} // namespace

Ragdoll Ragdoll::make(PhysicsWorld& physics, std::span<Node* const> boneNodes,
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

Ragdoll Ragdoll::makeArticulated(PhysicsWorld& physics, std::span<Node* const> boneNodes,
                                 const RagdollParams& params)
{
    Ragdoll rag;
    rag.physics_ = &physics;
    const std::size_t count = boneNodes.size();
    if (count == 0)
    {
        return rag;
    }

    std::unordered_map<const Node*, int> indexOf;
    for (std::size_t i = 0; i < count; ++i)
    {
        indexOf[boneNodes[i]] = static_cast<int>(i);
    }

    // Bind-pose world transform of each bone (the articulation is seeded to reproduce it).
    std::vector<Vec3> pos(count);
    std::vector<Quaternion> rot(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const Mat4& w = boneNodes[i]->composedWorld();
        pos[i] = translation(w);
        rot[i] = Quaternion::fromMatrix(w);
    }

    // Parent bone = nearest ancestor node that is itself a bone.
    std::vector<int> parent(count, -1);
    int root = -1;
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
        if (parent[i] < 0)
        {
            root = root < 0 ? static_cast<int>(i) : root; // first root wins (single-root model)
        }
    }
    if (root < 0)
    {
        return rag; // no root bone
    }

    // Topological order (parents before children): a breadth-first walk from the root so
    // addLink always sees an already-added parent link. Bones not reachable from the root
    // (a second skeleton root) are skipped — the single-root assumption.
    std::vector<std::vector<int>> childrenOf(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        if (parent[i] >= 0)
        {
            childrenOf[static_cast<std::size_t>(parent[i])].push_back(static_cast<int>(i));
        }
    }
    std::vector<int> order{root};
    for (std::size_t qi = 0; qi < order.size(); ++qi)
    {
        for (const int c : childrenOf[static_cast<std::size_t>(order[qi])])
        {
            order.push_back(c);
        }
    }

    const PhysicsArticulationHandle artHandle = physics.createArticulation();
    Articulation* art = physics.articulation(artHandle);
    art->baseFixed(false);

    const auto boneLength = [&](int i)
    {
        float length = params.defaultBoneLength;
        if (parent[static_cast<std::size_t>(i)] >= 0)
        {
            const float span = (pos[static_cast<std::size_t>(i)] -
                                pos[static_cast<std::size_t>(parent[static_cast<std::size_t>(i)])])
                                   .magnitude();
            if (span > 1.0e-4f)
            {
                length = span;
            }
        }
        return length;
    };

    std::vector<int> linkOf(count, -1);
    for (const int b : order)
    {
        const auto bi = static_cast<std::size_t>(b);
        const float halfHeight = 0.5f * boneLength(b);

        ArticulationLinkDesc desc;
        desc.mass = params.mass;
        desc.comLocal = Vec3{}; // capsule centred at the link origin
        desc.inertiaLocal = capsuleInertia(params.mass, params.radius, halfHeight);

        if (b == root)
        {
            art->addRootLink(desc);
            art->baseTransform(RigidTransform{rot[bi], pos[bi]});
            linkOf[bi] = 0;
        }
        else
        {
            const auto pb = static_cast<std::size_t>(parent[bi]);
            desc.parent = linkOf[pb];
            desc.joint = ArticulationJointType::Spherical;
            // Relative parent→child transform so forward kinematics at q = 0 reproduces the
            // bind pose (jointToChild stays identity — the joint sits at the child origin).
            desc.parentToJoint = RigidTransform{rot[pb].conjugate() * rot[bi],
                                                rot[pb].conjugate().rotate(pos[bi] - pos[pb])};
            if (params.coneTwist)
            {
                desc.swingLimit = params.swingLimit;
                desc.twistLimit = params.twistLimit;
                desc.limitStiffness = kRagdollLimitStiffness;
                desc.limitDamping = kRagdollLimitDamping;
            }
            linkOf[bi] = art->addLink(desc);
        }

        ColliderDesc collider;
        collider.shape = CapsuleShape{params.radius, halfHeight, Vec3{}};
        collider.collisionLayer = params.collisionLayer;
        collider.collisionMask = params.collisionMask;
        [[maybe_unused]] const auto lc =
            physics.attachLinkCollider(artHandle, linkOf[bi], collider);
    }
    art->forwardKinematics(); // so linkWorld is valid before the first step / activate

    rag.articulation_ = artHandle;
    for (std::size_t i = 0; i < count; ++i)
    {
        Bone bone;
        bone.node = boneNodes[i];
        bone.parent = parent[i];
        bone.link = linkOf[i];
        rag.bones_.push_back(bone);
    }
    return rag;
}

void Ragdoll::activate()
{
    if (physics_ == nullptr)
    {
        return;
    }
    // Seed each bone node's world-override from its simulated pose, so the skinning path
    // (Skin reads Node::composedWorld) renders the simulated skeleton. An articulated ragdoll
    // reads the link forward-kinematics transforms; a maximal one reads its bodies.
    if (articulated())
    {
        if (const Articulation* art = physics_->articulation(articulation_))
        {
            for (Bone& bone : bones_)
            {
                if (bone.link >= 0)
                {
                    const RigidTransform lw = art->linkWorld(static_cast<std::size_t>(bone.link));
                    bone.node->worldOverride(Mat4::translate(lw.translation) *
                                             lw.rotation.toMat4());
                }
            }
        }
    }
    else
    {
        for (Bone& bone : bones_)
        {
            const auto transform = physics_->bodyTransform(bone.body);
            if (transform.has_value())
            {
                bone.node->worldOverride(transform->world());
            }
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
