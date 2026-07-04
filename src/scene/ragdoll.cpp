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

struct BoneFrames
{
    std::vector<Vec3> pos;
    std::vector<Quaternion> rot;
    std::vector<int> parent;
};

[[nodiscard]] BoneFrames gatherBoneFrames(std::span<Node* const> boneNodes)
{
    const std::size_t count = boneNodes.size();

    // Node → bone index, for resolving each bone's parent bone (the nearest ancestor
    // node that is itself a bone).
    std::unordered_map<const Node*, int> indexOf;
    indexOf.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        indexOf[boneNodes[i]] = static_cast<int>(i);
    }

    // World pose of each bone from its current composed world (the bind/animated
    // pose the ragdoll is seeded from).
    BoneFrames frames;
    frames.pos.resize(count);
    frames.rot.resize(count);
    frames.parent.assign(count, -1);
    for (std::size_t i = 0; i < count; ++i)
    {
        const Mat4& w = boneNodes[i]->composedWorld();
        frames.pos[i] = translation(w);
        frames.rot[i] = Quaternion::fromMatrix(w);
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        for (const Node* p = boneNodes[i]->parent(); p != nullptr; p = p->parent())
        {
            const auto it = indexOf.find(p);
            if (it != indexOf.end())
            {
                frames.parent[i] = it->second;
                break;
            }
        }
    }

    return frames;
}

// Squared distance between two segments [p1,q1] and [p2,q2] (Ericson, Real-Time Collision
// Detection). Used to detect which bone capsules overlap in the bind pose.
[[nodiscard]] float segmentSegmentDistanceSq(const Vec3& p1, const Vec3& q1, const Vec3& p2,
                                             const Vec3& q2) noexcept
{
    const Vec3 d1 = q1 - p1;
    const Vec3 d2 = q2 - p2;
    const Vec3 r = p1 - p2;
    const float a = Vec3::dotProduct(d1, d1);
    const float e = Vec3::dotProduct(d2, d2);
    const float f = Vec3::dotProduct(d2, r);
    float s = 0.0f;
    float t = 0.0f;
    constexpr float eps = 1e-8f;
    if (a <= eps && e <= eps)
    {
        return Vec3::dotProduct(r, r); // both degenerate to points
    }
    if (a <= eps)
    {
        t = std::clamp(f / e, 0.0f, 1.0f);
    }
    else
    {
        const float c = Vec3::dotProduct(d1, r);
        if (e <= eps)
        {
            s = std::clamp(-c / a, 0.0f, 1.0f);
        }
        else
        {
            const float b = Vec3::dotProduct(d1, d2);
            const float denom = a * e - b * b;
            s = denom > eps ? std::clamp((b * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f)
            {
                t = 0.0f;
                s = std::clamp(-c / a, 0.0f, 1.0f);
            }
            else if (t > 1.0f)
            {
                t = 1.0f;
                s = std::clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    const Vec3 c1 = p1 + d1 * s;
    const Vec3 c2 = p2 + d2 * t;
    const Vec3 diff = c1 - c2;
    return Vec3::dotProduct(diff, diff);
}

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
    const BoneFrames frames = gatherBoneFrames(boneNodes);
    const std::vector<Vec3>& pos = frames.pos;
    const std::vector<Quaternion>& rot = frames.rot;
    const std::vector<int>& parent = frames.parent;

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

    const BoneFrames frames = gatherBoneFrames(boneNodes);
    const std::vector<Vec3>& pos = frames.pos;
    const std::vector<Quaternion>& rot = frames.rot;
    const std::vector<int>& parent = frames.parent;

    int root = -1;
    for (std::size_t i = 0; i < count; ++i)
    {
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
    std::vector<Vec3> capA(count); // capsule world endpoints (this joint …
    std::vector<Vec3> capB(count); // … back toward the parent), for the self-collision exclusion
    for (const int b : order)
    {
        const auto bi = static_cast<std::size_t>(b);
        const float halfHeight = 0.5f * boneLength(b);

        // This bone's own axis (joint→parent, the segment the capsule spans) expressed in the
        // link's local frame. Defaults to +Y; set from geometry for a non-root bone. The capsule
        // and the cone-twist twist axis both align to this — the rig's bones rarely run along the
        // link's local Y, so without it the capsule sticks out sideways (self-collision can't stop
        // the fold-through) and the limits are oriented arbitrarily (the skeleton collapses).
        Vec3 boneAxisLocal{0.0f, 1.0f, 0.0f};

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
            const Vec3 boneDirWorld = pos[bi] - pos[pb];
            if (boneDirWorld.magnitudeSquared() > 1.0e-8f)
            {
                boneAxisLocal = rot[bi].conjugate().rotate(Vec3::normalise(boneDirWorld));
            }
            desc.jointAxis = boneAxisLocal;
            if (params.coneTwist)
            {
                desc.swingLimit = params.swingLimit;
                desc.twistLimit = params.twistLimit;
            }
            linkOf[bi] = art->addLink(desc);
        }

        ColliderDesc collider;
        // Orient the capsule along the bone axis (its local Y rotated onto boneAxisLocal) so it
        // covers the actual limb; a capsule left on local Y would stick out sideways and miss it.
        collider.shape = CapsuleShape{params.radius, halfHeight, Vec3{}};
        collider.localRotation = Quaternion::fromVectors(Vec3{0.0f, 1.0f, 0.0f}, boneAxisLocal);
        // Articulated bones self-collide (so limbs stack into a plausible pose instead of folding
        // through the torso): they pair with each other in the broadphase; the articulation's
        // self-collision gather skips adjacent (parent-child) bones, which always overlap at their
        // shared joint. Layer kept so external masking still applies; mask opened to all.
        collider.collisionLayer = params.collisionLayer;
        collider.collisionMask = ~0U;
        [[maybe_unused]] const auto lc =
            physics.attachLinkCollider(artHandle, linkOf[bi], collider);

        // World endpoints of this capsule (straddling the joint along the bone axis by
        // ±halfHeight), matching the collider above, for the bind-pose overlap exclusion below.
        const Vec3 axWorld = rot[bi].rotate(boneAxisLocal) * halfHeight;
        capA[bi] = pos[bi] - axWorld;
        capB[bi] = pos[bi] + axWorld;
    }

    // Exclude self-collision between bones whose capsules already overlap in the bind pose
    // (structurally adjacent — arm-root vs torso, the two hip bones, …). Colliding those would
    // blast the skeleton apart on the first step; only bones that come together *later* (a
    // folding limb onto the torso) should generate self-contacts.
    for (std::size_t a = 0; a < count; ++a)
    {
        if (linkOf[a] < 0)
        {
            continue;
        }
        for (std::size_t b = a + 1; b < count; ++b)
        {
            if (linkOf[b] < 0)
            {
                continue;
            }
            const float d2 = segmentSegmentDistanceSq(capA[a], capB[a], capA[b], capB[b]);
            const float reach = 2.0f * params.radius + 0.02f; // both radii + a small margin
            if (d2 < reach * reach)
            {
                art->excludeSelfCollision(static_cast<std::size_t>(linkOf[a]),
                                          static_cast<std::size_t>(linkOf[b]));
            }
        }
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

void Ragdoll::syncNodes()
{
    if (physics_ == nullptr)
    {
        return;
    }
    // Push each bone node's world-override from its current simulated pose, so the skinning path
    // (Skin reads Node::composedWorld) renders the simulated skeleton. An articulated ragdoll
    // reads the link forward-kinematics transforms; a maximal one reads its bodies. Maximal bones
    // are also body-bound, so SceneGraph::applyPhysics keeps them in sync — but an articulated
    // ragdoll's bones are NOT body-bound, so this must run every frame after the physics step.
    if (articulated())
    {
        const Articulation* art = physics_->articulation(articulation_);
        if (art == nullptr)
        {
            return;
        }
        for (Bone& bone : bones_)
        {
            if (bone.link >= 0)
            {
                const RigidTransform lw = art->linkWorld(static_cast<std::size_t>(bone.link));
                bone.node->worldOverride(Mat4::translate(lw.translation) * lw.rotation.toMat4());
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
}

void Ragdoll::activate()
{
    syncNodes(); // seed the world-overrides from the initial simulated pose
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
