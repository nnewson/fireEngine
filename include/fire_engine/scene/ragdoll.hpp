#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/physics/physics_handle.hpp>

namespace fire_engine
{

class Node;
class PhysicsWorld;

// Tuning for an auto-built ragdoll. One capsule body per bone (radius `radius`,
// length from the bone-to-parent span or `defaultBoneLength` for a root/leaf), with
// a ball-socket joint to the parent bone. `coneTwist` adds a swing-cone + twist
// angular limit so the joints articulate like a skeleton rather than a free pivot.
struct RagdollParams
{
    float mass{1.0f};
    float radius{0.05f};
    float defaultBoneLength{0.2f};
    bool coneTwist{true};
    float swingLimit{0.7f}; // cone half-angle (radians, ~40°)
    float twistLimit{0.5f}; // ± twist (radians)
    // Bones share one collision layer and mask it out of their own collisions, so
    // adjacent (overlapping) capsules don't fight the joints — but still collide
    // with everything else (floors, props). Default: bit 1, all-but-self.
    std::uint32_t collisionLayer{1U << 1};
    std::uint32_t collisionMask{~(1U << 1)};
};

// Binds a skinned skeleton (a set of bone `Node`s — typically a Skin's joint nodes)
// to physics: a body + collider per bone and a joint linking each bone to its parent
// bone, seeded from the bones' current world pose. `activate()` hands the bones over
// to physics by setting their world-override, so the existing skinning path (Skin
// reads Node::composedWorld) renders the simulated pose; SceneGraph::applyPhysics
// keeps the overrides in sync each step. `deactivate()` returns them to animation.
class Ragdoll
{
public:
    Ragdoll() = default;

    [[nodiscard]]
    static Ragdoll make(PhysicsWorld& physics, std::span<Node* const> bones,
                        const RagdollParams& params = {});

    void activate();
    void deactivate();

    [[nodiscard]]
    bool active() const noexcept
    {
        return active_;
    }

    [[nodiscard]]
    std::size_t boneCount() const noexcept
    {
        return bones_.size();
    }

    [[nodiscard]]
    PhysicsBodyHandle body(std::size_t index) const noexcept
    {
        return bones_[index].body;
    }

    [[nodiscard]]
    PhysicsConstraintHandle joint(std::size_t index) const noexcept
    {
        return bones_[index].joint;
    }

    [[nodiscard]]
    const Node* node(std::size_t index) const noexcept
    {
        return bones_[index].node;
    }

private:
    struct Bone
    {
        Node* node{nullptr};
        PhysicsBodyHandle body;
        PhysicsConstraintHandle joint; // to the parent bone (invalid for a root)
        int parent{-1};                // index into bones_, or -1
    };

    PhysicsWorld* physics_{nullptr};
    std::vector<Bone> bones_;
    bool active_{false};
};

} // namespace fire_engine
