#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

#include <fire_engine/collision/narrow_phase.hpp>
#include <fire_engine/collision/sweep_and_prune_broad_phase.hpp>
#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/physics/collider_shape.hpp>
#include <fire_engine/physics/contact.hpp>
#include <fire_engine/physics/physics_body.hpp>
#include <fire_engine/physics/physics_handle.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

class PhysicsWorld
{
public:
    PhysicsWorld() = default;
    ~PhysicsWorld() = default;

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;
    PhysicsWorld(PhysicsWorld&&) noexcept = default;
    PhysicsWorld& operator=(PhysicsWorld&&) noexcept = default;

    [[nodiscard]]
    PhysicsBodyHandle createBody(const PhysicsBodyDesc& desc);

    [[nodiscard]]
    PhysicsColliderHandle createCollider(PhysicsBodyHandle bodyHandle, const ColliderDesc& desc);

    [[nodiscard]]
    bool destroyBody(PhysicsBodyHandle handle);

    void clear();
    void step(float fixedDt);

    [[nodiscard]]
    bool valid(PhysicsBodyHandle handle) const noexcept;

    [[nodiscard]]
    bool valid(PhysicsColliderHandle handle) const noexcept;

    [[nodiscard]]
    std::size_t bodyCount() const noexcept;

    [[nodiscard]]
    std::size_t colliderCount() const noexcept;

    // World-space collision primitives for the cloth/soft-body solver — each
    // active collider's shape composed with its body transform. Vulkan-free
    // output (mirrors the gatherLights / gatherEmitters pattern).
    [[nodiscard]]
    std::vector<ClothCollider> gatherColliders() const;

    [[nodiscard]]
    const PhysicsBody* body(PhysicsBodyHandle handle) const noexcept;

    [[nodiscard]]
    PhysicsBody* body(PhysicsBodyHandle handle) noexcept;

    [[nodiscard]]
    std::optional<Transform> bodyTransform(PhysicsBodyHandle handle) const noexcept;

    void setBodyTransform(PhysicsBodyHandle handle, const Transform& transform) noexcept;
    void setBodyVelocity(PhysicsBodyHandle handle, Vec3 velocity) noexcept;

    [[nodiscard]]
    const std::vector<CollisionPair>& possiblePairs() const noexcept
    {
        return broadPhase_.possiblePairs();
    }

    [[nodiscard]]
    bool validateBroadPhase() const
    {
        return broadPhase_.validate();
    }

private:
    struct BodyEntry
    {
        PhysicsBodyHandle handle;
        PhysicsBody body;
        Transform transform;
        Vec3 previousPosition{};
        bool active{true};
        std::vector<PhysicsColliderHandle> colliders;
    };

    struct ColliderEntry
    {
        PhysicsColliderHandle handle;
        PhysicsBodyHandle body;
        Collider collider;
        ColliderShape shape;
        PhysicsMaterial material;
        bool active{true};
    };

    struct ContactCandidate
    {
        BodyEntry* moving{nullptr};
        BodyEntry* target{nullptr};
        ColliderEntry* movingCollider{nullptr};
        ColliderEntry* targetCollider{nullptr};
    };

    struct SolverContact
    {
        float toi{0.0f};
        Vec3 normal{};
        BodyEntry* moving{nullptr};
        BodyEntry* target{nullptr};
        ColliderEntry* movingCollider{nullptr};
        ColliderEntry* targetCollider{nullptr};
    };

    PhysicsBodyHandle nextBodyHandle_{PhysicsBodyHandle{1U}};
    PhysicsColliderHandle nextColliderHandle_{PhysicsColliderHandle{1U}};
    std::vector<BodyEntry> bodies_;
    std::deque<ColliderEntry> colliders_;
    // Side-tables for O(1) lookup into the entry containers, keyed by handle
    // value and (for the broadphase's pair pointers) by Collider address.
    // Entries are tombstoned, never erased before clear(), so the stored indices
    // stay valid for the container's lifetime; find* still checks `active`.
    std::unordered_map<std::uint32_t, std::size_t> bodyIndexByHandle_;
    std::unordered_map<std::uint32_t, std::size_t> colliderIndexByHandle_;
    std::unordered_map<const Collider*, std::size_t> colliderIndexByPointer_;
    SweepAndPruneBroadPhase broadPhase_;
    NarrowPhase narrowPhase_;
    Vec3 gravity_{0.0f, -9.81f, 0.0f};

    [[nodiscard]]
    BodyEntry* findBody(PhysicsBodyHandle handle) noexcept;

    [[nodiscard]]
    const BodyEntry* findBody(PhysicsBodyHandle handle) const noexcept;

    [[nodiscard]]
    ColliderEntry* findCollider(PhysicsColliderHandle handle) noexcept;

    [[nodiscard]]
    const ColliderEntry* findCollider(PhysicsColliderHandle handle) const noexcept;

    [[nodiscard]]
    ColliderEntry* findCollider(const Collider* collider) noexcept;

    [[nodiscard]]
    AABB localBounds(const ColliderShape& shape) const noexcept;

    void updateCollider(ColliderEntry& collider);
    void resetCollider(ColliderEntry& collider);
    void updateColliders();
    void resetResolvedColliders();
    void capturePreviousPositions() noexcept;

    [[nodiscard]]
    std::vector<SolverContact> contacts();

    [[nodiscard]]
    std::optional<SolverContact> contactForPair(const CollisionPair& pair);

    [[nodiscard]]
    std::optional<ContactCandidate> contactCandidateForPair(const CollisionPair& pair);

    [[nodiscard]]
    bool applyResponses(std::vector<SolverContact>& contacts);

    [[nodiscard]]
    static bool movable(const BodyEntry& body) noexcept;

    [[nodiscard]]
    static bool movedThisFrame(const BodyEntry& body) noexcept;

    [[nodiscard]]
    static Vec3 frameDelta(const BodyEntry& body) noexcept;

    void moveToFrameTime(BodyEntry& body, float toi) noexcept;
    void slideFrameMovement(BodyEntry& body, float toi, Vec3 normal) noexcept;
};

} // namespace fire_engine
