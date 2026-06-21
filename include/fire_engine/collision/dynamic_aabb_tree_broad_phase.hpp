#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <fire_engine/collision/broad_phase.hpp>

namespace fire_engine
{

// Dynamic AABB tree (fat-AABB BVH) broadphase. Each collider is a leaf whose stored
// AABB is its swept world bounds enlarged by a margin, so small motion needs no
// re-insert; internal nodes union their children. Insertion uses a surface-area
// heuristic and the tree is kept balanced with rotations (à la Box2D's b2DynamicTree).
//
// Pairs are regenerated each update()/rebuild() by querying every leaf against the
// tree (precise swept-bounds overlap + layer/mask filter at the leaves) and sorting
// by collider id, so the possible-pair list is deterministic regardless of tree shape.
class DynamicAabbTreeBroadPhase final : public BroadPhase
{
public:
    DynamicAabbTreeBroadPhase() = default;
    ~DynamicAabbTreeBroadPhase() override = default;

    DynamicAabbTreeBroadPhase(const DynamicAabbTreeBroadPhase&) = delete;
    DynamicAabbTreeBroadPhase& operator=(const DynamicAabbTreeBroadPhase&) = delete;
    DynamicAabbTreeBroadPhase(DynamicAabbTreeBroadPhase&&) noexcept = default;
    DynamicAabbTreeBroadPhase& operator=(DynamicAabbTreeBroadPhase&&) noexcept = default;

    ColliderId addCollider(Collider& collider) override;

    [[nodiscard]]
    bool removeCollider(ColliderId colliderId) override;

    [[nodiscard]]
    bool removeCollider(Collider& collider) override;

    void clear() override;
    void update() override;
    void rebuild() override;

    [[nodiscard]]
    const std::vector<CollisionPair>& possiblePairs() const noexcept override
    {
        return possiblePairs_;
    }

    [[nodiscard]]
    bool validate() const override;

    [[nodiscard]]
    std::size_t colliderCount() const noexcept override
    {
        return leafByColliderId_.size();
    }

private:
    static constexpr int kNull = -1;

    struct Node
    {
        AABB box{};
        int parent{kNull};
        int child1{kNull};
        int child2{kNull};
        int height{0}; // 0 for a leaf, -1 for a free-list node
        ColliderId id{};
        Collider* collider{nullptr}; // leaf only

        [[nodiscard]]
        bool isLeaf() const noexcept
        {
            return child1 == kNull;
        }
    };

    [[nodiscard]] int allocateNode();
    void freeNode(int node) noexcept;
    void insertLeaf(int leaf);
    void removeLeaf(int leaf) noexcept;
    [[nodiscard]] int balance(int node);
    void refitAndBalance(int node);
    void regeneratePairs();
    void queryLeaf(int leaf);

    std::vector<Node> nodes_;
    int root_{kNull};
    int freeList_{kNull};
    ColliderId nextColliderId_{ColliderId{1U}};
    // Collider id value → leaf node index. Only ever looked up by key (never iterated
    // to produce the pair list, which is sorted), so its ordering can't leak out.
    std::unordered_map<std::uint32_t, int> leafByColliderId_;
    std::vector<CollisionPair> possiblePairs_;
    std::vector<int> queryStack_; // reused traversal stack
};

} // namespace fire_engine
