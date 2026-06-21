#include <fire_engine/collision/dynamic_aabb_tree_broad_phase.hpp>

#include <algorithm>

namespace fire_engine
{

namespace
{

// Fat-AABB margin (metres): leaves are enlarged by this so a small move keeps the
// collider inside its leaf box and avoids a re-insert. Matches Box2D's aabbExtension.
constexpr float kAabbMargin = 0.1f;

[[nodiscard]] AABB combine(const AABB& a, const AABB& b) noexcept
{
    return AABB{{std::min(a.min.x(), b.min.x()), std::min(a.min.y(), b.min.y()),
                 std::min(a.min.z(), b.min.z())},
                {std::max(a.max.x(), b.max.x()), std::max(a.max.y(), b.max.y()),
                 std::max(a.max.z(), b.max.z())}};
}

// Surface-area proxy (half-perimeter sum) used by the insertion heuristic.
[[nodiscard]] float perimeter(const AABB& a) noexcept
{
    const Vec3 d = a.max - a.min;
    return 2.0f * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
}

[[nodiscard]] bool overlaps(const AABB& a, const AABB& b) noexcept
{
    return a.min.x() <= b.max.x() && a.max.x() >= b.min.x() && a.min.y() <= b.max.y() &&
           a.max.y() >= b.min.y() && a.min.z() <= b.max.z() && a.max.z() >= b.min.z();
}

// True when `inner` is fully inside `outer` (used to decide whether a moved leaf has
// left its fat box and must be re-inserted).
[[nodiscard]] bool contains(const AABB& outer, const AABB& inner) noexcept
{
    return outer.min.x() <= inner.min.x() && outer.min.y() <= inner.min.y() &&
           outer.min.z() <= inner.min.z() && outer.max.x() >= inner.max.x() &&
           outer.max.y() >= inner.max.y() && outer.max.z() >= inner.max.z();
}

[[nodiscard]] AABB fatten(const AABB& a) noexcept
{
    const Vec3 margin{kAabbMargin, kAabbMargin, kAabbMargin};
    return AABB{a.min - margin, a.max + margin};
}

[[nodiscard]] bool canCollide(const Collider& a, const Collider& b) noexcept
{
    return (a.collisionMask() & b.collisionLayer()) != 0U &&
           (b.collisionMask() & a.collisionLayer()) != 0U;
}

} // namespace

int DynamicAabbTreeBroadPhase::allocateNode()
{
    int node = 0;
    if (freeList_ != kNull)
    {
        node = freeList_;
        freeList_ = nodes_[static_cast<std::size_t>(node)].child1;
        nodes_[static_cast<std::size_t>(node)] = Node{};
    }
    else
    {
        node = static_cast<int>(nodes_.size());
        nodes_.emplace_back();
    }
    return node;
}

void DynamicAabbTreeBroadPhase::freeNode(int node) noexcept
{
    nodes_[static_cast<std::size_t>(node)] = Node{};
    nodes_[static_cast<std::size_t>(node)].child1 = freeList_;
    nodes_[static_cast<std::size_t>(node)].height = -1;
    freeList_ = node;
}

void DynamicAabbTreeBroadPhase::insertLeaf(int leaf)
{
    if (root_ == kNull)
    {
        root_ = leaf;
        nodes_[static_cast<std::size_t>(leaf)].parent = kNull;
        return;
    }

    // Descend from the root to the best sibling by the surface-area heuristic.
    const AABB leafBox = nodes_[static_cast<std::size_t>(leaf)].box;
    int index = root_;
    while (!nodes_[static_cast<std::size_t>(index)].isLeaf())
    {
        const Node& node = nodes_[static_cast<std::size_t>(index)];
        const int child1 = node.child1;
        const int child2 = node.child2;

        const float area = perimeter(node.box);
        const float combinedArea = perimeter(combine(node.box, leafBox));

        // Cost of creating a new parent here.
        const float cost = 2.0f * combinedArea;
        // Minimum cost of pushing the leaf further down.
        const float inheritanceCost = 2.0f * (combinedArea - area);

        const auto descendCost = [&](int child)
        {
            const AABB merged = combine(leafBox, nodes_[static_cast<std::size_t>(child)].box);
            if (nodes_[static_cast<std::size_t>(child)].isLeaf())
            {
                return perimeter(merged) + inheritanceCost;
            }
            const float oldArea = perimeter(nodes_[static_cast<std::size_t>(child)].box);
            return (perimeter(merged) - oldArea) + inheritanceCost;
        };

        const float cost1 = descendCost(child1);
        const float cost2 = descendCost(child2);

        if (cost < cost1 && cost < cost2)
        {
            break;
        }
        index = cost1 < cost2 ? child1 : child2;
    }

    const int sibling = index;
    const int oldParent = nodes_[static_cast<std::size_t>(sibling)].parent;
    const int newParent = allocateNode();
    nodes_[static_cast<std::size_t>(newParent)].parent = oldParent;
    nodes_[static_cast<std::size_t>(newParent)].box =
        combine(leafBox, nodes_[static_cast<std::size_t>(sibling)].box);
    nodes_[static_cast<std::size_t>(newParent)].height =
        nodes_[static_cast<std::size_t>(sibling)].height + 1;
    nodes_[static_cast<std::size_t>(newParent)].child1 = sibling;
    nodes_[static_cast<std::size_t>(newParent)].child2 = leaf;
    nodes_[static_cast<std::size_t>(sibling)].parent = newParent;
    nodes_[static_cast<std::size_t>(leaf)].parent = newParent;

    if (oldParent != kNull)
    {
        if (nodes_[static_cast<std::size_t>(oldParent)].child1 == sibling)
        {
            nodes_[static_cast<std::size_t>(oldParent)].child1 = newParent;
        }
        else
        {
            nodes_[static_cast<std::size_t>(oldParent)].child2 = newParent;
        }
    }
    else
    {
        root_ = newParent;
    }

    refitAndBalance(nodes_[static_cast<std::size_t>(leaf)].parent);
}

void DynamicAabbTreeBroadPhase::removeLeaf(int leaf) noexcept
{
    if (leaf == root_)
    {
        root_ = kNull;
        return;
    }

    const int parent = nodes_[static_cast<std::size_t>(leaf)].parent;
    const int grandParent = nodes_[static_cast<std::size_t>(parent)].parent;
    const int sibling = nodes_[static_cast<std::size_t>(parent)].child1 == leaf
                            ? nodes_[static_cast<std::size_t>(parent)].child2
                            : nodes_[static_cast<std::size_t>(parent)].child1;

    if (grandParent != kNull)
    {
        if (nodes_[static_cast<std::size_t>(grandParent)].child1 == parent)
        {
            nodes_[static_cast<std::size_t>(grandParent)].child1 = sibling;
        }
        else
        {
            nodes_[static_cast<std::size_t>(grandParent)].child2 = sibling;
        }
        nodes_[static_cast<std::size_t>(sibling)].parent = grandParent;
        freeNode(parent);
        refitAndBalance(grandParent);
    }
    else
    {
        root_ = sibling;
        nodes_[static_cast<std::size_t>(sibling)].parent = kNull;
        freeNode(parent);
    }
}

void DynamicAabbTreeBroadPhase::refitAndBalance(int node)
{
    int index = node;
    while (index != kNull)
    {
        index = balance(index);
        const int child1 = nodes_[static_cast<std::size_t>(index)].child1;
        const int child2 = nodes_[static_cast<std::size_t>(index)].child2;
        nodes_[static_cast<std::size_t>(index)].height =
            1 + std::max(nodes_[static_cast<std::size_t>(child1)].height,
                         nodes_[static_cast<std::size_t>(child2)].height);
        nodes_[static_cast<std::size_t>(index)].box =
            combine(nodes_[static_cast<std::size_t>(child1)].box,
                    nodes_[static_cast<std::size_t>(child2)].box);
        index = nodes_[static_cast<std::size_t>(index)].parent;
    }
}

// AVL-style rotation: if `node` is unbalanced (child heights differ by > 1), rotate
// the taller grandchild up. Returns the (possibly new) subtree root.
int DynamicAabbTreeBroadPhase::balance(int node)
{
    Node& a = nodes_[static_cast<std::size_t>(node)];
    if (a.isLeaf() || a.height < 2)
    {
        return node;
    }

    const int iB = a.child1;
    const int iC = a.child2;
    const int balanceFactor =
        nodes_[static_cast<std::size_t>(iC)].height - nodes_[static_cast<std::size_t>(iB)].height;

    const auto rotateUp = [&](int iLow, int iHigh)
    {
        // Promote iHigh above node; reattach node's other child as iHigh's lighter child.
        Node& high = nodes_[static_cast<std::size_t>(iHigh)];
        const int iF = high.child1;
        const int iG = high.child2;

        high.child1 = node;
        high.parent = nodes_[static_cast<std::size_t>(node)].parent;
        nodes_[static_cast<std::size_t>(node)].parent = iHigh;

        const int parent = high.parent;
        if (parent != kNull)
        {
            if (nodes_[static_cast<std::size_t>(parent)].child1 == node)
            {
                nodes_[static_cast<std::size_t>(parent)].child1 = iHigh;
            }
            else
            {
                nodes_[static_cast<std::size_t>(parent)].child2 = iHigh;
            }
        }
        else
        {
            root_ = iHigh;
        }

        // Pick the taller of iHigh's children to keep as its child2; the other becomes
        // node's replacement child.
        const int taller = nodes_[static_cast<std::size_t>(iF)].height >
                                   nodes_[static_cast<std::size_t>(iG)].height
                               ? iF
                               : iG;
        const int shorter = taller == iF ? iG : iF;

        high.child2 = taller;
        // `iLow` is node's other (untouched) child, kept in place.
        if (nodes_[static_cast<std::size_t>(node)].child1 == iHigh)
        {
            nodes_[static_cast<std::size_t>(node)].child1 = shorter;
        }
        else
        {
            nodes_[static_cast<std::size_t>(node)].child2 = shorter;
        }
        nodes_[static_cast<std::size_t>(shorter)].parent = node;

        nodes_[static_cast<std::size_t>(node)].box =
            combine(nodes_[static_cast<std::size_t>(iLow)].box,
                    nodes_[static_cast<std::size_t>(shorter)].box);
        high.box = combine(nodes_[static_cast<std::size_t>(node)].box,
                           nodes_[static_cast<std::size_t>(taller)].box);
        nodes_[static_cast<std::size_t>(node)].height =
            1 + std::max(nodes_[static_cast<std::size_t>(iLow)].height,
                         nodes_[static_cast<std::size_t>(shorter)].height);
        high.height = 1 + std::max(nodes_[static_cast<std::size_t>(node)].height,
                                   nodes_[static_cast<std::size_t>(taller)].height);
        return iHigh;
    };

    if (balanceFactor > 1)
    {
        return rotateUp(iB, iC); // right child heavier
    }
    if (balanceFactor < -1)
    {
        return rotateUp(iC, iB); // left child heavier
    }
    return node;
}

ColliderId DynamicAabbTreeBroadPhase::addCollider(Collider& collider)
{
    const ColliderId id = nextColliderId_;
    nextColliderId_ = ColliderId{nextColliderId_.value() + 1U};
    collider.colliderId(id);

    const int leaf = allocateNode();
    nodes_[static_cast<std::size_t>(leaf)].box = fatten(collider.sweptWorldBounds());
    nodes_[static_cast<std::size_t>(leaf)].height = 0;
    nodes_[static_cast<std::size_t>(leaf)].id = id;
    nodes_[static_cast<std::size_t>(leaf)].collider = &collider;
    insertLeaf(leaf);
    leafByColliderId_.emplace(id.value(), leaf);
    return id;
}

bool DynamicAabbTreeBroadPhase::removeCollider(ColliderId colliderId)
{
    const auto it = leafByColliderId_.find(colliderId.value());
    if (it == leafByColliderId_.end())
    {
        return false;
    }
    const int leaf = it->second;
    removeLeaf(leaf);
    freeNode(leaf);
    leafByColliderId_.erase(it);
    return true;
}

bool DynamicAabbTreeBroadPhase::removeCollider(Collider& collider)
{
    return removeCollider(collider.colliderId());
}

void DynamicAabbTreeBroadPhase::clear()
{
    nodes_.clear();
    root_ = kNull;
    freeList_ = kNull;
    nextColliderId_ = ColliderId{1U};
    leafByColliderId_.clear();
    possiblePairs_.clear();
}

void DynamicAabbTreeBroadPhase::update()
{
    // Re-insert any leaf whose collider has moved out of its fat box, then refresh the
    // pair list. Leaf node indices are stable across re-insertion, so iterating the
    // (key→leaf) map is safe.
    for (const auto& [idValue, leaf] : leafByColliderId_)
    {
        const AABB tight = nodes_[static_cast<std::size_t>(leaf)].collider->sweptWorldBounds();
        if (!contains(nodes_[static_cast<std::size_t>(leaf)].box, tight))
        {
            removeLeaf(leaf);
            nodes_[static_cast<std::size_t>(leaf)].box = fatten(tight);
            insertLeaf(leaf);
        }
    }
    regeneratePairs();
}

void DynamicAabbTreeBroadPhase::rebuild()
{
    // Rebuild every leaf box from current bounds (re-inserting), then the pair list.
    for (const auto& [idValue, leaf] : leafByColliderId_)
    {
        removeLeaf(leaf);
        nodes_[static_cast<std::size_t>(leaf)].box =
            fatten(nodes_[static_cast<std::size_t>(leaf)].collider->sweptWorldBounds());
        insertLeaf(leaf);
    }
    regeneratePairs();
}

void DynamicAabbTreeBroadPhase::queryLeaf(int leaf)
{
    const Node& leafNode = nodes_[static_cast<std::size_t>(leaf)];
    const Collider* a = leafNode.collider;

    queryStack_.clear();
    if (root_ != kNull)
    {
        queryStack_.push_back(root_);
    }
    while (!queryStack_.empty())
    {
        const int n = queryStack_.back();
        queryStack_.pop_back();
        if (n == kNull || !overlaps(nodes_[static_cast<std::size_t>(n)].box, leafNode.box))
        {
            continue;
        }
        const Node& node = nodes_[static_cast<std::size_t>(n)];
        if (node.isLeaf())
        {
            // Emit each unordered pair once (id strictly greater) and only for a real
            // swept-bounds overlap that passes the layer/mask filter.
            if (leafNode.id < node.id &&
                overlaps(a->sweptWorldBounds(), node.collider->sweptWorldBounds()) &&
                canCollide(*a, *node.collider))
            {
                possiblePairs_.push_back(CollisionPair{leafNode.id, node.id, a, node.collider});
            }
        }
        else
        {
            queryStack_.push_back(node.child1);
            queryStack_.push_back(node.child2);
        }
    }
}

void DynamicAabbTreeBroadPhase::regeneratePairs()
{
    possiblePairs_.clear();
    for (const auto& [idValue, leaf] : leafByColliderId_)
    {
        queryLeaf(leaf);
    }
    // Sort by (firstId, secondId) so the list is independent of tree shape / map order.
    std::ranges::sort(possiblePairs_,
                      [](const CollisionPair& lhs, const CollisionPair& rhs)
                      {
                          if (lhs.firstId.value() != rhs.firstId.value())
                          {
                              return lhs.firstId.value() < rhs.firstId.value();
                          }
                          return lhs.secondId.value() < rhs.secondId.value();
                      });
}

bool DynamicAabbTreeBroadPhase::validate() const
{
    // Brute-force O(n²): every swept-bounds-overlapping, collidable pair must appear
    // exactly once in the maintained list, and vice versa.
    std::vector<Collider*> leaves;
    leaves.reserve(leafByColliderId_.size());
    for (const auto& [idValue, leaf] : leafByColliderId_)
    {
        leaves.push_back(nodes_[static_cast<std::size_t>(leaf)].collider);
    }

    std::size_t expected = 0;
    for (std::size_t i = 0; i < leaves.size(); ++i)
    {
        for (std::size_t j = i + 1; j < leaves.size(); ++j)
        {
            if (overlaps(leaves[i]->sweptWorldBounds(), leaves[j]->sweptWorldBounds()) &&
                canCollide(*leaves[i], *leaves[j]))
            {
                ++expected;
            }
        }
    }
    return expected == possiblePairs_.size();
}

} // namespace fire_engine
