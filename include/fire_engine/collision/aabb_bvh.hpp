#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <fire_engine/collision/aabb.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Smallest AABB enclosing both inputs.
[[nodiscard]] inline AABB aabbUnion(const AABB& a, const AABB& b) noexcept
{
    return AABB{{std::min(a.min.x(), b.min.x()), std::min(a.min.y(), b.min.y()),
                 std::min(a.min.z(), b.min.z())},
                {std::max(a.max.x(), b.max.x()), std::max(a.max.y(), b.max.y()),
                 std::max(a.max.z(), b.max.z())}};
}

// Surface-area proxy (half-perimeter sum) used by the BVH insertion heuristic.
[[nodiscard]] inline float aabbPerimeter(const AABB& a) noexcept
{
    const Vec3 d = a.max - a.min;
    return 2.0f * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
}

// Inclusive overlap on all three axes.
[[nodiscard]] inline bool aabbOverlaps(const AABB& a, const AABB& b) noexcept
{
    return a.min.x() <= b.max.x() && a.max.x() >= b.min.x() && a.min.y() <= b.max.y() &&
           a.max.y() >= b.min.y() && a.min.z() <= b.max.z() && a.max.z() >= b.min.z();
}

// True when `inner` is fully inside `outer`.
[[nodiscard]] inline bool aabbContains(const AABB& outer, const AABB& inner) noexcept
{
    return outer.min.x() <= inner.min.x() && outer.min.y() <= inner.min.y() &&
           outer.min.z() <= inner.min.z() && outer.max.x() >= inner.max.x() &&
           outer.max.y() >= inner.max.y() && outer.max.z() >= inner.max.z();
}

// Generic dynamic AABB bounding-volume hierarchy (fat-AABB BVH). Each leaf carries a
// payload `T`; internal nodes union their children. Insertion uses a surface-area
// heuristic and the tree is kept balanced with AVL-style rotations (à la Box2D's
// `b2DynamicTree`). Leaves are addressed by stable `int` proxy handles. Leaf boxes are
// fattened by `margin` on creation/move, so small motion keeps a proxy inside its leaf
// box and needs no re-insert. Supports both incremental use (a dynamic broadphase) and
// build-all-then-query (a static triangle mesh, with margin 0).
template <typename T>
class AabbBvh
{
public:
    static constexpr int kNull = -1;

    explicit AabbBvh(float margin = 0.1f) noexcept
        : margin_{margin}
    {
    }

    // Create a leaf for `payload` whose tight bounds are `box` (fattened internally by
    // the margin). Returns a stable proxy handle.
    [[nodiscard]] int createProxy(const AABB& box, const T& payload)
    {
        const int leaf = allocateNode();
        nodes_[static_cast<std::size_t>(leaf)].box = fatten(box);
        nodes_[static_cast<std::size_t>(leaf)].height = 0;
        nodes_[static_cast<std::size_t>(leaf)].payload = payload;
        insertLeaf(leaf);
        ++leafCount_;
        return leaf;
    }

    void destroyProxy(int proxy)
    {
        removeLeaf(proxy);
        freeNode(proxy);
        --leafCount_;
    }

    // Re-fit a moved proxy to tight bounds `box`. A cheap no-op while `box` stays inside
    // the leaf's stored (fat) box; otherwise re-inserts with a freshly fattened box.
    // Returns true if a re-insert happened.
    bool moveProxy(int proxy, const AABB& box)
    {
        if (aabbContains(nodes_[static_cast<std::size_t>(proxy)].box, box))
        {
            return false;
        }
        removeLeaf(proxy);
        nodes_[static_cast<std::size_t>(proxy)].box = fatten(box);
        insertLeaf(proxy);
        return true;
    }

    [[nodiscard]] const AABB& fatBounds(int proxy) const noexcept
    {
        return nodes_[static_cast<std::size_t>(proxy)].box;
    }

    [[nodiscard]] const T& payload(int proxy) const noexcept
    {
        return nodes_[static_cast<std::size_t>(proxy)].payload;
    }

    // Invoke `fn(int proxy)` for every leaf whose fat box overlaps `box`.
    template <typename Fn>
    void query(const AABB& box, Fn&& fn) const
    {
        if (root_ == kNull)
        {
            return;
        }
        queryStack_.clear();
        queryStack_.push_back(root_);
        while (!queryStack_.empty())
        {
            const int n = queryStack_.back();
            queryStack_.pop_back();
            if (n == kNull || !aabbOverlaps(nodes_[static_cast<std::size_t>(n)].box, box))
            {
                continue;
            }
            const Node& node = nodes_[static_cast<std::size_t>(n)];
            if (node.isLeaf())
            {
                fn(n);
            }
            else
            {
                queryStack_.push_back(node.child1);
                queryStack_.push_back(node.child2);
            }
        }
    }

    void clear() noexcept
    {
        nodes_.clear();
        root_ = kNull;
        freeList_ = kNull;
        leafCount_ = 0;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return leafCount_;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return leafCount_ == 0;
    }

private:
    struct Node
    {
        AABB box{};
        int parent{kNull};
        int child1{kNull};
        int child2{kNull};
        int height{0}; // 0 for a leaf, -1 for a free-list node
        T payload{};

        [[nodiscard]] bool isLeaf() const noexcept
        {
            return child1 == kNull;
        }
    };

    [[nodiscard]] AABB fatten(const AABB& a) const noexcept
    {
        const Vec3 m{margin_, margin_, margin_};
        return AABB{a.min - m, a.max + m};
    }

    [[nodiscard]] int allocateNode()
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

    void freeNode(int node) noexcept
    {
        nodes_[static_cast<std::size_t>(node)] = Node{};
        nodes_[static_cast<std::size_t>(node)].child1 = freeList_;
        nodes_[static_cast<std::size_t>(node)].height = -1;
        freeList_ = node;
    }

    void insertLeaf(int leaf)
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

            const float area = aabbPerimeter(node.box);
            const float combinedArea = aabbPerimeter(aabbUnion(node.box, leafBox));

            const float cost = 2.0f * combinedArea;
            const float inheritanceCost = 2.0f * (combinedArea - area);

            const auto descendCost = [&](int child)
            {
                const AABB merged = aabbUnion(leafBox, nodes_[static_cast<std::size_t>(child)].box);
                if (nodes_[static_cast<std::size_t>(child)].isLeaf())
                {
                    return aabbPerimeter(merged) + inheritanceCost;
                }
                const float oldArea = aabbPerimeter(nodes_[static_cast<std::size_t>(child)].box);
                return (aabbPerimeter(merged) - oldArea) + inheritanceCost;
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
            aabbUnion(leafBox, nodes_[static_cast<std::size_t>(sibling)].box);
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

    void removeLeaf(int leaf) noexcept
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

    void refitAndBalance(int node)
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
                aabbUnion(nodes_[static_cast<std::size_t>(child1)].box,
                          nodes_[static_cast<std::size_t>(child2)].box);
            index = nodes_[static_cast<std::size_t>(index)].parent;
        }
    }

    // AVL-style rotation: if `node`'s child heights differ by > 1, rotate the taller
    // grandchild up. Returns the (possibly new) subtree root.
    int balance(int node)
    {
        Node& a = nodes_[static_cast<std::size_t>(node)];
        if (a.isLeaf() || a.height < 2)
        {
            return node;
        }

        const int iB = a.child1;
        const int iC = a.child2;
        const int balanceFactor = nodes_[static_cast<std::size_t>(iC)].height -
                                  nodes_[static_cast<std::size_t>(iB)].height;

        const auto rotateUp = [&](int iLow, int iHigh)
        {
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

            const int taller = nodes_[static_cast<std::size_t>(iF)].height >
                                       nodes_[static_cast<std::size_t>(iG)].height
                                   ? iF
                                   : iG;
            const int shorter = taller == iF ? iG : iF;

            high.child2 = taller;
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
                aabbUnion(nodes_[static_cast<std::size_t>(iLow)].box,
                          nodes_[static_cast<std::size_t>(shorter)].box);
            high.box = aabbUnion(nodes_[static_cast<std::size_t>(node)].box,
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
            return rotateUp(iB, iC);
        }
        if (balanceFactor < -1)
        {
            return rotateUp(iC, iB);
        }
        return node;
    }

    float margin_;
    std::vector<Node> nodes_;
    int root_{kNull};
    int freeList_{kNull};
    std::size_t leafCount_{0};
    mutable std::vector<int> queryStack_;
};

} // namespace fire_engine
