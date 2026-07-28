#pragma once

#include <cstdint>
#include <functional>

namespace fire_engine
{

// A process-unique, monotonically allocated scene-node identity.
//
// LIVES IN core/ rather than scene/ on purpose. `Node` remains the identity AUTHORITY — it is the
// non-copyable scene instance, and a component (a copyable `Light`, say) cannot say which instance
// in the scene it belongs to. But the graphics seam (`Lighting`) has to carry the id, and a
// graphics header including a scene header would breach the layering rule transitively: the guard
// script only catches DIRECT forbidden includes, so that violation would have gone unnoticed.
// Putting the primitive here keeps the boundary intact without moving the authority.
//
// The two obvious cheaper handles both alias:
//   - a gathered-array index is traversal order, so adding, removing or reordering a node silently
//     re-points every key past it;
//   - a node POINTER can be recycled by the allocator, so a deleted node's address may reappear
//     as a different node and quietly inherit its history.
// Monotonic ids have neither failure: an id is never reused within the process.
//
// A STRONG type (enum class), not an alias: `NodeId` and `ShadowCasterId` are different identity
// domains, and as bare uint64_t aliases a caster id would compile happily where a light id was
// meant, producing a key that names nothing.
enum class NodeId : std::uint64_t
{
    // No node ever holds this; it marks "no node".
    Invalid = 0,
};

// Allocates the next id. Atomic because scene construction is not contractually single-threaded,
// and a duplicated id is the exact aliasing bug this type exists to prevent. Terminates rather than
// wrapping: a wrapped counter would hand out `Invalid` and then start reusing live identities,
// which corrupts every history keyed on them — unrecoverable, and worse than stopping.
[[nodiscard]] NodeId allocateNodeId() noexcept;

// Owns a node's identity and makes it MOVE-AWARE.
//
// A plain scalar member would be copied by a defaulted move, leaving the moved-from and moved-to
// objects both alive holding the same "unique" id. Here the identity follows the contents and the
// moved-from owner is issued a fresh one, so two live objects can never claim the same identity.
class NodeIdentity
{
public:
    NodeIdentity() noexcept
        : id_{allocateNodeId()}
    {
    }
    ~NodeIdentity() = default;

    // Non-copyable: copying an identity would defeat its purpose outright.
    NodeIdentity(const NodeIdentity&) = delete;
    NodeIdentity& operator=(const NodeIdentity&) = delete;

    NodeIdentity(NodeIdentity&& other) noexcept
        : id_{other.id_}
    {
        other.id_ = allocateNodeId();
    }
    NodeIdentity& operator=(NodeIdentity&& other) noexcept
    {
        if (this != &other)
        {
            id_ = other.id_;
            other.id_ = allocateNodeId();
        }
        return *this;
    }

    [[nodiscard]] NodeId value() const noexcept
    {
        return id_;
    }

private:
    NodeId id_;
};

} // namespace fire_engine

template <>
struct std::hash<fire_engine::NodeId>
{
    [[nodiscard]] std::size_t operator()(fire_engine::NodeId id) const noexcept
    {
        return std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(id));
    }
};
