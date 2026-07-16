#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/vertex.hpp>

// Shared mesh-topology / render-wedge primitives. glTF splits a corner into a render *wedge* per
// attribute (UV / normal / tangent) seam, so the same 3D position appears as several vertices. The
// simplifier, VIPM, and VDPM all collapse on **position-welded** topology and restore per-corner
// wedges at emit — and they must agree on both, or their recorded collapses stop lining up with
// each other's adjacency. These are the single source of truth for that agreement; they used to be
// copy-pasted into each module (and tests), which is exactly where the seam / chart bugs clustered.
namespace fire_engine::mesh_topology
{

// Original vertex -> canonical: the FIRST vertex at each EXACT (bit-for-bit) position.
// Deterministic, so every consumer produces identical canonical ids. A canonical id is itself an
// original vertex index (the first at its position), so it also indexes the original vertex array.
[[nodiscard]] std::vector<std::uint32_t> weldByPosition(std::span<const Vertex> vertices);

// Render-attribute distance between two wedges (UV0, UV1, normal, tangent) — the seam-preserving
// tie-break for restoring a corner to its own chart / shading identity. 0 for identical attributes.
[[nodiscard]] float wedgeDistance(const Vertex& a, const Vertex& b) noexcept;

// The wedge among `wedges` (original indices coincident in position) whose render attributes are
// nearest `source`. Empty `wedges` -> 0.
[[nodiscard]] std::uint32_t nearestWedge(std::span<const Vertex> vertices,
                                         std::span<const std::uint32_t> wedges,
                                         const Vertex& source) noexcept;

// Per canonical vertex, the original render wedges welded to it (its own vertex included). Indexed
// by canonical id; sized to `weld.size()`. Non-canonical slots are empty.
[[nodiscard]] std::vector<std::vector<std::uint32_t>>
canonicalWedges(std::span<const std::uint32_t> weld);

// The same canonical->wedge adjacency as `canonicalWedges`, but in a single flat CSR layout — the
// GPU uploader's shape (one storage buffer + one offset buffer, vs a vector-of-vectors that can't
// cross to a shader). `wedges[offsets[c] .. offsets[c + 1])` are canonical `c`'s wedges, in the
// SAME ascending-original-index order as `canonicalWedges(weld)[c]`, so a `nearestWedge` tie
// resolves to the identical (lowest-index) wedge and the two representations emit byte-for-byte the
// same indices. `offsets` is sized `weld.size() + 1`.
struct CanonicalWedgesCsr
{
    std::vector<std::uint32_t> wedges; // all wedges, grouped by canonical id
    std::vector<std::uint32_t>
        offsets; // size weld.size() + 1; canonical c = [offsets[c], offsets[c+1])

    // Canonical `c`'s wedges as a contiguous span (the argument `nearestWedge` wants).
    [[nodiscard]] std::span<const std::uint32_t> forCanonical(std::uint32_t c) const
    {
        return std::span{wedges}.subspan(offsets[c], offsets[c + 1] - offsets[c]);
    }
};

[[nodiscard]] CanonicalWedgesCsr canonicalWedgesCsr(std::span<const std::uint32_t> weld);

// The finest triangle set in canonical space: each input triangle mapped through `weld`, with
// post-weld degenerate (duplicate-corner) triangles dropped. Input face + corner order preserved
// (so winding is preserved). Both the CPU `ActiveFront` and the GPU-shaped `ParallelFront` build
// their finest-face set through this, so they walk identical topology.
[[nodiscard]] std::vector<std::array<std::uint32_t, 3>>
canonicalFaces(std::span<const std::uint32_t> weld, std::span<const std::uint32_t> indices);

} // namespace fire_engine::mesh_topology
