#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vertex.hpp>

namespace fire_engine
{

// Precomputed GPU wedge restoration (VDPM GPU emit, Stage B2). For each ORIGINAL vertex, the CPU's
// `nearestWedge` choice for its canonical vertex AND every ancestor up the removal-parent chain —
// so the GPU emit's scatter is PURE INTEGER indexing (no float `nearestWedge` on the GPU). That
// makes byte-identity with the CPU emit STRUCTURAL: Vulkan gives shader floating-point its own
// precision rules (contraction / reassociation not guaranteed to match the CPU), so recomputing the
// distance on the GPU could tie-break a near-equal candidate differently. Baking the CPU's
// strict-`<` decision in removes that risk entirely.
//
// CSR layout: original vertex v's choices are `choices[offsets[v] .. offsets[v + 1])`, indexed by
// DEPTH — `choices[offsets[v] + d]` is the wedge for the ancestor `d` removal-parent steps up
// `weld[v]`'s chain (d == 0 is the canonical vertex itself). At scatter the GPU walks `weld[oc]`'s
// chain to the active ancestor's depth `d` and reads `choices[offsets[oc] + d]`.
struct WedgeChoices
{
    std::vector<std::uint32_t>
        choices; // per (original vertex, ancestor depth): the restored wedge id
    std::vector<std::uint32_t> offsets; // size originalVertexCount + 1 (CSR)
    std::uint32_t maxDepth{0}; // max removal-parent chain depth (steps) — the shader loop bound

    // Original vertex v's choice list (depth 0 .. depthToRoot(weld[v])).
    [[nodiscard]] std::span<const std::uint32_t> forVertex(std::uint32_t v) const
    {
        return std::span{choices}.subspan(offsets[v], offsets[v + 1] - offsets[v]);
    }
};

// Build the choices from the mesh + forest + weld. The CSR wedge adjacency is derived internally
// from `weld` (`canonicalWedgesCsr`) — a caller passing a separately-built CSR could hand one that
// disagrees with `weld`, so it is not an argument. Every original vertex gets exactly
// `depthToRoot(weld[v]) + 1` choices (including depth 0). Offset accumulation is 64-bit and throws
// std::runtime_error if the total would exceed 32-bit GPU offsets; each removal-parent walk is
// capped at `forest.vertexCount` steps and throws on overrun (a cycle `validateForest` doesn't
// catch). Assumes `validateForest` has passed (the GPU-mesh boundary runs it) + `weld.size() ==
// vertices.size() == forest.vertexCount`. Vulkan-free + deterministic.
[[nodiscard]] WedgeChoices buildWedgeChoices(std::span<const Vertex> vertices,
                                             const VertexForest& forest,
                                             std::span<const std::uint32_t> weld);

// The removal-parent GPU acceleration structure (VDPM GPU emit, Stage B2): per canonical vertex,
// the vertex ONE removal-parent step up its chain — `splits[removingSplit[c]].parent`, or `c`
// itself at a root. It collapses the GPU ancestor walk to one dependent load per step (vs
// removingSplit → split record → parent). `removalParent[root] == root` is deliberate: an INACTIVE
// self-parent fails the walk immediately (a valid front has every root active). Assumes
// `validateForest` has passed. Sized to `forest.vertexCount`. Vulkan-free + deterministic.
[[nodiscard]] std::vector<std::uint32_t> buildRemovalParent(const VertexForest& forest);

} // namespace fire_engine
