#include <fire_engine/graphics/vdpm_wedge_choices.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <fire_engine/graphics/mesh_topology.hpp>

namespace fire_engine
{

WedgeChoices buildWedgeChoices(std::span<const Vertex> vertices, const VertexForest& forest,
                               std::span<const std::uint32_t> weld)
{
    const auto n = static_cast<std::uint32_t>(vertices.size());
    // The CSR is derived from weld here, so the two can't disagree. (validateForest is the caller's
    // gate for weld/forest consistency; this only trusts what it validated.)
    const mesh_topology::CanonicalWedgesCsr wedges = mesh_topology::canonicalWedgesCsr(weld);
    // A validated forest is still not proven acyclic in its removal-parent chain, so cap every walk
    // at vertexCount steps — a longer walk is a cycle.
    const std::uint32_t stepCap = forest.vertexCount;

    WedgeChoices wc;
    wc.offsets.assign(static_cast<std::size_t>(n) + 1, 0);

    // Pass 1: count choices per vertex = depthToRoot(weld[v]) + 1, accumulate offsets in 64-bit,
    // and track the max chain depth (the shader loop bound). Reject a total that won't fit a 32-bit
    // GPU offset before narrowing.
    std::uint64_t total = 0;
    for (std::uint32_t v = 0; v < n; ++v)
    {
        std::uint32_t c = weld[v];
        std::uint32_t depth = 0;
        while (forest.removingSplit[c] != kNoSplit)
        {
            c = forest.splits[forest.removingSplit[c]].parent;
            if (++depth > stepCap)
            {
                throw std::runtime_error("VDPM wedge choices: removal-parent chain cycle");
            }
        }
        wc.maxDepth = std::max(wc.maxDepth, depth);
        total += static_cast<std::uint64_t>(depth) + 1;
        if (total > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("VDPM wedge choices exceed 32-bit GPU offsets");
        }
        wc.offsets[v + 1] = static_cast<std::uint32_t>(total);
    }

    // Pass 2: fill each vertex's choices by walking weld[v]'s chain (depth 0 = the canonical
    // vertex), storing the CPU nearestWedge for each ancestor's bucket against v's own attributes.
    wc.choices.resize(static_cast<std::size_t>(total));
    for (std::uint32_t v = 0; v < n; ++v)
    {
        std::uint32_t c = weld[v];
        std::uint32_t idx = wc.offsets[v];
        while (true)
        {
            wc.choices[idx++] =
                mesh_topology::nearestWedge(vertices, wedges.forCanonical(c), vertices[v]);
            if (forest.removingSplit[c] == kNoSplit)
            {
                break;
            }
            c = forest.splits[forest.removingSplit[c]].parent;
        }
    }
    return wc;
}

} // namespace fire_engine
