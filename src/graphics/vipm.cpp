#include <fire_engine/graphics/vipm.hpp>

#include <cstdint>

namespace fire_engine
{

std::vector<MorphVertex> buildVipmMorphData(std::span<const Vertex> vertices,
                                            std::span<const MeshCollapse> collapses,
                                            std::span<const float> levelErrors)
{
    const std::size_t n = vertices.size();

    // A vertex geomorphs toward ALL the attributes of its resolved survivor — position, normal AND
    // texcoord — so the texture doesn't warp during the morph (see MorphVertex).
    auto morphToward = [&vertices](uint32_t target, float error)
    {
        const Vertex& t = vertices[target];
        return MorphVertex{t.position(), error, t.normal(), 0.0f, t.texCoord(), 0.0f, 0.0f};
    };

    // Default: never collapses — target is the vertex itself, sentinel error (never morphs).
    std::vector<MorphVertex> morph(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        morph[i] = morphToward(static_cast<uint32_t>(i), kVipmNeverCollapses);
    }

    // Union-find over the vertices. Subset placement means a collapse merges `removed` into `kept`
    // and `kept` keeps its original position, so after replaying every collapse up to some error,
    // find(v) is v's surviving representative — a vertex still present at that error — and its
    // position is simply that vertex's (unchanged) original position.
    // Index-fill by hand (not std::iota): std::ranges::iota isn't portable across the Apple-Clang
    // libc++ we build against, and plain std::iota trips modernize-use-ranges under the
    // now-blocking clang-tidy gate.
    std::vector<uint32_t> parent(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        parent[i] = static_cast<uint32_t>(i);
    }
    auto find = [&parent](uint32_t v)
    {
        while (parent[v] != v)
        {
            parent[v] = parent[parent[v]]; // path halving
            v = parent[v];
        }
        return v;
    };

    // Replay the recorded (finest -> coarsest) collapse stream, banded by the discrete level
    // errors. levelErrors[0] is LOD0 (0.0); levels 1.. are the coarser cut points in increasing
    // error. A vertex removed at error e falls in the band ending at the first level whose error >=
    // e; its geomorph target is its representative *after* that whole band is applied, i.e. a
    // vertex present at that coarser level. Reaching it exactly at morphFactor 1 makes the topology
    // swap seamless.
    std::size_t collapseIdx = 0;
    for (std::size_t level = 1; level < levelErrors.size(); ++level)
    {
        const float levelError = levelErrors[level];
        const std::size_t bandStart = collapseIdx;
        while (collapseIdx < collapses.size() && collapses[collapseIdx].error <= levelError)
        {
            const MeshCollapse& c = collapses[collapseIdx];
            parent[find(c.removed)] = find(c.kept);
            ++collapseIdx;
        }
        for (std::size_t i = bandStart; i < collapseIdx; ++i)
        {
            const MeshCollapse& c = collapses[i];
            morph[c.removed] = morphToward(find(c.removed), c.error);
        }
    }

    return morph;
}

} // namespace fire_engine
