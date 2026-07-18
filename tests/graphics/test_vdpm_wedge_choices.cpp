#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/mesh_topology.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_wedge_choices.hpp>
#include <fire_engine/graphics/vertex.hpp>

using namespace fire_engine;

namespace
{

struct Mesh
{
    std::vector<Vertex> verts;
    std::vector<std::uint32_t> indices;
};

Mesh makeGrid(int n)
{
    Mesh m;
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const float u = static_cast<float>(x) / static_cast<float>(n - 1);
            const float v = static_cast<float>(y) / static_cast<float>(n - 1);
            m.verts.push_back(Vertex{Vec3{static_cast<float>(x), static_cast<float>(y), 0.0f},
                                     Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{u, v}});
        }
    }
    for (int y = 0; y < n - 1; ++y)
    {
        for (int x = 0; x < n - 1; ++x)
        {
            const auto a = static_cast<std::uint32_t>((y * n) + x);
            const auto b = static_cast<std::uint32_t>((y * n) + x + 1);
            const auto c = static_cast<std::uint32_t>(((y + 1) * n) + x);
            const auto d = static_cast<std::uint32_t>(((y + 1) * n) + x + 1);
            m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
        }
    }
    return m;
}

// Per-corner-duplicated grid: every triangle carries its own three vertices at shared positions →
// multi-wedge canonicals, so nearestWedge genuinely chooses (distinct UVs per duplicate).
Mesh makeSeamedGrid(int n)
{
    const Mesh shared = makeGrid(n);
    Mesh dup;
    for (std::size_t t = 0; t < shared.indices.size(); ++t)
    {
        const Vertex& src = shared.verts[shared.indices[t]];
        Vertex v = src;
        v.texCoord(
            Vec2{src.texCoord().s() + (0.125f * static_cast<float>(t % 5)), src.texCoord().t()});
        dup.indices.push_back(static_cast<std::uint32_t>(dup.verts.size()));
        dup.verts.push_back(v);
    }
    return dup;
}

std::uint32_t depthToRoot(const VertexForest& f, std::uint32_t canonical)
{
    std::uint32_t c = canonical;
    std::uint32_t depth = 0;
    while (f.removingSplit[c] != kNoSplit)
    {
        c = f.splits[f.removingSplit[c]].parent;
        ++depth;
    }
    return depth;
}

} // namespace

TEST_CASE("buildWedgeChoices: choices equal nearestWedge for every ancestor depth", "[vdpm]")
{
    // The structural byte-identity guarantee: choices[offset[v] + d] must equal the CPU
    // nearestWedge for the ancestor d steps up weld[v]'s chain — for EVERY vertex and EVERY depth,
    // independent of any shader. Also: exactly depthToRoot + 1 choices per vertex, and every stored
    // wedge is a member of that ancestor's canonical bucket.
    for (const Mesh& m : {makeGrid(9), makeSeamedGrid(7)})
    {
        const QuadricSimplifier simp;
        const VertexForest forest =
            buildVertexForest(m.verts, simp.collapseSequence(m.verts, m.indices));
        const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(m.verts);
        const mesh_topology::CanonicalWedgesCsr csr = mesh_topology::canonicalWedgesCsr(weld);
        const WedgeChoices wc = buildWedgeChoices(m.verts, forest, weld);

        REQUIRE(wc.offsets.size() == m.verts.size() + 1);
        REQUIRE(wc.offsets.front() == 0u);
        REQUIRE(wc.offsets.back() == static_cast<std::uint32_t>(wc.choices.size()));

        std::uint32_t observedMaxDepth = 0;
        for (std::uint32_t v = 0; v < m.verts.size(); ++v)
        {
            const std::uint32_t depth = depthToRoot(forest, weld[v]);
            observedMaxDepth = std::max(observedMaxDepth, depth);
            const std::span<const std::uint32_t> choices = wc.forVertex(v);
            REQUIRE(choices.size() == depth + 1u); // exactly depthToRoot + 1 (includes depth 0)

            // Walk the chain and check each depth's stored choice against the live nearestWedge.
            std::uint32_t c = weld[v];
            for (std::uint32_t d = 0; d < choices.size(); ++d)
            {
                const std::span<const std::uint32_t> bucket = csr.forCanonical(c);
                const std::uint32_t expected =
                    mesh_topology::nearestWedge(m.verts, bucket, m.verts[v]);
                CHECK(choices[d] == expected);
                // and the stored wedge is a member of that ancestor's canonical bucket.
                CHECK(std::ranges::find(bucket, choices[d]) != bucket.end());
                if (forest.removingSplit[c] != kNoSplit)
                {
                    c = forest.splits[forest.removingSplit[c]].parent;
                }
            }
        }
        CHECK(wc.maxDepth == observedMaxDepth);
    }
}

TEST_CASE("buildWedgeChoices memory evidence (choice bytes vs the complete B2 static allocation)",
          "[vdpm]")
{
    // Evidence for the B2 gate: the precomputed-choice memory (absolute, per original vertex, and
    // as a ratio of the COMPLETE static B2 allocation — every buffer VdpmGpuMesh's full build
    // uploads), so the FP-safe wedge restoration's cost is on record against what it actually adds
    // to, not a partial footprint. The struct sizes mirror the asserted std430 ABI (render/ubo.hpp:
    // VdpmSplitGpu 48 B, VdpmPositionGpu 16 B); kept as literals so this graphics-layer test stays
    // Vulkan-free. (A helmet-scale number belongs to the render-integration stage, where the mesh
    // is already loaded through the Vulkan glTF path — there is no Vulkan-free full-attribute
    // loader to exercise it from a [vdpm] CPU test.)
    constexpr std::size_t kU32 = sizeof(std::uint32_t);
    constexpr std::size_t kSplitBytes = 48;    // VdpmSplitGpu
    constexpr std::size_t kPositionBytes = 16; // VdpmPositionGpu (padded vec4)
    for (const Mesh& m : {makeGrid(33), makeGrid(65), makeSeamedGrid(17)})
    {
        const QuadricSimplifier simp;
        const VertexForest forest =
            buildVertexForest(m.verts, simp.collapseSequence(m.verts, m.indices));
        const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(m.verts);
        const WedgeChoices wc = buildWedgeChoices(m.verts, forest, weld);
        const std::vector<std::uint32_t> removalParent = buildRemovalParent(forest);

        const std::size_t choiceBytes = (wc.choices.size() + wc.offsets.size()) * kU32;
        // The COMPLETE static emit allocation: splits + positions + indices + weld + removalParent
        // + wedge choices (choices + offsets) — every buffer the full VdpmGpuMesh build uploads.
        const std::size_t b2Bytes = (forest.splits.size() * kSplitBytes) +
                                    (m.verts.size() * kPositionBytes) + (m.indices.size() * kU32) +
                                    (weld.size() * kU32) + (removalParent.size() * kU32) +
                                    choiceBytes;
        WARN("wedge choices: " << choiceBytes << " B ("
                               << (static_cast<double>(choiceBytes) /
                                   static_cast<double>(std::max<std::size_t>(1, m.verts.size())))
                               << " B/vertex), maxDepth " << wc.maxDepth
                               << "; complete B2 static allocation " << b2Bytes
                               << " B (choices are "
                               << (100.0 * static_cast<double>(choiceBytes) /
                                   static_cast<double>(std::max<std::size_t>(1, b2Bytes)))
                               << "% of it)");
        CHECK(choiceBytes > 0);
        CHECK(b2Bytes > choiceBytes); // choices are a fraction of the whole static set
    }
}

TEST_CASE("buildRemovalParent: one removal-parent step per canonical, self at roots", "[vdpm]")
{
    for (const Mesh& m : {makeGrid(17), makeSeamedGrid(9)})
    {
        const QuadricSimplifier simp;
        const VertexForest forest =
            buildVertexForest(m.verts, simp.collapseSequence(m.verts, m.indices));
        const std::vector<std::uint32_t> parent = buildRemovalParent(forest);

        REQUIRE(parent.size() == forest.vertexCount);
        bool sawRoot = false;
        bool sawNonRoot = false;
        for (std::uint32_t c = 0; c < forest.vertexCount; ++c)
        {
            if (forest.removingSplit[c] == kNoSplit)
            {
                // A root is its OWN removal parent — an inactive self-parent then fails the GPU
                // ancestor walk immediately (a valid front keeps every root active).
                CHECK(parent[c] == c);
                sawRoot = true;
            }
            else
            {
                // Non-root: exactly the removing split's parent — one step toward an active
                // ancestor, matching the recursive activeAncestor walk.
                CHECK(parent[c] == forest.splits[forest.removingSplit[c]].parent);
                CHECK(parent[c] != c); // a split's parent is never the removed child itself
                sawNonRoot = true;
            }
        }
        CHECK(sawRoot);    // the coarsest vertices survive to the root front
        CHECK(sawNonRoot); // the mesh actually collapsed
    }
}
