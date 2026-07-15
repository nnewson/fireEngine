#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>

using namespace fire_engine;

namespace
{

struct Mesh
{
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
};

// An n x n vertex grid (2 triangles per quad) — no attribute seams. Its regular topology collapses
// in long serial chains, so the DAG is DEEP (rank scales ~linearly with n: ~250 at 65x65) — the
// opposite of a curved mesh, and the worst case for a rank-per-dispatch scheme.
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
            const auto a = static_cast<uint32_t>((y * n) + x);
            const auto b = static_cast<uint32_t>((y * n) + x + 1);
            const auto c = static_cast<uint32_t>(((y + 1) * n) + x);
            const auto d = static_cast<uint32_t>(((y + 1) * n) + x + 1);
            m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
        }
    }
    return m;
}

// A UV sphere — radial normals, a longitude seam, and pole fans (welded non-manifold edges), so its
// forest exercises the vl/vr non-monotone dependencies the DAG is really about.
Mesh makeUvSphere(int rings, int segments)
{
    Mesh m;
    constexpr float pi = 3.14159265f;
    for (int r = 0; r <= rings; ++r)
    {
        const float lat = pi * ((static_cast<float>(r) / static_cast<float>(rings)) - 0.5f);
        for (int s = 0; s <= segments; ++s)
        {
            const float lon = 2.0f * pi * static_cast<float>(s) / static_cast<float>(segments);
            const Vec3 nrm{std::cos(lat) * std::cos(lon), std::sin(lat),
                           std::cos(lat) * std::sin(lon)};
            m.verts.push_back(Vertex{nrm, Colour3{}, nrm,
                                     Vec2{static_cast<float>(s) / static_cast<float>(segments),
                                          static_cast<float>(r) / static_cast<float>(rings)}});
        }
    }
    const int stride = segments + 1;
    for (int r = 0; r < rings; ++r)
    {
        for (int s = 0; s < segments; ++s)
        {
            const auto a = static_cast<uint32_t>((r * stride) + s);
            const auto b = static_cast<uint32_t>((r * stride) + s + 1);
            const auto c = static_cast<uint32_t>(((r + 1) * stride) + s);
            const auto d = static_cast<uint32_t>(((r + 1) * stride) + s + 1);
            m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
        }
    }
    return m;
}

VertexForest forestOf(const Mesh& m)
{
    const QuadricSimplifier simp;
    return buildVertexForest(m.verts, simp.collapseSequence(m.verts, m.indices));
}

// The recursive oracle's refine/coarsen passes, replicated on an ActiveFront from GIVEN per-split
// scores — exactly `refineForView`'s update (coarse-first forceRefine over budget, then a
// fine-first coarsen fixpoint under coarsenBudget / back-facing), but decoupled from the scoring so
// it can be driven with the same score vector as ParallelFront.
void oracleUpdate(ActiveFront& f, std::span<const float> score,
                  std::span<const std::uint8_t> backface, float budget, float coarsenBudget)
{
    const auto n = static_cast<std::uint32_t>(f.forest().splits.size());
    for (std::uint32_t i = n; i-- > 0;)
    {
        if (backface[i] == 0 && !f.refined(i) && score[i] > budget)
        {
            f.forceRefine(i);
        }
    }
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (std::uint32_t i = 0; i < n; ++i)
        {
            if (f.refined(i) && (backface[i] != 0 || score[i] < coarsenBudget) && f.coarsen(i))
            {
                changed = true;
            }
        }
    }
}

} // namespace

TEST_CASE("dependency DAG: real forests are acyclic and rank-consistent", "[vdpm]")
{
    // The rank of every split must strictly exceed the rank of each split it depends on
    // (removingSplit of a non-root parent/vl/vr) — the property that makes a rank-ordered parallel
    // apply a correct substitute for the recursive forceRefine. Building it also proves acyclicity
    // (no throw). Reports maxRank: the number of rank-ordered passes a GPU refine/coarsen would
    // take (the dispatch-count evidence the arc's Stage 0 must gather before choosing a scheme).
    for (const Mesh& m : {makeGrid(17), makeUvSphere(24, 32)})
    {
        const VertexForest forest = forestOf(m);
        const DependencyDag dag = buildDependencyDag(forest); // throws on a cycle
        REQUIRE(dag.rank.size() == forest.splits.size());

        std::uint32_t checkedDeps = 0;
        for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(forest.splits.size()); ++s)
        {
            const VertexSplit& sp = forest.splits[s];
            for (const std::uint32_t dep : {sp.parent, sp.vl, sp.vr})
            {
                if (dep == kInvalidVertex)
                {
                    continue;
                }
                const std::uint32_t depSplit = forest.removingSplit[dep];
                if (depSplit == kNoSplit)
                {
                    continue; // root dependency: no ordering constraint
                }
                CHECK(dag.rank[s] > dag.rank[depSplit]);
                ++checkedDeps;
            }
        }
        CHECK(checkedDeps > 0);  // the sphere genuinely has non-root dependencies
        CHECK(dag.maxRank >= 1); // ...and therefore a non-trivial rank depth
        WARN("maxRank = " << dag.maxRank << " over " << forest.splits.size() << " splits");
    }
}

// Hidden ([.]) evidence: how the DAG rank depth scales with mesh size. Each frame runs ~3
// rank-ordered sequences (closure + refine + coarsen), so a naive rank-PER-instance dispatch is
// ~3·maxRank dispatches per instance — ~90 for a curved 12k-split mesh, ~750 for a 4k-split flat
// grid. That is enough command-recording overhead (especially on MoltenVK) that this does NOT by
// itself prove rank-per-dispatch viable: it only shows curved SYNTHETIC meshes have shallow DAGs.
// Still open (Stage B design): measure representative ASSETS (the helmet); prefer BATCHING all VDPM
// instances into each rank dispatch (rank count, not rank×instances); keep a work-queue/wave scheme
// as the fallback for deep forests (flat grids). Run with `./test_fire_engine [RankEvidence]`.
TEST_CASE("dependency DAG: rank depth vs mesh size", "[.][vdpm][RankEvidence]")
{
    for (const int n : {9, 17, 33, 65})
    {
        const VertexForest forest = forestOf(makeGrid(n));
        WARN("grid " << n << "x" << n << ": " << forest.splits.size()
                     << " splits, maxRank = " << buildDependencyDag(forest).maxRank);
    }
    for (const int rings : {12, 24, 48, 96})
    {
        const VertexForest forest = forestOf(makeUvSphere(rings, rings * 4 / 3));
        WARN("sphere " << rings << ": " << forest.splits.size()
                       << " splits, maxRank = " << buildDependencyDag(forest).maxRank);
    }
}

TEST_CASE("ParallelFront: rank-ordered update reproduces the recursive oracle's front", "[vdpm]")
{
    // The heart of Stage 0: the parallel refine/coarsen (requirement-closure + rank-ordered apply)
    // must produce the SAME persistent front as the recursive forceRefine + fine-first fixpoint,
    // for arbitrary per-split scores. Both fronts are driven with an identical sequence of
    // pseudo-random score/back-face vectors (a moving view), and their full state — every active
    // vertex and refined split — is compared each frame. Byte-identical here means the parallel
    // scheduling is a faithful substitute; only the (separately-modelled) repairs will legitimately
    // diverge later.
    constexpr float budget = 1.0f;
    constexpr float coarsenBudget = 0.6f;
    for (const Mesh& m : {makeGrid(17), makeUvSphere(24, 32)})
    {
        const QuadricSimplifier simp;
        const auto collapses = simp.collapseSequence(m.verts, m.indices);
        const VertexForest forest = buildVertexForest(m.verts, collapses);
        const auto n = static_cast<std::uint32_t>(forest.splits.size());

        ActiveFront oracle = ActiveFront::build(m.verts, m.indices, collapses);
        ParallelFront parallel = ParallelFront::build(forest);

        std::mt19937 rng(0xC0FFEE);
        std::uniform_real_distribution<float> scoreDist(0.0f, 2.0f); // straddles the budget
        std::size_t mismatches = 0;
        for (int frame = 0; frame < 8; ++frame)
        {
            std::vector<float> score(n);
            std::vector<std::uint8_t> backface(n);
            for (std::uint32_t i = 0; i < n; ++i)
            {
                score[i] = scoreDist(rng);
                backface[i] = (rng() & 7u) == 0u ? 1 : 0; // ~1/8 back-facing
            }
            oracleUpdate(oracle, score, backface, budget, coarsenBudget);
            parallel.applyView(score, backface, budget, coarsenBudget);
            parallel.validateInvariants(); // catch a corrupt dependent count before it goes latent

            for (std::uint32_t v = 0; v < forest.vertexCount; ++v)
            {
                mismatches += (oracle.active(v) != parallel.active(v)) ? 1 : 0;
            }
            for (std::uint32_t s = 0; s < n; ++s)
            {
                mismatches += (oracle.refined(s) != parallel.refined(s)) ? 1 : 0;
            }
        }
        CHECK(mismatches == 0);
    }
}

namespace
{
// A diamond forest: a base split B (rank 0), two splits L and R that BOTH depend on B (rank 1, so
// they share a same-rank dependency), and a top split T depending on both (rank 2). Vertex 1 is B's
// child and the parent of both L and R, so refining L and R both increment dependents_[1] and the
// closure marks B required from both — exactly the rank-local atomic sharing the GPU port must
// honour.
VertexForest makeDiamondForest()
{
    VertexForest f;
    f.vertexCount = 5;
    f.removingSplit = {kNoSplit, 0, 1, 2, 3}; // vertex v>0 removed by split v-1
    auto mk = [](std::uint32_t child, std::uint32_t parent, std::uint32_t vl)
    {
        VertexSplit s;
        s.child = child;
        s.parent = parent;
        s.vl = vl;
        s.vr = kInvalidVertex;
        return s;
    };
    f.splits = {mk(1, 0, 0),  // B: rank 0 (parent + vl are the root vertex 0)
                mk(2, 1, 0),  // L: parent = B's child ⇒ depends on B ⇒ rank 1
                mk(3, 1, 0),  // R: parent = B's child ⇒ depends on B ⇒ rank 1 (shares B with L)
                mk(4, 2, 3)}; // T: parent = L's child, vl = R's child ⇒ depends on L and R ⇒ rank 2
    return f;
}
} // namespace

TEST_CASE("ParallelFront: same-rank splits correctly share a dependency (diamond)", "[vdpm]")
{
    const VertexForest forest = makeDiamondForest();
    const DependencyDag dag = buildDependencyDag(forest);
    CHECK(dag.rank[0] == 0);
    CHECK(dag.rank[1] == 1);
    CHECK(dag.rank[2] == 1); // L and R are the same rank...
    CHECK(dag.rank[3] == 2);

    ParallelFront front = ParallelFront::build(forest);
    const std::vector<float> hot{9, 9, 9, 9}; // all over budget
    const std::vector<std::uint8_t> face{0, 0, 0, 0};
    front.applyView(hot, face, 1.0f, 0.6f);
    front
        .validateInvariants(); // ...and refining both must leave dependents_[1] == 2 (checked here)
    for (std::uint32_t v = 1; v <= 4; ++v)
    {
        CHECK(front.active(v)); // the whole diamond refined in
    }

    const std::vector<float> cold{0, 0, 0, 0};
    front.applyView(cold, face, 1.0f, 0.6f);
    front.validateInvariants();
    for (std::uint32_t v = 1; v <= 4; ++v)
    {
        CHECK_FALSE(front.active(v)); // ...and coarsened all the way back out
    }
}

TEST_CASE("validateForest: structurally malformed forests are rejected", "[vdpm]")
{
    VertexForest good = makeDiamondForest();
    CHECK_NOTHROW(validateForest(good));

    SECTION("removingSplit size mismatch")
    {
        VertexForest f = good;
        f.removingSplit.pop_back();
        CHECK_THROWS(validateForest(f));
    }
    SECTION("out-of-range vertex reference")
    {
        VertexForest f = good;
        f.splits[1].parent = 99;
        CHECK_THROWS(validateForest(f));
    }
    SECTION("child is not its own removing split")
    {
        VertexForest f = good;
        f.removingSplit[2] = 0; // vertex 2 is L's child but now points at B
        CHECK_THROWS(validateForest(f));
    }
    SECTION("out-of-range removing-split reference")
    {
        VertexForest f = good;
        f.removingSplit[1] = 42;
        CHECK_THROWS(validateForest(f));
    }
    SECTION("removingSplit/child reverse inconsistency")
    {
        VertexForest f = good;
        f.removingSplit[0] = 0; // vertex 0 (a root) aliases split 0, whose child is vertex 1, not 0
        CHECK_THROWS(validateForest(f));
    }
}

TEST_CASE("dependency DAG: a cyclic forest is rejected in all builds", "[vdpm]")
{
    // Two splits that each depend on the other's parent: split 0 needs vertex 1 active (removed by
    // split 1) and split 1 needs vertex 0 active (removed by split 0). A rank ordering cannot
    // exist, so the build must throw rather than silently emit ranks that would hang a GPU.
    VertexForest forest;
    forest.vertexCount = 3;
    forest.removingSplit = {0, 1,
                            kNoSplit}; // vertex 0 → split 0, vertex 1 → split 1, vertex 2 root
    VertexSplit s0;
    s0.child = 0;
    s0.parent = 1;
    s0.vl = 2;
    s0.vr = kInvalidVertex;
    VertexSplit s1;
    s1.child = 1;
    s1.parent = 0;
    s1.vl = 2;
    s1.vr = kInvalidVertex;
    forest.splits = {s0, s1};

    CHECK_THROWS(buildDependencyDag(forest));
}
