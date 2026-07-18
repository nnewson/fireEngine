#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/mesh_topology.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>

#include <support/vdpm.hpp>

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

// Per-corner-duplicated grid: every triangle carries its OWN three vertices at shared positions, so
// weldByPosition recovers a canonical carrying several render wedges (a position shared by k
// triangles → k wedges). Each duplicate gets a distinct UV so nearestWedge has a genuine choice —
// the multi-wedge seam-restoration path the emit must match the oracle on.
Mesh makeSeamedGrid(int n)
{
    const Mesh shared = makeGrid(n);
    Mesh dup;
    for (std::size_t t = 0; t < shared.indices.size(); ++t)
    {
        const Vertex& src = shared.verts[shared.indices[t]];
        Vertex v = src;
        v.texCoord(Vec2{src.texCoord().s() + (0.125f * static_cast<float>(t % 5)),
                        src.texCoord().t()}); // distinct, exactly-representable per-duplicate UV
        dup.indices.push_back(static_cast<std::uint32_t>(dup.verts.size()));
        dup.verts.push_back(v);
    }
    return dup;
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

// Emitted (non-degenerate) canonical triangle count for a settled front — a finest face survives
// iff its three active ancestors are distinct. Works on any front exposing forest()/active(), so it
// measures the sequential and parallel repaired fronts by the same rule.
template <class Front>
std::size_t emittedTriangles(const Front& front, std::span<const Vertex> vertices,
                             std::span<const std::uint32_t> indices)
{
    const VertexForest& f = front.forest();
    const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(vertices);
    auto ancestor = [&](std::uint32_t v)
    {
        while (!front.active(v))
        {
            v = f.splits[f.removingSplit[v]].parent;
        }
        return v;
    };
    std::size_t tris = 0;
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const std::uint32_t a0 = ancestor(weld[indices[i]]);
        const std::uint32_t a1 = ancestor(weld[indices[i + 1]]);
        const std::uint32_t a2 = ancestor(weld[indices[i + 2]]);
        if (a0 != a1 && a1 != a2 && a0 != a2)
        {
            ++tris;
        }
    }
    return tris;
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

TEST_CASE("dependency DAG: the shared dependency triple mirrors removingSplit per-slot", "[vdpm]")
{
    // The DAG's parent/vl/vr dependency triple is the ONE authority the closure + the GPU B3
    // uploader read (instead of each re-deriving removingSplit[...]). Every slot must equal the
    // removing split of its referenced vertex — kNoSplit for a root (always active) or the boundary
    // sentinel (no vr).
    auto depSplitOf = [](const VertexForest& f, std::uint32_t v)
    { return v == kInvalidVertex ? kNoSplit : f.removingSplit[v]; };

    for (const Mesh& m : {makeGrid(17), makeUvSphere(24, 32)})
    {
        const VertexForest forest = forestOf(m);
        const DependencyDag dag = buildDependencyDag(forest);
        REQUIRE(dag.dependencies.size() == forest.splits.size());

        for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(forest.splits.size()); ++s)
        {
            const VertexSplit& sp = forest.splits[s];
            const SplitDependencies& d = dag.dependencies[s];
            CHECK(d.parent == depSplitOf(forest, sp.parent));
            CHECK(d.vl == depSplitOf(forest, sp.vl));
            CHECK(d.vr == depSplitOf(forest, sp.vr));
        }
    }

    SECTION("a boundary split (vr sentinel) and a same-vertex parent/vl split (duplicate) — an "
            "explicit synthetic forest, not an incidental collapse topology")
    {
        // vertex 0 root; split 0 removes vertex 1 (from the root); split 1 removes vertex 2 with
        // BOTH parent and vl = vertex 1 and NO vr (boundary). So split 1's parent+vl slots both
        // carry split 0 (not de-duplicated), its vr slot is kNoSplit (the boundary sentinel —
        // checked directly here rather than relying on a real mesh keeping a boundary edge), and
        // refining it bumps dependents_[1] twice.
        VertexForest f;
        f.vertexCount = 3;
        f.removingSplit = {kNoSplit, 0, 1};
        auto mk = [](std::uint32_t child, std::uint32_t parent, std::uint32_t vl)
        {
            VertexSplit s;
            s.child = child;
            s.parent = parent;
            s.vl = vl;
            s.vr = kInvalidVertex;
            return s;
        };
        f.splits = {mk(1, 0, 0), mk(2, 1, 1)};

        const DependencyDag dag = buildDependencyDag(f);
        CHECK(dag.dependencies[1].parent == 0u);   // both slots resolve to split 0...
        CHECK(dag.dependencies[1].vl == 0u);       // ...not de-duplicated
        CHECK(dag.dependencies[1].vr == kNoSplit); // the boundary sentinel
        CHECK(dag.rank[1] == 1u);                  // still one rank past split 0

        ParallelFront front = ParallelFront::build(f);
        front.applyView(std::vector<float>{9, 9}, std::vector<std::uint8_t>{0, 0}, 1.0f, 0.6f);
        front.validateInvariants(); // reconstructs dependents_[1] == 2 from split 1's two slots
        CHECK(front.active(2));
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

TEST_CASE("ParallelFront full build: mesh context matches the ActiveFront oracle", "[vdpm]")
{
    // The full build (from a mesh) must produce the SAME finest canonical face set as ActiveFront,
    // so the parallel repairs + emit walk identical topology. Cross-check against the oracle's
    // finest set = its emitActiveCanonical() at full refinement (every finest face maps to itself).
    // Compare DIRECTLY — face order, corner order AND winding must match, so a reversed triangle
    // (the exact defect foldover repair cares about) would fail, not slip through a vertex sort.
    for (const Mesh& m : {makeGrid(9), makeUvSphere(12, 16)})
    {
        const QuadricSimplifier simp;
        const auto collapses = simp.collapseSequence(m.verts, m.indices);

        ParallelFront full = ParallelFront::build(m.verts, m.indices, collapses);
        const ParallelFront sched = ParallelFront::build(buildVertexForest(m.verts, collapses));
        CHECK(full.forest().splits.size() == sched.forest().splits.size());
        CHECK(full.dag().maxRank == sched.dag().maxRank);
        CHECK(full.hasMeshContext());
        CHECK_FALSE(sched.hasMeshContext()); // scheduling-only build carries no mesh context
        CHECK(sched.finestFaces().empty());

        ActiveFront oracle = ActiveFront::build(m.verts, m.indices, collapses);
        oracle.refineAll();
        CHECK(full.finestFaces() ==
              oracle.emitActiveCanonical()); // order + winding, not membership

        // The full build's scheduling still works (mesh context doesn't perturb the front state).
        const auto n = static_cast<std::uint32_t>(full.forest().splits.size());
        full.applyView(std::vector<float>(n, 9.0f), std::vector<std::uint8_t>(n, 0), 1.0f, 0.6f);
        full.validateInvariants();
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

TEST_CASE("ParallelFront::repairFront drives the settled front to zero invariant failures",
          "[vdpm]")
{
    // The snapshot repair's correctness contract: starting from an arbitrary settled front, detect
    // EVERY violation against that snapshot, close + apply in rank order, repeat — and end with
    // zero foldover AND zero coverage failures (the two independent invariants), measured by the
    // SHARED first-principles validators, NOT the repair's own accounting. The parallel repair
    // shares P2's per-face policy but not its sequential schedule, so it may reach a DIFFERENT
    // valid front — we assert the invariants, never front equality.
    constexpr float budget = 1.0f;
    constexpr float coarsenBudget = 0.6f;
    constexpr float vw = 1000.0f;
    constexpr float vh = 1000.0f;
    const Vec3 cam{0.0f, 0.0f, 2.4f};
    const Mat4 viewProj =
        Mat4::perspective(1.0f, 1.0f, 0.05f, 100.0f) * Mat4::lookAt(cam, {0, 0, 0}, {0, 1, 0});

    // A curved mesh viewed up close: partial fronts leave real silhouette coverage holes and
    // recession foldovers. Cull on (opaque) and off (double-sided), and identity / asymmetric
    // non-uniform / REFLECTED (negative-determinant, asymmetric) worlds — the reflected case flips
    // world winding, so it exercises the whole snapshot path's world-space foldover + cull
    // handling, not just the classifier unit tests.
    for (const bool cull : {true, false})
    {
        for (const Mat4& world : {Mat4::identity(), Mat4::scale(Vec3{1.4f, 0.7f, 1.2f}),
                                  Mat4::scale(Vec3{-1.3f, 0.8f, 1.1f})})
        {
            const Mesh m = makeUvSphere(20, 28);
            const QuadricSimplifier simp;
            const auto collapses = simp.collapseSequence(m.verts, m.indices);
            const VertexForest forest = buildVertexForest(m.verts, collapses);
            const auto n = static_cast<std::uint32_t>(forest.splits.size());

            ParallelFront front = ParallelFront::build(m.verts, m.indices, collapses);

            std::mt19937 rng(0x8E9A17u);
            std::uniform_real_distribution<float> scoreDist(0.0f, 2.0f);
            std::size_t preRepairViolations = 0;
            for (int frame = 0; frame < 6; ++frame)
            {
                std::vector<float> score(n);
                std::vector<std::uint8_t> backface(n);
                for (std::uint32_t i = 0; i < n; ++i)
                {
                    score[i] = scoreDist(rng);
                    backface[i] = 0; // let the repair, not a score, decide back-face handling
                }
                front.applyView(score, backface, budget, coarsenBudget);

                // The settled (pre-repair) front genuinely has violations to fix.
                preRepairViolations += test::foldoverCount(front, m.verts, m.indices, world) +
                                       test::coverageFailures(front, m.verts, m.indices, viewProj,
                                                              cam, world, vw, vh, cull);

                front.repairFront(m.verts, world, cam, viewProj, vw, vh, cull);
                front.validateInvariants(); // the front stayed structurally consistent

                CHECK(test::foldoverCount(front, m.verts, m.indices, world) == 0);
                CHECK(test::coverageFailures(front, m.verts, m.indices, viewProj, cam, world, vw,
                                             vh, cull) == 0);
                // The inflationary bound held: apply rounds are one fewer than detection passes.
                CHECK(front.repairDetectionPasses() <= n + 1);
                CHECK(front.repairApplyRounds() == front.repairDetectionPasses() - 1);
            }
            CHECK(preRepairViolations > 0); // the test actually exercised the repair
        }
    }
}

TEST_CASE("ParallelFront::repairFront overhead vs the sequential oracle (evidence)", "[vdpm]")
{
    // EVIDENCE, not an assertion: the snapshot repair and P2's sequential joint repair reach
    // DIFFERENT valid fronts (over-refining a region can dissolve a violation the other repairs via
    // another corner), and a different schedule can land either side of the other. So this reports
    // the triangle overhead both ways and only asserts the one hard STRUCTURAL bound — neither
    // repaired front can exceed the full finest detail. The numbers feed the Stage-0 gate review.
    constexpr float budget = 1.0f;
    constexpr float coarsenBudget = 0.6f;
    constexpr float vw = 1000.0f;
    constexpr float vh = 1000.0f;
    const Vec3 cam{0.0f, 0.0f, 2.4f};
    const Mat4 viewProj =
        Mat4::perspective(1.0f, 1.0f, 0.05f, 100.0f) * Mat4::lookAt(cam, {0, 0, 0}, {0, 1, 0});
    const Mat4 world = Mat4::identity();
    constexpr bool cull = true;

    const Mesh m = makeUvSphere(24, 32);
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);
    const auto n = static_cast<std::uint32_t>(forest.splits.size());
    // The finest canonical face set IS the full detail — its size bounds every repaired front.
    const std::size_t finestTris =
        ParallelFront::build(m.verts, m.indices, collapses).finestFaces().size();

    std::mt19937 rng(0x5EED1234u);
    std::uniform_real_distribution<float> scoreDist(0.0f, 2.0f);
    for (int frame = 0; frame < 4; ++frame)
    {
        std::vector<float> score(n);
        std::vector<std::uint8_t> backface(n, 0);
        for (std::uint32_t i = 0; i < n; ++i)
        {
            score[i] = scoreDist(rng);
        }

        // Drive BOTH fronts to the identical settled starting front (applyView == oracleUpdate),
        // then run each side's own repair from there.
        ActiveFront oracle = ActiveFront::build(m.verts, m.indices, collapses);
        ParallelFront parallel = ParallelFront::build(m.verts, m.indices, collapses);
        oracleUpdate(oracle, score, backface, budget, coarsenBudget);
        parallel.applyView(score, backface, budget, coarsenBudget);

        oracle.repairFront(m.verts, world, cam, viewProj, vw, vh, cull);
        parallel.repairFront(m.verts, world, cam, viewProj, vw, vh, cull);

        const std::size_t seqTris = emittedTriangles(oracle, m.verts, m.indices);
        const std::size_t parTris = emittedTriangles(parallel, m.verts, m.indices);

        // The only hard bound: neither repaired front exceeds the full finest detail.
        CHECK(parTris <= finestTris);
        CHECK(seqTris <= finestTris);

        const auto diff = static_cast<long long>(parTris) - static_cast<long long>(seqTris);
        const double ratio =
            seqTris > 0 ? static_cast<double>(parTris) / static_cast<double>(seqTris) : 0.0;
        WARN("frame " << frame << ": parallel " << parTris << " tris, sequential " << seqTris
                      << " tris, diff " << diff << ", ratio " << ratio << ", finest " << finestTris
                      << " ("
                      << (100.0 * static_cast<double>(parTris) /
                          static_cast<double>(std::max<std::size_t>(1, finestTris)))
                      << "% of finest); parallel detection passes "
                      << parallel.repairDetectionPasses() << ", apply rounds "
                      << parallel.repairApplyRounds() << ", splits refined "
                      << parallel.repairRefinedSplits());
    }
}

TEST_CASE("ParallelFront::repairFront honours the mesh-context contract", "[vdpm]")
{
    const Mesh m = makeUvSphere(10, 14);
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const Vec3 cam{0.0f, 0.0f, 3.0f};
    const Mat4 viewProj =
        Mat4::perspective(1.0f, 1.0f, 0.1f, 100.0f) * Mat4::lookAt(cam, {0, 0, 0}, {0, 1, 0});

    SECTION("a scheduling-only front has no mesh context ⇒ repairFront throws")
    {
        ParallelFront sched = ParallelFront::build(buildVertexForest(m.verts, collapses));
        REQUIRE_FALSE(sched.hasMeshContext());
        REQUIRE_THROWS_AS(
            sched.repairFront(m.verts, Mat4::identity(), cam, viewProj, 1000.0f, 1000.0f, true),
            std::logic_error);
    }
    SECTION("an empty-mesh full build has context and converges cleanly (no apply rounds)")
    {
        ParallelFront empty = ParallelFront::build({}, {}, {});
        REQUIRE(empty.hasMeshContext());
        REQUIRE_NOTHROW(
            empty.repairFront({}, Mat4::identity(), cam, viewProj, 1000.0f, 1000.0f, true));
        CHECK(empty.repairDetectionPasses() == 1); // one trivial scan of zero faces, then converged
        CHECK(empty.repairApplyRounds() == 0);
        CHECK(empty.repairRefinedSplits() == 0);
    }
    SECTION("a wrong-sized vertex span is rejected")
    {
        ParallelFront full = ParallelFront::build(m.verts, m.indices, collapses);
        std::vector<Vertex> truncated(m.verts.begin(), m.verts.end() - 1);
        REQUIRE_THROWS_AS(
            full.repairFront(truncated, Mat4::identity(), cam, viewProj, 1000.0f, 1000.0f, true),
            std::runtime_error);
    }
}

TEST_CASE("ParallelFront::emitActiveIndices is byte-identical to the oracle", "[vdpm]")
{
    // The Stage-0 emit contract: for the SAME front, the GPU-shaped compaction (survival flag →
    // prefix sum → stable scatter, CSR wedge restoration) produces the EXACT index buffer the
    // recursive oracle's emitActiveIndices does — indices, triangle order, AND restored wedges.
    // Both fronts are driven identically (applyView == oracleUpdate, no repair — the repairs
    // legitimately diverge, so emit is compared on identical state), and their front state is
    // asserted identical immediately before each emit so a future SCHEDULER regression surfaces as
    // a front mismatch, never as a spurious emitter failure.
    constexpr float budget = 1.0f;
    constexpr float coarsenBudget = 0.6f;
    for (const Mesh& m :
         {makeGrid(9), makeUvSphere(12, 16), makeUvSphere(24, 32), makeSeamedGrid(9)})
    {
        const QuadricSimplifier simp;
        const auto collapses = simp.collapseSequence(m.verts, m.indices);
        const VertexForest forest = buildVertexForest(m.verts, collapses);
        const auto n = static_cast<std::uint32_t>(forest.splits.size());

        ActiveFront oracle = ActiveFront::build(m.verts, m.indices, collapses);
        ParallelFront parallel = ParallelFront::build(m.verts, m.indices, collapses);

        auto assertIdenticalFrontThenEmit = [&]()
        {
            for (std::uint32_t v = 0; v < forest.vertexCount; ++v)
            {
                REQUIRE(oracle.active(v) == parallel.active(v)); // scheduler parity guard
            }
            for (std::uint32_t s = 0; s < n; ++s)
            {
                REQUIRE(oracle.refined(s) == parallel.refined(s));
            }
            const std::vector<std::uint32_t> par = parallel.emitActiveIndices(m.verts, m.indices);
            const std::vector<std::uint32_t> ora = oracle.emitActiveIndices(m.verts, m.indices);
            CHECK(par == ora); // indices, order, and wedges
            CHECK(par.size() % 3 == 0);
        };

        // Coarsest front (as built).
        assertIdenticalFrontThenEmit();

        // A sequence of pseudo-random views (a moving front).
        std::mt19937 rng(0xE312Du);
        std::uniform_real_distribution<float> scoreDist(0.0f, 2.0f);
        std::size_t emittedSomething = 0;
        for (int frame = 0; frame < 6; ++frame)
        {
            std::vector<float> score(n);
            std::vector<std::uint8_t> backface(n);
            for (std::uint32_t i = 0; i < n; ++i)
            {
                score[i] = scoreDist(rng);
                backface[i] = (rng() & 7u) == 0u ? 1 : 0;
            }
            oracleUpdate(oracle, score, backface, budget, coarsenBudget);
            parallel.applyView(score, backface, budget, coarsenBudget);
            assertIdenticalFrontThenEmit();
            emittedSomething += parallel.emitActiveIndices(m.verts, m.indices).size();
        }
        CHECK(emittedSomething > 0); // the views actually produced geometry

        // Full detail (over-budget everywhere ⇒ every split refined).
        oracle.refineAll();
        parallel.applyView(std::vector<float>(n, 9.0f), std::vector<std::uint8_t>(n, 0), budget,
                           coarsenBudget);
        assertIdenticalFrontThenEmit();
        // At full detail the emit reproduces the finest index stream (welded, degenerate-free).
        CHECK(parallel.emitActiveIndices(m.verts, m.indices).size() ==
              parallel.finestFaces().size() * 3);
    }
}

TEST_CASE("ParallelFront::emitActiveIndices: empty / all-degenerate / malformed", "[vdpm]")
{
    const Mesh m = makeUvSphere(10, 14);
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);

    SECTION("empty-mesh full build emits nothing (the zero-face prefix-sum path)")
    {
        ParallelFront empty = ParallelFront::build({}, {}, {});
        const auto out = empty.emitActiveIndices({}, {});
        CHECK(out.empty());
    }
    SECTION("an all-degenerate index stream emits nothing (faces drop, no last-element read)")
    {
        ParallelFront full = ParallelFront::build(m.verts, m.indices, collapses);
        // Every triangle repeats a corner ⇒ two welded corners share an ancestor ⇒ all dropped.
        std::vector<std::uint32_t> degenerate;
        for (std::uint32_t v = 0; v + 1 < static_cast<std::uint32_t>(m.verts.size()); ++v)
        {
            degenerate.insert(degenerate.end(), {v, v, v + 1});
        }
        const auto out = full.emitActiveIndices(m.verts, degenerate);
        CHECK(out.empty());
    }
    SECTION("a malformed (non-multiple-of-3) index stream is rejected")
    {
        ParallelFront full = ParallelFront::build(m.verts, m.indices, collapses);
        std::vector<std::uint32_t> bad{0, 1, 2, 3}; // 4 indices
        REQUIRE_THROWS_AS(full.emitActiveIndices(m.verts, bad), std::runtime_error);
    }
    SECTION("an out-of-range index is rejected")
    {
        ParallelFront full = ParallelFront::build(m.verts, m.indices, collapses);
        std::vector<std::uint32_t> bad{0, 1, static_cast<std::uint32_t>(m.verts.size())}; // == size
        REQUIRE_THROWS_AS(full.emitActiveIndices(m.verts, bad), std::runtime_error);
    }
    SECTION("a wrong-sized vertex span is rejected")
    {
        ParallelFront full = ParallelFront::build(m.verts, m.indices, collapses);
        std::vector<Vertex> truncated(m.verts.begin(), m.verts.end() - 1);
        REQUIRE_THROWS_AS(full.emitActiveIndices(truncated, m.indices), std::runtime_error);
    }
    SECTION("a scheduling-only front has no mesh context ⇒ emit throws")
    {
        ParallelFront sched = ParallelFront::build(buildVertexForest(m.verts, collapses));
        REQUIRE_THROWS_AS(sched.emitActiveIndices(m.verts, m.indices), std::logic_error);
    }
}

TEST_CASE("ParallelFront: applyView → repairFront → emitActiveIndices is self-consistent", "[vdpm]")
{
    // The REAL per-frame lifecycle. The parallel repair reaches a different valid front than the
    // oracle, so this can't compare to the oracle emit — instead it checks the emit against an
    // INDEPENDENT recomputation from the settled front: the surviving faces are exactly the
    // non-degenerate ones, IN ORIGINAL ORDER, and each emitted corner is the nearestWedge of its
    // original corner into its active ancestor's CSR bucket (and hence a member of that bucket).
    constexpr float budget = 1.0f;
    constexpr float coarsenBudget = 0.6f;
    constexpr float vw = 1000.0f;
    constexpr float vh = 1000.0f;
    const Vec3 cam{0.0f, 0.0f, 2.4f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj =
        Mat4::perspective(1.0f, 1.0f, 0.05f, 100.0f) * Mat4::lookAt(cam, {0, 0, 0}, {0, 1, 0});

    for (const bool cull : {true, false})
    {
        for (const Mesh& m : {makeUvSphere(20, 28), makeSeamedGrid(9)})
        {
            const QuadricSimplifier simp;
            const auto collapses = simp.collapseSequence(m.verts, m.indices);
            const VertexForest forest = buildVertexForest(m.verts, collapses);
            const auto n = static_cast<std::uint32_t>(forest.splits.size());
            const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(m.verts);

            ParallelFront front = ParallelFront::build(m.verts, m.indices, collapses);
            auto ancestor = [&](std::uint32_t v)
            {
                while (!front.active(v))
                {
                    v = front.forest().splits[front.forest().removingSplit[v]].parent;
                }
                return v;
            };

            std::mt19937 rng(0x11CEu);
            std::uniform_real_distribution<float> scoreDist(0.0f, 2.0f);
            std::size_t totalEmitted = 0;
            for (int frame = 0; frame < 4; ++frame)
            {
                std::vector<float> score(n);
                const std::vector<std::uint8_t> backface(n, 0);
                for (std::uint32_t i = 0; i < n; ++i)
                {
                    score[i] = scoreDist(rng);
                }
                front.applyView(score, backface, budget, coarsenBudget);
                front.repairFront(m.verts, world, cam, viewProj, vw, vh, cull);
                front.validateInvariants();

                const std::vector<std::uint32_t> out = front.emitActiveIndices(m.verts, m.indices);

                // Walk the original faces in order; each survivor must occupy the next output slot
                // (proves the compaction is stable + in original order), with wedge-restored
                // corners.
                std::size_t slot = 0;
                for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3)
                {
                    const std::array<std::uint32_t, 3> oc{m.indices[i], m.indices[i + 1],
                                                          m.indices[i + 2]};
                    const std::array<std::uint32_t, 3> anc{
                        ancestor(weld[oc[0]]), ancestor(weld[oc[1]]), ancestor(weld[oc[2]])};
                    if (anc[0] == anc[1] || anc[1] == anc[2] || anc[0] == anc[2])
                    {
                        continue; // degenerate at this front — dropped
                    }
                    REQUIRE((slot * 3) + 2 < out.size());
                    for (std::uint32_t k = 0; k < 3; ++k)
                    {
                        const std::uint32_t emitted = out[(slot * 3) + k];
                        const std::span<const std::uint32_t> bucket =
                            front.wedgesCsr().forCanonical(anc[k]);
                        // Belongs to the active ancestor's wedge bucket, and is exactly the wedge
                        // nearestWedge would restore for this original corner.
                        CHECK(weld[emitted] == anc[k]);
                        CHECK(std::ranges::find(bucket, emitted) != bucket.end());
                        CHECK(emitted ==
                              mesh_topology::nearestWedge(m.verts, bucket, m.verts[oc[k]]));
                    }
                    ++slot;
                }
                CHECK(out.size() == slot * 3); // no trailing/extra output beyond the survivors
                totalEmitted += out.size();
            }
            CHECK(totalEmitted > 0); // the lifecycle actually produced geometry
        }
    }
}

TEST_CASE("ParallelFront wedge-ABI evidence (CSR vs a precomputed corner→wedge map)", "[vdpm]")
{
    // Evidence for the Stage-0 gate: the CSR wedge memory (the chosen ABI) vs an upper bound on the
    // ALTERNATIVE precomputed corner→wedge map (one entry per finest-face corner per possible
    // active ancestor up its removal chain — it can grow ~Σ chain lengths and dwarf the CSR).
    // Reported so "CSR-first" is evidence-based, not assumed.
    for (const Mesh& m : {makeUvSphere(24, 32), makeSeamedGrid(17)})
    {
        const QuadricSimplifier simp;
        const auto collapses = simp.collapseSequence(m.verts, m.indices);
        const ParallelFront front = ParallelFront::build(m.verts, m.indices, collapses);
        const VertexForest& forest = front.forest();
        const mesh_topology::CanonicalWedgesCsr& csr = front.wedgesCsr();

        std::size_t maxWedges = 0;
        std::size_t multiWedgeCanonicals = 0;
        for (std::uint32_t c = 0; c < forest.vertexCount; ++c)
        {
            const std::size_t w = csr.forCanonical(c).size();
            maxWedges = std::max(maxWedges, w);
            multiWedgeCanonicals += w > 1 ? 1 : 0;
        }
        const std::size_t csrBytes =
            (csr.wedges.size() + csr.offsets.size()) * sizeof(std::uint32_t);

        // Upper bound on the precomputed map: for each finest-face corner, one wedge entry per
        // ancestor along its removal chain (its possible active ancestors across all fronts).
        auto chainLength = [&](std::uint32_t v)
        {
            std::size_t len = 1;
            while (forest.removingSplit[v] != kNoSplit)
            {
                v = forest.splits[forest.removingSplit[v]].parent;
                ++len;
            }
            return len;
        };
        std::size_t precomputedEntries = 0;
        for (const std::array<std::uint32_t, 3>& f : front.finestFaces())
        {
            for (const std::uint32_t c : f)
            {
                precomputedEntries += chainLength(c);
            }
        }
        const std::size_t precomputedBytes = precomputedEntries * sizeof(std::uint32_t);

        WARN("wedges: total " << csr.wedges.size() << ", max/canonical " << maxWedges
                              << ", multi-wedge canonicals " << multiWedgeCanonicals << "; CSR "
                              << csrBytes << " B vs precomputed-map upper bound "
                              << precomputedBytes << " B ("
                              << (static_cast<double>(precomputedBytes) /
                                  static_cast<double>(std::max<std::size_t>(1, csrBytes)))
                              << "x the CSR)");
        CHECK(csrBytes > 0);
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

TEST_CASE("validateFrontInvariants: passes a consistent front, rejects each tampered invariant",
          "[vdpm]")
{
    // The shared first-principles validator (also called by the GPU B3 harness on read-back state).
    // Base case: the diamond with only split 0 refined — vertex 1 active, dependents_[0] == 2
    // (split 0's parent + vl both reference the root vertex 0).
    const VertexForest forest = makeDiamondForest();
    const std::vector<std::uint8_t> active{1, 1, 0, 0, 0};
    const std::vector<std::uint8_t> refined{1, 0, 0, 0};
    const std::vector<std::uint32_t> dependents{2, 0, 0, 0, 0};
    CHECK_NOTHROW(validateFrontInvariants(forest, active, refined, dependents));

    SECTION("a refined split with an inactive dependency throws")
    {
        std::vector<std::uint8_t> a = active;
        a[0] = 0; // split 0 is refined but its parent/vl (root vertex 0) is now inactive
        CHECK_THROWS_AS(validateFrontInvariants(forest, a, refined, dependents), std::logic_error);
    }
    SECTION("a dependents count mismatch throws")
    {
        std::vector<std::uint32_t> d = dependents;
        d[0] = 1; // should be 2 (per-slot: split 0's parent + vl)
        CHECK_THROWS_AS(validateFrontInvariants(forest, active, refined, d), std::logic_error);
    }
    SECTION("an active/refined inconsistency throws")
    {
        std::vector<std::uint8_t> a = active;
        a[1] = 0; // vertex 1's removing split (0) is refined, so it must be active
        CHECK_THROWS_AS(validateFrontInvariants(forest, a, refined, dependents), std::logic_error);
    }
    SECTION("a wrong-sized state span throws")
    {
        std::vector<std::uint8_t> shortActive{1, 1, 0, 0}; // vertexCount is 5
        CHECK_THROWS_AS(validateFrontInvariants(forest, shortActive, refined, dependents),
                        std::logic_error);
    }
    SECTION("the uint32 (GPU read-back) overload accepts the same consistent front")
    {
        const std::vector<std::uint32_t> active32{1, 1, 0, 0, 0};
        const std::vector<std::uint32_t> refined32{1, 0, 0, 0};
        CHECK_NOTHROW(validateFrontInvariants(forest, active32, refined32, dependents));
    }
    SECTION("a flag that is not exactly 0 or 1 is rejected — NO lossy narrowing")
    {
        // The whole point of the templated (non-narrowing) validator: a corrupt GPU flag whose low
        // bits alias 0/1 (256 → 0, 257 → 1 under a uint8 truncation) must still be caught. A uint8
        // '2' is likewise not a truthy "active".
        std::vector<std::uint32_t> a32{1, 1, 0, 0, 0};
        a32[2] = 256u; // would truncate to 0 (a "false") — must throw instead
        CHECK_THROWS_AS(validateFrontInvariants(forest, a32, std::vector<std::uint32_t>{1, 0, 0, 0},
                                                dependents),
                        std::logic_error);
        a32[2] = 257u; // would truncate to 1 (a spurious "true") — must throw instead
        CHECK_THROWS_AS(validateFrontInvariants(forest, a32, std::vector<std::uint32_t>{1, 0, 0, 0},
                                                dependents),
                        std::logic_error);

        std::vector<std::uint8_t> a8 = active;
        a8[2] = 2; // a nonzero-but-not-1 uint8 is corruption, not "active"
        CHECK_THROWS_AS(validateFrontInvariants(forest, a8, refined, dependents), std::logic_error);
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
