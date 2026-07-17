#include <fire_engine/graphics/vdpm_parallel.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <fire_engine/graphics/mesh_topology.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

DependencyDag buildDependencyDag(const VertexForest& forest)
{
    validateForest(forest); // reject a malformed forest before it could fault a shader

    const std::size_t n = forest.splits.size();

    // Edges point dependency → dependent (`removingSplit[dep]` → `s`), plus an in-degree per split,
    // so a Kahn topological sweep can both order the DAG and compute longest-path ranks. A repeated
    // edge (two of parent/vl/vr share a remover) is harmless: it bumps in-degree and is decremented
    // the same number of times.
    std::vector<std::vector<std::uint32_t>> dependents(n);
    std::vector<std::uint32_t> inDegree(n, 0);
    auto addDependency = [&](std::uint32_t split, std::uint32_t depVertex)
    {
        if (depVertex == kInvalidVertex)
        {
            return; // boundary edge: no vr
        }
        const std::uint32_t depSplit = forest.removingSplit[depVertex];
        if (depSplit == kNoSplit)
        {
            return; // a root vertex is always active — no split has to fire first
        }
        dependents[depSplit].push_back(split);
        ++inDegree[split];
    };
    for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(n); ++s)
    {
        const VertexSplit& sp = forest.splits[s];
        addDependency(s, sp.parent);
        addDependency(s, sp.vl);
        addDependency(s, sp.vr);
    }

    // Kahn's algorithm: seed with the dependency-free splits (rank 0), and each time a split's last
    // dependency is satisfied, its rank is one past the deepest dependency. `queue` doubles as the
    // topological order; if it doesn't cover every split, an in-degree never reached 0 ⇒ a cycle.
    std::vector<std::uint32_t> rank(n, 0);
    std::vector<std::uint32_t> queue;
    queue.reserve(n);
    for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(n); ++s)
    {
        if (inDegree[s] == 0)
        {
            queue.push_back(s);
        }
    }
    for (std::size_t qi = 0; qi < queue.size(); ++qi)
    {
        const std::uint32_t s = queue[qi];
        for (const std::uint32_t d : dependents[s])
        {
            rank[d] = std::max(rank[d], rank[s] + 1);
            if (--inDegree[d] == 0)
            {
                queue.push_back(d);
            }
        }
    }
    if (queue.size() != n)
    {
        throw std::runtime_error("VDPM dependency graph has a cycle (unrefineable forest)");
    }

    DependencyDag dag;
    dag.maxRank = rank.empty() ? 0 : *std::ranges::max_element(rank);

    // Pack the splits by rank into the CSR layout (a counting sort = prefix-sum + scatter, the same
    // shape the GPU uploader uses): rank r occupies splitsByRank[rankOffsets[r] ..
    // rankOffsets[r+1]).
    dag.rankOffsets.assign(dag.maxRank + 2, 0);
    for (const std::uint32_t r : rank)
    {
        ++dag.rankOffsets[r + 1];
    }
    for (std::uint32_t r = 0; r < dag.maxRank + 1; ++r)
    {
        dag.rankOffsets[r + 1] += dag.rankOffsets[r];
    }
    dag.splitsByRank.resize(n);
    std::vector<std::uint32_t> cursor(dag.rankOffsets.begin(), dag.rankOffsets.end() - 1);
    for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(n); ++s)
    {
        dag.splitsByRank[cursor[rank[s]]++] = s;
    }
    dag.rank = std::move(rank);
    return dag;
}

ParallelFront ParallelFront::build(const VertexForest& forest)
{
    ParallelFront front;
    front.forest_ = forest;
    front.finishBuild();
    return front;
}

ParallelFront ParallelFront::build(std::span<const Vertex> vertices,
                                   std::span<const std::uint32_t> indices,
                                   std::span<const MeshCollapse> collapses)
{
    ParallelFront front;
    front.forest_ = buildVertexForest(vertices, collapses);
    // The mesh context the repairs + emit need, built exactly like ActiveFront::build so both walk
    // the identical topology.
    front.weld_ = mesh_topology::weldByPosition(vertices);
    front.finestFaces_ = mesh_topology::canonicalFaces(front.weld_, indices);
    front.wedgesCsr_ = mesh_topology::canonicalWedgesCsr(front.weld_);
    front.hasMeshContext_ = true;
    front.finishBuild();
    return front;
}

void ParallelFront::finishBuild()
{
    dag_ = buildDependencyDag(forest_); // validates the forest + builds the CSR rank layout

    const std::size_t n = forest_.splits.size();
    // Coarsest state: only never-removed (root) canonical vertices are active; no split refined.
    active_.assign(forest_.vertexCount, 0);
    for (std::uint32_t v = 0; v < forest_.vertexCount; ++v)
    {
        if (forest_.removingSplit[v] == kNoSplit)
        {
            active_[v] = 1;
        }
    }
    refined_.assign(n, 0);
    dependents_.assign(forest_.vertexCount, 0);
    required_.assign(n, 0);
}

bool ParallelFront::refineOne(std::uint32_t splitIndex)
{
    if (refined_[splitIndex] != 0)
    {
        return false;
    }
    const VertexSplit& s = forest_.splits[splitIndex];
    if (active_[s.parent] == 0 || active_[s.vl] == 0 ||
        (s.vr != kInvalidVertex && active_[s.vr] == 0))
    {
        return false;
    }
    refined_[splitIndex] = 1;
    active_[s.child] = 1;
    ++dependents_[s.parent];
    ++dependents_[s.vl];
    if (s.vr != kInvalidVertex)
    {
        ++dependents_[s.vr];
    }
    return true;
}

bool ParallelFront::coarsenOne(std::uint32_t splitIndex)
{
    if (refined_[splitIndex] == 0)
    {
        return false;
    }
    const VertexSplit& s = forest_.splits[splitIndex];
    if (dependents_[s.child] != 0)
    {
        return false; // the child props up a refined split — not a leaf
    }
    refined_[splitIndex] = 0;
    active_[s.child] = 0;
    --dependents_[s.parent];
    --dependents_[s.vl];
    if (s.vr != kInvalidVertex)
    {
        --dependents_[s.vr];
    }
    return true;
}

std::uint32_t ParallelFront::activeAncestor(std::uint32_t canonicalVertex) const
{
    // A root is always active and has removingSplit == kNoSplit, so an inactive vertex always has a
    // valid removing split whose parent is one step nearer an active ancestor.
    std::uint32_t v = canonicalVertex;
    while (active_[v] == 0)
    {
        v = forest_.splits[forest_.removingSplit[v]].parent;
    }
    return v;
}

std::span<const std::uint32_t> ParallelFront::rankSplits(std::uint32_t r) const
{
    return std::span{dag_.splitsByRank}.subspan(dag_.rankOffsets[r],
                                                dag_.rankOffsets[r + 1] - dag_.rankOffsets[r]);
}

std::uint32_t ParallelFront::closeAndApplyRequired()
{
    // CLOSE the already-seeded required set over the dependency DAG in DESCENDING rank: a required
    // split's dependency splits (which remove its parent/vl/vr and have strictly lower rank) must
    // also be required, so one high→low pass reaches every one before it is itself visited. (On the
    // GPU the marking is an atomic OR — several same-rank splits may mark the same dependency.)
    for (std::uint32_t r = dag_.maxRank + 1; r-- > 0;)
    {
        for (const std::uint32_t s : rankSplits(r))
        {
            if (required_[s] == 0)
            {
                continue;
            }
            const VertexSplit& sp = forest_.splits[s];
            for (const std::uint32_t dep : {sp.parent, sp.vl, sp.vr})
            {
                if (dep == kInvalidVertex)
                {
                    continue;
                }
                const std::uint32_t depSplit = forest_.removingSplit[dep];
                if (depSplit != kNoSplit)
                {
                    required_[depSplit] = 1u;
                }
            }
        }
    }

    // APPLY refinements in ASCENDING rank: each split's dependencies (lower rank) are already
    // refined, so its parent/vl/vr are active and refineOne succeeds — the non-recursive
    // forceRefine. A required, not-yet-refined split that FAILS to refine means its dependencies
    // weren't satisfied by rank order — a closure/rank bug, so treat it as an invariant violation,
    // not a silent skip.
    std::uint32_t refinedThisCall = 0;
    for (std::uint32_t r = 0; r <= dag_.maxRank; ++r)
    {
        for (const std::uint32_t s : rankSplits(r))
        {
            if (required_[s] != 0 && refined_[s] == 0)
            {
                if (!refineOne(s))
                {
                    throw std::logic_error("VDPM closeAndApplyRequired: required refine failed "
                                           "(unsatisfied dependency despite rank order)");
                }
                ++refinedThisCall;
            }
        }
    }
    return refinedThisCall;
}

void ParallelFront::applyView(std::span<const float> splitScore,
                              std::span<const std::uint8_t> splitBackface, float pixelBudget,
                              float coarsenBudget)
{
    const auto n = static_cast<std::uint32_t>(forest_.splits.size());
    if (splitScore.size() != n || splitBackface.size() != n)
    {
        throw std::runtime_error("VDPM applyView: score/backface span size != split count");
    }

    // (1) Seed the required set: every over-budget, non-back-facing split (refineForView's refine
    // predicate), then (2) close it over the DAG and apply in rank order (shared with the snapshot
    // repair). The refine count is irrelevant to a view update.
    for (std::uint32_t s = 0; s < n; ++s)
    {
        required_[s] = (splitBackface[s] == 0 && splitScore[s] > pixelBudget) ? 1u : 0u;
    }
    (void)closeAndApplyRequired();

    // (3) Coarsen in DESCENDING rank: a refined split under the coarsen budget (or back-facing)
    // whose child is a leaf collapses. A split's dependents have strictly higher rank, so
    // processing high→low coarsens them first (freeing the child) — the fine-first fixpoint in a
    // single pass.
    for (std::uint32_t r = dag_.maxRank + 1; r-- > 0;)
    {
        for (const std::uint32_t s : rankSplits(r))
        {
            if (refined_[s] != 0 && (splitBackface[s] != 0 || splitScore[s] < coarsenBudget))
            {
                coarsenOne(s);
            }
        }
    }
}

void ParallelFront::repairFront(std::span<const Vertex> vertices, const Mat4& world,
                                const Vec3& cameraPos, const Mat4& viewProj, float viewportWidth,
                                float viewportHeight, bool rasterBackfaceCulling)
{
    if (!hasMeshContext_)
    {
        // A scheduling-only front has no finest faces / weld, so there is nothing to repair and no
        // way to detect a violation — calling it is a usage error, not a silent no-op.
        throw std::logic_error("VDPM ParallelFront::repairFront: front has no mesh context");
    }
    if (vertices.size() != forest_.vertexCount)
    {
        // Canonical corner IDs (in finestFaces_ / their ancestors) index into the ORIGINAL vertex
        // array that built the mesh context — a wrong-sized span would be out-of-bounds. Reject it
        // rather than read past the end.
        throw std::runtime_error(
            "VDPM ParallelFront::repairFront: vertex span size != vertexCount");
    }
    repairDetectionPasses_ = 0;
    repairApplyRounds_ = 0;
    repairRefinedSplits_ = 0;
    // Upper bound on detection passes: each apply round refines >= 1 previously-inactive split, so
    // the apply rounds can't exceed the initially-unrefined count, and the detection passes are one
    // more (the final convergence-proving pass that marks nothing). A run past that is a
    // non-terminating detector — throw rather than spin.
    const auto initiallyUnrefined =
        static_cast<std::uint32_t>(std::ranges::count(refined_, std::uint8_t(0)));

    auto worldPos = [&](std::uint32_t v)
    {
        const Vec3 l = vertices[v].position();
        const Vec4 w = world * Vec4{l.x(), l.y(), l.z(), 1.0f};
        return Vec3{w.x(), w.y(), w.z()};
    };

    while (true)
    {
        // One DETECTION PASS: scan EVERY finest face against the current SETTLED front and mark
        // each violation's inactive-corner removing split required — a snapshot pass (no mutation
        // while detecting), unlike the sequential sweeps. Marking (not applying) is what lets the
        // whole pass run in parallel. Counted even when it marks nothing (the terminal convergence
        // pass) — it is a real GPU detect dispatch.
        ++repairDetectionPasses_;
        if (repairDetectionPasses_ > initiallyUnrefined + 1)
        {
            throw std::logic_error("VDPM ParallelFront::repairFront: exceeded the inflationary "
                                   "detection-pass bound (non-terminating detector)");
        }
        std::ranges::fill(required_, 0u);
        bool markedAny = false;
        auto markInactiveCorner = [&](std::uint32_t corner)
        {
            if (active_[corner] != 0)
            {
                return; // already at finest here — nothing to advance
            }
            required_[forest_.removingSplit[corner]] = 1u;
            markedAny = true;
        };

        for (const std::array<std::uint32_t, 3>& fc : finestFaces_)
        {
            const std::uint32_t a0 = activeAncestor(fc[0]);
            const std::uint32_t a1 = activeAncestor(fc[1]);
            const std::uint32_t a2 = activeAncestor(fc[2]);
            const bool degenerate = (a0 == a1 || a1 == a2 || a0 == a2);
            const std::array<Vec3, 3> original{worldPos(fc[0]), worldPos(fc[1]), worldPos(fc[2])};
            const std::array<Vec3, 3> replacement{worldPos(a0), worldPos(a1), worldPos(a2)};
            const std::array<bool, 3> inactive{active_[fc[0]] == 0, active_[fc[1]] == 0,
                                               active_[fc[2]] == 0};

            // Foldover ∪ coverage, via the SAME shared classifiers as the sequential sweeps — the
            // two detectors differ only in WHEN they apply (snapshot-mark here vs mutate-in-sweep).
            if (detail::isFoldover(original, replacement, degenerate))
            {
                for (const std::uint32_t c : fc)
                {
                    markInactiveCorner(c);
                }
            }
            const detail::CoverageRepair repair = detail::classifyCoverageRepair(
                original, replacement, degenerate, inactive, cameraPos, viewProj, viewportWidth,
                viewportHeight, rasterBackfaceCulling);
            switch (repair.kind)
            {
            case detail::CoverageRepairKind::None:
                break;
            case detail::CoverageRepairKind::AllInactiveCorners:
                for (const std::uint32_t c : fc)
                {
                    markInactiveCorner(c);
                }
                break;
            case detail::CoverageRepairKind::WorstInactiveCorner:
                markInactiveCorner(fc[repair.worstCorner]);
                break;
            }
        }

        if (!markedAny)
        {
            break; // the settled front has zero foldover + coverage violations — converged
        }
        // One APPLY ROUND: close the marked set over the DAG and apply in rank order. Every marked
        // split is an inactive corner's removing split (⇒ not yet refined), so the apply MUST
        // refine
        // >= 1; zero refines despite marks is a closure/rank bug.
        const std::uint32_t refined = closeAndApplyRequired();
        if (refined == 0)
        {
            throw std::logic_error("VDPM ParallelFront::repairFront: detected violations but "
                                   "refined nothing (closure/rank bug)");
        }
        repairRefinedSplits_ += refined;
        ++repairApplyRounds_;
    }
}

void ParallelFront::emitActiveIndices(std::span<const Vertex> vertices,
                                      std::span<const std::uint32_t> indices,
                                      std::vector<std::uint32_t>& out) const
{
    if (!hasMeshContext_)
    {
        throw std::logic_error("VDPM ParallelFront::emitActiveIndices: front has no mesh context");
    }
    if (vertices.size() != forest_.vertexCount)
    {
        throw std::runtime_error(
            "VDPM ParallelFront::emitActiveIndices: vertex span size != vertexCount");
    }
    if (indices.size() % 3 != 0)
    {
        // A malformed index stream (trailing 1-2 dangling values) is rejected, not silently
        // truncated — dropping vertices would emit a subtly wrong mesh.
        throw std::runtime_error(
            "VDPM ParallelFront::emitActiveIndices: index count is not a multiple of 3");
    }
    // GPU-sized arithmetic: face survival offsets and the output index count are uint32 on the GPU,
    // and the output count is bounded by the input (survivors * 3 <= indices.size()), so requiring
    // the index count to fit uint32 pins every count in the emit to 32 bits.
    if (indices.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::runtime_error(
            "VDPM ParallelFront::emitActiveIndices: index count exceeds 32-bit indexing");
    }
    // Every index must reference a real vertex — it is used to read weld_[idx] and vertices[idx],
    // so an out-of-range corner would be undefined behaviour. The API accepts a caller-supplied
    // stream (not only the guaranteed-valid stored one), so validate it; on the GPU this is the
    // upload-time gate on the static index buffer, analogous to validateForest.
    for (const std::uint32_t idx : indices)
    {
        if (idx >= vertices.size())
        {
            throw std::runtime_error(
                "VDPM ParallelFront::emitActiveIndices: index references an out-of-range vertex");
        }
    }
    const auto faceCount = static_cast<std::uint32_t>(indices.size() / 3);

    // Phase 1 — memoise activeAncestor(c) per canonical vertex. The front is settled at emit, so it
    // is stable; a vertex is a corner of many faces, so this avoids re-walking the parent chain per
    // corner. A parallel map on the GPU.
    ancestorCache_.assign(active_.size(), 0);
    for (std::uint32_t c = 0; c < ancestorCache_.size(); ++c)
    {
        ancestorCache_[c] = activeAncestor(c);
    }

    // Phase 2 — a SURVIVAL FLAG per original face: its three welded corners' active ancestors are
    // all distinct (else the face collapsed to a degenerate at this front and is dropped). Parallel
    // map — each face independent, reading only the ancestor cache.
    faceSurvive_.assign(faceCount, 0);
    for (std::uint32_t f = 0; f < faceCount; ++f)
    {
        const std::uint32_t a0 = ancestorCache_[weld_[indices[(3 * f) + 0]]];
        const std::uint32_t a1 = ancestorCache_[weld_[indices[(3 * f) + 1]]];
        const std::uint32_t a2 = ancestorCache_[weld_[indices[(3 * f) + 2]]];
        faceSurvive_[f] = (a0 != a1 && a1 != a2 && a0 != a2) ? 1u : 0u;
    }

    // Phase 3 — EXCLUSIVE PREFIX SUM of the flags → each surviving face's output slot (= survivors
    // before it = where a sequential append would place it, so the result is byte-identical to the
    // oracle's push_back order). A serial scan here; a work-efficient parallel scan on the GPU. The
    // running total after the last face is the surviving-face count — computed WITHOUT a
    // last-element read, so an empty face stream (faceCount == 0) yields zero cleanly.
    faceOutSlot_.assign(faceCount, 0);
    std::uint32_t running = 0;
    for (std::uint32_t f = 0; f < faceCount; ++f)
    {
        faceOutSlot_[f] = running;
        running += faceSurvive_[f];
    }
    const std::uint32_t survivingFaces = running;

    // Phase 4 — allocate the output ONCE, then STABLE SCATTER each surviving face's three corners
    // at out[3 * slot] (no atomic append; original face order preserved — blend/transmission need
    // it). Each corner is restored to the nearestWedge at its active ancestor's CSR bucket, so a
    // seam corner keeps its own chart/shading identity instead of snapping to one canonical. Every
    // slot in [0, survivingFaces*3) is written exactly once, so resize (no fill) suffices.
    out.resize(static_cast<std::size_t>(survivingFaces) * 3);
    for (std::uint32_t f = 0; f < faceCount; ++f)
    {
        if (faceSurvive_[f] == 0)
        {
            continue;
        }
        const std::size_t base = static_cast<std::size_t>(faceOutSlot_[f]) * 3;
        for (std::uint32_t k = 0; k < 3; ++k)
        {
            const std::uint32_t oc = indices[(3 * f) + k];
            const std::uint32_t anc = ancestorCache_[weld_[oc]];
            out[base + k] =
                mesh_topology::nearestWedge(vertices, wedgesCsr_.forCanonical(anc), vertices[oc]);
        }
    }
}

std::vector<std::uint32_t>
ParallelFront::emitActiveIndices(std::span<const Vertex> vertices,
                                 std::span<const std::uint32_t> indices) const
{
    std::vector<std::uint32_t> out;
    emitActiveIndices(vertices, indices, out);
    return out;
}

void ParallelFront::validateInvariants() const
{
    // Reconstruct the dependent counts from scratch over the refined splits, and check each refined
    // split's dependencies are active as we go.
    std::vector<std::uint32_t> expected(forest_.vertexCount, 0);
    for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(refined_.size()); ++s)
    {
        if (refined_[s] == 0)
        {
            continue;
        }
        const VertexSplit& sp = forest_.splits[s];
        if (active_[sp.parent] == 0 || active_[sp.vl] == 0 ||
            (sp.vr != kInvalidVertex && active_[sp.vr] == 0))
        {
            throw std::logic_error("ParallelFront: a refined split has an inactive dependency");
        }
        ++expected[sp.parent];
        ++expected[sp.vl];
        if (sp.vr != kInvalidVertex)
        {
            ++expected[sp.vr];
        }
    }
    for (std::uint32_t v = 0; v < static_cast<std::uint32_t>(forest_.vertexCount); ++v)
    {
        if (dependents_[v] != expected[v])
        {
            throw std::logic_error("ParallelFront: dependents_ count mismatch");
        }
        // A non-root vertex is active exactly when its removing split is refined; a root is always
        // active.
        const std::uint32_t rs = forest_.removingSplit[v];
        const bool shouldBeActive = rs == kNoSplit || refined_[rs] != 0;
        if ((active_[v] != 0) != shouldBeActive)
        {
            throw std::logic_error("ParallelFront: active/refined inconsistency");
        }
    }
}

} // namespace fire_engine
