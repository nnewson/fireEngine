#include <fire_engine/graphics/vdpm_parallel.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace fire_engine
{

void validateForest(const VertexForest& forest)
{
    const std::size_t n = forest.splits.size();
    const std::size_t vc = forest.vertexCount;
    constexpr std::size_t k32 = std::numeric_limits<std::uint32_t>::max();
    if (n > k32 || vc > k32)
    {
        throw std::runtime_error("VDPM forest exceeds 32-bit indexing");
    }
    if (forest.removingSplit.size() != vc)
    {
        throw std::runtime_error("VDPM forest: removingSplit size != vertexCount");
    }
    auto inRange = [vc](std::uint32_t v) { return static_cast<std::size_t>(v) < vc; };
    for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(n); ++s)
    {
        const VertexSplit& sp = forest.splits[s];
        // parent/child/vl must be real vertices; vr may be the boundary sentinel.
        if (!inRange(sp.parent) || !inRange(sp.child) || !inRange(sp.vl) ||
            (sp.vr != kInvalidVertex && !inRange(sp.vr)))
        {
            throw std::runtime_error("VDPM forest: split references an out-of-range vertex");
        }
        // Each split must be the removing split of its own child — which also proves children are
        // unique (two splits can't both be removingSplit[child]).
        if (forest.removingSplit[sp.child] != s)
        {
            throw std::runtime_error("VDPM forest: split/child removing-split inconsistency");
        }
    }
    for (std::uint32_t v = 0; v < static_cast<std::uint32_t>(vc); ++v)
    {
        const std::uint32_t rs = forest.removingSplit[v];
        if (rs == kNoSplit)
        {
            continue;
        }
        if (static_cast<std::size_t>(rs) >= n)
        {
            throw std::runtime_error("VDPM forest: removingSplit references an out-of-range split");
        }
        // Reverse of the split-loop check: the split that claims to remove `v` must have `v` as its
        // child. Without this a spare vertex can alias another's removing split — the DAG would
        // treat it as activated by that split, but refineOne only activates the split's recorded
        // child.
        if (forest.splits[rs].child != v)
        {
            throw std::runtime_error("VDPM forest: removingSplit/child reverse inconsistency");
        }
    }
}

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
    front.dag_ = buildDependencyDag(forest); // validates the forest + builds the CSR rank layout

    const std::size_t n = forest.splits.size();

    // Coarsest state: only never-removed (root) canonical vertices are active; no split refined.
    front.active_.assign(forest.vertexCount, 0);
    for (std::uint32_t v = 0; v < forest.vertexCount; ++v)
    {
        if (forest.removingSplit[v] == kNoSplit)
        {
            front.active_[v] = 1;
        }
    }
    front.refined_.assign(n, 0);
    front.dependents_.assign(forest.vertexCount, 0);
    front.required_.assign(n, 0);
    return front;
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

void ParallelFront::applyView(std::span<const float> splitScore,
                              std::span<const std::uint8_t> splitBackface, float pixelBudget,
                              float coarsenBudget)
{
    const auto n = static_cast<std::uint32_t>(forest_.splits.size());
    if (splitScore.size() != n || splitBackface.size() != n)
    {
        throw std::runtime_error("VDPM applyView: score/backface span size != split count");
    }

    // rank `r`'s splits, as a contiguous CSR range (a GPU rank dispatch).
    auto rankSplits = [this](std::uint32_t r)
    {
        return std::span{dag_.splitsByRank}.subspan(dag_.rankOffsets[r],
                                                    dag_.rankOffsets[r + 1] - dag_.rankOffsets[r]);
    };

    // (1) Seed the required set: every over-budget, non-back-facing split (refineForView's refine
    // predicate). Then CLOSE it over the dependency DAG in DESCENDING rank: a required split's
    // dependency splits (which remove its parent/vl/vr and have strictly lower rank) must also be
    // required, so one high→low pass reaches every one before it is itself visited. (On the GPU the
    // marking is an atomic OR — several same-rank splits may mark the same dependency.)
    for (std::uint32_t s = 0; s < n; ++s)
    {
        required_[s] = (splitBackface[s] == 0 && splitScore[s] > pixelBudget) ? 1u : 0u;
    }
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

    // (2) Apply refinements in ASCENDING rank: each split's dependencies (lower rank) are already
    // refined, so its parent/vl/vr are active and refineOne succeeds — the non-recursive
    // forceRefine. A required, not-yet-refined split that FAILS to refine means its dependencies
    // weren't satisfied by rank order — a closure/rank bug, so treat it as an invariant violation,
    // not a silent skip.
    for (std::uint32_t r = 0; r <= dag_.maxRank; ++r)
    {
        for (const std::uint32_t s : rankSplits(r))
        {
            if (required_[s] != 0 && refined_[s] == 0 && !refineOne(s))
            {
                throw std::logic_error("VDPM applyView: required refine failed (unsatisfied "
                                       "dependency despite rank order)");
            }
        }
    }

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
