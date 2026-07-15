#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/vdpm.hpp>

namespace fire_engine
{

// GPU-shaped CPU model of the VDPM active-front lifecycle (rendering-spine #3, the "GPU-driven
// front" arc, Stage 0). The production `ActiveFront` (vdpm.hpp) is RECURSIVE: `forceRefine` pulls
// in a split's dependency neighbourhood by recursion, and the repairs are sequential sweeps. A GPU
// can do neither. This module re-expresses the SAME lifecycle as data-parallel passes — the exact
// shape a compute implementation needs — so the hard parallel algorithm can be proven in plain C++,
// tested in CI against the recursive oracle, and its cost measured, BEFORE any GLSL is written. It
// is the arc's stop/go gate. Vulkan-free + headless-testable, like `ActiveFront`.

// The refine-dependency DAG of a vertex forest. Refining split `s` (activating its `child`) needs
// its `parent`, `vl`, and `vr` active; a non-root vertex `v` becomes active only when the split
// that removes it (`removingSplit[v]`) is refined. So `s` DEPENDS on
// `removingSplit[{parent,vl,vr}]` — the recursion `forceRefine` walks. `parent` is monotone
// (removed by a strictly later/coarser collapse) but `vl`/`vr` are NOT, so this is a general DAG,
// not the parent forest; a rank-ordered parallel apply (dependencies' ranks first) is the
// non-recursive analogue of `forceRefine`. `rank[s]` is the longest dependency chain to `s` (roots
// = 0); a split is applied in the pass for its rank, so `maxRank + 1` passes suffice. Building it
// also PROVES acyclicity (a cycle is an unrefineable front / GPU hang) AND structural validity.
//
// GPU-SHAPED LAYOUT: splits are stored packed by rank (`splitsByRank`) with CSR `rankOffsets`, the
// exact representation a compute uploader + shader consume — rank `r`'s splits are
// `splitsByRank[rankOffsets[r] .. rankOffsets[r + 1])`, so a rank dispatch is a contiguous range.
struct DependencyDag
{
    std::vector<std::uint32_t> rank;         // per split: longest dependency chain length (roots 0)
    std::vector<std::uint32_t> splitsByRank; // all split indices, ordered by ascending rank
    std::vector<std::uint32_t> rankOffsets; // size maxRank + 2; rank r = [offsets[r], offsets[r+1])
    std::uint32_t maxRank{0};               // highest rank ⇒ maxRank + 1 rank-ordered passes
};

// Build the dependency DAG + per-split topological ranks + the CSR rank layout. Throws
// std::runtime_error if the forest is CYCLIC or STRUCTURALLY MALFORMED (out-of-range / inconsistent
// references, sizes that don't fit the GPU's 32-bit indexing) — this is the gate before the forest
// is uploaded to the GPU, so it rejects anything a shader would fault on rather than assuming
// validity. See `validateForest`.
[[nodiscard]] DependencyDag buildDependencyDag(const VertexForest& forest);

// Structural validation of a vertex forest, independent of the DAG (called by buildDependencyDag).
// Throws std::runtime_error on any violation: `removingSplit` sized to `vertexCount`; every
// parent/child/vl (and vr unless the boundary sentinel) in range; every removing-split reference
// valid; each split's `child` its own removing split (⇒ children unique); split + vertex counts
// within 32-bit indexing.
void validateForest(const VertexForest& forest);

// A GPU-shaped active front: the same state as `ActiveFront` (per-vertex active, per-split refined,
// per-vertex dependent counts) but updated by RANK-ORDERED DATA-PARALLEL passes over the dependency
// DAG instead of the recursive `forceRefine` + fixpoint sweeps — the exact shape a compute
// implementation needs, with no recursion and no order-dependent iteration. Persistent across
// `applyView` calls, like `ActiveFront`. It takes PRECOMPUTED per-split scores (the scoring is
// separately tested via `detail::coneVisibility` + the channel tests) so the scheduling — the
// genuinely hard parallel algorithm — is validated in isolation against the recursive oracle.
class ParallelFront
{
public:
    [[nodiscard]] static ParallelFront build(const VertexForest& forest);

    // Update the persistent front for one frame. `splitScore` / `splitBackface` are per-split (one
    // entry per forest split — a wrong size throws). Mirrors `refineForView`'s refine + coarsen
    // passes exactly, but as rank-ordered parallel passes: (1) mark every over-budget,
    // non-back-facing split REQUIRED and close the requirement set through the dependency DAG (a
    // required split needs its dependencies refined); (2) apply the refinements in ASCENDING rank
    // (dependencies first); (3) coarsen every refined split now under `coarsenBudget` (or
    // back-facing) whose child is a leaf, in DESCENDING rank (dependents first).
    //
    // RANK-LOCAL CONCURRENCY CONTRACT (for the GLSL port): splits within a rank are CAUSALLY
    // independent (none is another's dependency), so they can run in one dispatch — but they are
    // NOT memory-independent: several may mark the same `required_` entry (closure) or add/subtract
    // the same `dependents_[v]` (refine/coarsen). The GPU therefore needs an atomic OR on
    // `required_` (hence uint32_t), atomic add/sub on `dependents_`, and a shader-storage
    // write→read barrier BETWEEN rank dispatches. The CPU model runs a rank serially, so the shared
    // writes are race-free here; the contract is documented so the port preserves it.
    void applyView(std::span<const float> splitScore, std::span<const std::uint8_t> splitBackface,
                   float pixelBudget, float coarsenBudget);

    // Self-check the front's internal consistency (throws std::logic_error on any violation), so a
    // corrupt `dependents_` count can't stay latent until a later coarsen trips on it:
    // `dependents_` matches a fresh reconstruction from the refined splits; every refined split's
    // dependencies are active; a non-root vertex is active exactly when its removing split is
    // refined. Cheap enough to run after every frame in tests.
    void validateInvariants() const;

    [[nodiscard]] bool active(std::uint32_t canonicalVertex) const
    {
        return active_[canonicalVertex] != 0;
    }
    [[nodiscard]] bool refined(std::uint32_t splitIndex) const
    {
        return refined_[splitIndex] != 0;
    }
    [[nodiscard]] const VertexForest& forest() const noexcept
    {
        return forest_;
    }
    [[nodiscard]] const DependencyDag& dag() const noexcept
    {
        return dag_;
    }

private:
    bool refineOne(std::uint32_t splitIndex);
    bool coarsenOne(std::uint32_t splitIndex);

    VertexForest forest_;
    DependencyDag dag_; // owns the packed-by-rank CSR layout
    // active_ / refined_ are SINGLE-writer within a rank (a vertex is activated only by its unique
    // removing split; a split writes only its own refined flag), so they need no atomic — uint8_t
    // is fine on the CPU. The eventual std430 GPU ABI should widen them to 32-bit entries (baseline
    // GLSL has no 8-bit storage without an extension), like `required_` / `dependents_`.
    std::vector<std::uint8_t> active_;      // per canonical vertex
    std::vector<std::uint8_t> refined_;     // per split
    std::vector<std::uint32_t> dependents_; // per canonical vertex: refined splits needing it
    // Per-split "required this frame" scratch, reused across frames. uint32_t (not a bit/byte) to
    // mirror the GLSL port, where the closure marks it with an atomic OR into baseline 32-bit
    // storage.
    std::vector<std::uint32_t> required_;
};

} // namespace fire_engine
