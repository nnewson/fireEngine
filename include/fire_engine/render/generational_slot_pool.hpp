#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <fire_engine/graphics/gpu_handle.hpp>

namespace fire_engine
{

// Hands out resource-table slot indices with a per-slot generation counter (the basis for
// CR-12/17 handle validation). acquire() recycles a released slot — reusing its index and
// returning its current generation — or grows the table; release() bumps the slot's generation
// (so any outstanding handle to it becomes stale) and returns the index to the free list. This
// keeps a churning table (e.g. render targets released and recreated every window resize)
// bounded instead of growing without limit. Pure index bookkeeping — no GPU state — so it is
// unit-tested directly; Resources holds the matching payload table in lockstep by index.
class GenerationalSlotPool
{
public:
    struct Slot
    {
        uint32_t index;
        uint32_t generation;
    };

    // Recycle a freed slot if one is available, else grow by one. The returned generation is
    // the slot's current generation, to be baked into the handle via makeHandle().
    [[nodiscard]] Slot acquire();

    // Retire slot `index`: bump its generation and return it to the free list.
    void release(uint32_t index);

    // True if `generation` still matches slot `index`'s live generation (i.e. the handle has
    // not been invalidated by a release + reuse).
    [[nodiscard]] bool valid(uint32_t index, uint32_t generation) const noexcept;

    // High-water slot count — the size the payload table must track. Stays bounded under
    // release/acquire churn because freed slots are recycled.
    [[nodiscard]] std::size_t slotCount() const noexcept
    {
        return generations_.size();
    }

private:
    std::vector<uint32_t> generations_;
    std::vector<uint32_t> free_;
};

} // namespace fire_engine
