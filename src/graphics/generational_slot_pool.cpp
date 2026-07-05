#include <fire_engine/graphics/generational_slot_pool.hpp>

namespace fire_engine
{

GenerationalSlotPool::Slot GenerationalSlotPool::acquire()
{
    if (!free_.empty())
    {
        const uint32_t index = free_.back();
        free_.pop_back();
        return {index, generations_[index]};
    }
    generations_.push_back(kFirstGeneration);
    return {static_cast<uint32_t>(generations_.size() - 1), kFirstGeneration};
}

void GenerationalSlotPool::release(uint32_t index)
{
    // Generations are 1-based: 0 is reserved as "never issued". A slot that has cycled through
    // all 255 live generations wraps back to 1, never to 0, so a packed handle can't collide with
    // a value-0 null handle (matters for PhysicsHandle, whose null is 0).
    uint32_t next = (generations_[index] + 1) & kHandleGenerationMask;
    if (next == 0)
    {
        next = kFirstGeneration;
    }
    generations_[index] = next;
    free_.push_back(index);
}

bool GenerationalSlotPool::valid(uint32_t index, uint32_t generation) const noexcept
{
    return index < generations_.size() && generations_[index] == generation;
}

} // namespace fire_engine
