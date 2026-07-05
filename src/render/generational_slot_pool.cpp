#include <fire_engine/render/generational_slot_pool.hpp>

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
    generations_.push_back(0);
    return {static_cast<uint32_t>(generations_.size() - 1), 0};
}

void GenerationalSlotPool::release(uint32_t index)
{
    generations_[index] = (generations_[index] + 1) & kHandleGenerationMask;
    free_.push_back(index);
}

bool GenerationalSlotPool::valid(uint32_t index, uint32_t generation) const noexcept
{
    return index < generations_.size() && generations_[index] == generation;
}

} // namespace fire_engine
