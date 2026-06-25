#pragma once

#include <cstddef>
#include <span>

namespace fire_engine
{

template <typename Entry>
[[nodiscard]] std::size_t findAnimationEntryIndex(std::span<const Entry> entries,
                                                  std::size_t id) noexcept
{
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        if (entries[i].id == id)
        {
            return i;
        }
    }
    return entries.size();
}

template <typename Entry>
bool selectAnimationEntry(std::span<const Entry> entries, std::size_t id, std::size_t& activeIndex,
                          std::size_t& activeId, bool& initialisedFlag) noexcept
{
    auto index = findAnimationEntryIndex(entries, id);
    if (index >= entries.size())
    {
        return false;
    }

    activeIndex = index;
    activeId = id;
    initialisedFlag = false;
    return true;
}

} // namespace fire_engine
