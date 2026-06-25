#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>

namespace fire_engine
{

template <typename Entry>
[[nodiscard]] std::optional<std::size_t>
findAnimationEntryIndex(std::span<const Entry> entries, std::size_t id) noexcept
{
    const auto found = std::ranges::find(entries, id, &Entry::id);
    if (found == entries.end())
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(found - entries.begin());
}

template <typename Entry>
bool selectAnimationEntry(std::span<const Entry> entries, std::size_t id, std::size_t& activeIndex,
                          std::size_t& activeId, bool& initialisedFlag) noexcept
{
    const auto index = findAnimationEntryIndex(entries, id);
    if (!index)
    {
        return false;
    }

    activeIndex = *index;
    activeId = id;
    initialisedFlag = false;
    return true;
}

} // namespace fire_engine
