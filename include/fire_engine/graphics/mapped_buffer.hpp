#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <span>
#include <type_traits>

#include <fire_engine/core/log.hpp>

namespace fire_engine
{

// Persistently-mapped host-visible GPU memory is passed around as std::span<std::byte> (not a raw
// void*) so the destination size travels with the pointer and every write is bounds-checkable
// (CR-21). These helpers are the single funnel for writing into such a span.

// Copy `bytes` from `src` into the mapped span. `src` must not overlap `dst` (it's device-visible
// memory being filled from a host struct). The bounds guard is release-visible, not just an
// `assert`: a debug build trips loudly at the source, and a release build CLAMPS (so a bad size can
// never corrupt neighbouring GPU allocations) and logs once instead of overflowing silently.
inline void writeMapped(std::span<std::byte> dst, const void* src, std::size_t bytes) noexcept
{
    assert(bytes <= dst.size() && "mapped write overflows the destination buffer");
    if (bytes > dst.size()) [[unlikely]]
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            log::error(log::category::render,
                       "writeMapped: {}-byte write clamped to {}-byte destination", bytes,
                       dst.size());
        }
        bytes = dst.size();
    }
    std::memcpy(dst.data(), src, bytes);
}

// Typed convenience: copy a trivially-copyable value (a UBO / SSBO record) into the span.
template <typename T>
inline void writeMapped(std::span<std::byte> dst, const T& value) noexcept
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "writeMapped copies raw bytes; T must be trivially copyable");
    writeMapped(dst, &value, sizeof(T));
}

} // namespace fire_engine
