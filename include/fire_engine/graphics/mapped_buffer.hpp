#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <span>
#include <type_traits>

namespace fire_engine
{

// Persistently-mapped host-visible GPU memory is passed around as std::span<std::byte> (not a raw
// void*) so the destination size travels with the pointer and every write is bounds-checkable
// (CR-21). These helpers are the single funnel for writing into such a span.

// Copy `bytes` from `src` into the mapped span, asserting the write fits. `src` must not overlap
// `dst` (it's device-visible memory being filled from a host struct).
inline void writeMapped(std::span<std::byte> dst, const void* src, std::size_t bytes) noexcept
{
    assert(bytes <= dst.size() && "mapped write overflows the destination buffer");
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
