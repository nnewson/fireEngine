#include <fire_engine/core/node_id.hpp>

#include <atomic>
#include <cstdlib>

#include <fire_engine/core/log.hpp>

namespace fire_engine
{

NodeId allocateNodeId() noexcept
{
    // Starts at 1 so NodeId::Invalid (0) is never handed out.
    static std::atomic<std::uint64_t> next{1};
    const std::uint64_t id = next.fetch_add(1, std::memory_order_relaxed);
    if (id == 0)
    {
        // The counter wrapped. Continuing would hand out Invalid and then REUSE live identities,
        // silently corrupting every history keyed on them. At ~2^64 allocations this is
        // unreachable in practice; if it ever happens, stopping is the only honest response.
        log::error(log::category::general,
                   "node id counter wrapped — identities are no longer unique");
        std::abort();
    }
    return static_cast<NodeId>(id);
}

} // namespace fire_engine
