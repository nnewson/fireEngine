#include <fire_engine/graphics/vdpm_gpu_registry.hpp>

#include <cstdint>
#include <stdexcept>

namespace fire_engine
{

bool VdpmWorkRequest::sameParams(const VdpmWorkRequest& other) const noexcept
{
    // Field-by-field so trailing/embedded padding never affects the result. The front handle is
    // the identity; the rest is what the compute reads, so a change in any of them is different
    // work.
    return front == other.front && world == other.world && uvScale == other.uvScale &&
           normalScale == other.normalScale && tangentScale == other.tangentScale &&
           rasterBackfaceCulling == other.rasterBackfaceCulling;
}

void selectVisibleVdpmRequests(std::span<const VdpmWorkRequest> requests,
                               std::span<const VdpmFrontHandle> visibleFronts,
                               std::vector<VdpmWorkRequest>& out, VdpmRequestSelectScratch& scratch)
{
    // Clear (retain capacity) — the whole selection is allocation-free once warm.
    out.clear();
    scratch.visibleFronts.clear();
    scratch.keptIndex.clear();

    for (const VdpmFrontHandle f : visibleFronts)
    {
        scratch.visibleFronts.insert(static_cast<std::uint32_t>(f));
    }

    for (const VdpmWorkRequest& req : requests)
    {
        const auto packed = static_cast<std::uint32_t>(req.front);
        if (!scratch.visibleFronts.contains(packed))
        {
            continue; // shadow-only / not camera-visible this frame
        }
        // keptIndex maps a kept front's packed handle to its slot in `out`, so a second request for
        // that front is compared against the one already accepted rather than blindly dropped
        // (first-wins) or duplicated.
        const auto it = scratch.keptIndex.find(packed);
        if (it == scratch.keptIndex.end())
        {
            scratch.keptIndex.emplace(packed, out.size());
            out.push_back(req);
            continue;
        }
        // A front is one persistent GPU state; scoring it twice with conflicting inputs in one
        // frame is a caller bug, not something to silently resolve first-wins.
        if (!out[it->second].sameParams(req))
        {
            throw std::logic_error(
                "selectVisibleVdpmRequests: conflicting work requests for one VDPM front in a "
                "single frame");
        }
        // Identical duplicate — already have it.
    }
}

} // namespace fire_engine
