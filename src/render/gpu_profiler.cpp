#include <fire_engine/render/gpu_profiler.hpp>

#include <fire_engine/render/device.hpp>

namespace fire_engine
{

GpuProfiler::GpuProfiler(const Device& device)
    : device_{&device.device()}
{
    const auto props = device.physicalDevice().getProperties();
    timestampPeriodNs_ = props.limits.timestampPeriod;

    // The graphics queue must support timestamps for our writeTimestamp2 calls.
    const auto queueFamilies = device.physicalDevice().getQueueFamilyProperties();
    const uint32_t validBits = queueFamilies[device.graphicsFamily()].timestampValidBits;

    if (timestampPeriodNs_ == 0.0f || validBits == 0)
    {
        // Timestamps unsupported (some MoltenVK setups) — stay disabled; the
        // overlay falls back to CPU frame timing only.
        return;
    }

    vk::QueryPoolCreateInfo ci{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = kQueriesPerFrame * kMaxFramesInFlight,
    };
    pool_ = vk::raii::QueryPool(*device_, ci);
    enabled_ = true;
}

void GpuProfiler::beginFrame(vk::CommandBuffer cmd, uint32_t frameIndex)
{
    if (!enabled_)
    {
        return;
    }
    cmd.resetQueryPool(*pool_, frameIndex * kQueriesPerFrame, kQueriesPerFrame);
    slotUsed_[frameIndex] = true;
}

void GpuProfiler::begin(vk::CommandBuffer cmd, uint32_t frameIndex, ProfilePass pass) const
{
    if (!enabled_)
    {
        return;
    }
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, *pool_,
                        queryIndex(frameIndex, pass, false));
}

void GpuProfiler::end(vk::CommandBuffer cmd, uint32_t frameIndex, ProfilePass pass) const
{
    if (!enabled_)
    {
        return;
    }
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *pool_,
                        queryIndex(frameIndex, pass, true));
}

void GpuProfiler::resolve(uint32_t frameIndex, FrameStats& out) const
{
    out.gpuValid = false;
    out.gpuTotalMs = 0.0f;
    out.passMs.fill(0.0f);
    if (!enabled_ || !slotUsed_[frameIndex])
    {
        return; // never reset yet (first cycle) — reading would be invalid
    }

    // Two uint64s per query (timestamp value + availability). Availability lets us
    // skip passes that didn't run this cycle (e.g. transmission with no draws)
    // without the whole read failing.
    constexpr uint32_t kStrideWords = 2;
    auto [result, data] = pool_.getResults<uint64_t>(
        frameIndex * kQueriesPerFrame, kQueriesPerFrame,
        kQueriesPerFrame * kStrideWords * sizeof(uint64_t), kStrideWords * sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
    if (result != vk::Result::eSuccess)
    {
        return; // results not ready yet (first cycle after create) — leave CPU-only
    }

    bool anyValid = false;
    for (uint32_t p = 0; p < kProfilePassCount; ++p)
    {
        const uint64_t beginVal = data[(p * 2 + 0) * kStrideWords];
        const uint64_t beginAvail = data[(p * 2 + 0) * kStrideWords + 1];
        const uint64_t endVal = data[(p * 2 + 1) * kStrideWords];
        const uint64_t endAvail = data[(p * 2 + 1) * kStrideWords + 1];
        if (beginAvail == 0 || endAvail == 0 || endVal < beginVal)
        {
            continue; // pass skipped this cycle, or wrapped — report 0
        }
        const float ms = static_cast<float>(endVal - beginVal) * timestampPeriodNs_ / 1.0e6f;
        out.passMs[p] = ms;
        out.gpuTotalMs += ms;
        anyValid = true;
    }
    out.gpuValid = anyValid;
}

} // namespace fire_engine
