#include <fire_engine/render/gpu_profiler.hpp>

#include <fire_engine/core/log.hpp>
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
    timestampValidBits_ = queueFamilies[device.graphicsFamily()].timestampValidBits;
    const uint32_t validBits = timestampValidBits_;

    if (timestampPeriodNs_ == 0.0f || validBits == 0)
    {
        // Genuinely unsupported — the overlay falls back to CPU frame timing only. Logged at WARN
        // with both numbers, because this is the ONLY reason per-pass GPU timings should ever be
        // missing, and for months it was blamed for a readback bug instead (see resolve()).
        log::warn(
            log::category::render,
            "GPU timestamps unsupported on this device (timestampPeriod={} ns, graphics-queue "
            "timestampValidBits={}) — per-pass GPU timings disabled",
            timestampPeriodNs_, validBits);
        return;
    }

    vk::QueryPoolCreateInfo ci{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = kQueriesPerFrame * kMaxFramesInFlight,
    };
    pool_ = vk::raii::QueryPool(*device_, ci);
    enabled_ = true;
    log::debug(log::category::render,
               "GPU timestamps enabled: timestampPeriod={} ns, validBits={}, {} queries per frame",
               timestampPeriodNs_, validBits, kQueriesPerFrame);
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
    // BOTTOM-of-pipe, like end() — see the header for why the two boundaries share one convention.
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *pool_,
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
    out.gpuTiming = GpuTimingState::Unsupported;
    out.gpuMeasuredPassSumMs = 0.0f;
    out.passMs.fill(0.0f);
    if (!enabled_)
    {
        return;
    }
    out.gpuTiming = GpuTimingState::WarmingUp;
    if (!slotUsed_[frameIndex])
    {
        return; // never reset yet (first cycle) — reading would be invalid
    }

    // Two uint64s per query (timestamp value + availability), so an unavailable query is
    // identifiable rather than fatal.
    constexpr uint32_t kStrideWords = 2;
    auto [result, data] = pool_.getResults<uint64_t>(
        frameIndex * kQueriesPerFrame, kQueriesPerFrame,
        static_cast<std::size_t>(kQueriesPerFrame) * kStrideWords * sizeof(uint64_t),
        kStrideWords * sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);

    // eNotReady IS THE NORMAL RESULT HERE, and treating it as failure is the bug that kept this
    // whole feature dark. Without VK_QUERY_RESULT_WAIT_BIT, vkGetQueryPoolResults returns
    // VK_NOT_READY when ANY query in the range is unavailable — and some always are: a pass that
    // did not run this frame (no transmissive draw, no spot light, no VDPM front) leaves its two
    // queries reset and never written, so they stay unavailable for that slot. With
    // VK_QUERY_RESULT_WITH_AVAILABILITY_BIT the per-query availability words are still written,
    // which is the entire reason we ask for them — so a NotReady buffer is a PARTIAL result to
    // filter, not a failed read. Bailing on it reported "unavailable" on every device and every
    // driver, a symptom that looked like a MoltenVK limitation and was parked as one.
    //
    // Anything else IS a real error (a lost device, a bad range): leave the slot WarmingUp and say
    // so rather than presenting whatever the buffer happens to hold.
    if (result != vk::Result::eSuccess && result != vk::Result::eNotReady)
    {
        log::warn(log::category::render, "GPU timestamp readback failed for slot {}: {}",
                  frameIndex, vk::to_string(result));
        return;
    }

    resolveTimestampWords(data, timestampPeriodNs_, timestampValidBits_, out);
}

void resolveTimestampWords(std::span<const std::uint64_t> words, float timestampPeriodNs,
                           std::uint32_t timestampValidBits, FrameStats& out) noexcept
{
    constexpr std::size_t kStrideWords = 2;
    out.gpuMeasuredPassSumMs = 0.0f;
    out.passMs.fill(0.0f);
    // No meaningful bits means no meaningful spans. The profiler refuses to enable in that case, so
    // this is defensive — but a zero mask would otherwise turn every delta into 0 and report a
    // frame of free passes, which is worse than reporting nothing.
    if (timestampValidBits == 0)
    {
        out.gpuTiming = GpuTimingState::WarmingUp;
        return;
    }
    // A short buffer is a programming error at the call site, not something to half-read: report
    // nothing rather than a total assembled from whatever fit.
    if (words.size() < static_cast<std::size_t>(kProfileQueriesPerFrame) * kStrideWords)
    {
        out.gpuTiming = GpuTimingState::WarmingUp;
        return;
    }

    const std::uint64_t mask = timestampMask(timestampValidBits);
    bool anyValid = false;
    for (std::uint32_t p = 0; p < kProfilePassCount; ++p)
    {
        const std::size_t beginWord = static_cast<std::size_t>(p) * 2 * kStrideWords;
        const std::size_t endWord = beginWord + kStrideWords;
        const std::uint64_t beginVal = words[beginWord];
        const std::uint64_t beginAvail = words[beginWord + 1];
        const std::uint64_t endVal = words[endWord];
        const std::uint64_t endAvail = words[endWord + 1];
        if (beginAvail == 0 || endAvail == 0)
        {
            continue; // pass did not run in this slot — report 0
        }
        // MODULAR in the queue's timestamp width. Vulkan guarantees only the low
        // `timestampValidBits` carry data and defines overflow as wrapping to zero inside that
        // width, so `end < begin` on the raw words is a legitimate wrap, not an anomaly — and on a
        // queue narrower than 64 bits the upper bits are undefined noise that must not reach the
        // subtraction at all. Masking both ends and masking the difference handles both facts in
        // one step: unsigned subtraction already wraps modulo 2^64, and the mask reduces it to
        // 2^timestampValidBits.
        const std::uint64_t ticks = ((endVal & mask) - (beginVal & mask)) & mask;
        const float ms = static_cast<float>(ticks) * timestampPeriodNs / 1.0e6f;
        out.passMs[p] = ms;
        // Only disjoint spans are summed: the VDPM breakdown rows sit INSIDE VdpmCompute, so adding
        // them too would inflate the reported sum by the whole VDPM cost whenever the single-front
        // breakdown is populated.
        if (profilePassContributesToMeasuredSum(static_cast<ProfilePass>(p)))
        {
            out.gpuMeasuredPassSumMs += ms;
        }
        anyValid = true;
    }
    out.gpuTiming = anyValid ? GpuTimingState::Valid : GpuTimingState::WarmingUp;
}

} // namespace fire_engine
