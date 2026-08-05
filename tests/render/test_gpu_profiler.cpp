#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include <fire_engine/render/gpu_profiler.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// The timestamp readback's arithmetic and availability policy, tested headlessly.
//
// This is where the bug lived that kept per-pass GPU timing dark on every device for months: the
// readback asked for per-query availability words and then threw them away, because
// `vkGetQueryPoolResults` returns VK_NOT_READY whenever ANY query in the range is unavailable — and
// some always are, since a pass that did not run leaves its two queries reset and never written.
// The Vulkan call itself needs a device; this half does not, and it is the half that was wrong.
// ---------------------------------------------------------------------------

namespace
{

constexpr std::size_t kStrideWords = 2;
constexpr std::size_t kWordCount = static_cast<std::size_t>(kProfileQueriesPerFrame) * kStrideWords;

// One slot's raw buffer, everything unavailable — the shape the driver writes for a frame in which
// nothing has completed.
std::vector<std::uint64_t> emptySlot()
{
    return std::vector<std::uint64_t>(kWordCount, 0);
}

// Fill one pass' begin/end pair as available, spanning [begin, end] ticks.
void writeSpan(std::vector<std::uint64_t>& words, ProfilePass pass, std::uint64_t begin,
               std::uint64_t end)
{
    const std::size_t beginWord = static_cast<std::size_t>(pass) * 2 * kStrideWords;
    words[beginWord] = begin;
    words[beginWord + 1] = 1;
    words[beginWord + kStrideWords] = end;
    words[beginWord + kStrideWords + 1] = 1;
}

[[nodiscard]] float passMs(const FrameStats& stats, ProfilePass pass)
{
    return stats.passMs[static_cast<std::size_t>(pass)];
}

} // namespace

TEST_CASE("GpuProfiler.AnEmptySlotIsWarmingUpNotValid", "[GpuProfiler]")
{
    // Nothing available yet. The distinction that matters: this is NOT `Unsupported` — the device
    // can time, it just has nothing to say — and the overlay renders the two differently now.
    FrameStats stats;
    resolveTimestampWords(emptySlot(), 1.0f, 64, stats);

    CHECK(stats.gpuTiming == GpuTimingState::WarmingUp);
    CHECK_FALSE(stats.gpuValid());
    CHECK(stats.gpuMeasuredPassSumMs == Catch::Approx(0.0f));
}

TEST_CASE("GpuProfiler.PassesThatDidNotRunAreSkippedRatherThanFailingTheRead", "[GpuProfiler]")
{
    // THE regression test for the parked bug. A real frame always mixes available and unavailable
    // queries — no transmissive draw, no spot light, no VDPM front — and that mixture is exactly
    // what makes the Vulkan call return VK_NOT_READY. The resolver must report the passes that DID
    // run.
    auto words = emptySlot();
    writeSpan(words, ProfilePass::ShadowCascades, 1'000, 3'000);
    writeSpan(words, ProfilePass::Forward, 10'000, 15'000);

    FrameStats stats;
    resolveTimestampWords(words, 1.0f, 64, stats); // 1 ns per tick, full-width queue

    CHECK(stats.gpuTiming == GpuTimingState::Valid);
    CHECK(stats.gpuValid());
    CHECK(passMs(stats, ProfilePass::ShadowCascades) == Catch::Approx(0.002f).margin(1e-6f));
    CHECK(passMs(stats, ProfilePass::Forward) == Catch::Approx(0.005f).margin(1e-6f));
    // Everything else stays 0 — reported as "did not run", not as missing data.
    CHECK(passMs(stats, ProfilePass::Transmission) == Catch::Approx(0.0f));
    CHECK(passMs(stats, ProfilePass::ShadowSpot) == Catch::Approx(0.0f));
    CHECK(stats.gpuMeasuredPassSumMs == Catch::Approx(0.007f).margin(1e-6f));
}

TEST_CASE("GpuProfiler.OneEndAvailableIsNotASpan", "[GpuProfiler]")
{
    // Half-written pairs are the shape of a pass whose begin landed in this slot and whose end did
    // not. Counting it would invent a span from a garbage value.
    auto words = emptySlot();
    const std::size_t beginWord = static_cast<std::size_t>(ProfilePass::Forward) * 2 * kStrideWords;
    words[beginWord] = 5'000;
    words[beginWord + 1] = 1;                // begin available
    words[beginWord + kStrideWords] = 999;   // end value present but...
    words[beginWord + kStrideWords + 1] = 0; // ...not available

    FrameStats stats;
    resolveTimestampWords(words, 1.0f, 64, stats);

    CHECK(stats.gpuTiming == GpuTimingState::WarmingUp);
    CHECK(passMs(stats, ProfilePass::Forward) == Catch::Approx(0.0f));
}

TEST_CASE("GpuProfiler.ADecreasingPairIsAWrapNotAnAnomaly", "[GpuProfiler]")
{
    // Vulkan defines timestamp overflow as wrapping to zero within `timestampValidBits`, so an end
    // value numerically below its begin is ORDINARY on a narrow queue — 8 bits at any real period
    // wraps constantly. Treating it as malformed (as this resolver first did) would report 0 ms for
    // every span that happened to straddle the wrap, i.e. silently drop real cost.
    auto words = emptySlot();
    writeSpan(words, ProfilePass::Forward, 250, 5); // wraps through 256

    FrameStats stats;
    resolveTimestampWords(words, 1.0f, 8, stats);

    CHECK(stats.gpuTiming == GpuTimingState::Valid);
    // 256 - 250 + 5 = 11 ticks.
    CHECK(passMs(stats, ProfilePass::Forward) == Catch::Approx(11.0e-6f).margin(1e-9f));
}

TEST_CASE("GpuProfiler.BitsAboveTheValidWidthCannotAffectTheResult", "[GpuProfiler]")
{
    // Vulkan leaves the bits above `timestampValidBits` UNDEFINED. A queue that returns junk up
    // there must produce the same span as one that returns zeros — otherwise the panel's numbers
    // depend on what a driver happens to leave in memory.
    auto clean = emptySlot();
    writeSpan(clean, ProfilePass::Forward, 250, 5);
    auto noisy = emptySlot();
    writeSpan(noisy, ProfilePass::Forward, 0xDEAD'BEEF'0000'0000ull | 250,
              0xFACE'0000'0000'0000ull | 5);

    FrameStats cleanStats;
    FrameStats noisyStats;
    resolveTimestampWords(clean, 1.0f, 8, cleanStats);
    resolveTimestampWords(noisy, 1.0f, 8, noisyStats);

    CHECK(passMs(noisyStats, ProfilePass::Forward) ==
          Catch::Approx(passMs(cleanStats, ProfilePass::Forward)).margin(1e-9f));
    CHECK(noisyStats.gpuMeasuredPassSumMs ==
          Catch::Approx(cleanStats.gpuMeasuredPassSumMs).margin(1e-9f));
}

TEST_CASE("GpuProfiler.AQueueWithNoValidBitsReportsNothing", "[GpuProfiler]")
{
    // Defensive: the profiler refuses to enable on such a queue, but a zero mask would otherwise
    // turn every delta into 0 and present a frame of free passes — worse than reporting nothing.
    auto words = emptySlot();
    writeSpan(words, ProfilePass::Forward, 0, 1'000'000);

    FrameStats stats;
    resolveTimestampWords(words, 1.0f, 0, stats);

    CHECK(stats.gpuTiming == GpuTimingState::WarmingUp);
    CHECK(stats.gpuMeasuredPassSumMs == Catch::Approx(0.0f));
}

TEST_CASE("GpuProfiler.TicksAreScaledByTheDevicesTimestampPeriod", "[GpuProfiler]")
{
    // The period is nanoseconds per tick and varies by device (1 on this Mac's MoltenVK, ~40 on
    // some desktop GPUs). Getting it wrong scales every number in the panel by a constant nobody
    // would notice — the totals would simply look plausible and wrong.
    auto words = emptySlot();
    writeSpan(words, ProfilePass::Forward, 0, 1'000'000); // 1e6 ticks

    FrameStats oneNs;
    resolveTimestampWords(words, 1.0f, 64, oneNs);
    CHECK(passMs(oneNs, ProfilePass::Forward) == Catch::Approx(1.0f).margin(1e-4f));

    FrameStats fortyNs;
    resolveTimestampWords(words, 40.0f, 64, fortyNs);
    CHECK(passMs(fortyNs, ProfilePass::Forward) == Catch::Approx(40.0f).margin(1e-3f));
}

TEST_CASE("GpuProfiler.VdpmBreakdownRowsReportButDoNotSumIntoTheMeasuredSum", "[GpuProfiler]")
{
    // The four VDPM stage rows are SUBRANGES of VdpmCompute. Adding them to the sum would inflate
    // it by the whole VDPM cost whenever the single-front breakdown is populated — and it is
    // populated exactly when someone is measuring VDPM, i.e. when the number matters most.
    auto words = emptySlot();
    writeSpan(words, ProfilePass::VdpmCompute, 0, 4'000'000); // 4 ms
    writeSpan(words, ProfilePass::VdpmScore, 0, 1'000'000);
    writeSpan(words, ProfilePass::VdpmApply, 1'000'000, 2'000'000);
    writeSpan(words, ProfilePass::VdpmRepair, 2'000'000, 3'000'000);
    writeSpan(words, ProfilePass::VdpmEmit, 3'000'000, 4'000'000);
    writeSpan(words, ProfilePass::Forward, 0, 2'000'000); // 2 ms, disjoint

    FrameStats stats;
    resolveTimestampWords(words, 1.0f, 64, stats);

    CHECK(passMs(stats, ProfilePass::VdpmScore) == Catch::Approx(1.0f).margin(1e-4f));
    CHECK(passMs(stats, ProfilePass::VdpmEmit) == Catch::Approx(1.0f).margin(1e-4f));
    // 4 (compute) + 2 (forward), NOT 4 + 4 (its stages) + 2.
    CHECK(stats.gpuMeasuredPassSumMs == Catch::Approx(6.0f).margin(1e-3f));
}

TEST_CASE("GpuProfiler.AShortBufferReportsNothingRatherThanAPartialSum", "[GpuProfiler]")
{
    // A caller that passes the wrong range gets no answer, not one assembled from whatever fit.
    auto words = emptySlot();
    writeSpan(words, ProfilePass::Forward, 0, 1'000'000);
    words.resize(words.size() / 2);

    FrameStats stats;
    resolveTimestampWords(words, 1.0f, 64, stats);

    CHECK(stats.gpuTiming == GpuTimingState::WarmingUp);
    CHECK(stats.gpuMeasuredPassSumMs == Catch::Approx(0.0f));
}
