#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/vdpm_scan.hpp>

using namespace fire_engine;

// Standalone GPU exclusive-scan stress harness ([.][gpu], local-only). Validates the recursive
// hierarchical scan (vdpm_scan_block/add) against a CPU exclusive prefix sum across the block-size
// boundaries — the primitive the B2 emit compaction depends on. Run: ./test_fire_engine "[gpu]".

namespace
{

constexpr std::uint32_t kB = kScanElementsPerBlock; // 256

struct ScanResult
{
    std::vector<std::uint32_t> output;
    std::uint32_t total{0};
};

// CPU authority: exclusive prefix sum + total.
std::vector<std::uint32_t> cpuExclusiveScan(std::span<const std::uint32_t> flags,
                                            std::uint32_t& total)
{
    std::vector<std::uint32_t> out(flags.size());
    std::uint32_t acc = 0;
    for (std::size_t i = 0; i < flags.size(); ++i)
    {
        out[i] = acc;
        acc += flags[i];
    }
    total = acc;
    return out;
}

// Persistent device + scan + buffers, sized to a max count. run() rewrites the mapped input (so
// scratch/output/total are REUSED across runs, never cleared — the dirty-scratch case), records the
// fill→scan→copy sequence with explicit barriers, submits, fence-waits, and reads back.
class ScanRunner
{
public:
    explicit ScanRunner(std::uint32_t maxCount)
        : device_(Device::headlessCompute()),
          resources_(device_),
          scan_(device_),
          pool_(
              device_.device(),
              vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        .queueFamilyIndex = device_.graphicsFamily()}),
          maxCount_(maxCount)
    {
        const Resources::MappedBufferSet in =
            resources_.createMappedDeviceAddressBuffers(bytes(maxCount_));
        inputBuf_ = in.buffers[0];
        inputMapped_ = in.mapped[0];
        inputAddr_ = resources_.bufferAddress(inputBuf_);

        outputBuf_ = resources_.createDeviceLocalStorageBuffer(
            bytes(maxCount_), nullptr, vk::BufferUsageFlagBits::eTransferSrc);
        outputAddr_ = resources_.bufferAddress(outputBuf_);
        totalBuf_ = resources_.createDeviceLocalStorageBuffer(
            sizeof(std::uint32_t), nullptr, vk::BufferUsageFlagBits::eTransferSrc);
        totalAddr_ = resources_.bufferAddress(totalBuf_);

        for (const std::uint32_t n : VdpmScan::scratchElementCounts(maxCount_))
        {
            const BufferHandle b = resources_.createDeviceLocalStorageBuffer(bytes(n), nullptr);
            scratch_.push_back(b);
            scratchAddr_.push_back(resources_.bufferAddress(b));
        }
        outReadback_ = resources_.createMappedReadbackBuffers(bytes(maxCount_));
        totalReadback_ = resources_.createMappedReadbackBuffers(sizeof(std::uint32_t));
    }
    ScanRunner(const ScanRunner&) = delete;
    ScanRunner& operator=(const ScanRunner&) = delete;
    ScanRunner(ScanRunner&&) = delete;
    ScanRunner& operator=(ScanRunner&&) = delete;
    ~ScanRunner() = default;

    ScanResult run(std::span<const std::uint32_t> flags)
    {
        const auto count = static_cast<std::uint32_t>(flags.size());
        if (count > 0)
        {
            std::memcpy(inputMapped_.data(), flags.data(), bytes(count));
        }
        // This count's internal scratch levels (a prefix of the max-count scratch — always big
        // enough since each level's element count is monotone in the total).
        const std::vector<std::uint32_t> levelCounts = VdpmScan::scratchElementCounts(count);
        const std::span<const std::uint64_t> levelAddrs{scratchAddr_.data(), levelCounts.size()};

        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        // Pre-zero the total (the empty-input contract; also defines the value when count == 0).
        // fillBuffer is a CLEAR command (eClear stage, eTransferWrite access), NOT a copy, so the
        // barrier before the scan's compute write must name eClear as the producer.
        cmd.fillBuffer(resources_.vulkanBuffer(totalBuf_), 0, sizeof(std::uint32_t), 0u);
        barrier(*cmd, vk::PipelineStageFlagBits2::eClear, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

        scan_.recordScan(*cmd, inputAddr_, outputAddr_, levelAddrs, totalAddr_, count);

        // Readback barrier: name BOTH possible producers of the buffers being copied — the scan's
        // compute write (count > 0) AND the fill/clear. For count == 0 no compute runs, so the fill
        // of the total is what the copy must wait on.
        barrier(*cmd,
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eClear,
                vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        if (count > 0)
        {
            cmd.copyBuffer(resources_.vulkanBuffer(outputBuf_),
                           resources_.vulkanBuffer(outReadback_.buffers[0]),
                           vk::BufferCopy{.size = bytes(count)});
        }
        cmd.copyBuffer(resources_.vulkanBuffer(totalBuf_),
                       resources_.vulkanBuffer(totalReadback_.buffers[0]),
                       vk::BufferCopy{.size = sizeof(std::uint32_t)});
        barrier(*cmd, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
        cmd.end();

        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        const vk::SubmitInfo2 submit{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo};
        const vk::raii::Fence fence(device_.device(), vk::FenceCreateInfo{});
        device_.graphicsQueue().submit2(submit, *fence);
        (void)device_.device().waitForFences(*fence, vk::True,
                                             std::numeric_limits<std::uint64_t>::max());

        ScanResult r;
        r.output.resize(count);
        if (count > 0)
        {
            std::memcpy(r.output.data(), outReadback_.mapped[0].data(), bytes(count));
        }
        std::memcpy(&r.total, totalReadback_.mapped[0].data(), sizeof(std::uint32_t));
        return r;
    }

private:
    static vk::DeviceSize bytes(std::uint32_t n)
    {
        return static_cast<vk::DeviceSize>(n) * sizeof(std::uint32_t);
    }
    static void barrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage,
                        vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                        vk::AccessFlags2 dstAccess)
    {
        const vk::MemoryBarrier2 mb{.srcStageMask = srcStage,
                                    .srcAccessMask = srcAccess,
                                    .dstStageMask = dstStage,
                                    .dstAccessMask = dstAccess};
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
    }

    Device device_;
    Resources resources_;
    VdpmScan scan_;
    vk::raii::CommandPool pool_;
    std::uint32_t maxCount_;
    BufferHandle inputBuf_{NullBuffer};
    std::span<std::byte> inputMapped_{};
    std::uint64_t inputAddr_{0};
    BufferHandle outputBuf_{NullBuffer};
    std::uint64_t outputAddr_{0};
    BufferHandle totalBuf_{NullBuffer};
    std::uint64_t totalAddr_{0};
    std::vector<BufferHandle> scratch_;
    std::vector<std::uint64_t> scratchAddr_;
    Resources::MappedBufferSet outReadback_;
    Resources::MappedBufferSet totalReadback_;
};

void expectScanMatches(ScanRunner& runner, std::span<const std::uint32_t> flags)
{
    std::uint32_t cpuTotal = 0;
    const std::vector<std::uint32_t> cpu = cpuExclusiveScan(flags, cpuTotal);
    const ScanResult gpu = runner.run(flags);
    CHECK(gpu.total == cpuTotal); // the total, tested separately from the slots
    REQUIRE(gpu.output.size() == cpu.size());
    for (std::size_t i = 0; i < cpu.size(); ++i)
    {
        CAPTURE(i);
        CHECK(gpu.output[i] == cpu[i]); // every exclusive slot
    }
}

} // namespace

TEST_CASE("VdpmScan::hierarchy is 64-bit safe at UINT32_MAX", "[vdpm]")
{
    // CPU-only (static, no device) — normal CI. groupCount's `n + block - 1` must not wrap near
    // UINT32_MAX. Expected levels: UINT32_MAX -> 16777216 -> 65536 -> 256.
    const std::vector<std::uint32_t> h =
        VdpmScan::hierarchy(std::numeric_limits<std::uint32_t>::max());
    REQUIRE(h.size() == 4);
    CHECK(h[0] == std::numeric_limits<std::uint32_t>::max());
    CHECK(h[1] == 16777216u);
    CHECK(h[2] == 65536u);
    CHECK(h[3] == 256u);
}

TEST_CASE("VDPM GPU scan matches a CPU exclusive prefix sum across block boundaries", "[.][gpu]")
{
    // Boundaries 0, 1, B-1, B, B+1, B^2-1, B^2, B^2+1 with all-zero / all-one / adversarial-mixed
    // flags — B^2+1 forces three hierarchy levels.
    const std::uint32_t maxCount = (kB * kB) + 1;
    ScanRunner runner(maxCount);

    const std::vector<std::uint32_t> counts{
        0, 1, kB - 1, kB, kB + 1, (kB * kB) - 1, kB * kB, (kB * kB) + 1};
    for (const std::uint32_t count : counts)
    {
        CAPTURE(count);
        // all-zero
        expectScanMatches(runner, std::vector<std::uint32_t>(count, 0u));
        // all-one
        expectScanMatches(runner, std::vector<std::uint32_t>(count, 1u));
        // adversarial mixed: 1s clustered at every block boundary (b-1, b, b+1) plus a
        // pseudo-random sprinkle, so a wrong block-sum or missing add surfaces exactly at the
        // seams.
        std::vector<std::uint32_t> mixed(count, 0u);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const std::uint32_t m = i % kB;
            const bool boundary = (m == 0) || (m == kB - 1);
            mixed[i] = (boundary || ((i * 2654435761u) >> 28) == 0u) ? 1u : 0u;
        }
        expectScanMatches(runner, mixed);
    }
}

TEST_CASE("VDPM GPU scan reuses dirty scratch (all-ones then all-zero, no clear)", "[.][gpu]")
{
    // The scan must overwrite EVERY valid output + block sum each run, so a second scan into the
    // same (uncleared) scratch/output can't inherit stale block sums. Run a full all-ones scan,
    // then an all-zero scan on the SAME runner without clearing anything.
    const std::uint32_t count = (kB * kB) + 1;
    ScanRunner runner(count);

    expectScanMatches(runner, std::vector<std::uint32_t>(count, 1u)); // dirties the scratch

    const ScanResult zero = runner.run(std::vector<std::uint32_t>(count, 0u));
    CHECK(zero.total == 0u);
    for (const std::uint32_t v : zero.output)
    {
        CHECK(v == 0u); // no stale block sums leaked in
    }
}
