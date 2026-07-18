#include <fire_engine/render/vdpm_scan.hpp>

#include <cstdint>
#include <stdexcept>

#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]] ComputePipelineConfig scanBlockConfig()
{
    ComputePipelineConfig config;
    config.compShaderPath = "vdpm_scan_block.comp.spv";
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eCompute, 0,
                                           static_cast<std::uint32_t>(sizeof(VdpmScanBlockPush)));
    return config;
}

[[nodiscard]] ComputePipelineConfig scanAddConfig()
{
    ComputePipelineConfig config;
    config.compShaderPath = "vdpm_scan_add.comp.spv";
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eCompute, 0,
                                           static_cast<std::uint32_t>(sizeof(VdpmScanAddPush)));
    return config;
}

[[nodiscard]] std::uint32_t groupCount(std::uint32_t n)
{
    // 64-bit intermediate: n + block - 1 wraps as uint32 near UINT32_MAX and would yield a zero or
    // wrong dispatch count.
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(n) + kScanElementsPerBlock - 1) /
                                      kScanElementsPerBlock);
}

// Global compute-write → compute-read/write barrier between scan levels. A global memory barrier
// (not per-buffer) because a level touches several buffers reached only by device address here.
void computeBarrier(vk::CommandBuffer cmd)
{
    const vk::MemoryBarrier2 mb{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
}

} // namespace

bool VdpmScan::deviceSupported(const Device& device)
{
    const vk::PhysicalDeviceLimits& limits = device.physicalDevice().getProperties().limits;
    return limits.maxComputeWorkGroupInvocations >= kScanElementsPerBlock &&
           limits.maxComputeWorkGroupSize[0] >= kScanElementsPerBlock &&
           limits.maxComputeSharedMemorySize >= kScanElementsPerBlock * sizeof(std::uint32_t);
}

VdpmScan::Checked VdpmScan::requireSupported(const Device& device)
{
    if (!deviceSupported(device))
    {
        throw std::runtime_error("VDPM GPU scan: device does not meet the compute limits (256)");
    }
    return {};
}

VdpmScan::VdpmScan(const Device& device)
    : VdpmScan(device, requireSupported(device))
{
}

VdpmScan::VdpmScan(const Device& device, Checked)
    : blockPipeline_(device, scanBlockConfig()),
      addPipeline_(device, scanAddConfig()),
      maxWorkGroupCountX_(
          device.physicalDevice().getProperties().limits.maxComputeWorkGroupCount[0])
{
}

std::vector<std::uint32_t> VdpmScan::hierarchy(std::uint32_t count)
{
    std::vector<std::uint32_t> levels;
    levels.push_back(count);
    while (levels.back() > kScanElementsPerBlock)
    {
        levels.push_back(groupCount(levels.back()));
    }
    return levels;
}

std::vector<std::uint32_t> VdpmScan::scratchElementCounts(std::uint32_t count)
{
    const std::vector<std::uint32_t> h = hierarchy(count);
    return {h.begin() + 1, h.end()}; // levels 1..K (empty when count <= block)
}

void VdpmScan::recordScan(vk::CommandBuffer cmd, std::uint64_t inputAddress,
                          std::uint64_t outputAddress,
                          std::span<const std::uint64_t> levelAddresses, std::uint64_t totalAddress,
                          std::uint32_t count) const
{
    if (count == 0)
    {
        return; // empty input: total is 0 (the caller pre-zeros it); no dispatch
    }
    const std::vector<std::uint32_t> h = hierarchy(count);
    const std::uint32_t topLevel = static_cast<std::uint32_t>(h.size()) - 1; // K
    if (levelAddresses.size() != topLevel)
    {
        throw std::runtime_error("VDPM scan: wrong scratch-level address count");
    }
    for (const std::uint32_t n : h)
    {
        if (groupCount(n) > maxWorkGroupCountX_)
        {
            throw std::runtime_error(
                "VDPM scan: level group count exceeds maxComputeWorkGroupCount");
        }
    }

    auto scanBlock = [&](std::uint64_t in, std::uint64_t out, std::uint64_t bs, std::uint32_t n)
    {
        const VdpmScanBlockPush push{
            .inputAddress = in, .outputAddress = out, .blockSumsAddress = bs, .count = n};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, blockPipeline_.pipeline());
        cmd.pushConstants<VdpmScanBlockPush>(blockPipeline_.pipelineLayout(),
                                             vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(groupCount(n), 1, 1);
    };
    auto scanAdd = [&](std::uint64_t out, std::uint64_t offsets, std::uint32_t n)
    {
        const VdpmScanAddPush push{.outputAddress = out, .offsetsAddress = offsets, .count = n};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, addPipeline_.pipeline());
        cmd.pushConstants<VdpmScanAddPush>(addPipeline_.pipelineLayout(),
                                           vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(groupCount(n), 1, 1);
    };

    // Single block: one scanBlock IS the global exclusive scan; its block total is the grand total.
    if (topLevel == 0)
    {
        scanBlock(inputAddress, outputAddress, totalAddress, h[0]);
        return;
    }

    // Upward: level 0 (external in/out) then each internal level scanned in place, block sums going
    // one level up; the top level's block sums are the grand total.
    scanBlock(inputAddress, outputAddress, levelAddresses[0], h[0]);
    computeBarrier(cmd);
    for (std::uint32_t k = 1; k < topLevel; ++k)
    {
        scanBlock(levelAddresses[k - 1], levelAddresses[k - 1], levelAddresses[k], h[k]);
        computeBarrier(cmd);
    }
    scanBlock(levelAddresses[topLevel - 1], levelAddresses[topLevel - 1], totalAddress,
              h[topLevel]);
    computeBarrier(cmd);

    // Downward: add each level's global block offsets (the scanned level above) back into it.
    for (std::uint32_t k = topLevel; k-- > 0;)
    {
        if (k == 0)
        {
            scanAdd(outputAddress, levelAddresses[0], h[0]);
        }
        else
        {
            scanAdd(levelAddresses[k - 1], levelAddresses[k], h[k]);
        }
        if (k > 0)
        {
            computeBarrier(cmd);
        }
    }
}

} // namespace fire_engine
