#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

#include <fire_engine/render/compute_pipeline.hpp>
#include <fire_engine/render/device.hpp>

namespace fire_engine
{

// GPU exclusive prefix-sum (VDPM Stage B2 emit compaction) — a recursive hierarchical scan over the
// `vdpm_scan_block` / `vdpm_scan_add` shaders, block size `kScanElementsPerBlock`. Each level scans
// its blocks and emits per-block totals; the totals are scanned the same way (recursively) and
// their offsets added back down, with a compute→compute barrier between EVERY level and EVERY add.
// Only the GPU primitive lives here; the caller owns the input/output/scratch buffers (per-frame in
// the emit) and any surrounding synchronisation.
class VdpmScan
{
public:
    // Does the device meet the fixed-local-size (256) requirements? Checked BEFORE building the
    // pipeline so an unsupported GPU rejects the VDPM GPU backend (caller keeps the CPU fallback),
    // rather than failing pipeline creation. Vulkan only guarantees lower minimums.
    [[nodiscard]] static bool deviceSupported(const Device& device);

    // Throws std::runtime_error if !deviceSupported (construct only after checking).
    explicit VdpmScan(const Device& device);

    // The per-level element counts for scanning `count` elements: n[0] = count, n[k+1] =
    // ceil(n[k] / block), terminating at the first level that fits ONE block (n[K] <= block). So
    // `size() - 1` internal scratch levels are needed; `count <= block` gives just {count}. Empty
    // input gives {0}.
    [[nodiscard]] static std::vector<std::uint32_t> hierarchy(std::uint32_t count);

    // Scratch buffer element counts (uint) needed to scan `count` elements: one per internal level
    // (hierarchy levels 1..K), each `hierarchy(count)[k]` long. Empty when count <= block. The
    // grand total is a separate single-uint buffer the caller supplies.
    [[nodiscard]] static std::vector<std::uint32_t> scratchElementCounts(std::uint32_t count);

    // Record the full exclusive scan: `input[count]` (device address) → `output` (device address);
    // the grand total is written to `totalAddress` (a single uint). `levelAddresses` are the
    // scratch buffers' device addresses, one per internal level (see scratchElementCounts), each
    // holding that level's block sums (scanned in place). Records NO barrier before the first
    // dispatch or after the last — the caller brackets it — but inserts a
    // compute-write→compute-read barrier between every internal level and every offset-add. A zero
    // `count` records nothing (the caller pre-zeros the total). Throws std::runtime_error if any
    // level's group count exceeds maxComputeWorkGroupCount[0].
    void recordScan(vk::CommandBuffer cmd, std::uint64_t inputAddress, std::uint64_t outputAddress,
                    std::span<const std::uint64_t> levelAddresses, std::uint64_t totalAddress,
                    std::uint32_t count) const;

private:
    // Token proving deviceSupported ran. The public ctor delegates through `requireSupported`,
    // whose argument is evaluated (and may throw) BEFORE the private target ctor builds the
    // pipelines — so an unsupported device rejects cleanly instead of faulting at fixed-local-size
    // pipeline creation.
    struct Checked
    {
    };
    [[nodiscard]] static Checked requireSupported(const Device& device);
    VdpmScan(const Device& device, Checked);

    ComputePipeline blockPipeline_; // vdpm_scan_block.comp
    ComputePipeline addPipeline_;   // vdpm_scan_add.comp
    std::uint32_t maxWorkGroupCountX_{0};
};

} // namespace fire_engine
