#pragma once

#include <array>
#include <cstdint>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/gpu_limits.hpp>

namespace fire_engine
{

class Device;

// Per-pass GPU timing via a timestamp query pool. Each pass writes a begin/end
// timestamp pair; results are read back one cycle later (the acquire fence
// guarantees the slot is complete), so there's no stall. Degrades gracefully to
// disabled when the device/queue doesn't support timestamps (e.g. some MoltenVK
// configurations) — callers fall back to CPU frame timing.
enum class ProfilePass : uint32_t
{
    Shadow,
    DepthPrepass,
    Ssao,
    Forward,
    Transmission,
    Taa,
    Particles,
    DebugDraw,
    Bloom,
    Post,
    Count
};

inline constexpr uint32_t kProfilePassCount = static_cast<uint32_t>(ProfilePass::Count);

// Resolved per-frame timings consumed by the overlay.
struct FrameStats
{
    // Wall-clock frame time (from the main-loop dt). Always valid.
    float cpuFrameMs{0.0f};
    // Per-pass GPU milliseconds; only meaningful when gpuValid. A pass that
    // didn't run this cycle (e.g. transmission with no transmissive draws)
    // reports 0.
    std::array<float, kProfilePassCount> passMs{};
    float gpuTotalMs{0.0f};
    bool gpuValid{false};
};

class GpuProfiler
{
public:
    explicit GpuProfiler(const Device& device);
    ~GpuProfiler() = default;

    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;
    GpuProfiler(GpuProfiler&&) noexcept = default;
    GpuProfiler& operator=(GpuProfiler&&) noexcept = default;

    // Reset this frame's query range. Must be recorded outside a render pass,
    // before any begin()/end() for the frame.
    void beginFrame(vk::CommandBuffer cmd, uint32_t frameIndex);
    void begin(vk::CommandBuffer cmd, uint32_t frameIndex, ProfilePass pass) const;
    void end(vk::CommandBuffer cmd, uint32_t frameIndex, ProfilePass pass) const;

    // Read back the results currently held in `frameIndex`'s slot (written a full
    // ring-cycle ago) into out.passMs / gpuTotalMs / gpuValid. No-op when timing
    // is unsupported.
    void resolve(uint32_t frameIndex, FrameStats& out) const;

    [[nodiscard]] bool enabled() const noexcept
    {
        return enabled_;
    }

private:
    [[nodiscard]] uint32_t queryIndex(uint32_t frameIndex, ProfilePass pass,
                                      bool end) const noexcept
    {
        return frameIndex * kQueriesPerFrame + static_cast<uint32_t>(pass) * 2 + (end ? 1u : 0u);
    }

    static constexpr uint32_t kQueriesPerFrame = kProfilePassCount * 2;

    const vk::raii::Device* device_{nullptr};
    vk::raii::QueryPool pool_{nullptr};
    float timestampPeriodNs_{0.0f};
    bool enabled_{false};
    // A slot must be reset (beginFrame) at least once before its results may be
    // read, or validation flags reading uninitialised queries on the first cycle.
    std::array<bool, kMaxFramesInFlight> slotUsed_{};
};

} // namespace fire_engine
