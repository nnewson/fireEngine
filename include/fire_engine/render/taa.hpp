#pragma once

#include <array>

#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;
class Swapchain;

// Temporal anti-aliasing subsystem (Roadmap: AA). Owns the screen-space velocity
// (motion-vector) target written by the forward pass as a second colour
// attachment, plus the resolve: two ping-pong history targets and a fullscreen
// pass that reprojects the previous frame's accumulation along the velocity
// buffer, neighbourhood-clamps it, and blends it with the current frame. The
// resolved image is blitted back into the offscreen HDR target so the downstream
// particle / bloom / post passes consume the anti-aliased result unchanged.
class Taa
{
public:
    Taa(const Device& device, const Swapchain& swapchain, Resources& resources,
        TextureHandle offscreenColour);
    ~Taa() = default;

    Taa(const Taa&) = delete;
    Taa& operator=(const Taa&) = delete;
    Taa(Taa&&) noexcept = default;
    Taa& operator=(Taa&&) noexcept = default;

    [[nodiscard]] TextureHandle velocityTarget() const noexcept
    {
        return velocityHandle_;
    }

    // Resolve the current frame: reproject + clamp + blend history, then blit the
    // result into the offscreen HDR target. currentFrame selects the ping-pong
    // parity (history slot written this frame == currentFrame).
    void recordResolve(vk::CommandBuffer cmd, uint32_t currentFrame);

    // Recreate the velocity + history targets at the current swapchain extent and
    // rebind the resolve descriptors against the (possibly new) offscreen target.
    // Resets the history-valid guard so the first post-resize frame uses current
    // colour only.
    void recreate(TextureHandle offscreenColour);

private:
    const Swapchain* swapchain_{nullptr};
    Resources* resources_{nullptr};
    Pipeline resolvePipeline_;
    TextureHandle offscreenHandle_{NullTexture};
    TextureHandle velocityHandle_{NullTexture};
    std::array<TextureHandle, kMaxFramesInFlight> historyHandles_{};
    std::array<DescriptorSetHandle, kMaxFramesInFlight> resolveDescSets_{};
    // Per ping-pong slot: has it been written at least once since the last
    // (re)create? Gates the history-valid push constant so the resolve never
    // samples undefined accumulation.
    std::array<bool, kMaxFramesInFlight> historyWritten_{};
};

} // namespace fire_engine
