#pragma once

#include <array>
#include <vector>

#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

class Device;

class PostProcessing
{
public:
    PostProcessing(const Device& device, const Swapchain& swapchain, Resources& resources);
    ~PostProcessing() = default;

    PostProcessing(const PostProcessing&) = delete;
    PostProcessing& operator=(const PostProcessing&) = delete;
    PostProcessing(PostProcessing&&) noexcept = default;
    PostProcessing& operator=(PostProcessing&&) noexcept = default;

    [[nodiscard]] TextureHandle offscreenColourTarget() const noexcept
    {
        return offscreenColourHandle_;
    }

    void transitionOffscreenForSampling(vk::CommandBuffer cmd) const;
    void recordBloomPasses(vk::CommandBuffer cmd) const;
    void recordPostProcessPass(vk::CommandBuffer cmd, uint32_t imageIndex,
                               uint32_t currentFrame) const;
    void recreate();

private:
    void buildBloomResources();

    const Swapchain* swapchain_{nullptr};
    Resources* resources_{nullptr};
    Pipeline postProcessPipeline_;
    Pipeline bloomDownsamplePipeline_;
    Pipeline bloomUpsamplePipeline_;
    TextureHandle offscreenColourHandle_{NullTexture};
    TextureHandle bloomChainHandle_{NullTexture};
    std::vector<DescriptorSetHandle> bloomDownDescSets_{};
    std::vector<DescriptorSetHandle> bloomUpDescSets_{};
    std::array<DescriptorSetHandle, kMaxFramesInFlight> postProcessDescSets_{NullDescriptorSet,
                                                                             NullDescriptorSet};
    BufferHandle postProcessIndexBuffer_{NullBuffer};
};

} // namespace fire_engine
