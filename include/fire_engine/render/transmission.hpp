#pragma once

#include <span>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

class Transmission
{
public:
    Transmission(const Swapchain& swapchain, Resources& resources,
                 TextureHandle offscreenColourHandle);
    ~Transmission() = default;

    Transmission(const Transmission&) = delete;
    Transmission& operator=(const Transmission&) = delete;
    Transmission(Transmission&&) noexcept = default;
    Transmission& operator=(Transmission&&) noexcept = default;

    void recordPass(vk::CommandBuffer cmd, std::span<const DrawCommand> transmissiveDraws,
                    vk::DescriptorSet globalSet) const;
    void recreate(TextureHandle offscreenColourHandle, TextureHandle velocityHandle);

private:
    void rebuildSceneColorChain();
    void recordSceneColorCapture(vk::CommandBuffer cmd) const;
    void recordForwardTransmissionPass(vk::CommandBuffer cmd,
                                       std::span<const DrawCommand> transmissiveDraws,
                                       vk::DescriptorSet globalSet) const;

    const Swapchain* swapchain_{nullptr};
    Resources* resources_{nullptr};
    TextureHandle offscreenColourHandle_{NullTexture};
    TextureHandle velocityHandle_{NullTexture};
    TextureHandle sceneColorHandle_{NullTexture};
    uint32_t sceneColorMipLevels_{0};
};

} // namespace fire_engine
