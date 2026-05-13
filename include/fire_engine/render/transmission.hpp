#pragma once

#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/render/render_pass.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

class Device;

class Transmission
{
public:
    Transmission(const Device& device, const Swapchain& swapchain, Resources& resources,
                 TextureHandle offscreenColourHandle);
    ~Transmission() = default;

    Transmission(const Transmission&) = delete;
    Transmission& operator=(const Transmission&) = delete;
    Transmission(Transmission&&) noexcept = default;
    Transmission& operator=(Transmission&&) noexcept = default;

    void recordPass(vk::CommandBuffer cmd, std::span<const DrawCommand> transmissiveDraws,
                    vk::DescriptorSet globalSet) const;
    void recreate(TextureHandle offscreenColourHandle);

private:
    void buildFramebuffer();
    void rebuildSceneColorChain();
    void recordSceneColorCapture(vk::CommandBuffer cmd) const;
    void recordForwardTransmissionPass(vk::CommandBuffer cmd,
                                       std::span<const DrawCommand> transmissiveDraws,
                                       vk::DescriptorSet globalSet) const;

    const Device* device_{nullptr};
    const Swapchain* swapchain_{nullptr};
    Resources* resources_{nullptr};
    RenderPass forwardTransmissionPass_;
    TextureHandle offscreenColourHandle_{NullTexture};
    TextureHandle sceneColorHandle_{NullTexture};
    uint32_t sceneColorMipLevels_{0};
};

} // namespace fire_engine
