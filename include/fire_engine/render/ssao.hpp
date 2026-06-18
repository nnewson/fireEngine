#pragma once

#include <array>
#include <cstdint>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/render_tunables.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;
class Swapchain;

// Screen-space ambient occlusion + contact shadows. Runs after the depth
// prepass: a fullscreen pass samples the shared scene depth, reconstructs
// view-space position + normal, estimates hemisphere occlusion (+ a sun-direction
// contact-shadow ray-march), and writes an R8G8 target (R = AO, G = contact). The
// forward shader samples it (forward set 1) to modulate ambient / sun visibility.
//
// Structured like Taa / ParticleSystem: renderer-owned, recreate() on resize.
// Always runs — disabling via the tunables zeroes the intensity so the pass
// writes 1.0, avoiding conditional passes / barriers in the renderer.
class Ssao
{
public:
    Ssao(const Device& device, const Swapchain& swapchain, Resources& resources);
    ~Ssao() = default;

    Ssao(const Ssao&) = delete;
    Ssao& operator=(const Ssao&) = delete;
    Ssao(Ssao&&) noexcept = default;
    Ssao& operator=(Ssao&&) noexcept = default;

    // The blurred AO target — what the forward shader samples.
    [[nodiscard]] TextureHandle aoTarget() const noexcept
    {
        return blurHandle_;
    }

    // Write this frame's UBO: the jittered projection the depth was rendered with,
    // the view-space sun direction (for contact shadows), and the live tunables.
    void update(const Mat4& jitteredProj, Vec3 sunViewDir, const RenderTunables& tunables,
                uint32_t frameIndex);

    // Record the SSAO pass: borrow the prepass depth (attachment -> read-only ->
    // attachment), write the AO target, and leave it shader-read for the forward
    // pass. Precondition: scene depth in DepthStencilAttachmentOptimal.
    void recordPass(vk::CommandBuffer cmd, uint32_t frameIndex) const;

    // Recreate the AO target at the current swapchain extent and rebind the
    // (recreated) scene-depth view. Also the one-time post-construction setup
    // (the swapchain depth image is created after this subsystem).
    void recreate();

private:
    void createDescriptors();
    void writeSceneDepth();

    const Device* device_{nullptr};
    const Swapchain* swapchain_{nullptr};
    Resources* resources_{nullptr};
    Pipeline pipeline_;                     // SSAO + contact ray-march -> raw AO
    Pipeline blurPipeline_;                 // bilateral blur: raw AO + depth -> smoothed AO
    TextureHandle aoHandle_{NullTexture};   // raw
    TextureHandle blurHandle_{NullTexture}; // smoothed (sampled by the forward pass)
    Resources::MappedBufferSet ubo_;
    vk::raii::DescriptorPool descriptorPool_{nullptr};
    vk::raii::DescriptorSets sets_{nullptr};     // SSAO: depth + UBO
    vk::raii::DescriptorSets blurSets_{nullptr}; // blur: raw AO + depth
    vk::raii::Sampler depthSampler_{nullptr};
    // CPU-generated hemisphere kernel, uploaded into each frame's UBO.
    std::array<std::array<float, 4>, kSsaoKernelSize> kernel_{};
    // proj[2][2] / proj[3][2] from the latest update(), for the blur's depth
    // linearisation push constant.
    float projC_{0.0f};
    float projD_{0.0f};
};

} // namespace fire_engine
