#pragma once

#include <array>
#include <cstdint>
#include <span>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/particle.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/render/compute_pipeline.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;
class Swapchain;

// Renderer-owned GPU particle system. Owns a persistent
// particle pool SSBO partitioned per emitter, simulates it with compute each
// frame, and renders it as instanced additive billboards into the HDR target.
// Structured like PostProcessing / Transmission. Emitters come from
// SceneGraph::gatherEmitters (the Light-style gather path).
class ParticleSystem
{
public:
    ParticleSystem(const Device& device, const Swapchain& swapchain, Resources& resources,
                   TextureHandle hdrTarget);
    ~ParticleSystem() = default;

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;
    ParticleSystem(ParticleSystem&&) noexcept = default;
    ParticleSystem& operator=(ParticleSystem&&) noexcept = default;

    // Write the per-frame UBO from gathered emitters + camera matrices, advancing
    // each emitter's spawn accumulator by spawnRate * dt to derive this frame's
    // integer spawn budget.
    void update(std::span<const EmitterState> emitters, const Mat4& view, const Mat4& proj,
                float dt, uint32_t frameIndex);

    // Clear spawn claims, dispatch the simulation, and barrier the pool for the
    // subsequent vertex-shader read.
    void recordSimulate(vk::CommandBuffer cmd, uint32_t frameIndex) const;

    // Render the pool as instanced billboards into the HDR target (additive),
    // bracketing with HDR layout barriers (ShaderReadOnly <-> ColorAttachment).
    void recordRender(vk::CommandBuffer cmd, uint32_t frameIndex) const;

    // Rebind the HDR target after a swapchain resize (the offscreen target is
    // recreated with a new handle).
    void recreate(TextureHandle hdrTarget);

private:
    void createDescriptors();
    // Bind the current swapchain scene-depth view into render-set binding 2.
    // Re-run after a swapchain resize (the depth view is recreated).
    void writeSceneDepth();

    const Device* device_{nullptr};
    const Swapchain* swapchain_{nullptr};
    Resources* resources_{nullptr};
    TextureHandle offscreenColourHandle_{NullTexture};

    ComputePipeline simulatePipeline_;
    Pipeline renderPipeline_;
    Resources::MappedBufferSet particlePool_; // shared SSBO (same handle per slot)
    Resources::MappedBufferSet spawnClaim_;   // shared SSBO (kMaxParticleEmitters counters)
    Resources::MappedBufferSet frameUbo_;     // per-frame UBO

    vk::raii::DescriptorPool descriptorPool_{nullptr};
    vk::raii::DescriptorSets computeSets_{nullptr};
    vk::raii::DescriptorSets renderSets_{nullptr};
    vk::raii::Sampler depthSampler_{nullptr};

    std::array<float, kMaxParticleEmitters> spawnAccumulators_{};
    uint32_t frameCounter_{0};

    static constexpr uint32_t kPoolSize = static_cast<uint32_t>(kMaxParticleEmitters) *
                                          static_cast<uint32_t>(kMaxParticlesPerEmitter);
    vk::DeviceSize poolBytes_{0};
    vk::DeviceSize claimBytes_{0};
};

} // namespace fire_engine
