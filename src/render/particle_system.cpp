#include <fire_engine/render/particle_system.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

namespace
{

// One Particle in the pool SSBO: 3 vec4 = 48 bytes (std430). Mirrors the
// Particle struct in shaders/particle_simulate.comp and particle.vert.
constexpr vk::DeviceSize kParticleBytes = 48;

ComputePipelineConfig simulateConfig()
{
    ComputePipelineConfig config;
    config.compShaderPath = "particle_simulate.comp.spv";
    config.bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    return config;
}

// HDR target layout transition for the additive particle pass.
void hdrBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout oldLayout,
                vk::ImageLayout newLayout, vk::PipelineStageFlags2 srcStage,
                vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                vk::AccessFlags2 dstAccess)
{
    vk::ImageMemoryBarrier2 b{
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}

} // namespace

ParticleSystem::ParticleSystem(const Device& device, const Swapchain& swapchain,
                               Resources& resources, TextureHandle hdrTarget)
    : device_{&device},
      swapchain_{&swapchain},
      resources_{&resources},
      offscreenColourHandle_{hdrTarget},
      simulatePipeline_(device, simulateConfig()),
      renderPipeline_(device, Pipeline::particleConfig(vk::Format::eR16G16B16A16Sfloat))
{
    poolBytes_ = static_cast<vk::DeviceSize>(kPoolSize) * kParticleBytes;
    claimBytes_ = static_cast<vk::DeviceSize>(kMaxParticleEmitters) * sizeof(uint32_t);

    // Zero-init the pool so every slot starts dead (lifetime 0 -> respawn).
    std::vector<std::byte> zeros(static_cast<std::size_t>(poolBytes_), std::byte{0});
    particlePool_ =
        resources_->createMappedStorageBuffer(static_cast<std::size_t>(poolBytes_), zeros.data());
    spawnClaim_ =
        resources_->createMappedStorageBuffer(static_cast<std::size_t>(claimBytes_), nullptr);
    frameUbo_ = resources_->createMappedUniformBuffers(sizeof(ParticleFrameUBO));

    depthSampler_ =
        vk::raii::Sampler(device.device(), vk::SamplerCreateInfo{
                                               .magFilter = vk::Filter::eNearest,
                                               .minFilter = vk::Filter::eNearest,
                                               .mipmapMode = vk::SamplerMipmapMode::eNearest,
                                               .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                                               .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                                               .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                                           });

    createDescriptors();
}

void ParticleSystem::createDescriptors()
{
    const vk::raii::Device& dev = device_->device();

    // Compute: 2 storage + 1 uniform per set. Render: 1 storage + 1 uniform.
    std::array<vk::DescriptorPoolSize, 3> sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3u * kMaxFramesInFlight},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 2u * kMaxFramesInFlight},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1u * kMaxFramesInFlight},
    };
    vk::DescriptorPoolCreateInfo poolCi{
        // raii DescriptorSets free individually on destruction.
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 2u * static_cast<uint32_t>(kMaxFramesInFlight),
        .poolSizeCount = static_cast<uint32_t>(sizes.size()),
        .pPoolSizes = sizes.data(),
    };
    descriptorPool_ = vk::raii::DescriptorPool(dev, poolCi);

    std::array<vk::DescriptorSetLayout, kMaxFramesInFlight> computeLayouts;
    computeLayouts.fill(simulatePipeline_.descriptorSetLayout());
    computeSets_ = vk::raii::DescriptorSets(
        dev, vk::DescriptorSetAllocateInfo{.descriptorPool = *descriptorPool_,
                                           .descriptorSetCount = kMaxFramesInFlight,
                                           .pSetLayouts = computeLayouts.data()});

    std::array<vk::DescriptorSetLayout, kMaxFramesInFlight> renderLayouts;
    renderLayouts.fill(renderPipeline_.descriptorSetLayout());
    renderSets_ = vk::raii::DescriptorSets(
        dev, vk::DescriptorSetAllocateInfo{.descriptorPool = *descriptorPool_,
                                           .descriptorSetCount = kMaxFramesInFlight,
                                           .pSetLayouts = renderLayouts.data()});

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        const vk::DescriptorBufferInfo poolInfo{resources_->vulkanBuffer(particlePool_.buffers[i]),
                                                0, poolBytes_};
        const vk::DescriptorBufferInfo claimInfo{resources_->vulkanBuffer(spawnClaim_.buffers[i]),
                                                 0, claimBytes_};
        const vk::DescriptorBufferInfo frameInfo{resources_->vulkanBuffer(frameUbo_.buffers[i]), 0,
                                                 sizeof(ParticleFrameUBO)};

        const vk::DescriptorSet computeSet = *computeSets_[i];
        const vk::DescriptorSet renderSet = *renderSets_[i];
        std::array<vk::WriteDescriptorSet, 5> writes{
            vk::WriteDescriptorSet{.dstSet = computeSet,
                                   .dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageBuffer,
                                   .pBufferInfo = &poolInfo},
            vk::WriteDescriptorSet{.dstSet = computeSet,
                                   .dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageBuffer,
                                   .pBufferInfo = &claimInfo},
            vk::WriteDescriptorSet{.dstSet = computeSet,
                                   .dstBinding = 2,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eUniformBuffer,
                                   .pBufferInfo = &frameInfo},
            vk::WriteDescriptorSet{.dstSet = renderSet,
                                   .dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageBuffer,
                                   .pBufferInfo = &poolInfo},
            vk::WriteDescriptorSet{.dstSet = renderSet,
                                   .dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eUniformBuffer,
                                   .pBufferInfo = &frameInfo},
        };
        dev.updateDescriptorSets(writes, {});
    }
    // Scene-depth binding (render set, binding 2) is written separately by
    // writeSceneDepth() once the swapchain depth image exists — the Renderer
    // creates depth resources after constructing the ParticleSystem.
}

void ParticleSystem::writeSceneDepth()
{
    const vk::raii::Device& dev = device_->device();
    const vk::DescriptorImageInfo depthInfo{
        .sampler = *depthSampler_,
        .imageView = swapchain_->depthView(),
        .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
    };
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        const vk::WriteDescriptorSet write{
            .dstSet = *renderSets_[i],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &depthInfo,
        };
        dev.updateDescriptorSets(write, {});
    }
}

void ParticleSystem::update(std::span<const EmitterState> emitters, const Mat4& view,
                            const Mat4& proj, float dt, uint32_t frameIndex)
{
    const uint32_t count =
        std::min<uint32_t>(static_cast<uint32_t>(emitters.size()), kMaxParticleEmitters);

    ParticleFrameUBO ubo{};
    ubo.view = view;
    ubo.proj = proj;
    ubo.dt = dt;
    ubo.frameCounter = frameCounter_++;
    ubo.emitterCount = count;
    ubo.particlesPerEmitter = static_cast<uint32_t>(kMaxParticlesPerEmitter);

    for (uint32_t e = 0; e < count; ++e)
    {
        const EmitterState& s = emitters[e];

        spawnAccumulators_[e] += s.spawnRate * dt;
        const float budget = std::floor(spawnAccumulators_[e]);
        spawnAccumulators_[e] -= budget;

        ParticleEmitterGpu& g = ubo.emitters[e];
        g.posCone[0] = s.worldPosition.x();
        g.posCone[1] = s.worldPosition.y();
        g.posCone[2] = s.worldPosition.z();
        g.posCone[3] = s.coneAngle;
        g.velLifetime[0] = s.baseVelocity.x();
        g.velLifetime[1] = s.baseVelocity.y();
        g.velLifetime[2] = s.baseVelocity.z();
        g.velLifetime[3] = s.lifetime;
        g.colourSize[0] = s.colour.r() * s.intensity;
        g.colourSize[1] = s.colour.g() * s.intensity;
        g.colourSize[2] = s.colour.b() * s.intensity;
        g.colourSize[3] = s.size;
        g.gravitySpawn[0] = s.gravity;
        g.gravitySpawn[1] = budget;
    }
    for (uint32_t e = count; e < kMaxParticleEmitters; ++e)
    {
        spawnAccumulators_[e] = 0.0f;
    }

    std::memcpy(frameUbo_.mapped[frameIndex], &ubo, sizeof(ubo));
}

void ParticleSystem::recordSimulate(vk::CommandBuffer cmd, uint32_t frameIndex) const
{
    const vk::Buffer claim = resources_->vulkanBuffer(spawnClaim_.buffers[frameIndex]);
    const vk::Buffer pool = resources_->vulkanBuffer(particlePool_.buffers[frameIndex]);

    // Reset the per-emitter spawn-claim counters to 0 on the GPU.
    cmd.fillBuffer(claim, 0, claimBytes_, 0u);
    recordBufferBarrier(cmd, makeBufferMemoryBarrier(vk::PipelineStageFlagBits2::eTransfer,
                                                     vk::AccessFlagBits2::eTransferWrite,
                                                     vk::PipelineStageFlagBits2::eComputeShader,
                                                     vk::AccessFlagBits2::eShaderStorageRead |
                                                         vk::AccessFlagBits2::eShaderStorageWrite,
                                                     claim, 0, claimBytes_));

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, simulatePipeline_.pipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, simulatePipeline_.pipelineLayout(), 0,
                           *computeSets_[frameIndex], {});
    const uint32_t groups = (kPoolSize + 63u) / 64u;
    cmd.dispatch(groups, 1, 1);

    // Pool written by compute -> read by the billboard vertex shader.
    recordBufferBarrier(cmd, makeBufferMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader,
                                                     vk::AccessFlagBits2::eShaderStorageWrite,
                                                     vk::PipelineStageFlagBits2::eVertexShader,
                                                     vk::AccessFlagBits2::eShaderStorageRead, pool,
                                                     0, poolBytes_));
}

void ParticleSystem::recordRender(vk::CommandBuffer cmd, uint32_t frameIndex) const
{
    const vk::Image hdrImage = resources_->vulkanImage(offscreenColourHandle_);
    const vk::ImageView hdrView = resources_->vulkanImageView(offscreenColourHandle_);
    const vk::Extent2D extent = swapchain_->extent();
    const vk::Rect2D area{.offset = {0, 0}, .extent = extent};

    // Scene depth: attachment -> read-only so the fragment shader can sample it
    // for soft-particle fade. Not needed downstream, so no restore.
    vk::ImageMemoryBarrier2 depthToRead{
        .srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                        vk::PipelineStageFlagBits2::eLateFragmentTests,
        .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .newLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = swapchain_->depthImage(),
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthToRead});

    // The HDR target is ShaderReadOnly coming out of the forward/transmission
    // passes; bring it back to a colour attachment to blend particles on top.
    hdrBarrier(cmd, hdrImage, vk::ImageLayout::eShaderReadOnlyOptimal,
               vk::ImageLayout::eColorAttachmentOptimal,
               vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
               vk::PipelineStageFlagBits2::eColorAttachmentOutput,
               vk::AccessFlagBits2::eColorAttachmentWrite);

    vk::RenderingAttachmentInfo colour{
        .imageView = hdrView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmd.beginRendering(makeRenderingInfo(area, {&colour, 1}, nullptr));
    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, area);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderPipeline_.pipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, renderPipeline_.pipelineLayout(), 0,
                           *renderSets_[frameIndex], {});
    const ParticleSoftPushConstants push{kCameraNearPlane, kCameraFarPlane, kParticleSoftFadeRange,
                                         0.0f};
    cmd.pushConstants<ParticleSoftPushConstants>(renderPipeline_.pipelineLayout(),
                                                 vk::ShaderStageFlagBits::eFragment, 0, push);
    cmd.draw(6, kPoolSize, 0, 0);
    cmd.endRendering();

    // Back to ShaderReadOnly for bloom / post-process sampling.
    hdrBarrier(cmd, hdrImage, vk::ImageLayout::eColorAttachmentOptimal,
               vk::ImageLayout::eShaderReadOnlyOptimal,
               vk::PipelineStageFlagBits2::eColorAttachmentOutput,
               vk::AccessFlagBits2::eColorAttachmentWrite,
               vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
}

void ParticleSystem::recreate(TextureHandle hdrTarget)
{
    offscreenColourHandle_ = hdrTarget;
    // The swapchain depth image/view were recreated; rebind the sampled depth.
    writeSceneDepth();
}

} // namespace fire_engine
