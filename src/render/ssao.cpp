#include <fire_engine/render/ssao.hpp>

#include <cstring>
#include <random>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

namespace
{

// R8G8 AO target: R = ambient occlusion, G = contact term. Must match
// Resources::createSsaoTarget.
constexpr vk::Format kAoFormat = vk::Format::eR8G8Unorm;

// Single-subresource layout transition (synchronization2). Dynamic rendering
// does no implicit transitions, so the SSAO pass cycles depth + AO explicitly.
void imageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageAspectFlags aspect,
                  vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                  vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                  vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
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
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = aspect,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}

} // namespace

Ssao::Ssao(const Device& device, const Swapchain& swapchain, Resources& resources)
    : device_{&device},
      swapchain_{&swapchain},
      resources_{&resources},
      pipeline_(device, Pipeline::ssaoConfig(kAoFormat)),
      blurPipeline_(device, Pipeline::ssaoBlurConfig(kAoFormat))
{
    ubo_ = resources_->createMappedUniformBuffers(sizeof(SsaoUBO));

    depthSampler_ =
        vk::raii::Sampler(device.device(), vk::SamplerCreateInfo{
                                               .magFilter = vk::Filter::eNearest,
                                               .minFilter = vk::Filter::eNearest,
                                               .mipmapMode = vk::SamplerMipmapMode::eNearest,
                                               .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                                               .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                                               .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                                           });

    // Hemisphere kernel: points in the +Z tangent hemisphere, biased toward the
    // origin (scale = lerp(0.1, 1, t^2)) so near-surface samples dominate.
    std::mt19937 rng{1337u};
    std::uniform_real_distribution<float> dist01{0.0f, 1.0f};
    std::uniform_real_distribution<float> distNeg{-1.0f, 1.0f};
    for (std::size_t i = 0; i < kSsaoKernelSize; ++i)
    {
        Vec3 s{distNeg(rng), distNeg(rng), dist01(rng)};
        s = Vec3::normalise(s) * dist01(rng);
        float t = static_cast<float>(i) / static_cast<float>(kSsaoKernelSize);
        float scale = 0.1f + 0.9f * t * t;
        s *= scale;
        kernel_[i] = {s.x(), s.y(), s.z(), 0.0f};
    }

    createDescriptors();
}

void Ssao::createDescriptors()
{
    const vk::raii::Device& dev = device_->device();
    // SSAO set: 1 CIS (depth) + 1 UBO. Blur set: 2 CIS (raw AO + depth).
    std::array<vk::DescriptorPoolSize, 2> sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 3u * kMaxFramesInFlight},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1u * kMaxFramesInFlight},
    };
    vk::DescriptorPoolCreateInfo poolCi{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 2u * static_cast<uint32_t>(kMaxFramesInFlight),
        .poolSizeCount = static_cast<uint32_t>(sizes.size()),
        .pPoolSizes = sizes.data(),
    };
    descriptorPool_ = vk::raii::DescriptorPool(dev, poolCi);

    std::array<vk::DescriptorSetLayout, kMaxFramesInFlight> layouts;
    layouts.fill(pipeline_.descriptorSetLayout());
    sets_ = vk::raii::DescriptorSets(
        dev, vk::DescriptorSetAllocateInfo{.descriptorPool = *descriptorPool_,
                                           .descriptorSetCount = kMaxFramesInFlight,
                                           .pSetLayouts = layouts.data()});

    std::array<vk::DescriptorSetLayout, kMaxFramesInFlight> blurLayouts;
    blurLayouts.fill(blurPipeline_.descriptorSetLayout());
    blurSets_ = vk::raii::DescriptorSets(
        dev, vk::DescriptorSetAllocateInfo{.descriptorPool = *descriptorPool_,
                                           .descriptorSetCount = kMaxFramesInFlight,
                                           .pSetLayouts = blurLayouts.data()});

    // Binding 1 (UBO) is stable; the depth + raw-AO image bindings are written by
    // writeSceneDepth() once the swapchain depth + AO targets exist (recreate()).
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        const vk::DescriptorBufferInfo uboInfo{resources_->vulkanBuffer(ubo_.buffers[i]), 0,
                                               sizeof(SsaoUBO)};
        const vk::WriteDescriptorSet write{.dstSet = *sets_[i],
                                           .dstBinding = 1,
                                           .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                           .pBufferInfo = &uboInfo};
        dev.updateDescriptorSets(write, {});
    }
}

void Ssao::writeSceneDepth()
{
    const vk::raii::Device& dev = device_->device();
    const vk::DescriptorImageInfo depthInfo{
        .sampler = *depthSampler_,
        .imageView = swapchain_->depthView(),
        .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
    };
    const vk::DescriptorImageInfo rawAoInfo{
        .sampler = resources_->vulkanSampler(aoHandle_),
        .imageView = resources_->vulkanImageView(aoHandle_),
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    constexpr auto kCis = vk::DescriptorType::eCombinedImageSampler;
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        std::array<vk::WriteDescriptorSet, 3> writes{
            // SSAO set: binding 0 = scene depth.
            vk::WriteDescriptorSet{.dstSet = *sets_[i],
                                   .dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = kCis,
                                   .pImageInfo = &depthInfo},
            // Blur set: binding 0 = raw AO, binding 1 = scene depth.
            vk::WriteDescriptorSet{.dstSet = *blurSets_[i],
                                   .dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = kCis,
                                   .pImageInfo = &rawAoInfo},
            vk::WriteDescriptorSet{.dstSet = *blurSets_[i],
                                   .dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = kCis,
                                   .pImageInfo = &depthInfo},
        };
        dev.updateDescriptorSets(writes, {});
    }
}

void Ssao::recreate()
{
    if (aoHandle_ != NullTexture)
    {
        resources_->releaseTexture(aoHandle_);
        resources_->releaseTexture(blurHandle_);
    }
    aoHandle_ = resources_->createSsaoTarget(swapchain_->extent());
    blurHandle_ = resources_->createSsaoTarget(swapchain_->extent());
    writeSceneDepth();
}

void Ssao::update(const Mat4& jitteredProj, Vec3 sunViewDir, const RenderTunables& tunables,
                  uint32_t frameIndex)
{
    const auto extent = swapchain_->extent();
    // proj[2][2] / proj[3][2] (column-major: operator[row,col]) feed the blur's
    // depth linearisation push constant.
    projC_ = jitteredProj[2, 2];
    projD_ = jitteredProj[2, 3];
    SsaoUBO data{};
    data.proj = jitteredProj;
    for (std::size_t i = 0; i < kSsaoKernelSize; ++i)
    {
        data.kernel[i][0] = kernel_[i][0];
        data.kernel[i][1] = kernel_[i][1];
        data.kernel[i][2] = kernel_[i][2];
        data.kernel[i][3] = kernel_[i][3];
    }
    data.params[0] = tunables.ssaoRadius;
    data.params[1] = tunables.ssaoBias;
    // Disabled => intensity 0 => the pass writes AO = 1 (no darkening).
    data.params[2] = tunables.ssaoEnabled ? tunables.ssaoIntensity : 0.0f;
    data.params[3] = tunables.ssaoPower;
    data.contact[0] = tunables.contactShadowLength;
    data.contact[1] = static_cast<float>(kContactShadowSteps);
    data.contact[2] = tunables.contactShadowsEnabled ? 1.0f : 0.0f;
    data.contact[3] = tunables.contactEdgeThreshold;
    data.sunViewDir[0] = sunViewDir.x();
    data.sunViewDir[1] = sunViewDir.y();
    data.sunViewDir[2] = sunViewDir.z();
    data.screen[0] = static_cast<float>(extent.width);
    data.screen[1] = static_cast<float>(extent.height);
    data.screen[2] = 1.0f / static_cast<float>(extent.width);
    data.screen[3] = 1.0f / static_cast<float>(extent.height);
    std::memcpy(ubo_.mapped[frameIndex], &data, sizeof(data));
}

void Ssao::recordPass(vk::CommandBuffer cmd, uint32_t frameIndex) const
{
    const auto extent = swapchain_->extent();
    vk::Rect2D renderArea{.offset = vk::Offset2D{.x = 0, .y = 0}, .extent = extent};

    // Borrow the prepass depth read-only for sampling.
    imageBarrier(cmd, swapchain_->depthImage(), vk::ImageAspectFlagBits::eDepth,
                 vk::ImageLayout::eDepthStencilAttachmentOptimal,
                 vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                 vk::PipelineStageFlagBits2::eLateFragmentTests,
                 vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                 vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);

    // AO target -> colour attachment (fullscreen overwrite, so discard the old).
    const vk::Image aoImage = resources_->vulkanImage(aoHandle_);
    imageBarrier(cmd, aoImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                 vk::ImageLayout::eColorAttachmentOptimal,
                 vk::PipelineStageFlagBits2::eFragmentShader, {},
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite);

    vk::RenderingAttachmentInfo colour{
        .imageView = resources_->vulkanImageView(aoHandle_),
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmd.beginRendering(makeRenderingInfo(renderArea, {&colour, 1}, nullptr));
    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_.pipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_.pipelineLayout(), 0,
                           *sets_[frameIndex], {});
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();

    // Raw AO -> shader-read so the bilateral blur can sample it.
    imageBarrier(cmd, aoImage, vk::ImageAspectFlagBits::eColor,
                 vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite,
                 vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);

    // Bilateral blur: raw AO + depth -> smoothed AO (the forward-sampled target).
    const vk::Image blurImage = resources_->vulkanImage(blurHandle_);
    imageBarrier(cmd, blurImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                 vk::ImageLayout::eColorAttachmentOptimal,
                 vk::PipelineStageFlagBits2::eFragmentShader, {},
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite);
    vk::RenderingAttachmentInfo blurColour{
        .imageView = resources_->vulkanImageView(blurHandle_),
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmd.beginRendering(makeRenderingInfo(renderArea, {&blurColour, 1}, nullptr));
    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, blurPipeline_.pipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, blurPipeline_.pipelineLayout(), 0,
                           *blurSets_[frameIndex], {});
    SsaoBlurPushConstants blurPush{};
    blurPush.texelSize[0] = 1.0f / static_cast<float>(extent.width);
    blurPush.texelSize[1] = 1.0f / static_cast<float>(extent.height);
    blurPush.projC = projC_;
    blurPush.projD = projD_;
    cmd.pushConstants<SsaoBlurPushConstants>(blurPipeline_.pipelineLayout(),
                                             vk::ShaderStageFlagBits::eFragment, 0, blurPush);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();

    // Smoothed AO -> shader-read for the forward pass; depth back to attachment so
    // the forward pass can load + test it.
    imageBarrier(cmd, blurImage, vk::ImageAspectFlagBits::eColor,
                 vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite,
                 vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
    imageBarrier(cmd, swapchain_->depthImage(), vk::ImageAspectFlagBits::eDepth,
                 vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                 vk::ImageLayout::eDepthStencilAttachmentOptimal,
                 vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
                 vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                 vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                     vk::AccessFlagBits2::eDepthStencilAttachmentRead);
}

} // namespace fire_engine
