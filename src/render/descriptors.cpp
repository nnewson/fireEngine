#include <fire_engine/render/descriptors.hpp>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

// Thin wrappers around vk::WriteDescriptorSet{...} so the long write array in
// writeGlobalBindings reads as one line per binding instead of five.
[[nodiscard]]
vk::WriteDescriptorSet writeBuffer(vk::DescriptorSet set, uint32_t binding, vk::DescriptorType type,
                                   const vk::DescriptorBufferInfo& info) noexcept
{
    return vk::WriteDescriptorSet{.dstSet = set,
                                  .dstBinding = binding,
                                  .descriptorCount = 1,
                                  .descriptorType = type,
                                  .pBufferInfo = &info};
}

[[nodiscard]]
vk::WriteDescriptorSet writeImage(vk::DescriptorSet set, uint32_t binding, vk::DescriptorType type,
                                  const vk::DescriptorImageInfo& info) noexcept
{
    return vk::WriteDescriptorSet{.dstSet = set,
                                  .dstBinding = binding,
                                  .descriptorCount = 1,
                                  .descriptorType = type,
                                  .pImageInfo = &info};
}

} // namespace

vk::DescriptorBufferInfo Descriptors::makeDescriptorBufferInfo(vk::Buffer buffer,
                                                               vk::DeviceSize range)
{
    return vk::DescriptorBufferInfo{
        .buffer = buffer,
        .offset = 0,
        .range = range,
    };
}

vk::DescriptorImageInfo Descriptors::makeDescriptorImageInfo(vk::Sampler sampler,
                                                             vk::ImageView imageView,
                                                             vk::ImageLayout imageLayout)
{
    return vk::DescriptorImageInfo{
        .sampler = sampler,
        .imageView = imageView,
        .imageLayout = imageLayout,
    };
}

Descriptors::Descriptors(const Device& device, const Pipeline& pipeline, const Resources& resources)
    : device_{&device},
      pipeline_{&pipeline},
      resources_{&resources}
{
}

Descriptors::DescriptorPoolEntry&
Descriptors::createDescriptorPool(std::span<const vk::DescriptorPoolSize> poolSizes,
                                  uint32_t maxSets)
{
    vk::DescriptorPoolCreateInfo ci{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    auto& poolEntry = descriptorPools_.emplace_back();
    poolEntry.pool = vk::raii::DescriptorPool(device_->device(), ci);
    return poolEntry;
}

std::vector<vk::raii::DescriptorSet>
Descriptors::allocateDescriptorSets(vk::DescriptorPool pool, vk::DescriptorSetLayout layout,
                                    uint32_t count) const
{
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);
    vk::DescriptorSetAllocateInfo ai{
        .descriptorPool = pool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
    };
    return device_->device().allocateDescriptorSets(ai);
}

DescriptorSetHandle Descriptors::registerDescriptorSet(vk::DescriptorSet set)
{
    auto dsHandle = static_cast<uint32_t>(descriptorSetTable_.size());
    descriptorSetTable_.push_back(set);
    return DescriptorSetHandle{dsHandle};
}

void Descriptors::retainDescriptorSets(DescriptorPoolEntry& poolEntry,
                                       std::vector<vk::raii::DescriptorSet>& sets)
{
    for (auto& s : sets)
    {
        poolEntry.sets.push_back(std::move(s));
    }
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::allocateFrameSets(DescriptorPoolEntry& poolEntry, vk::DescriptorSetLayout layout,
                               const FrameWriter& writeFrame)
{
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, kMaxFramesInFlight);

    std::array<DescriptorSetHandle, kMaxFramesInFlight> result{};
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        if (writeFrame)
        {
            writeFrame(*sets[i], i);
        }
        result[i] = registerDescriptorSet(*sets[i]);
    }

    retainDescriptorSets(poolEntry, sets);
    return result;
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::buildFrameSets(std::span<const vk::DescriptorPoolSize> poolSizes,
                            vk::DescriptorSetLayout layout, const FrameWriter& writeFrame)
{
    auto& poolEntry = createDescriptorPool(poolSizes, kMaxFramesInFlight);
    return allocateFrameSets(poolEntry, layout, writeFrame);
}

void pushForwardObjectDescriptors(vk::CommandBuffer cmd, const Resources& resources,
                                  vk::PipelineLayout layout, const DrawCommand& dc)
{
    // Forward set 0 is a push-descriptor layout: write the four per-object
    // vertex-stage buffers (frame/skin/morph UBOs + morph-targets SSBO) inline,
    // no allocated set. Material data is bindless (global set 2); WholeSize works
    // because each buffer was created exactly sized.
    const vk::DescriptorBufferInfo frameInfo{
        .buffer = resources.vulkanBuffer(dc.frameUbo), .offset = 0, .range = vk::WholeSize};
    const vk::DescriptorBufferInfo skinInfo{
        .buffer = resources.vulkanBuffer(dc.skinUbo), .offset = 0, .range = vk::WholeSize};
    const vk::DescriptorBufferInfo morphInfo{
        .buffer = resources.vulkanBuffer(dc.morphUbo), .offset = 0, .range = vk::WholeSize};
    const vk::DescriptorBufferInfo morphSsboInfo{
        .buffer = resources.vulkanBuffer(dc.morphSsbo), .offset = 0, .range = vk::WholeSize};

    constexpr auto kUbo = vk::DescriptorType::eUniformBuffer;
    constexpr auto kSsbo = vk::DescriptorType::eStorageBuffer;
    const std::array<vk::WriteDescriptorSet, 4> writes{{
        {.dstBinding = bindingIndex(ForwardBinding::Frame),
         .descriptorCount = 1,
         .descriptorType = kUbo,
         .pBufferInfo = &frameInfo},
        {.dstBinding = bindingIndex(ForwardBinding::Skin),
         .descriptorCount = 1,
         .descriptorType = kUbo,
         .pBufferInfo = &skinInfo},
        {.dstBinding = bindingIndex(ForwardBinding::Morph),
         .descriptorCount = 1,
         .descriptorType = kUbo,
         .pBufferInfo = &morphInfo},
        {.dstBinding = bindingIndex(ForwardBinding::MorphTargets),
         .descriptorCount = 1,
         .descriptorType = kSsbo,
         .pBufferInfo = &morphSsboInfo},
    }};
    // Core 1.4 entry point (vkCmdPushDescriptorSet) — the vcpkg loader exports
    // this, not the KHR-suffixed alias. Validity comes from the enabled
    // VK_KHR_push_descriptor extension plus the 1.4 device.
    cmd.pushDescriptorSet(vk::PipelineBindPoint::eGraphics, layout, 0, writes);
}

void pushShadowObjectDescriptors(vk::CommandBuffer cmd, const Resources& resources,
                                 vk::PipelineLayout layout, const DrawCommand& dc)
{
    // Shadow set 0 is a push-descriptor layout. Bindings 0..3 are per-object
    // vertex-stage buffers (ShadowUBO + skin/morph UBOs + morph SSBO); each was
    // created exactly sized, so WholeSize is correct. The shadow draw reuses the
    // forward skin/morph/morphSsbo handles carried on the DrawCommand.
    const vk::DescriptorBufferInfo shadowInfo{
        .buffer = resources.vulkanBuffer(dc.shadowUbo), .offset = 0, .range = vk::WholeSize};
    const vk::DescriptorBufferInfo skinInfo{
        .buffer = resources.vulkanBuffer(dc.skinUbo), .offset = 0, .range = vk::WholeSize};
    const vk::DescriptorBufferInfo morphInfo{
        .buffer = resources.vulkanBuffer(dc.morphUbo), .offset = 0, .range = vk::WholeSize};
    const vk::DescriptorBufferInfo morphSsboInfo{
        .buffer = resources.vulkanBuffer(dc.morphSsbo), .offset = 0, .range = vk::WholeSize};

    // Bindings 4/5 are the shared self-shadow first-depth image + sampler —
    // global resources (same for every shadow draw), only sampled by the second
    // self-shadow pass. The first/regular passes declare them but never read
    // them, so pushing them here is harmless (validation only checks layout for
    // descriptors a shader statically uses).
    const vk::DescriptorImageInfo selfShadowFirstInfo{
        .sampler = {},
        .imageView = resources.vulkanImageView(resources.sharedTextures().selfShadowFirstMap),
        .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal};
    const vk::DescriptorImageInfo selfShadowSamplerInfo{.sampler =
                                                            resources.vulkanShadowDebugSampler(),
                                                        .imageView = {},
                                                        .imageLayout = vk::ImageLayout::eUndefined};

    constexpr auto kUbo = vk::DescriptorType::eUniformBuffer;
    constexpr auto kSsbo = vk::DescriptorType::eStorageBuffer;
    constexpr auto kSi = vk::DescriptorType::eSampledImage;
    constexpr auto kSamp = vk::DescriptorType::eSampler;
    const std::array<vk::WriteDescriptorSet, 6> writes{{
        {.dstBinding = bindingIndex(ShadowBinding::Shadow),
         .descriptorCount = 1,
         .descriptorType = kUbo,
         .pBufferInfo = &shadowInfo},
        {.dstBinding = bindingIndex(ShadowBinding::Skin),
         .descriptorCount = 1,
         .descriptorType = kUbo,
         .pBufferInfo = &skinInfo},
        {.dstBinding = bindingIndex(ShadowBinding::Morph),
         .descriptorCount = 1,
         .descriptorType = kUbo,
         .pBufferInfo = &morphInfo},
        {.dstBinding = bindingIndex(ShadowBinding::MorphTargets),
         .descriptorCount = 1,
         .descriptorType = kSsbo,
         .pBufferInfo = &morphSsboInfo},
        {.dstBinding = bindingIndex(ShadowBinding::SelfShadowFirstMap),
         .descriptorCount = 1,
         .descriptorType = kSi,
         .pImageInfo = &selfShadowFirstInfo},
        {.dstBinding = bindingIndex(ShadowBinding::SelfShadowDepthSampler),
         .descriptorCount = 1,
         .descriptorType = kSamp,
         .pImageInfo = &selfShadowSamplerInfo},
    }};
    cmd.pushDescriptorSet(vk::PipelineBindPoint::eGraphics, layout, 0, writes);
}

void Descriptors::writeGlobalBindings(vk::DescriptorSet set, const GlobalDescriptorRequest& req,
                                      int frame) const
{
    const vk::DescriptorBufferInfo lightBufInfo =
        makeDescriptorBufferInfo(resources_->vulkanBuffer(req.lightBufs[frame]), sizeof(LightUBO));

    auto sampledImage = [this](TextureHandle handle)
    {
        return makeDescriptorImageInfo({}, resources_->vulkanImageView(handle),
                                       vk::ImageLayout::eDepthStencilReadOnlyOptimal);
    };
    auto combinedSampler = [this](TextureHandle handle)
    {
        return makeDescriptorImageInfo(resources_->vulkanSampler(handle),
                                       resources_->vulkanImageView(handle),
                                       vk::ImageLayout::eShaderReadOnlyOptimal);
    };
    auto plainSampler = [](vk::Sampler s)
    { return makeDescriptorImageInfo(s, {}, vk::ImageLayout::eUndefined); };

    const vk::DescriptorImageInfo shadowMapInfo = sampledImage(req.shadowMap);
    const vk::DescriptorImageInfo worldShadowMapInfo = sampledImage(req.worldShadowMap);
    const vk::DescriptorImageInfo selfShadowMapInfo = sampledImage(req.selfShadowMap);
    const vk::DescriptorImageInfo spotShadowMapInfo = sampledImage(req.spotShadowMap);
    const vk::DescriptorImageInfo pointShadowMapInfo = sampledImage(req.pointShadowMap);
    // shadowDebugImage is now the CSM depth map (sampled raw for the ShadowDepth
    // debug view), so it uses the depth-read-only layout like the other maps.
    const vk::DescriptorImageInfo shadowDebugImageInfo = sampledImage(req.shadowDebugImage);
    const vk::DescriptorImageInfo shadowCompareSamplerInfo =
        plainSampler(resources_->vulkanSampler(req.shadowMap));
    const vk::DescriptorImageInfo shadowDebugSamplerInfo =
        plainSampler(resources_->vulkanShadowDebugSampler());
    const vk::DescriptorImageInfo irradianceInfo = combinedSampler(req.irradianceMap);
    const vk::DescriptorImageInfo prefilteredInfo = combinedSampler(req.prefilteredMap);
    const vk::DescriptorImageInfo brdfLutInfo = combinedSampler(req.brdfLut);
    const vk::DescriptorImageInfo sceneColorInfo = combinedSampler(req.sceneColor);
    const vk::DescriptorImageInfo ssaoInfo = combinedSampler(req.ssaoMap);

    constexpr auto kUbo = vk::DescriptorType::eUniformBuffer;
    constexpr auto kCis = vk::DescriptorType::eCombinedImageSampler;
    constexpr auto kSi = vk::DescriptorType::eSampledImage;
    constexpr auto kSamp = vk::DescriptorType::eSampler;
    std::array<vk::WriteDescriptorSet, 14> writes = {{
        writeBuffer(set, bindingIndex(ForwardGlobalBinding::Light), kUbo, lightBufInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::ShadowMap), kSi, shadowMapInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::WorldShadowMap), kSi,
                   worldShadowMapInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::SelfShadowMap), kSi, selfShadowMapInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::SpotShadowMap), kSi, spotShadowMapInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::PointShadowMap), kSi,
                   pointShadowMapInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::ShadowDebugImage), kSi,
                   shadowDebugImageInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::ShadowCompareSampler), kSamp,
                   shadowCompareSamplerInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::ShadowDebugSampler), kSamp,
                   shadowDebugSamplerInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::IrradianceMap), kCis, irradianceInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::PrefilteredMap), kCis, prefilteredInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::BrdfLut), kCis, brdfLutInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::SceneColour), kCis, sceneColorInfo),
        writeImage(set, bindingIndex(ForwardGlobalBinding::SsaoMap), kCis, ssaoInfo),
    }};
    device_->device().updateDescriptorSets(writes, {});
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::createGlobalDescriptors(const GlobalDescriptorRequest& req)
{
    // One global set per frame-in-flight. Pool sizes are exact per the
    // ForwardGlobalBinding enum: 1 UBO + 5 CIS (IBL ×3 + sceneColor + ssao) + 6 SI
    // (shadow maps ×5 + debug colour) + 2 plain samplers (compare + debug).
    std::array<vk::DescriptorPoolSize, 4> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, kMaxFramesInFlight},
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight * 5},
        {vk::DescriptorType::eSampledImage, kMaxFramesInFlight * 6},
        {vk::DescriptorType::eSampler, kMaxFramesInFlight * 2},
    }};
    return buildFrameSets(poolSizes, pipeline_->globalDescriptorSetLayout(),
                          [&](vk::DescriptorSet set, int i) { writeGlobalBindings(set, req, i); });
}

void Descriptors::updateGlobalDescriptors(
    const std::array<DescriptorSetHandle, kMaxFramesInFlight>& sets,
    const GlobalDescriptorRequest& req)
{
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        writeGlobalBindings(vulkanDescriptorSet(sets[i]), req, i);
    }
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::createSingleUboDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& ubo,
                                        vk::DeviceSize uboSize)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, kMaxFramesInFlight},
    }};
    return buildFrameSets(poolSizes, layout,
                          [&](vk::DescriptorSet set, int i)
                          {
                              vk::DescriptorBufferInfo bufInfo = makeDescriptorBufferInfo(
                                  resources_->vulkanBuffer(ubo.buffers[i]), uboSize);
                              vk::WriteDescriptorSet write{
                                  .dstSet = set,
                                  .dstBinding = 0,
                                  .descriptorCount = 1,
                                  .descriptorType = vk::DescriptorType::eUniformBuffer,
                                  .pBufferInfo = &bufInfo,
                              };
                              device_->device().updateDescriptorSets(write, {});
                          });
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::createUboImageSamplerDescriptors(vk::DescriptorSetLayout layout,
                                              const MappedBufferSet& ubo, vk::DeviceSize uboSize,
                                              TextureHandle texture)
{
    std::array<vk::DescriptorPoolSize, 2> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, kMaxFramesInFlight},
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight},
    }};
    return buildFrameSets(
        poolSizes, layout,
        [&](vk::DescriptorSet set, int i)
        {
            vk::DescriptorBufferInfo bufInfo =
                makeDescriptorBufferInfo(resources_->vulkanBuffer(ubo.buffers[i]), uboSize);
            vk::DescriptorImageInfo texInfo = makeDescriptorImageInfo(
                resources_->vulkanSampler(texture), resources_->vulkanImageView(texture),
                vk::ImageLayout::eShaderReadOnlyOptimal);
            std::array<vk::WriteDescriptorSet, 2> writes = {{
                vk::WriteDescriptorSet{.dstSet = set,
                                       .dstBinding = bindingIndex(SkyboxBinding::Skybox),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &bufInfo},
                vk::WriteDescriptorSet{.dstSet = set,
                                       .dstBinding = bindingIndex(SkyboxBinding::Cubemap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &texInfo},
            }};
            device_->device().updateDescriptorSets(writes, {});
        });
}

std::array<DescriptorSetHandle, kMaxFramesInFlight> Descriptors::createSkyboxDescriptors(
    vk::DescriptorSetLayout layout, const MappedBufferSet& skyboxUbo, vk::DeviceSize skyboxUboSize,
    TextureHandle texture, const MappedBufferSet& lightUbo, vk::DeviceSize lightUboSize)
{
    std::array<vk::DescriptorPoolSize, 2> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, kMaxFramesInFlight * 2},
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight},
    }};
    return buildFrameSets(
        poolSizes, layout,
        [&](vk::DescriptorSet set, int i)
        {
            vk::DescriptorBufferInfo skyboxBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(skyboxUbo.buffers[i]), skyboxUboSize);
            vk::DescriptorBufferInfo lightBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(lightUbo.buffers[i]), lightUboSize);
            vk::DescriptorImageInfo texInfo = makeDescriptorImageInfo(
                resources_->vulkanSampler(texture), resources_->vulkanImageView(texture),
                vk::ImageLayout::eShaderReadOnlyOptimal);
            std::array<vk::WriteDescriptorSet, 3> writes = {{
                vk::WriteDescriptorSet{.dstSet = set,
                                       .dstBinding = 0,
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &skyboxBufInfo},
                vk::WriteDescriptorSet{.dstSet = set,
                                       .dstBinding = 1,
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &texInfo},
                vk::WriteDescriptorSet{.dstSet = set,
                                       .dstBinding = bindingIndex(SkyboxBinding::Light),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &lightBufInfo},
            }};
            device_->device().updateDescriptorSets(writes, {});
        });
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::createSingleImageSamplerDescriptors(vk::DescriptorSetLayout layout,
                                                 TextureHandle texture)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight},
    }};
    // Allocation only; the writes are issued through the shared update path so
    // creation and swapchain-resize recreation stay in lockstep.
    auto result = buildFrameSets(poolSizes, layout, {});
    updateSingleImageSamplerDescriptors(result, texture);
    return result;
}

void Descriptors::updateSingleImageSamplerDescriptors(
    const std::array<DescriptorSetHandle, kMaxFramesInFlight>& sets, TextureHandle texture)
{
    vk::DescriptorImageInfo imgInfo = makeDescriptorImageInfo(
        resources_->vulkanSampler(texture), resources_->vulkanImageView(texture),
        vk::ImageLayout::eShaderReadOnlyOptimal);
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        vk::WriteDescriptorSet write{
            .dstSet = descriptorSetTable_[static_cast<uint32_t>(sets[i])],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &imgInfo,
        };
        device_->device().updateDescriptorSets(write, {});
    }
}

DescriptorSetHandle Descriptors::createImageViewDescriptor(vk::DescriptorSetLayout layout,
                                                           vk::ImageView view, vk::Sampler sampler)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eCombinedImageSampler, 1},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, 1);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, 1);

    vk::DescriptorImageInfo imgInfo =
        makeDescriptorImageInfo(sampler, view, vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet write{
        .dstSet = *sets[0],
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &imgInfo,
    };
    device_->device().updateDescriptorSets(write, {});

    auto handle = registerDescriptorSet(*sets[0]);
    retainDescriptorSets(poolEntry, sets);
    return handle;
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::createPostProcessDescriptors(vk::DescriptorSetLayout layout, TextureHandle hdrTarget,
                                          TextureHandle bloomChain)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight * 2},
    }};
    // Allocation only; writes go through updatePostProcessDescriptors, which is
    // also the swapchain-resize recreation path.
    auto result = buildFrameSets(poolSizes, layout, {});
    updatePostProcessDescriptors(result, hdrTarget, bloomChain);
    return result;
}

void Descriptors::updatePostProcessDescriptors(
    const std::array<DescriptorSetHandle, kMaxFramesInFlight>& sets, TextureHandle hdrTarget,
    TextureHandle bloomChain)
{
    vk::ImageView bloomMip0 = resources_->vulkanBloomMipView(bloomChain, 0);
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        vk::DescriptorImageInfo hdrInfo = makeDescriptorImageInfo(
            resources_->vulkanSampler(hdrTarget), resources_->vulkanImageView(hdrTarget),
            vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo bloomInfo =
            makeDescriptorImageInfo(resources_->vulkanSampler(bloomChain), bloomMip0,
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
        std::array<vk::WriteDescriptorSet, 2> writes = {{
            vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(sets[i])],
                                   .dstBinding = bindingIndex(PostProcessBinding::HdrInput),
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &hdrInfo},
            vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(sets[i])],
                                   .dstBinding = bindingIndex(PostProcessBinding::BloomInput),
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &bloomInfo},
        }};
        device_->device().updateDescriptorSets(writes, {});
    }
}

std::array<DescriptorSetHandle, kMaxFramesInFlight> Descriptors::createTaaResolveDescriptors(
    vk::DescriptorSetLayout layout, TextureHandle currentColour, TextureHandle velocity,
    const std::array<TextureHandle, kMaxFramesInFlight>& history)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight * 3},
    }};
    auto result = buildFrameSets(poolSizes, layout, {});
    updateTaaResolveDescriptors(result, currentColour, velocity, history);
    return result;
}

void Descriptors::updateTaaResolveDescriptors(
    const std::array<DescriptorSetHandle, kMaxFramesInFlight>& sets, TextureHandle currentColour,
    TextureHandle velocity, const std::array<TextureHandle, kMaxFramesInFlight>& history)
{
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        // Parity i writes history[i], so it reads the opposite slot as "previous".
        const TextureHandle prevHistory = history[(kMaxFramesInFlight - 1) - i];

        vk::DescriptorImageInfo colourInfo = makeDescriptorImageInfo(
            resources_->vulkanSampler(currentColour), resources_->vulkanImageView(currentColour),
            vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo velocityInfo = makeDescriptorImageInfo(
            resources_->vulkanSampler(velocity), resources_->vulkanImageView(velocity),
            vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo historyInfo = makeDescriptorImageInfo(
            resources_->vulkanSampler(prevHistory), resources_->vulkanImageView(prevHistory),
            vk::ImageLayout::eShaderReadOnlyOptimal);

        const vk::DescriptorSet dst = descriptorSetTable_[static_cast<uint32_t>(sets[i])];
        std::array<vk::WriteDescriptorSet, 3> writes = {{
            vk::WriteDescriptorSet{.dstSet = dst,
                                   .dstBinding = bindingIndex(TaaBinding::CurrentColor),
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &colourInfo},
            vk::WriteDescriptorSet{.dstSet = dst,
                                   .dstBinding = bindingIndex(TaaBinding::Velocity),
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &velocityInfo},
            vk::WriteDescriptorSet{.dstSet = dst,
                                   .dstBinding = bindingIndex(TaaBinding::History),
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &historyInfo},
        }};
        device_->device().updateDescriptorSets(writes, {});
    }
}

vk::DescriptorSet Descriptors::vulkanDescriptorSet(DescriptorSetHandle handle) const noexcept
{
    return descriptorSetTable_[static_cast<uint32_t>(handle)];
}

} // namespace fire_engine
