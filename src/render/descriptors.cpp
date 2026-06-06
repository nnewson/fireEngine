#include <fire_engine/render/descriptors.hpp>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]]
TextureHandle materialTexture(const GeometryDescriptorInfo& geometry,
                              MaterialTextureSlot slot) noexcept
{
    return geometry.materialTextures[slotIndex(slot)];
}

// Thin wrappers around vk::WriteDescriptorSet{...} so the long write arrays in
// createObjectDescriptors / createShadowDescriptors / updateObjectGeometryTextures
// read as one line per binding instead of five.
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

ObjectDescriptorResult Descriptors::createObjectDescriptors(const ObjectDescriptorRequest& req)
{
    auto numGeometries = static_cast<uint32_t>(req.geometries.size());
    uint32_t totalSets = numGeometries * kMaxFramesInFlight;

    std::array<vk::DescriptorPoolSize, 3> poolSizes = {{
        // 4 uniform buffers per set: frame UBO, material UBO, skin UBO, morph UBO.
        {vk::DescriptorType::eUniformBuffer, totalSets * 4},
        // 10 combined samplers per set: baseColor, emissive, normal, mr,
        // occlusion, transmission, clearcoat ×3, thickness.
        {vk::DescriptorType::eCombinedImageSampler, totalSets * 10},
        {vk::DescriptorType::eStorageBuffer, totalSets},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, totalSets);

    ObjectDescriptorResult result;
    result.descSets.resize(numGeometries);

    for (uint32_t g = 0; g < numGeometries; ++g)
    {
        const auto& geo = req.geometries[g];
        auto sets = allocateDescriptorSets(*poolEntry.pool, pipeline_->descriptorSetLayout(),
                                           kMaxFramesInFlight);

        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            vk::DescriptorBufferInfo uboBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(req.uniformBufs[i]), sizeof(UniformBufferObject));
            vk::DescriptorBufferInfo matBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(geo.materialBufs[i]), sizeof(MaterialUBO));
            auto materialImageInfo = [&](MaterialTextureSlot slot)
            {
                const TextureHandle texture = materialTexture(geo, slot);
                return makeDescriptorImageInfo(resources_->vulkanSampler(texture),
                                               resources_->vulkanImageView(texture),
                                               vk::ImageLayout::eShaderReadOnlyOptimal);
            };

            vk::DescriptorImageInfo texInfo = materialImageInfo(MaterialTextureSlot::BaseColour);
            vk::DescriptorBufferInfo skinBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(geo.skinBufs[i]), sizeof(SkinUBO));
            vk::DescriptorBufferInfo morphUboBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(geo.morphUboBufs[i]), sizeof(MorphUBO));

            vk::DeviceSize ssboSize = geo.morphSsboSize > 0
                                          ? static_cast<vk::DeviceSize>(geo.morphSsboSize)
                                          : sizeof(float) * 4;
            vk::DescriptorBufferInfo morphSsboBufInfo =
                makeDescriptorBufferInfo(resources_->vulkanBuffer(geo.morphSsbo), ssboSize);

            vk::DescriptorImageInfo emissiveTexInfo =
                materialImageInfo(MaterialTextureSlot::Emissive);
            vk::DescriptorImageInfo normalTexInfo = materialImageInfo(MaterialTextureSlot::Normal);
            vk::DescriptorImageInfo mrTexInfo =
                materialImageInfo(MaterialTextureSlot::MetallicRoughness);
            vk::DescriptorImageInfo occTexInfo = materialImageInfo(MaterialTextureSlot::Occlusion);
            vk::DescriptorImageInfo transTexInfo =
                materialImageInfo(MaterialTextureSlot::Transmission);
            vk::DescriptorImageInfo ccTexInfo = materialImageInfo(MaterialTextureSlot::Clearcoat);
            vk::DescriptorImageInfo ccRoughTexInfo =
                materialImageInfo(MaterialTextureSlot::ClearcoatRoughness);
            vk::DescriptorImageInfo ccNormalTexInfo =
                materialImageInfo(MaterialTextureSlot::ClearcoatNormal);
            vk::DescriptorImageInfo thicknessTexInfo =
                materialImageInfo(MaterialTextureSlot::Thickness);
            const vk::DescriptorSet set = *sets[i];
            constexpr auto kUbo = vk::DescriptorType::eUniformBuffer;
            constexpr auto kSsbo = vk::DescriptorType::eStorageBuffer;
            constexpr auto kCis = vk::DescriptorType::eCombinedImageSampler;
            std::array<vk::WriteDescriptorSet, 15> writes = {{
                writeBuffer(set, bindingIndex(ForwardBinding::Frame), kUbo, uboBufInfo),
                writeBuffer(set, bindingIndex(ForwardBinding::Material), kUbo, matBufInfo),
                writeImage(set, bindingIndex(ForwardBinding::BaseColourTexture), kCis, texInfo),
                writeBuffer(set, bindingIndex(ForwardBinding::Skin), kUbo, skinBufInfo),
                writeBuffer(set, bindingIndex(ForwardBinding::Morph), kUbo, morphUboBufInfo),
                writeBuffer(set, bindingIndex(ForwardBinding::MorphTargets), kSsbo,
                            morphSsboBufInfo),
                writeImage(set, bindingIndex(ForwardBinding::EmissiveTexture), kCis,
                           emissiveTexInfo),
                writeImage(set, bindingIndex(ForwardBinding::NormalTexture), kCis, normalTexInfo),
                writeImage(set, bindingIndex(ForwardBinding::MetallicRoughnessTexture), kCis,
                           mrTexInfo),
                writeImage(set, bindingIndex(ForwardBinding::OcclusionTexture), kCis, occTexInfo),
                writeImage(set, bindingIndex(ForwardBinding::TransmissionTexture), kCis,
                           transTexInfo),
                writeImage(set, bindingIndex(ForwardBinding::ClearcoatTexture), kCis, ccTexInfo),
                writeImage(set, bindingIndex(ForwardBinding::ClearcoatRoughnessTexture), kCis,
                           ccRoughTexInfo),
                writeImage(set, bindingIndex(ForwardBinding::ClearcoatNormalTexture), kCis,
                           ccNormalTexInfo),
                writeImage(set, bindingIndex(ForwardBinding::ThicknessTexture), kCis,
                           thicknessTexInfo),
            }};
            device_->device().updateDescriptorSets(writes, {});

            result.descSets[g][i] = registerDescriptorSet(*sets[i]);
        }

        retainDescriptorSets(poolEntry, sets);
    }

    return result;
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
    auto sampledColourImage = [this](TextureHandle handle)
    {
        return makeDescriptorImageInfo({}, resources_->vulkanImageView(handle),
                                       vk::ImageLayout::eShaderReadOnlyOptimal);
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
    const vk::DescriptorImageInfo shadowDebugImageInfo = sampledColourImage(req.shadowDebugImage);
    const vk::DescriptorImageInfo shadowCompareSamplerInfo =
        plainSampler(resources_->vulkanSampler(req.shadowMap));
    const vk::DescriptorImageInfo shadowDebugSamplerInfo =
        plainSampler(resources_->vulkanShadowDebugSampler());
    const vk::DescriptorImageInfo irradianceInfo = combinedSampler(req.irradianceMap);
    const vk::DescriptorImageInfo prefilteredInfo = combinedSampler(req.prefilteredMap);
    const vk::DescriptorImageInfo brdfLutInfo = combinedSampler(req.brdfLut);
    const vk::DescriptorImageInfo sceneColorInfo = combinedSampler(req.sceneColor);

    constexpr auto kUbo = vk::DescriptorType::eUniformBuffer;
    constexpr auto kCis = vk::DescriptorType::eCombinedImageSampler;
    constexpr auto kSi = vk::DescriptorType::eSampledImage;
    constexpr auto kSamp = vk::DescriptorType::eSampler;
    std::array<vk::WriteDescriptorSet, 13> writes = {{
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
    }};
    device_->device().updateDescriptorSets(writes, {});
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::createGlobalDescriptors(const GlobalDescriptorRequest& req)
{
    // One global set per frame-in-flight. Pool sizes are exact per the
    // ForwardGlobalBinding enum: 1 UBO + 4 CIS (IBL ×3 + sceneColor) + 6 SI
    // (shadow maps ×5 + debug colour) + 2 plain samplers (compare + debug).
    std::array<vk::DescriptorPoolSize, 4> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, kMaxFramesInFlight},
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight * 4},
        {vk::DescriptorType::eSampledImage, kMaxFramesInFlight * 6},
        {vk::DescriptorType::eSampler, kMaxFramesInFlight * 2},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, kMaxFramesInFlight);
    auto sets = allocateDescriptorSets(*poolEntry.pool, pipeline_->globalDescriptorSetLayout(),
                                       kMaxFramesInFlight);

    std::array<DescriptorSetHandle, kMaxFramesInFlight> result;
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        writeGlobalBindings(*sets[i], req, i);
        result[i] = registerDescriptorSet(*sets[i]);
    }
    retainDescriptorSets(poolEntry, sets);
    return result;
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

void Descriptors::updateObjectGeometryTextures(DescriptorSetHandle set,
                                               const GeometryDescriptorInfo& geo)
{
    auto materialImageInfo = [&](MaterialTextureSlot slot)
    {
        const TextureHandle texture = materialTexture(geo, slot);
        return makeDescriptorImageInfo(resources_->vulkanSampler(texture),
                                       resources_->vulkanImageView(texture),
                                       vk::ImageLayout::eShaderReadOnlyOptimal);
    };

    vk::DescriptorImageInfo texInfo = materialImageInfo(MaterialTextureSlot::BaseColour);
    vk::DescriptorImageInfo emissiveTexInfo = materialImageInfo(MaterialTextureSlot::Emissive);
    vk::DescriptorImageInfo normalTexInfo = materialImageInfo(MaterialTextureSlot::Normal);
    vk::DescriptorImageInfo mrTexInfo = materialImageInfo(MaterialTextureSlot::MetallicRoughness);
    vk::DescriptorImageInfo occTexInfo = materialImageInfo(MaterialTextureSlot::Occlusion);
    vk::DescriptorImageInfo transTexInfo = materialImageInfo(MaterialTextureSlot::Transmission);
    vk::DescriptorImageInfo ccTexInfo = materialImageInfo(MaterialTextureSlot::Clearcoat);
    vk::DescriptorImageInfo ccRoughTexInfo =
        materialImageInfo(MaterialTextureSlot::ClearcoatRoughness);
    vk::DescriptorImageInfo ccNormalTexInfo =
        materialImageInfo(MaterialTextureSlot::ClearcoatNormal);
    vk::DescriptorImageInfo thicknessTexInfo = materialImageInfo(MaterialTextureSlot::Thickness);

    const vk::DescriptorSet dstSet = descriptorSetTable_[static_cast<uint32_t>(set)];
    constexpr auto kCis = vk::DescriptorType::eCombinedImageSampler;
    std::array<vk::WriteDescriptorSet, 10> writes = {{
        writeImage(dstSet, bindingIndex(ForwardBinding::BaseColourTexture), kCis, texInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::EmissiveTexture), kCis, emissiveTexInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::NormalTexture), kCis, normalTexInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::MetallicRoughnessTexture), kCis, mrTexInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::OcclusionTexture), kCis, occTexInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::TransmissionTexture), kCis, transTexInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::ClearcoatTexture), kCis, ccTexInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::ClearcoatRoughnessTexture), kCis,
                   ccRoughTexInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::ClearcoatNormalTexture), kCis,
                   ccNormalTexInfo),
        writeImage(dstSet, bindingIndex(ForwardBinding::ThicknessTexture), kCis, thicknessTexInfo),
    }};
    device_->device().updateDescriptorSets(writes, {});
}

ShadowDescriptorResult Descriptors::createShadowDescriptors(const ShadowDescriptorRequest& req)
{
    auto numGeometries = static_cast<uint32_t>(req.geometries.size());
    uint32_t totalSets = numGeometries * kMaxFramesInFlight;

    std::array<vk::DescriptorPoolSize, 4> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, totalSets * 3},
        {vk::DescriptorType::eStorageBuffer, totalSets},
        {vk::DescriptorType::eSampledImage, totalSets},
        {vk::DescriptorType::eSampler, totalSets},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, totalSets);

    ShadowDescriptorResult result;
    result.descSets.resize(numGeometries);

    for (uint32_t g = 0; g < numGeometries; ++g)
    {
        const auto& geo = req.geometries[g];
        auto sets = allocateDescriptorSets(*poolEntry.pool, shadowDescLayout_, kMaxFramesInFlight);

        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            vk::DescriptorBufferInfo shadowUboInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(geo.shadowUboBufs[i]), sizeof(ShadowUBO));
            vk::DescriptorBufferInfo skinBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(geo.skinBufs[i]), sizeof(SkinUBO));
            vk::DescriptorBufferInfo morphUboBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(geo.morphUboBufs[i]), sizeof(MorphUBO));
            vk::DeviceSize ssboSize = geo.morphSsboSize > 0
                                          ? static_cast<vk::DeviceSize>(geo.morphSsboSize)
                                          : sizeof(float) * 4;
            vk::DescriptorBufferInfo morphSsboInfo =
                makeDescriptorBufferInfo(resources_->vulkanBuffer(geo.morphSsbo), ssboSize);
            vk::DescriptorImageInfo selfShadowFirstInfo = makeDescriptorImageInfo(
                {}, resources_->vulkanImageView(resources_->sharedTextures().selfShadowFirstMap),
                vk::ImageLayout::eDepthStencilReadOnlyOptimal);
            vk::DescriptorImageInfo selfShadowSamplerInfo = makeDescriptorImageInfo(
                resources_->vulkanShadowDebugSampler(), {}, vk::ImageLayout::eUndefined);

            const vk::DescriptorSet set = *sets[i];
            constexpr auto kUbo = vk::DescriptorType::eUniformBuffer;
            constexpr auto kSsbo = vk::DescriptorType::eStorageBuffer;
            constexpr auto kSi = vk::DescriptorType::eSampledImage;
            constexpr auto kSamp = vk::DescriptorType::eSampler;
            std::array<vk::WriteDescriptorSet, 6> writes = {{
                writeBuffer(set, bindingIndex(ShadowBinding::Shadow), kUbo, shadowUboInfo),
                writeBuffer(set, bindingIndex(ShadowBinding::Skin), kUbo, skinBufInfo),
                writeBuffer(set, bindingIndex(ShadowBinding::Morph), kUbo, morphUboBufInfo),
                writeBuffer(set, bindingIndex(ShadowBinding::MorphTargets), kSsbo, morphSsboInfo),
                writeImage(set, bindingIndex(ShadowBinding::SelfShadowFirstMap), kSi,
                           selfShadowFirstInfo),
                writeImage(set, bindingIndex(ShadowBinding::SelfShadowDepthSampler), kSamp,
                           selfShadowSamplerInfo),
            }};
            device_->device().updateDescriptorSets(writes, {});

            result.descSets[g][i] = registerDescriptorSet(*sets[i]);
        }

        retainDescriptorSets(poolEntry, sets);
    }

    return result;
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::createSingleUboDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& ubo,
                                        vk::DeviceSize uboSize)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, kMaxFramesInFlight},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, kMaxFramesInFlight);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, kMaxFramesInFlight);

    std::array<DescriptorSetHandle, kMaxFramesInFlight> result{};
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        vk::DescriptorBufferInfo bufInfo =
            makeDescriptorBufferInfo(resources_->vulkanBuffer(ubo.buffers[i]), uboSize);
        vk::WriteDescriptorSet write{
            .dstSet = *sets[i],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &bufInfo,
        };
        device_->device().updateDescriptorSets(write, {});

        result[i] = registerDescriptorSet(*sets[i]);
    }

    retainDescriptorSets(poolEntry, sets);
    return result;
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
    auto& poolEntry = createDescriptorPool(poolSizes, kMaxFramesInFlight);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, kMaxFramesInFlight);

    std::array<DescriptorSetHandle, kMaxFramesInFlight> result{};
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        vk::DescriptorBufferInfo bufInfo =
            makeDescriptorBufferInfo(resources_->vulkanBuffer(ubo.buffers[i]), uboSize);
        vk::DescriptorImageInfo texInfo = makeDescriptorImageInfo(
            resources_->vulkanSampler(texture), resources_->vulkanImageView(texture),
            vk::ImageLayout::eShaderReadOnlyOptimal);
        std::array<vk::WriteDescriptorSet, 2> writes = {{
            vk::WriteDescriptorSet{.dstSet = *sets[i],
                                   .dstBinding = bindingIndex(SkyboxBinding::Skybox),
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eUniformBuffer,
                                   .pBufferInfo = &bufInfo},
            vk::WriteDescriptorSet{.dstSet = *sets[i],
                                   .dstBinding = bindingIndex(SkyboxBinding::Cubemap),
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &texInfo},
        }};
        device_->device().updateDescriptorSets(writes, {});
        result[i] = registerDescriptorSet(*sets[i]);
    }

    retainDescriptorSets(poolEntry, sets);
    return result;
}

std::array<DescriptorSetHandle, kMaxFramesInFlight> Descriptors::createSkyboxDescriptors(
    vk::DescriptorSetLayout layout, const MappedBufferSet& skyboxUbo, vk::DeviceSize skyboxUboSize,
    TextureHandle texture, const MappedBufferSet& lightUbo, vk::DeviceSize lightUboSize)
{
    std::array<vk::DescriptorPoolSize, 2> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, kMaxFramesInFlight * 2},
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, kMaxFramesInFlight);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, kMaxFramesInFlight);

    std::array<DescriptorSetHandle, kMaxFramesInFlight> result{};
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        vk::DescriptorBufferInfo skyboxBufInfo =
            makeDescriptorBufferInfo(resources_->vulkanBuffer(skyboxUbo.buffers[i]), skyboxUboSize);
        vk::DescriptorBufferInfo lightBufInfo =
            makeDescriptorBufferInfo(resources_->vulkanBuffer(lightUbo.buffers[i]), lightUboSize);
        vk::DescriptorImageInfo texInfo = makeDescriptorImageInfo(
            resources_->vulkanSampler(texture), resources_->vulkanImageView(texture),
            vk::ImageLayout::eShaderReadOnlyOptimal);
        std::array<vk::WriteDescriptorSet, 3> writes = {{
            vk::WriteDescriptorSet{.dstSet = *sets[i],
                                   .dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eUniformBuffer,
                                   .pBufferInfo = &skyboxBufInfo},
            vk::WriteDescriptorSet{.dstSet = *sets[i],
                                   .dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &texInfo},
            vk::WriteDescriptorSet{.dstSet = *sets[i],
                                   .dstBinding = bindingIndex(SkyboxBinding::Light),
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eUniformBuffer,
                                   .pBufferInfo = &lightBufInfo},
        }};
        device_->device().updateDescriptorSets(writes, {});
        result[i] = registerDescriptorSet(*sets[i]);
    }

    retainDescriptorSets(poolEntry, sets);
    return result;
}

std::array<DescriptorSetHandle, kMaxFramesInFlight>
Descriptors::createSingleImageSamplerDescriptors(vk::DescriptorSetLayout layout,
                                                 TextureHandle texture)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eCombinedImageSampler, kMaxFramesInFlight},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, kMaxFramesInFlight);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, kMaxFramesInFlight);

    std::array<DescriptorSetHandle, kMaxFramesInFlight> result{};
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        result[i] = registerDescriptorSet(*sets[i]);
    }

    retainDescriptorSets(poolEntry, sets);
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
    auto& poolEntry = createDescriptorPool(poolSizes, kMaxFramesInFlight);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, kMaxFramesInFlight);

    std::array<DescriptorSetHandle, kMaxFramesInFlight> result{};
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        result[i] = registerDescriptorSet(*sets[i]);
    }
    retainDescriptorSets(poolEntry, sets);

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

vk::DescriptorSet Descriptors::vulkanDescriptorSet(DescriptorSetHandle handle) const noexcept
{
    return descriptorSetTable_[static_cast<uint32_t>(handle)];
}

} // namespace fire_engine
