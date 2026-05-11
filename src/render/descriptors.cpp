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
    uint32_t totalSets = numGeometries * MAX_FRAMES_IN_FLIGHT;

    std::array<vk::DescriptorPoolSize, 5> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, totalSets * 5},
        // 14 combined samplers per set: baseColor, emissive, normal, mr,
        // occlusion, irradiance, prefiltered, brdfLut, transmission,
        // clearcoat, clearcoatRoughness, clearcoatNormal, sceneColor, thickness.
        {vk::DescriptorType::eCombinedImageSampler, totalSets * 14},
        // 6 sampled images per set: csm, world csm, self shadow, spot, point, debug colour.
        {vk::DescriptorType::eSampledImage, totalSets * 6},
        // 2 standalone samplers per set: compare sampler + non-compare debug sampler.
        {vk::DescriptorType::eSampler, totalSets * 2},
        {vk::DescriptorType::eStorageBuffer, totalSets},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, totalSets);

    ObjectDescriptorResult result;
    result.descSets.resize(numGeometries);

    for (uint32_t g = 0; g < numGeometries; ++g)
    {
        const auto& geo = req.geometries[g];
        auto sets = allocateDescriptorSets(*poolEntry.pool, pipeline_->descriptorSetLayout(),
                                           MAX_FRAMES_IN_FLIGHT);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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
            vk::DescriptorImageInfo sceneColorInfo =
                makeDescriptorImageInfo(resources_->vulkanSampler(req.sceneColor),
                                        resources_->vulkanImageView(req.sceneColor),
                                        vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::DescriptorImageInfo thicknessTexInfo =
                materialImageInfo(MaterialTextureSlot::Thickness);
            vk::DescriptorBufferInfo lightBufInfo = makeDescriptorBufferInfo(
                resources_->vulkanBuffer(req.lightBufs[i]), sizeof(LightUBO));
            // Shadow images use plain sampledImage descriptors; the comparison
            // sampler is shared across CSM/spot/point via its own descriptor
            // binding (Apple's per-stage sampler limit is 16).
            vk::DescriptorImageInfo shadowTexInfo =
                makeDescriptorImageInfo({}, resources_->vulkanImageView(req.shadowMap),
                                        vk::ImageLayout::eDepthStencilReadOnlyOptimal);
            vk::DescriptorImageInfo worldShadowTexInfo =
                makeDescriptorImageInfo({}, resources_->vulkanImageView(req.worldShadowMap),
                                        vk::ImageLayout::eDepthStencilReadOnlyOptimal);
            vk::DescriptorImageInfo selfShadowTexInfo =
                makeDescriptorImageInfo({}, resources_->vulkanImageView(req.selfShadowMap),
                                        vk::ImageLayout::eDepthStencilReadOnlyOptimal);
            vk::DescriptorImageInfo spotShadowImageInfo =
                makeDescriptorImageInfo({}, resources_->vulkanImageView(req.spotShadowMap),
                                        vk::ImageLayout::eDepthStencilReadOnlyOptimal);
            vk::DescriptorImageInfo pointShadowImageInfo =
                makeDescriptorImageInfo({}, resources_->vulkanImageView(req.pointShadowMap),
                                        vk::ImageLayout::eDepthStencilReadOnlyOptimal);
            vk::DescriptorImageInfo shadowCompareSamplerInfo = makeDescriptorImageInfo(
                resources_->vulkanSampler(req.shadowMap), {}, vk::ImageLayout::eUndefined);
            vk::DescriptorImageInfo shadowDebugSamplerInfo = makeDescriptorImageInfo(
                resources_->vulkanShadowDebugSampler(), {}, vk::ImageLayout::eUndefined);
            vk::DescriptorImageInfo shadowDebugImageInfo =
                makeDescriptorImageInfo({}, resources_->vulkanImageView(req.shadowDebugImage),
                                        vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::DescriptorImageInfo irradianceTexInfo =
                makeDescriptorImageInfo(resources_->vulkanSampler(req.irradianceMap),
                                        resources_->vulkanImageView(req.irradianceMap),
                                        vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::DescriptorImageInfo prefilteredTexInfo =
                makeDescriptorImageInfo(resources_->vulkanSampler(req.prefilteredMap),
                                        resources_->vulkanImageView(req.prefilteredMap),
                                        vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::DescriptorImageInfo brdfLutTexInfo = makeDescriptorImageInfo(
                resources_->vulkanSampler(req.brdfLut), resources_->vulkanImageView(req.brdfLut),
                vk::ImageLayout::eShaderReadOnlyOptimal);
            std::array<vk::WriteDescriptorSet, 28> writes = {{
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::Frame),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &uboBufInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::Material),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &matBufInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ForwardBinding::BaseColourTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &texInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::Skin),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &skinBufInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::Morph),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &morphUboBufInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::MorphTargets),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageBuffer,
                                       .pBufferInfo = &morphSsboBufInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::EmissiveTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &emissiveTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::NormalTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &normalTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ForwardBinding::MetallicRoughnessTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &mrTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::OcclusionTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &occTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::ShadowMap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage,
                                       .pImageInfo = &shadowTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::Light),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &lightBufInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::IrradianceMap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &irradianceTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::PrefilteredMap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &prefilteredTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::BrdfLut),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &brdfLutTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ForwardBinding::ShadowCompareSampler),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampler,
                                       .pImageInfo = &shadowCompareSamplerInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ForwardBinding::TransmissionTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &transTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::ClearcoatTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &ccTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ForwardBinding::ClearcoatRoughnessTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &ccRoughTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ForwardBinding::ClearcoatNormalTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &ccNormalTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::SceneColour),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &sceneColorInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::ThicknessTexture),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &thicknessTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::SpotShadowMap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage,
                                       .pImageInfo = &spotShadowImageInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::PointShadowMap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage,
                                       .pImageInfo = &pointShadowImageInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ForwardBinding::ShadowDebugSampler),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampler,
                                       .pImageInfo = &shadowDebugSamplerInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::ShadowDebugImage),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage,
                                       .pImageInfo = &shadowDebugImageInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::WorldShadowMap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage,
                                       .pImageInfo = &worldShadowTexInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ForwardBinding::SelfShadowMap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage,
                                       .pImageInfo = &selfShadowTexInfo},
            }};
            device_->device().updateDescriptorSets(writes, {});

            result.descSets[g][i] = registerDescriptorSet(*sets[i]);
        }

        retainDescriptorSets(poolEntry, sets);
    }

    return result;
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

    std::array<vk::WriteDescriptorSet, 10> writes = {{
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::BaseColourTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &texInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::EmissiveTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &emissiveTexInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::NormalTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &normalTexInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::MetallicRoughnessTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &mrTexInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::OcclusionTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &occTexInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::TransmissionTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &transTexInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::ClearcoatTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &ccTexInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding =
                                   bindingIndex(ForwardBinding::ClearcoatRoughnessTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &ccRoughTexInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::ClearcoatNormalTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &ccNormalTexInfo},
        vk::WriteDescriptorSet{.dstSet = descriptorSetTable_[static_cast<uint32_t>(set)],
                               .dstBinding = bindingIndex(ForwardBinding::ThicknessTexture),
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &thicknessTexInfo},
    }};
    device_->device().updateDescriptorSets(writes, {});
}

ShadowDescriptorResult Descriptors::createShadowDescriptors(const ShadowDescriptorRequest& req)
{
    auto numGeometries = static_cast<uint32_t>(req.geometries.size());
    uint32_t totalSets = numGeometries * MAX_FRAMES_IN_FLIGHT;

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
        auto sets =
            allocateDescriptorSets(*poolEntry.pool, shadowDescLayout_, MAX_FRAMES_IN_FLIGHT);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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
                {}, resources_->vulkanImageView(resources_->selfShadowFirstMap()),
                vk::ImageLayout::eDepthStencilReadOnlyOptimal);
            vk::DescriptorImageInfo selfShadowSamplerInfo = makeDescriptorImageInfo(
                resources_->vulkanShadowDebugSampler(), {}, vk::ImageLayout::eUndefined);

            std::array<vk::WriteDescriptorSet, 6> writes = {{
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ShadowBinding::Shadow),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &shadowUboInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ShadowBinding::Skin),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &skinBufInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ShadowBinding::Morph),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &morphUboBufInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding = bindingIndex(ShadowBinding::MorphTargets),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageBuffer,
                                       .pBufferInfo = &morphSsboInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ShadowBinding::SelfShadowFirstMap),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage,
                                       .pImageInfo = &selfShadowFirstInfo},
                vk::WriteDescriptorSet{.dstSet = *sets[i],
                                       .dstBinding =
                                           bindingIndex(ShadowBinding::SelfShadowDepthSampler),
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampler,
                                       .pImageInfo = &selfShadowSamplerInfo},
            }};
            device_->device().updateDescriptorSets(writes, {});

            result.descSets[g][i] = registerDescriptorSet(*sets[i]);
        }

        retainDescriptorSets(poolEntry, sets);
    }

    return result;
}

std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
Descriptors::createSingleUboDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& ubo,
                                        vk::DeviceSize uboSize)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, MAX_FRAMES_IN_FLIGHT);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, MAX_FRAMES_IN_FLIGHT);

    std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT> result{};
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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

std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
Descriptors::createUboImageSamplerDescriptors(vk::DescriptorSetLayout layout,
                                              const MappedBufferSet& ubo, vk::DeviceSize uboSize,
                                              TextureHandle texture)
{
    std::array<vk::DescriptorPoolSize, 2> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT},
        {vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, MAX_FRAMES_IN_FLIGHT);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, MAX_FRAMES_IN_FLIGHT);

    std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT> result{};
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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

std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT> Descriptors::createSkyboxDescriptors(
    vk::DescriptorSetLayout layout, const MappedBufferSet& skyboxUbo, vk::DeviceSize skyboxUboSize,
    TextureHandle texture, const MappedBufferSet& lightUbo, vk::DeviceSize lightUboSize)
{
    std::array<vk::DescriptorPoolSize, 2> poolSizes = {{
        {vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT * 2},
        {vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, MAX_FRAMES_IN_FLIGHT);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, MAX_FRAMES_IN_FLIGHT);

    std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT> result{};
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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

std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
Descriptors::createSingleImageSamplerDescriptors(vk::DescriptorSetLayout layout,
                                                 TextureHandle texture)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, MAX_FRAMES_IN_FLIGHT);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, MAX_FRAMES_IN_FLIGHT);

    std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT> result{};
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        result[i] = registerDescriptorSet(*sets[i]);
    }

    retainDescriptorSets(poolEntry, sets);
    updateSingleImageSamplerDescriptors(result, texture);
    return result;
}

void Descriptors::updateSingleImageSamplerDescriptors(
    const std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>& sets, TextureHandle texture)
{
    vk::DescriptorImageInfo imgInfo = makeDescriptorImageInfo(
        resources_->vulkanSampler(texture), resources_->vulkanImageView(texture),
        vk::ImageLayout::eShaderReadOnlyOptimal);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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

std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
Descriptors::createPostProcessDescriptors(vk::DescriptorSetLayout layout, TextureHandle hdrTarget,
                                          TextureHandle bloomChain)
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {{
        {vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT * 2},
    }};
    auto& poolEntry = createDescriptorPool(poolSizes, MAX_FRAMES_IN_FLIGHT);
    auto sets = allocateDescriptorSets(*poolEntry.pool, layout, MAX_FRAMES_IN_FLIGHT);

    std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT> result{};
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        result[i] = registerDescriptorSet(*sets[i]);
    }
    retainDescriptorSets(poolEntry, sets);

    updatePostProcessDescriptors(result, hdrTarget, bloomChain);
    return result;
}

void Descriptors::updatePostProcessDescriptors(
    const std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>& sets, TextureHandle hdrTarget,
    TextureHandle bloomChain)
{
    vk::ImageView bloomMip0 = resources_->vulkanBloomMipView(bloomChain, 0);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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
