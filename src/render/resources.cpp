#include <fire_engine/render/resources.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fire_engine/graphics/image.hpp>
#include <fire_engine/graphics/ktx_image.hpp>
#include <fire_engine/graphics/material_binding.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

Resources::Resources(const Device& device, const Pipeline& pipeline)
    : device_(&device),
      descriptors_(device, pipeline, *this)
{
    // Bindless materials set 2: one update-after-bind pool + one set, allocated from
    // the forward pipeline's bindless layout. registerBindlessTexture writes 2D
    // material textures into binding 0; registerMaterial fills the materials SSBO
    // bound at binding 1.
    if (pipeline.hasBindlessDescriptorSetLayout())
    {
        const std::array<vk::DescriptorPoolSize, 2> bindlessSizes{{
            {vk::DescriptorType::eCombinedImageSampler, kMaxBindlessTextures},
            {vk::DescriptorType::eStorageBuffer, 1},
        }};
        const vk::DescriptorPoolCreateInfo bindlessPoolCi{
            .flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind |
                     vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1,
            .poolSizeCount = static_cast<uint32_t>(bindlessSizes.size()),
            .pPoolSizes = bindlessSizes.data(),
        };
        bindlessPool_ = vk::raii::DescriptorPool(device.device(), bindlessPoolCi);

        const vk::DescriptorSetLayout layout = pipeline.bindlessDescriptorSetLayout();
        const vk::DescriptorSetAllocateInfo ai{
            .descriptorPool = *bindlessPool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        bindlessSet_ = std::move(device.device().allocateDescriptorSets(ai).front());

        // Global materials[] SSBO: one persistently-mapped host-visible buffer,
        // written per material on first registerMaterial. Bound once here.
        const std::size_t materialBytes = kMaxMaterials * sizeof(MaterialUBO);
        auto [matBuf, matMem] = device.createBuffer(
            static_cast<vk::DeviceSize>(materialBytes), vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        materialMapped_ = matMem.mapMemory(0, static_cast<vk::DeviceSize>(materialBytes));
        std::memset(materialMapped_, 0, materialBytes);
        materialBuffer_ = storeBuffer(std::move(matBuf), std::move(matMem));

        const vk::DescriptorBufferInfo matInfo{vulkanBuffer(materialBuffer_), 0,
                                               static_cast<vk::DeviceSize>(materialBytes)};
        const vk::WriteDescriptorSet matWrite{
            .dstSet = *bindlessSet_,
            .dstBinding = bindingIndex(BindlessBinding::Materials),
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &matInfo,
        };
        device.device().updateDescriptorSets(matWrite, {});
    }

    vk::CommandPoolCreateInfo poolCi{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = device.graphicsFamily(),
    };
    cmdPool_ = vk::raii::CommandPool(device.device(), poolCi);

    shadowDebugSampler_ = vk::raii::Sampler(
        device.device(), vk::SamplerCreateInfo{
                             .magFilter = vk::Filter::eNearest,
                             .minFilter = vk::Filter::eNearest,
                             .mipmapMode = vk::SamplerMipmapMode::eNearest,
                             .addressModeU = vk::SamplerAddressMode::eClampToBorder,
                             .addressModeV = vk::SamplerAddressMode::eClampToBorder,
                             .addressModeW = vk::SamplerAddressMode::eClampToBorder,
                             .mipLodBias = 0.0f,
                             .anisotropyEnable = vk::False,
                             .maxAnisotropy = 1.0f,
                             .compareEnable = vk::False,
                             .compareOp = vk::CompareOp::eAlways,
                             .minLod = 0.0f,
                             .maxLod = 1.0f,
                             .borderColor = vk::BorderColor::eFloatOpaqueWhite,
                             .unnormalizedCoordinates = vk::False,
                         });
}

// --- Buffer helpers ---

BufferHandle Resources::storeBuffer(vk::raii::Buffer buf, vk::raii::DeviceMemory mem)
{
    auto id = static_cast<uint32_t>(buffers_.size());
    buffers_.push_back({std::move(buf), std::move(mem)});
    return BufferHandle{id};
}

// --- Buffer creation ---

BufferHandle Resources::createVertexBuffer(std::span<const Vertex> vertices)
{
    vk::DeviceSize size = vertices.size_bytes();
    auto [buf, mem] = device_->createBuffer(size, vk::BufferUsageFlagBits::eVertexBuffer,
                                            vk::MemoryPropertyFlagBits::eHostVisible |
                                                vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data = mem.mapMemory(0, size);
    std::memcpy(data, vertices.data(), size);
    mem.unmapMemory();
    return storeBuffer(std::move(buf), std::move(mem));
}

BufferHandle Resources::createStorageBuffer(std::size_t size, const void* initialData)
{
    // eShaderDeviceAddress: the soft-body solver chains its buffers via 64-bit
    // GPU pointers (bufferDeviceAddress) instead of descriptor sets.
    auto [buf, mem] = device_->createBuffer(
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (initialData != nullptr)
    {
        void* data = mem.mapMemory(0, size);
        std::memcpy(data, initialData, size);
        mem.unmapMemory();
    }
    return storeBuffer(std::move(buf), std::move(mem));
}

BufferHandle Resources::createStorageVertexBuffer(std::span<const Vertex> vertices)
{
    vk::DeviceSize size = vertices.size_bytes();
    auto [buf, mem] = device_->createBuffer(
        size,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data = mem.mapMemory(0, size);
    std::memcpy(data, vertices.data(), size);
    mem.unmapMemory();
    return storeBuffer(std::move(buf), std::move(mem));
}

Resources::MappedBufferSet Resources::createMappedDeviceAddressBuffers(std::size_t size)
{
    // Per-frame, persistently-mapped storage buffers with a device address — for
    // the soft-body solver's per-frame collider buffer (written each frame from
    // the CPU, addressed by the compute shaders).
    MappedBufferSet result;
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        auto [buf, mem] = device_->createBuffer(
            static_cast<vk::DeviceSize>(size),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        result.mapped[i] = mem.mapMemory(0, static_cast<vk::DeviceSize>(size));
        result.buffers[i] = storeBuffer(std::move(buf), std::move(mem));
    }
    return result;
}

vk::DeviceAddress Resources::bufferAddress(BufferHandle handle) const noexcept
{
    return device_->device().getBufferAddress(
        vk::BufferDeviceAddressInfo{.buffer = vulkanBuffer(handle)});
}

BufferHandle Resources::createIndexBuffer(std::span<const uint16_t> indices)
{
    vk::DeviceSize size = indices.size_bytes();
    auto [buf, mem] = device_->createBuffer(size, vk::BufferUsageFlagBits::eIndexBuffer,
                                            vk::MemoryPropertyFlagBits::eHostVisible |
                                                vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data = mem.mapMemory(0, size);
    std::memcpy(data, indices.data(), size);
    mem.unmapMemory();
    return storeBuffer(std::move(buf), std::move(mem));
}

BufferHandle Resources::createIndexBuffer(std::span<const uint32_t> indices)
{
    vk::DeviceSize size = indices.size_bytes();
    auto [buf, mem] = device_->createBuffer(size, vk::BufferUsageFlagBits::eIndexBuffer,
                                            vk::MemoryPropertyFlagBits::eHostVisible |
                                                vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data = mem.mapMemory(0, size);
    std::memcpy(data, indices.data(), size);
    mem.unmapMemory();
    return storeBuffer(std::move(buf), std::move(mem));
}

uint32_t Resources::allocateObjectId() noexcept
{
    return nextObjectId_++;
}

// --- Texture creation ---

namespace
{

vk::SamplerAddressMode toVkAddressMode(WrapMode mode)
{
    switch (mode)
    {
    case WrapMode::MirroredRepeat:
        return vk::SamplerAddressMode::eMirroredRepeat;
    case WrapMode::ClampToEdge:
        return vk::SamplerAddressMode::eClampToEdge;
    default:
        return vk::SamplerAddressMode::eRepeat;
    }
}

vk::Filter toVkFilter(FilterMode mode)
{
    switch (mode)
    {
    case FilterMode::Nearest:
        return vk::Filter::eNearest;
    default:
        return vk::Filter::eLinear;
    }
}

vk::Format toVkTextureFormat(TextureEncoding encoding)
{
    return encoding == TextureEncoding::Srgb ? vk::Format::eR8G8B8A8Srgb
                                             : vk::Format::eR8G8B8A8Unorm;
}

std::size_t fallbackTextureIndex(Resources::FallbackTextureKind kind)
{
    return static_cast<std::size_t>(kind);
}

[[nodiscard]] bool supportsSampledTextureFormat(const Device& device, vk::Format format)
{
    vk::FormatProperties properties = device.physicalDevice().getFormatProperties(format);
    constexpr vk::FormatFeatureFlags required =
        vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
    return (properties.optimalTilingFeatures & required) == required;
}

[[nodiscard]] ktx_transcode_fmt_e chooseBasisTranscodeFormat(const Device& device,
                                                             TextureEncoding encoding)
{
    const vk::Format astcFormat = encoding == TextureEncoding::Srgb
                                      ? vk::Format::eAstc4x4SrgbBlock
                                      : vk::Format::eAstc4x4UnormBlock;
    if (supportsSampledTextureFormat(device, astcFormat))
    {
        return KTX_TTF_ASTC_4x4_RGBA;
    }

    const vk::Format bc7Format =
        encoding == TextureEncoding::Srgb ? vk::Format::eBc7SrgbBlock : vk::Format::eBc7UnormBlock;
    if (supportsSampledTextureFormat(device, bc7Format))
    {
        return KTX_TTF_BC7_RGBA;
    }

    const vk::Format etc2Format = encoding == TextureEncoding::Srgb
                                      ? vk::Format::eEtc2R8G8B8A8SrgbBlock
                                      : vk::Format::eEtc2R8G8B8A8UnormBlock;
    if (supportsSampledTextureFormat(device, etc2Format))
    {
        return KTX_TTF_ETC2_RGBA;
    }

    const vk::Format rgbaFormat =
        encoding == TextureEncoding::Srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
    if (supportsSampledTextureFormat(device, rgbaFormat))
    {
        return KTX_TTF_RGBA32;
    }

    throw std::runtime_error("No supported Vulkan transcode target for Basis-compressed KTX image");
}

[[nodiscard]] std::string ktxFailureDetail(KTX_error_code error)
{
    const char* reason = ktxErrorString(error);
    if (reason == nullptr || reason[0] == '\0')
    {
        return "unknown libktx error";
    }

    return reason;
}

[[nodiscard]] vk::ImageSubresourceRange
makeImageSubresourceRange(vk::ImageAspectFlags aspectMask, uint32_t baseMipLevel,
                          uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount)
{
    return vk::ImageSubresourceRange{
        .aspectMask = aspectMask,
        .baseMipLevel = baseMipLevel,
        .levelCount = levelCount,
        .baseArrayLayer = baseArrayLayer,
        .layerCount = layerCount,
    };
}

[[nodiscard]] vk::ImageSubresourceLayers makeImageSubresourceLayers(vk::ImageAspectFlags aspectMask,
                                                                    uint32_t mipLevel,
                                                                    uint32_t baseArrayLayer,
                                                                    uint32_t layerCount)
{
    return vk::ImageSubresourceLayers{
        .aspectMask = aspectMask,
        .mipLevel = mipLevel,
        .baseArrayLayer = baseArrayLayer,
        .layerCount = layerCount,
    };
}

[[nodiscard]] vk::ImageCreateInfo makeImageCreateInfo(vk::ImageCreateFlags flags,
                                                      vk::ImageType imageType, vk::Format format,
                                                      vk::Extent3D extent, uint32_t mipLevels,
                                                      uint32_t arrayLayers,
                                                      vk::ImageUsageFlags usage)
{
    return vk::ImageCreateInfo{
        .flags = flags,
        .imageType = imageType,
        .format = format,
        .extent = extent,
        .mipLevels = mipLevels,
        .arrayLayers = arrayLayers,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
}

[[nodiscard]] vk::MemoryAllocateInfo makeMemoryAllocateInfo(vk::MemoryRequirements requirements,
                                                            uint32_t memoryTypeIndex)
{
    return vk::MemoryAllocateInfo{
        .allocationSize = requirements.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
}

[[nodiscard]] vk::CommandBufferAllocateInfo
makeCommandBufferAllocateInfo(vk::CommandPool commandPool, uint32_t commandBufferCount)
{
    return vk::CommandBufferAllocateInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = commandBufferCount,
    };
}

[[nodiscard]] vk::CommandBufferBeginInfo makeOneTimeSubmitBeginInfo()
{
    return vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
}

[[nodiscard]] vk::ImageMemoryBarrier2
makeImageMemoryBarrier(vk::PipelineStageFlags2 srcStageMask, vk::AccessFlags2 srcAccessMask,
                       vk::PipelineStageFlags2 dstStageMask, vk::AccessFlags2 dstAccessMask,
                       vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::Image image,
                       vk::ImageSubresourceRange subresourceRange)
{
    return vk::ImageMemoryBarrier2{
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = subresourceRange,
    };
}

// Records a single image barrier through the synchronization2 path.
void recordImageBarrier(vk::CommandBuffer cmd, const vk::ImageMemoryBarrier2& barrier)
{
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

[[nodiscard]] vk::BufferImageCopy makeBufferImageCopy(vk::DeviceSize bufferOffset,
                                                      vk::ImageSubresourceLayers imageSubresource,
                                                      vk::Extent3D imageExtent)
{
    return vk::BufferImageCopy{
        .bufferOffset = bufferOffset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = imageSubresource,
        .imageOffset = vk::Offset3D{.x = 0, .y = 0, .z = 0},
        .imageExtent = imageExtent,
    };
}

[[nodiscard]] vk::ImageViewCreateInfo
makeImageViewCreateInfo(vk::Image image, vk::ImageViewType viewType, vk::Format format,
                        vk::ImageSubresourceRange subresourceRange)
{
    return vk::ImageViewCreateInfo{
        .image = image,
        .viewType = viewType,
        .format = format,
        .subresourceRange = subresourceRange,
    };
}

[[nodiscard]] vk::SamplerCreateInfo
makeSamplerCreateInfo(vk::Filter magFilter, vk::Filter minFilter, vk::SamplerMipmapMode mipmapMode,
                      vk::SamplerAddressMode addressModeU, vk::SamplerAddressMode addressModeV,
                      vk::SamplerAddressMode addressModeW, vk::Bool32 anisotropyEnable,
                      float maxAnisotropy, vk::Bool32 compareEnable, vk::CompareOp compareOp,
                      float minLod, float maxLod, vk::BorderColor borderColor)
{
    return vk::SamplerCreateInfo{
        .magFilter = magFilter,
        .minFilter = minFilter,
        .mipmapMode = mipmapMode,
        .addressModeU = addressModeU,
        .addressModeV = addressModeV,
        .addressModeW = addressModeW,
        .mipLodBias = 0.0f,
        .anisotropyEnable = anisotropyEnable,
        .maxAnisotropy = maxAnisotropy,
        .compareEnable = compareEnable,
        .compareOp = compareOp,
        .minLod = minLod,
        .maxLod = maxLod,
        .borderColor = borderColor,
        .unnormalizedCoordinates = vk::False,
    };
}

// Allocates a single primary command buffer from `pool`, runs `record(cmd)`
// inside a one-time-submit begin/end pair, submits the buffer on the graphics
// queue, and waits for the queue to idle before returning. Used for all the
// blocking, setup-time GPU work the renderer issues outside the main frame
// loop (texture uploads, layout transitions, etc).
template <typename F>
void withOneTimeSubmit(const Device& device, vk::CommandPool pool, F&& record)
{
    vk::CommandBufferAllocateInfo cmdAi = makeCommandBufferAllocateInfo(pool, 1);
    auto cmds = device.device().allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];
    cmd.begin(makeOneTimeSubmitBeginInfo());
    record(*cmd);
    cmd.end();
    vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
    vk::SubmitInfo2 submitInfo{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo};
    device.graphicsQueue().submit2(submitInfo);
    device.graphicsQueue().waitIdle();
}

// One-time transition Undefined → DepthStencilReadOnlyOptimal for every
// subresource of a depth image. Punctual-light shadow maps need this because
// only the layers/faces of *active* casters are touched by the shadow render
// pass each frame; unused layers would otherwise remain in UNDEFINED, but the
// forward shader's array sampler accesses the entire image.
void transitionDepthImageToReadOnly(const Device& device, vk::CommandPool pool, vk::Image image,
                                    uint32_t layerCount)
{
    withOneTimeSubmit(
        device, pool,
        [&](vk::CommandBuffer cmd)
        {
            vk::ImageMemoryBarrier2 toReadOnly = makeImageMemoryBarrier(
                vk::PipelineStageFlagBits2::eTopOfPipe, {},
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilReadOnlyOptimal, image,
                makeImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, layerCount));
            recordImageBarrier(cmd, toReadOnly);
        });
}

} // namespace

Resources::TextureEntry& Resources::appendTextureEntry(TextureHandle& handle, vk::Format format,
                                                       uint32_t mipLevels)
{
    handle = TextureHandle{static_cast<uint32_t>(textures_.size())};
    textures_.emplace_back();
    TextureEntry& entry = textures_.back();
    entry.format = format;
    entry.mipLevels = mipLevels;
    return entry;
}

void Resources::registerBindlessTexture(TextureHandle handle)
{
    if (!static_cast<bool>(*bindlessSet_))
    {
        return; // no forward pipeline / bindless set (e.g. headless contexts)
    }
    const TextureEntry& entry = textures_[static_cast<uint32_t>(handle)];
    const vk::DescriptorImageInfo info{*entry.sampler, *entry.view,
                                       vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::WriteDescriptorSet write{
        .dstSet = *bindlessSet_,
        .dstBinding = bindingIndex(BindlessBinding::Textures),
        .dstArrayElement = static_cast<uint32_t>(handle),
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &info,
    };
    device_->device().updateDescriptorSets(write, {});
}

uint32_t Resources::registerMaterial(const Material& material)
{
    if (materialMapped_ == nullptr)
    {
        return 0; // no bindless set (e.g. headless contexts)
    }
    if (auto it = materialIndices_.find(&material); it != materialIndices_.end())
    {
        return it->second;
    }
    if (materialCount_ >= kMaxMaterials)
    {
        throw std::runtime_error("Bindless material SSBO capacity (kMaxMaterials) exceeded");
    }

    const uint32_t index = materialCount_++;
    const MaterialUBO ubo = toMaterialUBO(material);
    std::memcpy(static_cast<char*>(materialMapped_) +
                    static_cast<std::size_t>(index) * sizeof(MaterialUBO),
                &ubo, sizeof(MaterialUBO));
    materialIndices_.emplace(&material, index);
    return index;
}

void Resources::allocateImage(TextureEntry& entry, const vk::ImageCreateInfo& imageInfo)
{
    entry.image = vk::raii::Image(device_->device(), imageInfo);

    auto imageRequirements = entry.image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocateInfo = makeMemoryAllocateInfo(
        imageRequirements, device_->findMemoryType(imageRequirements.memoryTypeBits,
                                                   vk::MemoryPropertyFlagBits::eDeviceLocal));
    entry.memory = vk::raii::DeviceMemory(device_->device(), allocateInfo);
    entry.image.bindMemory(*entry.memory, 0);
}

void Resources::createImageView(TextureEntry& entry, vk::ImageViewType viewType,
                                vk::ImageAspectFlags aspectMask, uint32_t baseMipLevel,
                                uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount)
{
    vk::ImageViewCreateInfo viewCi =
        makeImageViewCreateInfo(*entry.image, viewType, entry.format,
                                makeImageSubresourceRange(aspectMask, baseMipLevel, levelCount,
                                                          baseArrayLayer, layerCount));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);
}

void Resources::createSampler(TextureEntry& entry, const vk::SamplerCreateInfo& samplerInfo)
{
    entry.sampler = vk::raii::Sampler(device_->device(), samplerInfo);
}

void Resources::createSampledTextureSampler(TextureEntry& entry, const SamplerSettings& sampler,
                                            uint32_t mipLevels, vk::BorderColor borderColor)
{
    auto props = device_->physicalDevice().getProperties();
    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        toVkFilter(sampler.magFilter), toVkFilter(sampler.minFilter),
        vk::SamplerMipmapMode::eLinear, toVkAddressMode(sampler.wrapS),
        toVkAddressMode(sampler.wrapT), toVkAddressMode(sampler.wrapS), vk::True,
        props.limits.maxSamplerAnisotropy, vk::False, vk::CompareOp::eAlways, 0.0f,
        static_cast<float>(mipLevels - 1), borderColor);
    createSampler(entry, samplerCi);
}

void Resources::uploadImageFromHost(TextureEntry& entry, const void* pixels, vk::DeviceSize bytes,
                                    const vk::ImageCreateInfo& imageInfo,
                                    std::span<const vk::BufferImageCopy> regions)
{
    auto [stagingBuf, stagingMem] = device_->createBuffer(
        bytes, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* mapped = stagingMem.mapMemory(0, bytes);
    std::memcpy(mapped, pixels, static_cast<std::size_t>(bytes));
    stagingMem.unmapMemory();

    allocateImage(entry, imageInfo);

    const uint32_t mipLevels = imageInfo.mipLevels;
    const uint32_t arrayLayers = imageInfo.arrayLayers;

    withOneTimeSubmit(
        *device_, *cmdPool_,
        [&](vk::CommandBuffer cmd)
        {
            vk::ImageMemoryBarrier2 toTransfer = makeImageMemoryBarrier(
                vk::PipelineStageFlagBits2::eTopOfPipe, {}, vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal, *entry.image,
                makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0,
                                          arrayLayers));
            recordImageBarrier(cmd, toTransfer);

            cmd.copyBufferToImage(*stagingBuf, *entry.image, vk::ImageLayout::eTransferDstOptimal,
                                  regions);

            vk::ImageMemoryBarrier2 toShader = makeImageMemoryBarrier(
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
                vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                *entry.image,
                makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0,
                                          arrayLayers));
            recordImageBarrier(cmd, toShader);
        });
}

TextureHandle Resources::createUploaded2DTexture(const void* pixels, int width, int height,
                                                 vk::Format format, vk::DeviceSize bytesPerPixel,
                                                 const SamplerSettings& sampler,
                                                 vk::BorderColor borderColor)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, format);

    const vk::Extent3D extent{
        .width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height), .depth = 1};
    vk::BufferImageCopy region = makeBufferImageCopy(
        0, makeImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), extent);
    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format, extent, 1, 1,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    vk::DeviceSize bytes =
        static_cast<vk::DeviceSize>(width) * static_cast<vk::DeviceSize>(height) * bytesPerPixel;
    uploadImageFromHost(entry, pixels, bytes, imgCi,
                        std::span<const vk::BufferImageCopy>{&region, 1});

    createImageView(entry, vk::ImageViewType::e2D, vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    createSampledTextureSampler(entry, sampler, 1, borderColor);

    registerBindlessTexture(handle);
    return handle;
}

void Resources::createCubemapFaceViews(const Device& device, TextureEntry& entry)
{
    entry.faceViews.clear();
    entry.faceViews.reserve(static_cast<std::size_t>(entry.mipLevels) * 6);

    for (uint32_t mipLevel = 0; mipLevel < entry.mipLevels; ++mipLevel)
    {
        for (uint32_t face = 0; face < 6; ++face)
        {
            vk::ImageViewCreateInfo faceViewCi = makeImageViewCreateInfo(
                *entry.image, vk::ImageViewType::e2D, entry.format,
                makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, mipLevel, 1, face, 1));
            entry.faceViews.emplace_back(device.device(), faceViewCi);
        }
    }
}

TextureHandle Resources::createTexture(const Image& image, const SamplerSettings& sampler,
                                       TextureEncoding encoding)
{
    if (image.pixelType() == ImagePixelType::Float32)
    {
        return createTexture(image.dataf(), image.width(), image.height(), sampler);
    }

    return createTexture(image.data(), image.width(), image.height(), sampler, encoding);
}

TextureHandle Resources::createTexture(KtxImage&& image, const SamplerSettings& sampler,
                                       TextureEncoding encoding)
{
    if (image.empty())
    {
        throw std::runtime_error("Cannot upload an empty KTX image");
    }

    if (image.dimensions() != 2u || image.depth() != 1u || image.layers() != 1u ||
        image.faces() != 1u)
    {
        throw std::runtime_error("Only 2D single-layer KTX textures are currently supported");
    }

    if (image.needsTranscoding())
    {
        auto* texture = reinterpret_cast<ktxTexture2*>(image.native_handle());
        KTX_error_code error =
            ktxTexture2_TranscodeBasis(texture, chooseBasisTranscodeFormat(*device_, encoding), 0);
        if (error != KTX_SUCCESS)
        {
            throw std::runtime_error("Failed to transcode Basis-compressed KTX image (" +
                                     ktxFailureDetail(error) + ")");
        }
    }

    const vk::Format format = static_cast<vk::Format>(image.vkFormat());
    if (format == vk::Format::eUndefined)
    {
        throw std::runtime_error("KTX image does not expose a Vulkan format");
    }

    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, format, image.levels());

    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(entry.mipLevels);
    for (uint32_t level = 0; level < entry.mipLevels; ++level)
    {
        ktx_size_t offset = 0;
        KTX_error_code error =
            ktxTexture_GetImageOffset(image.native_handle(), level, 0, 0, &offset);
        if (error != KTX_SUCCESS)
        {
            throw std::runtime_error("Failed to read KTX mip offset (" + ktxFailureDetail(error) +
                                     ")");
        }

        regions.push_back(makeBufferImageCopy(
            static_cast<vk::DeviceSize>(offset),
            makeImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level, 0, 1),
            vk::Extent3D{.width = std::max(1u, image.width() >> level),
                         .height = std::max(1u, image.height() >> level),
                         .depth = 1}));
    }

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = image.width(), .height = image.height(), .depth = 1}, entry.mipLevels,
        1, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    uploadImageFromHost(entry, image.data(), static_cast<vk::DeviceSize>(image.size_bytes()), imgCi,
                        regions);

    createImageView(entry, vk::ImageViewType::e2D, vk::ImageAspectFlagBits::eColor, 0,
                    entry.mipLevels, 0, 1);
    createSampledTextureSampler(entry, sampler, entry.mipLevels, vk::BorderColor::eIntOpaqueBlack);

    registerBindlessTexture(handle);
    return handle;
}

TextureHandle Resources::createTexture(const uint8_t* pixels, int width, int height,
                                       const SamplerSettings& sampler, TextureEncoding encoding)
{
    return createUploaded2DTexture(pixels, width, height, toVkTextureFormat(encoding), 4, sampler,
                                   vk::BorderColor::eIntOpaqueBlack);
}

TextureHandle Resources::createTexture(const float* pixels, int width, int height,
                                       const SamplerSettings& sampler)
{
    return createUploaded2DTexture(pixels, width, height, vk::Format::eR32G32B32A32Sfloat,
                                   4 * sizeof(float), sampler, vk::BorderColor::eFloatOpaqueBlack);
}

TextureHandle Resources::createCubemapTexture(const float* pixels, uint32_t faceExtent,
                                              const SamplerSettings& sampler)
{
    return createCubemapTexture(pixels, faceExtent, 1, sampler);
}

TextureHandle Resources::createCubemapTexture(const float* pixels, uint32_t faceExtent,
                                              uint32_t mipLevels, const SamplerSettings& sampler)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eR32G32B32A32Sfloat, mipLevels);

    vk::DeviceSize imageSize = 0;
    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(static_cast<std::size_t>(mipLevels) * 6);
    {
        uint32_t levelExtent = faceExtent;
        vk::DeviceSize offset = 0;
        for (uint32_t level = 0; level < mipLevels; ++level)
        {
            vk::DeviceSize faceSize =
                static_cast<vk::DeviceSize>(levelExtent) * levelExtent * 4 * sizeof(float);
            for (uint32_t face = 0; face < 6; ++face)
            {
                regions.push_back(makeBufferImageCopy(
                    offset,
                    makeImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level, face, 1),
                    vk::Extent3D{.width = levelExtent, .height = levelExtent, .depth = 1}));
                offset += faceSize;
            }
            imageSize += faceSize * 6;
            levelExtent = std::max(1u, levelExtent / 2);
        }
    }

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        vk::ImageCreateFlagBits::eCubeCompatible, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = faceExtent, .height = faceExtent, .depth = 1}, mipLevels, 6,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    uploadImageFromHost(entry, pixels, imageSize, imgCi, regions);

    createImageView(entry, vk::ImageViewType::eCube, vk::ImageAspectFlagBits::eColor, 0, mipLevels,
                    0, 6);
    createCubemapFaceViews(*device_, entry);

    createSampledTextureSampler(entry, sampler, mipLevels, vk::BorderColor::eFloatOpaqueBlack);

    return handle;
}

TextureHandle Resources::createRenderTargetCubemap(uint32_t faceExtent, uint32_t mipLevels,
                                                   vk::Format format,
                                                   const SamplerSettings& sampler)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, format, mipLevels);

    // Transfer src/dst enable vkCmdBlitImage-based mip generation (used for
    // the skybox cubemap so the prefilter pass can do importance-sampled
    // mip-weighted lookups). Small memory/bandwidth cost when unused.
    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        vk::ImageCreateFlagBits::eCubeCompatible, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = faceExtent, .height = faceExtent, .depth = 1}, mipLevels, 6,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    allocateImage(entry, imgCi);

    createImageView(entry, vk::ImageViewType::eCube, vk::ImageAspectFlagBits::eColor, 0, mipLevels,
                    0, 6);
    createCubemapFaceViews(*device_, entry);

    createSampledTextureSampler(entry, sampler, mipLevels, vk::BorderColor::eFloatOpaqueBlack);

    return handle;
}

TextureHandle Resources::fallbackTexture(FallbackTextureKind kind)
{
    auto index = fallbackTextureIndex(kind);
    if (fallbackTextures_[index] == NullTexture)
    {
        fallbackTextures_[index] = createFallbackTexture(kind);
    }
    return fallbackTextures_[index];
}

TextureHandle Resources::createFallbackTexture(FallbackTextureKind kind)
{
    switch (kind)
    {
    case FallbackTextureKind::BaseColour:
    {
        static const uint8_t white[] = {255, 255, 255, 255};
        return createTexture(white, 1, 1, {}, TextureEncoding::Srgb);
    }
    case FallbackTextureKind::Emissive:
    {
        static const uint8_t black[] = {0, 0, 0, 255};
        return createTexture(black, 1, 1, {}, TextureEncoding::Srgb);
    }
    case FallbackTextureKind::Normal:
    {
        static const uint8_t flatNormal[] = {128, 128, 255, 255};
        return createTexture(flatNormal, 1, 1, {}, TextureEncoding::Linear);
    }
    case FallbackTextureKind::MetallicRoughness:
    case FallbackTextureKind::Occlusion:
    {
        static const uint8_t white[] = {255, 255, 255, 255};
        return createTexture(white, 1, 1, {}, TextureEncoding::Linear);
    }
    }

    return NullTexture;
}

// --- Shadow map ---

TextureHandle Resources::createShadowMap(uint32_t extent, uint32_t layerCount)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eD32Sfloat);

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = extent, .height = extent, .depth = 1}, 1, layerCount,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eTransferDst);
    allocateImage(entry, imgCi);

    // Main view: 2D for single-layer (sampler2DShadow), 2DArray for multi-layer
    // (sampler2DArrayShadow in the forward fragment shader).
    vk::ImageViewType mainViewType =
        layerCount > 1 ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, mainViewType, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, layerCount));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    // Per-layer 2D views for use as dynamic-rendering depth attachments — one
    // shadow iteration binds one layer's depth via baseArrayLayer=L.
    if (layerCount > 1)
    {
        entry.faceViews.reserve(layerCount);
        for (uint32_t layer = 0; layer < layerCount; ++layer)
        {
            vk::ImageViewCreateInfo layerCi = makeImageViewCreateInfo(
                *entry.image, vk::ImageViewType::e2D, entry.format,
                makeImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, layer, 1));
            entry.faceViews.emplace_back(device_->device(), layerCi);
        }
    }

    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
        vk::SamplerAddressMode::eClampToBorder, vk::SamplerAddressMode::eClampToBorder,
        vk::SamplerAddressMode::eClampToBorder, vk::False, 1.0f, vk::True,
        vk::CompareOp::eLessOrEqual, 0.0f, 1.0f, vk::BorderColor::eFloatOpaqueWhite);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    transitionDepthImageToReadOnly(*device_, *cmdPool_, *entry.image, layerCount);

    return handle;
}

vk::ImageView Resources::vulkanShadowMapLayerView(TextureHandle handle,
                                                  uint32_t layer) const noexcept
{
    const auto& entry = textures_[static_cast<uint32_t>(handle)];
    return *entry.faceViews[layer];
}

TextureHandle Resources::createPointShadowMap(uint32_t faceExtent, uint32_t cubeCount)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eD32Sfloat);

    const uint32_t totalLayers = 6u * cubeCount;
    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        vk::ImageCreateFlagBits::eCubeCompatible, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = faceExtent, .height = faceExtent, .depth = 1}, 1, totalLayers,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled);
    allocateImage(entry, imgCi);

    // Main view: cube array (samplerCubeArrayShadow in the forward fragment).
    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::eCubeArray, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, totalLayers));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    // Per-face 2D depth views — `6 * cube + face`. Used as dynamic-rendering
    // depth attachments during the depth pass (one face per begin/endRendering).
    entry.faceViews.reserve(totalLayers);
    for (uint32_t layer = 0; layer < totalLayers; ++layer)
    {
        vk::ImageViewCreateInfo faceCi = makeImageViewCreateInfo(
            *entry.image, vk::ImageViewType::e2D, entry.format,
            makeImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, layer, 1));
        entry.faceViews.emplace_back(device_->device(), faceCi);
    }

    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge, vk::False, 1.0f, vk::True,
        vk::CompareOp::eLessOrEqual, 0.0f, 1.0f, vk::BorderColor::eFloatOpaqueWhite);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    transitionDepthImageToReadOnly(*device_, *cmdPool_, *entry.image, totalLayers);

    return handle;
}

vk::ImageView Resources::vulkanPointShadowFaceView(TextureHandle handle, uint32_t cubeIndex,
                                                   uint32_t face) const noexcept
{
    const auto& entry = textures_[static_cast<uint32_t>(handle)];
    return *entry.faceViews[6u * cubeIndex + face];
}

TextureHandle Resources::createOffscreenColourTarget(vk::Extent2D extent)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eR16G16B16A16Sfloat);

    // TransferSrc: KHR_materials_transmission F3 blits this into the sceneColor
    // mip chain. TransferDst: the TAA resolve blits its history result back in
    // here so bloom / post sample the anti-aliased image. Sampled: post-process
    // and the TAA resolve read it. (TAA history targets reuse this same factory.)
    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = extent.width, .height = extent.height, .depth = 1}, 1, 1,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    allocateImage(entry, imgCi);

    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::e2D, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge, vk::False, 1.0f, vk::False, vk::CompareOp::eAlways,
        0.0f, 1.0f, vk::BorderColor::eFloatOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return handle;
}

TextureHandle Resources::createVelocityTarget(vk::Extent2D extent)
{
    TextureHandle handle;
    // RG16F screen-space motion vectors (TAA). Must match the velocity colour
    // format in Pipeline::forwardConfig / skyboxConfig.
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eR16G16Sfloat);

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = extent.width, .height = extent.height, .depth = 1}, 1, 1,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    allocateImage(entry, imgCi);

    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::e2D, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    // Nearest sampling — motion vectors must not be filtered across edges.
    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge, vk::False, 1.0f, vk::False, vk::CompareOp::eAlways,
        0.0f, 1.0f, vk::BorderColor::eFloatOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return handle;
}

TextureHandle Resources::createBloomChain(uint32_t width, uint32_t height, uint32_t mipLevels)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eR16G16B16A16Sfloat, mipLevels);

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = width, .height = height, .depth = 1}, mipLevels, 1,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    allocateImage(entry, imgCi);

    // Main view spans all mips — used as the post-process bloom input via mip 0.
    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::e2D, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    // Per-mip 2D views — each downsample/upsample pass binds one as
    // framebuffer attachment (write) and another as shader input (read).
    entry.faceViews.reserve(mipLevels);
    for (uint32_t m = 0; m < mipLevels; ++m)
    {
        vk::ImageViewCreateInfo mipCi = makeImageViewCreateInfo(
            *entry.image, vk::ImageViewType::e2D, entry.format,
            makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, m, 1, 0, 1));
        entry.faceViews.emplace_back(device_->device(), mipCi);
    }

    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge, vk::False, 1.0f, vk::False, vk::CompareOp::eAlways,
        0.0f, 0.0f, vk::BorderColor::eFloatOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return handle;
}

vk::ImageView Resources::vulkanBloomMipView(TextureHandle handle, uint32_t mipLevel) const noexcept
{
    return *textures_[static_cast<uint32_t>(handle)].faceViews[mipLevel];
}

TextureHandle Resources::createSceneColorTarget(uint32_t width, uint32_t height, uint32_t mipLevels)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eR16G16B16A16Sfloat, mipLevels);

    // KHR_materials_transmission F3 — receives a blit copy from the post-opaque
    // HDR target and then a vkCmdBlitImage chain for the remaining mips.
    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = width, .height = height, .depth = 1}, mipLevels, 1,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eSampled);
    allocateImage(entry, imgCi);

    // Initial transition Undefined → ShaderReadOnlyOptimal for ALL mips. The
    // forward descriptor set binds sceneColor at binding 20 on every draw —
    // including the non-transmissive ones in pass 1 — so the layout must
    // match what the descriptor was written with even before any capture
    // pass writes meaningful contents.
    withOneTimeSubmit(
        *device_, *cmdPool_,
        [&](vk::CommandBuffer cmd)
        {
            vk::ImageMemoryBarrier2 toShader = makeImageMemoryBarrier(
                vk::PipelineStageFlagBits2::eTopOfPipe, {},
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, *entry.image,
                makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1));
            recordImageBarrier(cmd, toShader);
        });

    // Main view spans all mips so the shader's textureLod sample picks a mip
    // from a roughness-driven LOD.
    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::e2D, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    // Linear min/mag + linear mip filter so frosted-glass roughness blends
    // smoothly between mips. ClampToEdge — refraction can read off-screen.
    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge, vk::False, 1.0f, vk::False, vk::CompareOp::eAlways,
        0.0f, static_cast<float>(mipLevels), vk::BorderColor::eFloatOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return handle;
}

void Resources::releaseTexture(TextureHandle handle)
{
    auto& entry = textures_[static_cast<uint32_t>(handle)];
    entry.sampler = vk::raii::Sampler{nullptr};
    entry.faceViews.clear();
    entry.view = vk::raii::ImageView{nullptr};
    entry.image = vk::raii::Image{nullptr};
    entry.memory = vk::raii::DeviceMemory{nullptr};
    entry.format = vk::Format::eUndefined;
    entry.mipLevels = 1;
}

// --- Mapped buffer sets ---

Resources::MappedBufferSet Resources::createMappedUniformBuffers(std::size_t size)
{
    MappedBufferSet result;
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        auto [buf, mem] = device_->createBuffer(
            static_cast<vk::DeviceSize>(size), vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        result.mapped[i] = mem.mapMemory(0, static_cast<vk::DeviceSize>(size));
        result.buffers[i] = storeBuffer(std::move(buf), std::move(mem));
    }
    return result;
}

Resources::MappedBufferSet Resources::createMappedStorageBuffer(std::size_t size,
                                                                const void* initialData)
{
    MappedBufferSet result;
    // Storage buffers are shared across frames — create a single buffer,
    // but fill both slots so callers can index by frame uniformly.
    // eTransferDst lets callers clear/upload via vkCmdFillBuffer / copy.
    auto [buf, mem] = device_->createBuffer(
        static_cast<vk::DeviceSize>(size),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (initialData != nullptr)
    {
        void* mapped = mem.mapMemory(0, static_cast<vk::DeviceSize>(size));
        std::memcpy(mapped, initialData, size);
        mem.unmapMemory();
    }
    auto handle = storeBuffer(std::move(buf), std::move(mem));
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        result.buffers[i] = handle;
        result.mapped[i] = nullptr;
    }
    return result;
}

// --- Pipeline registry ---

PipelineHandle Resources::registerPipeline(vk::Pipeline pipeline, vk::PipelineLayout layout)
{
    auto id = static_cast<uint32_t>(pipelines_.size());
    pipelines_.push_back({pipeline, layout});
    return PipelineHandle{id};
}

// --- Vulkan accessors ---

vk::Buffer Resources::vulkanBuffer(BufferHandle handle) const noexcept
{
    return *buffers_[static_cast<uint32_t>(handle)].buffer;
}

vk::ImageView Resources::vulkanImageView(TextureHandle handle) const noexcept
{
    return *textures_[static_cast<uint32_t>(handle)].view;
}

vk::ImageView Resources::vulkanCubemapFaceView(TextureHandle handle, uint32_t face,
                                               uint32_t mipLevel) const noexcept
{
    const auto& entry = textures_[static_cast<uint32_t>(handle)];
    std::size_t index = static_cast<std::size_t>(mipLevel) * 6 + face;
    return *entry.faceViews[index];
}

vk::Image Resources::vulkanImage(TextureHandle handle) const noexcept
{
    return *textures_[static_cast<uint32_t>(handle)].image;
}

vk::Sampler Resources::vulkanSampler(TextureHandle handle) const noexcept
{
    return *textures_[static_cast<uint32_t>(handle)].sampler;
}

vk::Sampler Resources::vulkanShadowDebugSampler() const noexcept
{
    return *shadowDebugSampler_;
}

vk::Format Resources::textureFormat(TextureHandle handle) const noexcept
{
    return textures_[static_cast<uint32_t>(handle)].format;
}

uint32_t Resources::textureMipLevels(TextureHandle handle) const noexcept
{
    return textures_[static_cast<uint32_t>(handle)].mipLevels;
}

vk::DescriptorSet Resources::vulkanDescriptorSet(DescriptorSetHandle handle) const noexcept
{
    return descriptors_.vulkanDescriptorSet(handle);
}

vk::Pipeline Resources::vulkanPipeline(PipelineHandle handle) const noexcept
{
    return pipelines_[static_cast<uint32_t>(handle)].pipeline;
}

vk::PipelineLayout Resources::vulkanPipelineLayout(PipelineHandle handle) const noexcept
{
    return pipelines_[static_cast<uint32_t>(handle)].layout;
}

} // namespace fire_engine
