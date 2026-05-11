#include <fire_engine/render/resources.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fire_engine/graphics/image.hpp>
#include <fire_engine/graphics/ktx_image.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

Resources::Resources(const Device& device, const Pipeline& pipeline)
    : device_(&device),
      pipeline_(&pipeline),
      descriptors_(device, pipeline, *this)
{
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

static vk::SamplerAddressMode toVkAddressMode(WrapMode mode)
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

static vk::Filter toVkFilter(FilterMode mode)
{
    switch (mode)
    {
    case FilterMode::Nearest:
        return vk::Filter::eNearest;
    default:
        return vk::Filter::eLinear;
    }
}

static vk::Format toVkTextureFormat(TextureEncoding encoding)
{
    return encoding == TextureEncoding::Srgb ? vk::Format::eR8G8B8A8Srgb
                                             : vk::Format::eR8G8B8A8Unorm;
}

static std::size_t fallbackTextureIndex(Resources::FallbackTextureKind kind)
{
    return static_cast<std::size_t>(kind);
}

[[nodiscard]] static bool supportsSampledTextureFormat(const Device& device, vk::Format format)
{
    vk::FormatProperties properties = device.physicalDevice().getFormatProperties(format);
    constexpr vk::FormatFeatureFlags required =
        vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
    return (properties.optimalTilingFeatures & required) == required;
}

[[nodiscard]] static ktx_transcode_fmt_e chooseBasisTranscodeFormat(const Device& device,
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

[[nodiscard]] static std::string ktxFailureDetail(KTX_error_code error)
{
    const char* reason = ktxErrorString(error);
    if (reason == nullptr || reason[0] == '\0')
    {
        return "unknown libktx error";
    }

    return reason;
}

[[nodiscard]] static vk::ImageSubresourceRange
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

[[nodiscard]] static vk::ImageSubresourceLayers
makeImageSubresourceLayers(vk::ImageAspectFlags aspectMask, uint32_t mipLevel,
                           uint32_t baseArrayLayer, uint32_t layerCount)
{
    return vk::ImageSubresourceLayers{
        .aspectMask = aspectMask,
        .mipLevel = mipLevel,
        .baseArrayLayer = baseArrayLayer,
        .layerCount = layerCount,
    };
}

[[nodiscard]] static vk::ImageCreateInfo
makeImageCreateInfo(vk::ImageCreateFlags flags, vk::ImageType imageType, vk::Format format,
                    vk::Extent3D extent, uint32_t mipLevels, uint32_t arrayLayers,
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

[[nodiscard]] static vk::MemoryAllocateInfo
makeMemoryAllocateInfo(vk::MemoryRequirements requirements, uint32_t memoryTypeIndex)
{
    return vk::MemoryAllocateInfo{
        .allocationSize = requirements.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
}

[[nodiscard]] static vk::CommandBufferAllocateInfo
makeCommandBufferAllocateInfo(vk::CommandPool commandPool, uint32_t commandBufferCount)
{
    return vk::CommandBufferAllocateInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = commandBufferCount,
    };
}

[[nodiscard]] static vk::CommandBufferBeginInfo makeOneTimeSubmitBeginInfo()
{
    return vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
}

[[nodiscard]] static vk::ImageMemoryBarrier
makeImageMemoryBarrier(vk::AccessFlags srcAccessMask, vk::AccessFlags dstAccessMask,
                       vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::Image image,
                       vk::ImageSubresourceRange subresourceRange)
{
    return vk::ImageMemoryBarrier{
        .srcAccessMask = srcAccessMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = subresourceRange,
    };
}

[[nodiscard]] static vk::BufferImageCopy
makeBufferImageCopy(vk::DeviceSize bufferOffset, vk::ImageSubresourceLayers imageSubresource,
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

[[nodiscard]] static vk::ImageViewCreateInfo
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

[[nodiscard]] static vk::SamplerCreateInfo
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

TextureHandle Resources::createTexture(KtxImage image, const SamplerSettings& sampler,
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

    auto id = static_cast<uint32_t>(textures_.size());
    textures_.emplace_back();
    auto& entry = textures_.back();
    entry.format = format;
    entry.mipLevels = image.levels();

    const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(image.size_bytes());
    auto [stagingBuf, stagingMem] = device_->createBuffer(
        imageSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data = stagingMem.mapMemory(0, imageSize);
    std::memcpy(data, image.data(), static_cast<std::size_t>(imageSize));
    stagingMem.unmapMemory();

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = image.width(), .height = image.height(), .depth = 1}, entry.mipLevels,
        1, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    entry.image = vk::raii::Image(device_->device(), imgCi);

    auto imgReq = entry.image.getMemoryRequirements();
    vk::MemoryAllocateInfo imgAi = makeMemoryAllocateInfo(
        imgReq,
        device_->findMemoryType(imgReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
    entry.memory = vk::raii::DeviceMemory(device_->device(), imgAi);
    entry.image.bindMemory(*entry.memory, 0);

    vk::CommandBufferAllocateInfo cmdAi = makeCommandBufferAllocateInfo(*cmdPool_, 1);
    auto cmds = device_->device().allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];

    cmd.begin(makeOneTimeSubmitBeginInfo());

    vk::ImageMemoryBarrier toTransfer = makeImageMemoryBarrier(
        {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, entry.mipLevels, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                        {}, {}, {}, toTransfer);

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
    cmd.copyBufferToImage(*stagingBuf, *entry.image, vk::ImageLayout::eTransferDstOptimal, regions);

    vk::ImageMemoryBarrier toShader = makeImageMemoryBarrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, entry.mipLevels, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, toShader);

    cmd.end();

    vk::CommandBuffer rawCmd = *cmd;
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &rawCmd,
    };
    device_->graphicsQueue().submit(submitInfo);
    device_->graphicsQueue().waitIdle();

    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::e2D, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, entry.mipLevels, 0, 1));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    auto props = device_->physicalDevice().getProperties();
    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        toVkFilter(sampler.magFilter), toVkFilter(sampler.minFilter),
        vk::SamplerMipmapMode::eLinear, toVkAddressMode(sampler.wrapS),
        toVkAddressMode(sampler.wrapT), toVkAddressMode(sampler.wrapS), vk::True,
        props.limits.maxSamplerAnisotropy, vk::False, vk::CompareOp::eAlways, 0.0f,
        static_cast<float>(entry.mipLevels - 1), vk::BorderColor::eIntOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return TextureHandle{id};
}

TextureHandle Resources::createTexture(const uint8_t* pixels, int width, int height,
                                       const SamplerSettings& sampler, TextureEncoding encoding)
{
    auto id = static_cast<uint32_t>(textures_.size());
    textures_.emplace_back();
    auto& entry = textures_.back();
    entry.format = toVkTextureFormat(encoding);

    vk::DeviceSize imageSize =
        static_cast<vk::DeviceSize>(width) * static_cast<vk::DeviceSize>(height) * 4;

    auto [stagingBuf, stagingMem] = device_->createBuffer(
        imageSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data = stagingMem.mapMemory(0, imageSize);
    std::memcpy(data, pixels, static_cast<std::size_t>(imageSize));
    stagingMem.unmapMemory();

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = static_cast<uint32_t>(width),
                     .height = static_cast<uint32_t>(height),
                     .depth = 1},
        1, 1, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    entry.image = vk::raii::Image(device_->device(), imgCi);

    auto imgReq = entry.image.getMemoryRequirements();
    vk::MemoryAllocateInfo imgAi = makeMemoryAllocateInfo(
        imgReq,
        device_->findMemoryType(imgReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
    entry.memory = vk::raii::DeviceMemory(device_->device(), imgAi);
    entry.image.bindMemory(*entry.memory, 0);

    vk::CommandBufferAllocateInfo cmdAi = makeCommandBufferAllocateInfo(*cmdPool_, 1);
    auto cmds = device_->device().allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];

    cmd.begin(makeOneTimeSubmitBeginInfo());

    vk::ImageMemoryBarrier toTransfer = makeImageMemoryBarrier(
        {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                        {}, {}, {}, toTransfer);

    vk::BufferImageCopy region =
        makeBufferImageCopy(0, makeImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                            vk::Extent3D{.width = static_cast<uint32_t>(width),
                                         .height = static_cast<uint32_t>(height),
                                         .depth = 1});
    cmd.copyBufferToImage(*stagingBuf, *entry.image, vk::ImageLayout::eTransferDstOptimal, region);

    vk::ImageMemoryBarrier toShader = makeImageMemoryBarrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, toShader);

    cmd.end();

    vk::CommandBuffer rawCmd = *cmd;
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &rawCmd,
    };
    device_->graphicsQueue().submit(submitInfo);
    device_->graphicsQueue().waitIdle();

    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::e2D, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    auto props = device_->physicalDevice().getProperties();
    vk::SamplerCreateInfo samplerCi =
        makeSamplerCreateInfo(toVkFilter(sampler.magFilter), toVkFilter(sampler.minFilter),
                              vk::SamplerMipmapMode::eLinear, toVkAddressMode(sampler.wrapS),
                              toVkAddressMode(sampler.wrapT), toVkAddressMode(sampler.wrapS),
                              vk::True, props.limits.maxSamplerAnisotropy, vk::False,
                              vk::CompareOp::eAlways, 0.0f, 0.0f, vk::BorderColor::eIntOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return TextureHandle{id};
}

TextureHandle Resources::createTexture(const float* pixels, int width, int height,
                                       const SamplerSettings& sampler)
{
    auto id = static_cast<uint32_t>(textures_.size());
    textures_.emplace_back();
    auto& entry = textures_.back();
    entry.format = vk::Format::eR32G32B32A32Sfloat;
    entry.mipLevels = 1;

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) *
                               static_cast<vk::DeviceSize>(height) * 4 * sizeof(float);

    auto [stagingBuf, stagingMem] = device_->createBuffer(
        imageSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data = stagingMem.mapMemory(0, imageSize);
    std::memcpy(data, pixels, static_cast<std::size_t>(imageSize));
    stagingMem.unmapMemory();

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = static_cast<uint32_t>(width),
                     .height = static_cast<uint32_t>(height),
                     .depth = 1},
        1, 1, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    entry.image = vk::raii::Image(device_->device(), imgCi);

    auto imgReq = entry.image.getMemoryRequirements();
    vk::MemoryAllocateInfo imgAi = makeMemoryAllocateInfo(
        imgReq,
        device_->findMemoryType(imgReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
    entry.memory = vk::raii::DeviceMemory(device_->device(), imgAi);
    entry.image.bindMemory(*entry.memory, 0);

    vk::CommandBufferAllocateInfo cmdAi = makeCommandBufferAllocateInfo(*cmdPool_, 1);
    auto cmds = device_->device().allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];

    cmd.begin(makeOneTimeSubmitBeginInfo());

    vk::ImageMemoryBarrier toTransfer = makeImageMemoryBarrier(
        {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                        {}, {}, {}, toTransfer);

    vk::BufferImageCopy region =
        makeBufferImageCopy(0, makeImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                            vk::Extent3D{.width = static_cast<uint32_t>(width),
                                         .height = static_cast<uint32_t>(height),
                                         .depth = 1});
    cmd.copyBufferToImage(*stagingBuf, *entry.image, vk::ImageLayout::eTransferDstOptimal, region);

    vk::ImageMemoryBarrier toShader = makeImageMemoryBarrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, toShader);

    cmd.end();

    vk::CommandBuffer rawCmd = *cmd;
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &rawCmd,
    };
    device_->graphicsQueue().submit(submitInfo);
    device_->graphicsQueue().waitIdle();

    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::e2D, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    auto props = device_->physicalDevice().getProperties();
    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        toVkFilter(sampler.magFilter), toVkFilter(sampler.minFilter),
        vk::SamplerMipmapMode::eLinear, toVkAddressMode(sampler.wrapS),
        toVkAddressMode(sampler.wrapT), toVkAddressMode(sampler.wrapS), vk::True,
        props.limits.maxSamplerAnisotropy, vk::False, vk::CompareOp::eAlways, 0.0f, 0.0f,
        vk::BorderColor::eFloatOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return TextureHandle{id};
}

TextureHandle Resources::createCubemapTexture(const float* pixels, uint32_t faceExtent,
                                              const SamplerSettings& sampler)
{
    return createCubemapTexture(pixels, faceExtent, 1, sampler);
}

TextureHandle Resources::createCubemapTexture(const float* pixels, uint32_t faceExtent,
                                              uint32_t mipLevels, const SamplerSettings& sampler)
{
    auto id = static_cast<uint32_t>(textures_.size());
    textures_.emplace_back();
    auto& entry = textures_.back();
    entry.format = vk::Format::eR32G32B32A32Sfloat;
    entry.mipLevels = mipLevels;

    vk::DeviceSize imageSize = 0;
    uint32_t levelExtent = faceExtent;
    for (uint32_t level = 0; level < mipLevels; ++level)
    {
        vk::DeviceSize faceSize =
            static_cast<vk::DeviceSize>(levelExtent) * levelExtent * 4 * sizeof(float);
        imageSize += faceSize * 6;
        levelExtent = std::max(1u, levelExtent / 2);
    }

    auto [stagingBuf, stagingMem] = device_->createBuffer(
        imageSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data = stagingMem.mapMemory(0, imageSize);
    std::memcpy(data, pixels, static_cast<std::size_t>(imageSize));
    stagingMem.unmapMemory();

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        vk::ImageCreateFlagBits::eCubeCompatible, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = faceExtent, .height = faceExtent, .depth = 1}, mipLevels, 6,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    entry.image = vk::raii::Image(device_->device(), imgCi);

    auto imgReq = entry.image.getMemoryRequirements();
    vk::MemoryAllocateInfo imgAi = makeMemoryAllocateInfo(
        imgReq,
        device_->findMemoryType(imgReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
    entry.memory = vk::raii::DeviceMemory(device_->device(), imgAi);
    entry.image.bindMemory(*entry.memory, 0);

    vk::CommandBufferAllocateInfo cmdAi = makeCommandBufferAllocateInfo(*cmdPool_, 1);
    auto cmds = device_->device().allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];
    cmd.begin(makeOneTimeSubmitBeginInfo());

    vk::ImageMemoryBarrier toTransfer = makeImageMemoryBarrier(
        {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 6));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                        {}, {}, {}, toTransfer);

    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(static_cast<std::size_t>(mipLevels) * 6);
    vk::DeviceSize offset = 0;
    levelExtent = faceExtent;
    for (uint32_t level = 0; level < mipLevels; ++level)
    {
        vk::DeviceSize faceSize =
            static_cast<vk::DeviceSize>(levelExtent) * levelExtent * 4 * sizeof(float);
        for (uint32_t face = 0; face < 6; ++face)
        {
            regions.push_back(makeBufferImageCopy(
                offset, makeImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level, face, 1),
                vk::Extent3D{.width = levelExtent, .height = levelExtent, .depth = 1}));
            offset += faceSize;
        }
        levelExtent = std::max(1u, levelExtent / 2);
    }
    cmd.copyBufferToImage(*stagingBuf, *entry.image, vk::ImageLayout::eTransferDstOptimal, regions);

    vk::ImageMemoryBarrier toShader = makeImageMemoryBarrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 6));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, toShader);
    cmd.end();

    vk::CommandBuffer rawCmd = *cmd;
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &rawCmd,
    };
    device_->graphicsQueue().submit(submitInfo);
    device_->graphicsQueue().waitIdle();

    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::eCube, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 6));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);
    createCubemapFaceViews(*device_, entry);

    auto props = device_->physicalDevice().getProperties();
    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        toVkFilter(sampler.magFilter), toVkFilter(sampler.minFilter),
        vk::SamplerMipmapMode::eLinear, toVkAddressMode(sampler.wrapS),
        toVkAddressMode(sampler.wrapT), toVkAddressMode(sampler.wrapS), vk::True,
        props.limits.maxSamplerAnisotropy, vk::False, vk::CompareOp::eAlways, 0.0f,
        static_cast<float>(mipLevels - 1), vk::BorderColor::eFloatOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return TextureHandle{id};
}

TextureHandle Resources::createRenderTargetCubemap(uint32_t faceExtent, uint32_t mipLevels,
                                                   vk::Format format,
                                                   const SamplerSettings& sampler)
{
    auto id = static_cast<uint32_t>(textures_.size());
    textures_.emplace_back();
    auto& entry = textures_.back();
    entry.format = format;
    entry.mipLevels = mipLevels;

    // Transfer src/dst enable vkCmdBlitImage-based mip generation (used for
    // the skybox cubemap so the prefilter pass can do importance-sampled
    // mip-weighted lookups). Small memory/bandwidth cost when unused.
    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        vk::ImageCreateFlagBits::eCubeCompatible, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = faceExtent, .height = faceExtent, .depth = 1}, mipLevels, 6,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    entry.image = vk::raii::Image(device_->device(), imgCi);

    auto imgReq = entry.image.getMemoryRequirements();
    vk::MemoryAllocateInfo imgAi = makeMemoryAllocateInfo(
        imgReq,
        device_->findMemoryType(imgReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
    entry.memory = vk::raii::DeviceMemory(device_->device(), imgAi);
    entry.image.bindMemory(*entry.memory, 0);

    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, vk::ImageViewType::eCube, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 6));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);
    createCubemapFaceViews(*device_, entry);

    auto props = device_->physicalDevice().getProperties();
    vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
        toVkFilter(sampler.magFilter), toVkFilter(sampler.minFilter),
        vk::SamplerMipmapMode::eLinear, toVkAddressMode(sampler.wrapS),
        toVkAddressMode(sampler.wrapT), toVkAddressMode(sampler.wrapS), vk::True,
        props.limits.maxSamplerAnisotropy, vk::False, vk::CompareOp::eAlways, 0.0f,
        static_cast<float>(mipLevels - 1), vk::BorderColor::eFloatOpaqueBlack);
    entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);

    return TextureHandle{id};
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

// One-time transition Undefined → DepthStencilReadOnlyOptimal for every
// subresource of a depth image. Punctual-light shadow maps need this because
// only the layers/faces of *active* casters are touched by the shadow render
// pass each frame; unused layers would otherwise remain in UNDEFINED, but the
// forward shader's array sampler accesses the entire image.
static void transitionDepthImageToReadOnly(const Device& device, vk::CommandPool pool,
                                           vk::Image image, uint32_t layerCount)
{
    vk::CommandBufferAllocateInfo cmdAi = makeCommandBufferAllocateInfo(pool, 1);
    auto cmds = device.device().allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];
    cmd.begin(makeOneTimeSubmitBeginInfo());
    vk::ImageMemoryBarrier toReadOnly = makeImageMemoryBarrier(
        {}, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilReadOnlyOptimal, image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, layerCount));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, toReadOnly);
    cmd.end();
    vk::CommandBuffer rawCmd = *cmd;
    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &rawCmd};
    device.graphicsQueue().submit(submitInfo);
    device.graphicsQueue().waitIdle();
}

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

    // Per-layer 2D views for use as framebuffer attachments — one shadow
    // framebuffer binds one layer's depth via baseArrayLayer=L.
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

    // Per-face 2D depth views — `6 * cube + face`. Used as framebuffer
    // attachments during the depth pass (one face per render-pass invocation).
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

TextureHandle Resources::createShadowColourAttachment(uint32_t extent, uint32_t layerCount,
                                                      bool sampled)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eB8G8R8A8Unorm);

    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = extent, .height = extent, .depth = 1}, 1, layerCount,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    allocateImage(entry, imgCi);

    vk::ImageViewType mainViewType =
        layerCount > 1 ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
    vk::ImageViewCreateInfo viewCi = makeImageViewCreateInfo(
        *entry.image, mainViewType, entry.format,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, layerCount));
    entry.view = vk::raii::ImageView(device_->device(), viewCi);

    if (layerCount > 1)
    {
        entry.faceViews.reserve(layerCount);
        for (uint32_t layer = 0; layer < layerCount; ++layer)
        {
            vk::ImageViewCreateInfo layerCi = makeImageViewCreateInfo(
                *entry.image, vk::ImageViewType::e2D, entry.format,
                makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, layer, 1));
            entry.faceViews.emplace_back(device_->device(), layerCi);
        }
    }

    if (sampled)
    {
        vk::SamplerCreateInfo samplerCi = makeSamplerCreateInfo(
            vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
            vk::SamplerAddressMode::eClampToBorder, vk::SamplerAddressMode::eClampToBorder,
            vk::SamplerAddressMode::eClampToBorder, vk::False, 1.0f, vk::False,
            vk::CompareOp::eAlways, 0.0f, 1.0f, vk::BorderColor::eFloatOpaqueWhite);
        entry.sampler = vk::raii::Sampler(device_->device(), samplerCi);
    }

    return handle;
}

vk::ImageView Resources::vulkanShadowColourLayerView(TextureHandle handle,
                                                     uint32_t layer) const noexcept
{
    const auto& entry = textures_[static_cast<uint32_t>(handle)];
    return *entry.faceViews[layer];
}

TextureHandle Resources::createOffscreenColourTarget(vk::Extent2D extent)
{
    TextureHandle handle;
    TextureEntry& entry = appendTextureEntry(handle, vk::Format::eR16G16B16A16Sfloat);

    // KHR_materials_transmission F3 also requires TransferSrc on the HDR
    // target so the sceneColor capture pass can blit from it.
    vk::ImageCreateInfo imgCi = makeImageCreateInfo(
        {}, vk::ImageType::e2D, entry.format,
        vk::Extent3D{.width = extent.width, .height = extent.height, .depth = 1}, 1, 1,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eTransferSrc);
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
    vk::CommandBufferAllocateInfo cmdAi = makeCommandBufferAllocateInfo(*cmdPool_, 1);
    auto cmds = device_->device().allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];
    cmd.begin(makeOneTimeSubmitBeginInfo());
    vk::ImageMemoryBarrier toShader = makeImageMemoryBarrier(
        {}, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eShaderReadOnlyOptimal, *entry.image,
        makeImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, toShader);
    cmd.end();
    vk::CommandBuffer rawCmd = *cmd;
    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &rawCmd};
    device_->graphicsQueue().submit(submitInfo);
    device_->graphicsQueue().waitIdle();

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
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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
    // but fill both slots so callers can index by frame uniformly
    auto [buf, mem] = device_->createBuffer(
        static_cast<vk::DeviceSize>(size), vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (initialData != nullptr)
    {
        void* mapped = mem.mapMemory(0, static_cast<vk::DeviceSize>(size));
        std::memcpy(mapped, initialData, size);
        mem.unmapMemory();
    }
    auto handle = storeBuffer(std::move(buf), std::move(mem));
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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
