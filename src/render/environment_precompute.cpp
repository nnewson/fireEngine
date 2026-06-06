#include <fire_engine/render/environment_precompute.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>

#include <fire_engine/graphics/image.hpp>
#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/cubemap_basis.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/render_pass.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

namespace
{

constexpr const char* defaultEnvironmentFilename = "skybox.hdr";

#ifndef FIRE_ENGINE_SOURCE_ASSET_DIR
#define FIRE_ENGINE_SOURCE_ASSET_DIR "assets"
#endif

#ifndef FIRE_ENGINE_BUILD_ASSET_DIR
#define FIRE_ENGINE_BUILD_ASSET_DIR "."
#endif

[[nodiscard]] std::string resolveSkyboxPath(const std::string& requestedPath)
{
    namespace fs = std::filesystem;

    if (!requestedPath.empty())
    {
        fs::path requested{requestedPath};
        if (fs::exists(requested))
        {
            return requested.string();
        }

        const std::array<fs::path, 3> requestedCandidates = {
            fs::path(FIRE_ENGINE_BUILD_ASSET_DIR) / requested,
            fs::path(FIRE_ENGINE_SOURCE_ASSET_DIR) / requested,
            fs::path("assets") / requested,
        };

        for (const auto& candidate : requestedCandidates)
        {
            if (fs::exists(candidate))
            {
                return candidate.string();
            }
        }

        throw std::runtime_error("Failed to locate requested HDR environment asset: " +
                                 requested.string());
    }

    const std::array<fs::path, 4> candidates = {
        fs::path(defaultEnvironmentFilename),
        fs::path(FIRE_ENGINE_BUILD_ASSET_DIR) / defaultEnvironmentFilename,
        fs::path(FIRE_ENGINE_SOURCE_ASSET_DIR) / defaultEnvironmentFilename,
        fs::path("assets") / defaultEnvironmentFilename,
    };

    for (const auto& candidate : candidates)
    {
        if (fs::exists(candidate))
        {
            return candidate.string();
        }
    }

    throw std::runtime_error(
        "Failed to locate HDR environment asset. Tried: " +
        fs::path(defaultEnvironmentFilename).string() + ", " +
        (fs::path(FIRE_ENGINE_BUILD_ASSET_DIR) / defaultEnvironmentFilename).string() + ", " +
        (fs::path(FIRE_ENGINE_SOURCE_ASSET_DIR) / defaultEnvironmentFilename).string() + ", " +
        (fs::path("assets") / defaultEnvironmentFilename).string());
}

[[nodiscard]] Image loadEnvironmentImage(const std::string& requestedPath, const char* stageName)
{
    try
    {
        std::string environmentPath = resolveSkyboxPath(requestedPath);
        Image hdr = Image::load_from_file(environmentPath);
        if (hdr.pixelType() != ImagePixelType::Float32)
        {
            throw std::runtime_error("decoded image is not HDR float data");
        }
        return hdr;
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("Environment bootstrap failed during ") + stageName +
                                 ": " + e.what());
    }
}

// Builds the kCubemapFaceCount face image views for a cubemap mip level, in the
// canonical +X,-X,+Y,-Y,+Z,-Z order (see cubemap_basis.hpp).
[[nodiscard]] std::array<vk::ImageView, kCubemapFaceCount>
cubemapFaceViews(Resources& resources, TextureHandle cubemap, uint32_t mipLevel = 0)
{
    std::array<vk::ImageView, kCubemapFaceCount> views{};
    for (uint32_t face = 0; face < kCubemapFaceCount; ++face)
    {
        views[face] = resources.vulkanCubemapFaceView(cubemap, face, mipLevel);
    }
    return views;
}

// Records, submits, and blocks on a one-shot transient command buffer. Every
// precompute pass is a blocking one-time submit, so this wraps the shared
// pool/alloc/begin/end/submit/waitIdle boilerplate; `record` receives the
// command buffer in the recording state.
template <typename RecordFn>
void oneTimeSubmit(const Device& device, RecordFn&& record)
{
    vk::CommandPoolCreateInfo poolCi{
        .flags = vk::CommandPoolCreateFlagBits::eTransient,
        .queueFamilyIndex = device.graphicsFamily(),
    };
    vk::raii::CommandPool commandPool(device.device(), poolCi);
    vk::CommandBufferAllocateInfo cmdAi{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    auto cmds = device.device().allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];
    cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    record(*cmd);

    cmd.end();
    vk::CommandBuffer rawCmd = *cmd;
    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &rawCmd};
    device.graphicsQueue().submit(submitInfo);
    device.graphicsQueue().waitIdle();
}

// Renders the six cube faces of `pass` through `pipeline`, binding `ds` and
// pushing the per-face constant returned by makePush(face). `extent` sizes the
// square viewport/scissor. Shared by the equirect→cubemap, irradiance, and
// per-mip prefilter passes.
template <typename MakePush>
void renderCubemapFaces(vk::CommandBuffer cmd, const RenderPass& pass, const Pipeline& pipeline,
                        vk::DescriptorSet ds, uint32_t extent, MakePush makePush)
{
    const vk::Viewport viewport =
        makeFullViewport(static_cast<float>(extent), static_cast<float>(extent));
    const vk::Rect2D renderArea{
        .offset = vk::Offset2D{.x = 0, .y = 0},
        .extent = vk::Extent2D{.width = extent, .height = extent},
    };
    const vk::ClearValue clearColour{
        .color = vk::ClearColorValue{.float32 = {{0.0f, 0.0f, 0.0f, 1.0f}}},
    };

    for (uint32_t face = 0; face < kCubemapFaceCount; ++face)
    {
        auto push = makePush(face);

        vk::RenderPassBeginInfo beginInfo{
            .renderPass = pass.renderPass(),
            .framebuffer = pass.framebuffer(face),
            .renderArea = renderArea,
            .clearValueCount = 1,
            .pClearValues = &clearColour,
        };
        cmd.beginRenderPass(beginInfo, vk::SubpassContents::eInline);
        cmd.setViewport(0, viewport);
        cmd.setScissor(0, renderArea);
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipelineLayout(), 0, ds,
                               {});
        cmd.pushConstants<decltype(push)>(pipeline.pipelineLayout(),
                                          vk::ShaderStageFlagBits::eFragment, 0, push);
        cmd.draw(3, 1, 0, 0);
        cmd.endRenderPass();
    }
}

// Generates mips 1..mipLevels-1 of an already-rendered cubemap (all six layers)
// by successive linear blits from the previous level. Expects mip 0 in
// eShaderReadOnlyOptimal on entry and leaves every level the same way.
void generateCubemapMipChain(vk::CommandBuffer cmd, vk::Image image, uint32_t baseExtent,
                             uint32_t mipLevels)
{
    if (mipLevels <= 1)
    {
        return;
    }

    auto barrier = [&](uint32_t baseMip, uint32_t mipCount, vk::ImageLayout oldLayout,
                       vk::ImageLayout newLayout, vk::AccessFlags srcAccess,
                       vk::AccessFlags dstAccess, vk::PipelineStageFlags srcStage,
                       vk::PipelineStageFlags dstStage)
    {
        vk::ImageMemoryBarrier b{
            .srcAccessMask = srcAccess,
            .dstAccessMask = dstAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange =
                vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                          .baseMipLevel = baseMip,
                                          .levelCount = mipCount,
                                          .baseArrayLayer = 0,
                                          .layerCount = kCubemapFaceCount},
        };
        cmd.pipelineBarrier(srcStage, dstStage, {}, {}, {}, b);
    };

    barrier(0, 1, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead,
            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer);

    int32_t srcExtent = static_cast<int32_t>(baseExtent);
    for (uint32_t mip = 1; mip < mipLevels; ++mip)
    {
        int32_t dstExtent = std::max(1, srcExtent / 2);

        barrier(mip, 1, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {},
                vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer);

        vk::ImageBlit blit{
            .srcSubresource =
                vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                           .mipLevel = mip - 1,
                                           .baseArrayLayer = 0,
                                           .layerCount = kCubemapFaceCount},
            .srcOffsets = {{vk::Offset3D{.x = 0, .y = 0, .z = 0},
                            vk::Offset3D{.x = srcExtent, .y = srcExtent, .z = 1}}},
            .dstSubresource =
                vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                           .mipLevel = mip,
                                           .baseArrayLayer = 0,
                                           .layerCount = kCubemapFaceCount},
            .dstOffsets = {{vk::Offset3D{.x = 0, .y = 0, .z = 0},
                            vk::Offset3D{.x = dstExtent, .y = dstExtent, .z = 1}}},
        };
        cmd.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image,
                      vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        barrier(mip - 1, 1, vk::ImageLayout::eTransferSrcOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eTransferRead,
                vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader);

        barrier(mip, 1, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);

        srcExtent = dstExtent;
    }

    barrier(mipLevels - 1, 1, vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eTransferRead,
            vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader);
}

} // namespace

EnvironmentPrecompute::EnvironmentPrecompute(const Device& device, Resources& resources,
                                             std::string environmentPath)
    : device_{&device},
      resources_{&resources},
      environmentPath_{std::move(environmentPath)}
{
}

void EnvironmentPrecompute::create(vk::DescriptorSetLayout skyboxDescriptorSetLayout,
                                   const Resources::MappedBufferSet& skyboxUbo,
                                   vk::DeviceSize skyboxUboSize,
                                   const Resources::MappedBufferSet& lightUbo,
                                   vk::DeviceSize lightUboSize)
{
    createSkyboxEnvironment(skyboxDescriptorSetLayout, skyboxUbo, skyboxUboSize, lightUbo,
                            lightUboSize);
    createIrradianceEnvironment();
    createPrefilteredEnvironment();
    createBrdfLut();
}

void EnvironmentPrecompute::createSkyboxEnvironment(
    vk::DescriptorSetLayout skyboxDescriptorSetLayout, const Resources::MappedBufferSet& skyboxUbo,
    vk::DeviceSize skyboxUboSize, const Resources::MappedBufferSet& lightUbo,
    vk::DeviceSize lightUboSize)
{
    Image hdr = loadEnvironmentImage(environmentPath_, "skybox cubemap creation");

    SamplerSettings sourceSampler{};
    sourceSampler.wrapS = WrapMode::Repeat;
    sourceSampler.wrapT = WrapMode::ClampToEdge;

    SamplerSettings cubemapSampler{};
    cubemapSampler.wrapS = WrapMode::ClampToEdge;
    cubemapSampler.wrapT = WrapMode::ClampToEdge;

    TextureHandle equirectHandle = resources_->createTexture(hdr, sourceSampler);

    try
    {
        skyboxCubemapHandle_ =
            resources_->createRenderTargetCubemap(kSkyboxCubemapExtent, kSkyboxCubemapMipLevels,
                                                  vk::Format::eR32G32B32A32Sfloat, cubemapSampler);

        RenderPass environmentPass = RenderPass::createOffscreenColour(
            *device_, resources_->textureFormat(skyboxCubemapHandle_));
        Pipeline environmentPipeline(
            *device_, Pipeline::environmentConvertConfig(environmentPass.renderPass()));

        auto faceViews = cubemapFaceViews(*resources_, skyboxCubemapHandle_);
        environmentPass.createColourFramebuffers(*device_, faceViews, kSkyboxCubemapExtent);

        auto captureSets = resources_->descriptors().createSingleImageSamplerDescriptors(
            environmentPipeline.descriptorSetLayout(), equirectHandle);
        const vk::DescriptorSet ds = resources_->vulkanDescriptorSet(captureSets[0]);

        oneTimeSubmit(*device_,
                      [&](vk::CommandBuffer cmd)
        {
            renderCubemapFaces(cmd, environmentPass, environmentPipeline, ds, kSkyboxCubemapExtent,
                               [](uint32_t face)
                               {
                                   EnvironmentCaptureUBO capture{};
                                   capture.faceIndex = static_cast<int>(face);
                                   capture.faceExtent = static_cast<int>(kSkyboxCubemapExtent);
                                   return capture;
                               });
            // Build the source mip chain so the prefilter pass can do
            // mip-weighted importance sampling against this cubemap.
            generateCubemapMipChain(cmd, resources_->vulkanImage(skyboxCubemapHandle_),
                                    kSkyboxCubemapExtent, kSkyboxCubemapMipLevels);
        });
    }
    catch (const std::exception& e)
    {
        resources_->releaseTexture(equirectHandle);
        throw std::runtime_error(std::string("Environment bootstrap failed during skybox cubemap "
                                             "capture: ") +
                                 e.what());
    }

    resources_->releaseTexture(equirectHandle);
    skyboxDescriptorSets_ = resources_->descriptors().createSkyboxDescriptors(
        skyboxDescriptorSetLayout, skyboxUbo, skyboxUboSize, skyboxCubemapHandle_, lightUbo,
        lightUboSize);
}

void EnvironmentPrecompute::createIrradianceEnvironment()
{
    try
    {
        SamplerSettings sampler{};
        sampler.wrapS = WrapMode::ClampToEdge;
        sampler.wrapT = WrapMode::ClampToEdge;

        irradianceCubemapHandle_ = resources_->createRenderTargetCubemap(
            kIrradianceCubemapExtent, 1, vk::Format::eR32G32B32A32Sfloat, sampler);

        RenderPass irradiancePass = RenderPass::createOffscreenColour(
            *device_, resources_->textureFormat(irradianceCubemapHandle_));
        Pipeline irradiancePipeline(
            *device_, Pipeline::irradianceConvolutionConfig(irradiancePass.renderPass()));

        auto faceViews = cubemapFaceViews(*resources_, irradianceCubemapHandle_);
        irradiancePass.createColourFramebuffers(*device_, faceViews, kIrradianceCubemapExtent);

        auto captureSets = resources_->descriptors().createSingleImageSamplerDescriptors(
            irradiancePipeline.descriptorSetLayout(), skyboxCubemapHandle_);
        const vk::DescriptorSet ds = resources_->vulkanDescriptorSet(captureSets[0]);

        oneTimeSubmit(*device_,
                      [&](vk::CommandBuffer cmd)
        {
            renderCubemapFaces(cmd, irradiancePass, irradiancePipeline, ds,
                               kIrradianceCubemapExtent,
                               [](uint32_t face)
                               {
                                   EnvironmentCaptureUBO capture{};
                                   capture.faceIndex = static_cast<int>(face);
                                   capture.faceExtent = static_cast<int>(kIrradianceCubemapExtent);
                                   return capture;
                               });
        });
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("Environment bootstrap failed during irradiance "
                                             "cubemap capture: ") +
                                 e.what());
    }
}

void EnvironmentPrecompute::createPrefilteredEnvironment()
{
    try
    {
        SamplerSettings sampler{};
        sampler.wrapS = WrapMode::ClampToEdge;
        sampler.wrapT = WrapMode::ClampToEdge;

        prefilteredCubemapHandle_ = resources_->createRenderTargetCubemap(
            kPrefilteredCubemapExtent, kPrefilteredCubemapMipLevels,
            vk::Format::eR32G32B32A32Sfloat, sampler);

        RenderPass prefilterPassTemplate = RenderPass::createOffscreenColour(
            *device_, resources_->textureFormat(prefilteredCubemapHandle_));
        Pipeline prefilterPipeline(
            *device_, Pipeline::prefilterEnvironmentConfig(prefilterPassTemplate.renderPass()));
        auto captureSets = resources_->descriptors().createSingleImageSamplerDescriptors(
            prefilterPipeline.descriptorSetLayout(), skyboxCubemapHandle_);

        std::vector<RenderPass> prefilterPasses;
        prefilterPasses.reserve(kPrefilteredCubemapMipLevels);

        uint32_t mipExtent = kPrefilteredCubemapExtent;
        for (uint32_t level = 0; level < kPrefilteredCubemapMipLevels; ++level)
        {
            RenderPass mipPass = RenderPass::createOffscreenColour(
                *device_, resources_->textureFormat(prefilteredCubemapHandle_));
            auto faceViews = cubemapFaceViews(*resources_, prefilteredCubemapHandle_, level);
            mipPass.createColourFramebuffers(*device_, faceViews, mipExtent);
            prefilterPasses.push_back(std::move(mipPass));
            mipExtent = std::max(1u, mipExtent / 2);
        }

        const vk::DescriptorSet ds = resources_->vulkanDescriptorSet(captureSets[0]);

        oneTimeSubmit(*device_,
                      [&](vk::CommandBuffer cmd)
        {
            mipExtent = kPrefilteredCubemapExtent;
            for (uint32_t level = 0; level < kPrefilteredCubemapMipLevels; ++level)
            {
                float roughness = kPrefilteredCubemapMipLevels > 1
                                      ? static_cast<float>(level) /
                                            static_cast<float>(kPrefilteredCubemapMipLevels - 1)
                                      : 0.0f;

                renderCubemapFaces(
                    cmd, prefilterPasses[level], prefilterPipeline, ds, mipExtent,
                    [&](uint32_t face)
                    {
                        EnvironmentPrefilterPushConstants capture{};
                        capture.faceIndex = static_cast<int>(face);
                        capture.faceExtent = static_cast<int>(mipExtent);
                        capture.roughness = roughness;
                        capture.sourceFaceExtent = static_cast<int>(kSkyboxCubemapExtent);
                        capture.sourceMaxMip = static_cast<float>(kSkyboxCubemapMipLevels - 1);
                        return capture;
                    });

                mipExtent = std::max(1u, mipExtent / 2);
            }
        });
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("Environment bootstrap failed during prefiltered "
                                             "cubemap capture: ") +
                                 e.what());
    }
}

void EnvironmentPrecompute::createBrdfLut()
{
    try
    {
        brdfLutHandle_ = resources_->createOffscreenColourTarget({kBrdfLutExtent, kBrdfLutExtent});

        RenderPass brdfPass =
            RenderPass::createOffscreenColour(*device_, resources_->textureFormat(brdfLutHandle_));
        Pipeline brdfPipeline(*device_, Pipeline::brdfIntegrationConfig(brdfPass.renderPass()));

        std::array<vk::ImageView, 1> views = {resources_->vulkanImageView(brdfLutHandle_)};
        brdfPass.createColourFramebuffers(*device_, views, kBrdfLutExtent);

        const vk::Viewport viewport = makeFullViewport(static_cast<float>(kBrdfLutExtent),
                                                       static_cast<float>(kBrdfLutExtent));
        const vk::Rect2D renderArea{
            .offset = vk::Offset2D{.x = 0, .y = 0},
            .extent = vk::Extent2D{.width = kBrdfLutExtent, .height = kBrdfLutExtent},
        };
        const vk::ClearValue clearColour{
            .color = vk::ClearColorValue{.float32 = {{0.0f, 0.0f, 0.0f, 1.0f}}},
        };

        oneTimeSubmit(*device_,
                      [&](vk::CommandBuffer cmd)
        {
            vk::RenderPassBeginInfo beginInfo{
                .renderPass = brdfPass.renderPass(),
                .framebuffer = brdfPass.framebuffer(0),
                .renderArea = renderArea,
                .clearValueCount = 1,
                .pClearValues = &clearColour,
            };
            cmd.beginRenderPass(beginInfo, vk::SubpassContents::eInline);
            cmd.setViewport(0, viewport);
            cmd.setScissor(0, renderArea);
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, brdfPipeline.pipeline());
            cmd.draw(3, 1, 0, 0);
            cmd.endRenderPass();
        });
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("Environment bootstrap failed during BRDF LUT "
                                             "capture: ") +
                                 e.what());
    }
}

} // namespace fire_engine
