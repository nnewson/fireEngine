#include <fire_engine/render/renderer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <unordered_map>

#include <fire_engine/graphics/image.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/view_basis.hpp>
#include <fire_engine/render/cubemap_basis.hpp>
#include <fire_engine/render/environment_precompute.hpp>
#include <fire_engine/render/render_context.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]]
const Lighting* primaryDirectionalLight(std::span<const Lighting> lights) noexcept
{
    for (const Lighting& light : lights)
    {
        if (light.type == 0)
        {
            return &light;
        }
    }
    return nullptr;
}

// Returns the slot in lightData.lights[] the light was packed into, or -1 if
// the light array was full and the light was discarded.
int packLight(LightUBO& lightData, int& slot, const Lighting& light) noexcept
{
    if (slot >= kMaxLights)
    {
        return -1;
    }

    int packedSlot = slot;
    LightData& dst = lightData.lights[slot++];
    dst.position[0] = light.worldPosition.x();
    dst.position[1] = light.worldPosition.y();
    dst.position[2] = light.worldPosition.z();
    dst.position[3] = static_cast<float>(light.type);
    dst.direction[0] = light.worldDirection.x();
    dst.direction[1] = light.worldDirection.y();
    dst.direction[2] = light.worldDirection.z();
    dst.direction[3] = light.range;
    dst.colour[0] = light.colour.r();
    dst.colour[1] = light.colour.g();
    dst.colour[2] = light.colour.b();
    dst.colour[3] = light.intensity;
    dst.cone[0] = light.innerConeCos;
    dst.cone[1] = light.outerConeCos;
    dst.cone[2] = -1.0f; // shadow index — overwritten if this light becomes a caster
    return packedSlot;
}

[[nodiscard]]
Mat4 fitSelfShadowMatrix(const Bounds3& bounds, Vec3 lightDir) noexcept
{
    if (!bounds.valid)
    {
        return Mat4::identity();
    }

    const Vec3 center = bounds.center();
    const float halfDiagonal = bounds.extent().magnitude() * 0.5f;
    const float padding = std::max(0.05f, halfDiagonal * 0.05f);
    const float radius = std::max(halfDiagonal + padding, 0.1f);
    const Vec3 up = stableUpForForward(lightDir);
    const Vec3 lightPos = center - lightDir * radius;
    const Mat4 view = Mat4::lookAt(lightPos, center, up);
    const Mat4 proj = Mat4::ortho(-radius, radius, -radius, radius, 0.0f, radius * 2.0f);
    return proj * view;
}

} // namespace

Renderer::Renderer(const Window& window, std::string environmentPath, RendererDebug debug)
    : device_(window),
      swapchain_(device_, window),
      forwardPass_(RenderPass::createForward(device_)),
      pipelineOpaque_(device_, Pipeline::forwardConfig(forwardPass_.renderPass())),
      pipelineOpaqueDoubleSided_(device_,
                                 Pipeline::forwardDoubleSidedConfig(forwardPass_.renderPass())),
      pipelineBlend_(device_, Pipeline::forwardBlendConfig(forwardPass_.renderPass())),
      skyboxPipeline_(device_, Pipeline::skyboxConfig(forwardPass_.renderPass())),
      frame_(device_, swapchain_),
      resources_(device_, pipelineOpaque_),
      postProcessing_(device_, swapchain_, resources_),
      transmission_(device_, swapchain_, resources_, postProcessing_.offscreenColourTarget()),
      shadows_(device_, resources_),
      environmentPath_(std::move(environmentPath)),
      debug_(debug)
{
    swapchain_.createDepthResources(device_);
    forwardPass_.createForwardFramebuffer(
        device_, resources_.vulkanImageView(postProcessing_.offscreenColourTarget()),
        swapchain_.depthView(), swapchain_.extent());
    transmission_.recreate(postProcessing_.offscreenColourTarget());
    forwardOpaqueHandle_ =
        resources_.registerPipeline(pipelineOpaque_.pipeline(), pipelineOpaque_.pipelineLayout());
    forwardOpaqueDoubleSidedHandle_ = resources_.registerPipeline(
        pipelineOpaqueDoubleSided_.pipeline(), pipelineOpaqueDoubleSided_.pipelineLayout());
    forwardBlendHandle_ =
        resources_.registerPipeline(pipelineBlend_.pipeline(), pipelineBlend_.pipelineLayout());
    skyboxPipelineHandle_ =
        resources_.registerPipeline(skyboxPipeline_.pipeline(), skyboxPipeline_.pipelineLayout());
    skyboxUbo_ = resources_.createMappedUniformBuffers(sizeof(SkyboxUBO));
    std::array<uint16_t, 3> skyboxIndices{0, 1, 2};
    skyboxIndexBuffer_ = resources_.createIndexBuffer(skyboxIndices);

    lightUbo_ = resources_.createMappedUniformBuffers(sizeof(LightUBO));
    resources_.lightBuffers(lightUbo_.buffers);
    EnvironmentPrecompute environmentPrecompute{device_, resources_, environmentPath_};
    environmentPrecompute.create(skyboxPipeline_.descriptorSetLayout(), skyboxUbo_,
                                 sizeof(SkyboxUBO), lightUbo_, sizeof(LightUBO));
    skyboxCubemapHandle_ = environmentPrecompute.skyboxCubemap();
    irradianceCubemapHandle_ = environmentPrecompute.irradianceCubemap();
    prefilteredCubemapHandle_ = environmentPrecompute.prefilteredCubemap();
    brdfLutHandle_ = environmentPrecompute.brdfLut();
    skyboxDescSets_ = environmentPrecompute.skyboxDescriptorSets();
    auto& shared = resources_.sharedTextures();
    shared.irradianceMap = irradianceCubemapHandle_;
    shared.prefilteredMap = prefilteredCubemapHandle_;
    shared.brdfLut = brdfLutHandle_;

    // All shared texture handles are populated by this point: shadow maps by
    // the Shadows constructor, sceneColor by transmission_.recreate above,
    // IBL textures by the just-completed environment precompute.
    globalDescSets_ =
        resources_.descriptors().createGlobalDescriptors(buildGlobalDescriptorRequest());

    imagesInFlight_.assign(swapchain_.images().size(), vk::Fence{});
}

GlobalDescriptorRequest Renderer::buildGlobalDescriptorRequest() const
{
    const auto& shared = resources_.sharedTextures();
    return GlobalDescriptorRequest{
        .lightBufs = lightUbo_.buffers,
        .shadowMap = shared.shadowMap,
        .worldShadowMap = shared.worldShadowMap,
        .selfShadowMap = shared.selfShadowMap,
        .spotShadowMap = shared.spotShadowMap,
        .pointShadowMap = shared.pointShadowMap,
        .shadowDebugImage = shared.shadowDebugImage,
        .irradianceMap = shared.irradianceMap,
        .prefilteredMap = shared.prefilteredMap,
        .brdfLut = shared.brdfLut,
        .sceneColor = shared.sceneColor,
    };
}

void Renderer::updateLightData(Vec3 cameraPosition, Vec3 cameraTarget, float aspect,
                               std::span<const Lighting> lights)
{
    // Pick a primary directional for CSM. First directional in the gather
    // order wins. Light::worldDirection is the light's forward ray direction
    // (glTF/KHR convention), so the shadow camera must look down that vector.
    // The forward shader negates it separately when it needs surface-to-light.
    const Lighting* primaryDirectional = primaryDirectionalLight(lights);
    directionalLightDir_ = primaryDirectional != nullptr
                               ? Vec3::normalise(primaryDirectional->worldDirection)
                               : Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f});

    LightUBO lightData{};
    computeShadowCascades(lightData, cameraPosition, cameraTarget, aspect);
    for (Mat4& m : lightData.selfShadowViewProj)
    {
        m = Mat4::identity();
    }

    // Pack lights into the UBO array. The primary directional (CSM source)
    // goes first so the shader can branch on i==0 for the shadow lookup.
    int slot = 0;
    activeSpotCasters_ = 0;
    activePointCasters_ = 0;
    auto packAndAssign = [&](const Lighting& L)
    {
        const int packed = packLight(lightData, slot, L);
        if (packed < 0)
        {
            return;
        }
        if (L.type == 2)
        {
            assignSpotShadow(lightData, packed, L);
        }
        else if (L.type == 1)
        {
            assignPointShadow(lightData, packed, L);
        }
    };
    if (primaryDirectional != nullptr)
    {
        packAndAssign(*primaryDirectional);
    }
    for (const auto& L : lights)
    {
        if (&L == primaryDirectional)
        {
            continue;
        }
        packAndAssign(L);
    }
    lightData.lightCount = slot;

    writeIblAndDebugParams(lightData);
    lightData_ = lightData;
    std::memcpy(lightUbo_.mapped[currentFrame_], &lightData_, sizeof(lightData_));
}

void Renderer::computeShadowCascades(LightUBO& out, Vec3 cameraPosition, Vec3 cameraTarget,
                                     float aspect)
{
    // Camera basis + light basis (shared by every cascade fit).
    const Vec3 lightDir = directionalLightDir_;
    const float tanHalfFov = std::tan(kCameraFovRadians * 0.5f);
    const ViewBasis basis = makeViewBasis(cameraPosition, cameraTarget);
    const Vec3 lightUp = stableUpForForward(lightDir);
    const Vec3 lightRight = normaliseOr(Vec3::crossProduct(lightDir, lightUp), {1.0f, 0.0f, 0.0f});
    const Vec3 lightUpOrtho = normaliseOr(Vec3::crossProduct(lightRight, lightDir), lightUp);
    const float shadowMapExtentF = static_cast<float>(kShadowMapExtent);

    // Bounding-sphere fit for a single sub-frustum slice. Captures the light
    // basis above, called once per cascade.
    auto fitCascade = [&](float sliceNear, float sliceFar) -> Mat4
    {
        const float nearH = tanHalfFov * sliceNear;
        const float nearW = nearH * aspect;
        const float farH = tanHalfFov * sliceFar;
        const float farW = farH * aspect;

        const Vec3 sliceNearCentre = cameraPosition + basis.forward * sliceNear;
        const Vec3 sliceFarCentre = cameraPosition + basis.forward * sliceFar;

        const std::array<Vec3, 8> corners{sliceNearCentre - basis.right * nearW - basis.up * nearH,
                                          sliceNearCentre + basis.right * nearW - basis.up * nearH,
                                          sliceNearCentre + basis.right * nearW + basis.up * nearH,
                                          sliceNearCentre - basis.right * nearW + basis.up * nearH,
                                          sliceFarCentre - basis.right * farW - basis.up * farH,
                                          sliceFarCentre + basis.right * farW - basis.up * farH,
                                          sliceFarCentre + basis.right * farW + basis.up * farH,
                                          sliceFarCentre - basis.right * farW + basis.up * farH};

        Vec3 frustumCentre{0.0f, 0.0f, 0.0f};
        for (const auto& c : corners)
        {
            frustumCentre += c;
        }
        frustumCentre /= 8.0f;

        float radius = 0.0f;
        for (const auto& c : corners)
        {
            radius = std::max(radius, (c - frustumCentre).magnitude());
        }
        radius = std::ceil(radius * 16.0f) / 16.0f;

        const float worldPerTexel = (2.0f * radius) / shadowMapExtentF;
        const float centreU = Vec3::dotProduct(frustumCentre, lightRight);
        const float centreV = Vec3::dotProduct(frustumCentre, lightUpOrtho);
        const float centreW = Vec3::dotProduct(frustumCentre, lightDir);
        const float snappedU = std::floor(centreU / worldPerTexel) * worldPerTexel;
        const float snappedV = std::floor(centreV / worldPerTexel) * worldPerTexel;
        const Vec3 snappedCentre =
            lightRight * snappedU + lightUpOrtho * snappedV + lightDir * centreW;

        const Vec3 lightPos = snappedCentre - lightDir * (radius + kShadowDepthBackExtend);
        const Mat4 lightView = Mat4::lookAt(lightPos, snappedCentre, lightUpOrtho);
        const Mat4 lightProj = Mat4::ortho(-radius, radius, -radius, radius, 0.0f,
                                           2.0f * radius + 2.0f * kShadowDepthBackExtend);
        return lightProj * lightView;
    };

    // Log-uniform cascade splits — Practical Split Scheme. Keeps close cascades
    // small for near-camera detail while still covering kShadowFarPlane.
    float splits[kShadowCascadeCount];
    for (uint32_t i = 0; i < kShadowCascadeCount; ++i)
    {
        const float p = static_cast<float>(i + 1) / static_cast<float>(kShadowCascadeCount);
        const float linear = kCameraNearPlane + (kShadowFarPlane - kCameraNearPlane) * p;
        const float logSplit = kCameraNearPlane * std::pow(kShadowFarPlane / kCameraNearPlane, p);
        splits[i] =
            kShadowCascadeSplitLambda * logSplit + (1.0f - kShadowCascadeSplitLambda) * linear;
    }

    Mat4 cascadeViewProj[kShadowCascadeCount];
    float sliceNear = kCameraNearPlane;
    for (uint32_t i = 0; i < kShadowCascadeCount; ++i)
    {
        cascadeViewProj[i] = fitCascade(sliceNear, splits[i]);
        sliceNear = splits[i];
    }

    // Reset the shadow matrix array each frame, then write cascade slots.
    shadowViewProjs_.fill(Mat4::identity());
    for (uint32_t i = 0; i < kShadowCascadeCount; ++i)
    {
        shadowViewProjs_[kShadowCascadeMatrixBase + i] = cascadeViewProj[i];
        out.cascadeViewProj[i] = cascadeViewProj[i];
        out.cascadeSplits[i] = splits[i];
    }
}

void Renderer::assignSpotShadow(LightUBO& out, int packedSlot, const Lighting& light)
{
    if (activeSpotCasters_ >= kMaxSpotShadowCasters)
    {
        return;
    }
    const int shadowIndex = activeSpotCasters_++;
    const float fov =
        std::max(2.0f * std::acos(std::clamp(light.outerConeCos, -1.0f, 1.0f)), 0.01f);
    const float far = light.range > 0.0f ? light.range : kPointShadowInfiniteRangeFallback;
    const Mat4 proj = Mat4::perspective(fov, 1.0f, kPointShadowNearPlane, far);
    const Vec3 dir = Vec3::normalise(light.worldDirection);
    const Vec3 up = stableUpForForward(dir);
    const Mat4 view = Mat4::lookAt(light.worldPosition, light.worldPosition + dir, up);
    const Mat4 viewProj = proj * view;
    shadowViewProjs_[kShadowSpotMatrixBase + shadowIndex] = viewProj;
    out.spotViewProj[shadowIndex] = viewProj;
    out.lights[packedSlot].cone[2] = static_cast<float>(shadowIndex);
}

void Renderer::assignPointShadow(LightUBO& out, int packedSlot, const Lighting& light)
{
    if (activePointCasters_ >= kMaxPointShadowCasters)
    {
        return;
    }
    const int shadowIndex = activePointCasters_++;
    const float far = light.range > 0.0f ? light.range : kPointShadowInfiniteRangeFallback;
    const Mat4 proj = Mat4::perspective(0.5f * pi, 1.0f, kPointShadowNearPlane, far);
    for (std::size_t face = 0; face < kCubemapFaceCount; ++face)
    {
        const Mat4 view =
            Mat4::lookAt(light.worldPosition, light.worldPosition + kCubemapFaceForward[face],
                         kCubemapFaceUp[face]);
        shadowViewProjs_[kShadowPointMatrixBase + kCubemapFaceCount * shadowIndex + face] =
            proj * view;
    }
    out.lights[packedSlot].cone[2] = static_cast<float>(shadowIndex);
    // Stash the effective range used for shadow projection so the shadow-pass
    // push-constant and the main-shader compare value agree.
    out.lights[packedSlot].direction[3] = far;
    pointCasters_[shadowIndex] = PointShadowCaster{light.worldPosition, far};
}

void Renderer::writeIblAndDebugParams(LightUBO& out) const
{
    const uint32_t mipLevels = prefilteredCubemapHandle_ != NullTexture
                                   ? resources_.textureMipLevels(prefilteredCubemapHandle_)
                                   : 1u;
    out.iblParams[0] = static_cast<float>(mipLevels > 0 ? mipLevels - 1 : 0);
    out.iblParams[1] = kDiffuseIblStrength;
    out.iblParams[2] = kSpecularIblStrength;
    out.shadowParams[0] = kShadowMinBias;
    out.shadowParams[1] = kShadowSlopeBias;
    out.shadowParams[2] = kShadowFilterRadius;
    out.shadowParams[3] = kShadowNormalOffset;
    out.pointSpotShadowParams[0] = kPointSpotShadowMinBias;
    out.pointSpotShadowParams[1] = kPointSpotShadowSlopeBias;
    out.environmentParams[0] = kSkyboxIntensity;
    out.environmentParams[1] = kEnvironmentShadowStrength;
    out.environmentParams[2] = static_cast<float>(debug_.view);
    out.environmentParams[3] = debug_.noShadows ? 1.0f : 0.0f;
}

void Renderer::assignSelfShadowSlots(std::vector<DrawCommand>& drawCommands)
{
    std::unordered_map<uint32_t, int> objectSlots;
    int nextSlot = 0;
    for (const auto& dc : drawCommands)
    {
        if (!dc.hasSkin || !dc.shadowBounds.valid || dc.objectId == 0)
        {
            continue;
        }
        if (objectSlots.contains(dc.objectId))
        {
            continue;
        }
        if (nextSlot >= kMaxSkinnedSelfShadowCasters)
        {
            break;
        }
        objectSlots.emplace(dc.objectId, nextSlot);
        lightData_.selfShadowViewProj[nextSlot] =
            fitSelfShadowMatrix(dc.shadowBounds, directionalLightDir_);
        ++nextSlot;
    }

    for (auto& dc : drawCommands)
    {
        auto it = objectSlots.find(dc.objectId);
        if (it == objectSlots.end())
        {
            dc.selfShadowSlot = -1;
            dc.selfShadowViewProj = Mat4::identity();
            continue;
        }
        dc.selfShadowSlot = it->second;
        dc.selfShadowViewProj = lightData_.selfShadowViewProj[it->second];
    }

    std::memcpy(lightUbo_.mapped[currentFrame_], &lightData_, sizeof(lightData_));
}

Renderer::DrawBuckets Renderer::buildDrawBuckets(const std::vector<DrawCommand>& drawCommands) const
{
    DrawBuckets buckets;
    buckets.opaque.reserve(drawCommands.size());
    for (const auto& dc : drawCommands)
    {
        if (dc.pipeline == shadows_.pipelineHandle())
        {
            buckets.shadow.push_back(dc);
            if (!dc.hasSkin)
            {
                buckets.worldShadow.push_back(dc);
            }
            else if (dc.selfShadowSlot >= 0)
            {
                buckets.selfShadow.push_back(dc);
            }
        }
        else if (dc.pipeline == forwardBlendHandle_)
        {
            buckets.blend.push_back(dc);
        }
        else if (dc.transmissive)
        {
            // KHR_materials_transmission F3 — defer to the second forward
            // sub-pass so the fragment shader can sample sceneColor.
            buckets.transmissive.push_back(dc);
        }
        else
        {
            buckets.opaque.push_back(dc);
        }
    }

    std::sort(buckets.blend.begin(), buckets.blend.end(),
              [](const DrawCommand& a, const DrawCommand& b) { return a.sortDepth > b.sortDepth; });
    std::sort(buckets.transmissive.begin(), buckets.transmissive.end(),
              [](const DrawCommand& a, const DrawCommand& b) { return a.sortDepth > b.sortDepth; });
    return buckets;
}

void Renderer::recordDrawBucket(vk::CommandBuffer cmd, const std::vector<DrawCommand>& bucket,
                                PipelineHandle& lastBoundPipeline) const
{
    for (const auto& dc : bucket)
    {
        const bool isForwardPipeline = dc.pipeline == forwardOpaqueHandle_ ||
                                       dc.pipeline == forwardOpaqueDoubleSidedHandle_ ||
                                       dc.pipeline == forwardBlendHandle_;
        if (dc.pipeline != lastBoundPipeline)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             resources_.vulkanPipeline(dc.pipeline));
            // Set 1 (forward globals) is bound whenever a forward pipeline
            // becomes active. Binding it through a forward layout is required
            // because the skybox pipeline (no globalBindings) shares this
            // bucket and its layout is set-1-incompatible — binding the
            // skybox pipeline disturbs set 1 per Vulkan layout-compatibility
            // rules, so we re-bind on every transition back to a forward
            // pipeline.
            if (isForwardPipeline)
            {
                vk::DescriptorSet globalSet =
                    resources_.vulkanDescriptorSet(globalDescSets_[currentFrame_]);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       resources_.vulkanPipelineLayout(dc.pipeline), 1, globalSet,
                                       {});
            }
            lastBoundPipeline = dc.pipeline;
        }
        if (dc.vertexBuffer != NullBuffer)
        {
            cmd.bindVertexBuffers(0, resources_.vulkanBuffer(dc.vertexBuffer), {vk::DeviceSize{0}});
        }

        vk::IndexType indexType =
            dc.indexType == DrawIndexType::UInt32 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
        cmd.bindIndexBuffer(resources_.vulkanBuffer(dc.indexBuffer), 0, indexType);

        vk::DescriptorSet ds = resources_.vulkanDescriptorSet(dc.descriptorSet);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               resources_.vulkanPipelineLayout(dc.pipeline), 0, ds, {});
        if (isForwardPipeline)
        {
            ForwardPushConstants pc{};
            pc.selfShadowSlot = dc.selfShadowSlot;
            cmd.pushConstants<ForwardPushConstants>(resources_.vulkanPipelineLayout(dc.pipeline),
                                                    vk::ShaderStageFlagBits::eFragment, 0, pc);
        }
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
}

void Renderer::recordForwardPass(vk::CommandBuffer cmd, const DrawBuckets& buckets)
{
    beginRenderPass(cmd);
    auto lastBoundPipeline = PipelineHandle{std::numeric_limits<uint32_t>::max()};
    recordDrawBucket(cmd, buckets.opaque, lastBoundPipeline);
    recordDrawBucket(cmd, buckets.blend, lastBoundPipeline);
    cmd.endRenderPass();
}

void Renderer::updateFrameLighting(SceneGraph& scene, Vec3 cameraPosition, Vec3 cameraTarget)
{
    const auto extent = swapchain_.extent();
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

    auto lights = scene.gatherLights();
    updateLightData(cameraPosition, cameraTarget, aspect, lights);
}

Renderer::DrawBuckets Renderer::collectDrawCommands(vk::CommandBuffer cmd, SceneGraph& scene,
                                                    Vec3 cameraPosition, Vec3 cameraTarget)
{
    std::vector<DrawCommand> drawCommands;
    recordSkybox(cameraPosition, cameraTarget, drawCommands);

    AlphaPipelines pipelines{forwardOpaqueHandle_, forwardOpaqueDoubleSidedHandle_,
                             forwardBlendHandle_};
    RenderContext ctx{device_,
                      swapchain_,
                      frame_,
                      pipelineOpaque_,
                      cmd,
                      currentFrame_,
                      cameraPosition,
                      cameraTarget,
                      &drawCommands,
                      pipelines,
                      shadows_.pipelineHandle(),
                      shadowViewProjs_};
    scene.render(ctx);

    assignSelfShadowSlots(drawCommands);
    return buildDrawBuckets(drawCommands);
}

void Renderer::recordShadowPass(vk::CommandBuffer cmd, const DrawBuckets& buckets)
{
    std::span<const PointShadowCaster> pointCasterSpan{
        pointCasters_.data(), static_cast<std::size_t>(activePointCasters_)};
    shadows_.recordPass(cmd, buckets.shadow, buckets.worldShadow, buckets.selfShadow,
                        activeSpotCasters_, pointCasterSpan);
}

void Renderer::recordTransmissionPass(vk::CommandBuffer cmd, const DrawBuckets& buckets)
{
    if (buckets.transmissive.empty())
    {
        return;
    }
    transmission_.recordPass(cmd, buckets.transmissive,
                             resources_.vulkanDescriptorSet(globalDescSets_[currentFrame_]));
}

void Renderer::recordPostProcessing(vk::CommandBuffer cmd, uint32_t imageIndex)
{
    postProcessing_.transitionOffscreenForSampling(cmd);
    postProcessing_.recordBloomPasses(cmd);
    postProcessing_.recordPostProcessPass(cmd, imageIndex, currentFrame_);
}

void Renderer::drawFrame(Window& display, SceneGraph& scene, Vec3 cameraPosition, Vec3 cameraTarget)
{
    auto imageIndex = acquireNextImage(display);
    if (!imageIndex)
    {
        return;
    }

    auto cmd = frame_.commandBuffer(currentFrame_);
    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});

    updateFrameLighting(scene, cameraPosition, cameraTarget);
    DrawBuckets buckets = collectDrawCommands(cmd, scene, cameraPosition, cameraTarget);

    recordShadowPass(cmd, buckets);
    recordForwardPass(cmd, buckets);
    recordTransmissionPass(cmd, buckets);
    recordPostProcessing(cmd, *imageIndex);

    cmd.end();
    submitAndPresent(display, cmd, *imageIndex);

    currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
}

std::optional<uint32_t> Renderer::acquireNextImage(Window& display)
{
    auto& dev = device_.device();

    (void)dev.waitForFences(frame_.inFlightFence(currentFrame_), vk::True, UINT64_MAX);

    auto [acquireResult, imageIndex] = (*dev).acquireNextImageKHR(
        swapchain_.swapchain(), UINT64_MAX, frame_.imageAvailable(currentFrame_));
    if (acquireResult == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapchain(display);
        return std::nullopt;
    }
    if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR)
    {
        throw std::runtime_error("failed to acquire swap chain image");
    }

    vk::Fence currentFrameFence = frame_.inFlightFence(currentFrame_);
    if (imagesInFlight_[imageIndex])
    {
        (void)dev.waitForFences(imagesInFlight_[imageIndex], vk::True, UINT64_MAX);
    }

    dev.resetFences(currentFrameFence);
    imagesInFlight_[imageIndex] = currentFrameFence;
    return imageIndex;
}

void Renderer::beginRenderPass(vk::CommandBuffer cmd)
{
    auto extent = swapchain_.extent();
    std::array<vk::ClearValue, 2> clears = {
        vk::ClearValue{.color = vk::ClearColorValue{.float32 = {{0.02f, 0.02f, 0.02f, 1.0f}}}},
        vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0}},
    };

    vk::Rect2D renderArea{
        .offset = vk::Offset2D{.x = 0, .y = 0},
        .extent = extent,
    };
    vk::RenderPassBeginInfo rpBegin{
        .renderPass = forwardPass_.renderPass(),
        .framebuffer = forwardPass_.framebuffer(0),
        .renderArea = renderArea,
        .clearValueCount = static_cast<uint32_t>(clears.size()),
        .pClearValues = clears.data(),
    };

    cmd.beginRenderPass(rpBegin, vk::SubpassContents::eInline);

    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);
}

void Renderer::submitAndPresent(Window& display, vk::CommandBuffer cmd, uint32_t imageIndex)
{
    auto imageAvail = frame_.imageAvailable(currentFrame_);
    auto renderDone = frame_.renderFinished(imageIndex);
    vk::SemaphoreSubmitInfo waitInfo{
        .semaphore = imageAvail,
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    };
    vk::SemaphoreSubmitInfo signalInfo{
        .semaphore = renderDone,
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    };
    vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = cmd};
    vk::SubmitInfo2 si{
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalInfo,
    };

    device_.graphicsQueue().submit2(si, frame_.inFlightFence(currentFrame_));

    auto swapchain = swapchain_.swapchain();
    vk::PresentInfoKHR pi{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderDone,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &imageIndex,
    };

    vk::Result presentResult = device_.presentQueue().presentKHR(pi);
    if (presentResult == vk::Result::eErrorOutOfDateKHR ||
        presentResult == vk::Result::eSuboptimalKHR || display.framebufferResized())
    {
        display.framebufferResized(false);
        recreateSwapchain(display);
    }
    else if (presentResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to present swap chain image");
    }
}

void Renderer::recordSkybox(Vec3 cameraPosition, Vec3 cameraTarget,
                            std::vector<DrawCommand>& drawCommands)
{
    const ViewBasis basis = makeViewBasis(cameraPosition, cameraTarget);

    constexpr float skyboxFov = kCameraFovRadians;
    auto extent = swapchain_.extent();
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    float tanHalfFov = std::tan(skyboxFov * 0.5f);

    SkyboxUBO data{};
    data.cameraForward[0] = basis.forward.x();
    data.cameraForward[1] = basis.forward.y();
    data.cameraForward[2] = basis.forward.z();
    data.cameraRight[0] = basis.right.x();
    data.cameraRight[1] = basis.right.y();
    data.cameraRight[2] = basis.right.z();
    data.cameraUp[0] = basis.up.x();
    data.cameraUp[1] = basis.up.y();
    data.cameraUp[2] = basis.up.z();
    data.viewParams[0] = tanHalfFov;
    data.viewParams[1] = aspect;
    std::memcpy(skyboxUbo_.mapped[currentFrame_], &data, sizeof(data));

    DrawCommand dc;
    dc.vertexBuffer = NullBuffer;
    dc.indexBuffer = skyboxIndexBuffer_;
    dc.indexCount = 3;
    dc.indexType = DrawIndexType::UInt16;
    dc.descriptorSet = skyboxDescSets_[currentFrame_];
    dc.pipeline = skyboxPipelineHandle_;
    drawCommands.push_back(dc);
}

void Renderer::recreateSwapchain(const Window& display)
{
    auto [w, h] = display.framebufferSize();
    while (w == 0 || h == 0)
    {
        std::tie(w, h) = display.framebufferSize();
        Window::waitEvents();
    }
    waitIdle();

    frame_.destroyRenderFinishedSemaphores();

    swapchain_.recreate(device_, display);
    postProcessing_.recreate();
    forwardPass_.createForwardFramebuffer(
        device_, resources_.vulkanImageView(postProcessing_.offscreenColourTarget()),
        swapchain_.depthView(), swapchain_.extent());
    transmission_.recreate(postProcessing_.offscreenColourTarget());
    // Rewrite the forward-globals (set 1) descriptors so they reference the
    // sampler/view from the freshly recreated sceneColor target (and any
    // other recreated shared texture) instead of the destroyed ones.
    resources_.descriptors().updateGlobalDescriptors(globalDescSets_,
                                                     buildGlobalDescriptorRequest());
    frame_.createRenderFinishedSemaphores(swapchain_.images().size());
    imagesInFlight_.assign(swapchain_.images().size(), vk::Fence{});
}

} // namespace fire_engine
