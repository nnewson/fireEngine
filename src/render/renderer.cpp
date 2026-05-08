#include <fire_engine/render/renderer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <span>

#include <fire_engine/graphics/image.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/render/environment_precompute.hpp>
#include <fire_engine/render/render_context.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{

namespace
{

struct CameraBasis
{
    Vec3 forward;
    Vec3 right;
    Vec3 up;
};

[[nodiscard]]
CameraBasis cameraBasis(Vec3 cameraPosition, Vec3 cameraTarget)
{
    const Vec3 forward = Vec3::normalise(cameraTarget - cameraPosition);
    const Vec3 worldUp{0.0f, 1.0f, 0.0f};
    const Vec3 right = Vec3::normalise(Vec3::crossProduct(forward, worldUp));
    const Vec3 up = Vec3::crossProduct(right, forward);
    return {forward, right, up};
}

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
    if (slot >= MAX_LIGHTS)
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
Vec3 pickLightUp(Vec3 dir) noexcept
{
    const Vec3 worldUp{0.0f, 1.0f, 0.0f};
    if (std::abs(Vec3::dotProduct(dir, worldUp)) > 0.99f)
    {
        return Vec3{0.0f, 0.0f, 1.0f};
    }
    return worldUp;
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
    const Vec3 up = pickLightUp(lightDir);
    const Vec3 lightPos = center - lightDir * radius;
    const Mat4 view = Mat4::lookAt(lightPos, center, up);
    const Mat4 proj = Mat4::ortho(-radius, radius, -radius, radius, 0.0f, radius * 2.0f);
    return proj * view;
}

} // namespace

Renderer::Renderer(const Window& window, std::string environmentPath, bool debugNormals,
                   bool debugNdotL, bool debugShadow, bool debugShadowDepth, bool noShadows)
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
      debugNormals_(debugNormals),
      debugNdotL_(debugNdotL),
      debugShadow_(debugShadow),
      debugShadowDepth_(debugShadowDepth),
      noShadows_(noShadows)
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
    resources_.irradianceMap(irradianceCubemapHandle_);
    resources_.prefilteredMap(prefilteredCubemapHandle_);
    resources_.brdfLut(brdfLutHandle_);
    imagesInFlight_.assign(swapchain_.images().size(), vk::Fence{});
}

void Renderer::updateLightData(Vec3 cameraPosition, Vec3 cameraTarget, float aspect,
                               std::span<const Lighting> lights)
{
    // Pick a primary directional for CSM. First directional in the gather
    // order wins. Light::worldDirection is the light's forward ray direction
    // (glTF/KHR convention), so the shadow camera must look down that vector.
    // The forward shader negates it separately when it needs surface-to-light.
    const Lighting* primaryDirectional = primaryDirectionalLight(lights);
    const Vec3 lightDir = primaryDirectional != nullptr
                              ? Vec3::normalise(primaryDirectional->worldDirection)
                              : Vec3::normalise(Vec3{1.0f, -1.0f, 1.0f});
    directionalLightDir_ = lightDir;

    // Camera basis + light basis (shared by every cascade fit).
    const float tanHalfFov = std::tan(cameraFovRadians * 0.5f);
    const CameraBasis basis = cameraBasis(cameraPosition, cameraTarget);
    const Vec3 worldUp{0.0f, 1.0f, 0.0f};

    Vec3 lightUp = worldUp;
    if (std::abs(Vec3::dotProduct(lightDir, lightUp)) > 0.99f)
    {
        lightUp = Vec3{0.0f, 0.0f, 1.0f};
    }
    const Vec3 lightRight = Vec3::normalise(Vec3::crossProduct(lightDir, lightUp));
    const Vec3 lightUpOrtho = Vec3::crossProduct(lightRight, lightDir);
    const float shadowMapExtentF = static_cast<float>(shadowMapExtent);

    // Bounding-sphere fit for a single sub-frustum slice. Matches the Stage 1
    // fit — extracted so CSM can reuse it per cascade.
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

        const Vec3 lightPos = snappedCentre - lightDir * (radius + shadowDepthBackExtend);
        const Mat4 lightView = Mat4::lookAt(lightPos, snappedCentre, lightUpOrtho);
        const Mat4 lightProj = Mat4::ortho(-radius, radius, -radius, radius, 0.0f,
                                           2.0f * radius + 2.0f * shadowDepthBackExtend);
        return lightProj * lightView;
    };

    // Log-uniform cascade splits (λ=0.5) — Practical Split Scheme. Keeps close
    // cascades small for near-camera detail while still covering shadowFarPlane.
    float splits[shadowCascadeCount];
    for (uint32_t i = 0; i < shadowCascadeCount; ++i)
    {
        const float p = static_cast<float>(i + 1) / static_cast<float>(shadowCascadeCount);
        const float linear = cameraNearPlane + (shadowFarPlane - cameraNearPlane) * p;
        const float logSplit = cameraNearPlane * std::pow(shadowFarPlane / cameraNearPlane, p);
        splits[i] = 0.5f * (linear + logSplit);
    }

    Mat4 cascadeViewProj[shadowCascadeCount];
    float sliceNear = cameraNearPlane;
    for (uint32_t i = 0; i < shadowCascadeCount; ++i)
    {
        cascadeViewProj[i] = fitCascade(sliceNear, splits[i]);
        sliceNear = splits[i];
    }

    // Reset the shadow matrix array each frame, then write cascades, spots,
    // and point faces in their fixed slots.
    shadowViewProjs_.fill(Mat4::identity());
    for (uint32_t i = 0; i < shadowCascadeCount; ++i)
    {
        shadowViewProjs_[SHADOW_CASCADE_MATRIX_BASE + i] = cascadeViewProj[i];
    }

    LightUBO lightData{};
    for (uint32_t i = 0; i < shadowCascadeCount; ++i)
    {
        lightData.cascadeViewProj[i] = cascadeViewProj[i];
        lightData.cascadeSplits[i] = splits[i];
    }
    for (Mat4& m : lightData.selfShadowViewProj)
    {
        m = Mat4::identity();
    }

    // Pack lights into the UBO array. The primary directional (CSM source)
    // goes first so the shader can branch on i==0 for the shadow lookup.
    int slot = 0;
    int activeSpotCasters = 0;
    int activePointCasters = 0;
    auto assignShadow = [&](int packedSlot, const Lighting& L)
    {
        if (packedSlot < 0)
        {
            return;
        }
        if (L.type == 2 && activeSpotCasters < MAX_SPOT_SHADOW_CASTERS)
        {
            const int shadowIndex = activeSpotCasters++;
            const float fov =
                std::max(2.0f * std::acos(std::clamp(L.outerConeCos, -1.0f, 1.0f)), 0.01f);
            const float far = L.range > 0.0f ? L.range : pointShadowInfiniteRangeFallback;
            const Mat4 proj = Mat4::perspective(fov, 1.0f, pointShadowNearPlane, far);
            const Vec3 dir = Vec3::normalise(L.worldDirection);
            const Vec3 up = pickLightUp(dir);
            const Mat4 view = Mat4::lookAt(L.worldPosition, L.worldPosition + dir, up);
            const Mat4 viewProj = proj * view;
            shadowViewProjs_[SHADOW_SPOT_MATRIX_BASE + shadowIndex] = viewProj;
            lightData.spotViewProj[shadowIndex] = viewProj;
            lightData.lights[packedSlot].cone[2] = static_cast<float>(shadowIndex);
        }
        else if (L.type == 1 && activePointCasters < MAX_POINT_SHADOW_CASTERS)
        {
            const int shadowIndex = activePointCasters++;
            const float far = L.range > 0.0f ? L.range : pointShadowInfiniteRangeFallback;
            const Mat4 proj = Mat4::perspective(0.5f * pi, 1.0f, pointShadowNearPlane, far);
            // Vulkan cubemap face order: +X, -X, +Y, -Y, +Z, -Z. Up vectors
            // match the IBL prefilter convention used elsewhere in the engine
            // so cube sampling stays consistent across face boundaries.
            constexpr Vec3 faceForward[6] = {
                Vec3{1.0f, 0.0f, 0.0f},  Vec3{-1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f},  Vec3{0.0f, 0.0f, -1.0f},
            };
            constexpr Vec3 faceUp[6] = {
                Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f},
                Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, -1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f},
            };
            for (uint32_t face = 0; face < 6; ++face)
            {
                const Mat4 view = Mat4::lookAt(L.worldPosition, L.worldPosition + faceForward[face],
                                               faceUp[face]);
                shadowViewProjs_[SHADOW_POINT_MATRIX_BASE + 6 * shadowIndex + face] = proj * view;
            }
            lightData.lights[packedSlot].cone[2] = static_cast<float>(shadowIndex);
            // Stash the effective range used for shadow projection so the
            // shadow pass push-constant + main-shader compare value agree.
            lightData.lights[packedSlot].direction[3] = far;
            pointCasters_[shadowIndex] = PointShadowCaster{L.worldPosition, far};
        }
    };
    if (primaryDirectional != nullptr)
    {
        const int packed = packLight(lightData, slot, *primaryDirectional);
        assignShadow(packed, *primaryDirectional);
    }
    for (const auto& L : lights)
    {
        if (&L == primaryDirectional)
        {
            continue;
        }
        const int packed = packLight(lightData, slot, L);
        assignShadow(packed, L);
    }
    lightData.lightCount = slot;
    activeSpotCasters_ = activeSpotCasters;
    activePointCasters_ = activePointCasters;
    uint32_t mipLevels = prefilteredCubemapHandle_ != NullTexture
                             ? resources_.textureMipLevels(prefilteredCubemapHandle_)
                             : 1u;
    lightData.iblParams[0] = static_cast<float>(mipLevels > 0 ? mipLevels - 1 : 0);
    lightData.iblParams[1] = diffuseIblStrength;
    lightData.iblParams[2] = specularIblStrength;
    lightData.shadowParams[0] = shadowMinBias;
    lightData.shadowParams[1] = shadowSlopeBias;
    lightData.shadowParams[2] = shadowFilterRadius;
    lightData.shadowParams[3] = shadowNormalOffset;
    lightData.pointSpotShadowParams[0] = pointSpotShadowMinBias;
    lightData.pointSpotShadowParams[1] = pointSpotShadowSlopeBias;
    lightData.environmentParams[0] = skyboxIntensity;
    lightData.environmentParams[1] = environmentShadowStrength;
    lightData.environmentParams[2] =
        debugNormals_ ? 1.0f
                      : (debugNdotL_ ? 2.0f
                                     : (debugShadow_ ? 3.0f
                                                     : (debugShadowDepth_ ? 4.0f : 0.0f)));
    lightData.environmentParams[3] = noShadows_ ? 1.0f : 0.0f;
    lightData_ = lightData;
    std::memcpy(lightUbo_.mapped[currentFrame_], &lightData_, sizeof(lightData_));
}

void Renderer::assignSelfShadowSlots(std::vector<DrawCommand>& drawCommands)
{
    for (Mat4& m : lightData_.selfShadowViewProj)
    {
        m = Mat4::identity();
    }

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
        if (nextSlot >= MAX_SKINNED_SELF_SHADOW_CASTERS)
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
        if (dc.pipeline != lastBoundPipeline)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             resources_.vulkanPipeline(dc.pipeline));
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
        const bool usesForwardPushConstants = dc.pipeline == forwardOpaqueHandle_ ||
                                              dc.pipeline == forwardOpaqueDoubleSidedHandle_ ||
                                              dc.pipeline == forwardBlendHandle_;
        if (usesForwardPushConstants)
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

    const auto extent = swapchain_.extent();
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

    auto lights = scene.gatherLights();
    updateLightData(cameraPosition, cameraTarget, aspect, lights);

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
    DrawBuckets buckets = buildDrawBuckets(drawCommands);
    std::span<const PointShadowCaster> pointCasterSpan{
        pointCasters_.data(), static_cast<std::size_t>(activePointCasters_)};
    shadows_.recordPass(cmd, buckets.shadow, buckets.worldShadow, buckets.selfShadow,
                        activeSpotCasters_, pointCasterSpan);
    recordForwardPass(cmd, buckets);
    if (!buckets.transmissive.empty())
    {
        transmission_.recordPass(cmd, buckets.transmissive);
    }
    postProcessing_.transitionOffscreenForSampling(cmd);
    postProcessing_.recordBloomPasses(cmd);
    postProcessing_.recordPostProcessPass(cmd, *imageIndex, currentFrame_);

    cmd.end();

    submitAndPresent(display, cmd, *imageIndex);
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
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

    vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, renderArea);
}

void Renderer::submitAndPresent(Window& display, vk::CommandBuffer cmd, uint32_t imageIndex)
{
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    auto imageAvail = frame_.imageAvailable(currentFrame_);
    auto renderDone = frame_.renderFinished(imageIndex);
    vk::SubmitInfo si{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &imageAvail,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &renderDone,
    };

    device_.graphicsQueue().submit(si, frame_.inFlightFence(currentFrame_));

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
    const CameraBasis basis = cameraBasis(cameraPosition, cameraTarget);

    constexpr float skyboxFov = cameraFovRadians;
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
    frame_.createRenderFinishedSemaphores(swapchain_.images().size());
    imagesInFlight_.assign(swapchain_.images().size(), vk::Fence{});
}

} // namespace fire_engine
