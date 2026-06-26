#include <fire_engine/render/renderer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>

#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/image.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/view_basis.hpp>
#include <fire_engine/render/cubemap_basis.hpp>
#include <fire_engine/render/environment_precompute.hpp>
#include <fire_engine/render/render_context.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{

namespace
{

// Single-subresource layout transition through synchronization2, used to cycle
// the forward HDR target and the shared depth image between attachment and
// shader-read layouts (dynamic rendering does no implicit transitions).
void forwardImageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageAspectFlags aspect,
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

// Radical-inverse Halton sample in the given base. Used to build the
// low-discrepancy sub-pixel jitter sequence for TAA.
[[nodiscard]]
float halton(uint32_t index, uint32_t base) noexcept
{
    float result = 0.0f;
    float invBase = 1.0f / static_cast<float>(base);
    float fraction = invBase;
    while (index > 0)
    {
        result += static_cast<float>(index % base) * fraction;
        index /= base;
        fraction *= invBase;
    }
    return result;
}

// Sub-pixel jitter offset in clip space for sample `index` at `extent`, applied
// to projection entries m[0,2]/m[1,2]. Halton(2,3) recentred to [-0.5, 0.5] and
// scaled to ±kJitterPixelRadius pixels (k = 4 * radius maps the recentred sample
// to a ±radius-pixel NDC offset). The sign is cosmetic — the velocity buffer is
// jitter-free, so the jitter cancels in accumulation.
[[nodiscard]]
std::pair<float, float> taaJitterOffset(uint32_t index, vk::Extent2D extent) noexcept
{
    constexpr float kJitterPixelRadius = 0.5f;
    constexpr float k = 4.0f * kJitterPixelRadius;
    const float jx = (halton(index + 1, 2) - 0.5f) * k / static_cast<float>(extent.width);
    const float jy = (halton(index + 1, 3) - 0.5f) * k / static_cast<float>(extent.height);
    return {jx, jy};
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
      pipelineOpaque_(device_, Pipeline::forwardConfig()),
      pipelineBlend_(device_, Pipeline::forwardBlendConfig()),
      skyboxPipeline_(device_, Pipeline::skyboxConfig()),
      depthPrepassPipeline_(device_, Pipeline::depthPrepassConfig()),
      frame_(device_, swapchain_),
      resources_(device_, pipelineOpaque_),
      postProcessing_(device_, swapchain_, resources_),
      transmission_(swapchain_, resources_, postProcessing_.offscreenColourTarget()),
      shadows_(device_, resources_),
      particles_(device_, swapchain_, resources_, postProcessing_.offscreenColourTarget()),
      taa_(device_, swapchain_, resources_, postProcessing_.offscreenColourTarget()),
      ssao_(device_, swapchain_, resources_),
      debugDraw_(device_, swapchain_, resources_),
      softBody_(device_, resources_),
      profiler_(device_),
      overlay_(device_, swapchain_, window, debug.overlayVisible),
      environmentPath_(std::move(environmentPath))
{
    // Seed the live tunables from the CLI debug flags so they carry over as the
    // overlay's initial state; everything else defaults from constants.hpp.
    tunables_.taaEnabled = debug.taa;
    tunables_.debugView = debug.view;
    tunables_.noShadows = debug.noShadows;
    tunables_.debugDrawAabbs = debug.physicsDebug;
    tunables_.debugDrawColliders = debug.physicsDebug;
    tunables_.debugDrawContacts = debug.physicsDebug;

    swapchain_.createDepthResources(device_);
    transmission_.recreate(postProcessing_.offscreenColourTarget(), taa_.velocityTarget());
    // Bind the now-created scene-depth image into the particle render set.
    particles_.recreate(postProcessing_.offscreenColourTarget());
    // Create the AO target + bind scene depth into the SSAO set (depth exists now).
    ssao_.recreate();
    resources_.sharedTextures().ssaoMap = ssao_.aoTarget();
    forwardOpaqueHandle_ =
        resources_.registerPipeline(pipelineOpaque_.pipeline(), pipelineOpaque_.pipelineLayout());
    forwardBlendHandle_ =
        resources_.registerPipeline(pipelineBlend_.pipeline(), pipelineBlend_.pipelineLayout());
    skyboxPipelineHandle_ =
        resources_.registerPipeline(skyboxPipeline_.pipeline(), skyboxPipeline_.pipelineLayout());
    depthPrepassHandle_ = resources_.registerPipeline(depthPrepassPipeline_.pipeline(),
                                                      depthPrepassPipeline_.pipelineLayout());
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

    imageTimelineValue_.assign(swapchain_.images().size(), 0);
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
        .ssaoMap = shared.ssaoMap,
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
        // The primary directional is packed first (slot 0); the overlay's sun
        // slider scales its intensity (colour.a) live.
        if (slot > 0)
        {
            lightData.lights[0].colour[3] *= tunables_.directionalIntensityScale;
        }
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
    out.iblParams[1] = tunables_.diffuseIbl;
    out.iblParams[2] = tunables_.specularIbl;
    out.shadowParams[0] = kShadowMinBias;
    out.shadowParams[1] = kShadowSlopeBias;
    out.shadowParams[2] = kShadowFilterRadius;
    out.shadowParams[3] = kShadowNormalOffset;
    out.pointSpotShadowParams[0] = kPointSpotShadowMinBias;
    out.pointSpotShadowParams[1] = kPointSpotShadowSlopeBias;
    out.environmentParams[0] = kSkyboxIntensity;
    out.environmentParams[1] = kEnvironmentShadowStrength;
    out.environmentParams[2] = static_cast<float>(tunables_.debugView);
    out.environmentParams[3] = tunables_.noShadows ? 1.0f : 0.0f;
}

void Renderer::assignSelfShadowSlots(std::span<DrawCommand> drawCommands)
{
    selfShadowSlotsScratch_.clear();
    selfShadowSlotsScratch_.reserve(
        std::min<std::size_t>(drawCommands.size(), kMaxSkinnedSelfShadowCasters));
    int nextSlot = 0;
    for (const auto& dc : drawCommands)
    {
        if (!dc.hasSkin || !dc.shadowBounds.valid || dc.objectId == 0)
        {
            continue;
        }
        if (selfShadowSlotsScratch_.contains(dc.objectId))
        {
            continue;
        }
        if (nextSlot >= kMaxSkinnedSelfShadowCasters)
        {
            break;
        }
        selfShadowSlotsScratch_.emplace(dc.objectId, nextSlot);
        lightData_.selfShadowViewProj[nextSlot] =
            fitSelfShadowMatrix(dc.shadowBounds, directionalLightDir_);
        ++nextSlot;
    }

    for (auto& dc : drawCommands)
    {
        auto it = selfShadowSlotsScratch_.find(dc.objectId);
        if (it == selfShadowSlotsScratch_.end())
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

void Renderer::clearDrawBuckets(DrawBuckets& buckets) noexcept
{
    buckets.shadow.clear();
    buckets.worldShadow.clear();
    buckets.selfShadow.clear();
    buckets.opaque.clear();
    buckets.blend.clear();
    buckets.transmissive.clear();
}

void Renderer::buildDrawBuckets(std::span<const DrawCommand> drawCommands,
                                DrawBuckets& buckets) const
{
    clearDrawBuckets(buckets);
    buckets.shadow.reserve(drawCommands.size());
    buckets.worldShadow.reserve(drawCommands.size());
    buckets.selfShadow.reserve(
        std::min<std::size_t>(drawCommands.size(), kMaxSkinnedSelfShadowCasters));
    buckets.opaque.reserve(drawCommands.size());
    buckets.blend.reserve(drawCommands.size());
    buckets.transmissive.reserve(drawCommands.size());
    // Camera-frustum cull the forward (non-shadow) draws; shadow draws are culled
    // per-cascade/-light in the shadow pass. A draw with invalid bounds (the skybox) is
    // never culled. The shadow buckets carry every caster — the shadow pass filters them.
    const bool cull = tunables_.cullingEnabled;
    const Frustum cameraFrustum = Frustum::fromViewProj(currentViewProj_);
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
            continue;
        }

        if (cull && !cameraFrustum.intersects(dc.shadowBounds))
        {
            continue;
        }
        if (dc.pipeline == forwardBlendHandle_)
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
}

void Renderer::recordDrawBucket(vk::CommandBuffer cmd, std::span<const DrawCommand> bucket,
                                PipelineHandle& lastBoundPipeline) const
{
    for (const auto& dc : bucket)
    {
        const bool isForwardPipeline =
            dc.pipeline == forwardOpaqueHandle_ || dc.pipeline == forwardBlendHandle_;
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
                // Set 2 — bindless materials (global texture array + materials SSBO).
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       resources_.vulkanPipelineLayout(dc.pipeline), 2,
                                       resources_.bindlessDescriptorSet(), {});
            }
            lastBoundPipeline = dc.pipeline;
        }
        // The merged opaque/double-sided pipeline declares cull mode dynamic;
        // set it per draw. Blend (static eNone) and skybox pipelines must not be
        // touched here — they did not declare the dynamic state.
        if (dc.pipeline == forwardOpaqueHandle_)
        {
            cmd.setCullMode(dc.doubleSided ? vk::CullModeFlagBits::eNone
                                           : vk::CullModeFlagBits::eBack);
        }
        if (dc.vertexBuffer != NullBuffer)
        {
            cmd.bindVertexBuffers(0, resources_.vulkanBuffer(dc.vertexBuffer), {vk::DeviceSize{0}});
        }

        vk::IndexType indexType =
            dc.indexType == DrawIndexType::UInt32 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
        cmd.bindIndexBuffer(resources_.vulkanBuffer(dc.indexBuffer), 0, indexType);

        if (isForwardPipeline)
        {
            // Forward set 0 is pushed inline (VK_KHR_push_descriptor) — no
            // per-object descriptor set — plus the per-draw push constants.
            // Skybox (also in this bucket) keeps its allocated set 0.
            pushForwardObjectDescriptors(cmd, resources_,
                                         resources_.vulkanPipelineLayout(dc.pipeline), dc);
            ForwardPushConstants pc{};
            pc.selfShadowSlot = dc.selfShadowSlot;
            pc.materialIndex = dc.materialIndex;
            cmd.pushConstants<ForwardPushConstants>(resources_.vulkanPipelineLayout(dc.pipeline),
                                                    vk::ShaderStageFlagBits::eFragment, 0, pc);
        }
        else
        {
            vk::DescriptorSet ds = resources_.vulkanDescriptorSet(dc.descriptorSet);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                   resources_.vulkanPipelineLayout(dc.pipeline), 0, ds, {});
        }
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
}

void Renderer::recordDepthPrepass(vk::CommandBuffer cmd, const DrawBuckets& buckets)
{
    const auto extent = swapchain_.extent();
    vk::Rect2D renderArea{.offset = vk::Offset2D{.x = 0, .y = 0}, .extent = extent};

    // Depth rests in DepthStencilReadOnlyOptimal between frames (last frame's
    // particle pass left it sampled) or Undefined on the first frame; we clear +
    // overwrite, so discard via Undefined into the depth-attachment layout.
    forwardImageBarrier(cmd, swapchain_.depthImage(), vk::ImageAspectFlagBits::eDepth,
                        vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eDepthStencilAttachmentOptimal,
                        vk::PipelineStageFlagBits2::eLateFragmentTests, {},
                        vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                        vk::AccessFlagBits2::eDepthStencilAttachmentWrite);

    const vk::ClearValue depthClear{.depthStencil =
                                        vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0}};
    vk::RenderingAttachmentInfo depth{
        .imageView = swapchain_.depthView(),
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = depthClear,
    };
    cmd.beginRendering(makeRenderingInfo(renderArea, {}, &depth));
    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);

    const vk::PipelineLayout layout = resources_.vulkanPipelineLayout(depthPrepassHandle_);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                     resources_.vulkanPipeline(depthPrepassHandle_));
    for (const auto& dc : buckets.opaque)
    {
        // buckets.opaque also carries the skybox (fullscreen triangle, no depth /
        // no per-object set 0) — only real forward-opaque geometry belongs here.
        if (dc.pipeline != forwardOpaqueHandle_)
        {
            continue;
        }
        cmd.setCullMode(dc.doubleSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
        if (dc.vertexBuffer != NullBuffer)
        {
            cmd.bindVertexBuffers(0, resources_.vulkanBuffer(dc.vertexBuffer), {vk::DeviceSize{0}});
        }
        vk::IndexType indexType =
            dc.indexType == DrawIndexType::UInt32 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
        cmd.bindIndexBuffer(resources_.vulkanBuffer(dc.indexBuffer), 0, indexType);
        pushForwardObjectDescriptors(cmd, resources_, layout, dc);
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
    cmd.endRendering();
}

void Renderer::recordSsaoPass(vk::CommandBuffer cmd)
{
    ssao_.recordPass(cmd, currentFrame_);
}

void Renderer::recordDebugDrawPass(vk::CommandBuffer cmd)
{
    if (!physicsDebugWanted())
    {
        return;
    }
    debugDraw_.record(cmd, postProcessing_.offscreenColourTarget(), currentViewProj_, physicsDebug_,
                      tunables_, currentFrame_);
}

void Renderer::recordForwardPass(vk::CommandBuffer cmd, const DrawBuckets& buckets)
{
    beginForwardRendering(cmd);
    auto lastBoundPipeline = PipelineHandle{std::numeric_limits<uint32_t>::max()};
    recordDrawBucket(cmd, buckets.opaque, lastBoundPipeline);
    recordDrawBucket(cmd, buckets.blend, lastBoundPipeline);
    endForwardRendering(cmd);
}

void Renderer::updateFrameLighting(SceneGraph& scene, Vec3 cameraPosition, Vec3 cameraTarget)
{
    const auto extent = swapchain_.extent();
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

    scene.gatherLights(lightScratch_);
    updateLightData(cameraPosition, cameraTarget, aspect, lightScratch_);
}

const Renderer::DrawBuckets& Renderer::collectDrawCommands(vk::CommandBuffer cmd, SceneGraph& scene,
                                                           Vec3 cameraPosition, Vec3 cameraTarget)
{
    drawCommandScratch_.clear();
    recordSkybox(cameraPosition, cameraTarget, drawCommandScratch_);

    AlphaPipelines pipelines{forwardOpaqueHandle_, forwardBlendHandle_};
    RenderContext ctx{device_,
                      swapchain_,
                      frame_,
                      pipelineOpaque_,
                      cmd,
                      currentFrame_,
                      cameraPosition,
                      cameraTarget,
                      view_,
                      jitteredProj_,
                      currentViewProj_,
                      previousViewProj_,
                      &drawCommandScratch_,
                      pipelines,
                      shadows_.pipelineHandle(),
                      shadowViewProjs_};

    // Coarse pre-cull: skip draw-building for rigid nodes outside every frustum (camera
    // plus all shadow casters). The union is a superset of what buildDrawBuckets /
    // shadows_ keep per pass, so a node dropped here is never wanted downstream. Inactive
    // shadow slots are identity matrices — harmless degenerate frustums that can only add
    // visibility, never remove it. Disabled toggle leaves culledNodes null (render all).
    frustumScratch_.clear();
    if (tunables_.cullingEnabled)
    {
        frustumScratch_.reserve(1 + shadowViewProjs_.size());
        frustumScratch_.push_back(Frustum::fromViewProj(currentViewProj_));
        for (const Mat4& shadowViewProj : shadowViewProjs_)
        {
            frustumScratch_.push_back(Frustum::fromViewProj(shadowViewProj));
        }
        ctx.culledNodes = &scene.cull(frustumScratch_);
        stats_.trackedNodes = static_cast<int>(scene.culler().trackedCount());
        stats_.culledNodes = static_cast<int>(scene.culler().culledCount());
    }
    else
    {
        stats_.trackedNodes = 0;
        stats_.culledNodes = 0;
    }

    scene.render(ctx);

    assignSelfShadowSlots(drawCommandScratch_);
    buildDrawBuckets(drawCommandScratch_, drawBucketsScratch_);
    return drawBucketsScratch_;
}

void Renderer::recordShadowPass(vk::CommandBuffer cmd, const DrawBuckets& buckets)
{
    std::span<const PointShadowCaster> pointCasterSpan{
        pointCasters_.data(), static_cast<std::size_t>(activePointCasters_)};
    shadows_.recordPass(cmd, buckets.shadow, buckets.worldShadow, buckets.selfShadow,
                        activeSpotCasters_, pointCasterSpan, shadowViewProjs_,
                        tunables_.cullingEnabled);
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
    profiler_.begin(cmd, currentFrame_, ProfilePass::Bloom);
    postProcessing_.recordBloomPasses(cmd);
    profiler_.end(cmd, currentFrame_, ProfilePass::Bloom);
    profiler_.begin(cmd, currentFrame_, ProfilePass::Post);
    postProcessing_.recordPostProcessPass(cmd, imageIndex, currentFrame_, tunables_.bloomStrength);
    profiler_.end(cmd, currentFrame_, ProfilePass::Post);
}

void Renderer::drawFrame(Window& display, SceneGraph& scene, Vec3 cameraPosition, Vec3 cameraTarget,
                         float dt)
{
    auto imageIndex = acquireNextImage(display);
    if (!imageIndex)
    {
        return;
    }

    // Read back the GPU timings written a ring-cycle ago into this slot (the
    // acquire timeline-wait guarantees that frame completed) and the wall-clock
    // CPU time.
    profiler_.resolve(currentFrame_, stats_);
    stats_.cpuFrameMs = dt * 1000.0f;

    // Start the ImGui frame before recording. GLFW events were already polled
    // this frame (Input::update), so the overlay's input state is current.
    overlay_.beginFrame();
    overlay_.buildUi(stats_, tunables_);

    auto cmd = frame_.commandBuffer(currentFrame_);
    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});
    // Reset this frame's timestamp range before any pass writes into it.
    profiler_.beginFrame(cmd, currentFrame_);

    // Soft-body (cloth) solve runs first: it writes solved positions + normals
    // into the cloth vertex buffers that the shadow + forward passes then read.
    const ClothSimParams clothParams{
        .substeps = static_cast<uint32_t>(std::max(1, tunables_.clothSubsteps)),
        .complianceScale = tunables_.clothComplianceScale,
        .damping = tunables_.clothDamping,
        .gravity = tunables_.clothGravity,
        .wind = {tunables_.clothWind[0], tunables_.clothWind[1], tunables_.clothWind[2]},
    };
    softBody_.recordSolve(cmd, dt, currentFrame_, clothColliders_, clothParams);

    // Per-frame camera matrices. The forward pass rasterises with jitteredProj_
    // (TAA sub-pixel jitter); currentViewProj_ is jitter-free so motion vectors
    // are independent of the jitter (it cancels in the resolve accumulation).
    const auto extent = swapchain_.extent();
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    view_ = Mat4::lookAt(cameraPosition, cameraTarget, {0.0f, 1.0f, 0.0f});
    const Mat4 unjitteredProj =
        Mat4::perspective(kCameraFovRadians, aspect, kCameraNearPlane, kCameraFarPlane);
    jitteredProj_ = unjitteredProj;
    if (tunables_.taaEnabled)
    {
        const auto [jx, jy] = taaJitterOffset(taaJitterIndex_, extent);
        jitteredProj_[0, 2] += jx;
        jitteredProj_[1, 2] += jy;
        taaJitterIndex_ = (taaJitterIndex_ + 1) % kTaaJitterSamples;
    }
    currentViewProj_ = unjitteredProj * view_;

    updateFrameLighting(scene, cameraPosition, cameraTarget);
    const DrawBuckets& buckets = collectDrawCommands(cmd, scene, cameraPosition, cameraTarget);

    // Particles render un-jittered (after TAA); feed them the plain proj. The
    // overlay's emitter scales are applied to a local copy of the gather.
    scene.gatherEmitters(emitterScratch_);
    for (auto& e : emitterScratch_)
    {
        e.spawnRate *= tunables_.particleRateScale;
        e.lifetime *= tunables_.particleLifetimeScale;
        e.size *= tunables_.particleSizeScale;
    }
    particles_.update(emitterScratch_, view_, unjitteredProj, dt, currentFrame_);

    profiler_.begin(cmd, currentFrame_, ProfilePass::Shadow);
    recordShadowPass(cmd, buckets);
    profiler_.end(cmd, currentFrame_, ProfilePass::Shadow);

    profiler_.begin(cmd, currentFrame_, ProfilePass::DepthPrepass);
    recordDepthPrepass(cmd, buckets);
    profiler_.end(cmd, currentFrame_, ProfilePass::DepthPrepass);

    // SSAO + contact shadows from the prepass depth; the forward pass samples the
    // AO target. Sun direction is rotated into view space for contact shadows.
    profiler_.begin(cmd, currentFrame_, ProfilePass::Ssao);
    const Vec4 sunView = view_ * Vec4{directionalLightDir_.x(), directionalLightDir_.y(),
                                      directionalLightDir_.z(), 0.0f};
    ssao_.update(jitteredProj_, static_cast<Vec3>(sunView), tunables_, currentFrame_);
    recordSsaoPass(cmd);
    profiler_.end(cmd, currentFrame_, ProfilePass::Ssao);

    profiler_.begin(cmd, currentFrame_, ProfilePass::Forward);
    recordForwardPass(cmd, buckets);
    profiler_.end(cmd, currentFrame_, ProfilePass::Forward);

    profiler_.begin(cmd, currentFrame_, ProfilePass::Transmission);
    recordTransmissionPass(cmd, buckets);
    profiler_.end(cmd, currentFrame_, ProfilePass::Transmission);

    // TAA resolve: reproject + accumulate history into the offscreen HDR target
    // before particles (which render un-jittered and stay out of the history).
    if (tunables_.taaEnabled)
    {
        profiler_.begin(cmd, currentFrame_, ProfilePass::Taa);
        taa_.recordResolve(cmd, currentFrame_, tunables_.taaHistoryBlend, tunables_.taaSharpen);
        profiler_.end(cmd, currentFrame_, ProfilePass::Taa);
    }

    profiler_.begin(cmd, currentFrame_, ProfilePass::Particles);
    recordParticlePass(cmd);
    profiler_.end(cmd, currentFrame_, ProfilePass::Particles);

    profiler_.begin(cmd, currentFrame_, ProfilePass::DebugDraw);
    recordDebugDrawPass(cmd);
    profiler_.end(cmd, currentFrame_, ProfilePass::DebugDraw);

    recordPostProcessing(cmd, *imageIndex);
    // Post-process leaves the swap image in ColorAttachmentOptimal; the overlay
    // draws over it, then we transition to present.
    overlay_.record(cmd, *swapchain_.imageViews()[*imageIndex], swapchain_.extent());
    transitionSwapchainToPresent(cmd, *imageIndex);

    cmd.end();
    submitAndPresent(display, cmd, *imageIndex);

    previousViewProj_ = currentViewProj_;
    currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
}

void Renderer::recordParticlePass(vk::CommandBuffer cmd)
{
    particles_.recordSimulate(cmd, currentFrame_);
    particles_.recordRender(cmd, currentFrame_);
}

std::optional<uint32_t> Renderer::acquireNextImage(Window& display)
{
    auto& dev = device_.device();

    // Wait until the last submit that used this frame-in-flight slot finished, so
    // its command buffer + per-frame UBOs are safe to overwrite.
    waitTimeline(frameTimelineValue_[currentFrame_]);

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

    // Wait until any earlier frame still rendering to this swapchain image is done
    // (matters when swapchain image count != frames in flight).
    waitTimeline(imageTimelineValue_[imageIndex]);
    return imageIndex;
}

void Renderer::waitTimeline(uint64_t value) const
{
    if (value == 0)
    {
        return;
    }
    vk::Semaphore sem = frame_.timeline();
    vk::SemaphoreWaitInfo wi{
        .semaphoreCount = 1,
        .pSemaphores = &sem,
        .pValues = &value,
    };
    (void)device_.device().waitSemaphores(wi, UINT64_MAX);
}

void Renderer::beginForwardRendering(vk::CommandBuffer cmd)
{
    auto extent = swapchain_.extent();
    vk::Rect2D renderArea{
        .offset = vk::Offset2D{.x = 0, .y = 0},
        .extent = extent,
    };

    const vk::Image hdrImage = resources_.vulkanImage(postProcessing_.offscreenColourTarget());
    const vk::Image velocityImage = resources_.vulkanImage(taa_.velocityTarget());
    const vk::Image depthImage = swapchain_.depthImage();

    // Dynamic rendering does no implicit attachment transitions. The HDR target
    // rests in ShaderReadOnly between frames (last frame's post-process sampled
    // it) and the depth in DepthStencilAttachmentOptimal; both are cleared this
    // pass (loadOp Clear), so we discard via Undefined and transition into the
    // attachment layouts, gated on the prior frame's reads/writes.
    forwardImageBarrier(cmd, hdrImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eColorAttachmentOptimal,
                        vk::PipelineStageFlagBits2::eFragmentShader, {},
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        vk::AccessFlagBits2::eColorAttachmentWrite);
    // Velocity target: also ShaderReadOnly between frames (TAA resolve samples it).
    forwardImageBarrier(cmd, velocityImage, vk::ImageAspectFlagBits::eColor,
                        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                        vk::PipelineStageFlagBits2::eFragmentShader, {},
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        vk::AccessFlagBits2::eColorAttachmentWrite);
    // The depth prepass already filled this buffer (and left it in the
    // attachment layout); the forward pass loads it (loadOp Load below), so keep
    // the layout and order the prepass depth writes before the forward depth
    // test/write (write-after-write hazard, same image).
    forwardImageBarrier(cmd, depthImage, vk::ImageAspectFlagBits::eDepth,
                        vk::ImageLayout::eDepthStencilAttachmentOptimal,
                        vk::ImageLayout::eDepthStencilAttachmentOptimal,
                        vk::PipelineStageFlagBits2::eLateFragmentTests,
                        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                        vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                        vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                            vk::AccessFlagBits2::eDepthStencilAttachmentRead);

    const vk::ClearValue colourClear{
        .color = vk::ClearColorValue{.float32 = {{0.02f, 0.02f, 0.02f, 1.0f}}}};
    const vk::ClearValue velocityClear{.color = vk::ClearColorValue{.float32 = {{0.0f, 0.0f}}}};
    std::array<vk::RenderingAttachmentInfo, 2> colours{
        vk::RenderingAttachmentInfo{
            .imageView = resources_.vulkanImageView(postProcessing_.offscreenColourTarget()),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = colourClear,
        },
        vk::RenderingAttachmentInfo{
            .imageView = resources_.vulkanImageView(taa_.velocityTarget()),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = velocityClear,
        },
    };
    // Load the depth the prepass wrote (forward tests LESS_OR_EQUAL against it).
    vk::RenderingAttachmentInfo depth{
        .imageView = swapchain_.depthView(),
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmd.beginRendering(makeRenderingInfo(renderArea, colours, &depth));

    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);
}

void Renderer::endForwardRendering(vk::CommandBuffer cmd)
{
    cmd.endRendering();

    // Mirror the old render pass's finalLayout: leave the HDR target in
    // ShaderReadOnly so the transmission scene-capture / bloom / post-process
    // can sample it. Depth keeps DepthStencilAttachmentOptimal for the
    // transmission load. Depth store was DontCare under render passes, but we
    // keep Store so the transmission depth-load has defined contents.
    const vk::Image hdrImage = resources_.vulkanImage(postProcessing_.offscreenColourTarget());
    forwardImageBarrier(
        cmd, hdrImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderRead);
    const vk::Image velocityImage = resources_.vulkanImage(taa_.velocityTarget());
    forwardImageBarrier(
        cmd, velocityImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderRead);
}

void Renderer::transitionSwapchainToPresent(vk::CommandBuffer cmd, uint32_t imageIndex)
{
    // The render→present dependency is carried by the renderFinished semaphore
    // signalled at submit, so dstStage is bottom-of-pipe with no access mask.
    forwardImageBarrier(cmd, swapchain_.images()[imageIndex], vk::ImageAspectFlagBits::eColor,
                        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        vk::AccessFlagBits2::eColorAttachmentWrite,
                        vk::PipelineStageFlagBits2::eBottomOfPipe, {});
}

void Renderer::submitAndPresent(Window& display, vk::CommandBuffer cmd, uint32_t imageIndex)
{
    auto imageAvail = frame_.imageAvailable(currentFrame_);
    auto renderDone = frame_.renderFinished(imageIndex);
    vk::SemaphoreSubmitInfo waitInfo{
        .semaphore = imageAvail,
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    };
    // Signal both: the binary renderDone (present waits on it) and the timeline at
    // the next monotonic value (CPU frame pacing waits on it). The timeline signal
    // is all-commands so its value only advances once the whole frame is done.
    const uint64_t signalValue = ++timelineValue_;
    const std::array<vk::SemaphoreSubmitInfo, 2> signalInfos{{
        {.semaphore = renderDone, .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput},
        {.semaphore = frame_.timeline(),
         .value = signalValue,
         .stageMask = vk::PipelineStageFlagBits2::eAllCommands},
    }};
    vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = cmd};
    vk::SubmitInfo2 si{
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo,
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signalInfos.size()),
        .pSignalSemaphoreInfos = signalInfos.data(),
    };

    device_.graphicsQueue().submit2(si);
    frameTimelineValue_[currentFrame_] = signalValue;
    imageTimelineValue_[imageIndex] = signalValue;

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
    transmission_.recreate(postProcessing_.offscreenColourTarget(), taa_.velocityTarget());
    particles_.recreate(postProcessing_.offscreenColourTarget());
    taa_.recreate(postProcessing_.offscreenColourTarget());
    ssao_.recreate();
    resources_.sharedTextures().ssaoMap = ssao_.aoTarget();
    // Rewrite the forward-globals (set 1) descriptors so they reference the
    // sampler/view from the freshly recreated sceneColor target (and any
    // other recreated shared texture) instead of the destroyed ones.
    resources_.descriptors().updateGlobalDescriptors(globalDescSets_,
                                                     buildGlobalDescriptorRequest());
    frame_.createRenderFinishedSemaphores(swapchain_.images().size());
    imageTimelineValue_.assign(swapchain_.images().size(), 0);
}

} // namespace fire_engine
