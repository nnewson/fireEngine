#include <fire_engine/graphics/mapped_buffer.hpp>
#include <fire_engine/render/renderer.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <span>
#include <utility>

#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/graphics/frustum.hpp>
#include <fire_engine/graphics/image.hpp>
#include <fire_engine/graphics/renderable_scene.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/view_basis.hpp>
#include <fire_engine/render/cubemap_basis.hpp>
#include <fire_engine/render/draw_record.hpp>
#include <fire_engine/render/environment_precompute.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/vdpm_scan.hpp>
#include <fire_engine/render/viewport.hpp>

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
    : device_(window, debug.requireValidation),
      swapchain_(device_, window, !debug.capturePath.empty()),
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
    tunables_.lodMode = debug.lodMode;
    tunables_.lodEnabled = debug.lod;
    capturePath_ = std::string{debug.capturePath};
    captureFrame_ = static_cast<std::uint64_t>(debug.captureFrame);
    if (captureWanted())
    {
        // Fail at STARTUP on an unsupported format rather than at the capture frame — a run that
        // renders for a while and only then reports it can't write the image wastes the operator's
        // time. Re-resolved when the copy is recorded, in case a recreation changed it.
        captureFormat_ = resolveCaptureFormat(swapchain_.format());
    }
    // B5c-4 default flip: an unset backend request resolves to ON wherever the device supports the
    // GPU-driven front (VdpmScan::deviceSupported — the same predicate that builds the manager
    // below), so the GPU path is the default; --vdpm-gpu / --no-vdpm-gpu force it explicitly.
    tunables_.vdpmGpuBackend = debug.vdpmGpuBackend.value_or(VdpmScan::deviceSupported(device_));
    tunables_.noShadows = debug.noShadows;
    tunables_.debugDrawAabbs = debug.physicsDebug;
    tunables_.debugDrawColliders = debug.physicsDebug;
    tunables_.debugDrawContacts = debug.physicsDebug;

    // No explicit skybox (empty path): still precompute IBL from the default environment, but
    // don't draw the sky as a background and calm the IBL to a neutral mid level. An explicit
    // skybox arg draws + lights at the full-strength constants.
    drawSkybox_ = !environmentPath_.empty();
    if (!drawSkybox_)
    {
        tunables_.diffuseIbl = kNoSkyboxDiffuseIblStrength;
        tunables_.specularIbl = kNoSkyboxSpecularIblStrength;
    }

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
    cameraUbo_ = resources_.createMappedUniformBuffers(sizeof(CameraUBO));
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

    // GPU-driven VDPM front manager (rendering-spine #3, Stage B5b). Built once here, gated ONLY on
    // the device's compute/scan capability — never on the runtime backend selector — so toggling
    // RenderTunables::vdpmGpuBackend at runtime works without a reload, and an unsupported device
    // simply leaves the manager null (the CPU front stays usable; construction never fails). The
    // per-mesh dispatch-limit fallback is handled inside the manager (logged once).
    if (VdpmScan::deviceSupported(device_))
    {
        vdpmManager_ = std::make_unique<VdpmGpuManager>(device_, resources_);
    }
    else
    {
        log::info(log::category::render,
                  "VDPM GPU backend unavailable: device does not meet the compute/scan limits; the "
                  "view-dependent LOD front will run on the CPU");
    }
    // Device capability is fixed for the renderer's lifetime, so publish it to the overlay ONCE
    // here rather than per frame: the first `buildUi` runs before the first `collectDrawCommands`,
    // so a per-frame assignment would report "unsupported" on frame 0 with `--overlay`.
    stats_.vdpmGpuAvailable = (vdpmManager_ != nullptr);
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
    // No upload here: assignSelfShadowSlots — which runs later this frame in collectDrawCommands,
    // before any submit — fills the self-shadow matrices and writes the whole struct once. That is
    // the single authoritative per-frame LightUBO upload; a write here would be immediately
    // overwritten (the struct is multi-KB), so don't reinstate one.
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
    // Joints is an overlay-only view with no shader branch (it suppresses geometry instead), so it
    // maps to None for the fragment shader's debug-view selector.
    const DebugView shaderView =
        tunables_.debugView == DebugView::Joints ? DebugView::None : tunables_.debugView;
    out.environmentParams[2] = static_cast<float>(shaderView);
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

    // The single authoritative per-frame LightUBO upload (updateLightData deliberately does not
    // write — see the note there). Runs unconditionally, even with no self-shadow casters, so the
    // rest of lightData_ (lights, cascades, IBL) still reaches the GPU.
    writeMapped(lightUbo_.mapped[currentFrame_], lightData_);
}

void Renderer::clearDrawBuckets(DrawBuckets& buckets) noexcept
{
    buckets.shadow.clear();
    buckets.worldShadow.clear();
    buckets.selfShadow.clear();
    buckets.opaque.clear();
    buckets.blend.clear();
    buckets.transmissive.clear();
    buckets.anySkinned = false;
}

void Renderer::buildDrawBuckets(std::span<const DrawCommand> drawCommands, DrawBuckets& buckets,
                                ShadowFrameStats& shadowStats) const
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
        buckets.anySkinned = buckets.anySkinned || dc.hasSkin;
        if (dc.pipeline == shadows_.pipelineHandle())
        {
            // ONCE per original command, before any bucket insertion: the split below is
            // deliberately NOT exclusive (every caster enters `shadow`, and additionally either
            // `worldShadow` or `selfShadow`), and each bucket is then replayed across cascades and
            // faces. Counting the reason anywhere downstream would multiply one decision by the
            // number of buckets and views it feeds. The per-view accounting below is the opposite:
            // it counts every replay, because each replay is real raster work.
            shadowStats.addLodReason(dc.shadowLodReason);
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
        const bool pipelineChanged = dc.pipeline != lastBoundPipeline;
        if (pipelineChanged)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             resources_.vulkanPipeline(dc.pipeline));
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
            // Forward set 0 is pushed inline (core 1.4 push descriptors) — no
            // per-object descriptor set — plus the per-draw push constants.
            // Skybox (also in this bucket) keeps its allocated set 0.
            pushForwardObjectDescriptors(cmd, resources_,
                                         resources_.vulkanPipelineLayout(dc.pipeline), dc);
            if (pipelineChanged)
            {
                // Establish push-descriptor set 0 before installing allocated
                // higher sets. Vulkan layout compatibility preserves set 0
                // when sets 1/2 are then bound through this same layout. This
                // ordering also avoids a Vulkan Validation Layers 1.4.350
                // first-use state-tracking defect in passes with no preceding
                // forward push (for example an all-blend scene).
                vk::DescriptorSet globalSet =
                    resources_.vulkanDescriptorSet(globalDescSets_[currentFrame_]);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       resources_.vulkanPipelineLayout(dc.pipeline), 1, globalSet,
                                       {});
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       resources_.vulkanPipelineLayout(dc.pipeline), 2,
                                       resources_.bindlessDescriptorSet(), {});
            }
            cmd.pushConstants<ForwardPushConstants>(resources_.vulkanPipelineLayout(dc.pipeline),
                                                    vk::ShaderStageFlagBits::eFragment, 0,
                                                    makeForwardPushConstants(dc));
        }
        else
        {
            vk::DescriptorSet ds = resources_.vulkanDescriptorSet(dc.descriptorSet);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                   resources_.vulkanPipelineLayout(dc.pipeline), 0, ds, {});
        }
        recordIndexedDraw(cmd, dc, resources_);
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
        recordIndexedDraw(cmd, dc, resources_);
    }
    cmd.endRendering();
}

void Renderer::recordSsaoPass(vk::CommandBuffer cmd)
{
    ssao_.recordPass(cmd, currentFrame_);
}

void Renderer::recordDebugDrawPass(vk::CommandBuffer cmd)
{
    // Query lines (the -q probe) draw independently of the --debug-physics categories,
    // so run the pass when either is present.
    if (!physicsDebugWanted() && physicsDebug_.queryLines.empty())
    {
        return;
    }
    debugDraw_.record(cmd, postProcessing_.offscreenColourTarget(), currentViewProj_, physicsDebug_,
                      tunables_, currentFrame_);
}

void Renderer::recordForwardPass(vk::CommandBuffer cmd, const DrawBuckets& buckets)
{
    beginForwardRendering(cmd);
    // The Joints debug view replaces the scene with the articulation gizmo/labels: skip the mesh
    // (and skybox) draws so nothing occludes the joints — the HDR target keeps its clear colour.
    if (tunables_.debugView != DebugView::Joints)
    {
        auto lastBoundPipeline = PipelineHandle{std::numeric_limits<uint32_t>::max()};
        recordDrawBucket(cmd, buckets.opaque, lastBoundPipeline);
        recordDrawBucket(cmd, buckets.blend, lastBoundPipeline);
    }
    endForwardRendering(cmd);
}

void Renderer::updateFrameLighting(RenderableScene& scene, Vec3 cameraPosition, Vec3 cameraTarget)
{
    const auto extent = swapchain_.extent();
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

    scene.gatherLights(lightScratch_);
    updateLightData(cameraPosition, cameraTarget, aspect, lightScratch_);
}

const Renderer::DrawBuckets& Renderer::collectDrawCommands(RenderableScene& scene,
                                                           Vec3 cameraPosition, Vec3 cameraTarget)
{
    drawCommandScratch_.clear();
    if (drawSkybox_)
    {
        recordSkybox(cameraPosition, cameraTarget, drawCommandScratch_);
    }

    // Per-frame camera UBO (set 1) — written once here for every forward draw this frame, rather
    // than baked into each object's per-object UBO. proj is the jittered projection (TAA).
    CameraUBO cameraData{};
    cameraData.view = view_;
    cameraData.proj = jitteredProj_;
    cameraData.cameraPos[0] = cameraPosition.x();
    cameraData.cameraPos[1] = cameraPosition.y();
    cameraData.cameraPos[2] = cameraPosition.z();
    cameraData.cameraPos[3] = 0.0f;
    cameraData.currentViewProj = currentViewProj_;
    cameraData.previousViewProj = previousViewProj_;
    writeMapped(cameraUbo_.mapped[currentFrame_], cameraData);

    // GPU-driven VDPM (Stage B5b): expose the per-frame request sink + the runtime backend selector
    // so Object appends visible fronts during collection. The compute only runs when the manager
    // exists (device capable) AND the selector is on; otherwise the sink stays unused and the CPU
    // front drives the draw.
    vdpmRequestScratch_.clear();
    const bool vdpmGpuActive = vdpmManager_ != nullptr && tunables_.vdpmGpuBackend;

    const auto extent = swapchain_.extent();
    const AlphaPipelines pipelines{forwardOpaqueHandle_, forwardBlendHandle_};
    const FrameInfo frame{.currentFrame = currentFrame_,
                          .viewportWidth = extent.width,
                          .viewportHeight = extent.height,
                          .cameraPosition = cameraPosition,
                          .cameraTarget = cameraTarget,
                          .view = view_,
                          .proj = jitteredProj_,
                          .currentViewProj = currentViewProj_,
                          .previousViewProj = previousViewProj_,
                          .cameraUbo = cameraUbo_.buffers[currentFrame_],
                          .pipelines = pipelines,
                          .lodEnabled = tunables_.lodEnabled,
                          .lodPixelErrorBudget = tunables_.lodPixelErrorBudget,
                          .lodMode = tunables_.lodMode,
                          .vdpmGpuBackend = vdpmGpuActive,
                          .vdpmRequestSink = vdpmGpuActive ? &vdpmRequestScratch_ : nullptr,
                          .shadowPipeline = shadows_.pipelineHandle(),
                          .shadowViewProjs = shadowViewProjs_};

    // Coarse pre-cull frustums: the camera plus every ACTIVE shadow caster. The union is a superset
    // of what buildDrawBuckets / shadows_ keep per pass, so a node dropped by all of them is never
    // wanted downstream. Only the active slots are pushed — the cascades (always all present) plus
    // the assigned spot / point-face slots (their counts were fixed in updateLightData, above).
    // Inactive slots hold identity matrices, whose frustum is a small NDC-cube box near the origin
    // that no pass renders into; skipping them keeps the union a superset AND tightens the cull
    // (they only ever spuriously added visibility). When culling is disabled we pass an empty span
    // and the scene draws everything.
    frustumScratch_.clear();
    if (tunables_.cullingEnabled)
    {
        const auto pushRange = [&](int base, int count)
        {
            for (int i = 0; i < count; ++i)
            {
                const auto slot = static_cast<std::size_t>(base) + static_cast<std::size_t>(i);
                frustumScratch_.push_back(Frustum::fromViewProj(shadowViewProjs_[slot]));
            }
        };
        frustumScratch_.reserve(1 + static_cast<std::size_t>(kShadowCascadeCount) +
                                static_cast<std::size_t>(activeSpotCasters_) +
                                kCubemapFaceCount * static_cast<std::size_t>(activePointCasters_));
        frustumScratch_.push_back(Frustum::fromViewProj(currentViewProj_));
        pushRange(kShadowCascadeMatrixBase, static_cast<int>(kShadowCascadeCount));
        pushRange(kShadowSpotMatrixBase, activeSpotCasters_);
        pushRange(kShadowPointMatrixBase,
                  static_cast<int>(kCubemapFaceCount) * activePointCasters_);
    }

    const CullStats cull = scene.buildDrawCommands(frame, frustumScratch_, drawCommandScratch_);
    stats_.trackedNodes = static_cast<int>(cull.tracked);
    stats_.culledNodes = static_cast<int>(cull.culled);
    stats_.vdpmFoldoversRepaired = static_cast<int>(cull.vdpmFoldoversRepaired);
    stats_.vdpmCoverageRepaired = static_cast<int>(cull.vdpmCoverageRepaired);
    stats_.vdpmGeometryTriggers = static_cast<int>(cull.vdpmChannels.geometryTriggers);
    stats_.vdpmUvTriggers = static_cast<int>(cull.vdpmChannels.uvTriggers);
    stats_.vdpmNormalTriggers = static_cast<int>(cull.vdpmChannels.normalTriggers);
    stats_.vdpmTangentTriggers = static_cast<int>(cull.vdpmChannels.tangentTriggers);
    stats_.vdpmMaxGeometryRatio = cull.vdpmChannels.maxGeometryRatio;
    stats_.vdpmMaxUvRatio = cull.vdpmChannels.maxUvRatio;
    stats_.vdpmMaxNormalRatio = cull.vdpmChannels.maxNormalRatio;
    stats_.vdpmMaxTangentRatio = cull.vdpmChannels.maxTangentRatio;

    assignSelfShadowSlots(drawCommandScratch_);
    // Reset this frame's slot before anything writes into it: the ring slot still holds the
    // counters from `kMaxFramesInFlight` frames ago, which have already been published.
    ShadowFrameStats& shadowStats = shadowStatsRing_[currentFrame_];
    shadowStats.reset();
    buildDrawBuckets(drawCommandScratch_, drawBucketsScratch_, shadowStats);

    // GPU-driven VDPM (Stage B5b): distil the request sink down to the fronts that are actually
    // camera-visible this frame, and (B5b-2) resolve each visible forward draw's buffers to the GPU
    // output. Object appended a request for every front on a coarse-cull survivor (camera ∪
    // shadow), but only fronts whose FORWARD draw survived into the forward buckets should run
    // their compute — a shadow-only instance must not. Walk the forward buckets: collect the
    // visible handles and point each tagged command at its front's GPU-emitted index + indirect
    // ring for this frame (resolveDrawBuffers throws if a tagged front doesn't resolve — an
    // invariant violation, since a GPU-backed draw carries indexCount 0 and a silent miss would
    // issue a zero-count draw). Then selectVisibleVdpmRequests filters the sink to the visible
    // subset and dedups (identical duplicate collapsed; conflicting params for one front throws).
    // The compute is recorded in drawFrame after collection; the compute→(index + indirect read)
    // barrier is delayed to just before the depth prepass so the shadow pass overlaps it.
    vdpmRecordScratch_.clear();
    if (vdpmGpuActive && !vdpmRequestScratch_.empty())
    {
        vdpmVisibleScratch_.clear();
        for (std::vector<DrawCommand>* bucket :
             {&drawBucketsScratch_.opaque, &drawBucketsScratch_.blend,
              &drawBucketsScratch_.transmissive})
        {
            for (DrawCommand& dc : *bucket)
            {
                if (dc.vdpmGpuFront != NullVdpmFront)
                {
                    vdpmVisibleScratch_.push_back(dc.vdpmGpuFront);
                    const VdpmGpuManager::DrawBuffers bufs =
                        vdpmManager_->resolveDrawBuffers(dc.vdpmGpuFront, currentFrame_);
                    dc.indexBuffer = bufs.index;
                    dc.indirectBuffer = bufs.indirect;
                    dc.indirectOffset = 0;
                }
            }
        }
        selectVisibleVdpmRequests(vdpmRequestScratch_, vdpmVisibleScratch_, vdpmRecordScratch_,
                                  vdpmSelectScratch_);
    }

    std::uint64_t triangles = 0;
    for (const std::vector<DrawCommand>* bucket :
         {&drawBucketsScratch_.opaque, &drawBucketsScratch_.blend,
          &drawBucketsScratch_.transmissive})
    {
        for (const DrawCommand& dc : *bucket)
        {
            triangles += dc.indexCount / 3;
        }
    }
    stats_.trianglesDrawn = triangles;
    stats_.trianglesOverflow = false;
    stats_.trianglesGpuPending = false; // set below iff the GPU backend records but isn't warm
    // (stats_.vdpmGpuAvailable is published once at construction — device capability is fixed and
    // the overlay reads it before this runs on frame 0.)
    return drawBucketsScratch_;
}

void Renderer::recordShadowPass(vk::CommandBuffer cmd, const DrawBuckets& buckets)
{
    std::span<const PointShadowCaster> pointCasterSpan{
        pointCasters_.data(), static_cast<std::size_t>(activePointCasters_)};
    // Self-shadow slots are assigned densely (assignSelfShadowSlots), so the
    // scratch map's size is the number of slots the pass must render; the world
    // CSM is wanted only when a skinned draw exists to sample it.
    shadows_.recordPass(cmd, buckets.shadow, buckets.worldShadow, buckets.selfShadow,
                        static_cast<int>(selfShadowSlotsScratch_.size()), activeSpotCasters_,
                        pointCasterSpan, shadowViewProjs_, tunables_.cullingEnabled,
                        buckets.anySkinned, shadowStatsRing_[currentFrame_], profiler_,
                        currentFrame_);
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

void Renderer::drawFrame(Window& display, RenderableScene& scene, float dt)
{
    // The scene owns the active camera; pull its pose through the seam. (The scene's per-frame
    // update — which refreshes camera world transforms — has already run in the main loop before
    // this.)
    const CameraView camera = scene.activeCamera();
    const Vec3 cameraPosition = camera.position;
    const Vec3 cameraTarget = camera.target;

    auto imageIndex = acquireNextImage(display);
    if (!imageIndex)
    {
        return;
    }

    // Publish the SHADOW counters this slot collected a ring-cycle ago, BEFORE the timings and the
    // overlay are built — the acquire timeline-wait is what guarantees that frame completed, so
    // this is the first moment its counters describe finished work. Consuming the validity bit
    // means only a later successful submission can re-arm the slot; during ring warm-up (or after a
    // frame that never reached submit) the flag is false and the panel says so. Deliberately
    // INDEPENDENT of gpuValid: timestamps may be unsupported on a device while these CPU-side
    // counters remain perfectly valid.
    stats_.shadowValid = std::exchange(shadowStatsSlotUsed_[currentFrame_], false);
    if (stats_.shadowValid)
    {
        stats_.shadow = shadowStatsRing_[currentFrame_];
    }
    else
    {
        stats_.shadow.reset();
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
    // Use the shared stable-up helper rather than a hardcoded world-up so a near-vertical look
    // direction can't degenerate the view basis (identical to {0,1,0} for every non-vertical view —
    // stableUpForForward returns it whenever |dot(forward, up)| < 0.99). The skybox/shadow paths
    // already go through view_basis.hpp; this keeps the main view consistent with them.
    view_ = Mat4::lookAt(cameraPosition, cameraTarget,
                         stableUpForForward(cameraTarget - cameraPosition));
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
    const DrawBuckets& buckets = collectDrawCommands(scene, cameraPosition, cameraTarget);

    // GPU-driven VDPM (Stage B5b): record the front lifecycle (score → refine/coarsen → repair →
    // emit) for each camera-visible front collected above. Recorded here after collection, BEFORE
    // the shadow pass; the frame-global camera/viewport/budget come from the jitter-free
    // view-projection (TAA jitter would thrash the coverage test). recordRequests adds NO consumer
    // barrier — the collectDrawCommands walk pointed each tagged draw at this front's GPU output
    // (resolveDrawBuffers), and the compute→(index + indirect read) barrier is recorded separately
    // just before the depth prepass (after the shadow pass, so shadow work overlaps the compute).
    // Perf instrumentation defaults to 0 for a frame that records no GPU VDPM compute.
    stats_.vdpmRecordCpuMs = 0.0f;
    stats_.vdpmFrontsRecorded = 0;
    stats_.vdpmMaxRankCount = 0;
    stats_.vdpmRepairRoundBudget = 0;
    stats_.vdpmAnalyticDispatches = 0;
    stats_.vdpmAnalyticBarriers = 0;
    stats_.vdpmApplyJobs = 0;
    stats_.vdpmRepairJobs = 0;
    stats_.vdpmRepairFronts = 0;
    stats_.vdpmMaxMarkedRounds = -1;
    stats_.vdpmSumMarkedRounds = 0;
    stats_.vdpmFallbackFronts = 0;
    stats_.vdpmNonCleanPrefix = 0;
    stats_.vdpmAncestorFailures = 0;
    stats_.vdpmFailFlagFronts = 0;
    stats_.vdpmEmittedOverflow = false;
    if (vdpmManager_ != nullptr && !vdpmRecordScratch_.empty())
    {
        const VdpmFrameGlobals globals{.viewProj = currentViewProj_,
                                       .cameraPos = cameraPosition,
                                       .projScaleY = std::abs(unjitteredProj[1, 1]),
                                       .viewportWidth = static_cast<float>(extent.width),
                                       .viewportHeight = static_cast<float>(extent.height),
                                       .pixelBudget = tunables_.lodPixelErrorBudget,
                                       .frameIndex = currentFrame_,
                                       .stageProfiler = &profiler_};
        // Perf instrumentation (no behaviour change): GPU timestamps around the compute
        // (ProfilePass::VdpmCompute) + CPU wall time of the recording itself (the MoltenVK
        // command-translation cost this arc suspects rivals shader time) + the analytic command
        // counts the manager tallied.
        // Per-front forward-draw counts (submitted-draw weighting for the health emitted total),
        // keyed BY HANDLE from the actual camera-visible forward buckets (not the deduped
        // requests).
        vdpmDrawCounts_.clear();
        for (const std::vector<DrawCommand>* bucket :
             {&drawBucketsScratch_.opaque, &drawBucketsScratch_.blend,
              &drawBucketsScratch_.transmissive})
        {
            for (const DrawCommand& dc : *bucket)
            {
                if (dc.vdpmGpuFront == NullVdpmFront)
                {
                    continue;
                }
                bool found = false;
                for (VdpmFrontDrawCount& c : vdpmDrawCounts_)
                {
                    if (c.front == dc.vdpmGpuFront)
                    {
                        c.drawCount += 1;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    vdpmDrawCounts_.push_back({.front = dc.vdpmGpuFront, .drawCount = 1});
                }
            }
        }
        // This frame's CPU-front (+ non-VDPM) triangle subtotal: GPU draws carry indexCount 0, so
        // stats_.trianglesDrawn (set in collectDrawCommands) IS exactly that subtotal. The manager
        // stores it and combines it with the delayed GPU emitted total, one frame at a time.
        const std::uint64_t cpuTriangleSubtotal = stats_.trianglesDrawn;

        profiler_.begin(cmd, currentFrame_, ProfilePass::VdpmCompute);
        const auto vdpmRecordStart = std::chrono::steady_clock::now();
        vdpmManager_->recordRequests(cmd, vdpmRecordScratch_, globals, vdpmDrawCounts_);
        const auto vdpmRecordEnd = std::chrono::steady_clock::now();
        profiler_.end(cmd, currentFrame_, ProfilePass::VdpmCompute);
        // Delayed scene-health readback recorded AFTER the timestamp closes (the copy stays out of
        // the compute measurement).
        vdpmManager_->recordDiagnosticReadback(cmd, currentFrame_, cpuTriangleSubtotal);

        stats_.vdpmRecordCpuMs =
            std::chrono::duration<float, std::milli>(vdpmRecordEnd - vdpmRecordStart).count();
        const VdpmGpuManager::ComputeStats& cs = vdpmManager_->lastComputeStats();
        stats_.vdpmFrontsRecorded = static_cast<int>(cs.frontsRecorded);
        stats_.vdpmMaxRankCount = static_cast<int>(cs.maxRankCount);
        stats_.vdpmRepairRoundBudget = static_cast<int>(cs.roundBudget);
        stats_.vdpmAnalyticDispatches = static_cast<int>(cs.analyticDispatches);
        stats_.vdpmAnalyticBarriers = static_cast<int>(cs.analyticBarriers);
        stats_.vdpmApplyJobs = static_cast<int>(cs.applyJobs);
        stats_.vdpmRepairJobs = static_cast<int>(cs.repairJobs);
        const VdpmGpuManager::SceneHealth& health = vdpmManager_->lastSceneHealth();
        if (health.valid)
        {
            // GPU backend active: the overlay triangle count becomes the frame-consistent (delayed)
            // combined CPU+GPU total. Pure-CPU scenes never reach here, so their count stays
            // current.
            stats_.trianglesDrawn = health.triangleTotal;
            stats_.trianglesOverflow = health.emittedOverflow;
        }
        else
        {
            // GPU fronts recorded but the emitted total isn't back yet (ring warming / post-gap):
            // trianglesDrawn is the CPU-only subtotal — INCOMPLETE, so flag it pending.
            stats_.trianglesGpuPending = true;
        }
        stats_.vdpmRepairFronts = static_cast<int>(health.repairFronts);
        stats_.vdpmMaxMarkedRounds = health.valid ? static_cast<int>(health.maxMarkedRounds) : -1;
        stats_.vdpmSumMarkedRounds = static_cast<int>(health.sumMarkedRounds);
        stats_.vdpmFallbackFronts = static_cast<int>(health.fallbackFronts);
        stats_.vdpmNonCleanPrefix = static_cast<int>(health.nonCleanPrefix);
        stats_.vdpmAncestorFailures = static_cast<int>(health.ancestorFailures);
        stats_.vdpmFailFlagFronts = static_cast<int>(health.failFlagFronts);
        stats_.vdpmEmittedOverflow = health.emittedOverflow;

        // Periodic perf sample (headless baseline complement to the overlay). CPU record ms is this
        // frame's; the GPU VdpmCompute ms was resolved from a ring-cycle ago (0 / invalid if the
        // device lacks timestamp support).
        if (++vdpmPerfLogCounter_ % 120 == 0)
        {
            const float gpuMs = stats_.passMs[static_cast<std::size_t>(ProfilePass::VdpmCompute)];
            log::debug(
                log::category::render,
                "VDPM GPU perf: record {:.3f} ms CPU | compute {:.3f} ms GPU (valid {}) | {} "
                "front(s), ~{} dispatches ({} apply + {} repair batched jobs), max {}/{} marked "
                "rounds",
                stats_.vdpmRecordCpuMs, gpuMs, stats_.gpuValid, stats_.vdpmFrontsRecorded,
                stats_.vdpmAnalyticDispatches, stats_.vdpmApplyJobs, stats_.vdpmRepairJobs,
                stats_.vdpmMaxMarkedRounds, stats_.vdpmRepairRoundBudget);
            // Per-stage breakdown (apply-kernel checkpoint). CPU ms is summed over the frame's
            // fronts (any count); GPU ms is meaningful only when ONE front recorded (single-shot
            // query slots) — it reads 0 otherwise. Confirms which stage owns the GPU time before
            // the kernel.
            const auto& scpu = vdpmManager_->lastComputeStats().stageCpuMs;
            auto stageGpu = [&](ProfilePass p)
            { return stats_.passMs[static_cast<std::size_t>(p)]; };
            log::debug(
                log::category::render,
                "VDPM GPU per-stage: CPU score {:.3f} apply {:.3f} repair {:.3f} emit {:.3f} "
                "ms | GPU (1-front) score {:.3f} apply {:.3f} repair {:.3f} emit {:.3f} ms",
                scpu[0], scpu[1], scpu[2], scpu[3], stageGpu(ProfilePass::VdpmScore),
                stageGpu(ProfilePass::VdpmApply), stageGpu(ProfilePass::VdpmRepair),
                stageGpu(ProfilePass::VdpmEmit));
        }
    }
    else if (vdpmManager_ != nullptr)
    {
        // GPU backend present but recording nothing this frame (toggled off, or no visible front):
        // invalidate this slot so its stale reduction isn't parsed a cycle later as fresh.
        vdpmManager_->invalidateHealthSlot(currentFrame_);
    }

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

    // No outer Shadow span: the five families are timed individually with bottom-to-bottom
    // boundaries inside recordPass, and an enclosing top-to-bottom timer would overlap them (and
    // would then have to be excluded from the frame total to avoid double-counting).
    recordShadowPass(cmd, buckets);

    // GPU-driven VDPM (Stage B5b-2): the VDPM compute (recorded above, after collection) wrote each
    // visible front's emitted index stream (scatter) + indirect command (finalize). The depth
    // prepass is the FIRST consumer (index assembly + drawIndexedIndirect), so order the compute
    // writes before it here — delayed until after the shadow pass (which never reads the VDPM
    // output: shadows keep discrete/direct and their draw copies cleared vdpmGpuFront) so
    // shadow-pass GPU work overlaps the compute. One global barrier covers both consumed buffers
    // for the prepass, forward, and transmission reads that follow.
    if (vdpmManager_ != nullptr && !vdpmRecordScratch_.empty())
    {
        const vk::MemoryBarrier2 computeToDraw{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask =
                vk::PipelineStageFlagBits2::eIndexInput | vk::PipelineStageFlagBits2::eDrawIndirect,
            .dstAccessMask =
                vk::AccessFlagBits2::eIndexRead | vk::AccessFlagBits2::eIndirectCommandRead,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &computeToDraw});
    }

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
    // Ragdoll joint labels: projected ImGui text (the marker gizmo lines go through debugDraw_).
    // Added to the foreground draw list here, after currentViewProj_ is refreshed above, so the
    // labels project with this frame's camera (not last frame's) and track the joints as it moves.
    overlay_.drawWorldLabels(physicsDebug_.jointLabels, currentViewProj_);
    // Post-process leaves the swap image in ColorAttachmentOptimal; the overlay
    // draws over it, then we transition to present.
    overlay_.record(cmd, *swapchain_.imageViews()[*imageIndex], swapchain_.extent());
    // Capture (if this is the requested frame) reads the FINAL swapchain content — after
    // post-process and the overlay, immediately before present — so the file is exactly what the
    // screen showed, tone-mapped and all. Capturing the HDR offscreen target instead would produce
    // an image no viewer ever saw.
    ++framesRendered_;
    const bool capturingThisFrame = captureWanted() && framesRendered_ == captureFrame_;
    if (capturingThisFrame)
    {
        recordCaptureCopy(cmd, *imageIndex);
    }
    transitionSwapchainToPresent(cmd, *imageIndex, capturingThisFrame);

    cmd.end();
    submitAndPresent(display, cmd, *imageIndex);
    if (capturingThisFrame)
    {
        writeCapture();
    }

    // Only now is this slot's collection eligible to become "completed": marking it after
    // recordShadowPass would mean "recorded", and a frame that threw or bailed before submit would
    // publish counters for work the GPU never ran.
    shadowStatsSlotUsed_[currentFrame_] = true;

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

void Renderer::transitionSwapchainToPresent(vk::CommandBuffer cmd, uint32_t imageIndex,
                                            bool afterCapture)
{
    // The render→present dependency is carried by the renderFinished semaphore
    // signalled at submit, so dstStage is bottom-of-pipe with no access mask.
    // After a capture the image is already in TransferSrcOptimal (the copy left it there), so
    // the present transition starts from that layout instead of ColorAttachmentOptimal.
    const vk::ImageLayout from = afterCapture ? vk::ImageLayout::eTransferSrcOptimal
                                              : vk::ImageLayout::eColorAttachmentOptimal;
    const vk::PipelineStageFlags2 srcStage =
        afterCapture ? vk::PipelineStageFlagBits2::eCopy
                     : vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    const vk::AccessFlags2 srcAccess = afterCapture ? vk::AccessFlagBits2::eTransferRead
                                                    : vk::AccessFlagBits2::eColorAttachmentWrite;
    forwardImageBarrier(cmd, swapchain_.images()[imageIndex], vk::ImageAspectFlagBits::eColor, from,
                        vk::ImageLayout::ePresentSrcKHR, srcStage, srcAccess,
                        vk::PipelineStageFlagBits2::eBottomOfPipe, {});
}

CaptureFormat Renderer::resolveCaptureFormat(vk::Format format)
{
    // Rejects anything that isn't 8-bit RGBA/BGRA rather than guessing: a wrong channel order
    // still yields a picture, just with the colours swapped — which would be committed as a
    // reference image and believed.
    switch (format)
    {
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        return CaptureFormat::Bgra8;
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        return CaptureFormat::Rgba8;
    default:
        break;
    }
    throw std::runtime_error(
        "--capture supports 8-bit RGBA/BGRA swapchain formats only; this surface uses " +
        std::string(vk::to_string(format)));
}

void Renderer::recordCaptureCopy(vk::CommandBuffer cmd, uint32_t imageIndex)
{
    // Snapshot the geometry AND the format now, with the copy. submitAndPresent may recreate
    // the swapchain (a resize, or an out-of-date present) before writeCapture runs, and the
    // buffer would then be decoded against an extent and format the pixels in it never had.
    captureExtent_ = swapchain_.extent();
    captureFormat_ = resolveCaptureFormat(swapchain_.format());
    const vk::Extent2D extent = captureExtent_;
    // Allocated on first use: a capture happens once, and a run without --capture must not pay
    // for a full-frame host-visible buffer.
    if (captureBuffer_.buffers[0] == NullBuffer)
    {
        captureBuffer_ = resources_.createMappedReadbackBuffers(
            static_cast<std::size_t>(extent.width) * extent.height * 4);
    }

    forwardImageBarrier(cmd, swapchain_.images()[imageIndex], vk::ImageAspectFlagBits::eColor,
                        vk::ImageLayout::eColorAttachmentOptimal,
                        vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        vk::AccessFlagBits2::eColorAttachmentWrite,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);

    // bufferRowLength/ImageHeight 0 ⇒ tightly packed to the copy extent, so the readback rows
    // have no padding and the converter's row pitch is exactly width * 4.
    const vk::BufferImageCopy region{
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = 0,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
        .imageExtent = {.width = extent.width, .height = extent.height, .depth = 1},
    };
    cmd.copyImageToBuffer(swapchain_.images()[imageIndex], vk::ImageLayout::eTransferSrcOptimal,
                          resources_.vulkanBuffer(captureBuffer_.buffers[0]), region);
}

void Renderer::writeCapture()
{
    // One-shot at shutdown: waiting for the whole device is simpler than threading a fence
    // through, and obviously correct — the copy must have completed before the mapping is read.
    device_.device().waitIdle();

    // The extent and format snapshotted WITH the copy, not the swapchain's current ones: a
    // recreation between the submit and here would otherwise reinterpret the captured pixels.
    const vk::Extent2D extent = captureExtent_;
    const std::vector<std::uint8_t> rgba =
        toRgba8(captureBuffer_.mapped[0], extent.width, extent.height,
                static_cast<std::size_t>(extent.width) * 4, captureFormat_);
    // Throwing rather than flagging: a capture command that prints an error and still exits 0
    // is worse than useless in a script, and the top-level handler already logs a fatal error
    // and returns EXIT_FAILURE.
    if (rgba.empty())
    {
        throw std::runtime_error("capture readback conversion failed for '" + capturePath_ + "'");
    }
    if (!writeRgba8Png(capturePath_.c_str(), rgba, extent.width, extent.height))
    {
        throw std::runtime_error("capture could not write '" + capturePath_ + "'");
    }
    // Only now: `captureComplete()` means the file EXISTS, which is what the main loop ends on.
    // Set before the write, a failure would briefly claim a capture that isn't there.
    captureDone_ = true;
    log::info(log::category::render, "captured frame {} ({}x{}) to '{}'", captureFrame_,
              extent.width, extent.height, capturePath_);
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
    // the next monotonic value (CPU frame pacing waits on it). BOTH are all-commands.
    //
    // renderDone must not be narrowed to eColorAttachmentOutput: work recorded AFTER the
    // colour attachment writes — the --capture image-to-buffer copy (eCopy) and the
    // present-layout transition — would not be covered, so presentation could race them.
    // Presentation needs the finished frame regardless, so there is no overlap to win here,
    // and synchronization validation is not enabled: zero VUIDs would not have caught it.
    const uint64_t signalValue = ++timelineValue_;
    const std::array<vk::SemaphoreSubmitInfo, 2> signalInfos{{
        {.semaphore = renderDone, .stageMask = vk::PipelineStageFlagBits2::eAllCommands},
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
    writeMapped(skyboxUbo_.mapped[currentFrame_], data);

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
