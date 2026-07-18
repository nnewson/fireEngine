#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/generational_slot_pool.hpp>
#include <fire_engine/render/vma.hpp>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/sampler_settings.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/descriptors.hpp>

namespace fire_engine
{

class Device;
class Image;
class KtxImage;
class Material;
class Pipeline;
class Vertex;

class Resources
{
public:
    enum class FallbackTextureKind
    {
        BaseColour,
        Emissive,
        Normal,
        MetallicRoughness,
        Occlusion,
    };

    Resources(const Device& device, const Pipeline& pipeline);
    // Compute-only construction (no graphics pipeline / bindless / descriptor infrastructure): sets
    // up ONLY the buffer table + upload command pool. For a headless compute device — the VDPM GPU
    // harness allocates SSBOs and records compute without any of the descriptor/texture machinery.
    // Resources still owns every GPU allocation. The descriptor-creating + texture APIs must not be
    // called on a compute-only Resources.
    explicit Resources(const Device& device);
    ~Resources() = default;

    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    Resources(Resources&&) noexcept = delete;
    Resources& operator=(Resources&&) noexcept = delete;

    // --- Buffer creation ---

    // Static mesh geometry: device-local (fast GPU reads), filled once via a staging copy.
    // Long-lived and never CPU-rewritten, so it does not belong in host-visible memory.
    [[nodiscard]] BufferHandle createVertexBuffer(std::span<const Vertex> vertices);
    // Vertex buffer that is also a storage buffer, so the cloth/soft-body compute
    // solver can write positions + normals into it in place each frame and the
    // forward/shadow passes read it as vertex input. Host-visible (mapped initial
    // upload) — CPU-seeded and GPU-mutated, unlike the static createVertexBuffer.
    [[nodiscard]] BufferHandle createStorageVertexBuffer(std::span<const Vertex> vertices);
    // Static index buffers (base mesh + every LOD cut): device-local, staged upload.
    [[nodiscard]] BufferHandle createIndexBuffer(std::span<const uint16_t> indices);
    [[nodiscard]] BufferHandle createIndexBuffer(std::span<const uint32_t> indices);

    // --- Texture creation ---

    [[nodiscard]] TextureHandle createTexture(const Image& image,
                                              const SamplerSettings& sampler = {},
                                              TextureEncoding encoding = TextureEncoding::Srgb);
    [[nodiscard]] TextureHandle createTexture(KtxImage&& image, const SamplerSettings& sampler = {},
                                              TextureEncoding encoding = TextureEncoding::Srgb);
    [[nodiscard]] TextureHandle createTexture(const uint8_t* pixels, int width, int height,
                                              const SamplerSettings& sampler = {},
                                              TextureEncoding encoding = TextureEncoding::Srgb);
    [[nodiscard]] TextureHandle createTexture(const float* pixels, int width, int height,
                                              const SamplerSettings& sampler = {});
    [[nodiscard]] TextureHandle createCubemapTexture(const float* pixels, uint32_t faceExtent,
                                                     const SamplerSettings& sampler = {});
    [[nodiscard]] TextureHandle createCubemapTexture(const float* pixels, uint32_t faceExtent,
                                                     uint32_t mipLevels,
                                                     const SamplerSettings& sampler = {});
    [[nodiscard]] TextureHandle createRenderTargetCubemap(uint32_t faceExtent, uint32_t mipLevels,
                                                          vk::Format format,
                                                          const SamplerSettings& sampler = {});
    [[nodiscard]] TextureHandle fallbackTexture(FallbackTextureKind kind);

    // --- Mapped buffer sets (per-frame, for UBOs and SSBOs) ---

    using MappedBufferSet = fire_engine::MappedBufferSet;

    [[nodiscard]] MappedBufferSet createMappedUniformBuffers(std::size_t size);
    // Per-frame, persistently-mapped host-visible vertex buffers for dynamic
    // CPU-built geometry (e.g. the physics debug-line endpoints).
    [[nodiscard]] MappedBufferSet createMappedVertexBuffers(std::size_t size);
    // Per-frame, persistently-mapped host-visible index buffers for VDPM's dynamic, view-dependent
    // index set (rebuilt each frame from the active front).
    [[nodiscard]] MappedBufferSet createMappedIndexBuffers(std::size_t size);
    // Per-frame, persistently-mapped host-visible INDIRECT-command buffers for VDPM's indirect
    // draws (one DrawIndexedIndirectCommand per instance, CPU-written from the emit count). The
    // subsequent queue submission makes the host write available to the draw; no explicit barrier.
    [[nodiscard]] MappedBufferSet createMappedIndirectBuffers(std::size_t size);
    // Per-frame, persistently-mapped host-visible buffers that are transfer-copy DESTINATIONS
    // (eTransferDst) — for reading a device-local buffer back to the CPU after a
    // transfer-write→host-read barrier + fence. Used by the headless VDPM GPU test harness.
    [[nodiscard]] MappedBufferSet createMappedReadbackBuffers(std::size_t size);
    // Single host-visible storage buffer (eStorageBuffer | eTransferDst) shared across all
    // frames-in-flight — the GPU serialises frames on the graphics queue, so no per-frame copies
    // are needed. eTransferDst lets callers clear/upload via vkCmdFillBuffer/copy. (Distinct from
    // createStorageBuffer, which adds eShaderDeviceAddress for the buffer_reference solver path.)
    [[nodiscard]] BufferHandle createSharedStorageBuffer(std::size_t size, const void* initialData);
    // Single persistent host-visible storage buffer with initial contents. Used
    // for the soft-body solver's particle / constraint buffers (per-instance sim
    // state, shared across frames — the GPU serialises frames on the graphics
    // queue, so no per-frame copies are needed).
    [[nodiscard]] BufferHandle createStorageBuffer(std::size_t size, const void* initialData);
    // Static, GPU-read-only storage buffer: device-local, filled once via a staging copy. For
    // long-lived SSBO data built at load and never CPU-rewritten (the VIPM geomorph table). Unlike
    // createStorageBuffer this is not host-visible and carries no device address — it is bound as a
    // plain SSBO, not chained by buffer_reference.
    [[nodiscard]] BufferHandle createStaticStorageBuffer(std::size_t size, const void* initialData);
    // Device-local storage buffer WITH a device address (BDA) — like createStaticStorageBuffer but
    // buffer_reference-addressable, for GPU compute that chains buffers by pointer (the VDPM GPU
    // front's static per-split metric input + persistent score/backface output). `additionalUsage`
    // ORs in extra usage flags on top of the storage+BDA+transfer-dst base — eTransferSrc for a
    // test readback, eIndexBuffer for a compute-emitted index stream (VDPM emit), eIndirectBuffer
    // for a compute-written indirect command (a superset arg rather than a growing boolean
    // interface).
    [[nodiscard]] BufferHandle
    createDeviceLocalStorageBuffer(std::size_t size, const void* initialData,
                                   vk::BufferUsageFlags additionalUsage = {});
    // The device's maxComputeWorkGroupCount[0] — the hard cap on a 1-D compute dispatch's group
    // count, so a caller can reject an over-large dispatch rather than fault the driver.
    [[nodiscard]] std::uint32_t maxComputeWorkGroupCountX() const noexcept;
    // Per-frame, persistently-mapped storage buffers with a device address (for
    // the soft-body solver's per-frame collider buffer, addressed via bDA).
    [[nodiscard]] MappedBufferSet createMappedDeviceAddressBuffers(std::size_t size);
    // GPU pointer (bufferDeviceAddress) of a buffer created with eShaderDeviceAddress.
    [[nodiscard]] vk::DeviceAddress bufferAddress(BufferHandle handle) const noexcept;
    [[nodiscard]] uint32_t allocateObjectId() noexcept;

    // --- Shadow map + offscreen textures ---

    // Allocates a 2D D32_SFLOAT image + view usable as a depth attachment and
    // as a sampled texture. Sampler uses comparison mode (CompareOp::eLess,
    // eClampToBorder, white border) so the forward shader can do hardware PCF
    // via sampler2DShadow. Returned handle slots into the existing texture
    // registry — retrieve view/sampler via the vulkan* accessors.
    // layerCount > 1 produces a 2D array image for cascaded shadow mapping;
    // per-layer 2D views are created for framebuffer attachment.
    [[nodiscard]] TextureHandle createShadowMap(uint32_t extent, uint32_t layerCount = 1);

    // Per-layer 2D view on a shadow-map array (framebuffer attachment use).
    [[nodiscard]] vk::ImageView vulkanShadowMapLayerView(TextureHandle handle,
                                                         uint32_t layer) const noexcept;

    // D32_SFLOAT cubemap-array depth image for point-light shadows. Total
    // layers = 6 * cubeCount. Main view is eCubeArray (sampled via
    // samplerCubeArrayShadow); per-face 2D views are created for framebuffer
    // attachment, indexed as `6 * cube + face`.
    [[nodiscard]] TextureHandle createPointShadowMap(uint32_t faceExtent, uint32_t cubeCount);

    // Per-face 2D depth view on a point shadow cubemap array. cubeIndex picks
    // the cube; face is 0..5 in Vulkan's cubemap face order (+X,-X,+Y,-Y,+Z,-Z).
    [[nodiscard]] vk::ImageView vulkanPointShadowFaceView(TextureHandle handle, uint32_t cubeIndex,
                                                          uint32_t face) const noexcept;

    // Allocates an R16G16B16A16_SFLOAT colour image sized to the given extent,
    // usable as both a colour attachment (forward pass target) and a sampled
    // texture (post-process input). Linear-filter sampler, ClampToEdge.
    [[nodiscard]] TextureHandle createOffscreenColourTarget(vk::Extent2D extent);

    // Allocates an R16G16_SFLOAT screen-space motion-vector target (TAA),
    // usable as a colour attachment and a sampled texture. Nearest-filter
    // sampler so velocities aren't blended across silhouettes.
    [[nodiscard]] TextureHandle createVelocityTarget(vk::Extent2D extent);

    // Allocates an R8_UNORM screen-space AO target (SSAO + contact term), usable
    // as a colour attachment (SSAO pass) and a sampled texture (forward shader).
    // Linear-filter sampler so the forward sampling smooths the AO.
    [[nodiscard]] TextureHandle createSsaoTarget(vk::Extent2D extent);

    // Multi-mip 2D HDR target used by the bloom downsample/upsample chain.
    // Per-mip 2D views are created for framebuffer attachment + shader input.
    [[nodiscard]] TextureHandle createBloomChain(uint32_t width, uint32_t height,
                                                 uint32_t mipLevels);

    [[nodiscard]] vk::ImageView vulkanBloomMipView(TextureHandle handle,
                                                   uint32_t mipLevel) const noexcept;

    // Multi-mip 2D HDR target used by KHR_materials_transmission F3
    // (screen-space refraction). Sized to the swapchain extent with
    // floor(log2(maxDim)) + 1 mips. Usage = TransferDst | TransferSrc | Sampled
    // so the renderer can blit the post-opaque HDR target into mip 0 and then
    // generate the rest of the chain via vkCmdBlitImage. Sampler uses linear
    // min/mag + linear mip-map for roughness-driven mip selection in the
    // shader's transmission lobe.
    [[nodiscard]] TextureHandle createSceneColorTarget(uint32_t width, uint32_t height,
                                                       uint32_t mipLevels);

    // Releases an existing offscreen / shadow texture entry so it can be
    // rebuilt at a new extent (e.g. on swapchain resize). The handle is
    // invalidated; callers must replace it with the result of a subsequent
    // createOffscreenColourTarget / createShadowColourAttachment call.
    void releaseTexture(TextureHandle handle);

    // True while `handle` still refers to a live texture — its slot's generation matches and
    // the slot has not been released. A handle to a since-released or recycled slot returns
    // false (CR-12/17), so stale handles are detectable rather than silently aliasing.
    [[nodiscard]] bool validTexture(TextureHandle handle) const noexcept;

    // Load-time upload batch (CR-16). Between begin/end, every texture upload records its
    // copy + layout transitions into one shared command buffer (retaining its staging buffer)
    // instead of submitting and blocking per texture; endUploadBatch submits once and waits on
    // a single fence. Brackets the whole asset-load phase, which completes before the first
    // frame. Uploads issued outside a batch fall back to an individual fenced submit.
    void beginUploadBatch();
    void endUploadBatch();

    [[nodiscard]] Descriptors& descriptors() noexcept
    {
        return descriptors_;
    }

    [[nodiscard]] const Descriptors& descriptors() const noexcept
    {
        return descriptors_;
    }

    // --- Bindless materials (forward set 2) ---

    // Write a 2D sampled texture into the global bindless array at array index ==
    // its handle value (the array is partially-bound, so non-2D handles just leave
    // gaps). Called from the 2D material-texture creation paths. update-after-bind
    // makes this safe even after the set has been bound in earlier frames.
    void registerBindlessTexture(TextureHandle handle);

    // The single global bindless descriptor set (forward set 2), bound once per
    // forward pass. Null until the owning forward pipeline declared a bindless set.
    [[nodiscard]] vk::DescriptorSet bindlessDescriptorSet() const noexcept
    {
        return static_cast<bool>(*bindlessSet_) ? *bindlessSet_ : vk::DescriptorSet{};
    }

    // Assign (or look up) a material's slot in the global materials[] SSBO, writing
    // its packed MaterialUBO on first registration. Deduplicated by Material
    // identity; idempotent, so draw setup can call it every frame cheaply. The
    // returned index goes into ForwardPushConstants::materialIndex. Returns 0 when
    // there's no bindless set (headless contexts).
    [[nodiscard]] uint32_t registerMaterial(const Material& material);

    // --- Shared light UBO (bound to every forward descriptor set) ---

    void lightBuffers(const std::array<BufferHandle, kMaxFramesInFlight>& bufs) noexcept
    {
        lightBuffers_ = bufs;
    }

    [[nodiscard]] const std::array<BufferHandle, kMaxFramesInFlight>& lightBuffers() const noexcept
    {
        return lightBuffers_;
    }

    // --- Shared textures bound to every forward / shadow descriptor set ---
    //
    // Each entry is populated once during renderer setup (Renderer for IBL,
    // Shadows for the shadow-pass attachments, Transmission for sceneColor)
    // and then read by descriptor-set writers and per-draw resolves. The
    // handles live together so callers can pass a single struct reference
    // through to descriptor-build helpers instead of threading eleven
    // arguments.
    struct SharedTextures
    {
        TextureHandle shadowMap{NullTexture};
        TextureHandle worldShadowMap{NullTexture};
        TextureHandle selfShadowMap{NullTexture};
        TextureHandle selfShadowFirstMap{NullTexture};
        TextureHandle spotShadowMap{NullTexture};
        TextureHandle pointShadowMap{NullTexture};
        TextureHandle shadowDebugImage{NullTexture};
        TextureHandle irradianceMap{NullTexture};
        TextureHandle prefilteredMap{NullTexture};
        TextureHandle brdfLut{NullTexture};
        // KHR_materials_transmission F3 — captured post-opaque scene-colour
        // mip chain. Bound at forward descriptor binding 20.
        TextureHandle sceneColor{NullTexture};
        // Screen-space AO + contact term (R8), written by the SSAO pass and
        // sampled in the forward shader to modulate ambient.
        TextureHandle ssaoMap{NullTexture};
    };

    [[nodiscard]] SharedTextures& sharedTextures() noexcept
    {
        return shared_;
    }

    [[nodiscard]] const SharedTextures& sharedTextures() const noexcept
    {
        return shared_;
    }

    // --- Pipeline registry ---
    // Pipelines are owned elsewhere (by Pipeline class); Resources stores raw
    // handles so Renderer can resolve PipelineHandle values stamped on
    // DrawCommands to the Vulkan pipeline/layout pair to bind.

    [[nodiscard]] PipelineHandle registerPipeline(vk::Pipeline pipeline, vk::PipelineLayout layout);

    // --- Vulkan accessors (for Renderer command recording) ---

    [[nodiscard]] vk::Buffer vulkanBuffer(BufferHandle handle) const noexcept;
    [[nodiscard]] vk::Image vulkanImage(TextureHandle handle) const noexcept;
    [[nodiscard]] vk::ImageView vulkanImageView(TextureHandle handle) const noexcept;
    [[nodiscard]] vk::ImageView vulkanCubemapFaceView(TextureHandle handle, uint32_t face,
                                                      uint32_t mipLevel = 0) const noexcept;
    [[nodiscard]] vk::Sampler vulkanSampler(TextureHandle handle) const noexcept;
    [[nodiscard]] vk::Sampler vulkanShadowDebugSampler() const noexcept;
    [[nodiscard]] vk::Format textureFormat(TextureHandle handle) const noexcept;
    [[nodiscard]] uint32_t textureMipLevels(TextureHandle handle) const noexcept;
    [[nodiscard]] vk::DescriptorSet vulkanDescriptorSet(DescriptorSetHandle handle) const noexcept;
    [[nodiscard]] vk::Pipeline vulkanPipeline(PipelineHandle handle) const noexcept;
    [[nodiscard]] vk::PipelineLayout vulkanPipelineLayout(PipelineHandle handle) const noexcept;

private:
    BufferHandle storeBuffer(UniqueVmaBuffer buffer);
    [[nodiscard]] BufferHandle createHostVisibleBuffer(vk::DeviceSize size,
                                                       vk::BufferUsageFlags usage,
                                                       const void* initialData = nullptr);
    // Device-local buffer filled once from `initialData` through a transient host-visible staging
    // buffer + a one-time copy submit. `usage` is OR'd with eTransferDst. For long-lived, GPU-only,
    // never-CPU-rewritten data (static vertices/indices, LOD index sets, VIPM tables).
    [[nodiscard]] BufferHandle createDeviceLocalBuffer(vk::DeviceSize size,
                                                       vk::BufferUsageFlags usage,
                                                       const void* initialData);
    [[nodiscard]] MappedBufferSet createMappedHostVisibleBuffers(std::size_t size,
                                                                 vk::BufferUsageFlags usage);
    [[nodiscard]] TextureHandle createFallbackTexture(FallbackTextureKind kind);

    const Device* device_;
    Descriptors descriptors_;

    // Global bindless material set (forward set 2): one update-after-bind pool, one
    // set, allocated from the forward pipeline's bindless layout. Empty when the
    // pipeline declared no bindless set.
    vk::raii::DescriptorPool bindlessPool_{nullptr};
    vk::raii::DescriptorSet bindlessSet_{nullptr};
    // Global materials[] SSBO (forward set 2, binding 1): one persistently-mapped
    // host-visible buffer, written per material on first registration. Material
    // identity → slot index. Immutable per slot once written, so shared across
    // frames without double-buffering.
    BufferHandle materialBuffer_{NullBuffer};
    std::span<std::byte> materialMapped_{};
    uint32_t materialCount_{0};
    std::unordered_map<const Material*, uint32_t> materialIndices_;

    vk::raii::CommandPool cmdPool_{nullptr};

    struct BufferEntry
    {
        // Owns the VkBuffer + its VMA sub-allocation as a unit (Approach A).
        UniqueVmaBuffer buffer;
    };
    std::vector<BufferEntry> buffers_;

    // Active load-time upload batch (CR-16); `active` is false outside a begin/endUploadBatch
    // scope. `staging` retains each upload's staging buffer until the single batch fence signals.
    struct UploadBatch
    {
        vk::raii::CommandBuffer cmd{nullptr};
        std::vector<UniqueVmaBuffer> staging;
        bool active{false};
    };
    UploadBatch uploadBatch_;

    struct TextureEntry
    {
        // Owns the VkImage + its VMA sub-allocation as a unit (Approach A); views/samplers
        // below stay stock vk::raii (their lifetime is independent of the allocation).
        UniqueVmaImage image;
        vk::raii::ImageView view{nullptr};
        vk::raii::Sampler sampler{nullptr};
        vk::Format format{vk::Format::eUndefined};
        std::vector<vk::raii::ImageView> faceViews;
        uint32_t mipLevels{1};
    };
    TextureEntry& appendTextureEntry(TextureHandle& handle, vk::Format format,
                                     uint32_t mipLevels = 1);
    void allocateImage(TextureEntry& entry, const vk::ImageCreateInfo& imageInfo);
    void createImageView(TextureEntry& entry, vk::ImageViewType viewType,
                         vk::ImageAspectFlags aspectMask, uint32_t baseMipLevel,
                         uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount);
    void createSampler(TextureEntry& entry, const vk::SamplerCreateInfo& samplerInfo);
    void createSampledTextureSampler(TextureEntry& entry, const SamplerSettings& sampler,
                                     uint32_t mipLevels, vk::BorderColor borderColor);
    struct Texture2DTargetDesc
    {
        vk::Format format{vk::Format::eUndefined};
        vk::Extent2D extent{};
        uint32_t mipLevels{1};
        vk::ImageUsageFlags usage{};
        vk::Filter filter{vk::Filter::eLinear};
        vk::SamplerMipmapMode mipmapMode{vk::SamplerMipmapMode::eNearest};
        float minLod{0.0f};
        float maxLod{1.0f};
        bool perMipViews{false};
        bool initialShaderReadOnlyLayout{false};
    };
    [[nodiscard]] TextureHandle createTexture2DTarget(const Texture2DTargetDesc& desc);
    [[nodiscard]] TextureHandle createUploaded2DTexture(const void* pixels, int width, int height,
                                                        vk::Format format,
                                                        vk::DeviceSize bytesPerPixel,
                                                        const SamplerSettings& sampler,
                                                        vk::BorderColor borderColor);
    // Stages `bytes` host bytes into a new device-local image described by
    // `imageInfo`, copies them via `regions`, and transitions the full
    // subresource range to ShaderReadOnlyOptimal. Populates entry.image and
    // entry.memory; the caller still owns view/sampler creation.
    void uploadImageFromHost(TextureEntry& entry, const void* pixels, vk::DeviceSize bytes,
                             const vk::ImageCreateInfo& imageInfo,
                             std::span<const vk::BufferImageCopy> regions);
    static void createCubemapFaceViews(const Device& device, TextureEntry& entry);
    std::vector<TextureEntry> textures_;
    // Slot lifecycle for textures_ (index + generation). Kept in lockstep with textures_ by
    // index; recycles released slots so the table stays bounded across resize churn (CR-17).
    GenerationalSlotPool textureSlots_;
    std::array<TextureHandle, 5> fallbackTextures_{NullTexture, NullTexture, NullTexture,
                                                   NullTexture, NullTexture};

    struct PipelineEntry
    {
        vk::Pipeline pipeline{};
        vk::PipelineLayout layout{};
    };
    std::vector<PipelineEntry> pipelines_;

    std::array<BufferHandle, kMaxFramesInFlight> lightBuffers_{NullBuffer, NullBuffer};
    SharedTextures shared_;
    vk::raii::Sampler shadowDebugSampler_{nullptr};
    uint32_t nextObjectId_{1};
};

} // namespace fire_engine
