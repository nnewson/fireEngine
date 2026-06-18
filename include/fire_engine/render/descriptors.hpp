#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/descriptor_bindings.hpp>

namespace fire_engine
{

class Device;
class Pipeline;
class Resources;
struct DrawCommand;

struct MappedBufferSet
{
    std::array<BufferHandle, kMaxFramesInFlight> buffers{NullBuffer, NullBuffer};
    std::array<MappedMemory, kMaxFramesInFlight> mapped{};
};

// Per-frame globals for the forward pipeline's set 1 — bound once at the
// start of each forward pass, reused by every draw. Lifetime = renderer
// lifetime; rewritten on swapchain resize when sceneColor / shadow maps get
// recreated.
struct GlobalDescriptorRequest
{
    std::array<BufferHandle, kMaxFramesInFlight> lightBufs{NullBuffer, NullBuffer};
    TextureHandle shadowMap{NullTexture};
    TextureHandle worldShadowMap{NullTexture};
    TextureHandle selfShadowMap{NullTexture};
    TextureHandle spotShadowMap{NullTexture};
    TextureHandle pointShadowMap{NullTexture};
    TextureHandle shadowDebugImage{NullTexture};
    TextureHandle irradianceMap{NullTexture};
    TextureHandle prefilteredMap{NullTexture};
    TextureHandle brdfLut{NullTexture};
    TextureHandle sceneColor{NullTexture};
    TextureHandle ssaoMap{NullTexture};
};

class Descriptors
{
public:
    Descriptors(const Device& device, const Pipeline& pipeline, const Resources& resources);
    ~Descriptors() = default;

    Descriptors(const Descriptors&) = delete;
    Descriptors& operator=(const Descriptors&) = delete;
    Descriptors(Descriptors&&) noexcept = default;
    Descriptors& operator=(Descriptors&&) noexcept = default;

    // Allocates kMaxFramesInFlight descriptor sets for the forward
    // pipeline's set 1 layout and writes every global binding (light UBO,
    // shadow maps, IBL, sceneColor).
    [[nodiscard]] std::array<DescriptorSetHandle, kMaxFramesInFlight>
    createGlobalDescriptors(const GlobalDescriptorRequest& req);

    // Rewrites every binding on the supplied global descriptor sets to point
    // at the textures/buffers in `req`. Called on swapchain resize after
    // Transmission::recreate (and any future shadow-map recreations) so the
    // existing sets stop dangling against destroyed samplers/views. Pool /
    // set allocation is untouched.
    void updateGlobalDescriptors(const std::array<DescriptorSetHandle, kMaxFramesInFlight>& sets,
                                 const GlobalDescriptorRequest& req);

    [[nodiscard]] std::array<DescriptorSetHandle, kMaxFramesInFlight>
    createSingleUboDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& ubo,
                               vk::DeviceSize uboSize);

    [[nodiscard]] std::array<DescriptorSetHandle, kMaxFramesInFlight>
    createUboImageSamplerDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& ubo,
                                     vk::DeviceSize uboSize, TextureHandle texture);

    [[nodiscard]] std::array<DescriptorSetHandle, kMaxFramesInFlight>
    createSkyboxDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& skyboxUbo,
                            vk::DeviceSize skyboxUboSize, TextureHandle texture,
                            const MappedBufferSet& lightUbo, vk::DeviceSize lightUboSize);

    [[nodiscard]] std::array<DescriptorSetHandle, kMaxFramesInFlight>
    createSingleImageSamplerDescriptors(vk::DescriptorSetLayout layout, TextureHandle texture);

    void updateSingleImageSamplerDescriptors(
        const std::array<DescriptorSetHandle, kMaxFramesInFlight>& sets, TextureHandle texture);

    [[nodiscard]] DescriptorSetHandle createImageViewDescriptor(vk::DescriptorSetLayout layout,
                                                                vk::ImageView view,
                                                                vk::Sampler sampler);

    [[nodiscard]] std::array<DescriptorSetHandle, kMaxFramesInFlight>
    createPostProcessDescriptors(vk::DescriptorSetLayout layout, TextureHandle hdrTarget,
                                 TextureHandle bloomChain);

    void
    updatePostProcessDescriptors(const std::array<DescriptorSetHandle, kMaxFramesInFlight>& sets,
                                 TextureHandle hdrTarget, TextureHandle bloomChain);

    // TAA resolve descriptors — one set per ping-pong parity. Set [p] reads the
    // current scene colour + velocity and the *opposite* history slot
    // (history[1 - p]) as the previous frame's accumulation.
    [[nodiscard]] std::array<DescriptorSetHandle, kMaxFramesInFlight>
    createTaaResolveDescriptors(vk::DescriptorSetLayout layout, TextureHandle currentColour,
                                TextureHandle velocity,
                                const std::array<TextureHandle, kMaxFramesInFlight>& history);

    void
    updateTaaResolveDescriptors(const std::array<DescriptorSetHandle, kMaxFramesInFlight>& sets,
                                TextureHandle currentColour, TextureHandle velocity,
                                const std::array<TextureHandle, kMaxFramesInFlight>& history);

    [[nodiscard]] vk::DescriptorSet vulkanDescriptorSet(DescriptorSetHandle handle) const noexcept;

private:
    struct DescriptorPoolEntry
    {
        vk::raii::DescriptorPool pool{nullptr};
        std::vector<vk::raii::DescriptorSet> sets;
    };

    [[nodiscard]] static vk::DescriptorBufferInfo makeDescriptorBufferInfo(vk::Buffer buffer,
                                                                           vk::DeviceSize range);
    [[nodiscard]] static vk::DescriptorImageInfo
    makeDescriptorImageInfo(vk::Sampler sampler, vk::ImageView imageView,
                            vk::ImageLayout imageLayout);
    // Submits the 13-write update for one frame's global descriptor set.
    // Shared by createGlobalDescriptors (initial) and updateGlobalDescriptors
    // (swapchain resize) so the binding order stays in lockstep with
    // ForwardGlobalBinding in a single place.
    void writeGlobalBindings(vk::DescriptorSet set, const GlobalDescriptorRequest& req,
                             int frame) const;

    DescriptorPoolEntry& createDescriptorPool(std::span<const vk::DescriptorPoolSize> poolSizes,
                                              uint32_t maxSets);
    [[nodiscard]] std::vector<vk::raii::DescriptorSet>
    allocateDescriptorSets(vk::DescriptorPool pool, vk::DescriptorSetLayout layout,
                           uint32_t count) const;

    // Called once per frame-in-flight to populate that frame's descriptor set.
    // The callback builds its DescriptorImageInfo/BufferInfo locals and issues
    // updateDescriptorSets itself, so those infos stay alive across the write.
    using FrameWriter = std::function<void(vk::DescriptorSet set, int frame)>;

    // Shared create envelope: allocate kMaxFramesInFlight sets of `layout` from
    // `poolEntry`, run `writeFrame` on each (skipped when empty — callers that
    // populate via a separate update*() helper pass {}), register and retain
    // them, and return the per-frame handles.
    std::array<DescriptorSetHandle, kMaxFramesInFlight>
    allocateFrameSets(DescriptorPoolEntry& poolEntry, vk::DescriptorSetLayout layout,
                      const FrameWriter& writeFrame);

    // allocateFrameSets plus a fresh pool sized by `poolSizes` (maxSets =
    // kMaxFramesInFlight) — the common single-group-per-frame case.
    std::array<DescriptorSetHandle, kMaxFramesInFlight>
    buildFrameSets(std::span<const vk::DescriptorPoolSize> poolSizes,
                   vk::DescriptorSetLayout layout, const FrameWriter& writeFrame);
    [[nodiscard]] DescriptorSetHandle registerDescriptorSet(vk::DescriptorSet set);
    void retainDescriptorSets(DescriptorPoolEntry& poolEntry,
                              std::vector<vk::raii::DescriptorSet>& sets);

    const Device* device_{nullptr};
    const Pipeline* pipeline_{nullptr};
    const Resources* resources_{nullptr};
    std::vector<DescriptorPoolEntry> descriptorPools_{};
    std::vector<vk::DescriptorSet> descriptorSetTable_{};
};

// Pushes a forward draw's per-object set 0 (frame/skin/morph UBOs + morph SSBO)
// inline via VK_KHR_push_descriptor — no allocated descriptor set. Shared by the
// forward pass and the transmission pass; the buffer handles come off the
// DrawCommand. `layout` is the draw's pipeline layout (set 0 is a push layout).
void pushForwardObjectDescriptors(vk::CommandBuffer cmd, const Resources& resources,
                                  vk::PipelineLayout layout, const DrawCommand& dc);

// Pushes a shadow draw's per-object set 0 inline via VK_KHR_push_descriptor — no
// allocated descriptor set. Bindings 0..3 are the per-object ShadowUBO + the
// skin/morph UBOs + morph SSBO carried on the DrawCommand; bindings 4/5 are the
// shared self-shadow first-depth image + sampler, read straight from Resources
// (global, identical for every shadow draw). `layout` is the shadow pipeline
// layout (set 0 is a push layout).
void pushShadowObjectDescriptors(vk::CommandBuffer cmd, const Resources& resources,
                                 vk::PipelineLayout layout, const DrawCommand& dc);

} // namespace fire_engine
