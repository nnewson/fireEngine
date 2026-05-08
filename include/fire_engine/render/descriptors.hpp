#pragma once

#include <array>
#include <cstddef>
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

struct MappedBufferSet
{
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> buffers{NullBuffer, NullBuffer};
    std::array<MappedMemory, MAX_FRAMES_IN_FLIGHT> mapped{};
};

struct GeometryDescriptorInfo
{
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> materialBufs{NullBuffer, NullBuffer};
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> skinBufs{NullBuffer, NullBuffer};
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> morphUboBufs{NullBuffer, NullBuffer};
    BufferHandle morphSsbo{NullBuffer};
    std::size_t morphSsboSize{0};
    std::array<TextureHandle, materialTextureSlotCount> materialTextures{
        NullTexture, NullTexture, NullTexture, NullTexture, NullTexture,
        NullTexture, NullTexture, NullTexture, NullTexture, NullTexture};
};

struct ObjectDescriptorRequest
{
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> uniformBufs{NullBuffer, NullBuffer};
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> lightBufs{NullBuffer, NullBuffer};
    TextureHandle shadowMap{NullTexture};
    TextureHandle worldShadowMap{NullTexture};
    TextureHandle selfShadowMap{NullTexture};
    TextureHandle spotShadowMap{NullTexture};
    TextureHandle pointShadowMap{NullTexture};
    TextureHandle shadowDebugImage{NullTexture};
    TextureHandle irradianceMap{NullTexture};
    TextureHandle prefilteredMap{NullTexture};
    TextureHandle brdfLut{NullTexture};
    // KHR_materials_transmission F3 — captured post-opaque HDR scene colour
    // mip chain. Bound at forward descriptor binding 20.
    TextureHandle sceneColor{NullTexture};
    std::vector<GeometryDescriptorInfo> geometries;
};

struct ObjectDescriptorResult
{
    std::vector<std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>> descSets;
};

struct ShadowGeometryDescriptorInfo
{
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> shadowUboBufs{NullBuffer, NullBuffer};
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> skinBufs{NullBuffer, NullBuffer};
    std::array<BufferHandle, MAX_FRAMES_IN_FLIGHT> morphUboBufs{NullBuffer, NullBuffer};
    BufferHandle morphSsbo{NullBuffer};
    std::size_t morphSsboSize{0};
};

struct ShadowDescriptorRequest
{
    std::vector<ShadowGeometryDescriptorInfo> geometries;
};

struct ShadowDescriptorResult
{
    std::vector<std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>> descSets;
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

    [[nodiscard]] ObjectDescriptorResult
    createObjectDescriptors(const ObjectDescriptorRequest& req);
    void updateObjectGeometryTextures(DescriptorSetHandle set,
                                      const GeometryDescriptorInfo& geometry);
    [[nodiscard]] ShadowDescriptorResult
    createShadowDescriptors(const ShadowDescriptorRequest& req);

    void shadowDescriptorSetLayout(vk::DescriptorSetLayout layout) noexcept
    {
        shadowDescLayout_ = layout;
    }

    [[nodiscard]] vk::DescriptorSetLayout shadowDescriptorSetLayout() const noexcept
    {
        return shadowDescLayout_;
    }

    [[nodiscard]] std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
    createSingleUboDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& ubo,
                               vk::DeviceSize uboSize);

    [[nodiscard]] std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
    createUboImageSamplerDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& ubo,
                                     vk::DeviceSize uboSize, TextureHandle texture);

    [[nodiscard]] std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
    createSkyboxDescriptors(vk::DescriptorSetLayout layout, const MappedBufferSet& skyboxUbo,
                            vk::DeviceSize skyboxUboSize, TextureHandle texture,
                            const MappedBufferSet& lightUbo, vk::DeviceSize lightUboSize);

    [[nodiscard]] std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
    createSingleImageSamplerDescriptors(vk::DescriptorSetLayout layout, TextureHandle texture);

    void updateSingleImageSamplerDescriptors(
        const std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>& sets, TextureHandle texture);

    [[nodiscard]] DescriptorSetHandle createImageViewDescriptor(vk::DescriptorSetLayout layout,
                                                                vk::ImageView view,
                                                                vk::Sampler sampler);

    [[nodiscard]] std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>
    createPostProcessDescriptors(vk::DescriptorSetLayout layout, TextureHandle hdrTarget,
                                 TextureHandle bloomChain);

    void
    updatePostProcessDescriptors(const std::array<DescriptorSetHandle, MAX_FRAMES_IN_FLIGHT>& sets,
                                 TextureHandle hdrTarget, TextureHandle bloomChain);

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

    DescriptorPoolEntry& createDescriptorPool(std::span<const vk::DescriptorPoolSize> poolSizes,
                                              uint32_t maxSets);
    [[nodiscard]] std::vector<vk::raii::DescriptorSet>
    allocateDescriptorSets(vk::DescriptorPool pool, vk::DescriptorSetLayout layout,
                           uint32_t count) const;
    [[nodiscard]] DescriptorSetHandle registerDescriptorSet(vk::DescriptorSet set);
    void retainDescriptorSets(DescriptorPoolEntry& poolEntry,
                              std::vector<vk::raii::DescriptorSet>& sets);

    const Device* device_{nullptr};
    const Pipeline* pipeline_{nullptr};
    const Resources* resources_{nullptr};
    std::vector<DescriptorPoolEntry> descriptorPools_{};
    std::vector<vk::DescriptorSet> descriptorSetTable_{};
    vk::DescriptorSetLayout shadowDescLayout_{};
};

} // namespace fire_engine
