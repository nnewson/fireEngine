#pragma once

#include <array>
#include <string>

#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;

class EnvironmentPrecompute
{
public:
    EnvironmentPrecompute(const Device& device, Resources& resources, std::string environmentPath);
    ~EnvironmentPrecompute() = default;

    EnvironmentPrecompute(const EnvironmentPrecompute&) = delete;
    EnvironmentPrecompute& operator=(const EnvironmentPrecompute&) = delete;
    EnvironmentPrecompute(EnvironmentPrecompute&&) noexcept = default;
    EnvironmentPrecompute& operator=(EnvironmentPrecompute&&) noexcept = default;

    void create(vk::DescriptorSetLayout skyboxDescriptorSetLayout,
                const Resources::MappedBufferSet& skyboxUbo, vk::DeviceSize skyboxUboSize,
                const Resources::MappedBufferSet& lightUbo, vk::DeviceSize lightUboSize);

    [[nodiscard]] TextureHandle skyboxCubemap() const noexcept
    {
        return skyboxCubemapHandle_;
    }

    [[nodiscard]] TextureHandle irradianceCubemap() const noexcept
    {
        return irradianceCubemapHandle_;
    }

    [[nodiscard]] TextureHandle prefilteredCubemap() const noexcept
    {
        return prefilteredCubemapHandle_;
    }

    [[nodiscard]] TextureHandle brdfLut() const noexcept
    {
        return brdfLutHandle_;
    }

    [[nodiscard]] const std::array<DescriptorSetHandle, kMaxFramesInFlight>&
    skyboxDescriptorSets() const noexcept
    {
        return skyboxDescriptorSets_;
    }

private:
    void createSkyboxEnvironment(vk::DescriptorSetLayout skyboxDescriptorSetLayout,
                                 const Resources::MappedBufferSet& skyboxUbo,
                                 vk::DeviceSize skyboxUboSize,
                                 const Resources::MappedBufferSet& lightUbo,
                                 vk::DeviceSize lightUboSize);
    void createIrradianceEnvironment();
    void createPrefilteredEnvironment();
    void createBrdfLut();

    const Device* device_{nullptr};
    Resources* resources_{nullptr};
    std::string environmentPath_;
    TextureHandle skyboxCubemapHandle_{NullTexture};
    TextureHandle irradianceCubemapHandle_{NullTexture};
    TextureHandle prefilteredCubemapHandle_{NullTexture};
    TextureHandle brdfLutHandle_{NullTexture};
    std::array<DescriptorSetHandle, kMaxFramesInFlight> skyboxDescriptorSets_{};
};

} // namespace fire_engine
