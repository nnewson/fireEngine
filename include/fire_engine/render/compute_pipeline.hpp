#pragma once

#include <string>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/render/device.hpp>

namespace fire_engine
{

// Describes a compute pipeline: a single compute shader, its descriptor-set
// bindings (one set), and push-constant ranges. The graphics counterpart is
// PipelineConfig in pipeline.hpp.
struct ComputePipelineConfig
{
    std::string compShaderPath;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    std::vector<vk::PushConstantRange> pushConstantRanges;
};

// Owns a compute VkPipeline plus its single descriptor-set layout and pipeline
// layout. Mirrors the graphics Pipeline class but with one compute stage and a
// VkComputePipelineCreateInfo.
class ComputePipeline
{
public:
    ComputePipeline(const Device& device, const ComputePipelineConfig& config);
    ~ComputePipeline() = default;

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(ComputePipeline&&) noexcept = default;
    ComputePipeline& operator=(ComputePipeline&&) noexcept = default;

    [[nodiscard]] vk::DescriptorSetLayout descriptorSetLayout() const noexcept
    {
        return *descSetLayout_;
    }
    [[nodiscard]] vk::PipelineLayout pipelineLayout() const noexcept
    {
        return *pipelineLayout_;
    }
    [[nodiscard]] vk::Pipeline pipeline() const noexcept
    {
        return *pipeline_;
    }

private:
    const vk::raii::Device* device_{nullptr};
    vk::raii::DescriptorSetLayout descSetLayout_{nullptr};
    vk::raii::PipelineLayout pipelineLayout_{nullptr};
    vk::raii::Pipeline pipeline_{nullptr};
};

// Synchronization2 buffer-memory-barrier helpers. The renderer only had image
// barriers; compute work that hands a buffer to a later stage (e.g. a particle
// SSBO written by compute then read by the vertex shader) needs these.
[[nodiscard]] vk::BufferMemoryBarrier2
makeBufferMemoryBarrier(vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                        vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess,
                        vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size);

void recordBufferBarrier(vk::CommandBuffer cmd, const vk::BufferMemoryBarrier2& barrier);

} // namespace fire_engine
