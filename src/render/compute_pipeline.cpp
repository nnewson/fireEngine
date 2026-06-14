#include <array>
#include <cstddef>
#include <cstdint>

#include <fire_engine/core/shader_loader.hpp>
#include <fire_engine/render/compute_pipeline.hpp>

namespace fire_engine
{

ComputePipeline::ComputePipeline(const Device& device, const ComputePipelineConfig& config)
    : device_(&device.device())
{
    vk::DescriptorSetLayoutCreateInfo dslci{
        .bindingCount = static_cast<uint32_t>(config.bindings.size()),
        .pBindings = config.bindings.data(),
    };
    descSetLayout_ = vk::raii::DescriptorSetLayout(*device_, dslci);

    std::array<vk::DescriptorSetLayout, 1> setLayouts{*descSetLayout_};
    vk::PipelineLayoutCreateInfo plci{
        .setLayoutCount = 1,
        .pSetLayouts = setLayouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(config.pushConstantRanges.size()),
        .pPushConstantRanges = config.pushConstantRanges.data(),
    };
    pipelineLayout_ = vk::raii::PipelineLayout(*device_, plci);

    auto code = ShaderLoader::load_from_file(config.compShaderPath);
    vk::ShaderModuleCreateInfo smci{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };
    vk::raii::ShaderModule module(*device_, smci);

    vk::PipelineShaderStageCreateInfo stage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *module,
        .pName = "main",
    };
    vk::ComputePipelineCreateInfo cpci{
        .stage = stage,
        .layout = *pipelineLayout_,
    };
    pipeline_ = vk::raii::Pipeline(*device_, nullptr, cpci);
}

vk::BufferMemoryBarrier2 makeBufferMemoryBarrier(vk::PipelineStageFlags2 srcStage,
                                                 vk::AccessFlags2 srcAccess,
                                                 vk::PipelineStageFlags2 dstStage,
                                                 vk::AccessFlags2 dstAccess, vk::Buffer buffer,
                                                 vk::DeviceSize offset, vk::DeviceSize size)
{
    return vk::BufferMemoryBarrier2{
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .buffer = buffer,
        .offset = offset,
        .size = size,
    };
}

void recordBufferBarrier(vk::CommandBuffer cmd, const vk::BufferMemoryBarrier2& barrier)
{
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier});
}

} // namespace fire_engine
