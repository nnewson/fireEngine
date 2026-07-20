#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

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

    auto code = ShaderLoader::load_spirv_from_file(config.compShaderPath);
    vk::ShaderModuleCreateInfo smci{
        .codeSize = code.size() * sizeof(std::uint32_t),
        .pCode = code.data(),
    };
    vk::raii::ShaderModule module(*device_, smci);

    vk::PipelineShaderStageCreateInfo stage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *module,
        .pName = "main",
    };

    // Specialization constants (explicit IDs → contiguous uint32 data + one map entry each). Built
    // here so it outlives cpci; stays null when there are none, so unspecialized pipelines are
    // unchanged. Duplicate IDs are a programming error (undefined which value wins) → reject.
    std::vector<vk::SpecializationMapEntry> specEntries;
    std::vector<std::uint32_t> specData;
    vk::SpecializationInfo specInfo{};
    if (!config.specConstants.empty())
    {
        specEntries.reserve(config.specConstants.size());
        specData.reserve(config.specConstants.size());
        for (const ComputeSpecializationConstant& sc : config.specConstants)
        {
            for (const vk::SpecializationMapEntry& e : specEntries)
            {
                if (e.constantID == sc.id)
                {
                    throw std::runtime_error(
                        "ComputePipeline: duplicate specialization constant ID");
                }
            }
            specEntries.push_back(vk::SpecializationMapEntry{
                .constantID = sc.id,
                .offset = static_cast<std::uint32_t>(specData.size() * sizeof(std::uint32_t)),
                .size = sizeof(std::uint32_t)});
            specData.push_back(sc.value);
        }
        specInfo =
            vk::SpecializationInfo{.mapEntryCount = static_cast<std::uint32_t>(specEntries.size()),
                                   .pMapEntries = specEntries.data(),
                                   .dataSize = specData.size() * sizeof(std::uint32_t),
                                   .pData = specData.data()};
        stage.pSpecializationInfo = &specInfo;
    }

    vk::ComputePipelineCreateInfo cpci{
        .stage = stage,
        .layout = *pipelineLayout_,
    };
    pipeline_ = vk::raii::Pipeline(*device_, device.pipelineCache(), cpci);
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
