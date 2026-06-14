#include <fire_engine/render/compute_self_test.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fire_engine/render/device.hpp>

namespace fire_engine
{

namespace
{

constexpr uint32_t kElementCount = 1024;
constexpr uint32_t kIncrement = 1;
constexpr uint32_t kLocalSizeX = 64;

// Matches the push_constant block in compute_selftest.comp.
struct SelfTestPush
{
    uint32_t count;
    uint32_t increment;
};

} // namespace

ComputePipelineConfig computeSelfTestConfig()
{
    ComputePipelineConfig config;
    config.compShaderPath = "compute_selftest.comp.spv";
    config.bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eCompute, 0,
                                           static_cast<uint32_t>(sizeof(SelfTestPush)));
    return config;
}

void runComputeSelfTest(const Device& device)
{
    const vk::raii::Device& dev = device.device();
    const vk::DeviceSize bufferSize = sizeof(uint32_t) * kElementCount;

    // Host-visible/coherent SSBO, owned locally so it can be mapped for readback.
    // Seed values[i] = i so the dispatch's per-index coverage is verifiable.
    auto [buffer, memory] = device.createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                                                vk::MemoryPropertyFlagBits::eHostVisible |
                                                    vk::MemoryPropertyFlagBits::eHostCoherent);
    {
        auto* values = static_cast<uint32_t*>(memory.mapMemory(0, bufferSize));
        for (uint32_t i = 0; i < kElementCount; ++i)
        {
            values[i] = i;
        }
        memory.unmapMemory();
    }

    ComputePipeline pipeline(device, computeSelfTestConfig());

    // Standalone descriptor pool + set bound to the SSBO.
    vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
    };
    vk::DescriptorPoolCreateInfo poolCi{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    vk::raii::DescriptorPool descriptorPool(dev, poolCi);

    vk::DescriptorSetLayout setLayout = pipeline.descriptorSetLayout();
    vk::DescriptorSetAllocateInfo setAi{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &setLayout,
    };
    vk::raii::DescriptorSets sets(dev, setAi);
    vk::DescriptorSet set = *sets[0];

    vk::DescriptorBufferInfo bufInfo{
        .buffer = *buffer,
        .offset = 0,
        .range = bufferSize,
    };
    vk::WriteDescriptorSet write{
        .dstSet = set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pBufferInfo = &bufInfo,
    };
    dev.updateDescriptorSets(write, {});

    // Transient command buffer: dispatch -> buffer barrier -> dispatch. The
    // barrier (compute write -> compute read) is what actually gets exercised;
    // waitIdle alone would mask a missing or wrong barrier.
    vk::CommandPoolCreateInfo cmdPoolCi{
        .flags = vk::CommandPoolCreateFlagBits::eTransient,
        .queueFamilyIndex = device.graphicsFamily(),
    };
    vk::raii::CommandPool cmdPool(dev, cmdPoolCi);
    vk::CommandBufferAllocateInfo cmdAi{
        .commandPool = *cmdPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    auto cmds = dev.allocateCommandBuffers(cmdAi);
    auto& cmd = cmds[0];

    const uint32_t groups = (kElementCount + kLocalSizeX - 1) / kLocalSizeX;
    const SelfTestPush push{kElementCount, kIncrement};

    cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.pipelineLayout(), 0, set, {});
    cmd.pushConstants<SelfTestPush>(pipeline.pipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                                    push);
    cmd.dispatch(groups, 1, 1);

    recordBufferBarrier(*cmd, makeBufferMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader,
                                                      vk::AccessFlagBits2::eShaderStorageWrite,
                                                      vk::PipelineStageFlagBits2::eComputeShader,
                                                      vk::AccessFlagBits2::eShaderStorageRead,
                                                      *buffer, 0, bufferSize));

    cmd.dispatch(groups, 1, 1);
    cmd.end();

    vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
    vk::SubmitInfo2 submit{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo};
    device.graphicsQueue().submit2(submit);
    device.graphicsQueue().waitIdle();

    // Readback: each element saw two increments, so values[i] == i + 2 * increment.
    auto* values = static_cast<uint32_t*>(memory.mapMemory(0, bufferSize));
    for (uint32_t i = 0; i < kElementCount; ++i)
    {
        const uint32_t expected = i + 2u * kIncrement;
        if (values[i] != expected)
        {
            memory.unmapMemory();
            throw std::runtime_error("compute self-test FAILED at index " + std::to_string(i) +
                                     ": expected " + std::to_string(expected) + ", got " +
                                     std::to_string(values[i]));
        }
    }
    memory.unmapMemory();
    std::cout << "compute path OK" << std::endl;
}

} // namespace fire_engine
