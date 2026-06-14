#include <gtest/gtest.h>

#include <fire_engine/render/compute_self_test.hpp>

using fire_engine::computeSelfTestConfig;

TEST(ComputePipelineConfig, SelfTestConfigBindsOneStorageBuffer)
{
    auto config = computeSelfTestConfig();

    EXPECT_EQ(config.compShaderPath, "compute_selftest.comp.spv");

    ASSERT_EQ(config.bindings.size(), 1u);
    EXPECT_EQ(config.bindings[0].binding, 0u);
    EXPECT_EQ(config.bindings[0].descriptorType, vk::DescriptorType::eStorageBuffer);
    EXPECT_EQ(config.bindings[0].descriptorCount, 1u);
    EXPECT_EQ(config.bindings[0].stageFlags, vk::ShaderStageFlagBits::eCompute);
}

TEST(ComputePipelineConfig, SelfTestConfigHasComputePushConstant)
{
    auto config = computeSelfTestConfig();

    ASSERT_EQ(config.pushConstantRanges.size(), 1u);
    EXPECT_EQ(config.pushConstantRanges[0].stageFlags, vk::ShaderStageFlagBits::eCompute);
    EXPECT_EQ(config.pushConstantRanges[0].offset, 0u);
    // Push block is two uint32s (count + increment).
    EXPECT_EQ(config.pushConstantRanges[0].size, 2u * sizeof(uint32_t));
}
