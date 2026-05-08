#include <gtest/gtest.h>

#include <vector>

#include <fire_engine/graphics/draw_command.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(DrawCommand, DefaultVertexBufferIsNull)
{
    DrawCommand cmd;
    EXPECT_EQ(cmd.vertexBuffer, NullBuffer);
}

TEST(DrawCommand, DefaultIndexBufferIsNull)
{
    DrawCommand cmd;
    EXPECT_EQ(cmd.indexBuffer, NullBuffer);
}

TEST(DrawCommand, DefaultIndexCountIsZero)
{
    DrawCommand cmd;
    EXPECT_EQ(cmd.indexCount, 0u);
}

TEST(DrawCommand, DefaultIndexTypeIsUInt16)
{
    DrawCommand cmd;
    EXPECT_EQ(cmd.indexType, DrawIndexType::UInt16);
}

TEST(DrawCommand, DefaultDescriptorSetIsNull)
{
    DrawCommand cmd;
    EXPECT_EQ(cmd.descriptorSet, NullDescriptorSet);
}

TEST(DrawCommand, DefaultPipelineIsNull)
{
    DrawCommand cmd;
    EXPECT_EQ(cmd.pipeline, NullPipeline);
}

TEST(DrawCommand, DefaultSortDepthIsZero)
{
    DrawCommand cmd;
    EXPECT_FLOAT_EQ(cmd.sortDepth, 0.0f);
}

TEST(DrawCommand, AssignSortDepth)
{
    DrawCommand cmd;
    cmd.sortDepth = 42.5f;
    EXPECT_FLOAT_EQ(cmd.sortDepth, 42.5f);
}

TEST(DrawCommand, DefaultTransmissiveIsFalse)
{
    DrawCommand cmd;
    EXPECT_FALSE(cmd.transmissive);
}

TEST(DrawCommand, DefaultSelfShadowMetadataIsDisabled)
{
    DrawCommand cmd;
    EXPECT_EQ(cmd.objectId, 0u);
    EXPECT_FALSE(cmd.hasSkin);
    EXPECT_EQ(cmd.selfShadowSlot, -1);
    EXPECT_FALSE(cmd.shadowBounds.valid);
    EXPECT_EQ(cmd.selfShadowViewProj, Mat4::identity());
}

TEST(DrawCommand, AssignTransmissive)
{
    DrawCommand cmd;
    cmd.transmissive = true;
    EXPECT_TRUE(cmd.transmissive);
}

// ---------------------------------------------------------------------------
// Member assignment
// ---------------------------------------------------------------------------

TEST(DrawCommand, AssignVertexBuffer)
{
    DrawCommand cmd;
    cmd.vertexBuffer = BufferHandle{5};
    EXPECT_EQ(static_cast<uint32_t>(cmd.vertexBuffer), 5u);
}

TEST(DrawCommand, AssignIndexBuffer)
{
    DrawCommand cmd;
    cmd.indexBuffer = BufferHandle{10};
    EXPECT_EQ(static_cast<uint32_t>(cmd.indexBuffer), 10u);
}

TEST(DrawCommand, AssignIndexCount)
{
    DrawCommand cmd;
    cmd.indexCount = 36;
    EXPECT_EQ(cmd.indexCount, 36u);
}

TEST(DrawCommand, AssignIndexType)
{
    DrawCommand cmd;
    cmd.indexType = DrawIndexType::UInt32;
    EXPECT_EQ(cmd.indexType, DrawIndexType::UInt32);
}

TEST(DrawCommand, AssignDescriptorSet)
{
    DrawCommand cmd;
    cmd.descriptorSet = DescriptorSetHandle{2};
    EXPECT_EQ(static_cast<uint32_t>(cmd.descriptorSet), 2u);
}

TEST(DrawCommand, AssignPipeline)
{
    DrawCommand cmd;
    cmd.pipeline = PipelineHandle{7};
    EXPECT_EQ(static_cast<uint32_t>(cmd.pipeline), 7u);
}

// ---------------------------------------------------------------------------
// Aggregate initialization
// ---------------------------------------------------------------------------

TEST(DrawCommand, AggregateInit)
{
    DrawCommand cmd{BufferHandle{1}, BufferHandle{2}, 24, DescriptorSetHandle{3},
                    PipelineHandle{4}};
    EXPECT_EQ(static_cast<uint32_t>(cmd.vertexBuffer), 1u);
    EXPECT_EQ(static_cast<uint32_t>(cmd.indexBuffer), 2u);
    EXPECT_EQ(cmd.indexCount, 24u);
    EXPECT_EQ(static_cast<uint32_t>(cmd.descriptorSet), 3u);
    EXPECT_EQ(static_cast<uint32_t>(cmd.pipeline), 4u);
}

// ---------------------------------------------------------------------------
// Collection usage
// ---------------------------------------------------------------------------

TEST(DrawCommand, StorableInVector)
{
    std::vector<DrawCommand> commands;
    commands.push_back({BufferHandle{0}, BufferHandle{1}, 6, DescriptorSetHandle{0}});
    commands.push_back({BufferHandle{2}, BufferHandle{3}, 12, DescriptorSetHandle{1}});
    EXPECT_EQ(commands.size(), 2u);
    EXPECT_EQ(commands[0].indexCount, 6u);
    EXPECT_EQ(commands[1].indexCount, 12u);
}

TEST(DrawCommand, ReserveAndEmplaceBack)
{
    std::vector<DrawCommand> commands;
    commands.reserve(3);
    for (uint32_t i = 0; i < 3; ++i)
    {
        DrawCommand cmd;
        cmd.vertexBuffer = BufferHandle{i};
        cmd.indexCount = (i + 1) * 6;
        commands.push_back(cmd);
    }
    EXPECT_EQ(commands.size(), 3u);
    EXPECT_EQ(commands[2].indexCount, 18u);
}

// ---------------------------------------------------------------------------
// Copy semantics
// ---------------------------------------------------------------------------

TEST(DrawCommand, CopyPreservesAllFields)
{
    DrawCommand original{BufferHandle{1}, BufferHandle{2}, 36, DescriptorSetHandle{4},
                         PipelineHandle{5}};
    original.indexType = DrawIndexType::UInt32;
    DrawCommand copy = original;
    EXPECT_EQ(copy.vertexBuffer, original.vertexBuffer);
    EXPECT_EQ(copy.indexBuffer, original.indexBuffer);
    EXPECT_EQ(copy.indexCount, original.indexCount);
    EXPECT_EQ(copy.indexType, original.indexType);
    EXPECT_EQ(copy.descriptorSet, original.descriptorSet);
    EXPECT_EQ(copy.pipeline, original.pipeline);
}
