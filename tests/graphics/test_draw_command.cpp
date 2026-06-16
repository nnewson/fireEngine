#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <fire_engine/graphics/draw_command.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST_CASE("DrawCommand.DefaultVertexBufferIsNull", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK(cmd.vertexBuffer == NullBuffer);
}

TEST_CASE("DrawCommand.DefaultIndexBufferIsNull", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK(cmd.indexBuffer == NullBuffer);
}

TEST_CASE("DrawCommand.DefaultIndexCountIsZero", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK(cmd.indexCount == 0u);
}

TEST_CASE("DrawCommand.DefaultIndexTypeIsUInt16", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK(cmd.indexType == DrawIndexType::UInt16);
}

TEST_CASE("DrawCommand.DefaultDescriptorSetIsNull", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK(cmd.descriptorSet == NullDescriptorSet);
}

TEST_CASE("DrawCommand.DefaultPipelineIsNull", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK(cmd.pipeline == NullPipeline);
}

TEST_CASE("DrawCommand.DefaultSortDepthIsZero", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK(cmd.sortDepth == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("DrawCommand.AssignSortDepth", "[DrawCommand]")
{
    DrawCommand cmd;
    cmd.sortDepth = 42.5f;
    CHECK(cmd.sortDepth == Catch::Approx(42.5f).margin(1e-5f));
}

TEST_CASE("DrawCommand.DefaultTransmissiveIsFalse", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK_FALSE(cmd.transmissive);
}

TEST_CASE("DrawCommand.DefaultSelfShadowMetadataIsDisabled", "[DrawCommand]")
{
    DrawCommand cmd;
    CHECK(cmd.objectId == 0u);
    CHECK_FALSE(cmd.hasSkin);
    CHECK(cmd.selfShadowSlot == -1);
    CHECK_FALSE(cmd.shadowBounds.valid);
    CHECK(cmd.selfShadowViewProj == Mat4::identity());
}

TEST_CASE("DrawCommand.AssignTransmissive", "[DrawCommand]")
{
    DrawCommand cmd;
    cmd.transmissive = true;
    CHECK(cmd.transmissive);
}

// ---------------------------------------------------------------------------
// Member assignment
// ---------------------------------------------------------------------------

TEST_CASE("DrawCommand.AssignVertexBuffer", "[DrawCommand]")
{
    DrawCommand cmd;
    cmd.vertexBuffer = BufferHandle{5};
    CHECK(static_cast<uint32_t>(cmd.vertexBuffer) == 5u);
}

TEST_CASE("DrawCommand.AssignIndexBuffer", "[DrawCommand]")
{
    DrawCommand cmd;
    cmd.indexBuffer = BufferHandle{10};
    CHECK(static_cast<uint32_t>(cmd.indexBuffer) == 10u);
}

TEST_CASE("DrawCommand.AssignIndexCount", "[DrawCommand]")
{
    DrawCommand cmd;
    cmd.indexCount = 36;
    CHECK(cmd.indexCount == 36u);
}

TEST_CASE("DrawCommand.AssignIndexType", "[DrawCommand]")
{
    DrawCommand cmd;
    cmd.indexType = DrawIndexType::UInt32;
    CHECK(cmd.indexType == DrawIndexType::UInt32);
}

TEST_CASE("DrawCommand.AssignDescriptorSet", "[DrawCommand]")
{
    DrawCommand cmd;
    cmd.descriptorSet = DescriptorSetHandle{2};
    CHECK(static_cast<uint32_t>(cmd.descriptorSet) == 2u);
}

TEST_CASE("DrawCommand.AssignPipeline", "[DrawCommand]")
{
    DrawCommand cmd;
    cmd.pipeline = PipelineHandle{7};
    CHECK(static_cast<uint32_t>(cmd.pipeline) == 7u);
}

// ---------------------------------------------------------------------------
// Aggregate initialization
// ---------------------------------------------------------------------------

TEST_CASE("DrawCommand.AggregateInit", "[DrawCommand]")
{
    DrawCommand cmd{BufferHandle{1}, BufferHandle{2}, 24, DescriptorSetHandle{3},
                    PipelineHandle{4}};
    CHECK(static_cast<uint32_t>(cmd.vertexBuffer) == 1u);
    CHECK(static_cast<uint32_t>(cmd.indexBuffer) == 2u);
    CHECK(cmd.indexCount == 24u);
    CHECK(static_cast<uint32_t>(cmd.descriptorSet) == 3u);
    CHECK(static_cast<uint32_t>(cmd.pipeline) == 4u);
}

// ---------------------------------------------------------------------------
// Collection usage
// ---------------------------------------------------------------------------

TEST_CASE("DrawCommand.StorableInVector", "[DrawCommand]")
{
    std::vector<DrawCommand> commands;
    commands.push_back({BufferHandle{0}, BufferHandle{1}, 6, DescriptorSetHandle{0}});
    commands.push_back({BufferHandle{2}, BufferHandle{3}, 12, DescriptorSetHandle{1}});
    REQUIRE(commands.size() == 2u);
    CHECK(commands[0].indexCount == 6u);
    CHECK(commands[1].indexCount == 12u);
}

TEST_CASE("DrawCommand.ReserveAndEmplaceBack", "[DrawCommand]")
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
    REQUIRE(commands.size() == 3u);
    CHECK(commands[2].indexCount == 18u);
}

// ---------------------------------------------------------------------------
// Copy semantics
// ---------------------------------------------------------------------------

TEST_CASE("DrawCommand.CopyPreservesAllFields", "[DrawCommand]")
{
    DrawCommand original{BufferHandle{1}, BufferHandle{2}, 36, DescriptorSetHandle{4},
                         PipelineHandle{5}};
    original.indexType = DrawIndexType::UInt32;
    DrawCommand copy = original;
    CHECK(copy.vertexBuffer == original.vertexBuffer);
    CHECK(copy.indexBuffer == original.indexBuffer);
    CHECK(copy.indexCount == original.indexCount);
    CHECK(copy.indexType == original.indexType);
    CHECK(copy.descriptorSet == original.descriptorSet);
    CHECK(copy.pipeline == original.pipeline);
}
