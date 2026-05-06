#include <gtest/gtest.h>

#include <fire_engine/graphics/frame_info.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(FrameInfo, DefaultCurrentFrameIsZero)
{
    FrameInfo info;
    EXPECT_EQ(info.currentFrame, 0u);
}

TEST(FrameInfo, DefaultViewportWidthIsZero)
{
    FrameInfo info;
    EXPECT_EQ(info.viewportWidth, 0u);
}

TEST(FrameInfo, DefaultViewportHeightIsZero)
{
    FrameInfo info;
    EXPECT_EQ(info.viewportHeight, 0u);
}

TEST(FrameInfo, DefaultCameraPositionIsOrigin)
{
    FrameInfo info;
    EXPECT_FLOAT_EQ(info.cameraPosition.x(), 0.0f);
    EXPECT_FLOAT_EQ(info.cameraPosition.y(), 0.0f);
    EXPECT_FLOAT_EQ(info.cameraPosition.z(), 0.0f);
}

TEST(FrameInfo, DefaultCameraTargetIsOrigin)
{
    FrameInfo info;
    EXPECT_FLOAT_EQ(info.cameraTarget.x(), 0.0f);
    EXPECT_FLOAT_EQ(info.cameraTarget.y(), 0.0f);
    EXPECT_FLOAT_EQ(info.cameraTarget.z(), 0.0f);
}

TEST(FrameInfo, DefaultAlphaPipelinesAreNull)
{
    FrameInfo info;
    EXPECT_EQ(info.pipelines.opaque, NullPipeline);
    EXPECT_EQ(info.pipelines.opaqueDoubleSided, NullPipeline);
    EXPECT_EQ(info.pipelines.blend, NullPipeline);
}

TEST(FrameInfo, DefaultShadowPipelineIsNull)
{
    FrameInfo info;
    EXPECT_EQ(info.shadowPipeline, NullPipeline);
}

TEST(FrameInfo, DefaultShadowViewProjsAreZero)
{
    FrameInfo info;
    Mat4 zero;
    for (const Mat4& m : info.shadowViewProjs)
    {
        EXPECT_EQ(m, zero);
    }
}

TEST(FrameInfo, AssignShadowPipelineRoundTrip)
{
    FrameInfo info;
    info.shadowPipeline = PipelineHandle{7};
    EXPECT_EQ(info.shadowPipeline, PipelineHandle{7});
}

TEST(FrameInfo, AssignShadowViewProjsRoundTrip)
{
    FrameInfo info;
    Mat4 id = Mat4::identity();
    for (Mat4& m : info.shadowViewProjs)
    {
        m = id;
    }
    for (const Mat4& m : info.shadowViewProjs)
    {
        EXPECT_EQ(m, id);
    }
}

// ---------------------------------------------------------------------------
// Aggregate initialization
// ---------------------------------------------------------------------------

TEST(FrameInfo, AggregateInit)
{
    FrameInfo info{1, 1920, 1080, {2.0f, 3.0f, 4.0f}, {0.0f, 0.0f, 0.0f}};
    EXPECT_EQ(info.currentFrame, 1u);
    EXPECT_EQ(info.viewportWidth, 1920u);
    EXPECT_EQ(info.viewportHeight, 1080u);
    EXPECT_FLOAT_EQ(info.cameraPosition.x(), 2.0f);
    EXPECT_FLOAT_EQ(info.cameraPosition.y(), 3.0f);
    EXPECT_FLOAT_EQ(info.cameraPosition.z(), 4.0f);
}

// ---------------------------------------------------------------------------
// Member assignment
// ---------------------------------------------------------------------------

TEST(FrameInfo, AssignCurrentFrame)
{
    FrameInfo info;
    info.currentFrame = 1;
    EXPECT_EQ(info.currentFrame, 1u);
}

TEST(FrameInfo, AssignViewportDimensions)
{
    FrameInfo info;
    info.viewportWidth = 2560;
    info.viewportHeight = 1440;
    EXPECT_EQ(info.viewportWidth, 2560u);
    EXPECT_EQ(info.viewportHeight, 1440u);
}

TEST(FrameInfo, AssignCameraVectors)
{
    FrameInfo info;
    info.cameraPosition = {5.0f, 10.0f, 15.0f};
    info.cameraTarget = {0.0f, 1.0f, 0.0f};
    EXPECT_FLOAT_EQ(info.cameraPosition.x(), 5.0f);
    EXPECT_FLOAT_EQ(info.cameraTarget.y(), 1.0f);
}

// ---------------------------------------------------------------------------
// Copy semantics
// ---------------------------------------------------------------------------

TEST(FrameInfo, CopyPreservesAllFields)
{
    FrameInfo original{0, 800, 600, {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    FrameInfo copy = original;
    EXPECT_EQ(copy.currentFrame, original.currentFrame);
    EXPECT_EQ(copy.viewportWidth, original.viewportWidth);
    EXPECT_EQ(copy.viewportHeight, original.viewportHeight);
    EXPECT_EQ(copy.cameraPosition, original.cameraPosition);
    EXPECT_EQ(copy.cameraTarget, original.cameraTarget);
}
