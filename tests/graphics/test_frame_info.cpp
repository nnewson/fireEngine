#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/frame_info.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST_CASE("FrameInfo.DefaultCurrentFrameIsZero", "[FrameInfo]")
{
    FrameInfo info;
    CHECK(info.currentFrame == 0u);
}

TEST_CASE("FrameInfo.DefaultViewportWidthIsZero", "[FrameInfo]")
{
    FrameInfo info;
    CHECK(info.viewportWidth == 0u);
}

TEST_CASE("FrameInfo.DefaultViewportHeightIsZero", "[FrameInfo]")
{
    FrameInfo info;
    CHECK(info.viewportHeight == 0u);
}

TEST_CASE("FrameInfo.DefaultCameraPositionIsOrigin", "[FrameInfo]")
{
    FrameInfo info;
    CHECK(info.cameraPosition.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(info.cameraPosition.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(info.cameraPosition.z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("FrameInfo.DefaultCameraTargetIsOrigin", "[FrameInfo]")
{
    FrameInfo info;
    CHECK(info.cameraTarget.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(info.cameraTarget.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(info.cameraTarget.z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("FrameInfo.DefaultAlphaPipelinesAreNull", "[FrameInfo]")
{
    FrameInfo info;
    CHECK(info.pipelines.opaque == NullPipeline);
    CHECK(info.pipelines.blend == NullPipeline);
}

TEST_CASE("FrameInfo.DefaultShadowPipelineIsNull", "[FrameInfo]")
{
    FrameInfo info;
    CHECK(info.shadowPipeline == NullPipeline);
}

TEST_CASE("FrameInfo.DefaultShadowViewProjsAreZero", "[FrameInfo]")
{
    FrameInfo info;
    Mat4 zero;
    for (const Mat4& m : info.shadowViewProjs)
    {
        CHECK(m == zero);
    }
}

TEST_CASE("FrameInfo.AssignShadowPipelineRoundTrip", "[FrameInfo]")
{
    FrameInfo info;
    info.shadowPipeline = PipelineHandle{7};
    CHECK(info.shadowPipeline == PipelineHandle{7});
}

TEST_CASE("FrameInfo.AssignShadowViewProjsRoundTrip", "[FrameInfo]")
{
    FrameInfo info;
    Mat4 id = Mat4::identity();
    for (Mat4& m : info.shadowViewProjs)
    {
        m = id;
    }
    for (const Mat4& m : info.shadowViewProjs)
    {
        CHECK(m == id);
    }
}

// ---------------------------------------------------------------------------
// Aggregate initialization
// ---------------------------------------------------------------------------

TEST_CASE("FrameInfo.AggregateInit", "[FrameInfo]")
{
    FrameInfo info{1, 1920, 1080, {2.0f, 3.0f, 4.0f}, {0.0f, 0.0f, 0.0f}};
    CHECK(info.currentFrame == 1u);
    CHECK(info.viewportWidth == 1920u);
    CHECK(info.viewportHeight == 1080u);
    CHECK(info.cameraPosition.x() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(info.cameraPosition.y() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(info.cameraPosition.z() == Catch::Approx(4.0f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// Member assignment
// ---------------------------------------------------------------------------

TEST_CASE("FrameInfo.AssignCurrentFrame", "[FrameInfo]")
{
    FrameInfo info;
    info.currentFrame = 1;
    CHECK(info.currentFrame == 1u);
}

TEST_CASE("FrameInfo.AssignViewportDimensions", "[FrameInfo]")
{
    FrameInfo info;
    info.viewportWidth = 2560;
    info.viewportHeight = 1440;
    CHECK(info.viewportWidth == 2560u);
    CHECK(info.viewportHeight == 1440u);
}

TEST_CASE("FrameInfo.AssignCameraVectors", "[FrameInfo]")
{
    FrameInfo info;
    info.cameraPosition = {5.0f, 10.0f, 15.0f};
    info.cameraTarget = {0.0f, 1.0f, 0.0f};
    CHECK(info.cameraPosition.x() == Catch::Approx(5.0f).margin(1e-5f));
    CHECK(info.cameraTarget.y() == Catch::Approx(1.0f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// Copy semantics
// ---------------------------------------------------------------------------

TEST_CASE("FrameInfo.CopyPreservesAllFields", "[FrameInfo]")
{
    FrameInfo original{0, 800, 600, {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    FrameInfo copy = original;
    CHECK(copy.currentFrame == original.currentFrame);
    CHECK(copy.viewportWidth == original.viewportWidth);
    CHECK(copy.viewportHeight == original.viewportHeight);
    CHECK(copy.cameraPosition == original.cameraPosition);
    CHECK(copy.cameraTarget == original.cameraTarget);
}
