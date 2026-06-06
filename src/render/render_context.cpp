#include <fire_engine/render/render_context.hpp>

#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

FrameInfo RenderContext::frameInfo() const noexcept
{
    auto extent = swapchain.extent();
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const Mat4 view = Mat4::lookAt(cameraPosition, cameraTarget, {0, 1, 0});
    const Mat4 proj =
        Mat4::perspective(kCameraFovRadians, aspect, kCameraNearPlane, kCameraFarPlane);
    return {currentFrame, extent.width, extent.height, cameraPosition, cameraTarget,
            view,         proj,         pipelines,     shadowPipeline, shadowViewProjs};
}

} // namespace fire_engine
