#include <fire_engine/render/render_context.hpp>

#include <fire_engine/render/swapchain.hpp>

namespace fire_engine
{

FrameInfo RenderContext::frameInfo() const noexcept
{
    auto extent = swapchain.extent();
    return {currentFrame,     extent.width, extent.height,  cameraPosition,
            cameraTarget,     view,         proj,           currentViewProj,
            previousViewProj, pipelines,    shadowPipeline, shadowViewProjs};
}

} // namespace fire_engine
