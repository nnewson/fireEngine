#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

class Device;
class Swapchain;
class Frame;
class Pipeline;

struct RenderContext
{
    const Device& device;
    const Swapchain& swapchain;
    Frame& frame;
    const Pipeline& pipeline;
    vk::CommandBuffer commandBuffer;
    uint32_t currentFrame;
    Vec3 cameraPosition;
    Vec3 cameraTarget;
    // Per-frame camera matrices supplied by the Renderer. proj is the jittered
    // projection (TAA, rasterisation only); currentViewProj/previousViewProj are
    // jitter-free for motion vectors. Forwarded straight to FrameInfo.
    Mat4 view{Mat4::identity()};
    Mat4 proj{Mat4::identity()};
    Mat4 currentViewProj{Mat4::identity()};
    Mat4 previousViewProj{Mat4::identity()};
    std::vector<DrawCommand>* drawCommands{nullptr};
    AlphaPipelines pipelines{};
    PipelineHandle shadowPipeline{NullPipeline};
    std::array<Mat4, kShadowTotalMatrixCount> shadowViewProjs{};

    [[nodiscard]] FrameInfo frameInfo() const noexcept;
};

} // namespace fire_engine
