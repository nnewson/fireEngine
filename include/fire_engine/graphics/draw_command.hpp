#pragma once

#include <cstdint>

#include <fire_engine/graphics/bounds.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

enum class DrawIndexType : uint8_t
{
    UInt16,
    UInt32,
};

struct DrawCommand
{
    constexpr DrawCommand() = default;
    constexpr DrawCommand(BufferHandle vertexBuffer, BufferHandle indexBuffer, uint32_t indexCount,
                          DescriptorSetHandle descriptorSet = NullDescriptorSet,
                          PipelineHandle pipeline = NullPipeline, float sortDepth = 0.0f,
                          DrawIndexType indexType = DrawIndexType::UInt16) noexcept
        : vertexBuffer(vertexBuffer),
          indexBuffer(indexBuffer),
          indexCount(indexCount),
          indexType(indexType),
          descriptorSet(descriptorSet),
          pipeline(pipeline),
          sortDepth(sortDepth)
    {
    }

    BufferHandle vertexBuffer{NullBuffer};
    BufferHandle indexBuffer{NullBuffer};
    uint32_t indexCount{0};
    DrawIndexType indexType{DrawIndexType::UInt16};
    DescriptorSetHandle descriptorSet{NullDescriptorSet};
    PipelineHandle pipeline{NullPipeline};
    float sortDepth{0.0f};
    // KHR_materials_transmission F3: when true, this draw must run AFTER
    // the scene-colour capture so its fragment shader can sample the
    // post-opaque HDR target via screen-space refraction.
    bool transmissive{false};
    uint32_t objectId{0};
    bool hasSkin{false};
    int selfShadowSlot{-1};
    Bounds3 shadowBounds{};
    Mat4 selfShadowViewProj{Mat4::identity()};
};

} // namespace fire_engine
