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
    // Set 0 for skybox draws (an allocated descriptor set). Forward, transmissive,
    // and shadow draws leave this null and instead carry the per-object buffers
    // below, which the renderer pushes inline via VK_KHR_push_descriptor.
    DescriptorSetHandle descriptorSet{NullDescriptorSet};
    // Forward set-0 push buffers: frame/skin/morph UBOs + morph-targets SSBO.
    BufferHandle frameUbo{NullBuffer};
    BufferHandle skinUbo{NullBuffer};
    BufferHandle morphUbo{NullBuffer};
    BufferHandle morphSsbo{NullBuffer};
    // Shadow set-0 push buffer (binding 0): per-object ShadowUBO. Shadow draws
    // reuse skin/morph/morphSsbo above; the shared self-shadow image+sampler
    // (bindings 4/5) are global, pushed from Resources by the shadow pass.
    BufferHandle shadowUbo{NullBuffer};
    PipelineHandle pipeline{NullPipeline};
    float sortDepth{0.0f};
    // KHR_materials_transmission F3: when true, this draw must run AFTER
    // the scene-colour capture so its fragment shader can sample the
    // post-opaque HDR target via screen-space refraction.
    bool transmissive{false};
    // Drives the forward pipeline's dynamic cull mode: double-sided draws cull
    // nothing, single-sided cull back faces. Opaque and double-sided draws share
    // one forward pipeline (set per draw via VK_DYNAMIC_STATE_CULL_MODE); only
    // relevant for the merged opaque/double-sided pipeline, ignored for blend.
    bool doubleSided{false};
    uint32_t objectId{0};
    bool hasSkin{false};
    int selfShadowSlot{-1};
    // Index into the global bindless materials[] SSBO for this draw's material.
    uint32_t materialIndex{0};
    Bounds3 shadowBounds{};
    Mat4 selfShadowViewProj{Mat4::identity()};
};

} // namespace fire_engine
