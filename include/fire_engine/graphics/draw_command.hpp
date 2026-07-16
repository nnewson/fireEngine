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

// Vulkan-free mirror of VkDrawIndexedIndirectCommand — the 5-word record a `drawIndexedIndirect`
// reads from a GPU buffer. Kept in `graphics/` (Vulkan-free) so `object.cpp` can write it via
// `writeMapped` without a Vulkan type crossing the layering boundary, exactly the host↔GPU struct
// discipline `ubo.hpp` uses. `render/` static_asserts this matches VkDrawIndexedIndirectCommand
// (size, per-field offset, alignment) so a silent layout drift can't corrupt the draw. Today the
// CPU fills it from the VDPM emit's index count; when the front moves to the GPU (Stage B5) a
// compute shader writes the identical record and the draw is unchanged.
struct DrawIndexedIndirectCommand
{
    uint32_t indexCount{0};
    uint32_t instanceCount{0};
    uint32_t firstIndex{0};
    int32_t vertexOffset{0};
    uint32_t firstInstance{0};
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
    // Forward set-0 push buffers: per-object (frame) + per-frame camera + skin/morph UBOs + morph
    // SSBO. cameraUbo is the same handle for every draw in a frame (written once by the Renderer).
    BufferHandle frameUbo{NullBuffer};
    BufferHandle cameraUbo{NullBuffer};
    BufferHandle skinUbo{NullBuffer};
    // Previous-frame skin matrices (forward set-0 PrevSkin binding) for skinned motion vectors.
    // Identity for non-skinned draws. Not used by the shadow pass.
    BufferHandle prevSkinUbo{NullBuffer};
    BufferHandle morphUbo{NullBuffer};
    BufferHandle morphSsbo{NullBuffer};
    // VIPM per-vertex geomorph SSBO (Continuous LOD). The geometry's morph buffer for VIPM meshes,
    // else a dummy; the vertex shader only reads it when MorphUBO::morphFactor > 0.
    BufferHandle vipmBuffer{NullBuffer};
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
    // Selected discrete LOD level for this draw (0 = full mesh). Only used by the LOD debug tint.
    uint32_t lodLevel{0};
    Bounds3 shadowBounds{};
    Mat4 selfShadowViewProj{Mat4::identity()};
    // Indirect draw (rendering-spine #3, GPU-driven-front Stage A). When `indirectBuffer` is not
    // NullBuffer the renderer records `drawIndexedIndirect` from the `DrawIndexedIndirectCommand`
    // at `indirectOffset` in that buffer, instead of `drawIndexed(indexCount, ...)`. Only the VDPM
    // (view-dependent LOD) draws set it; every other path (static / discrete LOD / VIPM / skybox /
    // shadow) leaves it null and draws directly. `indexCount` stays populated for the triangle
    // overlay even on indirect draws. `indirectOffset` is a Vulkan-free byte offset (no Vulkan type
    // in this layer).
    BufferHandle indirectBuffer{NullBuffer};
    uint64_t indirectOffset{0};
};

} // namespace fire_engine
