#pragma once

#include <array>
#include <cstdint>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

// Pipeline handles covering the three forward alpha variants. Object::render
// picks one per geometry from the material's alphaMode/doubleSided flags. The
// renderer later buckets blend draws so they can be sorted back-to-front.
struct AlphaPipelines
{
    PipelineHandle opaque{NullPipeline};
    PipelineHandle opaqueDoubleSided{NullPipeline};
    PipelineHandle blend{NullPipeline};
};

struct FrameInfo
{
    uint32_t currentFrame{0};
    uint32_t viewportWidth{0};
    uint32_t viewportHeight{0};
    Vec3 cameraPosition;
    Vec3 cameraTarget;
    AlphaPipelines pipelines{};
    PipelineHandle shadowPipeline{NullPipeline};
    // Light-space view-projection matrices for every shadow caster — cascades,
    // spot lights, and the six faces of each point light. Layout matches
    // ShadowUBO::lightViewProj. Object::render copies the full array into the
    // per-draw ShadowUBO; the shadow vertex shader picks one via push constant.
    std::array<Mat4, kShadowTotalMatrixCount> shadowViewProjs{};
};

} // namespace fire_engine
