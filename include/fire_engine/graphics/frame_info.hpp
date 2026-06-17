#pragma once

#include <array>
#include <cstdint>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// Pipeline handles covering the forward alpha variants. Object::render picks
// one per geometry from the material's alphaMode flag: opaque and double-sided
// share `opaque` (cull mode is set per draw via dynamic state, see
// DrawCommand::doubleSided); BLEND materials use `blend`, which keeps a static
// blend equation (dynamic blend is unsupported on MoltenVK). The renderer
// buckets blend draws so they can be sorted back-to-front.
struct AlphaPipelines
{
    PipelineHandle opaque{NullPipeline};
    PipelineHandle blend{NullPipeline};
};

struct FrameInfo
{
    uint32_t currentFrame{0};
    uint32_t viewportWidth{0};
    uint32_t viewportHeight{0};
    Vec3 cameraPosition;
    Vec3 cameraTarget;
    // Camera view and projection for the frame, computed once when the
    // FrameInfo is built (see RenderContext::frameInfo) rather than per object.
    // proj is the jittered projection (TAA) — used for rasterisation only.
    Mat4 view{Mat4::identity()};
    Mat4 proj{Mat4::identity()};
    // Jitter-free current/previous view-projection for motion vectors (TAA).
    Mat4 currentViewProj{Mat4::identity()};
    Mat4 previousViewProj{Mat4::identity()};
    AlphaPipelines pipelines{};
    PipelineHandle shadowPipeline{NullPipeline};
    // Light-space view-projection matrices for every shadow caster — cascades,
    // spot lights, and the six faces of each point light. Layout matches
    // ShadowUBO::lightViewProj. Object::render copies the full array into the
    // per-draw ShadowUBO; the shadow vertex shader picks one via push constant.
    std::array<Mat4, kShadowTotalMatrixCount> shadowViewProjs{};
};

} // namespace fire_engine
