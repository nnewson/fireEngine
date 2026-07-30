#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

struct VdpmWorkRequest; // graphics/vdpm_gpu_registry.hpp — the sink holds these by value

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
    // FrameInfo is built (by the renderer, per frame) rather than per object.
    // proj is the jittered projection (TAA) — used for rasterisation only.
    Mat4 view{Mat4::identity()};
    Mat4 proj{Mat4::identity()};
    // Jitter-free current/previous view-projection for motion vectors (TAA).
    Mat4 currentViewProj{Mat4::identity()};
    Mat4 previousViewProj{Mat4::identity()};
    // Per-frame camera UBO buffer for this frame slot (forward set-0 Camera binding, pushed per
    // draw). Written once per frame by the Renderer; the object draw-build copies it into each
    // DrawCommand.
    BufferHandle cameraUbo{NullBuffer};
    AlphaPipelines pipelines{};
    // Discrete mesh LOD (set by the renderer from RenderTunables). When enabled, the object
    // draw-build picks a coarser index set for distant/small static meshes within this pixel
    // budget.
    bool lodEnabled{false};
    float lodPixelErrorBudget{2.0f};
    // Shadow LOD, SEPARATELY (SH-03 slice 6). `--no-shadow-lod` forces every shadow caster to LOD0
    // while leaving the forward selection alone, which is the control an A/B needs: measuring
    // shadow LOD against `--no-lod` also changes the visible geometry, so the comparison contains
    // differences that have nothing to do with shadows. A tiny budget is NOT a substitute — the
    // selector still runs, and any cut whose estimated deviation is 0 (or below the budget) is
    // still eligible, so "budget 0.001" means "almost always LOD0", not "LOD0".
    bool shadowLodEnabled{true};
    // LOD strategy (from RenderTunables). Continuous enables the VIPM geomorph on the forward pass.
    LodMode lodMode{LodMode::Discrete};
    // GPU-driven VDPM (rendering-spine #3, Stage B5b): the global CPU/GPU backend selector and the
    // renderer-owned per-frame work-request sink. When `vdpmGpuBackend` is set and a binding
    // carries a GPU front handle, Object appends its VdpmWorkRequest to `vdpmRequestSink` and tags
    // the forward DrawCommand with the front handle; the renderer harvests the camera-visible
    // forward draws, records their front compute, and draws directly from the GPU-emitted
    // index/indirect buffers (the per-instance CPU front work is skipped). Null sink / false
    // selector ⇒ pure CPU front.
    bool vdpmGpuBackend{false};
    std::vector<VdpmWorkRequest>* vdpmRequestSink{nullptr};
    PipelineHandle shadowPipeline{NullPipeline};
    // Light-space view-projection matrices for every shadow caster — cascades,
    // spot lights, and the six faces of each point light. Layout matches
    // ShadowUBO::lightViewProj. Object::render copies the full array into the
    // per-draw ShadowUBO; the shadow vertex shader picks one via push constant.
    std::array<Mat4, kShadowTotalMatrixCount> shadowViewProjs{};
};

} // namespace fire_engine
