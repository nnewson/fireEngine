#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

// The VdpmMeshHandle / VdpmFrontHandle identity handles live in gpu_handle.hpp alongside the other
// GPU resource handles.

// One camera-visible GPU-driven VDPM front to run this frame — the *semantic* per-instance inputs
// only. Frame-global data (jitter-free viewProj, viewport, refine/coarsen/repair budgets, camera
// pose) is NOT repeated here: the Renderer owns it and supplies it once, deriving the per-frame
// VdpmViewParams/VdpmRepairParams from these instance values + the frame globals. Emitted by
// Object during draw collection; gathered + deduped by the Renderer.
struct VdpmWorkRequest
{
    VdpmFrontHandle front{NullVdpmFront};
    Mat4 world;
    // Per-material attribute-deviation channel scales (geometry is implicit at 1.0). Two instances
    // sharing a mesh but wearing different materials refine differently, so these are per-request.
    float uvScale{1.0f};
    float normalScale{1.0f};
    float tangentScale{1.0f};
    // The raster back-face-culling policy of the instance's material (double-sided/blended
    // materials keep their back-faces, so coverage repair must protect them).
    bool rasterBackfaceCulling{true};

    // Member-by-member equality — NEVER memcmp (Mat4 + the trailing bool leave padding). Two
    // requests for the same front with identical parameters are the same work; the Renderer dedups
    // on this so a front rendered into several passes runs its compute once.
    [[nodiscard]] bool sameParams(const VdpmWorkRequest& other) const noexcept;
};

// The Vulkan-free seam graphics/ uses to register GPU-driven VDPM meshes/fronts, implemented by
// render::VdpmGpuManager and threaded through the load path. graphics/ never sees render/, so the
// concrete manager (which owns all Vulkan state) is reached only through this interface. All calls
// are load-time (never per-frame). registerMesh/createFront return the Null* handle when the GPU
// backend is unavailable or the mesh is ineligible, so the caller falls back to the CPU front.
class VdpmGpuRegistry
{
public:
    VdpmGpuRegistry() = default;
    VdpmGpuRegistry(const VdpmGpuRegistry&) = delete;
    VdpmGpuRegistry& operator=(const VdpmGpuRegistry&) = delete;
    VdpmGpuRegistry(VdpmGpuRegistry&&) = delete;
    VdpmGpuRegistry& operator=(VdpmGpuRegistry&&) = delete;
    virtual ~VdpmGpuRegistry() = default;

    // False when the device lacks the compute/scan capability the GPU front needs — the whole GPU
    // backend is then unavailable and every caller stays on the CPU front.
    [[nodiscard]] virtual bool available() const noexcept = 0;

    // Register a geometry's static forest once (shared by every instance of the geometry). Returns
    // NullVdpmMesh if unavailable or the forest is ineligible (logged once with the reason).
    [[nodiscard]] virtual VdpmMeshHandle registerMesh(std::span<const Vertex> vertices,
                                                      std::span<const std::uint32_t> indices,
                                                      std::span<const MeshCollapse> collapses) = 0;

    // Create a per-instance front over a registered mesh. Returns NullVdpmFront if `mesh` is null
    // or invalid.
    [[nodiscard]] virtual VdpmFrontHandle createFront(VdpmMeshHandle mesh) = 0;
};

// Caller-owned scratch for selectVisibleVdpmRequests, so the per-frame selection allocates nothing
// steady-state: the containers are cleared (capacity retained) on each call, not reconstructed.
struct VdpmRequestSelectScratch
{
    std::unordered_set<std::uint32_t> visibleFronts;          // packed handles present this frame
    std::unordered_map<std::uint32_t, std::size_t> keptIndex; // packed front → its slot in `out`
};

// Reduce a frame's raw work-request sink to the deduped set whose front is camera-visible — the
// exact list the manager records compute for — writing it into `out` (cleared first) and reusing
// `scratch`. A request is kept iff its front appears in `visibleFronts` (the fronts whose FORWARD
// draw survived the camera cull; a shadow-only instance's front is absent, so it never runs
// compute). Two requests for the SAME front are collapsed: with identical parameters
// (VdpmWorkRequest::sameParams) to a single entry, but with DIFFERING parameters it throws
// std::logic_error — a front is one persistent GPU state and must never be scored twice in a frame
// with conflicting inputs. First-seen order is preserved. `out` + `scratch` are caller-owned so the
// whole selection is allocation-free once warm. Vulkan-free + pure, so the runtime dedup contract
// is unit-tested directly (not just sameParams in isolation).
void selectVisibleVdpmRequests(std::span<const VdpmWorkRequest> requests,
                               std::span<const VdpmFrontHandle> visibleFronts,
                               std::vector<VdpmWorkRequest>& out,
                               VdpmRequestSelectScratch& scratch);

} // namespace fire_engine
