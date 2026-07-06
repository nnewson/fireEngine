#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// One recorded edge collapse: vertex `removed` is merged into `kept`. Phase 1 uses *subset*
// (endpoint) collapse — `kept` stays at its original position, so every simplified level indexes
// the unchanged original vertex buffer. `position` is `kept`'s position (recorded for the Phase 2/3
// progressive-mesh hierarchy, where a vertex split reintroduces `removed` at its original spot and
// geomorphs it out from `kept`). `error` is the quadric error this collapse incurs.
struct MeshCollapse
{
    uint32_t kept{0};
    uint32_t removed{0};
    Vec3 position{};
    float error{0.0f};
};

// Result of a simplification: an index buffer into the ORIGINAL vertex array (no vertex data
// moves), plus the largest collapse error applied to reach the requested ratio.
struct SimplifiedMesh
{
    std::vector<uint32_t> indices;
    float error{0.0f};
};

// Backend-swappable triangle-mesh simplifier. Our quadric-error-metric implementation is the first
// backend; the interface lets a different one (meshoptimizer, or a test oracle) drop in without
// touching call sites.
class MeshSimplifier
{
public:
    MeshSimplifier() = default;
    virtual ~MeshSimplifier() = default;

    MeshSimplifier(const MeshSimplifier&) = default;
    MeshSimplifier& operator=(const MeshSimplifier&) = default;
    MeshSimplifier(MeshSimplifier&&) = default;
    MeshSimplifier& operator=(MeshSimplifier&&) = default;

    // Simplify to roughly `targetRatio` (0..1) of the input triangle count. Returns indices into
    // the original vertices. Degenerate/boundary-locked meshes may simplify less than requested.
    [[nodiscard]] virtual SimplifiedMesh simplify(std::span<const Vertex> vertices,
                                                  std::span<const uint32_t> indices,
                                                  float targetRatio) const = 0;
};

// Garland–Heckbert quadric-error-metric edge-collapse simplifier (subset placement +
// boundary-quadric preservation + a normal-flip veto). Deterministic. Vulkan-free — the whole thing
// is CPU + headless- testable.
class QuadricSimplifier : public MeshSimplifier
{
public:
    [[nodiscard]] SimplifiedMesh simplify(std::span<const Vertex> vertices,
                                          std::span<const uint32_t> indices,
                                          float targetRatio) const override;

    // The full greedy coarsening stream (finest → coarsest) — the raw material for the Phase 2/3
    // progressive-mesh hierarchy. `simplify(ratio)` is exactly the mesh after replaying the prefix
    // of this stream that reaches the target triangle count.
    [[nodiscard]] std::vector<MeshCollapse>
    collapseSequence(std::span<const Vertex> vertices, std::span<const uint32_t> indices) const;
};

} // namespace fire_engine
