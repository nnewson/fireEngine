#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

// View-independent progressive meshing (VIPM, rendering-spine #3 Phase 2). Layers a GPU geomorph on
// top of Phase 1's discrete LODs so the level transitions dissolve instead of popping: as the mesh
// coarsens, a collapsing vertex slides from its own position to `targetPosition`, reaching it
// exactly when the topology swaps to the coarser level — so the swap is invisible.

// Per-original-vertex geomorph data. Parallel to the vertex buffer; used only on the VIPM path (the
// discrete-LOD and skinned/cloth paths never bind it, so the shared Vertex layout is untouched).
struct MorphVertex
{
    // ALL the attributes this vertex geomorphs toward as it collapses: those of the *resolved*
    // surviving vertex at the coarser side of its collapse band (chains A->B->C followed through).
    // The shader morphs position, normal, tangent, and both texcoord sets together so that at
    // morphFactor 1 the vertex is *fully* coincident with a vertex present after the swap.
    // Position-only morphing warps the texture and snaps the attributes at the swap. Own attributes
    // for non-collapsing vertices (they never morph).
    Vec3 targetPosition{};
    // The 1-based LOD level this original vertex first disappears into. The shader compares this
    // with MorphUBO::vipmTargetLevel, so non-monotonic collapse errors cannot trigger an early
    // morph. Sentinel = never removed by the built LOD cuts.
    float collapseLevel{0.0f};
    Vec3 targetNormal{};
    float _pad0{0.0f};
    Vec4 targetTangent{};
    Vec2 targetTexCoord{};
    Vec2 targetTexCoord1{};
};

// std430-compatible: four tightly-packed vec4s (posLevel | normal,_ | tangent | uv0,uv1), uploaded
// straight into a storage buffer the vertex shader indexes by vertex index.
static_assert(sizeof(Vec3) == 12, "MorphVertex's std430 layout assumes a tightly packed Vec3");
static_assert(sizeof(MorphVertex) == 64, "MorphVertex must match its std430 shader layout");

// Sentinel collapse level for a vertex that never collapses within the built LOD levels.
inline constexpr float kVipmNeverCollapses = 1.0e30f;

// Build per-original-vertex morph data from the recorded collapse stream and exact discrete LOD
// cuts. For a vertex first removed by LOD level L, its target is the full render-attribute wedge it
// resolves to after replaying exactly `lods[L].collapseCount` collapses. Vulkan-free +
// deterministic (headless-tested).
[[nodiscard]] std::vector<MorphVertex> buildVipmMorphData(std::span<const Vertex> vertices,
                                                          std::span<const MeshCollapse> collapses,
                                                          std::span<const ProgressiveLod> lods);

// Continuous LOD selection for VIPM: which discrete topology level to draw, plus how far (0..1) the
// geomorph toward the next coarser level has progressed and that next level's exact ordinal. The
// shader morphs each drawn vertex whose removal level equals `targetLevel` by `morphFactor`.
// `level` matches what discrete `selectLod` would pick at the same distance, so the two modes agree
// on topology and the swap lands exactly when `morphFactor` reaches 1.
struct VipmSelection
{
    std::size_t level{0};
    float morphFactor{0.0f};
    std::size_t targetLevel{0};
};

[[nodiscard]] inline VipmSelection selectVipm(std::span<const GeometryLod> lods, float distance,
                                              float projScaleY, float viewportHeight,
                                              float pixelErrorBudget) noexcept
{
    const std::size_t level =
        selectLod(lods, distance, projScaleY, viewportHeight, pixelErrorBudget);
    VipmSelection sel{level, 0.0f, 0};
    if (level + 1 >= lods.size())
    {
        return sel; // coarsest level — nothing to morph toward
    }

    // World-space error the current view tolerates (invert selectLod's projection), then how far it
    // has advanced from this level's error toward the next level's.
    const float d = distance > 1e-3f ? distance : 1e-3f;
    const float tolerated = pixelErrorBudget * 2.0f * d / (projScaleY * viewportHeight);
    const float lo = lods[level].error;
    const float hi = lods[level + 1].error;
    sel.targetLevel = level + 1;
    sel.morphFactor = hi > lo ? std::clamp((tolerated - lo) / (hi - lo), 0.0f, 1.0f) : 0.0f;
    return sel;
}

} // namespace fire_engine
