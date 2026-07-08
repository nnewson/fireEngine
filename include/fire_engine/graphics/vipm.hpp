#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/vec3.hpp>

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
    // The position this vertex geomorphs toward as it collapses: the *resolved* surviving vertex at
    // the coarser side of its collapse band (chains A->B->C followed through), so at morphFactor 1
    // it is coincident with a vertex that's present after the swap. Own position for non-collapsing
    // vertices.
    Vec3 targetPosition{};
    // The LOD error at which this vertex is removed; a large sentinel for vertices that never
    // collapse within the built levels (they never morph).
    float collapseError{0.0f};
};

// std430-compatible: { vec3; float } packs to 16 bytes with the error in the vec3's 4th slot, so
// the array uploads straight into a storage buffer the vertex shader indexes by vertex index.
static_assert(sizeof(Vec3) == 12, "MorphVertex's std430 layout assumes a tightly packed Vec3");
static_assert(sizeof(MorphVertex) == 16, "MorphVertex must match its std430 shader layout");

// Sentinel collapse error for a vertex that never collapses within the built LOD levels.
inline constexpr float kVipmNeverCollapses = 1.0e30f;

// Build per-original-vertex morph data from the recorded collapse stream and the discrete LOD
// errors
// (`levelErrors[0]` is LOD0 == 0.0, the rest are the coarser levels in increasing order). For a
// vertex removed in band (levelErrors[L-1], levelErrors[L]], its target is the position of the
// vertex it resolves to after replaying every collapse up to levelErrors[L] — a vertex present at
// level L. Vulkan-free + deterministic (headless-tested).
[[nodiscard]] std::vector<MorphVertex> buildVipmMorphData(std::span<const Vertex> vertices,
                                                          std::span<const MeshCollapse> collapses,
                                                          std::span<const float> levelErrors);

// Continuous LOD selection for VIPM: which discrete topology level to draw, plus how far (0..1) the
// geomorph toward the next coarser level has progressed and that next level's error. The shader
// morphs each drawn vertex whose `collapseError <= nextLevelError` by `morphFactor`. `level`
// matches what discrete `selectLod` would pick at the same distance, so the two modes agree on
// topology and the swap lands exactly when `morphFactor` reaches 1.
struct VipmSelection
{
    std::size_t level{0};
    float morphFactor{0.0f};
    float nextLevelError{0.0f};
};

[[nodiscard]] inline VipmSelection selectVipm(std::span<const GeometryLod> lods, float distance,
                                              float projScaleY, float viewportHeight,
                                              float pixelError) noexcept
{
    const std::size_t level = selectLod(lods, distance, projScaleY, viewportHeight, pixelError);
    VipmSelection sel{level, 0.0f, 0.0f};
    if (level + 1 >= lods.size())
    {
        return sel; // coarsest level — nothing to morph toward
    }

    // World-space error the current view tolerates (invert selectLod's projection), then how far it
    // has advanced from this level's error toward the next level's.
    const float d = distance > 1e-3f ? distance : 1e-3f;
    const float tolerated = pixelError * 2.0f * d / (projScaleY * viewportHeight);
    const float lo = lods[level].error;
    const float hi = lods[level + 1].error;
    sel.nextLevelError = hi;
    sel.morphFactor = hi > lo ? std::clamp((tolerated - lo) / (hi - lo), 0.0f, 1.0f) : 0.0f;
    return sel;
}

} // namespace fire_engine
