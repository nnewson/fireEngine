#include <fire_engine/graphics/shadow_caster_deformation.hpp>

#include <fire_engine/graphics/geometry.hpp>

namespace fire_engine
{

ShadowCasterDeformation shadowCasterDeformation(const Geometry& geometry,
                                                bool objectDeforms) noexcept
{
    // Morph CAPABILITY, not the current weights. All-zero weights are this frame's value, not a
    // property of the caster: an animation can drive them at any time, and a caster that changed
    // classification mid-animation would swap error models — and with them its whole selection
    // history — at a moment nothing else in the frame marks as special.
    const bool morphCapable = !geometry.morphPositions().empty() ||
                              !geometry.morphNormals().empty() || !geometry.morphTangents().empty();
    // A compute pass rewrites these vertices after the simplifier measured them. Cloth is the only
    // storage-vertex geometry today AND it is single-level, so it would resolve to the whole mesh
    // regardless — it is safe by accident. Classifying it explicitly is what keeps that accident
    // from becoming load-bearing the day storage-vertex geometry gains an LOD chain.
    const bool computeWritesVertices = geometry.storageVertices();

    return objectDeforms || morphCapable || computeWritesVertices
               ? ShadowCasterDeformation::Deformable
               : ShadowCasterDeformation::Rigid;
}

} // namespace fire_engine
