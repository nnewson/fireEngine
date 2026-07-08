#include <fire_engine/graphics/geometry.hpp>

#include <algorithm>
#include <string>

#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vipm.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

void Geometry::load(Resources& resources)
{
    vertexBuffer_ = storageVertices_ ? resources.createStorageVertexBuffer(vertices_)
                                     : resources.createVertexBuffer(vertices_);
    indexBuffer_ = resources.createIndexBuffer(indices_);

    // LOD0 is the full mesh. Build coarser discrete levels for static meshes large enough to be
    // worth it — every level indexes the same (unchanged) vertex buffer, so this only uploads index
    // data.
    lods_.clear();
    lods_.push_back(GeometryLod{indexBuffer_, indexCount(), 0.0f});
    if (!storageVertices_ && indices_.size() / 3 >= kMinLodTriangles)
    {
        const QuadricSimplifier simplifier;
        for (const float ratio : kLodRatios)
        {
            const SimplifiedMesh simplified = simplifier.simplify(vertices_, indices_, ratio);
            // Skip a level the mesh couldn't actually coarsen (boundary-locked / already minimal).
            if (simplified.indices.empty() || simplified.indices.size() >= indices_.size())
            {
                continue;
            }
            const BufferHandle buffer = resources.createIndexBuffer(simplified.indices);
            lods_.push_back(GeometryLod{buffer, static_cast<uint32_t>(simplified.indices.size()),
                                        std::max(0.0f, simplified.error)});
        }
        if (lods_.size() > 1)
        {
            std::string levels;
            for (const GeometryLod& lod : lods_)
            {
                levels += " " + std::to_string(lod.indexCount / 3) +
                          "t(e=" + std::to_string(lod.error) + ")";
            }
            log::debug(log::category::render, "LOD built {} levels from {} tris:{}", lods_.size(),
                       indices_.size() / 3, levels);

            // VIPM (Continuous LOD): reshape the same collapse stream into per-vertex geomorph
            // data, banded by the discrete level errors, and upload it as a parallel storage
            // buffer. It is bound only on the Continuous draw path; the discrete draw ignores it.
            const std::vector<MeshCollapse> collapses =
                simplifier.collapseSequence(vertices_, indices_);
            std::vector<float> levelErrors;
            levelErrors.reserve(lods_.size());
            for (const GeometryLod& lod : lods_)
            {
                levelErrors.push_back(lod.error);
            }
            const std::vector<MorphVertex> morph =
                buildVipmMorphData(vertices_, collapses, levelErrors);
            morphBuffer_ =
                resources.createStorageBuffer(morph.size() * sizeof(MorphVertex), morph.data());
        }
    }
}

} // namespace fire_engine
