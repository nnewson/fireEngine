#include <fire_engine/graphics/geometry.hpp>

#include <algorithm>
#include <string>

#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm_gpu_registry.hpp>
#include <fire_engine/graphics/vipm.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

void Geometry::load(Resources& resources, VdpmGpuRegistry* registry)
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
        const ProgressiveMesh progressive =
            simplifier.buildProgressive(vertices_, indices_, kLodRatios);
        for (std::size_t i = 1; i < progressive.lods.size(); ++i)
        {
            const ProgressiveLod& simplified = progressive.lods[i];
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
            // data, banded by exact discrete LOD cuts, and upload it as a parallel storage buffer.
            // It is bound only on the Continuous draw path; the discrete draw ignores it.
            const std::vector<MorphVertex> morph =
                buildVipmMorphData(vertices_, progressive.collapses, progressive.lods);
            morphBuffer_ = resources.createStaticStorageBuffer(morph.size() * sizeof(MorphVertex),
                                                               morph.data());

            // VDPM (View-dependent LOD): keep the collapse stream so an ActiveFront can be built
            // per instance and refined per frame into a dynamic index buffer.
            collapses_ = progressive.collapses;
        }
    }

    // GPU-driven VDPM (Stage B5b): register this geometry's forest with the GPU backend once, so
    // every instance can create a per-instance front over the shared mesh. Registered after the
    // collapse stream is populated; a null registry (CPU backend / unsupported device) or an
    // ineligible mesh leaves vdpmMeshHandle_ == NullVdpmMesh.
    if (registry != nullptr && hasVdpmData())
    {
        vdpmMeshHandle_ = registry->registerMesh(vertices_, indices_, collapses_);
    }
}

} // namespace fire_engine
