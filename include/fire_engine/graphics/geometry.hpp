#pragma once

#include <cstdint>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vertex.hpp>

namespace fire_engine
{

class Material;
class Resources;
class VdpmGpuRegistry;

class Geometry
{
public:
    Geometry() = default;
    ~Geometry() = default;

    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) noexcept = default;
    Geometry& operator=(Geometry&&) noexcept = default;

    // `registry`, when non-null and the geometry carries a collapse stream, registers this
    // geometry's static forest once with the GPU-driven VDPM backend and records the returned mesh
    // handle (shared by every instance). Null, or a mesh the backend rejects, leaves
    // vdpmMeshHandle() == NullVdpmMesh, so every instance stays on the CPU front. No default —
    // every caller states its registration choice (pass Renderer::vdpmRegistry(), or nullptr for
    // none).
    void load(Resources& resources, VdpmGpuRegistry* registry);

    [[nodiscard]] bool loaded() const noexcept
    {
        return vertexBuffer_ != NullBuffer;
    }

    [[nodiscard]] const std::vector<Vertex>& vertices() const noexcept
    {
        return vertices_;
    }
    void vertices(std::vector<Vertex> v) noexcept
    {
        vertices_ = std::move(v);
    }

    [[nodiscard]] const std::vector<uint32_t>& indices() const noexcept
    {
        return indices_;
    }
    void indices(std::vector<uint32_t> v) noexcept
    {
        indices_ = std::move(v);
    }
    void indices(std::vector<uint16_t> v) noexcept
    {
        indices_.assign(v.begin(), v.end());
    }

    [[nodiscard]] const Material& material() const noexcept
    {
        return *material_;
    }
    void material(const Material* m) noexcept
    {
        material_ = m;
    }

    [[nodiscard]] BufferHandle vertexBuffer() const noexcept
    {
        return vertexBuffer_;
    }

    [[nodiscard]] BufferHandle indexBuffer() const noexcept
    {
        return indexBuffer_;
    }

    [[nodiscard]] uint32_t indexCount() const noexcept
    {
        return static_cast<uint32_t>(indices_.size());
    }

    [[nodiscard]] DrawIndexType indexType() const noexcept
    {
        return DrawIndexType::UInt32;
    }

    // Discrete levels of detail, built at load() time from the base mesh (LOD0 = the full mesh;
    // empty for deformable or small meshes). All levels index the same, unchanged vertex buffer.
    [[nodiscard]] const std::vector<GeometryLod>& lods() const noexcept
    {
        return lods_;
    }

    // VIPM (Continuous LOD): per-vertex geomorph data, parallel to the vertex buffer and indexed by
    // vertex index in the shader. Built alongside the discrete LODs; NullBuffer when the mesh has
    // no coarser levels (small/deformable meshes), in which case only the discrete path is
    // available.
    [[nodiscard]] BufferHandle morphBuffer() const noexcept
    {
        return morphBuffer_;
    }
    [[nodiscard]] bool hasVipmData() const noexcept
    {
        return morphBuffer_ != NullBuffer;
    }

    // VDPM (View-dependent LOD): the recorded collapse stream, kept so an ActiveFront can be built
    // per instance and refined per frame. Empty for meshes with no progressive LODs.
    [[nodiscard]] const std::vector<MeshCollapse>& collapses() const noexcept
    {
        return collapses_;
    }
    [[nodiscard]] bool hasVdpmData() const noexcept
    {
        return !collapses_.empty();
    }

    // GPU-driven VDPM mesh handle for this geometry (rendering-spine #3, Stage B5b), set by load()
    // when a registry is supplied and the forest is GPU-eligible; NullVdpmMesh otherwise. Instances
    // create their per-instance front over it via VdpmGpuRegistry::createFront.
    [[nodiscard]] VdpmMeshHandle vdpmMeshHandle() const noexcept
    {
        return vdpmMeshHandle_;
    }

    [[nodiscard]] bool castsShadow() const noexcept
    {
        return castsShadow_;
    }
    void castsShadow(bool value) noexcept
    {
        castsShadow_ = value;
    }

    // When set, load() allocates the vertex buffer with storage usage so the
    // soft-body compute solver can write it in place each frame (cloth meshes).
    void storageVertices(bool value) noexcept
    {
        storageVertices_ = value;
    }
    [[nodiscard]] bool storageVertices() const noexcept
    {
        return storageVertices_;
    }

    [[nodiscard]] const std::vector<std::vector<Vec3>>& morphPositions() const noexcept
    {
        return morphPositions_;
    }
    void morphPositions(std::vector<std::vector<Vec3>> v) noexcept
    {
        morphPositions_ = std::move(v);
    }

    [[nodiscard]] const std::vector<std::vector<Vec3>>& morphNormals() const noexcept
    {
        return morphNormals_;
    }
    void morphNormals(std::vector<std::vector<Vec3>> v) noexcept
    {
        morphNormals_ = std::move(v);
    }

    // glTF morph TANGENT deltas — Vec3 per vertex per target (no handedness;
    // the base tangent's .w stays unchanged across targets per spec).
    [[nodiscard]] const std::vector<std::vector<Vec3>>& morphTangents() const noexcept
    {
        return morphTangents_;
    }
    void morphTangents(std::vector<std::vector<Vec3>> v) noexcept
    {
        morphTangents_ = std::move(v);
    }

    [[nodiscard]] std::size_t morphTargetCount() const noexcept
    {
        return morphPositions_.size();
    }

private:
    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    const Material* material_{nullptr};

    std::vector<std::vector<Vec3>> morphPositions_;
    std::vector<std::vector<Vec3>> morphNormals_;
    std::vector<std::vector<Vec3>> morphTangents_;

    BufferHandle vertexBuffer_{NullBuffer};
    BufferHandle indexBuffer_{NullBuffer};
    std::vector<GeometryLod> lods_;
    BufferHandle morphBuffer_{NullBuffer};
    std::vector<MeshCollapse> collapses_;
    VdpmMeshHandle vdpmMeshHandle_{NullVdpmMesh};
    bool castsShadow_{true};
    bool storageVertices_{false};
};

} // namespace fire_engine
