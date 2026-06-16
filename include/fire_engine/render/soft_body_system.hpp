#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/render/compute_pipeline.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;

// Renderer-owned GPU soft-body (cloth) solver (Roadmap #2). Owns the XPBD compute
// pipelines and, per registered cloth, the particle + constraint buffers and a
// descriptor set. Each frame `recordSolve` runs the substep loop (predict →
// per-colour distance solve → finalize) writing solved positions + normals into
// the cloth's render vertex buffer, which the forward/shadow passes then read.
//
// v1 wires the solver with descriptor sets (the ParticleSystem pattern);
// bufferDeviceAddress is a deferred follow-up (see roadmap.md). Cloths are
// registered imperatively via addCloth — a Cloth scene component + gather is a
// later refactor.
class SoftBodySystem
{
public:
    SoftBodySystem(const Device& device, Resources& resources);
    ~SoftBodySystem() = default;

    SoftBodySystem(const SoftBodySystem&) = delete;
    SoftBodySystem& operator=(const SoftBodySystem&) = delete;
    SoftBodySystem(SoftBodySystem&&) noexcept = default;
    SoftBodySystem& operator=(SoftBodySystem&&) noexcept = default;

    // Register a cloth: allocate + upload its particle/constraint buffers and bind
    // a solver descriptor set over them and the (compute-writable) render vertex
    // buffer. `vertexBuffer` is the Geometry's storage vertex buffer.
    void addCloth(const ClothMesh& mesh, BufferHandle vertexBuffer);

    [[nodiscard]] bool empty() const noexcept
    {
        return cloths_.empty();
    }

    // Record the substep dispatch chain for every cloth, ending with a
    // compute-write → vertex-input-read barrier on each render vertex buffer.
    // Recorded before the shadow pass.
    void recordSolve(vk::CommandBuffer cmd, float dt) const;

private:
    struct Cloth
    {
        BufferHandle particles{NullBuffer};
        BufferHandle constraints{NullBuffer};
        BufferHandle verts{NullBuffer};
        vk::raii::DescriptorSet set{nullptr};
        uint32_t particleCount{0};
        std::vector<uint32_t> colourRanges;
        uint32_t resX{0};
        uint32_t resZ{0};
    };

    const Device* device_{nullptr};
    Resources* resources_{nullptr};
    ComputePipeline predict_;
    ComputePipeline solve_;
    ComputePipeline finalize_;
    vk::raii::DescriptorPool pool_{nullptr};
    std::vector<Cloth> cloths_;
};

} // namespace fire_engine
