#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>
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

    // Record the substep dispatch chain for every cloth (predict → solve → collide
    // per substep, then finalize), ending with a compute-write → vertex-input-read
    // barrier on each render vertex buffer. Uploads `colliders` into this frame's
    // collider UBO first. Recorded before the shadow pass.
    void recordSolve(vk::CommandBuffer cmd, float dt, uint32_t frameIndex,
                     std::span<const ClothCollider> colliders) const;

    static constexpr uint32_t kMaxClothColliders = 16;

private:
    struct Cloth
    {
        BufferHandle particles{NullBuffer};
        BufferHandle constraints{NullBuffer};
        BufferHandle verts{NullBuffer};
        // One descriptor set per frame-in-flight: 0/1/2 are the shared particle/
        // constraint/vertex buffers, binding 3 is that frame's collider UBO.
        std::vector<vk::raii::DescriptorSet> sets;
        uint32_t particleCount{0};
        std::vector<uint32_t> colourRanges;
        uint32_t resX{0};
        uint32_t resZ{0};
    };

    const Device* device_{nullptr};
    Resources* resources_{nullptr};
    ComputePipeline predict_;
    ComputePipeline solve_;
    ComputePipeline collide_;
    ComputePipeline finalize_;
    vk::raii::DescriptorPool pool_{nullptr};
    // Per-frame collider UBO (count + ClothCollider[kMaxClothColliders]), rewritten
    // each frame from the gathered world colliders.
    Resources::MappedBufferSet colliderUbo_;
    std::vector<Cloth> cloths_;
};

} // namespace fire_engine
