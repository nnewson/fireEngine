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

// Live solver parameters (overlay-tunable). gravity/wind are accelerations
// applied in the predict step; complianceScale is a global multiplier on every
// constraint's authored (per-type) compliance, so 1.0 keeps the authored
// stiffness and the slider softens/stiffens the whole cloth; substeps trades cost
// for stability.
struct ClothSimParams
{
    uint32_t substeps{20};
    float complianceScale{1.0f};
    float damping{0.99f};
    float gravity{-9.8f};
    float wind[3]{0.0f, 0.0f, 0.0f};
};

// Renderer-owned GPU soft-body (cloth) solver (Roadmap #2). Owns the XPBD compute
// pipelines and, per registered cloth, the particle + constraint buffers. Each
// frame `recordSolve` runs the substep loop (predict → per-colour distance solve
// → collide → finalize) writing solved positions + normals into the cloth's
// render vertex buffer, which the forward/shadow passes then read.
//
// The solver is descriptor-free: every buffer (particles, constraints, render
// verts, per-frame colliders) reaches the shaders as a 64-bit bufferDeviceAddress
// pointer carried in the push constant. Cloths are registered imperatively via
// addCloth — a Cloth scene component + gather is a later refactor.
class SoftBodySystem
{
public:
    SoftBodySystem(const Device& device, Resources& resources);
    ~SoftBodySystem() = default;

    SoftBodySystem(const SoftBodySystem&) = delete;
    SoftBodySystem& operator=(const SoftBodySystem&) = delete;
    SoftBodySystem(SoftBodySystem&&) noexcept = default;
    SoftBodySystem& operator=(SoftBodySystem&&) noexcept = default;

    // Register a cloth: allocate + upload its particle/constraint buffers and cache
    // their GPU addresses (plus the compute-writable render vertex buffer's).
    // `vertexBuffer` is the Geometry's storage vertex buffer.
    void addCloth(const ClothMesh& mesh, BufferHandle vertexBuffer);

    [[nodiscard]] bool empty() const noexcept
    {
        return cloths_.empty();
    }

    // Record the substep dispatch chain for every cloth (predict → solve → collide
    // per substep, then finalize), ending with a compute-write → vertex-input-read
    // barrier on each render vertex buffer. Uploads `colliders` into this frame's
    // collider buffer first. Recorded before the shadow pass.
    void recordSolve(vk::CommandBuffer cmd, float dt, uint32_t frameIndex,
                     std::span<const ClothCollider> colliders, const ClothSimParams& params) const;

    static constexpr uint32_t kMaxClothColliders = 16;

private:
    struct Cloth
    {
        BufferHandle particles{NullBuffer};
        BufferHandle constraints{NullBuffer};
        BufferHandle verts{NullBuffer};
        BufferHandle indices{NullBuffer};    // triangle list (for normal recompute)
        BufferHandle adjOffsets{NullBuffer}; // CSR per-vertex normal adjacency
        BufferHandle adjTris{NullBuffer};
        // GPU pointers (bufferDeviceAddress) of the above, pushed to the solver.
        vk::DeviceAddress particlesAddr{0};
        vk::DeviceAddress constraintsAddr{0};
        vk::DeviceAddress vertsAddr{0};
        vk::DeviceAddress indicesAddr{0};
        vk::DeviceAddress adjOffsetsAddr{0};
        vk::DeviceAddress adjTrisAddr{0};
        uint32_t particleCount{0};
        std::vector<uint32_t> colourRanges;
    };

    Resources* resources_{nullptr};
    ComputePipeline predict_;
    ComputePipeline solve_;
    ComputePipeline collide_;
    ComputePipeline finalize_;
    // Per-frame collider buffer (count + ClothCollider[kMaxClothColliders]) +
    // its device address, rewritten each frame from the gathered world colliders.
    Resources::MappedBufferSet colliderBuffers_;
    std::array<vk::DeviceAddress, kMaxFramesInFlight> colliderAddrs_{};
    std::vector<Cloth> cloths_;
};

} // namespace fire_engine
