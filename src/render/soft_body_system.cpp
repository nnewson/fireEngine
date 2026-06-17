#include <fire_engine/render/soft_body_system.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>

#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/render/device.hpp>

namespace fire_engine
{

namespace
{

// XPBD substepping, solver tunables, etc. now come from ClothSimParams (overlay).
constexpr float kMaxFrameDt = 1.0f / 30.0f; // clamp to avoid blow-ups on a hitch
constexpr uint32_t kLocalSize = 64;

// The finalize shader treats the render vertex buffer as a flat float array; this
// must stay in lockstep with the C++ Vertex layout (position at float 0, normal at
// float 6). Vertex is standard-layout with position_/colour_/normal_ first.
static_assert(sizeof(Vertex) == 100, "cloth_finalize.comp assumes a 100-byte (25-float) Vertex");

// GPU mirrors. std430: 3×vec4 = 48 bytes; {uint,uint,float,float} = 16 bytes.
struct ParticleGpu
{
    float pos[4];  // xyz, w = invMass
    float prev[4]; // xyz
    float vel[4];  // xyz
};

struct ConstraintGpu
{
    uint32_t a;
    uint32_t b;
    float restLength;
    float compliance;
};

// Shared push block, mirroring the `Push` block in the cloth_*.comp shaders. The
// two vec4s lead so they land at 16-byte-aligned offsets; then the seven 64-bit
// buffer_reference pointers (8 bytes each, the first aligned at offset 32); then
// the scalars. 112 bytes total.
struct ClothPush
{
    float gravity[4]{0.0f, -9.8f, 0.0f, 0.0f};
    float wind[4]{0.0f, 0.0f, 0.0f, 0.0f};
    vk::DeviceAddress particles{0};
    vk::DeviceAddress constraints{0};
    vk::DeviceAddress verts{0};
    vk::DeviceAddress colliders{0};
    vk::DeviceAddress indices{0};
    vk::DeviceAddress adjOffsets{0};
    vk::DeviceAddress adjTris{0};
    float dt{0.0f};
    uint32_t particleCount{0};
    uint32_t rangeBegin{0};
    uint32_t rangeEnd{0};
    float damping{0.99f};
    float complianceScale{1.0f};
};
static_assert(sizeof(ClothPush) == 112, "ClothPush must match the std430 shader push layout");

// std430 collider buffer mirroring the `Colliders` buffer_reference in the
// shaders. ClothCollider is 64 bytes; count is padded to 16 before the array.
struct ClothColliderUboGpu
{
    int32_t count{0};
    int32_t pad[3]{};
    ClothCollider colliders[SoftBodySystem::kMaxClothColliders]{};
};

[[nodiscard]] ComputePipelineConfig clothConfig(const char* shader)
{
    // No descriptor bindings — every buffer reaches the shader as a 64-bit GPU
    // pointer (bufferDeviceAddress) carried in the push constant.
    ComputePipelineConfig config;
    config.compShaderPath = shader;
    config.pushConstantRanges.emplace_back(vk::ShaderStageFlagBits::eCompute, 0,
                                           static_cast<uint32_t>(sizeof(ClothPush)));
    return config;
}

[[nodiscard]] uint32_t groups(uint32_t count)
{
    return (count + kLocalSize - 1) / kLocalSize;
}

} // namespace

SoftBodySystem::SoftBodySystem(const Device& device, Resources& resources)
    : resources_{&resources},
      predict_(device, clothConfig("cloth_predict.comp.spv")),
      solve_(device, clothConfig("cloth_solve.comp.spv")),
      collide_(device, clothConfig("cloth_collide.comp.spv")),
      finalize_(device, clothConfig("cloth_finalize.comp.spv"))
{
    // Per-frame collider buffer (device-addressable storage) + cached addresses.
    colliderBuffers_ = resources_->createMappedDeviceAddressBuffers(sizeof(ClothColliderUboGpu));
    for (int f = 0; f < kMaxFramesInFlight; ++f)
    {
        colliderAddrs_[f] = resources_->bufferAddress(colliderBuffers_.buffers[f]);
    }
}

void SoftBodySystem::addCloth(const ClothMesh& mesh, BufferHandle vertexBuffer)
{
    const uint32_t count = static_cast<uint32_t>(mesh.particles.size());

    std::vector<ParticleGpu> particles(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        const ClothParticle& src = mesh.particles[i];
        particles[i] = ParticleGpu{
            .pos = {src.position.x(), src.position.y(), src.position.z(), src.invMass},
            .prev = {src.position.x(), src.position.y(), src.position.z(), 0.0f},
            .vel = {0.0f, 0.0f, 0.0f, 0.0f},
        };
    }

    std::vector<ConstraintGpu> constraints(mesh.constraints.size());
    for (std::size_t i = 0; i < mesh.constraints.size(); ++i)
    {
        const ClothConstraint& c = mesh.constraints[i];
        constraints[i] = ConstraintGpu{
            .a = c.a, .b = c.b, .restLength = c.restLength, .compliance = c.compliance};
    }

    Cloth cloth;
    cloth.particles =
        resources_->createStorageBuffer(particles.size() * sizeof(ParticleGpu), particles.data());
    cloth.constraints = resources_->createStorageBuffer(constraints.size() * sizeof(ConstraintGpu),
                                                        constraints.data());
    cloth.verts = vertexBuffer;
    cloth.particleCount = count;
    cloth.colourRanges = mesh.colourRanges;

    // Index list + CSR normal adjacency: constant per cloth, read by the finalize
    // pass to recompute per-vertex normals from the current particle positions.
    cloth.indices = resources_->createStorageBuffer(mesh.indices.size() * sizeof(uint32_t),
                                                    mesh.indices.data());
    cloth.adjOffsets = resources_->createStorageBuffer(
        mesh.normalAdjOffsets.size() * sizeof(uint32_t), mesh.normalAdjOffsets.data());
    cloth.adjTris = resources_->createStorageBuffer(mesh.normalAdjTris.size() * sizeof(uint32_t),
                                                    mesh.normalAdjTris.data());

    // Cache the GPU pointers the solver pushes each dispatch (bufferDeviceAddress).
    cloth.particlesAddr = resources_->bufferAddress(cloth.particles);
    cloth.constraintsAddr = resources_->bufferAddress(cloth.constraints);
    cloth.vertsAddr = resources_->bufferAddress(cloth.verts);
    cloth.indicesAddr = resources_->bufferAddress(cloth.indices);
    cloth.adjOffsetsAddr = resources_->bufferAddress(cloth.adjOffsets);
    cloth.adjTrisAddr = resources_->bufferAddress(cloth.adjTris);

    cloths_.push_back(std::move(cloth));
}

void SoftBodySystem::recordSolve(vk::CommandBuffer cmd, float dt, uint32_t frameIndex,
                                 std::span<const ClothCollider> colliders,
                                 const ClothSimParams& params) const
{
    if (cloths_.empty())
    {
        return;
    }

    const uint32_t substeps = std::max(1u, params.substeps);
    const float frameDt = std::min(std::max(dt, 0.0f), kMaxFrameDt);
    const float subDt = frameDt / static_cast<float>(substeps);
    if (subDt <= 0.0f)
    {
        return;
    }

    // Upload this frame's world colliders into its UBO (per-frame, so we don't
    // race a still-in-flight read).
    ClothColliderUboGpu ubo{};
    ubo.count = static_cast<int32_t>(std::min<std::size_t>(colliders.size(), kMaxClothColliders));
    for (int32_t k = 0; k < ubo.count; ++k)
    {
        ubo.colliders[k] = colliders[static_cast<std::size_t>(k)];
    }
    std::memcpy(colliderBuffers_.mapped[frameIndex], &ubo, sizeof(ubo));

    auto particleBarrier = [&](const Cloth& c)
    {
        recordBufferBarrier(
            cmd, makeBufferMemoryBarrier(
                     vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                     vk::PipelineStageFlagBits2::eComputeShader,
                     vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                     resources_->vulkanBuffer(c.particles), 0, vk::WholeSize));
    };

    for (const Cloth& c : cloths_)
    {
        const uint32_t particleGroups = groups(c.particleCount);

        ClothPush push;
        push.particles = c.particlesAddr;
        push.constraints = c.constraintsAddr;
        push.verts = c.vertsAddr;
        push.colliders = colliderAddrs_[frameIndex];
        push.indices = c.indicesAddr;
        push.adjOffsets = c.adjOffsetsAddr;
        push.adjTris = c.adjTrisAddr;
        push.gravity[0] = 0.0f;
        push.gravity[1] = params.gravity;
        push.gravity[2] = 0.0f;
        push.wind[0] = params.wind[0];
        push.wind[1] = params.wind[1];
        push.wind[2] = params.wind[2];
        push.dt = subDt;
        push.particleCount = c.particleCount;
        push.damping = params.damping;
        push.complianceScale = params.complianceScale;

        const uint32_t numColours =
            c.colourRanges.empty() ? 0u : static_cast<uint32_t>(c.colourRanges.size() - 1);

        for (uint32_t s = 0; s < substeps; ++s)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, predict_.pipeline());
            cmd.pushConstants<ClothPush>(predict_.pipelineLayout(),
                                         vk::ShaderStageFlagBits::eCompute, 0, push);
            cmd.dispatch(particleGroups, 1, 1);
            particleBarrier(c);

            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, solve_.pipeline());
            for (uint32_t col = 0; col < numColours; ++col)
            {
                push.rangeBegin = c.colourRanges[col];
                push.rangeEnd = c.colourRanges[col + 1];
                const uint32_t n = push.rangeEnd - push.rangeBegin;
                if (n == 0)
                {
                    continue;
                }
                cmd.pushConstants<ClothPush>(solve_.pipelineLayout(),
                                             vk::ShaderStageFlagBits::eCompute, 0, push);
                cmd.dispatch(groups(n), 1, 1);
                particleBarrier(c);
            }

            // Collision projection (after the distance solve), reads the collider
            // UBO + particles, writes particles.
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, collide_.pipeline());
            cmd.pushConstants<ClothPush>(collide_.pipelineLayout(),
                                         vk::ShaderStageFlagBits::eCompute, 0, push);
            cmd.dispatch(particleGroups, 1, 1);
            particleBarrier(c);
        }

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, finalize_.pipeline());
        cmd.pushConstants<ClothPush>(finalize_.pipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                                     0, push);
        cmd.dispatch(particleGroups, 1, 1);

        // Solved positions + normals are now in the render vertex buffer: order the
        // compute writes before the shadow/forward vertex-input reads.
        recordBufferBarrier(
            cmd, makeBufferMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader,
                                         vk::AccessFlagBits2::eShaderWrite,
                                         vk::PipelineStageFlagBits2::eVertexAttributeInput,
                                         vk::AccessFlagBits2::eVertexAttributeRead,
                                         resources_->vulkanBuffer(c.verts), 0, vk::WholeSize));
    }
}

} // namespace fire_engine
