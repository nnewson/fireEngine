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

// XPBD: substepping (many small steps, one constraint pass each) is more stable
// than iterating a single large step. Tunable via the overlay in a later stage.
constexpr uint32_t kSubsteps = 20;
constexpr float kGravity = -9.8f;
constexpr float kDamping = 0.99f;
constexpr float kMaxFrameDt = 1.0f / 30.0f; // clamp to avoid blow-ups on a hitch
constexpr uint32_t kMaxCloths = 8;
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

// Shared push block (mirrors the `Push` block in the cloth_*.comp shaders).
struct ClothPush
{
    float gravity[4]{0.0f, kGravity, 0.0f, 0.0f};
    float dt{0.0f};
    uint32_t particleCount{0};
    uint32_t rangeBegin{0};
    uint32_t rangeEnd{0};
    float damping{kDamping};
    uint32_t resX{0};
    uint32_t resZ{0};
    uint32_t pad{0};
};

// std140 collider UBO mirroring the `Colliders` block in cloth_collide.comp.
// ClothCollider is 64 bytes (std140 element stride); count is padded to 16.
struct ClothColliderUboGpu
{
    int32_t count{0};
    int32_t pad[3]{};
    ClothCollider colliders[SoftBodySystem::kMaxClothColliders]{};
};

[[nodiscard]] ComputePipelineConfig clothConfig(const char* shader)
{
    ComputePipelineConfig config;
    config.compShaderPath = shader;
    config.bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // particles
        {1, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // constraints
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // verts
        {3, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // colliders
    };
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
    : device_{&device},
      resources_{&resources},
      predict_(device, clothConfig("cloth_predict.comp.spv")),
      solve_(device, clothConfig("cloth_solve.comp.spv")),
      collide_(device, clothConfig("cloth_collide.comp.spv")),
      finalize_(device, clothConfig("cloth_finalize.comp.spv"))
{
    // kMaxFramesInFlight descriptor sets per cloth (one per frame), each with 3
    // storage buffers + 1 uniform (the per-frame collider UBO).
    constexpr uint32_t maxSets = kMaxCloths * kMaxFramesInFlight;
    std::array<vk::DescriptorPoolSize, 2> poolSizes{{
        {vk::DescriptorType::eStorageBuffer, maxSets * 3},
        {vk::DescriptorType::eUniformBuffer, maxSets},
    }};
    vk::DescriptorPoolCreateInfo ci{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    pool_ = vk::raii::DescriptorPool(device_->device(), ci);

    colliderUbo_ = resources_->createMappedUniformBuffers(sizeof(ClothColliderUboGpu));
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
    cloth.resX = mesh.resX;
    cloth.resZ = mesh.resZ;

    vk::DescriptorSetLayout layout = predict_.descriptorSetLayout();
    for (uint32_t f = 0; f < kMaxFramesInFlight; ++f)
    {
        vk::DescriptorSetAllocateInfo ai{
            .descriptorPool = *pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        vk::raii::DescriptorSets sets(device_->device(), ai);
        cloth.sets.push_back(std::move(sets[0]));

        const std::array<vk::DescriptorBufferInfo, 3> storageInfos{{
            {resources_->vulkanBuffer(cloth.particles), 0, vk::WholeSize},
            {resources_->vulkanBuffer(cloth.constraints), 0, vk::WholeSize},
            {resources_->vulkanBuffer(cloth.verts), 0, vk::WholeSize},
        }};
        const vk::DescriptorBufferInfo colliderInfo{
            resources_->vulkanBuffer(colliderUbo_.buffers[f]), 0, vk::WholeSize};
        const vk::DescriptorSet dst = *cloth.sets[f];
        std::array<vk::WriteDescriptorSet, 4> writes{{
            {.dstSet = dst,
             .dstBinding = 0,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .pBufferInfo = &storageInfos[0]},
            {.dstSet = dst,
             .dstBinding = 1,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .pBufferInfo = &storageInfos[1]},
            {.dstSet = dst,
             .dstBinding = 2,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .pBufferInfo = &storageInfos[2]},
            {.dstSet = dst,
             .dstBinding = 3,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eUniformBuffer,
             .pBufferInfo = &colliderInfo},
        }};
        device_->device().updateDescriptorSets(writes, {});
    }

    cloths_.push_back(std::move(cloth));
}

void SoftBodySystem::recordSolve(vk::CommandBuffer cmd, float dt, uint32_t frameIndex,
                                 std::span<const ClothCollider> colliders) const
{
    if (cloths_.empty())
    {
        return;
    }

    const float frameDt = std::min(std::max(dt, 0.0f), kMaxFrameDt);
    const float subDt = frameDt / static_cast<float>(kSubsteps);
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
    std::memcpy(colliderUbo_.mapped[frameIndex], &ubo, sizeof(ubo));

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
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, predict_.pipelineLayout(), 0,
                               *c.sets[frameIndex], {});
        const uint32_t particleGroups = groups(c.particleCount);

        ClothPush push;
        push.dt = subDt;
        push.particleCount = c.particleCount;
        push.resX = c.resX;
        push.resZ = c.resZ;

        const uint32_t numColours =
            c.colourRanges.empty() ? 0u : static_cast<uint32_t>(c.colourRanges.size() - 1);

        for (uint32_t s = 0; s < kSubsteps; ++s)
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
