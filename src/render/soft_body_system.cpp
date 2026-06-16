#include <fire_engine/render/soft_body_system.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

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

[[nodiscard]] ComputePipelineConfig clothConfig(const char* shader)
{
    ComputePipelineConfig config;
    config.compShaderPath = shader;
    config.bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // particles
        {1, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // constraints
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // verts
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
      finalize_(device, clothConfig("cloth_finalize.comp.spv"))
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes{
        {{vk::DescriptorType::eStorageBuffer, kMaxCloths * 3}}};
    vk::DescriptorPoolCreateInfo ci{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = kMaxCloths,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    pool_ = vk::raii::DescriptorPool(device_->device(), ci);
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
    vk::DescriptorSetAllocateInfo ai{
        .descriptorPool = *pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    vk::raii::DescriptorSets sets(device_->device(), ai);
    cloth.set = std::move(sets[0]);

    const std::array<vk::DescriptorBufferInfo, 3> infos{{
        {resources_->vulkanBuffer(cloth.particles), 0, vk::WholeSize},
        {resources_->vulkanBuffer(cloth.constraints), 0, vk::WholeSize},
        {resources_->vulkanBuffer(cloth.verts), 0, vk::WholeSize},
    }};
    std::array<vk::WriteDescriptorSet, 3> writes;
    for (uint32_t b = 0; b < 3; ++b)
    {
        writes[b] = vk::WriteDescriptorSet{
            .dstSet = *cloth.set,
            .dstBinding = b,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &infos[b],
        };
    }
    device_->device().updateDescriptorSets(writes, {});

    cloths_.push_back(std::move(cloth));
}

void SoftBodySystem::recordSolve(vk::CommandBuffer cmd, float dt) const
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
                               *c.set, {});
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
