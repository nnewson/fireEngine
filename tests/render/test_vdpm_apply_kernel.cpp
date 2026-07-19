#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/vdpm_gpu.hpp>

#include <support/vdpm.hpp>

using namespace fire_engine;

// VDPM persistent APPLY kernel cross-check ([.][gpu], local only). The kernel
// (vdpm_apply_kernel.comp, recordApplyKernel) must be a BIT-EXACT port of the multi-dispatch
// recorder (recordApplyScoredView) for the SAME scores — apply is integer state, a single
// deterministic pass (no FP fixpoint), so the two GPU paths agree exactly. Method (Stage-3
// CrossCheckRunner shape): two buildRuntime fronts from one mesh, scored identically (deterministic
// ⇒ identical GPU score buffers — PROVEN by reading both back), then apply one via the kernel and
// one via the recorder; assert active/refined/dependents/ required/failFlags EXACTLY equal +
// validateFrontInvariants on the kernel front. The recorder is itself cross-checked element-exact
// against the CPU oracle (ParallelFront::applyView) in test_vdpm_gpu_front.cpp, so kernel ==
// recorder gives kernel == oracle-scheduling transitively.

namespace
{

struct Mesh
{
    std::vector<Vertex> verts;
    std::vector<std::uint32_t> indices;
};

Mesh uvSphere(int rings, int segments)
{
    Mesh m;
    const float pi = 3.14159265f;
    for (int r = 0; r <= rings; ++r)
    {
        const float lat = pi * ((static_cast<float>(r) / static_cast<float>(rings)) - 0.5f);
        for (int s = 0; s <= segments; ++s)
        {
            const float lon = 2.0f * pi * static_cast<float>(s) / static_cast<float>(segments);
            const Vec3 nrm{std::cos(lat) * std::cos(lon), std::sin(lat),
                           std::cos(lat) * std::sin(lon)};
            m.verts.push_back(Vertex{nrm, Colour3{}, nrm, Vec2{0.0f, 0.0f}});
        }
    }
    const int stride = segments + 1;
    for (int r = 0; r < rings; ++r)
    {
        for (int s = 0; s < segments; ++s)
        {
            const auto a = static_cast<std::uint32_t>((r * stride) + s);
            const auto b = static_cast<std::uint32_t>((r * stride) + s + 1);
            const auto c = static_cast<std::uint32_t>(((r + 1) * stride) + s);
            const auto d = static_cast<std::uint32_t>(((r + 1) * stride) + s + 1);
            m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
        }
    }
    return m;
}

// The score view the runtime front's recordScore consumes (cone predicate + channel scales).
VdpmViewParams scoreViewOf(const Mat4& world, const Vec3& cam, float vh, bool cull)
{
    return makeVdpmViewParams(world, cam, 1.0f, vh, 2.0f, cull, 1.0f, 1.0f, 1.0f);
}

// Read-back-front adapter for validateFrontInvariants' Front interface.
struct GpuFrontView
{
    const VertexForest& forest_;
    std::span<const std::uint32_t> active_;
    [[nodiscard]] const VertexForest& forest() const noexcept
    {
        return forest_;
    }
    [[nodiscard]] bool active(std::uint32_t v) const
    {
        return active_[v] != 0u;
    }
};

constexpr float kVh = 768.0f;

struct ApplyState
{
    std::vector<std::uint32_t> scores; // splitCount * 6 raw uint words (VdpmScoreOut = 24 B)
    std::vector<std::uint32_t> active;
    std::vector<std::uint32_t> refined;
    std::vector<std::uint32_t> dependents;
    std::vector<std::uint32_t> required;
    std::array<std::uint32_t, 2> failFlags{};
};

// Two runtime fronts from one mesh: frontK driven by the apply KERNEL, frontR by the recorder. Both
// persistent, so a sequence of applyOnce() calls exercises multi-frame refine → coarsen.
class ApplyCrossCheckRunner
{
public:
    ApplyCrossCheckRunner(std::span<const Vertex> verts, std::span<const std::uint32_t> indices,
                          const VertexForest& forest)
        : device_(Device::headlessCompute()),
          resources_(device_),
          scorePipeline_(device_, vdpmScorePipelineConfig()),
          refinePipelines_(device_),
          kernel_(device_),
          pool_(
              device_.device(),
              vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        .queueFamilyIndex = device_.graphicsFamily()}),
          mesh_(VdpmGpuMesh::build(resources_, verts, indices, forest)),
          frontK_(VdpmGpuFront::buildRuntime(resources_, mesh_)),
          frontR_(VdpmGpuFront::buildRuntime(resources_, mesh_)),
          vertexCount_(forest.vertexCount),
          splitCount_(static_cast<std::uint32_t>(forest.splits.size()))
    {
    }
    ApplyCrossCheckRunner(const ApplyCrossCheckRunner&) = delete;
    ApplyCrossCheckRunner& operator=(const ApplyCrossCheckRunner&) = delete;
    ApplyCrossCheckRunner(ApplyCrossCheckRunner&&) = delete;
    ApplyCrossCheckRunner& operator=(ApplyCrossCheckRunner&&) = delete;
    ~ApplyCrossCheckRunner() = default;

    [[nodiscard]] static bool supported()
    {
        return VdpmApplyKernel::deviceSupported(Device::headlessCompute());
    }

    struct Pair
    {
        ApplyState k;
        ApplyState r;
    };

    // ONE submit: score + apply BOTH fronts (kernel vs recorder) for the given view/budgets, then
    // read back scores + state of each. State persists in the fronts across calls.
    [[nodiscard]] Pair applyOnce(const VdpmViewParams& scoreView, float pixelBudget,
                                 float coarsenBudget)
    {
        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        // frontK: score → apply KERNEL (its own leading score→kernel barrier). frontR: score →
        // apply RECORDER (recordApplyScoredView's leading barrier). Separate fronts, disjoint
        // buffers.
        frontK_.recordScore(*cmd, scorePipeline_, 0, scoreView);
        frontK_.recordApplyKernel(*cmd, kernel_, 0, pixelBudget, coarsenBudget);
        frontR_.recordScore(*cmd, scorePipeline_, 0, scoreView);
        frontR_.recordApplyScoredView(*cmd, refinePipelines_, resources_, pixelBudget,
                                      coarsenBudget);

        const vk::MemoryBarrier2 toTransfer{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toTransfer});

        auto copyBack = [&](BufferHandle src, std::uint32_t words)
        {
            const vk::DeviceSize size = static_cast<vk::DeviceSize>(words) * sizeof(std::uint32_t);
            Resources::MappedBufferSet host = resources_.createMappedReadbackBuffers(size);
            cmd.copyBuffer(resources_.vulkanBuffer(src), resources_.vulkanBuffer(host.buffers[0]),
                           vk::BufferCopy{.size = size});
            return host;
        };
        struct Handles
        {
            Resources::MappedBufferSet scores, active, refined, dependents, required, failFlags;
        };
        const std::uint32_t scoreWords =
            splitCount_ * (sizeof(VdpmScoreOut) / sizeof(std::uint32_t));
        auto queue = [&](VdpmGpuFront& f)
        {
            return Handles{.scores = copyBack(f.outputBuffer(), scoreWords),
                           .active = copyBack(f.activeStateBuffer(), vertexCount_),
                           .refined = copyBack(f.refinedStateBuffer(), splitCount_),
                           .dependents = copyBack(f.dependentsStateBuffer(), vertexCount_),
                           .required = copyBack(f.requiredStateBuffer(), splitCount_),
                           .failFlags = copyBack(f.failFlagsBuffer(), 2)};
        };
        const Handles hk = queue(frontK_);
        const Handles hr = queue(frontR_);

        const vk::MemoryBarrier2 toHost{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
        cmd.end();

        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        const vk::SubmitInfo2 submit{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo};
        const vk::raii::Fence fence(device_.device(), vk::FenceCreateInfo{});
        device_.graphicsQueue().submit2(submit, *fence);
        (void)device_.device().waitForFences(*fence, vk::True,
                                             std::numeric_limits<std::uint64_t>::max());

        auto read = [](const Resources::MappedBufferSet& set, std::uint32_t count)
        {
            std::vector<std::uint32_t> out(count);
            std::memcpy(out.data(), set.mapped[0].data(), count * sizeof(std::uint32_t));
            return out;
        };
        auto parse = [&](const Handles& h)
        {
            ApplyState st;
            st.scores = read(h.scores, scoreWords);
            st.active = read(h.active, vertexCount_);
            st.refined = read(h.refined, splitCount_);
            st.dependents = read(h.dependents, vertexCount_);
            st.required = read(h.required, splitCount_);
            std::ranges::copy(read(h.failFlags, 2), st.failFlags.begin());
            return st;
        };
        return Pair{.k = parse(hk), .r = parse(hr)};
    }

private:
    Device device_;
    Resources resources_;
    ComputePipeline scorePipeline_;
    VdpmRefinePipelines refinePipelines_;
    VdpmApplyKernel kernel_;
    vk::raii::CommandPool pool_;
    VdpmGpuMesh mesh_;
    VdpmGpuFront frontK_;
    VdpmGpuFront frontR_;
    std::uint32_t vertexCount_;
    std::uint32_t splitCount_;
};

// Assert one applyOnce result: identical scores (proven, not assumed), bit-exact apply state, and a
// valid kernel front.
void checkApply(const ApplyCrossCheckRunner::Pair& p, const VertexForest& forest)
{
    // Identical GPU score inputs — REQUIRE (a mismatch makes the state comparison meaningless).
    REQUIRE(p.k.scores == p.r.scores);
    // Bit-exact apply: kernel == recorder across ALL persistent + transient state.
    CHECK(p.k.active == p.r.active);
    CHECK(p.k.refined == p.r.refined);
    CHECK(p.k.dependents == p.r.dependents);
    CHECK(p.k.required == p.r.required);
    CHECK(p.k.failFlags == p.r.failFlags);
    // The kernel front is genuinely valid (both GPU paths could share a defect the mutual
    // comparison can't see); no refine failure / dependents underflow.
    CHECK(p.k.failFlags[0] == 0u);
    CHECK(p.k.failFlags[1] == 0u);
    CHECK_NOTHROW(validateFrontInvariants(forest, p.k.active, p.k.refined, p.k.dependents));
}

VertexForest sphereForest(const Mesh& m)
{
    const QuadricSimplifier simp;
    return buildVertexForest(m.verts, simp.collapseSequence(m.verts, m.indices));
}

} // namespace

TEST_CASE("VDPM apply kernel == recorder: full refine then full coarsen", "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    const Mesh m = uvSphere(18, 24);
    const VertexForest forest = sphereForest(m);
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    const VdpmViewParams near = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.6f}, kVh, true);

    // Frame 1: a tiny budget drives a deep refine. Frame 2: a huge budget coarsens the persistent
    // front back down — exercises the coarsen descending pass on a heavily-refined front.
    checkApply(runner.applyOnce(near, 0.5f, kVdpmCoarsenRatio * 0.5f), forest);
    checkApply(runner.applyOnce(near, 1.0e9f, kVdpmCoarsenRatio * 1.0e9f), forest);
}

TEST_CASE("VDPM apply kernel == recorder: alternating budgets, back-to-back persistent frames",
          "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    const Mesh m = uvSphere(16, 20);
    const VertexForest forest = sphereForest(m);
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 2.2f}, kVh, true);

    // The persistent front swings between refine-heavy and coarsen-heavy each frame; the hysteresis
    // dead-band means the two directions take different splits, so every frame must still match.
    for (const float budget : {2.0f, 8.0f, 3.0f, 12.0f, 1.0f})
    {
        checkApply(runner.applyOnce(view, budget, kVdpmCoarsenRatio * budget), forest);
    }
}

TEST_CASE("VDPM apply kernel == recorder: diamond dependencies (shared parents/vl/vr)", "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    // A denser sphere maximises shared dependency vertices (a split's vl/vr feed several finer
    // splits — the DAG diamonds the closure must fan through), stressing the atomic-OR closure +
    // atomic-add/sub dependents under concurrency.
    const Mesh m = uvSphere(24, 32);
    const VertexForest forest = sphereForest(m);
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.9f}, kVh, true);
    checkApply(runner.applyOnce(view, 0.8f, kVdpmCoarsenRatio * 0.8f), forest);
    checkApply(runner.applyOnce(view, 25.0f, kVdpmCoarsenRatio * 25.0f), forest);
}

TEST_CASE("VDPM apply kernel == recorder: back-facing coarsening (cull on)", "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    const Mesh m = uvSphere(18, 24);
    const VertexForest forest = sphereForest(m);
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    // Refine front-on, then move the camera to the far side: the now back-facing splits score 0 and
    // must coarsen (the backface branch of applyCoarsenEligible) — kernel + recorder identically.
    const VdpmViewParams front = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.7f}, kVh, true);
    const VdpmViewParams behind = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, -1.7f}, kVh, true);
    checkApply(runner.applyOnce(front, 0.6f, kVdpmCoarsenRatio * 0.6f), forest);
    checkApply(runner.applyOnce(behind, 0.6f, kVdpmCoarsenRatio * 0.6f), forest);
}
