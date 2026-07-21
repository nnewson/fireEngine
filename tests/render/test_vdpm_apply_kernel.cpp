#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
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

// A flat n×n grid — its QEM collapse stream forms a DEEP rank chain (high maxRank), stressing the
// kernel's descending close/coarsen rank loops and the wgsync between many ranks.
Mesh grid(int n)
{
    Mesh m;
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            m.verts.push_back(Vertex{Vec3{static_cast<float>(x), static_cast<float>(y), 0.0f},
                                     Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f}});
        }
    }
    for (int y = 0; y < n - 1; ++y)
    {
        for (int x = 0; x < n - 1; ++x)
        {
            const auto a = static_cast<std::uint32_t>((y * n) + x);
            const auto b = static_cast<std::uint32_t>((y * n) + x + 1);
            const auto c = static_cast<std::uint32_t>(((y + 1) * n) + x);
            const auto d = static_cast<std::uint32_t>(((y + 1) * n) + x + 1);
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

Mat4 lookAtProj(const Vec3& cam)
{
    Mat4 view = Mat4::identity();
    view[0, 3] = -cam.x();
    view[1, 3] = -cam.y();
    view[2, 3] = -cam.z();
    Mat4 proj = Mat4::identity();
    proj[2, 2] = -1.0f;
    proj[3, 2] = -1.0f;
    proj[3, 3] = 0.0f;
    return proj * view;
}

VdpmRepairParams repairParamsOf(const Mat4& world, const Vec3& cam, float vw, float vh, bool cull)
{
    VdpmRepairParams p{};
    p.world = world;
    p.viewProj = lookAtProj(cam);
    p.cameraPos[0] = cam.x();
    p.cameraPos[1] = cam.y();
    p.cameraPos[2] = cam.z();
    p.viewport[0] = vw;
    p.viewport[1] = vh;
    p.viewport[2] = cull ? 1.0f : 0.0f;
    return p;
}

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
    // read back scores + state of each. State persists in the fronts across calls. When
    // `dirtyFailFlags` is set, BOTH fronts' failFlags are prefilled with 0xFFFFFFFF before the
    // apply — so a clean apply reading back 0 directly proves the reset (the kernel's in-kernel
    // clear, the recorder's leading fillBuffer), not merely that the buffers started clean.
    [[nodiscard]] Pair applyOnce(const VdpmViewParams& scoreView, float pixelBudget,
                                 float coarsenBudget, bool dirtyFailFlags = false)
    {
        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        if (dirtyFailFlags)
        {
            for (VdpmGpuFront* f : {&frontK_, &frontR_})
            {
                cmd.fillBuffer(resources_.vulkanBuffer(f->failFlagsBuffer()), 0,
                               2 * sizeof(std::uint32_t), 0xFFFFFFFFu);
            }
            const vk::MemoryBarrier2 clearToCompute{
                .srcStageMask = vk::PipelineStageFlagBits2::eClear,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                                 vk::AccessFlagBits2::eShaderStorageWrite,
            };
            cmd.pipelineBarrier2(
                vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});
        }

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
            Resources::MappedBufferSet host;
            if (words == 0) // zero-split front: no split-sized buffer to read (VMA rejects 0 bytes)
            {
                return host;
            }
            const vk::DeviceSize size = static_cast<vk::DeviceSize>(words) * sizeof(std::uint32_t);
            host = resources_.createMappedReadbackBuffers(size);
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
            if (count > 0)
            {
                std::memcpy(out.data(), set.mapped[0].data(), count * sizeof(std::uint32_t));
            }
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

    // Test-only underflow gate: assumes frontK is already refined (dependents > 0), then CORRUPTS
    // (zeroes) its dependents and coarsens — every coarsen atomic-sub now decrements a 0 dependent,
    // which the kernel's subOne must detect and flag in failFlags[1]. The corruption is a raw
    // fillBuffer here (test-only; NOT a production state-mutation API). Returns frontK's failFlags
    // after the coarsen (the in-kernel reset clears them at entry, so a set [1] is genuinely from
    // this coarsen).
    [[nodiscard]] std::array<std::uint32_t, 2>
    corruptDependentsThenCoarsenK(const VdpmViewParams& view, float coarsenBudget)
    {
        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        cmd.fillBuffer(resources_.vulkanBuffer(frontK_.dependentsStateBuffer()), 0,
                       static_cast<vk::DeviceSize>(vertexCount_) * sizeof(std::uint32_t), 0u);
        const vk::MemoryBarrier2 clearToCompute{
            .srcStageMask = vk::PipelineStageFlagBits2::eClear,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask =
                vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &clearToCompute});
        frontK_.recordScore(*cmd, scorePipeline_, 0, view);
        frontK_.recordApplyKernel(*cmd, kernel_, 0, coarsenBudget,
                                  kVdpmCoarsenRatio * coarsenBudget);

        const vk::MemoryBarrier2 toTransfer{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toTransfer});
        Resources::MappedBufferSet host =
            resources_.createMappedReadbackBuffers(2 * sizeof(std::uint32_t));
        cmd.copyBuffer(resources_.vulkanBuffer(frontK_.failFlagsBuffer()),
                       resources_.vulkanBuffer(host.buffers[0]),
                       vk::BufferCopy{.size = 2 * sizeof(std::uint32_t)});
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
        std::array<std::uint32_t, 2> flags{};
        std::memcpy(flags.data(), host.mapped[0].data(), sizeof(flags));
        return flags;
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

VertexForest qemForest(const Mesh& m)
{
    const QuadricSimplifier simp;
    return buildVertexForest(m.verts, simp.collapseSequence(m.verts, m.indices));
}

double medianMs(std::vector<double> v)
{
    if (v.empty())
    {
        return 0.0;
    }
    std::ranges::sort(v);
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[(n / 2) - 1] + v[n / 2]);
}

// Workgroup-size sweep (apply-kernel arc, adj 3) on one mesh — GPU-timestamp the apply dispatch at
// 64/128/256 and WARN the median. Statistically meaningful, NOT one-shot: three independent,
// identically-initialised persistent fronts (one per size, so mutation doesn't bias later samples);
// pipelines + warm-up OUTSIDE timing; alternating refine/coarsen so every size sees equivalent
// work; median over repeats with the per-cycle size order ROTATED to blunt thermal/DVFS bias. Read
// the numbers and bake the winner into VdpmApplyKernel::kLocalSize — or keep 256 on a material tie.
void applySizeSweep(const Mesh& m, const char* label)
{
    Device device = Device::headlessCompute();
    const vk::PhysicalDeviceLimits& limits = device.physicalDevice().getProperties().limits;
    const auto qfp = device.physicalDevice().getQueueFamilyProperties();
    if (limits.timestampPeriod == 0.0f || qfp[device.graphicsFamily()].timestampValidBits == 0)
    {
        WARN(label << ": device has no compute-queue timestamp support — sweep skipped");
        return;
    }
    Resources resources(device);
    ComputePipeline scorePipeline(device, vdpmScorePipelineConfig());
    const VertexForest forest = qemForest(m);
    const VdpmGpuMesh mesh = VdpmGpuMesh::build(resources, m.verts, m.indices, forest);

    constexpr std::array<std::uint32_t, 3> sizes{64u, 128u, 256u};
    std::array<std::optional<VdpmApplyKernel>, 3> kernels; // non-movable → optional + emplace
    std::vector<VdpmGpuFront> fronts;
    for (std::size_t i = 0; i < sizes.size(); ++i)
    {
        kernels[i].emplace(device, sizes[i]);
        fronts.push_back(VdpmGpuFront::buildRuntime(resources, mesh));
    }

    const vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});
    const vk::raii::QueryPool queryPool(
        device.device(),
        vk::QueryPoolCreateInfo{.queryType = vk::QueryType::eTimestamp, .queryCount = 2});
    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 2.5f}, kVh, true);

    // Score + GPU-timestamped apply on front i (persistent), returning the apply's GPU ms.
    auto timedApply = [&](std::size_t i, float budget) -> double
    {
        auto cmds = device.device().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{.commandPool = *pool,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1});
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.resetQueryPool(*queryPool, 0, 2);
        fronts[i].recordScore(*cmd, scorePipeline, 0, view);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *queryPool, 0);
        fronts[i].recordApplyKernel(*cmd, *kernels[i], 0, budget, kVdpmCoarsenRatio * budget);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *queryPool, 1);
        cmd.end();
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        const vk::SubmitInfo2 submit{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo};
        const vk::raii::Fence fence(device.device(), vk::FenceCreateInfo{});
        device.graphicsQueue().submit2(submit, *fence);
        (void)device.device().waitForFences(*fence, vk::True,
                                            std::numeric_limits<std::uint64_t>::max());
        const auto [res, data] = queryPool.getResults<std::uint64_t>(
            0, 2, 2 * sizeof(std::uint64_t), sizeof(std::uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        return static_cast<double>(data[1] - data[0]) *
               static_cast<double>(limits.timestampPeriod) / 1.0e6;
    };

    // Warm-up OUTSIDE timing: one refine + one coarsen per front (pipeline/first-submit costs
    // paid).
    for (std::size_t i = 0; i < sizes.size(); ++i)
    {
        (void)timedApply(i, 0.4f);
        (void)timedApply(i, 1.0e9f);
    }
    // Timed cycles: alternating refine/coarsen budgets (equivalent work per size), size order
    // rotated per cycle.
    constexpr int kCycles = 24;
    std::array<std::vector<double>, 3> samples;
    for (int c = 0; c < kCycles; ++c)
    {
        const float budget = (c % 2 == 0) ? 0.4f : 1.0e9f;
        for (std::size_t k = 0; k < sizes.size(); ++k)
        {
            const std::size_t i = (k + static_cast<std::size_t>(c)) % sizes.size();
            samples[i].push_back(timedApply(i, budget));
        }
    }
    for (std::size_t i = 0; i < sizes.size(); ++i)
    {
        WARN(label << " local_size_x=" << sizes[i] << ": median apply " << medianMs(samples[i])
                   << " ms GPU (" << samples[i].size() << " samples)");
    }
    SUCCEED("sweep reported medians for "
            << label); // evidence test — the WARN'd medians are the output
}

} // namespace

TEST_CASE("VDPM apply kernel == recorder: full refine then full coarsen", "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    const Mesh m = uvSphere(18, 24);
    const VertexForest forest = qemForest(m);
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
    const VertexForest forest = qemForest(m);
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
    const VertexForest forest = qemForest(m);
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
    const VertexForest forest = qemForest(m);
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    // Refine front-on, then move the camera to the far side: the now back-facing splits score 0 and
    // must coarsen (the backface branch of applyCoarsenEligible) — kernel + recorder identically.
    const VdpmViewParams front = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.7f}, kVh, true);
    const VdpmViewParams behind = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, -1.7f}, kVh, true);
    checkApply(runner.applyOnce(front, 0.6f, kVdpmCoarsenRatio * 0.6f), forest);
    checkApply(runner.applyOnce(behind, 0.6f, kVdpmCoarsenRatio * 0.6f), forest);
}

TEST_CASE("VDPM apply kernel: in-kernel failFlags reset (prefilled nonzero → 0)", "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    const Mesh m = uvSphere(18, 24);
    const VertexForest forest = qemForest(m);
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 2.0f}, kVh, true);
    // Prefill BOTH failFlags with 0xFFFFFFFF, then run a CLEAN apply: reading back 0 proves the
    // kernel's in-kernel reset ran (a clean apply on initially-clean buffers would leave 0 either
    // way — this starts dirty, so 0 can only come from the reset). checkApply asserts both == 0.
    checkApply(runner.applyOnce(view, 1.0f, kVdpmCoarsenRatio * 1.0f, /*dirtyFailFlags=*/true),
               forest);
}

TEST_CASE("VDPM apply kernel: coarsen dependents-underflow sets failFlags[1]", "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    const Mesh m = uvSphere(16, 20);
    const VertexForest forest = qemForest(m);
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.8f}, kVh, true);
    // Refine frontK (dependents > 0), then corrupt its dependents to 0 and coarsen: every
    // collapse's atomic-sub decrements a 0 → the kernel's subOne must flag failFlags[1].
    (void)runner.applyOnce(view, 0.5f, kVdpmCoarsenRatio * 0.5f);
    const std::array<std::uint32_t, 2> flags =
        runner.corruptDependentsThenCoarsenK(view, /*coarsenBudget=*/1.0e9f);
    CHECK(flags[1] != 0u); // dependents-underflow detected
}

TEST_CASE("VDPM apply kernel == recorder: zero-split front is a clean no-op (no dispatch)",
          "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    // A single triangle can't collapse — build a genuinely zero-split forest (all verts roots). The
    // kernel's recordApplyKernel must early-out (splitCount == 0, no dispatch) exactly as the
    // recorder does; the fronts stay identical + valid.
    Mesh m;
    m.verts = {Vertex{Vec3{0.0f, 0.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f}},
               Vertex{Vec3{1.0f, 0.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{1.0f, 0.0f}},
               Vertex{Vec3{0.0f, 1.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 1.0f}}};
    m.indices = {0, 1, 2};
    const VertexForest forest = buildVertexForest(m.verts, {}); // no collapses → zero splits
    REQUIRE(forest.splits.empty());
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 2.0f}, kVh, false);
    checkApply(runner.applyOnce(view, 1.0f, kVdpmCoarsenRatio * 1.0f), forest);
}

TEST_CASE("VDPM apply kernel == recorder: deep rank chain (descending-loop boundary)", "[.][gpu]")
{
    if (!ApplyCrossCheckRunner::supported())
    {
        return;
    }
    // A grid's collapse stream is a DEEP rank chain (high maxRank) — the descending close/coarsen
    // loops walk many ranks with a wgsync between each, and the last rank (0) is the boundary.
    const Mesh m = grid(33);
    const VertexForest forest = qemForest(m); // QuadricSimplifier over the grid
    ApplyCrossCheckRunner runner(m.verts, m.indices, forest);
    const VdpmViewParams view =
        scoreViewOf(Mat4::identity(), Vec3{16.0f, 16.0f, 30.0f}, kVh, false);
    checkApply(runner.applyOnce(view, 0.3f, kVdpmCoarsenRatio * 0.3f), forest);     // deep refine
    checkApply(runner.applyOnce(view, 1.0e9f, kVdpmCoarsenRatio * 1.0e9f), forest); // full coarsen
}

TEST_CASE("VDPM apply kernel: workgroup size 0 is rejected", "[.][gpu]")
{
    const Device device = Device::headlessCompute();
    // A zero local_size_x is an invalid pipeline — deviceSupported must say no, and the ctor must
    // reject it (via requireSupported) rather than build a broken pipeline.
    CHECK_FALSE(VdpmApplyKernel::deviceSupported(device, 0u));
    // Sanity: a valid size passes. Use 1 (any compute device runs 1-wide groups) — a
    // recorder-fallback device that can't run 256-wide would otherwise fail this positive check.
    CHECK(VdpmApplyKernel::deviceSupported(device, 1u));
    CHECK_THROWS_AS(VdpmApplyKernel(device, 0u), std::runtime_error);
}

TEST_CASE("VDPM apply kernel: split-bearing front with ZERO repair faces (recordFrame, both paths)",
          "[.][gpu]")
{
    // A REACHABLE runtime state: a split-bearing forest but an ALL-DEGENERATE index stream (every
    // face's three corners are vertex 0 → all weld together → 0 canonical repair faces, but raw
    // faceCount > 0). Repair early-outs (finestFaceCount == 0), so recordFrame's conditional
    // apply→emit barrier is the ONLY thing ordering the apply's `active` writes before emit's
    // ancestor read — exercise BOTH the persistent-kernel and recorder selections. Emission is
    // empty (every face degenerate) with a canonical zero-index indirect command; failFlags stay
    // clean.
    if (!VdpmApplyKernel::deviceSupported(Device::headlessCompute()) ||
        !VdpmRepairKernel::deviceSupported(Device::headlessCompute()))
    {
        return;
    }
    Device device = Device::headlessCompute();
    Resources resources(device);
    ComputePipeline scorePipeline(device, vdpmScorePipelineConfig());
    VdpmRefinePipelines refinePipelines(device);
    VdpmRepairPipelines repairPipelines(device);
    VdpmEmitPipelines emitPipelines(device);
    VdpmApplyKernel applyKernel(device);
    VdpmRepairKernel repairKernel(device);
    const vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});

    const Mesh m = uvSphere(14, 18);
    const VertexForest forest = qemForest(m);
    const std::vector<std::uint32_t> degenerate(m.indices.size(), 0u);
    const VdpmGpuMesh mesh = VdpmGpuMesh::build(resources, m.verts, degenerate, forest);
    REQUIRE(mesh.binding().splitCount > 0u);
    REQUIRE(mesh.binding().finestFaceCount == 0u);
    REQUIRE(mesh.binding().faceCount > 0u); // raw faces DO exist — emit runs over them
    VdpmGpuFront front = VdpmGpuFront::buildRuntime(resources, mesh);
    const std::uint32_t vertexCount = forest.vertexCount;
    const std::uint32_t splitCount = static_cast<std::uint32_t>(forest.splits.size());

    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.7f}, kVh, true);
    const VdpmRepairParams repairParams{}; // unused — repair early-outs at finestFaceCount == 0

    auto runFrame = [&](const VdpmApplyKernel* ak, const VdpmRepairKernel* rk)
    {
        auto cmds = device.device().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{.commandPool = *pool,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1});
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        front.recordFrame(*cmd, scorePipeline, refinePipelines, repairPipelines, emitPipelines,
                          resources, 0, view, repairParams, 1.0f, kVdpmCoarsenRatio * 1.0f,
                          kVdpmGpuRepairRoundBudget, ak, rk, nullptr);

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
            Resources::MappedBufferSet host =
                resources.createMappedReadbackBuffers(words * sizeof(std::uint32_t));
            cmd.copyBuffer(resources.vulkanBuffer(src), resources.vulkanBuffer(host.buffers[0]),
                           vk::BufferCopy{.size = words * sizeof(std::uint32_t)});
            return host;
        };
        const Resources::MappedBufferSet counters = copyBack(front.countersBuffer(0), 3);
        const Resources::MappedBufferSet indirect = copyBack(front.emittedIndirectBuffer(0), 5);
        const Resources::MappedBufferSet failFlags = copyBack(front.failFlagsBuffer(), 2);
        const Resources::MappedBufferSet active = copyBack(front.activeStateBuffer(), vertexCount);
        const Resources::MappedBufferSet refined = copyBack(front.refinedStateBuffer(), splitCount);
        const Resources::MappedBufferSet deps =
            copyBack(front.dependentsStateBuffer(), vertexCount);
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
        const vk::raii::Fence fence(device.device(), vk::FenceCreateInfo{});
        device.graphicsQueue().submit2(submit, *fence);
        (void)device.device().waitForFences(*fence, vk::True,
                                            std::numeric_limits<std::uint64_t>::max());
        auto read = [](const Resources::MappedBufferSet& set, std::uint32_t count)
        {
            std::vector<std::uint32_t> out(count);
            std::memcpy(out.data(), set.mapped[0].data(), count * sizeof(std::uint32_t));
            return out;
        };
        CHECK(read(counters, 3)[2] == 0u); // emission empty — every face is degenerate
        const std::vector<std::uint32_t> ind = read(indirect, 5);
        CHECK(ind[0] == 0u); // indexCount == 0 (canonical zero-index draw)
        CHECK(ind[1] == 1u); // instanceCount == 1 (still a valid indirect command)
        const std::vector<std::uint32_t> ff = read(failFlags, 2);
        CHECK(ff[0] == 0u); // no refine failure
        CHECK(ff[1] == 0u); // no dependents underflow
        CHECK_NOTHROW(validateFrontInvariants(forest, read(active, vertexCount),
                                              read(refined, splitCount), read(deps, vertexCount)));
    };

    runFrame(&applyKernel, &repairKernel); // both kernels
    runFrame(nullptr, nullptr);            // both recorders — the apply→emit barrier path
}

// Front-batching go/no-go benchmark (front-batching arc, Stage A). N fronts, each a
// single-workgroup apply/repair. SERIAL = N back-to-back per-front dispatches (each
// recordApplyKernel/recordRepairKernel carries its own leading barrier, so the N dispatches
// serialise — the current production shape). BATCHED = fill an N-job array + ONE dispatch(N), so
// the N fronts' workgroups run concurrently and fill the GPU. GPU-timestamped, median over repeats.
// If BATCHED is materially faster than SERIAL, front-batching is justified (adj 1 — a real
// GPU-concurrency measurement, not an estimate). Run with
// `./test_fire_engine "[BatchBench]"` and read the WARN'd medians.
TEST_CASE("VDPM front-batch benchmark: serial vs dispatch(N) apply + repair",
          "[.][gpu][BatchBench]")
{
    Device device = Device::headlessCompute(); // ONE device for both capability queries (adj 2)
    if (!VdpmApplyKernel::deviceSupported(device) || !VdpmRepairKernel::deviceSupported(device))
    {
        return;
    }
    const vk::PhysicalDeviceLimits& limits = device.physicalDevice().getProperties().limits;
    const auto qfp = device.physicalDevice().getQueueFamilyProperties();
    if (limits.timestampPeriod == 0.0f || qfp[device.graphicsFamily()].timestampValidBits == 0)
    {
        WARN("device has no compute-queue timestamp support — batch benchmark skipped");
        return;
    }
    Resources resources(device);
    ComputePipeline scorePipeline(device, vdpmScorePipelineConfig());
    VdpmApplyKernel applyKernel(device);
    VdpmRepairKernel repairKernel(device);
    const vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});
    const vk::raii::QueryPool queryPool(
        device.device(),
        vk::QueryPoolCreateInfo{.queryType = vk::QueryType::eTimestamp, .queryCount = 2});

    constexpr std::uint32_t kN = 13; // ~TransmissionTest front count
    const Mesh m = uvSphere(24, 32); // helmet-ish scale
    const VertexForest forest = qemForest(m);
    const VdpmGpuMesh mesh = VdpmGpuMesh::build(resources, m.verts, m.indices, forest);
    // TWO identical front sets: A driven SERIALLY, B BATCHED. State-fair (adj 1): serial and
    // batched never see each other's mutated output — each rep resets BOTH to identical roots, then
    // times serial(A) vs batched(B) on that SAME pre-state, asserting the sets stay identical
    // throughout.
    std::vector<VdpmGpuFront> setA, setB;
    setA.reserve(kN);
    setB.reserve(kN);
    for (std::uint32_t i = 0; i < kN; ++i)
    {
        setA.push_back(VdpmGpuFront::buildRuntime(resources, mesh));
        setB.push_back(VdpmGpuFront::buildRuntime(resources, mesh));
    }
    const Resources::MappedBufferSet applyJobs =
        resources.createMappedDeviceAddressBuffers(kN * sizeof(VdpmApplyJobGpu));
    const Resources::MappedBufferSet repairJobs =
        resources.createMappedDeviceAddressBuffers(kN * sizeof(VdpmRepairJobGpu));
    const std::uint32_t vertexCount = forest.vertexCount;

    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.7f}, kVh, true);
    const VdpmRepairParams params =
        repairParamsOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.7f}, 1024.0f, kVh, true);
    constexpr float kBudget = 0.5f;
    constexpr float kCoarsen = kVdpmCoarsenRatio * kBudget;

    auto computeBarrier = [](vk::CommandBuffer cmd)
    {
        const vk::MemoryBarrier2 mb{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask =
                vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
    };
    auto oneShot = [&]() -> vk::raii::CommandBuffer
    {
        auto cmds = device.device().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{.commandPool = *pool,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1});
        cmds[0].begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return std::move(cmds[0]);
    };
    auto submitWait = [&](vk::raii::CommandBuffer& cmd)
    {
        cmd.end();
        const vk::CommandBufferSubmitInfo ci{.commandBuffer = *cmd};
        const vk::SubmitInfo2 sub{.commandBufferInfoCount = 1, .pCommandBufferInfos = &ci};
        const vk::raii::Fence fence(device.device(), vk::FenceCreateInfo{});
        device.graphicsQueue().submit2(sub, *fence);
        (void)device.device().waitForFences(*fence, vk::True,
                                            std::numeric_limits<std::uint64_t>::max());
    };
    auto submitTimed = [&](auto&& record) -> double
    {
        vk::raii::CommandBuffer cmd = oneShot();
        cmd.resetQueryPool(*queryPool, 0, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *queryPool, 0);
        record(*cmd);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *queryPool, 1);
        submitWait(cmd);
        const auto [res, data] = queryPool.getResults<std::uint64_t>(
            0, 2, 2 * sizeof(std::uint64_t), sizeof(std::uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        return static_cast<double>(data[1] - data[0]) *
               static_cast<double>(limits.timestampPeriod) / 1.0e6;
    };

    // Coarsen a set fully to roots (apply with huge budgets — nothing refines, everything coarsens)
    // + score for the next real apply. Deterministic, so both sets land on identical roots +
    // scores.
    auto resetAndScore = [&](vk::CommandBuffer cmd, std::vector<VdpmGpuFront>& set)
    {
        for (VdpmGpuFront& f : set)
        {
            f.recordScore(cmd, scorePipeline, 0, view);
            f.recordApplyKernel(cmd, applyKernel, 0, 1.0e9f, 1.0e9f);
        }
    };
    // Concatenate every front's `active` (proves the two sets stay bit-identical — the fairness +
    // batched==serial guard).
    auto readActive = [&](std::vector<VdpmGpuFront>& set)
    {
        vk::raii::CommandBuffer cmd = oneShot();
        std::vector<Resources::MappedBufferSet> hosts;
        hosts.reserve(set.size());
        for (VdpmGpuFront& f : set)
        {
            Resources::MappedBufferSet h =
                resources.createMappedReadbackBuffers(vertexCount * sizeof(std::uint32_t));
            cmd.copyBuffer(resources.vulkanBuffer(f.activeStateBuffer()),
                           resources.vulkanBuffer(h.buffers[0]),
                           vk::BufferCopy{.size = vertexCount * sizeof(std::uint32_t)});
            hosts.push_back(h);
        }
        const vk::MemoryBarrier2 toHost{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
        submitWait(cmd);
        std::vector<std::uint32_t> out;
        out.reserve(set.size() * vertexCount);
        for (const Resources::MappedBufferSet& h : hosts)
        {
            const auto* p = reinterpret_cast<const std::uint32_t*>(h.mapped[0].data());
            out.insert(out.end(), p, p + vertexCount);
        }
        return out;
    };
    auto writeJobs = [&](const Resources::MappedBufferSet& dst, auto makeJob)
    {
        for (std::uint32_t i = 0; i < kN; ++i)
        {
            const auto job = makeJob(i);
            std::memcpy(dst.mapped[0].data() + i * sizeof(job), &job, sizeof(job));
        }
    };
    auto serialApply = [&](vk::CommandBuffer cmd, std::vector<VdpmGpuFront>& set)
    {
        for (VdpmGpuFront& f : set)
        {
            f.recordApplyKernel(cmd, applyKernel, 0, kBudget, kCoarsen);
        }
    };
    auto batchedApply = [&](vk::CommandBuffer cmd, std::vector<VdpmGpuFront>& set)
    {
        writeJobs(applyJobs,
                  [&](std::uint32_t i) { return set[i].prepareApplyJob(0, kBudget, kCoarsen); });
        computeBarrier(cmd);
        const VdpmApplyKernelPush push{
            .jobsAddress = resources.bufferAddress(applyJobs.buffers[0]), .jobCount = kN, .pad = 0};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, applyKernel.pipeline().pipeline());
        cmd.pushConstants<VdpmApplyKernelPush>(applyKernel.pipeline().pipelineLayout(),
                                               vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(kN, 1, 1);
    };
    auto serialRepair = [&](vk::CommandBuffer cmd, std::vector<VdpmGpuFront>& set)
    {
        for (VdpmGpuFront& f : set)
        {
            f.recordRepairKernel(cmd, repairKernel, 0, params, kVdpmGpuRepairRoundBudget);
        }
    };
    auto batchedRepair = [&](vk::CommandBuffer cmd, std::vector<VdpmGpuFront>& set)
    {
        writeJobs(repairJobs, [&](std::uint32_t i)
                  { return set[i].prepareRepairJob(0, params, kVdpmGpuRepairRoundBudget); });
        computeBarrier(cmd);
        const VdpmRepairKernelPush push{.jobsAddress =
                                            resources.bufferAddress(repairJobs.buffers[0]),
                                        .jobCount = kN,
                                        .pad = 0};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, repairKernel.pipeline().pipeline());
        cmd.pushConstants<VdpmRepairKernelPush>(repairKernel.pipeline().pipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(kN, 1, 1);
    };

    constexpr int kReps = 16;
    std::array<std::vector<double>, 4>
        s; // [serialApply, batchedApply, serialRepair, batchedRepair]
    for (int r = 0; r < kReps; ++r)
    {
        const bool serialFirst = (r % 2 == 0); // rotate order to blunt thermal/warm-up bias
        // Pre-apply: reset BOTH sets to identical roots (+ score), assert identical.
        {
            vk::raii::CommandBuffer cmd = oneShot();
            resetAndScore(*cmd, setA);
            resetAndScore(*cmd, setB);
            submitWait(cmd);
        }
        REQUIRE(readActive(setA) == readActive(setB)); // pre-apply state identical (both roots)
        auto apA = [&] { return submitTimed([&](vk::CommandBuffer c) { serialApply(c, setA); }); };
        auto apB = [&] { return submitTimed([&](vk::CommandBuffer c) { batchedApply(c, setB); }); };
        if (serialFirst)
        {
            s[0].push_back(apA());
            s[1].push_back(apB());
        }
        else
        {
            s[1].push_back(apB());
            s[0].push_back(apA());
        }
        REQUIRE(readActive(setA) ==
                readActive(setB)); // post-apply identical (fresh holes to repair)
        auto rpA = [&] { return submitTimed([&](vk::CommandBuffer c) { serialRepair(c, setA); }); };
        auto rpB = [&]
        { return submitTimed([&](vk::CommandBuffer c) { batchedRepair(c, setB); }); };
        if (serialFirst)
        {
            s[2].push_back(rpA());
            s[3].push_back(rpB());
        }
        else
        {
            s[3].push_back(rpB());
            s[2].push_back(rpA());
        }
    }
    WARN("apply  N=" << kN << ": serial " << medianMs(s[0]) << " ms vs batched " << medianMs(s[1])
                     << " ms GPU");
    WARN("repair N=" << kN << ": serial " << medianMs(s[2]) << " ms vs batched " << medianMs(s[3])
                     << " ms GPU");
    SUCCEED("batch benchmark reported serial vs dispatch(N) medians");
}

// ============================================================================================
// FUSION CEILING — is an apply+repair FUSED kernel worth PROTOTYPING? ([.][gpu][FusionCeiling])
// A hidden fenced measurement (evidence → docs/lod.md), NOT production plumbing. Fusion's only win
// is reclaiming the PER-WORKGROUP TAIL: across the GLOBAL apply→repair barrier, early-finishing
// apply workgroups idle until the SLOWEST clears before ANY repair workgroup starts. A
// command-level "apply-end→repair-start" gap CANNOT see that (a barrier is a dependency, not GPU
// work), so we bound it analytically instead:
//   (2) optimistic CEILING: saving ≲ min(Tapply, Trepair) + a small launch/sync allowance (fusion
//       also drops one dispatch launch + swaps the device-wide barrier for workgroup-local sync).
//       If that is a tiny fraction of the FULL lifecycle → skip fusion outright (no kernel).
//   (3) TAIL heuristic (whether a PROTOTYPE is warranted, NOT a runtime prediction — scheduling,
//   wave
//       capacity, and fused register pressure move the real number): separate-stage floor
//       max(Ai)+max(Ri) vs fused ideal max(Ai+Ri), over HETEROGENEOUS fronts (identical fronts have
//       ZERO tail by construction). A small delta ⇒ the heaviest front dominates both stages.
//   EMIT's lifecycle share is MEASURED here too (emit is 91/106 dispatches, but the TIME must be
//   measured — the shaders may be cheap despite the count).
// State-fair: each timed region runs on a freshly reset+scored pre-state; warm-up excluded;
// medians.
// ============================================================================================
TEST_CASE("VDPM apply+repair fusion ceiling", "[.][gpu][FusionCeiling]")
{
    Device device = Device::headlessCompute();
    if (!VdpmApplyKernel::deviceSupported(device) || !VdpmRepairKernel::deviceSupported(device))
    {
        return;
    }
    const vk::PhysicalDeviceLimits& limits = device.physicalDevice().getProperties().limits;
    const auto qfp = device.physicalDevice().getQueueFamilyProperties();
    if (limits.timestampPeriod == 0.0f || qfp[device.graphicsFamily()].timestampValidBits == 0)
    {
        WARN("no compute-queue timestamp support — fusion-ceiling benchmark skipped");
        return;
    }
    Resources resources(device);
    ComputePipeline scorePipeline(device, vdpmScorePipelineConfig());
    VdpmEmitPipelines emitPipelines(device);
    VdpmApplyKernel applyKernel(device);
    VdpmRepairKernel repairKernel(device);
    const vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});
    const vk::raii::QueryPool queryPool(
        device.device(),
        vk::QueryPoolCreateInfo{.queryType = vk::QueryType::eTimestamp, .queryCount = 2});

    // A PROCEDURAL PROXY multi-front set to exercise the fusion-tail methodology — NOT
    // TransmissionTest's meshes. It does NOT reproduce the real collapse forests, DAG ranks
    // (TransmissionTest's live max rank is ~45 — these proxies differ), per-instance transforms, or
    // materials: every job here shares ONE identity world / view / repair params. So any zero-tail
    // result below is a property of THESE benchmark jobs, not a guarantee about the renderer's
    // instances (identical geometry != identical VDPM work — scoring / active state / coverage /
    // convergence are instance- and view-dependent). The skip-fusion decision is a complexity/value
    // JUDGMENT, not a scene-specific measurement (a scene-specific verdict would need the real
    // meshes
    // + transforms + acceptance camera loaded, per docs/lod.md). Heterogeneous (varied cost)
    // because an identical set has zero tail trivially. Meshes kept alive: a front holds device
    // ADDRESSES.
    const std::array<Mesh, 5> specs{uvSphere(12, 16), uvSphere(22, 30), uvSphere(34, 46), grid(48),
                                    uvSphere(18, 24)};
    constexpr std::uint32_t kMax = 13;
    std::vector<VdpmGpuMesh> meshes;
    std::vector<VdpmGpuFront> fronts;
    meshes.reserve(kMax);
    fronts.reserve(kMax);
    for (std::uint32_t i = 0; i < kMax; ++i)
    {
        const Mesh& mm = specs[i % specs.size()];
        meshes.push_back(VdpmGpuMesh::build(resources, mm.verts, mm.indices, qemForest(mm)));
        fronts.push_back(VdpmGpuFront::buildRuntime(resources, meshes.back()));
    }
    const Resources::MappedBufferSet applyJobs =
        resources.createMappedDeviceAddressBuffers(kMax * sizeof(VdpmApplyJobGpu));
    const Resources::MappedBufferSet repairJobs =
        resources.createMappedDeviceAddressBuffers(kMax * sizeof(VdpmRepairJobGpu));
    const VdpmViewParams view = scoreViewOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.7f}, kVh, true);
    const VdpmRepairParams params =
        repairParamsOf(Mat4::identity(), Vec3{0.0f, 0.0f, 1.7f}, 1024.0f, kVh, true);
    constexpr float kBudget = 0.5f;
    constexpr float kCoarsen = kVdpmCoarsenRatio * kBudget;

    auto computeBarrier = [](vk::CommandBuffer cmd)
    {
        const vk::MemoryBarrier2 mb{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask =
                vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
    };
    auto oneShot = [&]() -> vk::raii::CommandBuffer
    {
        auto cmds = device.device().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{.commandPool = *pool,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1});
        cmds[0].begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return std::move(cmds[0]);
    };
    auto submitWait = [&](vk::raii::CommandBuffer& cmd)
    {
        cmd.end();
        const vk::CommandBufferSubmitInfo ci{.commandBuffer = *cmd};
        const vk::SubmitInfo2 sub{.commandBufferInfoCount = 1, .pCommandBufferInfos = &ci};
        const vk::raii::Fence fence(device.device(), vk::FenceCreateInfo{});
        device.graphicsQueue().submit2(sub, *fence);
        (void)device.device().waitForFences(*fence, vk::True,
                                            std::numeric_limits<std::uint64_t>::max());
    };
    auto submitTimed = [&](auto&& record) -> double
    {
        vk::raii::CommandBuffer cmd = oneShot();
        cmd.resetQueryPool(*queryPool, 0, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *queryPool, 0);
        record(*cmd);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *queryPool, 1);
        submitWait(cmd);
        const auto [res, data] = queryPool.getResults<std::uint64_t>(
            0, 2, 2 * sizeof(std::uint64_t), sizeof(std::uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        return static_cast<double>(data[1] - data[0]) *
               static_cast<double>(limits.timestampPeriod) / 1.0e6;
    };

    // Reset [0,n) to roots (+ score) so every timed region starts from an identical pre-state.
    auto resetAndScore = [&](std::uint32_t n)
    {
        vk::raii::CommandBuffer cmd = oneShot();
        for (std::uint32_t i = 0; i < n; ++i)
        {
            fronts[i].recordScore(*cmd, scorePipeline, 0, view);
            fronts[i].recordApplyKernel(*cmd, applyKernel, 0, 1.0e9f, 1.0e9f);
        }
        submitWait(cmd);
    };
    auto scoreAll = [&](vk::CommandBuffer cmd, std::uint32_t n)
    {
        for (std::uint32_t i = 0; i < n; ++i)
        {
            fronts[i].recordScore(cmd, scorePipeline, 0, view);
        }
    };
    // dispatch(n) of a batch, with a leading score/apply→kernel barrier (jobs filled from fronts).
    auto batchedApply = [&](vk::CommandBuffer cmd, std::uint32_t n)
    {
        for (std::uint32_t i = 0; i < n; ++i)
        {
            const VdpmApplyJobGpu j = fronts[i].prepareApplyJob(0, kBudget, kCoarsen);
            std::memcpy(applyJobs.mapped[0].data() + i * sizeof(j), &j, sizeof(j));
        }
        computeBarrier(cmd);
        applyKernel.recordDispatch(cmd, resources.bufferAddress(applyJobs.buffers[0]), n);
    };
    auto batchedRepair = [&](vk::CommandBuffer cmd, std::uint32_t n)
    {
        for (std::uint32_t i = 0; i < n; ++i)
        {
            const VdpmRepairJobGpu j =
                fronts[i].prepareRepairJob(0, params, kVdpmGpuRepairRoundBudget);
            std::memcpy(repairJobs.mapped[0].data() + i * sizeof(j), &j, sizeof(j));
        }
        computeBarrier(cmd);
        repairKernel.recordDispatch(cmd, resources.bufferAddress(repairJobs.buffers[0]), n);
    };
    auto emitAll = [&](vk::CommandBuffer cmd, std::uint32_t n)
    {
        for (std::uint32_t i = 0; i < n; ++i)
        {
            fronts[i].recordEmitFromFront(cmd, emitPipelines, resources, 0);
        }
    };

    constexpr int kWarmup = 2;
    constexpr int kReps = 7;
    auto med = [&](auto&& sample) -> double
    {
        std::vector<double> v;
        v.reserve(kReps);
        for (int r = 0; r < kWarmup + kReps; ++r)
        {
            const double t = sample();
            if (r >= kWarmup)
            {
                v.push_back(t);
            }
        }
        return medianMs(v);
    };

    // Batched stage timings over the first n fronts, each on a fresh reset pre-state.
    auto tApply = [&](std::uint32_t n)
    {
        return med(
            [&]
            {
                resetAndScore(n);
                return submitTimed([&](vk::CommandBuffer c) { batchedApply(c, n); });
            });
    };
    auto tRepair = [&](std::uint32_t n)
    {
        return med(
            [&]
            {
                resetAndScore(n);
                {
                    auto c = oneShot();
                    batchedApply(*c, n);
                    submitWait(c);
                } // to the applied state
                return submitTimed([&](vk::CommandBuffer c) { batchedRepair(c, n); });
            });
    };
    auto tSeq = [&](std::uint32_t n)
    {
        return med(
            [&]
            {
                resetAndScore(n);
                return submitTimed(
                    [&](vk::CommandBuffer c)
                    {
                        batchedApply(c, n);
                        batchedRepair(c, n);
                    });
            });
    };
    auto tEmit = [&](std::uint32_t n)
    {
        return med(
            [&]
            {
                resetAndScore(n);
                {
                    auto c = oneShot();
                    batchedApply(*c, n);
                    batchedRepair(*c, n);
                    submitWait(c);
                }
                return submitTimed(
                    [&](vk::CommandBuffer c)
                    {
                        // The repair→emit compute-stage boundary is part of emit's real ENTRY cost
                        // (production records it before the emit reads the repair-written active
                        // state) — time it inside the region.
                        computeBarrier(c);
                        emitAll(c, n);
                    });
            });
    };
    auto tFull = [&](std::uint32_t n)
    {
        return med(
            [&]
            {
                resetAndScore(n);
                return submitTimed(
                    [&](vk::CommandBuffer c)
                    {
                        scoreAll(c, n);
                        batchedApply(c, n);
                        batchedRepair(c, n);
                        computeBarrier(c); // repair → emit
                        emitAll(c, n);
                    });
            });
    };
    // Per-front A_i / R_i (single-workgroup dispatch(1)) for the tail heuristic. NOTE: scores are
    // state-INDEPENDENT (view + split geometry only), so the ONE reset score is valid for the timed
    // stage — no re-score (which, after an apply that read the score buffer, would be a WAR
    // rewrite), and the applies' own leading barriers order the roots→test-state dependency.
    auto aOf = [&](std::uint32_t i)
    {
        return med(
            [&]
            {
                {
                    auto c = oneShot();
                    fronts[i].recordScore(*c, scorePipeline, 0, view);
                    fronts[i].recordApplyKernel(*c, applyKernel, 0, 1.0e9f, 1.0e9f); // → roots
                    submitWait(c);
                }
                return submitTimed(
                    [&](vk::CommandBuffer c)
                    { fronts[i].recordApplyKernel(c, applyKernel, 0, kBudget, kCoarsen); });
            });
    };
    auto rOf = [&](std::uint32_t i)
    {
        return med(
            [&]
            {
                {
                    auto c = oneShot();
                    fronts[i].recordScore(*c, scorePipeline, 0, view);
                    fronts[i].recordApplyKernel(*c, applyKernel, 0, 1.0e9f, 1.0e9f);    // → roots
                    fronts[i].recordApplyKernel(*c, applyKernel, 0, kBudget, kCoarsen); // → test
                    submitWait(c);
                }
                return submitTimed(
                    [&](vk::CommandBuffer c)
                    {
                        fronts[i].recordRepairKernel(c, repairKernel, 0, params,
                                                     kVdpmGpuRepairRoundBudget);
                    });
            });
    };

    // (4) Scaling sweep.
    for (const std::uint32_t n : {1u, 2u, 4u, 8u, kMax})
    {
        const double full = tFull(n);
        const double ap = tApply(n);
        const double rp = tRepair(n);
        const double sq = tSeq(n);
        const double em = tEmit(n);
        WARN("N=" << n << ": full " << full << " | apply " << ap << " | repair " << rp << " | seq "
                  << sq << " | emit " << em << " ms GPU");
    }

    // (2) Ceiling + (3) tail + emit share, at the full N=kMax heterogeneous composition.
    const double full = tFull(kMax);
    const double ap = tApply(kMax);
    const double rp = tRepair(kMax);
    const double em = tEmit(kMax);
    double maxA = 0.0;
    double maxR = 0.0;
    double maxAplusR = 0.0;
    std::uint32_t argMaxA = 0;
    std::uint32_t argMaxR = 0;
    std::uint32_t argMaxAR = 0;
    for (std::uint32_t i = 0; i < kMax; ++i)
    {
        const double a = aOf(i);
        const double r = rOf(i);
        if (a > maxA)
        {
            maxA = a;
            argMaxA = i;
        }
        if (r > maxR)
        {
            maxR = r;
            argMaxR = i;
        }
        if (a + r > maxAplusR)
        {
            maxAplusR = a + r;
            argMaxAR = i;
        }
    }
    const double ceiling = std::min(ap, rp);                  // optimistic overlap recoverable
    const double separateFloor = maxA + maxR;                 // batched-separate ideal
    const double fusedIdealFloor = maxAplusR;                 // fused ideal
    const double tailDelta = separateFloor - fusedIdealFloor; // the reclaimable tail
    auto dims = [&](std::uint32_t i)
    {
        return "front[" + std::to_string(i) + "]{rank=" + std::to_string(fronts[i].rankCount()) +
               ",face=" + std::to_string(fronts[i].faceCount()) +
               ",finest=" + std::to_string(fronts[i].finestFaceCount()) + "}";
    };

    WARN("CEILING: min(Tapply,Trepair)=" << ceiling << " ms is " << (100.0 * ceiling / full)
                                         << "% of full lifecycle " << full
                                         << " ms (+ a small launch/sync allowance) — <~10% ⇒ skip");
    WARN("EMIT share: " << em << " ms = " << (100.0 * em / full) << "% of full lifecycle");
    WARN("TAIL: separate floor max(Ai)+max(Ri)="
         << separateFloor << " ms vs fused ideal max(Ai+Ri)=" << fusedIdealFloor
         << " ms → reclaimable " << tailDelta << " ms (" << (100.0 * tailDelta / full)
         << "% of full)");
    // Argmax evidence: a near-zero tail means the SAME front is the critical path in both stages
    // (argMaxA == argMaxR) — no cross-front tail on THIS proxy. The skip-fusion decision is a
    // complexity/value judgment (docs/lod.md), not a scene-specific verdict from these proxies.
    WARN("ARGMAX: maxA " << dims(argMaxA) << "=" << maxA << " ms | maxR " << dims(argMaxR) << "="
                         << maxR << " ms | max(A+R) " << dims(argMaxAR) << "=" << maxAplusR
                         << " ms  (same front dominates both ⇔ tail≈0)");
    SUCCEED("fusion-ceiling benchmark reported the ceiling + tail bounds");
}

// Evidence sweep (not a pass/fail gate): pick the kernel's production workgroup size. Run with
// `./test_fire_engine "[ApplySizeSweep]"` from the build dir and read the WARN'd medians.
TEST_CASE("VDPM apply kernel workgroup-size sweep (64/128/256)", "[.][gpu][ApplySizeSweep]")
{
    // All three candidates are constructed, so gate on the LARGEST (256) — 64-only hardware would
    // otherwise throw when building the 128/256 kernels.
    if (!VdpmApplyKernel::deviceSupported(Device::headlessCompute(), 256u))
    {
        return;
    }
    applySizeSweep(uvSphere(30, 40), "sphere(30,40)"); // curved / helmet-like ranks
    applySizeSweep(grid(49), "grid(49)");              // deeper rank chain
}
