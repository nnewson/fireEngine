#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
#include <fire_engine/render/vdpm_gpu.hpp>

#include <support/vdpm.hpp>

using namespace fire_engine;

// VDPM persistent repair KERNEL harness (perf arc, Stage 2). Tagged [.][gpu] — local only. Drives a
// buildRuntime front through recordScore + recordApplyScoredView (settle) → recordRepairKernel (the
// single-dispatch repair fixpoint under test), and reads back the settled state + roundHistory +
// repairControl.
// Each TEST_CASE exercises one control-flow shape the kernel introduces (the exhaustive GPU↔GPU /
// GPU↔CPU equivalence is Stage 3); together they make the kernel reviewable as working code.

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

VdpmRepairParams repairParamsOf(const Mat4& world, const Mat4& viewProj, const Vec3& cam, float vw,
                                float vh, bool cull)
{
    VdpmRepairParams p{};
    p.world = world;
    p.viewProj = viewProj;
    p.cameraPos[0] = cam.x();
    p.cameraPos[1] = cam.y();
    p.cameraPos[2] = cam.z();
    p.viewport[0] = vw;
    p.viewport[1] = vh;
    p.viewport[2] = cull ? 1.0f : 0.0f;
    return p;
}

// The score view params (cone predicate + channel scales) the runtime front's recordScore consumes
// — matches the repair params' world/cam/cull so the settle + repair see the same view.
VdpmViewParams scoreViewOf(const Mat4& world, const Vec3& cam, float vh, bool cull)
{
    return makeVdpmViewParams(world, cam, 1.0f, vh, 2.0f, cull, 1.0f, 1.0f, 1.0f);
}

// Read-back-front adapter satisfying the test::foldoverCount / coverageFailures Front interface.
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

struct Result
{
    std::vector<std::uint32_t> active;
    std::vector<std::uint32_t> refined;
    std::vector<std::uint32_t> dependents;
    std::array<std::uint32_t, 4> control{}; // anyMarked, ancestorFailure, fallbackFired, pad
    std::array<std::uint32_t, 2> failFlags{};
    std::vector<std::uint32_t> roundHistory;
};

class KernelRunner
{
public:
    KernelRunner(std::span<const Vertex> verts, std::span<const std::uint32_t> indices,
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
          front_(VdpmGpuFront::buildRuntime(resources_, mesh_)),
          vertexCount_(forest.vertexCount),
          splitCount_(static_cast<std::uint32_t>(forest.splits.size())),
          finestFaceCount_(mesh_.binding().finestFaceCount)
    {
    }
    KernelRunner(const KernelRunner&) = delete;
    KernelRunner& operator=(const KernelRunner&) = delete;
    KernelRunner(KernelRunner&&) = delete;
    KernelRunner& operator=(KernelRunner&&) = delete;
    ~KernelRunner() = default;

    [[nodiscard]] static bool supported()
    {
        return VdpmRepairKernel::deviceSupported(Device::headlessCompute());
    }

    // applyView (settle) into the persistent front, then the persistent-kernel repair for one frame
    // slot; read back the settled state + roundHistory + control.
    [[nodiscard]] Result settleAndRepair(const VdpmViewParams& scoreView, float budget,
                                         const VdpmRepairParams& params, std::uint32_t frameIndex,
                                         std::uint32_t roundBudget, bool dirtyHistoryFirst = false)
    {
        return run(
            [&](vk::CommandBuffer cmd)
            {
                front_.recordScore(cmd, scorePipeline_, frameIndex, scoreView);
                front_.recordApplyScoredView(cmd, refinePipelines_, resources_, budget,
                                             kVdpmCoarsenRatio * budget);
                if (dirtyHistoryFirst)
                {
                    // Pre-fill the ENTIRE history with 1s, so the kernel's entry clear (which must
                    // cover the full capacity, not just [0, roundBudget)) is directly testable: any
                    // slot the kernel doesn't write must read back 0, not this 1.
                    cmd.fillBuffer(resources_.vulkanBuffer(front_.roundHistoryBuffer()), 0,
                                   kVdpmGpuRepairRoundBudget * sizeof(std::uint32_t), 0xFFFFFFFFu);
                    const vk::MemoryBarrier2 clearToCompute{
                        .srcStageMask = vk::PipelineStageFlagBits2::eClear,
                        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                    };
                    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                            .pMemoryBarriers = &clearToCompute});
                }
                front_.recordRepairKernel(cmd, kernel_, frameIndex, params, roundBudget);
            });
    }

    // Two applyView+kernel repairs in ONE submit, into DISTINCT frame slots — proves the job/params
    // rings for slot N+1 don't clobber slot N in flight. Reads back the final front state.
    [[nodiscard]] Result twoSlots(const VdpmViewParams& scoreViewA, float budgetA,
                                  const VdpmRepairParams& paramsA, const VdpmViewParams& scoreViewB,
                                  float budgetB, const VdpmRepairParams& paramsB)
    {
        return run(
            [&](vk::CommandBuffer cmd)
            {
                front_.recordScore(cmd, scorePipeline_, /*frameIndex=*/0, scoreViewA);
                front_.recordApplyScoredView(cmd, refinePipelines_, resources_, budgetA,
                                             kVdpmCoarsenRatio * budgetA);
                front_.recordRepairKernel(cmd, kernel_, /*frameIndex=*/0, paramsA, 24);
                // The cross-frame LIFECYCLE barrier recordFrame owns: order the prior frame's
                // compute reads/writes (score buffer + persistent state) before this frame's score
                // WRITE + reuse — the score-buffer WAR B5a fixed. The two-slot rings + this barrier
                // are what let two frames run back-to-back in ONE submit.
                const vk::MemoryBarrier2 lifecycle{
                    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                                     vk::AccessFlagBits2::eShaderStorageWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                                     vk::AccessFlagBits2::eShaderStorageWrite,
                };
                cmd.pipelineBarrier2(
                    vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &lifecycle});
                front_.recordScore(cmd, scorePipeline_, /*frameIndex=*/1, scoreViewB);
                front_.recordApplyScoredView(cmd, refinePipelines_, resources_, budgetB,
                                             kVdpmCoarsenRatio * budgetB);
                front_.recordRepairKernel(cmd, kernel_, /*frameIndex=*/1, paramsB, 24);
            });
    }

    [[nodiscard]] VdpmGpuFront& front() noexcept
    {
        return front_;
    }
    [[nodiscard]] VdpmRepairKernel& kernel() noexcept
    {
        return kernel_;
    }
    [[nodiscard]] Resources& resources() noexcept
    {
        return resources_;
    }

private:
    template <class RecordFn>
    [[nodiscard]] Result run(RecordFn&& record)
    {
        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        record(*cmd);

        const vk::MemoryBarrier2 toTransfer{
            .srcStageMask =
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eClear,
            .srcAccessMask =
                vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toTransfer});

        auto copyBack = [&](BufferHandle src, std::uint32_t count)
        {
            const vk::DeviceSize size = static_cast<vk::DeviceSize>(count) * sizeof(std::uint32_t);
            Resources::MappedBufferSet host = resources_.createMappedReadbackBuffers(size);
            cmd.copyBuffer(resources_.vulkanBuffer(src), resources_.vulkanBuffer(host.buffers[0]),
                           vk::BufferCopy{.size = size});
            return host;
        };
        const Resources::MappedBufferSet a = copyBack(front_.activeStateBuffer(), vertexCount_);
        const Resources::MappedBufferSet r = copyBack(front_.refinedStateBuffer(), splitCount_);
        const Resources::MappedBufferSet d = copyBack(front_.dependentsStateBuffer(), vertexCount_);
        const Resources::MappedBufferSet ctrl = copyBack(front_.repairControlBuffer(), 4);
        const Resources::MappedBufferSet ff = copyBack(front_.failFlagsBuffer(), 2);
        const Resources::MappedBufferSet hist =
            copyBack(front_.roundHistoryBuffer(), kVdpmGpuRepairRoundBudget);

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
        Result st;
        st.active = read(a, vertexCount_);
        st.refined = read(r, splitCount_);
        st.dependents = read(d, vertexCount_);
        std::ranges::copy(read(ctrl, 4), st.control.begin());
        std::ranges::copy(read(ff, 2), st.failFlags.begin());
        st.roundHistory = read(hist, kVdpmGpuRepairRoundBudget);
        return st;
    }

    Device device_;
    Resources resources_;
    ComputePipeline scorePipeline_;
    VdpmRefinePipelines refinePipelines_;
    VdpmRepairKernel kernel_;
    vk::raii::CommandPool pool_;
    VdpmGpuMesh mesh_;
    VdpmGpuFront front_;
    std::uint32_t vertexCount_;
    std::uint32_t splitCount_;
    std::uint32_t finestFaceCount_;
};

// Common invariants a settled+repaired front must satisfy for view (world/cam/viewProj/cull).
void checkHoleFree(const VertexForest& forest, const Result& st, std::span<const Vertex> verts,
                   std::span<const std::uint32_t> indices, const Mat4& world, const Mat4& viewProj,
                   const Vec3& cam, float vw, float vh, bool cull)
{
    CHECK(st.failFlags[0] == 0u); // no refine failure
    CHECK(st.failFlags[1] == 0u); // no dependents underflow
    CHECK(st.control[1] == 0u);   // no ancestor failures
    CHECK_NOTHROW(validateFrontInvariants(forest, st.active, st.refined, st.dependents));
    const GpuFrontView view{forest, st.active};
    CHECK(test::foldoverCount(view, verts, indices, world) == 0);
    CHECK(test::coverageFailures(view, verts, indices, viewProj, cam, world, vw, vh, cull) == 0);
}

constexpr float kVw = 1024.0f;
constexpr float kVh = 768.0f;

} // namespace

TEST_CASE("VDPM repair kernel: an already-full-detail front converges on round 0 (no marks)",
          "[.][gpu]")
{
    if (!KernelRunner::supported())
    {
        return;
    }
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(16, 20);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const bool cull = false;
    const float budget = 1e-6f; // tiny ⇒ FULL detail ⇒ hole-free ⇒ first detect clean

    KernelRunner runner(m.verts, m.indices, forest);
    const Result st = runner.settleAndRepair(scoreViewOf(world, cam, kVh, cull), budget,
                                             repairParamsOf(world, viewProj, cam, kVw, kVh, cull),
                                             /*frameIndex=*/0, /*roundBudget=*/24);

    CHECK(st.control[0] == 0u);      // final anyMarked clean
    CHECK(st.control[2] == 0u);      // no fallback
    CHECK(st.roundHistory[0] == 0u); // round 0 already clean
    for (std::uint32_t h : st.roundHistory)
    {
        CHECK(h == 0u); // nothing marked, ever
    }
    checkHoleFree(forest, st, m.verts, m.indices, world, viewProj, cam, kVw, kVh, cull);
}

TEST_CASE(
    "VDPM repair kernel: a partial front repairs to marked-then-clean, hole-free, no fallback",
    "[.][gpu]")
{
    if (!KernelRunner::supported())
    {
        return;
    }
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(18, 24);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const bool cull = true;
    const float budget = 4.0f; // a selective front that genuinely leaves holes

    // Confirm the pre-repair front really needs repair (the CPU oracle on the applyView-settled
    // front).
    ParallelFront cpu = ParallelFront::build(m.verts, m.indices, collapses);
    {
        const VdpmViewParams view = scoreViewOf(world, cam, kVh, cull);
        std::vector<float> scalar(forest.splits.size());
        std::vector<std::uint8_t> backface(forest.splits.size());
        for (std::size_t s = 0; s < forest.splits.size(); ++s)
        {
            const VertexSplit& sp = forest.splits[s];
            const VdpmSplitScore sc = scoreVdpmSplit(view, sp, m.verts[sp.parent].position(),
                                                     m.verts[sp.child].position());
            scalar[s] = sc.score();
            backface[s] = sc.backface;
        }
        cpu.applyView(scalar, backface, budget, kVdpmCoarsenRatio * budget);
        REQUIRE((test::foldoverCount(cpu, m.verts, m.indices, world) +
                 test::coverageFailures(cpu, m.verts, m.indices, viewProj, cam, world, kVw, kVh,
                                        cull)) > 0);
    }

    KernelRunner runner(m.verts, m.indices, forest);
    const Result st = runner.settleAndRepair(scoreViewOf(world, cam, kVh, cull), budget,
                                             repairParamsOf(world, viewProj, cam, kVw, kVh, cull),
                                             /*frameIndex=*/0, /*roundBudget=*/24);

    CHECK(st.control[0] == 0u); // converged
    CHECK(st.control[2] == 0u); // no fallback
    // History is a marked-then-clean prefix: at least round 0 marked, then all zero.
    CHECK(st.roundHistory[0] == 1u);
    bool seenZero = false;
    for (std::uint32_t h : st.roundHistory)
    {
        if (h == 0u)
        {
            seenZero = true;
        }
        else
        {
            CHECK_FALSE(seenZero); // a 1 after a 0 would be a repair/sync bug
        }
    }
    checkHoleFree(forest, st, m.verts, m.indices, world, viewProj, cam, kVw, kVh, cull);
}

TEST_CASE("VDPM repair kernel: budget 0 forces the full-detail fallback, still hole-free",
          "[.][gpu]")
{
    if (!KernelRunner::supported())
    {
        return;
    }
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(16, 20);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const bool cull = true;
    const float budget =
        4.0f; // a partial front WITH holes; budget 0 ⇒ no bounded rounds ⇒ fallback

    KernelRunner runner(m.verts, m.indices, forest);
    const Result st = runner.settleAndRepair(scoreViewOf(world, cam, kVh, cull), budget,
                                             repairParamsOf(world, viewProj, cam, kVw, kVh, cull),
                                             /*frameIndex=*/0, /*roundBudget=*/0);

    CHECK(st.control[2] == 1u); // fallback FIRED
    for (std::uint32_t h : st.roundHistory)
    {
        CHECK(h == 0u); // no bounded rounds recorded any history
    }
    // Fallback drives to FULL detail: every split refined.
    CHECK(std::ranges::all_of(st.refined, [](std::uint32_t v) { return v != 0u; }));
    checkHoleFree(forest, st, m.verts, m.indices, world, viewProj, cam, kVw, kVh, cull);
}

TEST_CASE("VDPM repair kernel: coarsest front (deepest ancestor chains) resolves without failure",
          "[.][gpu]")
{
    if (!KernelRunner::supported())
    {
        return;
    }
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(16, 20);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const bool cull = true;
    const float budget =
        1e6f; // huge ⇒ COARSEST front ⇒ every non-root walks its full removal chain

    KernelRunner runner(m.verts, m.indices, forest);
    const Result st = runner.settleAndRepair(scoreViewOf(world, cam, kVh, cull), budget,
                                             repairParamsOf(world, viewProj, cam, kVw, kVh, cull),
                                             /*frameIndex=*/0, /*roundBudget=*/24);

    // The ancestor resolve walks the deepest chains (coarsest front) up to maxDepth TRANSITIONS and
    // must accept the exact-maxDepth root — zero ancestor failures.
    CHECK(st.control[1] == 0u);
    checkHoleFree(forest, st, m.verts, m.indices, world, viewProj, cam, kVw, kVh, cull);
}

TEST_CASE("VDPM repair kernel: an oversized round budget is rejected before recording", "[.][gpu]")
{
    if (!KernelRunner::supported())
    {
        return;
    }
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(12, 16);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    KernelRunner runner(m.verts, m.indices, forest);
    // makeRepairJob / recordRepairKernel must reject a budget past the allocated history capacity.
    REQUIRE_THROWS_AS(runner.front().makeRepairJob(0, kVdpmGpuRepairRoundBudget + 1),
                      std::logic_error);
}

TEST_CASE("VDPM repair kernel: two back-to-back frame slots do not clobber each other's rings",
          "[.][gpu]")
{
    if (!KernelRunner::supported())
    {
        return;
    }
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(16, 20);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    // MATERIALLY different views/params per slot, so a params-ring mix-up would corrupt the result
    // (not just pass on identical inputs). Slot A: camera on +z, cull ON, a partial budget. Slot B:
    // camera on -z (front-facing flips), cull OFF, full detail. Each repair must produce a valid
    // hole-free front FOR ITS OWN view.
    const Vec3 camA{0.0f, 0.0f, 2.5f};
    const Mat4 worldA = Mat4::identity();
    const Mat4 viewProjA = lookAtProj(camA);
    const bool cullA = true;
    const Vec3 camB{0.0f, 0.0f, -2.5f};
    Mat4 worldB = Mat4::identity();
    worldB[0, 3] = 0.3f; // a translated world too
    const Mat4 viewProjB = lookAtProj(camB);
    const bool cullB = false;

    KernelRunner runner(m.verts, m.indices, forest);
    const Result st = runner.twoSlots(scoreViewOf(worldA, camA, kVh, cullA), 4.0f,
                                      repairParamsOf(worldA, viewProjA, camA, kVw, kVh, cullA),
                                      scoreViewOf(worldB, camB, kVh, cullB), 1e-6f,
                                      repairParamsOf(worldB, viewProjB, camB, kVw, kVh, cullB));

    // The final front is frame B's — must be valid + hole-free for frame B's view.
    checkHoleFree(forest, st, m.verts, m.indices, worldB, viewProjB, camB, kVw, kVh, cullB);
}

TEST_CASE("VDPM repair kernel: a short budget clears the full history tail on reused storage",
          "[.][gpu]")
{
    if (!KernelRunner::supported())
    {
        return;
    }
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(18, 24);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const bool cull = true;
    const VdpmRepairParams params = repairParamsOf(world, viewProj, cam, kVw, kVh, cull);

    // Pre-dirty the WHOLE history buffer with 1s, then run a budget-2 repair. The kernel must clear
    // the full capacity at entry, so every slot from the budget (2) up reads back 0 — a
    // [0, roundBudget)-only clear would leave the pre-dirtied 1s in the tail.
    KernelRunner runner(m.verts, m.indices, forest);
    const Result st = runner.settleAndRepair(scoreViewOf(world, cam, kVh, cull), 4.0f, params,
                                             /*frameIndex=*/0, /*roundBudget=*/2,
                                             /*dirtyHistoryFirst=*/true);
    for (std::size_t k = 2; k < st.roundHistory.size(); ++k)
    {
        CHECK(st.roundHistory[k] == 0u); // full-capacity clear, not the pre-dirtied 0xFFFFFFFF
    }
    checkHoleFree(forest, st, m.verts, m.indices, world, viewProj, cam, kVw, kVh, cull);
}
