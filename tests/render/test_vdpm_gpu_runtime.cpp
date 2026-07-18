#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/mesh_topology.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/vdpm_gpu.hpp>

#include <support/vdpm.hpp>

using namespace fire_engine;

// GPU VDPM runtime harness (rendering-spine #3, GPU-driven-front Stage B5a). Tagged [.][gpu].
// Drives the COMBINED runtime front: buildRuntime + recordFrame (score → apply-scored → repair →
// emit) with the draw-consumed outputs ringed per frame slot. Split contract (once GPU scoring +
// GPU repair are chained, legitimate FP-boundary divergence can produce a different valid front):
// OFF-THRESHOLD fixtures require exact front + emitted-index equality; GENERAL fixtures require a
// valid, hole-free front, emitted ≤ finest, clean diagnostics, and a correct 5-word indirect
// command.

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

struct FrameReadback
{
    std::vector<std::uint32_t> active;
    std::vector<std::uint32_t> refined;
    std::vector<std::uint32_t> dependents;
    std::array<std::uint32_t, 2> failFlags{0, 0};
    std::array<std::uint32_t, 4> control{0, 0, 0, 0};
    std::vector<std::uint32_t> emittedIndices; // truncated to counters[2]
    std::array<std::uint32_t, 3> counters{0, 0, 0};
    DrawIndexedIndirectCommand indirect{};
};

// A headless device + a persistent runtime front (score + refine/coarsen + repair + emit).
class RuntimeRunner
{
public:
    RuntimeRunner(std::span<const Vertex> verts, std::span<const std::uint32_t> indices,
                  const VertexForest& forest)
        : device_(Device::headlessCompute()),
          resources_(device_),
          scorePipeline_(device_, vdpmScorePipelineConfig()),
          refinePipelines_(device_),
          repairPipelines_(device_),
          emitPipelines_(device_),
          pool_(
              device_.device(),
              vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        .queueFamilyIndex = device_.graphicsFamily()}),
          mesh_(VdpmGpuMesh::build(resources_, verts, indices, forest)),
          front_(VdpmGpuFront::buildRuntime(resources_, mesh_)),
          vertexCount_(forest.vertexCount),
          splitCount_(static_cast<std::uint32_t>(forest.splits.size())),
          faceCount_(mesh_.binding().faceCount)
    {
    }
    RuntimeRunner(const RuntimeRunner&) = delete;
    RuntimeRunner& operator=(const RuntimeRunner&) = delete;
    RuntimeRunner(RuntimeRunner&&) = delete;
    RuntimeRunner& operator=(RuntimeRunner&&) = delete;
    ~RuntimeRunner() = default;

    // Record ONE frame into slot `frameIndex`, submit, wait, read back that slot + the front state.
    struct FrameArgs
    {
        std::uint32_t frameIndex;
        VdpmViewParams scoreView;
        VdpmRepairParams repairParams;
        float budget;
        float coarsen;
        std::uint32_t roundBudget;
    };

    [[nodiscard]] FrameReadback frame(std::uint32_t frameIndex, const VdpmViewParams& scoreView,
                                      const VdpmRepairParams& repairParams, float budget,
                                      float coarsen, std::uint32_t roundBudget)
    {
        std::array<FrameArgs, 1> args{
            FrameArgs{frameIndex, scoreView, repairParams, budget, coarsen, roundBudget}};
        std::array<FrameReadback, 1> out;
        record(args, out);
        return out[0];
    }

    // Record several frames back-to-back in ONE submit (no CPU wait between), reading each slot
    // back.
    template <std::size_t N>
    void record(const std::array<FrameArgs, N>& frames, std::array<FrameReadback, N>& out)
    {
        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        for (const FrameArgs& f : frames)
        {
            front_.recordFrame(*cmd, scorePipeline_, refinePipelines_, repairPipelines_,
                               emitPipelines_, resources_, f.frameIndex, f.scoreView,
                               f.repairParams, f.budget, f.coarsen, f.roundBudget);
        }

        // Compute-write → transfer-read; counters/state are compute- and clear-written.
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

        struct Copy
        {
            Resources::MappedBufferSet host;
            std::uint32_t words;
        };
        std::vector<Copy> copies;
        auto add = [&](BufferHandle src, std::uint32_t words) -> Copy&
        {
            Copy c;
            c.words = words;
            if (words > 0)
            {
                const vk::DeviceSize size =
                    static_cast<vk::DeviceSize>(words) * sizeof(std::uint32_t);
                c.host = resources_.createMappedReadbackBuffers(size);
                cmd.copyBuffer(resources_.vulkanBuffer(src),
                               resources_.vulkanBuffer(c.host.buffers[0]),
                               vk::BufferCopy{.size = size});
            }
            copies.push_back(c);
            return copies.back();
        };
        // Per-frame ring outputs, then the shared persistent state (read once — it reflects the
        // last frame's settled front).
        std::vector<std::size_t> emittedIdx, countersIdx, indirectIdx;
        for (const FrameArgs& f : frames)
        {
            emittedIdx.push_back(copies.size());
            add(front_.emittedIndicesBuffer(f.frameIndex), faceCount_ * 3);
            countersIdx.push_back(copies.size());
            add(front_.countersBuffer(f.frameIndex), 3);
            indirectIdx.push_back(copies.size());
            add(front_.emittedIndirectBuffer(f.frameIndex),
                sizeof(DrawIndexedIndirectCommand) / sizeof(std::uint32_t));
        }
        const std::size_t activeIdx = copies.size();
        add(front_.activeStateBuffer(), vertexCount_);
        const std::size_t refinedIdx = copies.size();
        add(front_.refinedStateBuffer(), splitCount_);
        const std::size_t depIdx = copies.size();
        add(front_.dependentsStateBuffer(), vertexCount_);
        const std::size_t flagsIdx = copies.size();
        add(front_.failFlagsBuffer(), 2);
        const std::size_t ctrlIdx = copies.size();
        add(front_.repairControlBuffer(), 4);

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

        auto readVec = [&](std::size_t ci)
        {
            std::vector<std::uint32_t> v(copies[ci].words);
            if (!v.empty())
            {
                std::memcpy(v.data(), copies[ci].host.mapped[0].data(),
                            v.size() * sizeof(std::uint32_t));
            }
            return v;
        };
        const std::vector<std::uint32_t> active = readVec(activeIdx);
        const std::vector<std::uint32_t> refined = readVec(refinedIdx);
        const std::vector<std::uint32_t> dependents = readVec(depIdx);
        const std::vector<std::uint32_t> flags = readVec(flagsIdx);
        const std::vector<std::uint32_t> ctrl = readVec(ctrlIdx);
        for (std::size_t i = 0; i < frames.size(); ++i)
        {
            FrameReadback& rb = out[i];
            rb.active = active;
            rb.refined = refined;
            rb.dependents = dependents;
            std::ranges::copy(flags, rb.failFlags.begin());
            std::ranges::copy(ctrl, rb.control.begin());
            const std::vector<std::uint32_t> counters = readVec(countersIdx[i]);
            std::ranges::copy(counters, rb.counters.begin());
            const std::vector<std::uint32_t> ind = readVec(indirectIdx[i]);
            rb.indirect.indexCount = ind[0];
            rb.indirect.instanceCount = ind[1];
            rb.indirect.firstIndex = ind[2];
            rb.indirect.vertexOffset = static_cast<std::int32_t>(ind[3]);
            rb.indirect.firstInstance = ind[4];
            const std::vector<std::uint32_t> allIdx = readVec(emittedIdx[i]);
            rb.emittedIndices.assign(allIdx.begin(), allIdx.begin() + rb.counters[2]);
        }
    }

private:
    Device device_;
    Resources resources_;
    ComputePipeline scorePipeline_;
    VdpmRefinePipelines refinePipelines_;
    VdpmRepairPipelines repairPipelines_;
    VdpmEmitPipelines emitPipelines_;
    vk::raii::CommandPool pool_;
    VdpmGpuMesh mesh_;
    VdpmGpuFront front_;
    std::uint32_t vertexCount_;
    std::uint32_t splitCount_;
    std::uint32_t faceCount_;
};

// The full CPU lifecycle (scoreVdpmSplit → applyView → repairFront → emitActiveIndices) — the
// oracle.
std::vector<std::uint32_t> cpuLifecycle(ParallelFront& front, std::span<const Vertex> verts,
                                        std::span<const std::uint32_t> indices,
                                        const VertexForest& forest, const Mat4& world,
                                        const Vec3& cam, const Mat4& viewProj, float vw, float vh,
                                        bool cull, float budget, float coarsen)
{
    const VdpmViewParams view =
        makeVdpmViewParams(world, cam, 1.0f, vh, 2.0f, cull, 1.0f, 0.5f, 0.5f);
    const auto n = static_cast<std::uint32_t>(forest.splits.size());
    std::vector<float> scalar(n);
    std::vector<std::uint8_t> backface(n);
    for (std::uint32_t s = 0; s < n; ++s)
    {
        const VertexSplit& sp = forest.splits[s];
        const VdpmSplitScore sc =
            scoreVdpmSplit(view, sp, verts[sp.parent].position(), verts[sp.child].position());
        scalar[s] = sc.score();
        backface[s] = sc.backface;
    }
    front.applyView(scalar, backface, budget, coarsen);
    front.repairFront(verts, world, cam, viewProj, vw, vh, cull);
    return front.emitActiveIndices(verts, indices);
}

// The score view params matching the CPU (same makeVdpmViewParams args).
VdpmViewParams scoreViewOf(const Mat4& world, const Vec3& cam, float vh, bool cull)
{
    return makeVdpmViewParams(world, cam, 1.0f, vh, 2.0f, cull, 1.0f, 0.5f, 0.5f);
}

} // namespace

TEST_CASE("VDPM GPU runtime: full lifecycle produces a valid hole-free front + indirect command",
          "[.][gpu]")
{
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(18, 24);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const float vw = 1024.0f;
    const float vh = 768.0f;
    const bool cull = true;

    RuntimeRunner runner(m.verts, m.indices, forest);
    const FrameReadback rb =
        runner.frame(0, scoreViewOf(world, cam, vh, cull),
                     repairParamsOf(world, viewProj, cam, vw, vh, cull), 1.0f, 0.6f, 24);

    // GENERAL contract: clean diagnostics, valid + hole-free front, emitted ≤ finest, indirect
    // valid.
    CHECK(rb.failFlags[0] == 0u);
    CHECK(rb.failFlags[1] == 0u);
    CHECK(rb.control[1] == 0u); // ancestor failures
    CHECK_NOTHROW(validateFrontInvariants(forest, rb.active, rb.refined, rb.dependents));

    const GpuFrontView view{forest, rb.active};
    CHECK(test::foldoverCount(view, m.verts, m.indices, world) == 0);
    CHECK(test::coverageFailures(view, m.verts, m.indices, viewProj, cam, world, vw, vh, cull) ==
          0);

    const std::size_t finestTris =
        ParallelFront::build(m.verts, m.indices, collapses).finestFaces().size();
    CHECK(rb.emittedIndices.size() <= finestTris * 3);
    CHECK(rb.emittedIndices.size() % 3 == 0);

    // The 5-word indirect command: indexCount == counters[2] == emitted size; rest fixed.
    CHECK(rb.indirect.indexCount == rb.counters[2]);
    CHECK(rb.indirect.indexCount == rb.emittedIndices.size());
    CHECK(rb.indirect.instanceCount == 1u);
    CHECK(rb.indirect.firstIndex == 0u);
    CHECK(rb.indirect.vertexOffset == 0);
    CHECK(rb.indirect.firstInstance == 0u);
}

TEST_CASE("VDPM GPU runtime: OFF-THRESHOLD (full detail) matches the CPU front + emit exactly",
          "[.][gpu]")
{
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(16, 20);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const float vw = 1024.0f;
    const float vh = 768.0f;
    // Cull OFF so no split is zeroed for back-facing — with a budget FAR below every channel score,
    // EVERY split refines to full detail (unambiguously off every threshold), and full detail is
    // hole-free so the repair is a clean no-op. The GPU + CPU fronts + emits must then be EXACT
    // (integer scheduling + a no-op repair leave no room for FP-boundary divergence).
    const bool cull = false;
    const float budget = 1e-6f;

    RuntimeRunner runner(m.verts, m.indices, forest);
    const FrameReadback rb =
        runner.frame(0, scoreViewOf(world, cam, vh, cull),
                     repairParamsOf(world, viewProj, cam, vw, vh, cull), budget, budget * 0.6f, 24);

    ParallelFront cpu = ParallelFront::build(m.verts, m.indices, collapses);
    const std::vector<std::uint32_t> cpuEmit = cpuLifecycle(
        cpu, m.verts, m.indices, forest, world, cam, viewProj, vw, vh, cull, budget, budget * 0.6f);

    // Off every threshold ⇒ GPU front == CPU front EXACTLY (active + refined), + identical emit. (A
    // handful of near-planar collapses have zero error and legitimately never refine — GPU and CPU
    // agree on those too, so the fronts still match element-for-element.)
    std::size_t stateMismatch = 0;
    for (std::uint32_t v = 0; v < forest.vertexCount; ++v)
    {
        stateMismatch += ((rb.active[v] != 0u) != cpu.active(v)) ? 1 : 0;
    }
    for (std::uint32_t s = 0; s < forest.splits.size(); ++s)
    {
        stateMismatch += ((rb.refined[s] != 0u) != cpu.refined(s)) ? 1 : 0;
    }
    CHECK(stateMismatch == 0);
    CHECK(rb.control[2] == 0u); // no full-detail fallback needed
    CHECK(rb.emittedIndices == cpuEmit);
    CHECK(rb.indirect.indexCount == cpuEmit.size());
}

TEST_CASE("VDPM GPU runtime: frame-ring keeps distinct outputs across two back-to-back frames",
          "[.][gpu]")
{
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(16, 20);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);

    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const float vw = 1024.0f;
    const float vh = 768.0f;
    const bool cull = true;
    const VdpmViewParams sv = scoreViewOf(world, cam, vh, cull);
    const VdpmRepairParams rp = repairParamsOf(world, viewProj, cam, vw, vh, cull);

    RuntimeRunner runner(m.verts, m.indices, forest);
    // Frame 0 (slot 0): a tiny budget → full detail. Frame 1 (slot 1): a huge budget → coarsest.
    // Recorded back-to-back in ONE submit, no CPU wait. If the outputs weren't ringed, frame 1
    // would overwrite slot 0 and both would show the coarse result.
    std::array<RuntimeRunner::FrameArgs, 2> args{
        RuntimeRunner::FrameArgs{0, sv, rp, 1e-6f, 6e-7f, 24},
        RuntimeRunner::FrameArgs{1, sv, rp, 1e6f, 6e5f, 24}};
    std::array<FrameReadback, 2> rb;
    runner.record(args, rb);

    // Slot 0 kept the full-detail emit; slot 1 is the (much smaller) coarse emit.
    CHECK(rb[0].counters[2] > rb[1].counters[2]);
    CHECK(rb[0].indirect.indexCount == rb[0].counters[2]);
    CHECK(rb[1].indirect.indexCount == rb[1].counters[2]);

    // Slot 0 == the CPU full-detail emit (the persistent front was full detail after frame 0).
    ParallelFront cpu = ParallelFront::build(m.verts, m.indices, collapses);
    const std::vector<std::uint32_t> cpuFull = cpuLifecycle(
        cpu, m.verts, m.indices, forest, world, cam, viewProj, vw, vh, cull, 1e-6f, 6e-7f);
    CHECK(rb[0].emittedIndices == cpuFull);
}

TEST_CASE("VDPM GPU runtime: a zero-split mesh records two frames without racing the emit scratch",
          "[.][gpu]")
{
    // A single triangle can't be simplified — zero splits, one face. score/apply/repair all
    // early-out, so ONLY the lifecycle-boundary barrier at recordFrame's start orders the two
    // back-to-back emits' shared internal scratch. Validation layers (on in Dev) would flag a
    // missing dependency.
    Mesh m;
    m.verts = {Vertex{Vec3{-0.5f, -0.5f, 1.0f}, Colour3{}, Vec3{0, 0, 1}, Vec2{0, 0}},
               Vertex{Vec3{0.5f, -0.5f, 1.0f}, Colour3{}, Vec3{0, 0, 1}, Vec2{1, 0}},
               Vertex{Vec3{0.0f, 0.5f, 1.0f}, Colour3{}, Vec3{0, 0, 1}, Vec2{0, 1}}};
    m.indices = {0, 1, 2};
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);
    REQUIRE(forest.splits.empty());

    const Vec3 cam{0.0f, 0.0f, 3.0f};
    const VdpmViewParams sv = scoreViewOf(Mat4::identity(), cam, 768.0f, false);
    const VdpmRepairParams rp =
        repairParamsOf(Mat4::identity(), lookAtProj(cam), cam, 1024.0f, 768.0f, false);

    RuntimeRunner runner(m.verts, m.indices, forest);
    std::array<RuntimeRunner::FrameArgs, 2> args{
        RuntimeRunner::FrameArgs{0, sv, rp, 1.0f, 0.6f, 4},
        RuntimeRunner::FrameArgs{1, sv, rp, 1.0f, 0.6f, 4}};
    std::array<FrameReadback, 2> rb;
    runner.record(args, rb);

    // Both slots emit the single (front-facing, all-roots-active) triangle — 3 indices — cleanly.
    for (const FrameReadback& f : rb)
    {
        CHECK(f.control[1] == 0u); // no ancestor failures
        CHECK(f.counters[2] == 3u);
        CHECK(f.indirect.indexCount == 3u);
        CHECK(f.indirect.instanceCount == 1u);
        CHECK(f.emittedIndices == std::vector<std::uint32_t>{0u, 1u, 2u});
    }
}
