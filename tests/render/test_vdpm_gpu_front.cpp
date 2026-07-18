#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/vdpm_gpu.hpp>

using namespace fire_engine;

// GPU VDPM refine/coarsen harness (rendering-spine #3, GPU-driven-front Stage B3). Tagged [.][gpu]
// so normal CTest (+ the no-ICD CI runners) SKIP it. Runs the GPU refine/coarsen passes against the
// CPU `ParallelFront` model with an IDENTICAL score/backface sequence, and asserts the persistent
// state (active/refined/dependents) is ELEMENT-EXACT every frame + the shared
// `validateFrontInvariants` passes on the read-back state + the invariant-failure flags stay 0.
// Isolation harness: the scores are uploaded (not GPU-computed), so scheduling is integer-exact
// once the decisions are made.

namespace
{

struct Mesh
{
    std::vector<Vertex> verts;
    std::vector<std::uint32_t> indices;
};

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

VertexForest forestOf(const Mesh& m)
{
    const QuadricSimplifier simp;
    return buildVertexForest(m.verts, simp.collapseSequence(m.verts, m.indices));
}

// A diamond forest (same-rank atomic sharing): B at rank 0, L/R at rank 1 sharing B, T at rank 2.
VertexForest diamondForest()
{
    VertexForest f;
    f.vertexCount = 5;
    f.removingSplit = {kNoSplit, 0, 1, 2, 3};
    auto mk = [](std::uint32_t child, std::uint32_t parent, std::uint32_t vl)
    {
        VertexSplit s;
        s.child = child;
        s.parent = parent;
        s.vl = vl;
        s.vr = kInvalidVertex;
        return s;
    };
    f.splits = {mk(1, 0, 0), mk(2, 1, 0), mk(3, 1, 0), mk(4, 2, 3)};
    return f;
}

// A linear removal chain of `n` splits (rank 0..n-1) — the deepest DAG shape: split i removes
// vertex i+1 with parent = vertex i. Fully refines then fully coarsens in one reverse sweep.
VertexForest chainForest(std::uint32_t n)
{
    VertexForest f;
    f.vertexCount = n + 1;
    f.removingSplit.assign(n + 1, kNoSplit);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        f.removingSplit[i + 1] = i;
        VertexSplit s;
        s.child = i + 1;
        s.parent = i;
        s.vl = i;
        s.vr = kInvalidVertex;
        f.splits.push_back(s);
    }
    return f;
}

// Build one VdpmScoreOut per split from a scalar decision score + backface flag, placing the scalar
// in `channel` (0=geometry,1=uv,2=normal,3=tangent) so the GPU's max-reduction is exercised on each
// channel. The CPU model is driven with the SAME scalar, so the decision is identical either way.
std::vector<VdpmScoreOut> makeScores(std::span<const float> scalar,
                                     std::span<const std::uint8_t> backface, int channel = 0)
{
    std::vector<VdpmScoreOut> out(scalar.size());
    for (std::size_t i = 0; i < scalar.size(); ++i)
    {
        VdpmScoreOut o{};
        (channel == 0   ? o.geometry
         : channel == 1 ? o.uv
         : channel == 2 ? o.normal
                        : o.tangent) = scalar[i];
        o.backface = backface[i];
        out[i] = o;
    }
    return out;
}

// Persistent state read back from the GPU front.
struct GpuState
{
    std::vector<std::uint32_t> active;
    std::vector<std::uint32_t> refined;
    std::vector<std::uint32_t> dependents;
    std::array<std::uint32_t, 2> failFlags{0, 0};
};

// A headless compute device + a persistent GPU front built from a forest, driven frame by frame.
// Non-movable; build one per case.
class FrontRunner
{
public:
    FrontRunner(std::span<const Vertex> verts, const VertexForest& forest)
        : device_(Device::headlessCompute()),
          resources_(device_),
          pipelines_(device_),
          pool_(
              device_.device(),
              vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        .queueFamilyIndex = device_.graphicsFamily()}),
          mesh_(VdpmGpuMesh::build(resources_, verts, forest)),
          front_(VdpmGpuFront::buildWithFront(resources_, mesh_)),
          vertexCount_(forest.vertexCount),
          splitCount_(static_cast<std::uint32_t>(forest.splits.size()))
    {
    }
    FrontRunner(const FrontRunner&) = delete;
    FrontRunner& operator=(const FrontRunner&) = delete;
    FrontRunner(FrontRunner&&) = delete;
    FrontRunner& operator=(FrontRunner&&) = delete;
    ~FrontRunner() = default;

    // Record one applyView, submit, wait, read back the persistent state.
    [[nodiscard]] GpuState applyView(std::span<const VdpmScoreOut> scores, float budget,
                                     float coarsen)
    {
        return run(
            [&](vk::CommandBuffer cmd)
            { front_.recordApplyView(cmd, pipelines_, resources_, scores, budget, coarsen); });
    }

    // Record TWO applyView calls back-to-back in ONE submit (no CPU wait between) — the inter-call
    // barrier contract. Identical scores both cycles (a single score buffer; per-frame buffers are
    // a B5 concern), so the second cycle is the settled front re-applied.
    [[nodiscard]] GpuState applyViewTwice(std::span<const VdpmScoreOut> scores, float budget,
                                          float coarsen)
    {
        return run(
            [&](vk::CommandBuffer cmd)
            {
                front_.recordApplyView(cmd, pipelines_, resources_, scores, budget, coarsen);
                front_.recordApplyView(cmd, pipelines_, resources_, scores, budget, coarsen);
            });
    }

private:
    template <class RecordFn>
    [[nodiscard]] GpuState run(RecordFn&& record)
    {
        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        record(*cmd);

        // Producer → transfer-read for all state buffers. active/refined/dependents are compute-
        // written; failFlags is normally last written by recordApplyView's fillBuffer CLEAR (eClear
        // / eTransferWrite) when no shader reports a failure — so the source scope must name BOTH
        // producers, not compute alone, or the clear-written flags race the copy.
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

        // A zero-count state buffer (e.g. `refined` for a split-free forest) is null — skip its
        // copy (VMA rejects a zero-byte allocation) and leave its read-back vector empty.
        auto copyBack = [&](BufferHandle src, std::uint32_t count)
        {
            Resources::MappedBufferSet host;
            if (count == 0)
            {
                return host;
            }
            const vk::DeviceSize size = static_cast<vk::DeviceSize>(count) * sizeof(std::uint32_t);
            host = resources_.createMappedReadbackBuffers(size);
            cmd.copyBuffer(resources_.vulkanBuffer(src), resources_.vulkanBuffer(host.buffers[0]),
                           vk::BufferCopy{.size = size});
            return host;
        };
        const Resources::MappedBufferSet activeHost =
            copyBack(front_.activeStateBuffer(), vertexCount_);
        const Resources::MappedBufferSet refinedHost =
            copyBack(front_.refinedStateBuffer(), splitCount_);
        const Resources::MappedBufferSet depHost =
            copyBack(front_.dependentsStateBuffer(), vertexCount_);
        const Resources::MappedBufferSet flagHost = copyBack(front_.failFlagsBuffer(), 2);

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

        GpuState st;
        st.active.resize(vertexCount_);
        st.refined.resize(splitCount_);
        st.dependents.resize(vertexCount_);
        if (vertexCount_ > 0)
        {
            std::memcpy(st.active.data(), activeHost.mapped[0].data(),
                        vertexCount_ * sizeof(std::uint32_t));
            std::memcpy(st.dependents.data(), depHost.mapped[0].data(),
                        vertexCount_ * sizeof(std::uint32_t));
        }
        if (splitCount_ > 0)
        {
            std::memcpy(st.refined.data(), refinedHost.mapped[0].data(),
                        splitCount_ * sizeof(std::uint32_t));
        }
        std::memcpy(st.failFlags.data(), flagHost.mapped[0].data(), 2 * sizeof(std::uint32_t));
        return st;
    }

    Device device_;
    Resources resources_;
    VdpmRefinePipelines pipelines_;
    vk::raii::CommandPool pool_;
    VdpmGpuMesh mesh_;
    VdpmGpuFront front_;
    std::uint32_t vertexCount_;
    std::uint32_t splitCount_;
};

// Cross-check a GPU state against the CPU model: element-exact active/refined/dependents, the
// shared first-principles validator on the read-back state, and zero invariant-failure flags.
void expectMatchesCpu(const GpuState& gpu, const ParallelFront& cpu, const VertexForest& forest)
{
    CHECK(gpu.failFlags[0] == 0u); // refine failure
    CHECK(gpu.failFlags[1] == 0u); // dependents underflow

    // The read-back state is internally consistent (dependents reconstructs from refined, etc.).
    CHECK_NOTHROW(validateFrontInvariants(forest, gpu.active, gpu.refined, gpu.dependents));

    std::size_t mismatches = 0;
    for (std::uint32_t v = 0; v < forest.vertexCount; ++v)
    {
        mismatches += (gpu.active[v] != (cpu.active(v) ? 1u : 0u)) ? 1 : 0;
        mismatches += (gpu.dependents[v] != cpu.dependents(v)) ? 1 : 0;
    }
    for (std::uint32_t s = 0; s < forest.splits.size(); ++s)
    {
        mismatches += (gpu.refined[s] != (cpu.refined(s) ? 1u : 0u)) ? 1 : 0;
    }
    CHECK(mismatches == 0);
}

} // namespace

TEST_CASE("VDPM GPU refine/coarsen: a moving-view score sequence matches the CPU model", "[.][gpu]")
{
    for (const Mesh& m : {grid(17), uvSphere(20, 28)})
    {
        const VertexForest forest = forestOf(m);
        const auto n = static_cast<std::uint32_t>(forest.splits.size());
        REQUIRE(n > 1);

        FrontRunner runner(m.verts, forest);
        ParallelFront cpu = ParallelFront::build(forest);

        std::mt19937 rng(0xB3F00D);
        std::uniform_real_distribution<float> dist(0.0f, 2.0f); // straddles the budget
        constexpr float budget = 1.0f;
        constexpr float coarsen = 0.6f;
        for (int frame = 0; frame < 8; ++frame)
        {
            std::vector<float> scalar(n);
            std::vector<std::uint8_t> backface(n, 0);
            for (std::uint32_t i = 0; i < n; ++i)
            {
                scalar[i] = dist(rng);
                backface[i] = (rng() & 7u) == 0u ? 1u : 0u; // ~1/8 back-facing
            }
            const std::vector<VdpmScoreOut> scores = makeScores(scalar, backface);
            const GpuState gpu = runner.applyView(scores, budget, coarsen);
            cpu.applyView(scalar, backface, budget, coarsen);
            CAPTURE(frame);
            expectMatchesCpu(gpu, cpu, forest);
        }
    }
}

TEST_CASE("VDPM GPU refine/coarsen: each score channel drives the decision (max reduction)",
          "[.][gpu]")
{
    const VertexForest forest = forestOf(grid(9));
    const auto n = static_cast<std::uint32_t>(forest.splits.size());
    const std::vector<std::uint8_t> face(n, 0);
    const std::vector<float> hot(n, 9.0f); // over budget in the tested channel only

    // For each channel in turn, the scalar lives ONLY in that channel (the others are 0). The CPU
    // is driven with the scalar directly. If the GPU max-reduction ignored the channel, its score
    // would be 0 (no refine) and the front would diverge from the CPU's full refine — so matching
    // proves every channel feeds the decision.
    for (int channel = 0; channel < 4; ++channel)
    {
        FrontRunner runner(std::vector<Vertex>(forest.vertexCount), forest);
        ParallelFront cpu = ParallelFront::build(forest);
        const GpuState gpu = runner.applyView(makeScores(hot, face, channel), 1.0f, 0.6f);
        cpu.applyView(hot, face, 1.0f, 0.6f);
        CAPTURE(channel);
        expectMatchesCpu(gpu, cpu, forest);
    }
}

TEST_CASE("VDPM GPU refine/coarsen: diamond (same-rank atomic sharing)", "[.][gpu]")
{
    const VertexForest forest = diamondForest();
    // A dummy vertex array sized to vertexCount (the front topology uses only the forest; positions
    // are unused by refine/coarsen).
    const std::vector<Vertex> verts(forest.vertexCount);
    FrontRunner runner(verts, forest);
    ParallelFront cpu = ParallelFront::build(forest);

    const std::vector<std::uint8_t> face(forest.splits.size(), 0);

    SECTION("all over budget → the whole diamond refines; dependents_[1] == 2 (L and R share B)")
    {
        const std::vector<float> hot(forest.splits.size(), 9.0f);
        const GpuState gpu = runner.applyView(makeScores(hot, face), 1.0f, 0.6f);
        cpu.applyView(hot, face, 1.0f, 0.6f);
        expectMatchesCpu(gpu, cpu, forest);
        CHECK(gpu.dependents[1] == 2u);
    }
    SECTION("then all under budget → coarsens all the way back out")
    {
        const std::vector<float> hot(forest.splits.size(), 9.0f);
        (void)runner.applyView(makeScores(hot, face), 1.0f, 0.6f);
        cpu.applyView(hot, face, 1.0f, 0.6f);
        const std::vector<float> cold(forest.splits.size(), 0.0f);
        const GpuState gpu = runner.applyView(makeScores(cold, face), 1.0f, 0.6f);
        cpu.applyView(cold, face, 1.0f, 0.6f);
        expectMatchesCpu(gpu, cpu, forest);
        for (std::uint32_t v = 1; v <= 4; ++v)
        {
            CHECK(gpu.active[v] == 0u);
        }
    }
}

TEST_CASE("VDPM GPU refine/coarsen: deep chain fully refines then fully coarsens", "[.][gpu]")
{
    const VertexForest forest = chainForest(24); // rank 0..23
    const std::vector<Vertex> verts(forest.vertexCount);
    FrontRunner runner(verts, forest);
    ParallelFront cpu = ParallelFront::build(forest);
    const std::vector<std::uint8_t> face(forest.splits.size(), 0);

    const std::vector<float> hot(forest.splits.size(), 9.0f);
    GpuState gpu = runner.applyView(makeScores(hot, face), 1.0f, 0.6f);
    cpu.applyView(hot, face, 1.0f, 0.6f);
    expectMatchesCpu(gpu, cpu, forest);
    for (std::uint32_t v = 0; v <= 24; ++v)
    {
        CHECK(gpu.active[v] == 1u); // the whole chain refined in
    }

    const std::vector<float> cold(forest.splits.size(), 0.0f);
    gpu = runner.applyView(makeScores(cold, face), 1.0f, 0.6f);
    cpu.applyView(cold, face, 1.0f, 0.6f);
    expectMatchesCpu(gpu, cpu, forest);
    for (std::uint32_t v = 1; v <= 24; ++v)
    {
        CHECK(gpu.active[v] == 0u); // ...and coarsened back out in one reverse sweep
    }
}

TEST_CASE("VDPM GPU refine/coarsen: alternating hot/cold frames (required overwrite)", "[.][gpu]")
{
    const VertexForest forest = forestOf(uvSphere(16, 20));
    const auto n = static_cast<std::uint32_t>(forest.splits.size());
    FrontRunner runner(std::vector<Vertex>(forest.vertexCount), forest);
    ParallelFront cpu = ParallelFront::build(forest);
    const std::vector<std::uint8_t> face(n, 0);

    for (int frame = 0; frame < 6; ++frame)
    {
        const float v = (frame % 2 == 0) ? 9.0f : 0.0f; // hot, cold, hot, ...
        const std::vector<float> scalar(n, v);
        const GpuState gpu = runner.applyView(makeScores(scalar, face), 1.0f, 0.6f);
        cpu.applyView(scalar, face, 1.0f, 0.6f);
        CAPTURE(frame);
        expectMatchesCpu(gpu, cpu, forest);
    }
}

TEST_CASE("VDPM GPU refine/coarsen: back-facing splits coarsen", "[.][gpu]")
{
    const VertexForest forest = forestOf(uvSphere(16, 20));
    const auto n = static_cast<std::uint32_t>(forest.splits.size());
    FrontRunner runner(std::vector<Vertex>(forest.vertexCount), forest);
    ParallelFront cpu = ParallelFront::build(forest);

    // Refine everything, then mark all back-facing (score high but backface set) → all must
    // coarsen.
    const std::vector<float> hot(n, 9.0f);
    (void)runner.applyView(makeScores(hot, std::vector<std::uint8_t>(n, 0)), 1.0f, 0.6f);
    cpu.applyView(hot, std::vector<std::uint8_t>(n, 0), 1.0f, 0.6f);

    const std::vector<std::uint8_t> allBack(n, 1);
    const GpuState gpu = runner.applyView(makeScores(hot, allBack), 1.0f, 0.6f);
    cpu.applyView(hot, allBack, 1.0f, 0.6f);
    expectMatchesCpu(gpu, cpu, forest);
    // Back-facing everywhere ⇒ nothing refined; a non-root vertex (one with a removing split) is
    // active exactly when that split is refined, so all must be inactive — the coarsest front.
    // (Roots stay active; a real sphere has many, so we key off the forest, not vertex 0.)
    for (std::uint32_t v = 0; v < forest.vertexCount; ++v)
    {
        if (forest.removingSplit[v] != kNoSplit)
        {
            CHECK(gpu.active[v] == 0u);
        }
    }
    CHECK(std::ranges::none_of(gpu.refined, [](std::uint32_t r) { return r != 0u; }));
}

TEST_CASE("VDPM GPU refine/coarsen: two apply cycles back-to-back (inter-call barrier)", "[.][gpu]")
{
    const VertexForest forest = forestOf(grid(13));
    const auto n = static_cast<std::uint32_t>(forest.splits.size());
    FrontRunner runner(std::vector<Vertex>(forest.vertexCount), forest);
    ParallelFront cpu = ParallelFront::build(forest);

    std::mt19937 rng(0x2CE11);
    std::uniform_real_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> scalar(n);
    std::vector<std::uint8_t> face(n, 0);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        scalar[i] = dist(rng);
    }
    const std::vector<VdpmScoreOut> scores = makeScores(scalar, face);

    // Two GPU cycles in one submit (no CPU wait between); the CPU model applies the SAME scores
    // twice. The second cycle re-applies the settled front, so the two must agree — proving the
    // inter-call barriers serialise the persistent state.
    const GpuState gpu = runner.applyViewTwice(scores, 1.0f, 0.6f);
    cpu.applyView(scalar, face, 1.0f, 0.6f);
    cpu.applyView(scalar, face, 1.0f, 0.6f);
    expectMatchesCpu(gpu, cpu, forest);
}

TEST_CASE("VDPM GPU refine/coarsen: single-rank and empty forests", "[.][gpu]")
{
    SECTION("single-rank forest (one root vertex, splits all rank 0)")
    {
        // A star: root vertex 0; splits each remove a distinct leaf directly from the root, so
        // every split is rank 0 (no split depends on another).
        VertexForest f;
        const std::uint32_t leaves = 5;
        f.vertexCount = leaves + 1;
        f.removingSplit.assign(f.vertexCount, kNoSplit);
        for (std::uint32_t i = 0; i < leaves; ++i)
        {
            f.removingSplit[i + 1] = i;
            VertexSplit s;
            s.child = i + 1;
            s.parent = 0;
            s.vl = 0;
            s.vr = kInvalidVertex;
            f.splits.push_back(s);
        }
        REQUIRE(buildDependencyDag(f).maxRank == 0u);

        FrontRunner runner(std::vector<Vertex>(f.vertexCount), f);
        ParallelFront cpu = ParallelFront::build(f);
        const std::vector<float> hot(leaves, 9.0f);
        const std::vector<std::uint8_t> face(leaves, 0);
        const GpuState gpu = runner.applyView(makeScores(hot, face), 1.0f, 0.6f);
        cpu.applyView(hot, face, 1.0f, 0.6f);
        expectMatchesCpu(gpu, cpu, f);
    }
    SECTION("split-free forest (roots only) records nothing and stays at the coarsest front")
    {
        VertexForest f;
        f.vertexCount = 3;
        f.removingSplit.assign(3, kNoSplit); // all roots ⇒ all active, no splits
        FrontRunner runner(std::vector<Vertex>(f.vertexCount), f);
        const GpuState gpu = runner.applyView({}, 1.0f, 0.6f);
        CHECK(gpu.failFlags[0] == 0u);
        CHECK(gpu.failFlags[1] == 0u);
        for (std::uint32_t v = 0; v < 3; ++v)
        {
            CHECK(gpu.active[v] == 1u); // every root active
        }
    }
    SECTION("genuinely empty forest (zero vertices, zero splits) — failFlags still readable")
    {
        VertexForest f; // vertexCount 0, no splits, no removingSplit
        FrontRunner runner({}, f);
        const GpuState gpu = runner.applyView({}, 1.0f, 0.6f);
        CHECK(gpu.active.empty());
        CHECK(gpu.refined.empty());
        CHECK(gpu.failFlags[0] == 0u); // the diagnostic buffer exists even for an empty front
        CHECK(gpu.failFlags[1] == 0u);
    }
}

// Acceptance evidence for the rank-per-dispatch scheme (Stage B3): the per-instance dispatch +
// barrier counts (analytic, exact), the DAG rank depth per mesh class, and the measured per-frame
// CPU record+submit+wait cost for a SINGLE instance. It records the numbers so the B5 integration
// can decide whether to batch instances into each rank dispatch (or a work queue) if the
// per-instance dispatch count dominates — but it does NOT prove rank-per-dispatch viable at scale.
// MULTI-INSTANCE scaling, a GPU/CPU-time split (timestamp queries), and a representative ASSET (the
// helmet, which needs the Vulkan glTF path) are all deferred to B5 render integration. Run with
// `./test_fire_engine "[B3Evidence]"` from the build dir.
TEST_CASE("VDPM GPU refine/coarsen: dispatch + rank evidence", "[.][gpu][B3Evidence]")
{
    struct Case
    {
        const char* name;
        Mesh mesh;
    };
    std::vector<Case> cases;
    cases.push_back({"sphere(24,32)", uvSphere(24, 32)});
    cases.push_back({"grid(65) [deep]", grid(65)});

    for (const Case& c : cases)
    {
        const VertexForest forest = forestOf(c.mesh);
        const auto n = static_cast<std::uint32_t>(forest.splits.size());
        const std::uint32_t maxRank = buildDependencyDag(forest).maxRank;

        // Per applyView: mark(1) + close(maxRank+1) + refine(maxRank+1) + coarsen(maxRank+1).
        const std::uint32_t dispatches = 1 + 3 * (maxRank + 1);
        // prior(1) + clear(1) + recorder-leading(1) + close(maxRank+1) + refine-between(maxRank) +
        // recorder-trailing(1) + coarsen-between(maxRank).
        const std::uint32_t barriers = 5 + 3 * maxRank;

        FrontRunner runner(c.mesh.verts, forest);
        std::mt19937 rng(0xE7); // NOLINT — deterministic evidence
        std::uniform_real_distribution<float> dist(0.0f, 2.0f);
        const std::vector<std::uint8_t> face(n, 0);

        constexpr int frames = 60;
        const auto t0 = std::chrono::steady_clock::now();
        for (int f = 0; f < frames; ++f)
        {
            std::vector<float> scalar(n);
            for (std::uint32_t i = 0; i < n; ++i)
            {
                scalar[i] = dist(rng);
            }
            (void)runner.applyView(makeScores(scalar, face), 1.0f, 0.6f);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double msPerFrame =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / frames;

        WARN(c.name << ": " << n << " splits, maxRank " << maxRank << " ⇒ " << dispatches
                    << " dispatches + " << barriers << " barriers per instance per frame; "
                    << msPerFrame << " ms/frame (1 instance, record+submit+wait)");
        CHECK(dispatches > 0);
    }
}
