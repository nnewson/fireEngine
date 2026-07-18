#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

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

// GPU VDPM repair harness (rendering-spine #3, GPU-driven-front Stage B4). Tagged [.][gpu]. Runs
// the GPU foldover/coverage repair against the CPU model + the shared first-principles violation
// counters. The AUTHORITATIVE contract (FP-robust): after repair the GPU front has zero
// CPU-classified foldovers + zero coverage failures, valid invariants, no GPU failure/ancestor
// flags, and emitted triangles no greater than full detail. Plus DIRECT per-face classifier
// readback (every branch) and the fallback path.

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
            const float u = static_cast<float>(x) / static_cast<float>(n - 1);
            const float v = static_cast<float>(y) / static_cast<float>(n - 1);
            m.verts.push_back(Vertex{Vec3{u - 0.5f, v - 0.5f, 0.0f}, Colour3{},
                                     Vec3{0.0f, 0.0f, 1.0f}, Vec2{u, v}});
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

std::vector<VdpmScoreOut> makeScores(std::span<const float> scalar,
                                     std::span<const std::uint8_t> backface)
{
    std::vector<VdpmScoreOut> out(scalar.size());
    for (std::size_t i = 0; i < scalar.size(); ++i)
    {
        out[i] = VdpmScoreOut{.geometry = scalar[i], .backface = backface[i]};
    }
    return out;
}

VdpmRepairParams makeParams(const Mat4& world, const Mat4& viewProj, const Vec3& cam, float vw,
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

// A read-back GPU front wrapped to satisfy the test:: violation counters (forest() + active()).
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

struct RepairState
{
    std::vector<std::uint32_t> active;
    std::vector<std::uint32_t> refined;
    std::vector<std::uint32_t> dependents;
    std::array<std::uint32_t, 4> control{0, 0, 0, 0}; // anyMarked, ancestorFailure, fallback, pad
    std::array<std::uint32_t, 2> failFlags{0, 0};     // B3 refine failure, dependents underflow
    std::vector<std::uint32_t> classification;        // per finest face (recordDetectClassify only)
};

// A headless compute device + a persistent full-mesh GPU front (score + refine/coarsen + repair).
class RepairRunner
{
public:
    RepairRunner(std::span<const Vertex> verts, std::span<const std::uint32_t> indices,
                 const VertexForest& forest)
        : device_(Device::headlessCompute()),
          resources_(device_),
          refinePipelines_(device_),
          repairPipelines_(device_),
          pool_(
              device_.device(),
              vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        .queueFamilyIndex = device_.graphicsFamily()}),
          mesh_(VdpmGpuMesh::build(resources_, verts, indices, forest)),
          front_(VdpmGpuFront::buildWithFront(resources_, mesh_,
                                              /*withClassificationReadback=*/true)),
          vertexCount_(forest.vertexCount),
          splitCount_(static_cast<std::uint32_t>(forest.splits.size())),
          finestFaceCount_(mesh_.binding().finestFaceCount)
    {
    }
    RepairRunner(const RepairRunner&) = delete;
    RepairRunner& operator=(const RepairRunner&) = delete;
    RepairRunner(RepairRunner&&) = delete;
    RepairRunner& operator=(RepairRunner&&) = delete;
    ~RepairRunner() = default;

    [[nodiscard]] std::uint32_t finestFaceCount() const noexcept
    {
        return finestFaceCount_;
    }

    // applyView (settle) then recordRepair; read back the front state + control.
    [[nodiscard]] RepairState settleAndRepair(std::span<const VdpmScoreOut> scores, float budget,
                                              float coarsen, const VdpmRepairParams& params,
                                              std::uint32_t roundBudget)
    {
        return run(
            [&](vk::CommandBuffer cmd)
            {
                front_.recordApplyView(cmd, refinePipelines_, resources_, scores, budget, coarsen);
                front_.recordRepair(cmd, refinePipelines_, repairPipelines_, resources_, params,
                                    roundBudget);
            },
            /*wantClassification=*/false);
    }

    // applyView (settle) then a single classifying detect (no repair mutation).
    [[nodiscard]] RepairState settleAndClassify(std::span<const VdpmScoreOut> scores, float budget,
                                                float coarsen, const VdpmRepairParams& params)
    {
        return run(
            [&](vk::CommandBuffer cmd)
            {
                front_.recordApplyView(cmd, refinePipelines_, resources_, scores, budget, coarsen);
                front_.recordDetectClassify(cmd, repairPipelines_, resources_, params);
            },
            /*wantClassification=*/true);
    }

private:
    template <class RecordFn>
    [[nodiscard]] RepairState run(RecordFn&& record, bool wantClassification)
    {
        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        record(*cmd);

        // Producer → transfer barrier. repairControl is CLEAR- and compute-written, so name both.
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
        const Resources::MappedBufferSet a = copyBack(front_.activeStateBuffer(), vertexCount_);
        const Resources::MappedBufferSet r = copyBack(front_.refinedStateBuffer(), splitCount_);
        const Resources::MappedBufferSet d = copyBack(front_.dependentsStateBuffer(), vertexCount_);
        const Resources::MappedBufferSet ctrl = copyBack(front_.repairControlBuffer(), 4);
        const Resources::MappedBufferSet ff = copyBack(front_.failFlagsBuffer(), 2);
        const Resources::MappedBufferSet cls =
            wantClassification ? copyBack(front_.repairClassificationBuffer(), finestFaceCount_)
                               : Resources::MappedBufferSet{};

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

        RepairState st;
        auto read = [](const Resources::MappedBufferSet& set, std::uint32_t count)
        {
            std::vector<std::uint32_t> out(count);
            if (count > 0)
            {
                std::memcpy(out.data(), set.mapped[0].data(), count * sizeof(std::uint32_t));
            }
            return out;
        };
        st.active = read(a, vertexCount_);
        st.refined = read(r, splitCount_);
        st.dependents = read(d, vertexCount_);
        const std::vector<std::uint32_t> c = read(ctrl, 4);
        std::ranges::copy(c, st.control.begin());
        const std::vector<std::uint32_t> flags = read(ff, 2);
        std::ranges::copy(flags, st.failFlags.begin());
        if (wantClassification)
        {
            st.classification = read(cls, finestFaceCount_);
        }
        return st;
    }

    Device device_;
    Resources resources_;
    VdpmRefinePipelines refinePipelines_;
    VdpmRepairPipelines repairPipelines_;
    vk::raii::CommandPool pool_;
    VdpmGpuMesh mesh_;
    VdpmGpuFront front_;
    std::uint32_t vertexCount_;
    std::uint32_t splitCount_;
    std::uint32_t finestFaceCount_;
};

// A simple perspective-ish view: camera on +z looking at the origin, a diagonal viewProj that
// projects the unit-ish meshes into NDC. Kept jitter-free (as repairFront requires).
Mat4 lookAtProj(const Vec3& cam)
{
    // Translate camera to origin then a mild perspective foreshortening on z. Column-major Mat4.
    Mat4 view = Mat4::identity();
    view[0, 3] = -cam.x();
    view[1, 3] = -cam.y();
    view[2, 3] = -cam.z();
    Mat4 proj = Mat4::identity();
    proj[2, 2] = -1.0f;
    proj[3, 2] = -1.0f; // w = -z (so a point in front, z<cam, has w>0)
    proj[3, 3] = 0.0f;
    return proj * view;
}

// The CPU per-face classification packed the SAME way as the GPU (`kVdpmDetect*`), against a given
// active state — the oracle for the direct classifier test.
std::uint32_t cpuClassify(const std::array<std::uint32_t, 3>& fc, std::span<const Vertex> verts,
                          const VertexForest& forest, std::span<const std::uint32_t> active,
                          const VdpmRepairParams& params)
{
    auto ancestor = [&](std::uint32_t v)
    {
        while (active[v] == 0u)
        {
            v = forest.splits[forest.removingSplit[v]].parent;
        }
        return v;
    };
    const Mat4 world = params.world;
    const Mat4 viewProj = params.viewProj;
    const Vec3 cam{params.cameraPos[0], params.cameraPos[1], params.cameraPos[2]};
    auto wp = [&](std::uint32_t v)
    {
        const Vec3 p = verts[v].position();
        const Vec4 w = world * Vec4{p.x(), p.y(), p.z(), 1.0f};
        return Vec3{w.x(), w.y(), w.z()};
    };
    const std::uint32_t a0 = ancestor(fc[0]);
    const std::uint32_t a1 = ancestor(fc[1]);
    const std::uint32_t a2 = ancestor(fc[2]);
    const bool degenerate = (a0 == a1 || a1 == a2 || a0 == a2);
    const std::array<Vec3, 3> original{wp(fc[0]), wp(fc[1]), wp(fc[2])};
    const std::array<Vec3, 3> replacement{wp(a0), wp(a1), wp(a2)};
    const std::array<bool, 3> inactive{active[fc[0]] == 0u, active[fc[1]] == 0u,
                                       active[fc[2]] == 0u};

    std::uint32_t packed = 0;
    if (detail::isFoldover(original, replacement, degenerate))
    {
        packed |= kVdpmDetectFoldoverBit;
    }
    const detail::CoverageRepair rep = detail::classifyCoverageRepair(
        original, replacement, degenerate, inactive, cam, viewProj, params.viewport[0],
        params.viewport[1], params.viewport[2] != 0.0f);
    std::uint32_t kind = 0;
    std::uint32_t worstLocal = 0;
    if (rep.kind == detail::CoverageRepairKind::AllInactiveCorners)
    {
        kind = 1;
    }
    else if (rep.kind == detail::CoverageRepairKind::WorstInactiveCorner)
    {
        kind = 2;
        worstLocal = rep.worstCorner;
    }
    packed |= (kind << kVdpmDetectCoverageKindShift);
    packed |= (worstLocal << kVdpmDetectWorstCornerShift);
    return packed;
}

} // namespace

TEST_CASE("VDPM GPU repair: settled+repaired front has zero foldover + zero coverage failures",
          "[.][gpu]")
{
    const QuadricSimplifier simp;
    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 viewProj = lookAtProj(cam);
    const float vw = 1024.0f;
    const float vh = 768.0f;

    struct Config
    {
        const char* name;
        Mat4 world;
        bool cull;
    };
    const std::array<Config, 4> configs{
        Config{"identity, cull", Mat4::identity(), true},
        Config{"non-uniform, no-cull", Mat4::scale(Vec3{1.6f, 0.7f, 1.3f}), false},
        Config{"reflected, cull", Mat4::scale(Vec3{-1.0f, 1.0f, 1.0f}), true},
        Config{"reflected, no-cull", Mat4::scale(Vec3{1.0f, -1.0f, 1.0f}), false},
    };

    for (const Mesh& m : {uvSphere(18, 24), grid(21)})
    {
        const auto collapses = simp.collapseSequence(m.verts, m.indices);
        const VertexForest forest = buildVertexForest(m.verts, collapses);
        const auto n = static_cast<std::uint32_t>(forest.splits.size());
        REQUIRE(n > 1);
        const std::size_t finestTris =
            ParallelFront::build(m.verts, m.indices, collapses).finestFaces().size();
        const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(m.verts);
        const std::vector<std::array<std::uint32_t, 3>> finestFaces =
            mesh_topology::canonicalFaces(weld, m.indices);
        const std::vector<std::uint8_t> backface(n, 0);

        RepairRunner runner(m.verts, m.indices, forest); // ONE persistent front across the configs

        // Each config re-settles the persistent front to a COARSE state (scores under budget) —
        // real foldover/coverage holes under that transform/cull/view — then repairs. Sweeping the
        // matrix on ONE runner exercises persistent TRANSFORM/CULL changes: config k+1's applyView
        // must re-settle the front config k's repair left refined (coarsening back what its view no
        // longer needs) before repairing for the new view. (A budget-driven persistent change is
        // asserted separately below.)
        for (const Config& cfg : configs)
        {
            CAPTURE(cfg.name);
            const float budget = 1.0f;
            const std::vector<float> scalar(n, 0.2f);
            const VdpmRepairParams params = makeParams(cfg.world, viewProj, cam, vw, vh, cfg.cull);

            ParallelFront cpuSettled = ParallelFront::build(m.verts, m.indices, collapses);
            cpuSettled.applyView(scalar, backface, budget, 0.6f);
            REQUIRE((test::foldoverCount(cpuSettled, m.verts, m.indices, cfg.world) +
                     test::coverageFailures(cpuSettled, m.verts, m.indices, viewProj, cam,
                                            cfg.world, vw, vh, cfg.cull)) >
                    0); // really needs repair

            const RepairState st =
                runner.settleAndRepair(makeScores(scalar, backface), budget, 0.6f, params,
                                       /*roundBudget=*/24);

            // Diagnostics clean (B3 failFlags AND B4 control).
            CHECK(st.failFlags[0] == 0u); // refine failure
            CHECK(st.failFlags[1] == 0u); // dependents underflow
            CHECK(st.control[1] == 0u);   // ancestor failures
            CHECK(st.control[2] == 0u);   // converged in the round budget (no full-detail fallback)
            CHECK_NOTHROW(validateFrontInvariants(forest, st.active, st.refined, st.dependents));

            // AUTHORITATIVE: zero CPU-classified violations on the GPU front for this view.
            const GpuFrontView view{forest, st.active};
            CHECK(test::foldoverCount(view, m.verts, m.indices, cfg.world) == 0);
            CHECK(test::coverageFailures(view, m.verts, m.indices, viewProj, cam, cfg.world, vw, vh,
                                         cfg.cull) == 0);

            // Emitted triangles never exceed full detail (the general bound).
            std::size_t emitted = 0;
            for (const std::array<std::uint32_t, 3>& fc : finestFaces)
            {
                auto anc = [&](std::uint32_t v)
                {
                    while (st.active[v] == 0u)
                    {
                        v = forest.splits[forest.removingSplit[v]].parent;
                    }
                    return v;
                };
                const std::uint32_t x = anc(fc[0]);
                const std::uint32_t y = anc(fc[1]);
                const std::uint32_t z = anc(fc[2]);
                emitted += (x != y && y != z && x != z) ? 1 : 0;
            }
            CHECK(emitted <= finestTris);
        }

        // Persistent BUDGET change: on a FRESH front (coarsest baseline — the matrix runner above
        // is already repair-refined, and coarsen hysteresis would keep that detail regardless of
        // budget), scores that STRADDLE the two budgets. Settling the LOOSER budget then the
        // TIGHTER one on the SAME persistent front refines strictly more (coarse→fine is monotone).
        // Comparing the applyView-settled fronts (pre-repair) shows the persistent front responds
        // to the budget.
        RepairRunner budgetRunner(m.verts, m.indices, forest);
        const VdpmRepairParams idParams = makeParams(Mat4::identity(), viewProj, cam, vw, vh, true);
        std::vector<float> straddle(n);
        for (std::uint32_t i = 0; i < n; ++i)
        {
            straddle[i] = (i % 3 == 0) ? 1.8f : (i % 3 == 1 ? 1.2f : 0.4f);
        }
        const std::vector<std::uint32_t> loose =
            budgetRunner.settleAndClassify(makeScores(straddle, backface), 1.5f, 0.6f, idParams)
                .active;
        const std::vector<std::uint32_t> tight =
            budgetRunner.settleAndClassify(makeScores(straddle, backface), 1.0f, 0.6f, idParams)
                .active;
        CHECK(tight != loose); // budget 1.0 additionally refines the score-1.2 splits
    }
}

TEST_CASE("VDPM GPU repair: per-face classification matches the CPU classifiers (all branches)",
          "[.][gpu]")
{
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(16, 20);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);
    const auto n = static_cast<std::uint32_t>(forest.splits.size());

    RepairRunner runner(m.verts, m.indices, forest);
    const Vec3 cam{0.0f, 0.0f, 2.5f};
    // A NON-UNIFORM world: per the isFoldover contract it can flip the relative orientation of a
    // face vs its active-ancestor replacement, so this reliably exercises the foldover branch too.
    const VdpmRepairParams params = makeParams(Mat4::scale(Vec3{1.7f, 0.5f, 1.2f}), lookAtProj(cam),
                                               cam, 1024.0f, 768.0f, true);

    const std::vector<std::uint32_t> weld = mesh_topology::weldByPosition(m.verts);
    const std::vector<std::array<std::uint32_t, 3>> finestFaces =
        mesh_topology::canonicalFaces(weld, m.indices);
    const std::vector<std::uint8_t> backface(n, 0);

    // Several MIXED fronts (deterministic pseudo-random scores straddling the budget) → different
    // active sets so the classifier's branches (foldover / all-inactive / worst-corner / none) are
    // collectively exercised. On EVERY face of EVERY front the GPU classification must EXACTLY
    // equal the CPU classifier (they run the identical policy; a stubbed/wrong branch would diverge
    // where it occurs).
    std::size_t totalMismatches = 0;
    std::array<std::size_t, 3> coverageKindSeen{0, 0, 0};
    std::size_t foldoversSeen = 0;
    std::mt19937 rng(0xB4C1A55);
    std::uniform_real_distribution<float> dist(0.0f, 2.0f);
    for (int trial = 0; trial < 6; ++trial)
    {
        std::vector<float> scalar(n);
        for (std::uint32_t i = 0; i < n; ++i)
        {
            scalar[i] = dist(rng);
        }
        const RepairState st =
            runner.settleAndClassify(makeScores(scalar, backface), 1.0f, 0.6f, params);
        REQUIRE(st.classification.size() == finestFaces.size());
        for (std::size_t f = 0; f < finestFaces.size(); ++f)
        {
            const std::uint32_t cpu =
                cpuClassify(finestFaces[f], m.verts, forest, st.active, params);
            totalMismatches += (cpu != st.classification[f]) ? 1 : 0;
            coverageKindSeen[(cpu >> kVdpmDetectCoverageKindShift) & kVdpmDetectCoverageKindMask]++;
            foldoversSeen += (cpu & kVdpmDetectFoldoverBit) ? 1 : 0;
        }
    }
    CHECK(totalMismatches == 0); // the faithful-port proof, across many faces + branches
    // The mixed fronts collectively exercise all three coverage kinds (a shader that stubbed one
    // would be caught by the exact-match check where that branch occurs).
    CHECK(coverageKindSeen[0] > 0); // None
    CHECK(coverageKindSeen[1] > 0); // AllInactiveCorners
    CHECK(coverageKindSeen[2] > 0); // WorstInactiveCorner
    CHECK(foldoversSeen > 0);       // the foldover branch really ran (non-uniform world)
    WARN("classifier branches over 6 mixed fronts: None "
         << coverageKindSeen[0] << ", AllInactive " << coverageKindSeen[1] << ", Worst "
         << coverageKindSeen[2] << ", foldovers " << foldoversSeen);
}

TEST_CASE("VDPM GPU repair: a tiny round budget forces the full-detail fallback", "[.][gpu]")
{
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(18, 24);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);
    const auto n = static_cast<std::uint32_t>(forest.splits.size());

    RepairRunner runner(m.verts, m.indices, forest);
    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const VdpmRepairParams params = makeParams(world, viewProj, cam, 1024.0f, 768.0f, true);

    // A coarse front needs several repair rounds; budget 0 → the final detect still finds
    // violations → the fallback fires and drives to full detail (guaranteed hole-free).
    const std::vector<float> cold(n, 0.0f);
    const std::vector<std::uint8_t> backface(n, 0);
    const RepairState st =
        runner.settleAndRepair(makeScores(cold, backface), 1.0f, 0.6f, params, /*roundBudget=*/0);

    CHECK(st.control[2] == 1u);   // fallback fired
    CHECK(st.control[1] == 0u);   // no ancestor failures
    CHECK(st.failFlags[0] == 0u); // no refine failure even on the fallback path
    CHECK(st.failFlags[1] == 0u); // no dependents underflow
    CHECK_NOTHROW(validateFrontInvariants(forest, st.active, st.refined, st.dependents));
    // Full detail ⇒ every split refined ⇒ zero violations trivially.
    CHECK(std::ranges::all_of(st.refined, [](std::uint32_t r) { return r != 0u; }));
    const GpuFrontView view{forest, st.active};
    CHECK(test::foldoverCount(view, m.verts, m.indices, world) == 0);
    CHECK(test::coverageFailures(view, m.verts, m.indices, viewProj, cam, world, 1024.0f, 768.0f,
                                 true) == 0);
}

TEST_CASE("VDPM GPU repair: a full-detail mesh (faces, zero splits) classifies without required",
          "[.][gpu]")
{
    // A single triangle can't be simplified — zero splits but one finest face. recordDetectClassify
    // must not fillBuffer the (null) required buffer; every corner is a root (active), so the
    // detector runs and every face classifies as None with nothing marked.
    Mesh m;
    m.verts = {Vertex{Vec3{-0.5f, -0.5f, 0.0f}, Colour3{}, Vec3{0, 0, 1}, Vec2{0, 0}},
               Vertex{Vec3{0.5f, -0.5f, 0.0f}, Colour3{}, Vec3{0, 0, 1}, Vec2{1, 0}},
               Vertex{Vec3{0.0f, 0.5f, 0.0f}, Colour3{}, Vec3{0, 0, 1}, Vec2{0, 1}}};
    m.indices = {0, 1, 2};
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);
    REQUIRE(forest.splits.empty());         // nothing to simplify
    REQUIRE(!forest.removingSplit.empty()); // still 3 (root) vertices

    RepairRunner runner(m.verts, m.indices, forest);
    REQUIRE(runner.finestFaceCount() == 1u);
    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const VdpmRepairParams params =
        makeParams(Mat4::identity(), lookAtProj(cam), cam, 1024.0f, 768.0f, true);

    // No splits ⇒ applyView/repair are no-ops; the classifying detect still runs over the one face.
    const RepairState st = runner.settleAndClassify({}, 1.0f, 0.6f, params);
    REQUIRE(st.classification.size() == 1u);
    CHECK(st.control[0] == 0u); // nothing marked (all corners active)
    CHECK(st.control[1] == 0u); // no ancestor failures
    CHECK((st.classification[0] & kVdpmDetectFoldoverBit) == 0u);
    CHECK(((st.classification[0] >> kVdpmDetectCoverageKindShift) & kVdpmDetectCoverageKindMask) ==
          0u); // None
}

TEST_CASE("VDPM GPU repair: evidence (rounds vs the CPU model, fallback)", "[.][gpu][B4Evidence]")
{
    const QuadricSimplifier simp;
    const Mesh m = uvSphere(20, 28);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VertexForest forest = buildVertexForest(m.verts, collapses);
    const auto n = static_cast<std::uint32_t>(forest.splits.size());

    RepairRunner runner(m.verts, m.indices, forest);
    const Vec3 cam{0.0f, 0.0f, 2.5f};
    const Mat4 world = Mat4::identity();
    const Mat4 viewProj = lookAtProj(cam);
    const VdpmRepairParams params = makeParams(world, viewProj, cam, 1024.0f, 768.0f, true);

    const std::vector<float> scalar(n, 1.2f);
    const std::vector<std::uint8_t> backface(n, 0);

    // CPU reference: actual detection passes + apply rounds.
    ParallelFront cpu = ParallelFront::build(m.verts, m.indices, collapses);
    cpu.applyView(scalar, backface, 1.0f, 0.6f);
    cpu.repairFront(m.verts, world, cam, viewProj, 1024.0f, 768.0f, true);

    const std::uint32_t budget = 8;
    const RepairState st =
        runner.settleAndRepair(makeScores(scalar, backface), 1.0f, 0.6f, params, budget);

    WARN("CPU repair: detectionPasses "
         << cpu.repairDetectionPasses() << ", applyRounds " << cpu.repairApplyRounds()
         << ", refinedSplits " << cpu.repairRefinedSplits() << " | GPU: recorded dispatch budget "
         << budget << ", fallbackFired " << st.control[2]);
    CHECK(st.control[2] == 0u); // a generous budget converges without the fallback
}
