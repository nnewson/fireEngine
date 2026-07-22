#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <fire_engine/core/gltf_loader.hpp>
#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/vdpm_gpu.hpp>
#include <fire_engine/scene/scene_graph.hpp>

#include <support/vdpm.hpp>

using namespace fire_engine;

// ============================================================================================
// B5c-2 — HELMET WEDGE/RANK EVIDENCE ([.][gpu], local only). The REAL DamagedHelmet asset, decoded
// through the production GltfLoader::loadScene + the collapse stream Geometry::load already built
// (geometry.collapses()) — no re-simplify, no bespoke parser. Two cleanly-separated claims:
//   A. From a CPU-uploaded active set, the GPU emit is BYTE-IDENTICAL to ParallelFront::emitActive-
//      Indices at roots / mid / full — proving real-asset emit ORDER + seam wedge restoration at
//      nonzero ancestor depths (asserted actually reached, so the match can't be vacuous).
//   B. The complete GPU lifecycle on the helmet is invariant-valid + hole-free + clean-failFlags +
//      bounded — WITHOUT overclaiming FP identity (only exact when a clear score margin + a
//      full-detail reference are first proven).
// ============================================================================================

namespace
{
// A read-back GPU front wrapped for the test:: violation counters (forest() + active()).
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

// The B5c-2 fixture: the real helmet via a production load + a headless device + the shared
// pipelines. The load is heavy (textures/materials); each of the two claims constructs its own
// harness so they stay independently pass/failable — the duplicated load is accepted for that
// separation (this is a local-only [.][gpu] test).
class HelmetHarness
{
public:
    HelmetHarness()
        : device_(Device::headlessCompute()),
          resources_(device_),
          scorePipeline_(device_, vdpmScorePipelineConfig()),
          refinePipelines_(device_),
          repairPipelines_(device_),
          emitPipelines_(device_),
          pool_(device_.device(), vk::CommandPoolCreateInfo{
                                      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                      .queueFamilyIndex = device_.graphicsFamily()})
    {
        // Production decode: loadScene builds every Geometry (incl. its progressive-mesh collapse
        // stream) exactly as the app does. vdpmRegistry = nullptr ⇒ CPU-only geometry.
        SceneGraph scene;
        Assets assets;
        PhysicsWorld physics;
        GltfLoader::loadScene(std::string(FIRE_ENGINE_BUILD_ASSET_DIR) +
                                  "/DamagedHelmet/DamagedHelmet.gltf",
                              scene, resources_, assets, physics);

        // Select the helmet geometry UNAMBIGUOUSLY: exactly one VDPM-eligible geometry.
        const Geometry* helmet = nullptr;
        std::size_t eligible = 0;
        for (std::size_t i = 0; i < assets.geometryCount(); ++i)
        {
            if (assets.geometry(i).hasVdpmData())
            {
                helmet = &assets.geometry(i);
                ++eligible;
            }
        }
        REQUIRE(eligible == 1);
        REQUIRE(helmet != nullptr);

        verts_ = helmet->vertices();       // copy CPU geometry out before assets/scene die
        indices_ = helmet->indices();      //
        collapses_ = helmet->collapses();  // the PRODUCTION collapse stream (not a re-simplify)
        REQUIRE_FALSE(collapses_.empty()); // eligibility ⇒ non-empty
        forest_ = buildVertexForest(verts_, collapses_);
        mesh_ = VdpmGpuMesh::build(resources_, verts_, indices_, forest_);

        // The production per-front kernels when supported (else recordFrame uses the recorders).
        if (VdpmApplyKernel::deviceSupported(device_))
        {
            applyKernel_.emplace(device_);
        }
        if (VdpmRepairKernel::deviceSupported(device_))
        {
            repairKernel_.emplace(device_);
        }
    }
    HelmetHarness(const HelmetHarness&) = delete;
    HelmetHarness& operator=(const HelmetHarness&) = delete;
    HelmetHarness(HelmetHarness&&) = delete;
    HelmetHarness& operator=(HelmetHarness&&) = delete;
    ~HelmetHarness() = default;

    [[nodiscard]] std::span<const Vertex> verts() const
    {
        return verts_;
    }
    [[nodiscard]] std::span<const std::uint32_t> indices() const
    {
        return indices_;
    }
    [[nodiscard]] const std::vector<MeshCollapse>& collapses() const
    {
        return collapses_;
    }
    [[nodiscard]] const VertexForest& forest() const
    {
        return forest_;
    }
    [[nodiscard]] const VdpmGpuMesh& mesh() const
    {
        return *mesh_;
    }
    [[nodiscard]] Resources& resources()
    {
        return resources_;
    }

    // CLAIM A: emit a CPU-uploaded active set on a buildWithEmit GPU front, read back the emitted
    // index stream (truncated to counters[2]) + the ancestor-failure count. Mirrors the B2 emit
    // harness.
    struct EmitResult
    {
        std::vector<std::uint32_t> indices;
        std::uint32_t failureCount{0};
    };
    [[nodiscard]] EmitResult emit(std::span<const std::uint32_t> active)
    {
        VdpmGpuFront front = VdpmGpuFront::buildWithEmit(resources_, *mesh_);
        const std::uint32_t faceCount = front.faceCount();
        const vk::DeviceSize countersSize = 3 * sizeof(std::uint32_t);
        const vk::DeviceSize idxSize =
            static_cast<vk::DeviceSize>(faceCount) * 3 * sizeof(std::uint32_t);
        const Resources::MappedBufferSet countersHost =
            resources_.createMappedReadbackBuffers(countersSize);
        const Resources::MappedBufferSet idxHost =
            faceCount > 0 ? resources_.createMappedReadbackBuffers(idxSize)
                          : Resources::MappedBufferSet{};

        vk::raii::CommandBuffer cmd = begin();
        front.recordEmit(*cmd, emitPipelines_, resources_, active);
        // counters: mixed clear/compute source; emitted indices: compute source.
        barrier(*cmd,
                vk::PipelineStageFlagBits2::eClear | vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderStorageWrite);
        cmd.copyBuffer(resources_.vulkanBuffer(front.countersBuffer()),
                       resources_.vulkanBuffer(countersHost.buffers[0]),
                       vk::BufferCopy{.size = countersSize});
        if (faceCount > 0)
        {
            barrier(*cmd, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderStorageWrite);
            cmd.copyBuffer(resources_.vulkanBuffer(front.emittedIndicesBuffer()),
                           resources_.vulkanBuffer(idxHost.buffers[0]),
                           vk::BufferCopy{.size = idxSize});
        }
        toHostBarrier(*cmd);
        submit(cmd);

        std::array<std::uint32_t, 3> counters{};
        std::memcpy(counters.data(), countersHost.mapped[0].data(), countersSize);
        EmitResult out;
        out.failureCount = counters[0];
        const std::uint32_t emitted = counters[2];
        REQUIRE(counters[1] <= faceCount);    // survivors ≤ faces
        REQUIRE(emitted == 3u * counters[1]); // 3 indices per survivor
        REQUIRE(emitted <= 3u * faceCount);   // fits the worst-case allocation
        if (faceCount > 0 && emitted > 0)
        {
            out.indices.resize(emitted);
            std::memcpy(out.indices.data(), idxHost.mapped[0].data(),
                        emitted * sizeof(std::uint32_t));
        }
        return out;
    }

    // CLAIM B: the COMPLETE GPU lifecycle (score → apply → repair → emit via recordFrame, the
    // production per-front path with kernels when supported) on a runtime front at a real view; the
    // settled state + emitted stream read back for invariant/hole/failFlag/bounded checks.
    struct LifecycleResult
    {
        std::vector<std::uint32_t> active;
        std::vector<std::uint32_t> refined;
        std::vector<std::uint32_t> dependents;
        std::array<std::uint32_t, 2> failFlags{0, 0};
        std::vector<std::uint32_t> emitted;
    };
    [[nodiscard]] LifecycleResult runtimeLifecycle(const VdpmViewParams& view,
                                                   const VdpmRepairParams& repair, float budget)
    {
        VdpmGpuFront front = VdpmGpuFront::buildRuntime(resources_, *mesh_);
        const std::uint32_t vc = mesh_->binding().vertexCount;
        const std::uint32_t sc = mesh_->binding().splitCount;
        const std::uint32_t fc = front.faceCount();
        const VdpmApplyKernel* const ak = applyKernel_ ? &*applyKernel_ : nullptr;
        const VdpmRepairKernel* const rk = repairKernel_ ? &*repairKernel_ : nullptr;

        vk::raii::CommandBuffer cmd = begin();
        front.recordFrame(*cmd, scorePipeline_, refinePipelines_, repairPipelines_, emitPipelines_,
                          resources_, 0, view, repair, budget, kVdpmCoarsenRatio * budget,
                          kVdpmGpuRepairRoundBudget, ak, rk, nullptr);
        // recordFrame records no consumer barrier — order all front-state + emit writes → the
        // copies.
        barrier(*cmd,
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eClear,
                vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite);
        const Resources::MappedBufferSet a = copyState(*cmd, front.activeStateBuffer(), vc);
        const Resources::MappedBufferSet r = copyState(*cmd, front.refinedStateBuffer(), sc);
        const Resources::MappedBufferSet d = copyState(*cmd, front.dependentsStateBuffer(), vc);
        const Resources::MappedBufferSet ff = copyState(*cmd, front.failFlagsBuffer(), 2);
        const Resources::MappedBufferSet ct = copyState(*cmd, front.countersBuffer(0), 3);
        const Resources::MappedBufferSet em =
            fc > 0 ? copyState(*cmd, front.emittedIndicesBuffer(0), 3 * fc)
                   : Resources::MappedBufferSet{};
        toHostBarrier(*cmd);
        submit(cmd);

        LifecycleResult res;
        res.active = read(a, vc);
        res.refined = read(r, sc);
        res.dependents = read(d, vc);
        const std::vector<std::uint32_t> f = read(ff, 2);
        res.failFlags = {f[0], f[1]};
        std::array<std::uint32_t, 3> counters{};
        std::memcpy(counters.data(), ct.mapped[0].data(), sizeof(counters));
        // Guard the emitted count BEFORE it drives a mapped memcpy: a broken shader must not cause
        // an out-of-bounds read of the 3*fc-word `em` buffer. counters[0]==0 also ensures Claim B
        // cannot pass while emit silently drops faces after an ancestor-resolution failure.
        REQUIRE(counters[0] == 0u);               // no ancestor-resolution failures
        REQUIRE(counters[1] <= fc);               // survivors ≤ faces
        REQUIRE(counters[2] == 3u * counters[1]); // 3 indices per survivor
        REQUIRE(counters[2] <= 3u * fc);          // fits the worst-case allocation
        const std::uint32_t emitted = counters[2];
        if (fc > 0 && emitted > 0)
        {
            res.emitted.resize(emitted);
            std::memcpy(res.emitted.data(), em.mapped[0].data(), emitted * sizeof(std::uint32_t));
        }
        return res;
    }

private:
    [[nodiscard]] Resources::MappedBufferSet copyState(vk::CommandBuffer cmd, BufferHandle src,
                                                       std::uint32_t words)
    {
        Resources::MappedBufferSet host =
            resources_.createMappedReadbackBuffers(words * sizeof(std::uint32_t));
        cmd.copyBuffer(resources_.vulkanBuffer(src), resources_.vulkanBuffer(host.buffers[0]),
                       vk::BufferCopy{.size = words * sizeof(std::uint32_t)});
        return host;
    }
    static std::vector<std::uint32_t> read(const Resources::MappedBufferSet& host,
                                           std::uint32_t words)
    {
        std::vector<std::uint32_t> out(words);
        std::memcpy(out.data(), host.mapped[0].data(), words * sizeof(std::uint32_t));
        return out;
    }

    [[nodiscard]] vk::raii::CommandBuffer begin()
    {
        auto cmds = device_.device().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{.commandPool = *pool_,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1});
        cmds[0].begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return std::move(cmds[0]);
    }
    void submit(vk::raii::CommandBuffer& cmd)
    {
        cmd.end();
        const vk::CommandBufferSubmitInfo ci{.commandBuffer = *cmd};
        const vk::SubmitInfo2 s{.commandBufferInfoCount = 1, .pCommandBufferInfos = &ci};
        const vk::raii::Fence fence(device_.device(), vk::FenceCreateInfo{});
        device_.graphicsQueue().submit2(s, *fence);
        (void)device_.device().waitForFences(*fence, vk::True,
                                             std::numeric_limits<std::uint64_t>::max());
    }
    static void barrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage,
                        vk::AccessFlags2 srcAccess)
    {
        const vk::MemoryBarrier2 mb{.srcStageMask = srcStage,
                                    .srcAccessMask = srcAccess,
                                    .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
                                    .dstAccessMask = vk::AccessFlagBits2::eTransferRead};
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
    }
    static void toHostBarrier(vk::CommandBuffer cmd)
    {
        const vk::MemoryBarrier2 mb{.srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                                    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                    .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                                    .dstAccessMask = vk::AccessFlagBits2::eHostRead};
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
    }

    Device device_;
    Resources resources_;
    ComputePipeline scorePipeline_;
    VdpmRefinePipelines refinePipelines_;
    VdpmRepairPipelines repairPipelines_;
    VdpmEmitPipelines emitPipelines_;
    std::optional<VdpmApplyKernel> applyKernel_;
    std::optional<VdpmRepairKernel> repairKernel_;
    vk::raii::CommandPool pool_;
    std::vector<Vertex> verts_;
    std::vector<std::uint32_t> indices_;
    std::vector<MeshCollapse> collapses_;
    VertexForest forest_;
    std::optional<VdpmGpuMesh> mesh_;
};

// The CPU active set as a per-canonical 0/1 uint32 span (recordEmit's upload format).
std::vector<std::uint32_t> activeMask(const ParallelFront& f, std::uint32_t vertexCount)
{
    std::vector<std::uint32_t> a(vertexCount);
    for (std::uint32_t v = 0; v < vertexCount; ++v)
    {
        a[v] = f.active(v) ? 1u : 0u;
    }
    return a;
}
std::uint32_t activeCount(const ParallelFront& f, std::uint32_t vertexCount)
{
    std::uint32_t c = 0;
    for (std::uint32_t v = 0; v < vertexCount; ++v)
    {
        c += f.active(v) ? 1u : 0u;
    }
    return c;
}
// How many finest-face corners are RESTORED via an active ancestor (nonzero ancestor depth), and
// how many of those ancestors are MULTI-WEDGE canonicals — proving the emit's seam path is actually
// reached rather than trivially all-active.
struct RestorationStats
{
    std::size_t substituted{0};
    std::size_t multiWedge{0};
};
RestorationStats restorationStats(const ParallelFront& f)
{
    RestorationStats s;
    const VertexForest& forest = f.forest();
    const auto& csr = f.wedgesCsr();
    // The coarsest active ancestor up the removal chain (mirrors ParallelFront::activeAncestor,
    // which is private): removal-parent(c) = the vertex c collapses into =
    // splits[removingSplit[c]].parent.
    auto activeAncestor = [&](std::uint32_t c)
    {
        std::uint32_t u = c;
        std::size_t steps = 0;
        while (!f.active(u) && forest.removingSplit[u] != kNoSplit && steps <= forest.vertexCount)
        {
            u = forest.splits[forest.removingSplit[u]].parent;
            ++steps;
        }
        return u;
    };
    // finestFaces() is the order-preserving canonical (welded, weld-degenerate-dropped) proxy for
    // the original surviving faces. The scatter resolves each corner to its active ancestor and
    // keeps the face ONLY if the three resolved ancestors stay distinct; a face that collapses
    // under resolution is dropped before its corners' wedge-choice lookups take effect. Count
    // restoration only among SURVIVING faces so multiWedge strictly witnesses the seam lookup the
    // scatter runs.
    for (const std::array<std::uint32_t, 3>& face : f.finestFaces())
    {
        const std::array<std::uint32_t, 3> anc = {activeAncestor(face[0]), activeAncestor(face[1]),
                                                  activeAncestor(face[2])};
        if (anc[0] == anc[1] || anc[1] == anc[2] || anc[0] == anc[2])
        {
            continue; // non-surviving (degenerate under resolution) — not emitted
        }
        for (std::size_t i = 0; i < 3; ++i)
        {
            if (anc[i] != face[i]) // an inactive corner restored up its removal chain (depth ≥ 1)
            {
                ++s.substituted;
                if (csr.offsets[anc[i] + 1] - csr.offsets[anc[i]] > 1)
                {
                    ++s.multiWedge; // restored to a multi-wedge (seam) canonical
                }
            }
        }
    }
    return s;
}
} // namespace

TEST_CASE("VDPM helmet: byte-identical emit from a CPU active set (roots/mid/full)", "[.][gpu]")
{
    HelmetHarness helmet;
    const VertexForest& forest = helmet.forest();
    const std::uint32_t vertexCount = forest.vertexCount;
    const std::uint32_t splitCount = static_cast<std::uint32_t>(forest.splits.size());
    const std::vector<std::uint8_t> noBackface(splitCount, 0u);
    constexpr float kBudget = 0.5f;
    constexpr float kCoarsen = kVdpmCoarsenRatio * kBudget;

    // A refine state is chosen by SYNTHETIC per-split scores (deterministic, view-free): a score
    // above the budget refines that split, below leaves it. This lets us pin roots / mid / full
    // exactly, independent of the real asset's view-dependent scoring.
    auto refineTo = [&](std::span<const float> scores)
    {
        ParallelFront cpu =
            ParallelFront::build(helmet.verts(), helmet.indices(), helmet.collapses());
        cpu.applyView(scores, noBackface, kBudget, kCoarsen);
        return cpu;
    };
    auto claimA = [&](const ParallelFront& cpu)
    {
        const std::vector<std::uint32_t> active = activeMask(cpu, vertexCount);
        const std::vector<std::uint32_t> cpuIndices =
            cpu.emitActiveIndices(helmet.verts(), helmet.indices());
        const HelmetHarness::EmitResult gpu = helmet.emit(active);
        CHECK(gpu.failureCount == 0u);
        CHECK(gpu.indices == cpuIndices); // BYTE-IDENTICAL emit on the real asset
    };

    const std::vector<float> zero(splitCount, 0.0f);
    const std::vector<float> huge(splitCount, 1.0e9f);
    std::vector<float> half(splitCount, 0.0f);
    for (std::uint32_t s = 0; s < splitCount; ++s)
    {
        half[s] = (s % 2u == 0u) ? 1.0e9f : 0.0f; // deterministic ~half
    }

    const ParallelFront roots = refineTo(zero);
    const ParallelFront mid = refineTo(half);
    const ParallelFront full = refineTo(huge);

    claimA(roots); // deepest restoration (every non-root corner substitutes)
    claimA(mid);   // mid-depth restoration
    claimA(full);  // no restoration (all active) — the emit-order-at-full check

    // The three states are genuinely ordered, so "mid" really is between (not a degenerate alias).
    const std::uint32_t rc = activeCount(roots, vertexCount);
    const std::uint32_t mc = activeCount(mid, vertexCount);
    const std::uint32_t fc = activeCount(full, vertexCount);
    CHECK(rc < mc);
    CHECK(mc < fc);

    // The non-full cases MUST actually reach nonzero ancestor depth + multi-wedge (seam) buckets —
    // else the byte-identity above would be vacuous (all corners active, no restoration exercised).
    const RestorationStats rootsStats = restorationStats(roots);
    const RestorationStats midStats = restorationStats(mid);
    CHECK(rootsStats.substituted > 0);
    CHECK(rootsStats.multiWedge > 0);
    CHECK(midStats.substituted > 0);
    CHECK(midStats.multiWedge > 0);
    CHECK(restorationStats(full).substituted == 0); // full detail restores nothing

    // Static device-local footprint actually uploaded for this real mesh: the full B2–B4 buffer set
    // (splits/positions/frontSplits/splitsByRank/rankRanges/indices/weld/removalParent/wedgeChoices/
    // wedgeOffsets/finestFaces/removingSplit), and — reported separately — the precomputed
    // wedge-choice map (choices + offsets) the emit shader walks for the seam restoration proven
    // above.
    const std::size_t staticBytes = helmet.mesh().staticByteFootprint();
    const std::size_t wedgeMapBytes = helmet.mesh().wedgeMapByteFootprint();
    CHECK(staticBytes > 0);
    CHECK(wedgeMapBytes > 0);
    CHECK(wedgeMapBytes <= staticBytes);

    WARN("helmet Claim A: verts " << vertexCount << " splits " << splitCount << " maxRank "
                                  << helmet.mesh().binding().maxRank << " | active roots " << rc
                                  << " mid " << mc << " full " << fc << " | roots restored "
                                  << rootsStats.substituted << " (multi-wedge "
                                  << rootsStats.multiWedge << "), mid restored "
                                  << midStats.substituted << " (multi-wedge " << midStats.multiWedge
                                  << ") | static footprint " << staticBytes << " B (wedge map "
                                  << wedgeMapBytes << " B)");
}

TEST_CASE("VDPM helmet: full GPU lifecycle is valid, hole-free, and bounded", "[.][gpu]")
{
    HelmetHarness helmet;
    const VertexForest& forest = helmet.forest();
    const std::uint32_t vertexCount = forest.vertexCount;
    const std::uint32_t splitCount = static_cast<std::uint32_t>(forest.splits.size());
    const std::uint32_t finestFaceCount = helmet.mesh().binding().finestFaceCount;

    // A real view of the helmet at origin: camera on +Z, tiny budget + cull OFF to push toward full
    // detail (but NOT assumed — checked below).
    const Mat4 world = Mat4::identity();
    const Vec3 cam{0.0f, 0.0f, 3.0f};
    auto lookAtProj = [](const Vec3& c)
    {
        Mat4 v = Mat4::identity();
        v[0, 3] = -c.x();
        v[1, 3] = -c.y();
        v[2, 3] = -c.z();
        Mat4 p = Mat4::identity();
        p[2, 2] = -1.0f;
        p[3, 2] = -1.0f;
        p[3, 3] = 0.0f;
        return p * v;
    };
    const Mat4 viewProj = lookAtProj(cam);
    constexpr float kVw = 1280.0f;
    constexpr float kVh = 720.0f;
    constexpr float kBudget = 1.0e-6f;
    const bool cull = false;

    const VdpmViewParams view =
        makeVdpmViewParams(world, cam, 1.0f, kVh, kVdpmSilhouetteBoost, cull, 1.0f, 1.0f, 1.0f);
    VdpmRepairParams repair{};
    repair.world = world;
    repair.viewProj = viewProj;
    repair.cameraPos[0] = cam.x();
    repair.cameraPos[1] = cam.y();
    repair.cameraPos[2] = cam.z();
    repair.viewport[0] = kVw;
    repair.viewport[1] = kVh;
    repair.viewport[2] = cull ? 1.0f : 0.0f;

    const HelmetHarness::LifecycleResult gpu = helmet.runtimeLifecycle(view, repair, kBudget);

    // (1) Persistent state invariants + (2) clean B3 failure flags.
    CHECK_NOTHROW(validateFrontInvariants(forest, gpu.active, gpu.refined, gpu.dependents));
    CHECK(gpu.failFlags[0] == 0u); // no refine failure
    CHECK(gpu.failFlags[1] == 0u); // no dependents underflow

    // (3) HOLE-FREE: the CPU foldover + coverage classifiers over the read-back GPU active set find
    // no violation (the P2 contract the whole repair stage exists to satisfy, on the real asset).
    const GpuFrontView gpuView{forest, gpu.active};
    CHECK(test::foldoverCount(gpuView, helmet.verts(), helmet.indices(), world) == 0);
    CHECK(test::coverageFailures(gpuView, helmet.verts(), helmet.indices(), viewProj, cam, world,
                                 kVw, kVh, cull) == 0);

    // (4) BOUNDED: never more than full detail.
    REQUIRE(gpu.emitted.size() % 3 == 0);
    CHECK(gpu.emitted.size() <= 3u * finestFaceCount);
    CHECK_FALSE(gpu.emitted.empty());

    // GUARDED exact-lifecycle case: byte-identity is only legitimate when BOTH fronts independently
    // reach FULL DETAIL (repair is then a no-op and no FP decision can diverge). Don't assume the
    // tiny budget forces it on the real asset — verify it, and otherwise SKIP (Claim A already
    // supplies the structural byte-identity proof).
    std::uint32_t gpuActive = 0;
    for (const std::uint32_t v : gpu.active)
    {
        gpuActive += (v != 0u) ? 1u : 0u;
    }
    ParallelFront cpu = ParallelFront::build(helmet.verts(), helmet.indices(), helmet.collapses());
    std::vector<float> score(splitCount);
    std::vector<std::uint8_t> backface(splitCount);
    for (std::uint32_t s = 0; s < splitCount; ++s)
    {
        const VertexSplit& sp = forest.splits[s];
        const VdpmSplitScore sc = scoreVdpmSplit(view, sp, helmet.verts()[sp.parent].position(),
                                                 helmet.verts()[sp.child].position());
        score[s] = sc.score();
        backface[s] = sc.backface;
    }
    cpu.applyView(score, backface, kBudget, kVdpmCoarsenRatio * kBudget);
    cpu.repairFront(helmet.verts(), world, cam, viewProj, kVw, kVh, cull);
    const std::vector<std::uint32_t> cpuEmit =
        cpu.emitActiveIndices(helmet.verts(), helmet.indices());
    std::uint32_t cpuActive = 0;
    for (std::uint32_t v = 0; v < vertexCount; ++v)
    {
        cpuActive += cpu.active(v) ? 1u : 0u;
    }

    if (gpuActive == vertexCount && cpuActive == vertexCount)
    {
        CHECK(gpu.emitted == cpuEmit); // both full detail ⇒ exact, no FP divergence possible
        WARN("helmet Claim B: full-detail reference reached — exact GPU==CPU lifecycle verified ("
             << gpu.emitted.size() / 3 << " tris)");
    }
    else
    {
        WARN(
            "helmet Claim B: reference NOT full detail (gpu active "
            << gpuActive << "/" << vertexCount << ", cpu active " << cpuActive
            << ") — exact-lifecycle byte match skipped; validity/holes/bounds hold, Claim A covers "
               "structural parity. Emitted GPU "
            << gpu.emitted.size() / 3 << " tris, CPU " << cpuEmit.size() / 3);
    }
}
