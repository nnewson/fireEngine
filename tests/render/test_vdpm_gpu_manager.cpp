#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_gpu_registry.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/vdpm_gpu_manager.hpp>

using namespace fire_engine;

// GPU-driven VDPM manager (rendering-spine #3, Stage B5b-1). Tagged [.][gpu] — local only (CI has
// no ICD). Exercises the device-bound half of the registration seam: registerMesh / createFront
// return valid generational handles, stale handles resolve to null, and recordRequests records the
// full front lifecycle for a live request with zero validation errors.

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

} // namespace

TEST_CASE("VdpmGpuManager registers meshes/fronts and records a frame", "[.][gpu]")
{
    Device device = Device::headlessCompute();
    Resources resources(device);
    VdpmGpuManager manager(device, resources);
    REQUIRE(manager.available());

    const Mesh m = uvSphere(16, 20);
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    REQUIRE_FALSE(collapses.empty());

    const VdpmMeshHandle mesh = manager.registerMesh(m.verts, m.indices, collapses);
    REQUIRE(mesh != NullVdpmMesh);

    const VdpmFrontHandle front = manager.createFront(mesh);
    REQUIRE(front != NullVdpmFront);

    // A live front resolves its draw-consumed ring buffers; a null / stale handle resolves to null.
    REQUIRE(manager.frontIndexBuffer(front, 0) != NullBuffer);
    REQUIRE(manager.frontIndirectBuffer(front, 0) != NullBuffer);
    REQUIRE(manager.frontIndexBuffer(NullVdpmFront, 0) == NullBuffer);
    const VdpmFrontHandle stale = makeHandle<VdpmFrontHandle>(handleIndex(front), 99);
    REQUIRE(manager.frontIndexBuffer(stale, 0) == NullBuffer);

    // resolveDrawBuffers (the B5b-2 draw switch): a live front yields a non-null index + indirect
    // pair, but a stale / null handle is an INVARIANT VIOLATION (a GPU-backed draw carries
    // indexCount 0, so a silent miss would issue a zero-count draw) and throws — never resolves to
    // null.
    const VdpmGpuManager::DrawBuffers db = manager.resolveDrawBuffers(front, 0);
    REQUIRE(db.index != NullBuffer);
    REQUIRE(db.indirect != NullBuffer);
    REQUIRE_THROWS_AS(manager.resolveDrawBuffers(stale, 0), std::logic_error);
    REQUIRE_THROWS_AS(manager.resolveDrawBuffers(NullVdpmFront, 0), std::logic_error);

    // createFront on a null / bogus mesh yields a null front (CPU fallback).
    REQUIRE(manager.createFront(NullVdpmMesh) == NullVdpmFront);

    // Record one frame at a tiny budget with cull OFF (no split is zeroed for back-facing) so the
    // front refines to full detail, and read back the GPU-written indirect command. Cull-off + a
    // budget far below every channel score makes the front unambiguously off every threshold, so
    // the GPU emit equals the CPU oracle's emit EXACTLY (integer scheduling + a hole-free no-op
    // repair leave no FP-boundary divergence). The expected index count is derived from the CPU
    // lifecycle below (which naturally accounts for welded seam/pole degenerates), and all five
    // indirect fields must be the canonical one-instance draw. A clean submit with validation
    // layers on also proves 0-VUID.
    VdpmWorkRequest req;
    req.front = front;
    req.world = Mat4::identity();
    req.rasterBackfaceCulling = false;
    const std::array<VdpmWorkRequest, 1> requests{req};
    const VdpmFrameGlobals globals{.viewProj = lookAtProj(Vec3{0.0f, 0.0f, 3.0f}),
                                   .cameraPos = Vec3{0.0f, 0.0f, 3.0f},
                                   .projScaleY = 1.0f,
                                   .viewportWidth = 1280.0f,
                                   .viewportHeight = 720.0f,
                                   .pixelBudget = 1e-6f,
                                   .frameIndex = 0};

    // The CPU oracle emit for this exact request + globals (the same makeVdpmViewParams inputs the
    // manager derives, the same score → applyView → repairFront → emitActiveIndices lifecycle). Its
    // index count is the exact full-detail target the GPU indirect command must match.
    const VertexForest forest = buildVertexForest(m.verts, collapses);
    ParallelFront cpu = ParallelFront::build(m.verts, m.indices, collapses);
    const VdpmViewParams view =
        makeVdpmViewParams(req.world, globals.cameraPos, globals.projScaleY, globals.viewportHeight,
                           kVdpmSilhouetteBoost, req.rasterBackfaceCulling, req.uvScale,
                           req.normalScale, req.tangentScale);
    std::vector<float> scalar(forest.splits.size());
    std::vector<std::uint8_t> backface(forest.splits.size());
    for (std::size_t s = 0; s < forest.splits.size(); ++s)
    {
        const VertexSplit& sp = forest.splits[s];
        const VdpmSplitScore sc =
            scoreVdpmSplit(view, sp, m.verts[sp.parent].position(), m.verts[sp.child].position());
        scalar[s] = sc.score();
        backface[s] = sc.backface;
    }
    cpu.applyView(scalar, backface, globals.pixelBudget, kVdpmCoarsenRatio * globals.pixelBudget);
    cpu.repairFront(m.verts, req.world, globals.cameraPos, globals.viewProj, globals.viewportWidth,
                    globals.viewportHeight, req.rasterBackfaceCulling);
    const std::vector<std::uint32_t> cpuEmit = cpu.emitActiveIndices(m.verts, m.indices);
    REQUIRE(cpuEmit.size() % 3 == 0);
    REQUIRE_FALSE(cpuEmit.empty());

    vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});
    const vk::CommandBufferAllocateInfo ai{
        .commandPool = *pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
    auto cmds = device.device().allocateCommandBuffers(ai);
    vk::raii::CommandBuffer& cmd = cmds[0];
    cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    manager.recordRequests(*cmd, requests, globals);

    // Copy the GPU indirect command (5 uint32) to a host-visible readback. The indirect buffer is
    // compute-written (finalize) and clear-written, so the source scope is compute|clear.
    constexpr std::uint32_t kIndirectWords =
        sizeof(DrawIndexedIndirectCommand) / sizeof(std::uint32_t);
    constexpr vk::DeviceSize kIndirectBytes = kIndirectWords * sizeof(std::uint32_t);
    const Resources::MappedBufferSet host = resources.createMappedReadbackBuffers(kIndirectBytes);
    const vk::MemoryBarrier2 toCopy{
        .srcStageMask =
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eClear,
        .srcAccessMask =
            vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toCopy});
    cmd.copyBuffer(resources.vulkanBuffer(manager.frontIndirectBuffer(front, 0)),
                   resources.vulkanBuffer(host.buffers[0]), vk::BufferCopy{.size = kIndirectBytes});
    const vk::MemoryBarrier2 toHost{
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
    cmd.end();

    const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
    const vk::SubmitInfo2 submit{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo};
    const vk::raii::Fence fence(device.device(), vk::FenceCreateInfo{});
    device.graphicsQueue().submit2(submit, *fence);
    REQUIRE(device.device().waitForFences(*fence, vk::True, UINT64_MAX) == vk::Result::eSuccess);

    std::array<std::uint32_t, kIndirectWords> ind{};
    std::memcpy(ind.data(), host.mapped[0].data(), sizeof(ind));
    // indexCount, instanceCount, firstIndex, vertexOffset, firstInstance.
    REQUIRE(ind[0] == static_cast<std::uint32_t>(cpuEmit.size())); // exact CPU-oracle full detail
    REQUIRE(ind[1] == 1u);
    REQUIRE(ind[2] == 0u);
    REQUIRE(ind[3] == 0u);
    REQUIRE(ind[4] == 0u);
}

// ============================================================================================
// Batched (stage-major) recordRequests == per-front recordFrame — BIT-EXACT ([.][gpu], front-
// batching arc, adj 6). The manager's dispatch(N)-per-stage batched path must produce
// byte-identical front state + draw output to N independent recordFrame calls with the SAME derived
// inputs. The batched fronts live in the manager (read via frontForTest); the reference fronts are
// driven directly with the SAME both-kernel path and identical (deterministic) mesh GPU data, so
// any divergence is a batching bug (job mis-indexing, compaction, a missing/coarse barrier,
// chunking, or a growth hazard), not a scoring/kernel difference — those are cross-checked
// elsewhere.
// ============================================================================================
namespace
{
// One collapse sequence for a mesh (empty ⇒ a zero-split front).
[[nodiscard]] std::vector<MeshCollapse> collapsesOf(const Mesh& m)
{
    const QuadricSimplifier simp;
    return simp.collapseSequence(m.verts, m.indices);
}

// A parallel batched/reference front pair over ONE mesh spec. `indices` may be degenerated (all 0)
// to force finestFaceCount == 0 (an apply-only front: in the apply batch, out of the repair batch).
struct XCheckEntry
{
    // The per-request instance params (distinct across entries so identical scores can't mask a
    // job mis-index).
    Mat4 world;
    bool cull{true};
    float uvScale{1.0f};
    float normalScale{1.0f};
    float tangentScale{1.0f};
    // Sizes (known from the reference forest, used to size every state readback).
    std::uint32_t vertexCount{0};
    std::uint32_t splitCount{0};
    VdpmFrontHandle batched{NullVdpmFront};
    std::size_t refIndex{0}; // into the runner's reference-front vector
};

class BatchXCheckRunner
{
public:
    BatchXCheckRunner()
        : device_(Device::headlessCompute()),
          resources_(device_),
          scorePipeline_(device_, vdpmScorePipelineConfig()),
          refinePipelines_(device_),
          repairPipelines_(device_),
          emitPipelines_(device_),
          applyKernel_(device_),
          repairKernel_(device_),
          manager_(device_, resources_),
          pool_(device_.device(), vk::CommandPoolCreateInfo{
                                      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                      .queueFamilyIndex = device_.graphicsFamily()})
    {
    }

    [[nodiscard]] static bool supported()
    {
        Device d = Device::headlessCompute();
        return VdpmApplyKernel::deviceSupported(d) && VdpmRepairKernel::deviceSupported(d);
    }

    // Register a parallel pair. `degenerate` forces finestFaceCount==0 (apply-only). Returns the
    // entry index.
    std::size_t add(const Mesh& m, const Mat4& world, bool cull, float uvScale, float normalScale,
                    float tangentScale, bool degenerate)
    {
        const std::vector<MeshCollapse> collapses = collapsesOf(m);
        const std::vector<std::uint32_t> indices =
            degenerate ? std::vector<std::uint32_t>(m.indices.size(), 0u) : m.indices;
        const VertexForest forest = buildVertexForest(m.verts, collapses);

        // Batched side (in the manager).
        const VdpmMeshHandle meshB = manager_.registerMesh(m.verts, indices, collapses);
        REQUIRE(meshB != NullVdpmMesh);
        const VdpmFrontHandle frontB = manager_.createFront(meshB);
        REQUIRE(frontB != NullVdpmFront);

        // Reference side (owned directly; identical deterministic GPU data).
        refMeshes_.push_back(VdpmGpuMesh::build(resources_, m.verts, indices, forest));
        refFronts_.push_back(VdpmGpuFront::buildRuntime(resources_, refMeshes_.back()));

        entries_.push_back(
            XCheckEntry{.world = world,
                        .cull = cull,
                        .uvScale = uvScale,
                        .normalScale = normalScale,
                        .tangentScale = tangentScale,
                        .vertexCount = static_cast<std::uint32_t>(forest.vertexCount),
                        .splitCount = static_cast<std::uint32_t>(forest.splits.size()),
                        .batched = frontB,
                        .refIndex = refFronts_.size() - 1});
        return entries_.size() - 1;
    }

    // Force the batched chunk cap (0 ⇒ device cap) so the chunked advanced-BDA dispatch is
    // exercised.
    void setChunkCap(std::uint32_t cap)
    {
        manager_.setTestGroupCapOverride(cap);
    }

    // Drive frame `fi` for the entries named by `order` (a permutation/subset of entry indices):
    // batched via manager.recordRequests, reference via per-front recordFrame with the SAME derived
    // inputs, then assert every front's state + draw output bit-exact.
    void runAndCompare(std::uint32_t fi, std::span<const std::size_t> order, float pixelBudget,
                       const Vec3& cameraPos)
    {
        vk::raii::CommandBuffer cmd = begin();
        recordBoth(*cmd, fi, order, pixelBudget, cameraPos);
        submit(cmd);
        for (const std::size_t e : order)
        {
            compareEntry(e, fi);
        }
    }

    // Record TWO batches into ONE command buffer, submitted ONCE: a small batch (frame slot
    // `slotSmall`, budget `budgetSmall`) that fills job array v1, then a superset (slot
    // `slotSuper`, budget `budgetSuper`) that GROWS the arrays to v2. When the grown batch is
    // recorded, the small batch's dispatch is already RECORDED-BUT-NOT-SUBMITTED against v1 — so
    // v1's buffer must survive the reallocation (Resources never frees it). DIFFERENT slots +
    // budgets make the two batches non-idempotent: the small batch's slot-`slotSmall` emit proves
    // v1's dispatch read the right jobs, the superset's slot-`slotSuper` emit proves the grown
    // array — each checked at its own slot (the per-front state, single-buffered, ends at the
    // superset's; the reference mirrors both passes, so it matches). `superset` must contain every
    // `small` entry.
    void runGrowthInOneSubmit(std::span<const std::size_t> small, std::uint32_t slotSmall,
                              float budgetSmall, std::span<const std::size_t> superset,
                              std::uint32_t slotSuper, float budgetSuper, const Vec3& cameraPos)
    {
        vk::raii::CommandBuffer cmd = begin();
        recordBoth(*cmd, slotSmall, small, budgetSmall, cameraPos); // job array v1, slot slotSmall
        recordBoth(*cmd, slotSuper, superset, budgetSuper,
                   cameraPos); // GROWS to v2, slot slotSuper
        submit(cmd);
        for (const std::size_t e : small)
        {
            compareEntry(e, slotSmall);
        }
        for (const std::size_t e : superset)
        {
            compareEntry(e, slotSuper);
        }
    }

private:
    // Record both sides (batched via the manager, reference via per-front recordFrame with
    // identical derived inputs) for `order` into `cmd`. No submit.
    void recordBoth(vk::CommandBuffer cmd, std::uint32_t fi, std::span<const std::size_t> order,
                    float pixelBudget, const Vec3& cameraPos)
    {
        const float coarsenBudget = kVdpmCoarsenRatio * pixelBudget;
        const VdpmFrameGlobals globals{.viewProj = lookAtProj(cameraPos),
                                       .cameraPos = cameraPos,
                                       .projScaleY = 1.0f,
                                       .viewportWidth = 1280.0f,
                                       .viewportHeight = 720.0f,
                                       .pixelBudget = pixelBudget,
                                       .frameIndex = fi};
        std::vector<VdpmWorkRequest> requests;
        requests.reserve(order.size());
        for (const std::size_t e : order)
        {
            const XCheckEntry& en = entries_[e];
            VdpmWorkRequest req;
            req.front = en.batched;
            req.world = en.world;
            req.rasterBackfaceCulling = en.cull;
            req.uvScale = en.uvScale;
            req.normalScale = en.normalScale;
            req.tangentScale = en.tangentScale;
            requests.push_back(req);
        }
        manager_.recordRequests(cmd, requests, globals); // batched path (>= 1 front, no profiler)
        for (const std::size_t e : order)
        {
            const XCheckEntry& en = entries_[e];
            const VdpmViewParams view = makeVdpmViewParams(
                en.world, globals.cameraPos, globals.projScaleY, globals.viewportHeight,
                kVdpmSilhouetteBoost, en.cull, en.uvScale, en.normalScale, en.tangentScale);
            VdpmRepairParams repair{};
            repair.world = en.world;
            repair.viewProj = globals.viewProj;
            repair.cameraPos[0] = globals.cameraPos.x();
            repair.cameraPos[1] = globals.cameraPos.y();
            repair.cameraPos[2] = globals.cameraPos.z();
            repair.viewport[0] = globals.viewportWidth;
            repair.viewport[1] = globals.viewportHeight;
            repair.viewport[2] = en.cull ? 1.0f : 0.0f;
            refFronts_[en.refIndex].recordFrame(
                cmd, scorePipeline_, refinePipelines_, repairPipelines_, emitPipelines_, resources_,
                fi, view, repair, pixelBudget, coarsenBudget, kVdpmGpuRepairRoundBudget,
                &applyKernel_, &repairKernel_, nullptr);
        }
    }

    vk::raii::CommandBuffer begin()
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
        REQUIRE(device_.device().waitForFences(*fence, vk::True, UINT64_MAX) ==
                vk::Result::eSuccess);
    }
    // Copy `words` uint32 from a device buffer (written in a PRIOR, already-fenced submit) to host.
    std::vector<std::uint32_t> read(BufferHandle h, std::uint32_t words)
    {
        if (words == 0 || h == NullBuffer)
        {
            return {};
        }
        const Resources::MappedBufferSet host =
            resources_.createMappedReadbackBuffers(words * sizeof(std::uint32_t));
        vk::raii::CommandBuffer cmd = begin();
        cmd.copyBuffer(resources_.vulkanBuffer(h), resources_.vulkanBuffer(host.buffers[0]),
                       vk::BufferCopy{.size = words * sizeof(std::uint32_t)});
        const vk::MemoryBarrier2 toHost{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
        submit(cmd);
        std::vector<std::uint32_t> out(words);
        std::memcpy(out.data(), host.mapped[0].data(), words * sizeof(std::uint32_t));
        return out;
    }
    void compareEntry(std::size_t e, std::uint32_t fi)
    {
        const XCheckEntry& en = entries_[e];
        const VdpmGpuFront* b = manager_.frontForTest(en.batched);
        REQUIRE(b != nullptr);
        const VdpmGpuFront& r = refFronts_[en.refIndex];
        const std::uint32_t v = en.vertexCount;
        const std::uint32_t s = en.splitCount;

        // Persistent front state: active/dependents per vertex, refined/required per split,
        // failFlags (2). BIT-EXACT — the whole point of the cross-check.
        CHECK(read(b->activeStateBuffer(), v) == read(r.activeStateBuffer(), v));
        CHECK(read(b->refinedStateBuffer(), s) == read(r.refinedStateBuffer(), s));
        CHECK(read(b->dependentsStateBuffer(), v) == read(r.dependentsStateBuffer(), v));
        CHECK(read(b->requiredStateBuffer(), s) == read(r.requiredStateBuffer(), s));
        CHECK(read(b->failFlagsBuffer(), 2) == read(r.failFlagsBuffer(), 2));
        const std::vector<std::uint32_t> bff = read(b->failFlagsBuffer(), 2);
        CHECK(bff[0] == 0u); // no refine failure
        CHECK(bff[1] == 0u); // no dependents underflow

        // Draw output: the 5-word indirect command, then indexCount emitted indices.
        constexpr std::uint32_t kIndirectWords =
            sizeof(DrawIndexedIndirectCommand) / sizeof(std::uint32_t);
        const std::vector<std::uint32_t> bInd = read(b->emittedIndirectBuffer(fi), kIndirectWords);
        const std::vector<std::uint32_t> rInd = read(r.emittedIndirectBuffer(fi), kIndirectWords);
        CHECK(bInd == rInd);
        REQUIRE(bInd.size() == kIndirectWords);
        CHECK(bInd[1] == 1u); // one instance
        const std::uint32_t indexCount = bInd[0];
        CHECK(read(b->emittedIndicesBuffer(fi), indexCount) ==
              read(r.emittedIndicesBuffer(fi), indexCount));
    }

    Device device_;
    Resources resources_;
    ComputePipeline scorePipeline_;
    VdpmRefinePipelines refinePipelines_;
    VdpmRepairPipelines repairPipelines_;
    VdpmEmitPipelines emitPipelines_;
    VdpmApplyKernel applyKernel_;
    VdpmRepairKernel repairKernel_;
    VdpmGpuManager manager_;
    vk::raii::CommandPool pool_;
    std::vector<VdpmGpuMesh> refMeshes_;
    std::vector<VdpmGpuFront> refFronts_;
    std::vector<XCheckEntry> entries_;
};

Mat4 translate(float x, float y, float z)
{
    Mat4 m = Mat4::identity();
    m[0, 3] = x;
    m[1, 3] = y;
    m[2, 3] = z;
    return m;
}
} // namespace

TEST_CASE("VdpmGpuManager batched recordRequests == per-front recordFrame (bit-exact)", "[.][gpu]")
{
    if (!BatchXCheckRunner::supported())
    {
        return; // device lacks a persistent kernel — batching never engages
    }

    const Mesh sphere = uvSphere(16, 20);
    const Vec3 cam{0.0f, 0.0f, 3.0f};

    SECTION("distinct params + compaction (full / apply-only / zero-split) + shuffled order")
    {
        BatchXCheckRunner run;
        // Two FULL fronts (distinct worlds ⇒ distinct scores ⇒ a job mis-index would diverge), one
        // APPLY-ONLY front (degenerate indices ⇒ finestFaceCount 0 ⇒ in the apply batch, out of the
        // repair batch: Nr < Na), one ZERO-SPLIT front (no collapses ⇒ in NEITHER batch, still
        // emitted). Distinct culls/material scales too.
        const std::size_t full0 =
            run.add(sphere, translate(0.0f, 0.0f, 0.0f), true, 1.0f, 1.0f, 1.0f, false);
        const std::size_t full1 =
            run.add(sphere, translate(0.4f, 0.1f, 0.0f), false, 2.0f, 0.5f, 1.5f, false);
        const std::size_t applyOnly =
            run.add(sphere, translate(-0.3f, 0.2f, 0.0f), true, 1.0f, 1.0f, 1.0f, true);
        const Mesh tri{
            {Vertex{Vec3{0.0f, 0.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f}},
             Vertex{Vec3{1.0f, 0.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f}},
             Vertex{Vec3{0.0f, 1.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f}}},
            {0u, 1u, 2u}};
        const std::size_t zeroSplit =
            run.add(tri, translate(0.0f, -0.5f, 0.0f), true, 1.0f, 1.0f, 1.0f, false);

        // Shuffled order (not entry order) — the compaction must key each job to its own front.
        const std::array<std::size_t, 4> order{full1, zeroSplit, applyOnly, full0};
        run.runAndCompare(0, order, /*pixelBudget=*/1e-4f, cam);
    }

    SECTION("N = 1 batched (single job) matches per-front")
    {
        BatchXCheckRunner run;
        const std::size_t only =
            run.add(sphere, translate(0.0f, 0.0f, 0.0f), false, 1.0f, 1.0f, 1.0f, false);
        const std::array<std::size_t, 1> order{only};
        run.runAndCompare(0, order, 1e-4f, cam);
    }

    SECTION("job-array growth across two frame slots with different budgets, in one submit")
    {
        BatchXCheckRunner run;
        std::vector<std::size_t> all;
        all.reserve(5);
        for (int i = 0; i < 5; ++i)
        {
            all.push_back(run.add(sphere, translate(0.15f * static_cast<float>(i), 0.0f, 0.0f),
                                  (i % 2) == 0, 1.0f, 1.0f, 1.0f, false));
        }
        // Small batch (slot 0, grows capacity 0→2) THEN the full set (slot 1, grows 2→8) in ONE
        // command buffer / ONE submit. Different slots + budgets ⇒ the two batches do genuinely
        // different work, so the small batch's slot-0 emit (against the pre-growth array v1) and
        // the grown slot-1 emit are BOTH checked bit-exact — proving v1's BDA survived the
        // reallocation and the grown array indexes correctly.
        const std::array<std::size_t, 2> small{all[0], all[1]};
        run.runGrowthInOneSubmit(small, /*slotSmall=*/0, /*budgetSmall=*/1e-4f,
                                 std::span<const std::size_t>{all}, /*slotSuper=*/1,
                                 /*budgetSuper=*/5e-4f, cam);
    }

    SECTION("forced small chunk cap exercises the advanced-BDA chunked dispatch")
    {
        BatchXCheckRunner run;
        std::vector<std::size_t> all;
        all.reserve(5);
        for (int i = 0; i < 5; ++i)
        {
            all.push_back(run.add(sphere, translate(0.15f * static_cast<float>(i), 0.0f, 0.0f),
                                  (i % 2) == 0, 1.0f, 1.0f, 1.0f, false));
        }
        // Cap 2 over 5 full fronts ⇒ apply and repair each dispatch in ⌈5/2⌉ = 3 chunks, advancing
        // the job BDA by firstJob*stride. Bit-exact vs per-front ⇒ the advanced-address dispatch
        // reads the right jobs (the device cap is far too large to ever chunk in practice).
        run.setChunkCap(2);
        run.runAndCompare(0, std::span<const std::size_t>{all}, 1e-4f, cam);
    }
}

// Scene-health readback ring (B5c-1): the delayed readback must reflect the frame
// kMaxFramesInFlight ago and keep the frame slots isolated. Drive frames with a CONSTANT view (⇒
// constant GPU emitted count) but a DISTINCT cpuTriangleSubtotal each frame; once warm, each read =
// subtotal(f-K) + a constant emitted-triangle term, so consecutive valid reads step by exactly the
// per-frame subtotal delta. A wrong delay or cross-slot contamination breaks that constant step.
TEST_CASE("VdpmGpuManager scene-health readback: K-frame delay + slot isolation", "[.][gpu]")
{
    Device device = Device::headlessCompute();
    Resources resources(device);
    VdpmGpuManager manager(device, resources);
    REQUIRE(manager.available());

    const Mesh m = uvSphere(16, 20);
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VdpmMeshHandle mesh = manager.registerMesh(m.verts, m.indices, collapses);
    REQUIRE(mesh != NullVdpmMesh);
    const VdpmFrontHandle front = manager.createFront(mesh);
    REQUIRE(front != NullVdpmFront);

    VdpmWorkRequest req;
    req.front = front;
    req.world = Mat4::identity();
    req.rasterBackfaceCulling = false; // full detail (no back-face zeroing) ⇒ constant emit
    const std::array<VdpmWorkRequest, 1> requests{req};

    const vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});

    constexpr int kFrames = 8;
    constexpr std::uint64_t kStep = 1000;
    std::array<std::int64_t, kFrames> reads{};
    reads.fill(-1);
    for (int f = 0; f < kFrames; ++f)
    {
        const std::uint32_t slot = static_cast<std::uint32_t>(f % kMaxFramesInFlight);
        const VdpmFrameGlobals globals{.viewProj = lookAtProj(Vec3{0.0f, 0.0f, 3.0f}),
                                       .cameraPos = Vec3{0.0f, 0.0f, 3.0f},
                                       .projScaleY = 1.0f,
                                       .viewportWidth = 1280.0f,
                                       .viewportHeight = 720.0f,
                                       .pixelBudget = 1e-6f,
                                       .frameIndex = slot};
        auto cmds = device.device().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{.commandPool = *pool,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1});
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        manager.recordRequests(*cmd, requests, globals);
        manager.recordDiagnosticReadback(*cmd, slot, kStep * static_cast<std::uint64_t>(f + 1));
        cmd.end();
        const vk::CommandBufferSubmitInfo ci{.commandBuffer = *cmd};
        const vk::SubmitInfo2 s{.commandBufferInfoCount = 1, .pCommandBufferInfos = &ci};
        const vk::raii::Fence fence(device.device(), vk::FenceCreateInfo{});
        device.graphicsQueue().submit2(s, *fence);
        REQUIRE(device.device().waitForFences(*fence, vk::True, UINT64_MAX) ==
                vk::Result::eSuccess);

        const VdpmGpuManager::SceneHealth& h = manager.lastSceneHealth();
        if (h.valid)
        {
            reads[static_cast<std::size_t>(f)] = static_cast<std::int64_t>(h.triangleTotal);
        }
    }

    // From K+1 on, both the current and previous reads are valid AND from settled frames; their
    // difference is the per-frame subtotal step (the constant emitted term cancels).
    int checked = 0;
    for (int f = kMaxFramesInFlight + 1; f < kFrames; ++f)
    {
        REQUIRE(reads[static_cast<std::size_t>(f)] >= 0);
        REQUIRE(reads[static_cast<std::size_t>(f - 1)] >= 0);
        CHECK(reads[static_cast<std::size_t>(f)] - reads[static_cast<std::size_t>(f - 1)] ==
              static_cast<std::int64_t>(kStep));
        ++checked;
    }
    CHECK(checked > 0);
}

// B5c-1 draw-count validation: a supplied drawCounts mapping must be a complete 1:1 keyed cover of
// the request set (no zero-count / unknown / duplicate / missing entry) — a silent default-to-1
// would let a filtering mismatch produce plausible-but-false totals.
TEST_CASE("VdpmGpuManager recordRequests rejects a malformed draw-count mapping", "[.][gpu]")
{
    Device device = Device::headlessCompute();
    Resources resources(device);
    VdpmGpuManager manager(device, resources);
    const Mesh m = uvSphere(12, 16);
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VdpmMeshHandle mesh = manager.registerMesh(m.verts, m.indices, collapses);
    const VdpmFrontHandle a = manager.createFront(mesh);
    const VdpmFrontHandle b = manager.createFront(mesh);
    REQUIRE(a != NullVdpmFront);
    REQUIRE(b != NullVdpmFront);
    const VdpmFrontHandle bogus = makeHandle<VdpmFrontHandle>(handleIndex(a), 77);

    auto reqOf = [](VdpmFrontHandle f)
    {
        VdpmWorkRequest r;
        r.front = f;
        r.world = Mat4::identity();
        r.rasterBackfaceCulling = false;
        return r;
    };
    const std::array<VdpmWorkRequest, 2> requests{reqOf(a), reqOf(b)};
    const VdpmFrameGlobals globals{.viewProj = lookAtProj(Vec3{0.0f, 0.0f, 3.0f}),
                                   .cameraPos = Vec3{0.0f, 0.0f, 3.0f},
                                   .projScaleY = 1.0f,
                                   .viewportWidth = 1280.0f,
                                   .viewportHeight = 720.0f,
                                   .pixelBudget = 1e-4f,
                                   .frameIndex = 0};

    const vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});
    auto cmds = device.device().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
        .commandPool = *pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1});
    vk::raii::CommandBuffer& cmd = cmds[0];
    cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    using DC = VdpmFrontDrawCount;
    // zero count
    const std::array<DC, 2> zero{DC{a, 0}, DC{b, 1}};
    REQUIRE_THROWS_AS(manager.recordRequests(*cmd, requests, globals, zero), std::logic_error);
    // unknown front (not in requests)
    const std::array<DC, 3> unknown{DC{a, 1}, DC{b, 1}, DC{bogus, 1}};
    REQUIRE_THROWS_AS(manager.recordRequests(*cmd, requests, globals, unknown), std::logic_error);
    // duplicate
    const std::array<DC, 3> dup{DC{a, 1}, DC{a, 1}, DC{b, 1}};
    REQUIRE_THROWS_AS(manager.recordRequests(*cmd, requests, globals, dup), std::logic_error);
    // missing (b has no entry)
    const std::array<DC, 1> missing{DC{a, 1}};
    REQUIRE_THROWS_AS(manager.recordRequests(*cmd, requests, globals, missing), std::logic_error);
    // a complete valid mapping is accepted
    const std::array<DC, 2> ok{DC{a, 1}, DC{b, 2}};
    REQUIRE_NOTHROW(manager.recordRequests(*cmd, requests, globals, ok));
}

// B5c-1 stale-diagnostics guard: after an inactive gap (no fronts recorded), the readback must NOT
// parse an old slot as fresh — it stays invalid until the ring re-warms with live data.
TEST_CASE("VdpmGpuManager scene-health readback invalidates across an inactive gap", "[.][gpu]")
{
    Device device = Device::headlessCompute();
    Resources resources(device);
    VdpmGpuManager manager(device, resources);
    const Mesh m = uvSphere(12, 16);
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VdpmMeshHandle mesh = manager.registerMesh(m.verts, m.indices, collapses);
    const VdpmFrontHandle front = manager.createFront(mesh);
    REQUIRE(front != NullVdpmFront);
    VdpmWorkRequest req;
    req.front = front;
    req.world = Mat4::identity();
    req.rasterBackfaceCulling = false;
    const std::array<VdpmWorkRequest, 1> requests{req};

    const vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});
    auto activeFrame = [&](int f) -> bool
    {
        const std::uint32_t slot = static_cast<std::uint32_t>(f % kMaxFramesInFlight);
        const VdpmFrameGlobals globals{.viewProj = lookAtProj(Vec3{0.0f, 0.0f, 3.0f}),
                                       .cameraPos = Vec3{0.0f, 0.0f, 3.0f},
                                       .projScaleY = 1.0f,
                                       .viewportWidth = 1280.0f,
                                       .viewportHeight = 720.0f,
                                       .pixelBudget = 1e-6f,
                                       .frameIndex = slot};
        auto cmds = device.device().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{.commandPool = *pool,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1});
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        manager.recordRequests(*cmd, requests, globals);
        manager.recordDiagnosticReadback(*cmd, slot, 1000);
        cmd.end();
        const vk::CommandBufferSubmitInfo ci{.commandBuffer = *cmd};
        const vk::SubmitInfo2 s{.commandBufferInfoCount = 1, .pCommandBufferInfos = &ci};
        const vk::raii::Fence fence(device.device(), vk::FenceCreateInfo{});
        device.graphicsQueue().submit2(s, *fence);
        REQUIRE(device.device().waitForFences(*fence, vk::True, UINT64_MAX) ==
                vk::Result::eSuccess);
        return manager.lastSceneHealth().valid;
    };

    int f = 0;
    for (; f < kMaxFramesInFlight + 2; ++f) // warm the ring
    {
        activeFrame(f);
    }
    REQUIRE(manager.lastSceneHealth().valid);

    // Inactive gap: the Renderer invalidates every slot it doesn't record into.
    for (std::uint32_t s = 0; s < kMaxFramesInFlight; ++s)
    {
        manager.invalidateHealthSlot(s);
    }
    // Resume: the first kMaxFramesInFlight active frames read invalidated slots → invalid; then the
    // slots re-warm and validity returns.
    for (int i = 0; i < kMaxFramesInFlight; ++i, ++f)
    {
        CHECK_FALSE(activeFrame(f)); // reading a slot invalidated during the gap
    }
    CHECK(activeFrame(f)); // ring re-warmed
}

// B5c-1 accounting: the scene-health reduction adds EXACTLY one dispatch + one barrier on top of
// the front-lifecycle totals — the manager owns this increment, so pin it here.
TEST_CASE("VdpmGpuManager: the health stage adds exactly {+1 dispatch, +1 barrier}", "[.][gpu]")
{
    Device device = Device::headlessCompute();
    Resources resources(device);
    VdpmGpuManager manager(device, resources);
    const Mesh m = uvSphere(12, 16);
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    const VdpmMeshHandle mesh = manager.registerMesh(m.verts, m.indices, collapses);
    const VdpmFrontHandle front = manager.createFront(mesh);
    REQUIRE(front != NullVdpmFront);

    // A single unprofiled front routes to the batched path, so its lifecycle cost is
    // analyticBatchedCost — the manager's reported total must be exactly that + the health {1,1}.
    const VdpmGpuFront* f = manager.frontForTest(front);
    REQUIRE(f != nullptr);
    const std::array<VdpmGpuFront::FrontDims, 1> dims{
        {{f->rankCount(), f->faceCount(), f->finestFaceCount()}}};
    const VdpmGpuFront::ComputeCost lifecycle =
        VdpmGpuFront::analyticBatchedCost(dims, resources.maxComputeWorkGroupCountX());

    VdpmWorkRequest req;
    req.front = front;
    req.world = Mat4::identity();
    req.rasterBackfaceCulling = false;
    const std::array<VdpmWorkRequest, 1> requests{req};
    const VdpmFrameGlobals globals{.viewProj = lookAtProj(Vec3{0.0f, 0.0f, 3.0f}),
                                   .cameraPos = Vec3{0.0f, 0.0f, 3.0f},
                                   .projScaleY = 1.0f,
                                   .viewportWidth = 1280.0f,
                                   .viewportHeight = 720.0f,
                                   .pixelBudget = 1e-4f,
                                   .frameIndex = 0}; // no stageProfiler ⇒ batched path

    const vk::raii::CommandPool pool(
        device.device(),
        vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queueFamilyIndex = device.graphicsFamily()});
    auto cmds = device.device().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
        .commandPool = *pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1});
    vk::raii::CommandBuffer& cmd = cmds[0];
    cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    manager.recordRequests(*cmd, requests, globals); // accounting filled synchronously; no submit
    cmd.end();

    CHECK(manager.lastComputeStats().analyticDispatches == lifecycle.dispatches + 1);
    CHECK(manager.lastComputeStats().analyticBarriers == lifecycle.barriers + 1);
}
