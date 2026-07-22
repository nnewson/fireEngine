#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/vdpm_gpu.hpp>

using namespace fire_engine;

// VDPM scene-health REDUCTION ([.][gpu], local only). Drives vdpm_health_reduce.comp directly with
// controlled per-front source buffers (counters / repairControl / roundHistory / failFlags) so each
// health term + the manual 64-bit emitted-index carry can be checked in isolation — the manager
// wires the same shader over live fronts, so this pins the reduce logic without a full lifecycle.

namespace
{
// One front's controlled health inputs.
struct FrontInput
{
    std::uint32_t emitted{0}; // counters[2]
    std::uint32_t drawMultiplier{1};
    std::uint32_t ancestorFailure{0};   // repairControl[1]
    std::uint32_t fallbackFired{0};     // repairControl[2]
    std::uint32_t failFlag0{0};         // failFlags[0]
    std::uint32_t failFlag1{0};         // failFlags[1]
    std::vector<std::uint32_t> history; // roundHistory (length kVdpmGpuRepairRoundBudget)
    bool repairPresent{true};
};

class ReduceHarness
{
public:
    ReduceHarness()
        : device_(Device::headlessCompute()),
          resources_(device_),
          pipeline_(device_, vdpmHealthReducePipelineConfig()),
          pool_(device_.device(), vk::CommandPoolCreateInfo{
                                      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                      .queueFamilyIndex = device_.graphicsFamily()})
    {
    }

    // Reduce `fronts` and return the scene health.
    VdpmSceneHealthGpu reduce(const std::vector<FrontInput>& fronts)
    {
        const std::uint32_t n = static_cast<std::uint32_t>(fronts.size());
        constexpr std::uint32_t kB = kVdpmGpuRepairRoundBudget;
        // Per-front source regions packed into one host-visible BDA buffer each.
        Resources::MappedBufferSet counters =
            resources_.createMappedDeviceAddressBuffers(n * 3 * sizeof(std::uint32_t));
        Resources::MappedBufferSet control =
            resources_.createMappedDeviceAddressBuffers(n * 4 * sizeof(std::uint32_t));
        Resources::MappedBufferSet history =
            resources_.createMappedDeviceAddressBuffers(n * kB * sizeof(std::uint32_t));
        Resources::MappedBufferSet fail =
            resources_.createMappedDeviceAddressBuffers(n * 2 * sizeof(std::uint32_t));
        Resources::MappedBufferSet jobs =
            resources_.createMappedDeviceAddressBuffers(n * sizeof(VdpmHealthJobGpu));
        Resources::MappedBufferSet scene =
            resources_.createMappedDeviceAddressBuffers(sizeof(VdpmSceneHealthGpu));

        auto* cPtr = reinterpret_cast<std::uint32_t*>(counters.mapped[0].data());
        auto* rcPtr = reinterpret_cast<std::uint32_t*>(control.mapped[0].data());
        auto* rhPtr = reinterpret_cast<std::uint32_t*>(history.mapped[0].data());
        auto* ffPtr = reinterpret_cast<std::uint32_t*>(fail.mapped[0].data());
        std::memset(cPtr, 0, n * 3 * sizeof(std::uint32_t));
        std::memset(rcPtr, 0, n * 4 * sizeof(std::uint32_t));
        std::memset(rhPtr, 0, n * kB * sizeof(std::uint32_t));
        std::memset(ffPtr, 0, n * 2 * sizeof(std::uint32_t));

        const std::uint64_t cBase = resources_.bufferAddress(counters.buffers[0]);
        const std::uint64_t rcBase = resources_.bufferAddress(control.buffers[0]);
        const std::uint64_t rhBase = resources_.bufferAddress(history.buffers[0]);
        const std::uint64_t ffBase = resources_.bufferAddress(fail.buffers[0]);
        for (std::uint32_t i = 0; i < n; ++i)
        {
            const FrontInput& f = fronts[i];
            cPtr[i * 3 + 2] = f.emitted;
            rcPtr[i * 4 + 1] = f.ancestorFailure;
            rcPtr[i * 4 + 2] = f.fallbackFired;
            ffPtr[i * 2 + 0] = f.failFlag0;
            ffPtr[i * 2 + 1] = f.failFlag1;
            for (std::uint32_t r = 0; r < kB && r < f.history.size(); ++r)
            {
                rhPtr[i * kB + r] = f.history[r];
            }
            const VdpmHealthJobGpu job{
                .countersAddress = cBase + i * 3 * sizeof(std::uint32_t),
                .repairControlAddress = rcBase + i * 4 * sizeof(std::uint32_t),
                .roundHistoryAddress = rhBase + i * kB * sizeof(std::uint32_t),
                .failFlagsAddress = ffBase + i * 2 * sizeof(std::uint32_t),
                .roundBudget = kB,
                .drawMultiplier = f.drawMultiplier,
                .repairPresent = f.repairPresent ? 1u : 0u,
                .pad = 0};
            std::memcpy(jobs.mapped[0].data() + i * sizeof(job), &job, sizeof(job));
        }
        // Prefill the scene buffer with 0xFF to prove the shader FULLY overwrites it.
        std::memset(scene.mapped[0].data(), 0xFF, sizeof(VdpmSceneHealthGpu));

        auto cmds = device_.device().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{.commandPool = *pool_,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1});
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        const VdpmHealthReducePush push{.jobsAddress = resources_.bufferAddress(jobs.buffers[0]),
                                        .sceneHealthAddress =
                                            resources_.bufferAddress(scene.buffers[0]),
                                        .jobCount = n,
                                        .pad = 0};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_.pipeline());
        cmd.pushConstants<VdpmHealthReducePush>(pipeline_.pipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch(1, 1, 1);
        cmd.end();
        const vk::CommandBufferSubmitInfo ci{.commandBuffer = *cmd};
        const vk::SubmitInfo2 s{.commandBufferInfoCount = 1, .pCommandBufferInfos = &ci};
        const vk::raii::Fence fence(device_.device(), vk::FenceCreateInfo{});
        device_.graphicsQueue().submit2(s, *fence);
        REQUIRE(device_.device().waitForFences(*fence, vk::True,
                                               std::numeric_limits<std::uint64_t>::max()) ==
                vk::Result::eSuccess);
        VdpmSceneHealthGpu out{};
        std::memcpy(&out, scene.mapped[0].data(), sizeof(out));
        return out;
    }

private:
    Device device_;
    Resources resources_;
    ComputePipeline pipeline_;
    vk::raii::CommandPool pool_;
};

std::vector<std::uint32_t> markedThenClean(std::uint32_t marked)
{
    std::vector<std::uint32_t> h(kVdpmGpuRepairRoundBudget, 0u);
    for (std::uint32_t r = 0; r < marked && r < h.size(); ++r)
    {
        h[r] = 1u;
    }
    return h;
}
} // namespace

TEST_CASE("VDPM health reduce: mixed repair, multipliers, flags", "[.][gpu]")
{
    ReduceHarness harness;

    SECTION("mixed full / no-repair fronts + submitted-draw weighting")
    {
        // Front 0: full repair, 100 emitted ×2 draws, 3 marked rounds. Front 1: no repair, 50
        // emitted ×1, a B3 fail flag. Front 2: full repair, 10 emitted ×3, 0 marked, fallback
        // fired.
        const VdpmSceneHealthGpu h = harness.reduce({
            FrontInput{.emitted = 100, .drawMultiplier = 2, .history = markedThenClean(3)},
            FrontInput{.emitted = 50, .failFlag0 = 1, .repairPresent = false},
            FrontInput{.emitted = 10,
                       .drawMultiplier = 3,
                       .fallbackFired = 1,
                       .history = markedThenClean(0)},
        });
        CHECK(h.emittedIndexTotalLo == 100u * 2 + 50u + 10u * 3); // 280
        CHECK(h.emittedIndexTotalHi == 0u);
        CHECK(h.emittedIndexOverflow == 0u);
        CHECK(h.repairFronts == 2u);    // fronts 0 + 2 (front 1 repairPresent=false)
        CHECK(h.maxMarkedRounds == 3u); // front 0
        CHECK(h.sumMarkedRounds == 3u); // 3 + 0
        CHECK(h.fallbackFronts == 1u);  // front 2
        CHECK(h.nonCleanPrefixFronts == 0u);
        CHECK(h.ancestorFailureFronts == 0u);
        CHECK(h.failFlagFronts == 1u); // front 1 (failFlags read for EVERY front)
    }

    SECTION("dirty history prefix (marked round after a clean one) is flagged")
    {
        std::vector<std::uint32_t> dirty(kVdpmGpuRepairRoundBudget, 0u);
        dirty[0] = 1u; // marked
        dirty[1] = 0u; // clean
        dirty[2] = 1u; // marked AFTER clean → non-clean prefix
        const VdpmSceneHealthGpu h = harness.reduce({FrontInput{.history = dirty}});
        CHECK(h.nonCleanPrefixFronts == 1u);
        CHECK(h.maxMarkedRounds ==
              1u); // LEADING run only (round 0); the round-2 mark is the anomaly
    }

    SECTION("ancestor failure + fallback are counted per front")
    {
        const VdpmSceneHealthGpu h = harness.reduce({
            FrontInput{.ancestorFailure = 1, .history = markedThenClean(1)},
            FrontInput{.fallbackFired = 1, .history = markedThenClean(2)},
        });
        CHECK(h.ancestorFailureFronts == 1u);
        CHECK(h.fallbackFronts == 1u);
        CHECK(h.repairFronts == 2u);
    }

    SECTION("64-bit emitted-index carry — accumulation across fronts")
    {
        // Two fronts each emit 0x80000000 → 0x1_0000_0000: lo wraps to 0, hi carries to 1.
        const VdpmSceneHealthGpu h = harness.reduce({
            FrontInput{.emitted = 0x80000000u, .repairPresent = false},
            FrontInput{.emitted = 0x80000000u, .repairPresent = false},
        });
        CHECK(h.emittedIndexTotalLo == 0u);
        CHECK(h.emittedIndexTotalHi == 1u);
        CHECK(h.emittedIndexOverflow == 0u);
    }

    SECTION("64-bit carry — a SINGLE weighted product crosses 32 bits")
    {
        // One front: 0x10000000 emitted × 20 draws = 0x1_4000_0000 (> 2^32) — proves the PRODUCT is
        // 64-bit (umulExtended), not a 32-bit multiply that would truncate to 0x40000000.
        const VdpmSceneHealthGpu h = harness.reduce({
            FrontInput{.emitted = 0x10000000u, .drawMultiplier = 20, .repairPresent = false},
        });
        CHECK(h.emittedIndexTotalLo == 0x40000000u);
        CHECK(h.emittedIndexTotalHi == 1u);
        CHECK(h.emittedIndexOverflow == 0u);
    }
    // (The manager never reduces 0 fronts — recordHealthReduction is gated on a non-empty request
    // set — so an empty reduction is not a reachable/tested path; the exact checks above already
    // prove the shader fully overwrites the 0xFF-prefilled output.)
}

TEST_CASE("prepareHealthJob repairPresent matches the runtime repair predicate", "[.][gpu]")
{
    Device device = Device::headlessCompute();
    Resources resources(device);

    // A ZERO-SPLIT faced front: one triangle, no collapses ⇒ splitCount 0 but finestFaceCount 1. It
    // keeps repair buffers but never runs repair, so it must NOT be reported as a repair front.
    const std::vector<Vertex> triVerts{
        Vertex{Vec3{0.0f, 0.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f}},
        Vertex{Vec3{1.0f, 0.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f}},
        Vertex{Vec3{0.0f, 1.0f, 0.0f}, Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f}}};
    const std::vector<std::uint32_t> triIndices{0u, 1u, 2u};
    const VertexForest triForest = buildVertexForest(triVerts, {}); // no collapses ⇒ 0 splits
    REQUIRE(triForest.splits.empty());
    const VdpmGpuMesh triMesh = VdpmGpuMesh::build(resources, triVerts, triIndices, triForest);
    REQUIRE(triMesh.binding().finestFaceCount > 0u); // has a face...
    REQUIRE(triMesh.binding().splitCount == 0u);     // ...but no splits
    const VdpmGpuFront triFront = VdpmGpuFront::buildRuntime(resources, triMesh);
    CHECK(triFront.prepareHealthJob(0, 1).repairPresent == 0u); // splitCount 0 ⇒ no repair

    // A normal multi-split front DOES repair.
    std::vector<Vertex> sv;
    std::vector<std::uint32_t> si;
    constexpr float kPi = 3.14159265f;
    constexpr int kRings = 8;
    constexpr int kSegs = 12;
    for (int r = 0; r <= kRings; ++r)
    {
        const float lat = kPi * ((static_cast<float>(r) / kRings) - 0.5f);
        for (int s = 0; s <= kSegs; ++s)
        {
            const float lon = 2.0f * kPi * static_cast<float>(s) / kSegs;
            const Vec3 n{std::cos(lat) * std::cos(lon), std::sin(lat),
                         std::cos(lat) * std::sin(lon)};
            sv.push_back(Vertex{n, Colour3{}, n, Vec2{0.0f, 0.0f}});
        }
    }
    constexpr int kStride = kSegs + 1;
    for (int r = 0; r < kRings; ++r)
    {
        for (int s = 0; s < kSegs; ++s)
        {
            const auto a = static_cast<std::uint32_t>((r * kStride) + s);
            const auto b = static_cast<std::uint32_t>((r * kStride) + s + 1);
            const auto c = static_cast<std::uint32_t>(((r + 1) * kStride) + s);
            const auto d = static_cast<std::uint32_t>(((r + 1) * kStride) + s + 1);
            si.insert(si.end(), {a, b, d, a, d, c});
        }
    }
    const QuadricSimplifier simp;
    const auto collapses = simp.collapseSequence(sv, si);
    REQUIRE_FALSE(collapses.empty());
    const VertexForest sForest = buildVertexForest(sv, collapses);
    const VdpmGpuMesh sMesh = VdpmGpuMesh::build(resources, sv, si, sForest);
    const VdpmGpuFront sFront = VdpmGpuFront::buildRuntime(resources, sMesh);
    CHECK(sFront.prepareHealthJob(0, 1).repairPresent == 1u);
}
