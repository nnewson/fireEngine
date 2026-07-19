#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vdpm_parallel.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/compute_pipeline.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/vdpm_gpu.hpp>

using Catch::Approx;
using namespace fire_engine;

// GPU VDPM harness (rendering-spine #3, GPU-driven-front Stage B1). Tagged [.][gpu] so normal CTest
// (and the no-ICD CI runners) SKIP it — it needs a real Vulkan device. Run locally with:
//   ./test_fire_engine "[gpu]"
// from the build dir. It cross-checks the GPU compute port against the pure CPU scoring authority.

TEST_CASE("VDPM GPU: a surface-free compute device initialises", "[.][gpu]")
{
    // The one genuine unknown for Stage B1: MoltenVK offscreen compute with no swapchain. If this
    // throws, diagnose THAT failure (no hidden-window fallback) — MoltenVK supports compute without
    // a surface.
    const Device device = Device::headlessCompute();
    CHECK(*device.device() != nullptr);
    CHECK(*device.physicalDevice() != nullptr);
    CHECK(*device.graphicsQueue() != nullptr); // the shared graphics+compute queue

    // No surface + no present queue in headless mode.
    CHECK(*device.surface() == nullptr);
    CHECK(*device.presentQueue() == nullptr);

    // The selected family really is graphics+compute (the same-queue production path).
    const auto families = device.physicalDevice().getQueueFamilyProperties();
    REQUIRE(device.graphicsFamily() < families.size());
    const vk::QueueFlags flags = families[device.graphicsFamily()].queueFlags;
    CHECK(static_cast<bool>(flags & vk::QueueFlagBits::eGraphics));
    CHECK(static_cast<bool>(flags & vk::QueueFlagBits::eCompute));
}

// The ABI pack helpers are pure CPU field copies, so they run in NORMAL CI ([vdpm], not [gpu]) —
// no device needed. They pin the uploader faithful to the scoring authority, the third leg of the
// "one scoring" contract (oracle / uploader / shader).

TEST_CASE("packVdpmSplit copies the split's cone + errors + IDs", "[vdpm]")
{
    VertexSplit s;
    s.parent = 7;
    s.child = 42;
    s.normalConeAxis = Vec3{0.0f, 1.0f, 0.0f};
    s.normalConeCos = 0.25f;
    s.supportRadius = 1.5f;
    s.error = 2.0f;
    s.uvError = 3.0f;
    s.normalError = 0.5f;
    s.tangentError = 0.75f;

    const VdpmSplitGpu g = packVdpmSplit(s);
    CHECK(g.coneAxisCos[0] == 0.0f);
    CHECK(g.coneAxisCos[1] == 1.0f);
    CHECK(g.coneAxisCos[2] == 0.0f);
    CHECK(g.coneAxisCos[3] == 0.25f); // cos packed into .w
    CHECK(g.supportRadius == 1.5f);
    CHECK(g.error == 2.0f);
    CHECK(g.uvError == 3.0f);
    CHECK(g.normalError == 0.5f);
    CHECK(g.tangentError == 0.75f);
    CHECK(g.parentId == 7u);
    CHECK(g.childId == 42u);
}

TEST_CASE("packVdpmPosition packs xyz into a padded vec4", "[vdpm]")
{
    const VdpmPositionGpu g = packVdpmPosition(Vec3{1.0f, -2.0f, 3.0f});
    CHECK(g.position[0] == 1.0f);
    CHECK(g.position[1] == -2.0f);
    CHECK(g.position[2] == 3.0f);
    CHECK(g.position[3] == 0.0f);
}

TEST_CASE("packVdpmScoreParams images VdpmViewParams column-major with flags + addresses", "[vdpm]")
{
    // A FULLY NON-SYMMETRIC linear part with nine distinct entries, so a row/column transpose in
    // the packer would be caught (a symmetric or diagonal matrix could not). Invertible (det = -3)
    // so the cone stays usable. Translated so the affine term is distinct too.
    Mat4 world = Mat4::identity();
    world[0, 0] = 1.0f;
    world[0, 1] = 2.0f;
    world[0, 2] = 3.0f;
    world[1, 0] = 4.0f;
    world[1, 1] = 5.0f;
    world[1, 2] = 6.0f;
    world[2, 0] = 7.0f;
    world[2, 1] = 8.0f;
    world[2, 2] = 10.0f;
    world[0, 3] = 10.0f;
    world[1, 3] = 20.0f;
    world[2, 3] = 30.0f;
    const Vec3 cam{1.0f, 2.0f, 3.0f};
    const VdpmViewParams v =
        makeVdpmViewParams(world, cam, 1.25f, 1000.0f, 2.0f, true, 0.5f, 0.6f, 0.7f);

    const VdpmScoreParams p = packVdpmScoreParams(v, 0x1111u, 0x2222u, 0x3333u, 99u);

    // GLSL mat3 is column-major: packed column c, row r == worldLinear[r, c]. Assert all NINE
    // entries (transpose-detecting) plus all THREE padding slots.
    const std::array<const float*, 3> cols{p.worldLinearCol0, p.worldLinearCol1, p.worldLinearCol2};
    for (int c = 0; c < 3; ++c)
    {
        CHECK(cols[c][0] == v.worldLinear[0, c]);
        CHECK(cols[c][1] == v.worldLinear[1, c]);
        CHECK(cols[c][2] == v.worldLinear[2, c]);
        CHECK(cols[c][3] == 0.0f); // column padding
    }

    CHECK(p.worldTranslationMinusCamera[0] == v.worldTranslationMinusCamera.x());
    CHECK(p.worldTranslationMinusCamera[1] == v.worldTranslationMinusCamera.y());
    CHECK(p.worldTranslationMinusCamera[2] == v.worldTranslationMinusCamera.z());
    CHECK(p.cameraObj[0] == v.cameraObj.x());

    CHECK(p.worldLengthScale == v.worldLengthScale);
    CHECK(p.facingSign == v.facingSign);
    CHECK(p.projScaleY == v.projScaleY);
    CHECK(p.halfViewport == v.halfViewport); // == viewportHeight/2 = 500
    CHECK(p.silhouetteBoost == v.silhouetteBoost);
    CHECK(p.uvScale == v.uvScale);
    CHECK(p.normalScale == v.normalScale);
    CHECK(p.tangentScale == v.tangentScale);

    CHECK(p.coneUsable == 1u); // flags as uint32, not bool
    CHECK(p.coneCullEnabled == 1u);
    CHECK(p.splitCount == 99u);
    CHECK(p.splitsAddress == 0x1111u);
    CHECK(p.positionsAddress == 0x2222u);
    CHECK(p.outputsAddress == 0x3333u);
}

TEST_CASE("validateVdpmRankRanges accepts a contiguous partition, rejects malformed ones", "[vdpm]")
{
    using R = RankRange;

    SECTION("valid contiguous partition of splitCount")
    {
        const std::array<R, 3> ok{R{0, 4}, R{4, 3}, R{7, 5}}; // covers [0,12)
        REQUIRE_NOTHROW(validateVdpmRankRanges(ok, 12));
    }
    SECTION("empty ranges cover exactly a zero-split mesh")
    {
        REQUIRE_NOTHROW(validateVdpmRankRanges(std::span<const R>{}, 0));
    }
    SECTION("gap between ranks throws")
    {
        const std::array<R, 2> gap{R{0, 4}, R{5, 3}}; // 4 != 5
        REQUIRE_THROWS_AS(validateVdpmRankRanges(gap, 8), std::runtime_error);
    }
    SECTION("overlap between ranks throws")
    {
        const std::array<R, 2> overlap{R{0, 4}, R{3, 3}}; // 4 != 3
        REQUIRE_THROWS_AS(validateVdpmRankRanges(overlap, 7), std::runtime_error);
    }
    SECTION("wrong terminal count throws")
    {
        const std::array<R, 2> partition{R{0, 4}, R{4, 3}}; // covers 7, not 8
        REQUIRE_THROWS_AS(validateVdpmRankRanges(partition, 8), std::runtime_error);
    }
    SECTION("first rank not at offset 0 throws")
    {
        const std::array<R, 1> shifted{R{1, 4}};
        REQUIRE_THROWS_AS(validateVdpmRankRanges(shifted, 4), std::runtime_error);
    }
    SECTION("counts that would wrap under 32-bit arithmetic are still rejected")
    {
        // Rank 0 covers [0, UINT32_MAX); rank 1 starts at UINT32_MAX (offset check passes) and adds
        // 5. A 32-bit accumulator wraps UINT32_MAX + 5 to 4 and would WRONGLY accept splitCount 4;
        // the 64-bit accumulator reaches UINT32_MAX + 5 and rejects. This pins the 64-bit choice.
        const std::array<R, 2> wrapping{
            R{0, std::numeric_limits<std::uint32_t>::max()},
            R{std::numeric_limits<std::uint32_t>::max(), 5},
        };
        REQUIRE_THROWS_AS(validateVdpmRankRanges(wrapping, 4), std::runtime_error);
    }
}

// ============================================================================================
// GPU score harness ([.][gpu], local-only — needs a real Vulkan device). Cross-checks
// shaders/vdpm_score.comp against the CPU scoring authority (scoreVdpmSplit) on a headless device.
// ============================================================================================

namespace
{

// A headless compute device + compute-only Resources + the score pipeline + a command pool. Records
// the dispatch, the compute→transfer→host readback barriers, submits, fence-waits, and returns the
// per-split output. Non-movable (Resources/pipeline hold pointers to `device_`); build one per
// case.
class GpuScorer
{
public:
    GpuScorer()
        : device_(Device::headlessCompute()),
          resources_(device_),
          pipeline_(device_, vdpmScorePipelineConfig()),
          pool_(device_.device(), vk::CommandPoolCreateInfo{
                                      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                      .queueFamilyIndex = device_.graphicsFamily()})
    {
    }
    GpuScorer(const GpuScorer&) = delete;
    GpuScorer& operator=(const GpuScorer&) = delete;
    GpuScorer(GpuScorer&&) = delete;
    GpuScorer& operator=(GpuScorer&&) = delete;
    ~GpuScorer() = default;

    [[nodiscard]] std::vector<VdpmScoreOut>
    score(std::span<const Vertex> verts, const VertexForest& forest, const VdpmViewParams& view)
    {
        const VdpmGpuMesh mesh = VdpmGpuMesh::build(resources_, verts, forest);
        VdpmGpuFront front = VdpmGpuFront::build(resources_, mesh);
        const std::uint32_t n = mesh.splitCount();
        if (n == 0)
        {
            return {};
        }
        const vk::DeviceSize outSize = static_cast<vk::DeviceSize>(n) * sizeof(VdpmScoreOut);
        const Resources::MappedBufferSet readback = resources_.createMappedReadbackBuffers(outSize);

        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        // Slot 0 always (we fence-wait before any reuse). Score dispatch — recordScore adds NO
        // barriers; the consumer (this harness) owns the transfer synchronisation.
        front.recordScore(*cmd, pipeline_, 0, view);
        const vk::Buffer outBuf = resources_.vulkanBuffer(front.outputBuffer());
        // compute shader write → transfer read (the copy).
        recordBufferBarrier(
            *cmd, makeBufferMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader,
                                          vk::AccessFlagBits2::eShaderStorageWrite,
                                          vk::PipelineStageFlagBits2::eCopy,
                                          vk::AccessFlagBits2::eTransferRead, outBuf, 0, outSize));
        const vk::Buffer hostBuf = resources_.vulkanBuffer(readback.buffers[0]);
        cmd.copyBuffer(outBuf, hostBuf, vk::BufferCopy{.size = outSize});
        // transfer write → host read (the host barrier exposes the copy to CPU; coherence only
        // removes the invalidate).
        recordBufferBarrier(*cmd, makeBufferMemoryBarrier(vk::PipelineStageFlagBits2::eCopy,
                                                          vk::AccessFlagBits2::eTransferWrite,
                                                          vk::PipelineStageFlagBits2::eHost,
                                                          vk::AccessFlagBits2::eHostRead, hostBuf,
                                                          0, outSize));
        cmd.end();

        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        const vk::SubmitInfo2 submit{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo};
        const vk::raii::Fence fence(device_.device(), vk::FenceCreateInfo{});
        device_.graphicsQueue().submit2(submit, *fence);
        (void)device_.device().waitForFences(*fence, vk::True,
                                             std::numeric_limits<std::uint64_t>::max());

        std::vector<VdpmScoreOut> out(n);
        std::memcpy(out.data(), readback.mapped[0].data(), outSize);
        return out;
    }

private:
    Device device_;
    Resources resources_;
    ComputePipeline pipeline_;
    vk::raii::CommandPool pool_;
};

// Compare the GPU output against the CPU authority for every split: scores CLOSE (finite, >= 0),
// backface EXACT. `verts` supplies parent/child object-space positions (canonical IDs index it).
void expectMatchesCpu(std::span<const VdpmScoreOut> gpu, std::span<const Vertex> verts,
                      const VertexForest& forest, const VdpmViewParams& view)
{
    REQUIRE(gpu.size() == forest.splits.size());
    auto close = [](float g, float cpu) { return g == Approx(cpu).epsilon(1e-4f).margin(1e-4f); };
    for (std::size_t i = 0; i < forest.splits.size(); ++i)
    {
        const VertexSplit& s = forest.splits[i];
        const VdpmSplitScore cpu =
            scoreVdpmSplit(view, s, verts[s.parent].position(), verts[s.child].position());
        const VdpmScoreOut& g = gpu[i];
        CAPTURE(i);
        // finite + non-negative everywhere.
        CHECK(std::isfinite(g.geometry));
        CHECK(g.geometry >= 0.0f);
        CHECK(std::isfinite(g.uv));
        CHECK(g.uv >= 0.0f);
        CHECK(std::isfinite(g.normal));
        CHECK(std::isfinite(g.tangent));
        CHECK(std::isfinite(g.straddle));
        // exact back-face decision (fixtures are away from the cone's geometric boundary).
        CHECK(g.backface == cpu.backface);
        // every channel close (not just the max), and straddle as its own component.
        CHECK(close(g.geometry, cpu.geometry));
        CHECK(close(g.uv, cpu.uv));
        CHECK(close(g.normal, cpu.normal));
        CHECK(close(g.tangent, cpu.tangent));
        CHECK(close(g.straddle, cpu.straddle));
    }
}

// A UV sphere (radial normals) → a real forest with genuine cone/back-face + normal-channel
// content.
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
VertexForest forestOf(const Mesh& m)
{
    const QuadricSimplifier simp;
    return buildVertexForest(m.verts, simp.collapseSequence(m.verts, m.indices));
}

// A shared-vertex grid (single wedge per canonical).
Mesh grid(int n)
{
    Mesh m;
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const float u = static_cast<float>(x) / static_cast<float>(n - 1);
            const float v = static_cast<float>(y) / static_cast<float>(n - 1);
            m.verts.push_back(Vertex{Vec3{static_cast<float>(x), static_cast<float>(y), 0.0f},
                                     Colour3{}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{u, v}});
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

// A per-corner-duplicated grid: every triangle carries its own three vertices at shared positions
// with slightly varied UVs → every canonical is multi-wedge, so nearestWedge genuinely tie-breaks
// and the GPU restore must reproduce the CPU's exact wedge choice (the byte-identity risk).
Mesh seamedGrid(int n)
{
    const Mesh shared = grid(n);
    Mesh dup;
    for (std::size_t t = 0; t < shared.indices.size(); ++t)
    {
        const Vertex& src = shared.verts[shared.indices[t]];
        Vertex v = src;
        v.texCoord(
            Vec2{src.texCoord().s() + (0.125f * static_cast<float>(t % 5)), src.texCoord().t()});
        dup.indices.push_back(static_cast<std::uint32_t>(dup.verts.size()));
        dup.verts.push_back(v);
    }
    return dup;
}

// A tiny SYNTHETIC forest with hand-set per-channel errors, so each channel can be made dominant
// and every channel is non-zero (a real sphere has no UV/tangent deviation). Two splits
// (multi-split mandatory — a wrong output stride can't pass on element 0 alone).
struct Synthetic
{
    std::vector<Vertex> verts;
    VertexForest forest;
};
Synthetic syntheticForest()
{
    Synthetic s;
    // 4 canonical vertices at distinct positions (front of the camera at +z).
    for (const Vec3 p : {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 1, 0}})
    {
        s.verts.push_back(Vertex{p, Colour3{}, Vec3{0, 0, 1}, Vec2{0, 0}});
    }
    s.forest.vertexCount = s.verts.size();
    auto mk =
        [](std::uint32_t parent, std::uint32_t child, float err, float uv, float nrm, float tan)
    {
        VertexSplit v;
        v.parent = parent;
        v.child = child;
        v.error = err;
        v.uvError = uv;
        v.normalError = nrm;
        v.tangentError = tan;
        v.supportRadius = 0.1f;
        v.normalConeAxis = Vec3{0.0f, 0.0f, 1.0f}; // faces +z (the camera) — never back-facing
        v.normalConeCos = 0.99f;
        return v;
    };
    // Split 0: geometry + normal set; split 1: UV + tangent set — each channel exercised as a
    // winner.
    s.forest.splits = {mk(0, 1, 0.5f, 0.0f, 0.7f, 0.0f), mk(2, 3, 0.0f, 0.4f, 0.0f, 0.9f)};
    // Each split removes its child — validateForest (the VdpmGpuMesh boundary) requires the forward
    // + reverse removingSplit/child consistency, so wire it up (split 0 removes vertex 1, split 1
    // removes vertex 3).
    s.forest.removingSplit.assign(s.verts.size(), kNoSplit);
    s.forest.removingSplit[1] = 0;
    s.forest.removingSplit[3] = 1;
    return s;
}

} // namespace

TEST_CASE("VDPM GPU score matches the CPU authority on a real forest", "[.][gpu]")
{
    GpuScorer scorer;
    const Mesh m = uvSphere(16, 20);
    const VertexForest forest = forestOf(m);
    REQUIRE(forest.splits.size() > 1); // multi-split
    const Vec3 cam{0.0f, 0.0f, 3.0f};

    SECTION("identity world, cull on")
    {
        const VdpmViewParams v =
            makeVdpmViewParams(Mat4::identity(), cam, 1.0f, 1000.0f, 2.0f, true, 1.0f, 0.5f, 0.5f);
        expectMatchesCpu(scorer.score(m.verts, forest, v), m.verts, forest, v);
    }
    SECTION("non-uniform world")
    {
        const Mat4 world = Mat4::scale(Vec3{1.6f, 0.7f, 1.3f});
        const VdpmViewParams v =
            makeVdpmViewParams(world, cam, 1.0f, 1000.0f, 2.0f, true, 1.0f, 0.5f, 0.5f);
        expectMatchesCpu(scorer.score(m.verts, forest, v), m.verts, forest, v);
    }
    SECTION("reflected (negative-determinant) world")
    {
        const Mat4 world = Mat4::scale(Vec3{-1.0f, 1.0f, 1.0f});
        const VdpmViewParams v =
            makeVdpmViewParams(world, cam, 1.0f, 1000.0f, 2.0f, true, 1.0f, 0.5f, 0.5f);
        expectMatchesCpu(scorer.score(m.verts, forest, v), m.verts, forest, v);
    }
    SECTION("culling disabled")
    {
        const VdpmViewParams v =
            makeVdpmViewParams(Mat4::identity(), cam, 1.0f, 1000.0f, 2.0f, false, 1.0f, 0.5f, 0.5f);
        expectMatchesCpu(scorer.score(m.verts, forest, v), m.verts, forest, v);
    }
}

TEST_CASE("VDPM GPU score: every channel exercised as a winner (synthetic forest)", "[.][gpu]")
{
    GpuScorer scorer;
    const Synthetic s = syntheticForest();
    const Vec3 cam{0.5f, 0.5f, 4.0f};

    // Crank each channel's scale in turn so it dominates, and confirm ALL channels still match.
    for (const std::array<float, 3> scales :
         {std::array<float, 3>{1.0f, 1.0f, 1.0f}, std::array<float, 3>{50.0f, 1.0f, 1.0f},
          std::array<float, 3>{1.0f, 50.0f, 1.0f}, std::array<float, 3>{1.0f, 1.0f, 50.0f}})
    {
        const VdpmViewParams v = makeVdpmViewParams(Mat4::identity(), cam, 1.0f, 1000.0f, 2.0f,
                                                    true, scales[0], scales[1], scales[2]);
        expectMatchesCpu(scorer.score(s.verts, s.forest, v), s.verts, s.forest, v);
    }
}

TEST_CASE("VDPM GPU score: singular world + camera-inside-support corner cases", "[.][gpu]")
{
    GpuScorer scorer;
    const Synthetic s = syntheticForest();

    SECTION("singular (zero-scale) world: cone unusable, never culls, still scores")
    {
        const Mat4 singular = Mat4::scale(Vec3{1.0f, 1.0f, 0.0f}); // det 0
        const VdpmViewParams v = makeVdpmViewParams(singular, Vec3{0, 0, 4}, 1.0f, 1000.0f, 2.0f,
                                                    true, 1.0f, 1.0f, 1.0f);
        const auto gpu = scorer.score(s.verts, s.forest, v);
        expectMatchesCpu(gpu, s.verts, s.forest, v);
        for (const VdpmScoreOut& o : gpu)
        {
            CHECK(o.backface == 0u); // singular ⇒ never culls
        }
    }
    SECTION("camera inside the support sphere: never culls, max straddle")
    {
        // Camera essentially at a split's parent (support radius 0.1) — the near-sphere branch.
        const VdpmViewParams v = makeVdpmViewParams(Mat4::identity(), Vec3{0.0f, 0.0f, 0.02f}, 1.0f,
                                                    1000.0f, 2.0f, true, 1.0f, 1.0f, 1.0f);
        expectMatchesCpu(scorer.score(s.verts, s.forest, v), s.verts, s.forest, v);
    }
}

TEST_CASE("VDPM GPU score: a zero-split forest dispatches nothing", "[.][gpu]")
{
    GpuScorer scorer;
    VertexForest empty;
    empty.vertexCount = 0;
    const VdpmViewParams v = makeVdpmViewParams(Mat4::identity(), Vec3{0, 0, 3}, 1.0f, 1000.0f,
                                                2.0f, true, 1.0f, 1.0f, 1.0f);
    const auto out = scorer.score({}, empty, v);
    CHECK(out.empty());
}

TEST_CASE("VdpmGpuMesh::build is the validation boundary (rejects malformed forests)", "[.][gpu]")
{
    const Device device = Device::headlessCompute();
    Resources resources(device);

    // Vertex count must match the forest.
    VertexForest mismatched;
    mismatched.vertexCount = 3;
    mismatched.removingSplit.assign(3, kNoSplit);
    const std::vector<Vertex> twoVerts(2);
    REQUIRE_THROWS_AS(VdpmGpuMesh::build(resources, twoVerts, mismatched), std::runtime_error);

    // Structural: a split claims to remove vertex 1, but removingSplit[1] doesn't point back.
    VertexForest bad;
    bad.vertexCount = 4;
    bad.removingSplit.assign(4, kNoSplit);
    VertexSplit s;
    s.parent = 0;
    s.child = 1;
    s.vl = 0;
    bad.splits = {s};
    const std::vector<Vertex> fourVerts(4);
    REQUIRE_THROWS_AS(VdpmGpuMesh::build(resources, fourVerts, bad), std::runtime_error);
}

// ============================================================================================
// GPU emit harness ([.][gpu], local-only). Cross-checks the four emit passes (shaders/vdpm_ancestor
// | survival | scatter | emit_finalize) + the exclusive scan against the CPU emit authority
// (ParallelFront::emitActiveIndices) on a headless device. The GPU emit MUST be BYTE-IDENTICAL to
// the CPU — same indices, same order, same restored wedges — for any valid settled front.
// ============================================================================================

namespace
{

// The GPU emit result: the emitted index stream (already truncated to counters[2]) + the ancestor
// failure count (must be 0 on a valid front).
struct EmitResult
{
    std::vector<std::uint32_t> indices;
    std::uint32_t failureCount{0};
    std::uint32_t emittedCount{0};
};

// A headless compute device + Resources + the five emit pipelines + a command pool. Builds a full
// (score + emit) mesh/front per call, records the emit for an uploaded front, and reads back the
// emitted indices + counters. Non-movable; build one per case.
class GpuEmitter
{
public:
    GpuEmitter()
        : device_(Device::headlessCompute()),
          resources_(device_),
          pipelines_(device_),
          pool_(device_.device(), vk::CommandPoolCreateInfo{
                                      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                      .queueFamilyIndex = device_.graphicsFamily()})
    {
    }
    GpuEmitter(const GpuEmitter&) = delete;
    GpuEmitter& operator=(const GpuEmitter&) = delete;
    GpuEmitter(GpuEmitter&&) = delete;
    GpuEmitter& operator=(GpuEmitter&&) = delete;
    ~GpuEmitter() = default;

    [[nodiscard]] EmitResult emit(std::span<const Vertex> verts,
                                  std::span<const std::uint32_t> indices,
                                  const VertexForest& forest, std::span<const std::uint32_t> active)
    {
        const VdpmGpuMesh mesh = VdpmGpuMesh::build(resources_, verts, indices, forest);
        VdpmGpuFront front = VdpmGpuFront::buildWithEmit(resources_, mesh);
        const std::uint32_t faceCount = front.faceCount();

        const vk::DeviceSize countersSize = 3 * sizeof(std::uint32_t);
        const vk::DeviceSize idxSize =
            static_cast<vk::DeviceSize>(faceCount) * 3 * sizeof(std::uint32_t);
        const Resources::MappedBufferSet countersHost =
            resources_.createMappedReadbackBuffers(countersSize);
        // A zero-face mesh has no emitted-index buffer; only the counters are read.
        const Resources::MappedBufferSet idxHost =
            faceCount > 0 ? resources_.createMappedReadbackBuffers(idxSize)
                          : Resources::MappedBufferSet{};

        const vk::CommandBufferAllocateInfo ai{.commandPool = *pool_,
                                               .level = vk::CommandBufferLevel::ePrimary,
                                               .commandBufferCount = 1};
        auto cmds = device_.device().allocateCommandBuffers(ai);
        vk::raii::CommandBuffer& cmd = cmds[0];
        cmd.begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        front.recordEmit(*cmd, pipelines_, resources_, active);

        // The counters buffer is CLEAR-written for counters[0] (valid front — no atomic) and, on a
        // zero-face mesh, counters[1]; only the scatter path writes it via compute. So the counters
        // readback barrier's source scope names BOTH the clear and the compute writes (per
        // recordEmit's consumer contract) — a compute-only source would race the clear-written
        // values.
        recordBufferBarrier(
            *cmd,
            makeBufferMemoryBarrier(
                vk::PipelineStageFlagBits2::eClear | vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderStorageWrite,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead,
                resources_.vulkanBuffer(front.countersBuffer()), 0, countersSize));
        cmd.copyBuffer(resources_.vulkanBuffer(front.countersBuffer()),
                       resources_.vulkanBuffer(countersHost.buffers[0]),
                       vk::BufferCopy{.size = countersSize});
        if (faceCount > 0)
        {
            // The emitted-index stream is purely compute-written (the scatter) → compute→copy.
            recordBufferBarrier(
                *cmd, makeBufferMemoryBarrier(
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderStorageWrite,
                          vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead,
                          resources_.vulkanBuffer(front.emittedIndicesBuffer()), 0, idxSize));
            cmd.copyBuffer(resources_.vulkanBuffer(front.emittedIndicesBuffer()),
                           resources_.vulkanBuffer(idxHost.buffers[0]),
                           vk::BufferCopy{.size = idxSize});
        }

        // Global transfer-write → host-read barrier exposing both copies to the CPU.
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

        std::array<std::uint32_t, 3> counters{};
        std::memcpy(counters.data(), countersHost.mapped[0].data(), countersSize);

        EmitResult result;
        result.failureCount = counters[0];
        const std::uint32_t survivingFaces = counters[1];
        result.emittedCount = counters[2];
        // Sanity-bound the GPU counts BEFORE trusting them for resize/memcpy: no more survivors
        // than faces, the index count is exactly 3 per survivor, and it fits the worst-case
        // allocation.
        REQUIRE(survivingFaces <= faceCount);
        REQUIRE(result.emittedCount == 3u * survivingFaces);
        REQUIRE(result.emittedCount <= 3u * faceCount);
        if (faceCount > 0 && result.emittedCount > 0)
        {
            result.indices.resize(result.emittedCount);
            std::memcpy(result.indices.data(), idxHost.mapped[0].data(),
                        static_cast<std::size_t>(result.emittedCount) * sizeof(std::uint32_t));
        }
        return result;
    }

private:
    Device device_;
    Resources resources_;
    VdpmEmitPipelines pipelines_;
    vk::raii::CommandPool pool_;
};

// The settled front's per-canonical active flags as the GPU-shaped uint32 array (0/1).
[[nodiscard]] std::vector<std::uint32_t> activeFlags(const ParallelFront& front)
{
    const std::uint32_t n = front.forest().vertexCount;
    std::vector<std::uint32_t> a(n);
    for (std::uint32_t v = 0; v < n; ++v)
    {
        a[v] = front.active(v) ? 1u : 0u;
    }
    return a;
}

// Assert the GPU emit is byte-identical to the CPU emit for the SAME settled front, and that no
// ancestor walk failed (a valid front always resolves).
void expectEmitMatchesCpu(GpuEmitter& emitter, std::span<const Vertex> verts,
                          std::span<const std::uint32_t> indices, const ParallelFront& front)
{
    const std::vector<std::uint32_t> active = activeFlags(front);
    const std::vector<std::uint32_t> cpu = front.emitActiveIndices(verts, indices);
    const EmitResult gpu = emitter.emit(verts, indices, front.forest(), active);

    CHECK(gpu.failureCount == 0u);
    CHECK(gpu.emittedCount == cpu.size());
    CHECK(gpu.indices == cpu); // byte-identical: indices, order, AND restored wedges
}

} // namespace

TEST_CASE("VDPM GPU emit: byte-identical to the CPU emit across settled fronts", "[.][gpu]")
{
    GpuEmitter emitter;
    const QuadricSimplifier simp;

    // A sphere (multi-wedge only at poles) and a per-corner-seamed grid (every canonical
    // multi-wedge → nearestWedge genuinely tie-breaks) exercise the wedge-restoration path.
    struct Case
    {
        const char* name;
        Mesh mesh;
    };
    std::vector<Case> cases;
    cases.push_back({"sphere", uvSphere(16, 20)});
    cases.push_back({"seamed-grid", seamedGrid(9)});

    for (const Case& c : cases)
    {
        CAPTURE(c.name);
        const auto collapses = simp.collapseSequence(c.mesh.verts, c.mesh.indices);

        SECTION(std::string(c.name) + " coarsest front (deepest chains → maxDepth walks)")
        {
            // A freshly-built front sits at the coarsest front (roots only), so the deepest finest
            // vertex walks exactly its full removal chain — byte-identity here proves deep-chain
            // ancestor resolution against the recursive CPU activeAncestor.
            const ParallelFront front =
                ParallelFront::build(c.mesh.verts, c.mesh.indices, collapses);
            expectEmitMatchesCpu(emitter, c.mesh.verts, c.mesh.indices, front);
        }
        SECTION(std::string(c.name) + " partially-refined front")
        {
            ParallelFront front = ParallelFront::build(c.mesh.verts, c.mesh.indices, collapses);
            const auto n = static_cast<std::uint32_t>(front.forest().splits.size());
            // Deterministic pseudo-random scores straddling the budget → a mixed front.
            std::mt19937 rng(0x5EED);
            std::uniform_real_distribution<float> dist(0.0f, 2.0f);
            std::vector<float> score(n);
            std::vector<std::uint8_t> backface(n, 0);
            for (std::uint32_t i = 0; i < n; ++i)
            {
                score[i] = dist(rng);
            }
            front.applyView(score, backface, 1.0f, 0.6f);
            front.validateInvariants();
            expectEmitMatchesCpu(emitter, c.mesh.verts, c.mesh.indices, front);
        }
        SECTION(std::string(c.name) + " fully-refined front")
        {
            ParallelFront front = ParallelFront::build(c.mesh.verts, c.mesh.indices, collapses);
            const auto n = static_cast<std::uint32_t>(front.forest().splits.size());
            // Every split over budget → full detail; every finest vertex active (depth-0 restore).
            front.applyView(std::vector<float>(n, 9.0f), std::vector<std::uint8_t>(n, 0), 1.0f,
                            0.6f);
            front.validateInvariants();
            expectEmitMatchesCpu(emitter, c.mesh.verts, c.mesh.indices, front);
        }
        SECTION(std::string(c.name) + " repaired front (realistic view)")
        {
            ParallelFront front = ParallelFront::build(c.mesh.verts, c.mesh.indices, collapses);
            const auto n = static_cast<std::uint32_t>(front.forest().splits.size());
            std::mt19937 rng(0xC0FFEE);
            std::uniform_real_distribution<float> dist(0.0f, 2.0f);
            std::vector<float> score(n);
            std::vector<std::uint8_t> backface(n, 0);
            for (std::uint32_t i = 0; i < n; ++i)
            {
                score[i] = dist(rng);
            }
            front.applyView(score, backface, 1.0f, 0.6f);
            const Mat4 world = Mat4::identity();
            const Vec3 cam{0.0f, 0.0f, 3.0f};
            const Mat4 viewProj = Mat4::identity();
            front.repairFront(c.mesh.verts, world, cam, viewProj, 1000.0f, 1000.0f, true);
            front.validateInvariants();
            expectEmitMatchesCpu(emitter, c.mesh.verts, c.mesh.indices, front);
        }
    }
}

TEST_CASE("VDPM GPU emit: determinism (identical bytes across repeated emits)", "[.][gpu]")
{
    GpuEmitter emitter;
    const QuadricSimplifier simp;
    const Mesh m = seamedGrid(9);
    const auto collapses = simp.collapseSequence(m.verts, m.indices);
    ParallelFront front = ParallelFront::build(m.verts, m.indices, collapses);
    const auto n = static_cast<std::uint32_t>(front.forest().splits.size());
    front.applyView(std::vector<float>(n, 9.0f), std::vector<std::uint8_t>(n, 0), 1.0f, 0.6f);
    const std::vector<std::uint32_t> active = activeFlags(front);

    const EmitResult a = emitter.emit(m.verts, m.indices, front.forest(), active);
    const EmitResult b = emitter.emit(m.verts, m.indices, front.forest(), active);
    CHECK(a.indices == b.indices);
    CHECK(a.emittedCount == b.emittedCount);
    CHECK(a.failureCount == 0u);
    CHECK(b.failureCount == 0u);
}

// ---- Synthetic fixtures pinning the ancestor bound + the degenerate-face path ----

TEST_CASE("VDPM GPU emit: a genuinely empty mesh emits nothing", "[.][gpu]")
{
    GpuEmitter emitter;
    VertexForest empty;
    empty.vertexCount = 0;
    const EmitResult r = emitter.emit({}, {}, empty, {});
    CHECK(r.emittedCount == 0u);
    CHECK(r.failureCount == 0u);
    CHECK(r.indices.empty());
}

TEST_CASE("VDPM GPU emit: all faces collapse but faceCount > 0 (distinct from empty mesh)",
          "[.][gpu]")
{
    // A single triangle (v0,v1,v2) whose corner v2 collapses to v0. At the coarsest front (only the
    // roots v0,v1 active) the face resolves to ancestors (v0,v1,v0) — two equal → it degenerates,
    // so NOTHING is emitted even though faceCount == 1. Distinct from the empty-mesh case above.
    GpuEmitter emitter;
    std::vector<Vertex> verts;
    for (const Vec3 p : {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}})
    {
        verts.push_back(Vertex{p, Colour3{}, Vec3{0, 0, 1}, Vec2{0, 0}});
    }
    const std::vector<std::uint32_t> indices{0, 1, 2};

    VertexForest forest;
    forest.vertexCount = 3;
    forest.removingSplit.assign(3, kNoSplit);
    forest.removingSplit[2] = 0; // vertex 2 is removed by split 0
    VertexSplit split;
    split.parent = 0;
    split.child = 2;
    forest.splits = {split};

    // Coarsest front: only the roots (v0, v1) active; v2 inactive → resolves up to v0.
    const std::vector<std::uint32_t> active{1u, 1u, 0u};
    const EmitResult r = emitter.emit(verts, indices, forest, active);
    CHECK(r.failureCount == 0u); // v2 resolves to v0 — it does not fail
    CHECK(r.emittedCount == 0u); // the lone face degenerates
    CHECK(r.indices.empty());

    // With v2 active too, the same face survives → its three original corners emit.
    const std::vector<std::uint32_t> full{1u, 1u, 1u};
    const EmitResult r2 = emitter.emit(verts, indices, forest, full);
    CHECK(r2.failureCount == 0u);
    CHECK(r2.emittedCount == 3u);
    CHECK(r2.indices == std::vector<std::uint32_t>{0u, 1u, 2u});
}

TEST_CASE("VDPM GPU emit: deepest chain — active root exactly maxDepth away resolves, no failure",
          "[.][gpu]")
{
    // A linear removal chain v3 → v2 → v1 → v0 (each removed by a split whose parent is the
    // previous vertex): maxDepth == 3. With only the root v0 active, the deepest vertex v3 must be
    // reached AFTER exactly maxDepth transitions. The ancestor loop tests the vertex reached after
    // the final allowed transition, so v3 resolves to v0 (depth 3) rather than tripping the bound —
    // a wrong off-by-one would reject it and bump the failure counter.
    GpuEmitter emitter;
    std::vector<Vertex> verts;
    for (const Vec3 p : {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{2, 0, 0}, Vec3{3, 0, 0}})
    {
        verts.push_back(Vertex{p, Colour3{}, Vec3{0, 0, 1}, Vec2{0, 0}});
    }
    // One triangle referencing the two deepest vertices + the root, so the ancestor walk for v3
    // (and v2) actually runs.
    const std::vector<std::uint32_t> indices{0, 2, 3};

    VertexForest forest;
    forest.vertexCount = 4;
    forest.removingSplit.assign(4, kNoSplit);
    forest.removingSplit[1] = 2; // v1 removed by split 2 (parent v0)
    forest.removingSplit[2] = 1; // v2 removed by split 1 (parent v1)
    forest.removingSplit[3] = 0; // v3 removed by split 0 (parent v2)
    auto mk = [](std::uint32_t parent, std::uint32_t child)
    {
        VertexSplit s;
        s.parent = parent;
        s.child = child;
        return s;
    };
    forest.splits = {mk(2, 3), mk(1, 2), mk(0, 1)}; // split i removes child i via removingSplit

    // Only the root v0 active — v1, v2, v3 all resolve up the chain to v0.
    const std::vector<std::uint32_t> active{1u, 0u, 0u, 0u};
    const EmitResult r = emitter.emit(verts, indices, forest, active);
    CHECK(r.failureCount == 0u); // v3 at exactly maxDepth resolves — the bound is not off by one
    CHECK(r.emittedCount == 0u); // all three corners collapse to v0 → the face degenerates
}
