#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include <fire_engine/graphics/mesh_simplifier.hpp>
#include <fire_engine/graphics/vdpm.hpp>
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
    s.forest.removingSplit.assign(s.verts.size(), kNoSplit);
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
