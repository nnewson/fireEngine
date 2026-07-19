#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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
