#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include <fire_engine/collision/aabb.hpp>
#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/contact.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/render_tunables.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;
class Swapchain;

// Physics debug data the main loop forwards to the renderer each frame: world
// AABBs (broadphase bounds), authored collider shapes (from
// PhysicsWorld::gatherColliders), and the last step's approximate contacts. All
// Vulkan-free.
// An arbitrary coloured line segment — used by the query-probe demo to draw rays and
// overlap markers from PhysicsWorld queries.
struct DebugLine
{
    Vec3 a{};
    Vec3 b{};
    Colour3 colour{1.0f, 1.0f, 1.0f};
};

struct PhysicsDebugData
{
    std::vector<AABB> aabbs;
    std::vector<ClothCollider> shapes;
    std::vector<DebugContact> contacts;
    // Parallel to `shapes` (1 = asleep): sleeping bodies draw in a distinct colour.
    std::vector<std::uint8_t> shapesAsleep;
    // Free-form coloured lines (query rays / overlap markers); always drawn when present.
    std::vector<DebugLine> queryLines;
};

// Renderer-owned immediate-mode wireframe debug pass. Each frame it builds
// line-list geometry (reusing the standard Vertex) from PhysicsDebugData into a
// per-frame mapped vertex buffer and draws it into the HDR target, after
// particles and before post-process. Depth test is toggled (x-ray vs
// depth-tested) via the dynamic-state line pipeline.
class DebugDraw
{
public:
    DebugDraw(const Device& device, const Swapchain& swapchain, Resources& resources);
    ~DebugDraw() = default;

    DebugDraw(const DebugDraw&) = delete;
    DebugDraw& operator=(const DebugDraw&) = delete;
    DebugDraw(DebugDraw&&) noexcept = default;
    DebugDraw& operator=(DebugDraw&&) noexcept = default;

    // Build + draw the enabled debug categories into `hdrTarget`. No-op (records
    // nothing) when nothing is enabled or there's no geometry. Precondition: HDR
    // in ColorAttachmentOptimal; scene depth in DepthStencilReadOnlyOptimal.
    void record(vk::CommandBuffer cmd, TextureHandle hdrTarget, const Mat4& viewProj,
                const PhysicsDebugData& data, const RenderTunables& tunables, uint32_t frameIndex);

private:
    void buildLines(const PhysicsDebugData& data, const RenderTunables& tunables);
    void addLine(Vec3 a, Vec3 b, Colour3 colour);
    void addAabb(const AABB& box, Colour3 colour);
    void addBox(Vec3 center, Vec3 halfExtents, const Quaternion& orientation, Colour3 colour);
    void addSphere(Vec3 center, float radius, Colour3 colour);
    void addCapsule(Vec3 p0, Vec3 p1, float radius, Colour3 colour);
    void addContact(const DebugContact& contact);

    const Swapchain* swapchain_{nullptr};
    Resources* resources_{nullptr};
    Pipeline pipeline_;
    Resources::MappedBufferSet vertexBuffers_;
    std::vector<Vertex> lines_; // CPU scratch, rebuilt each frame
};

} // namespace fire_engine
