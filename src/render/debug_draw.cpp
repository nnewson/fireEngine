#include <fire_engine/render/debug_draw.hpp>

#include <array>
#include <cmath>
#include <cstring>

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/math/vec4.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/swapchain.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

namespace
{

constexpr vk::Format kHdrFormat = vk::Format::eR16G16B16A16Sfloat;
// Capacity of the per-frame debug vertex buffer (line endpoints). Generous for
// the collider/contact counts the engine handles; excess geometry is dropped.
constexpr std::size_t kMaxDebugLineVertices = 1U << 15U;
constexpr int kCircleSegments = 24;

// Category colours.
constexpr Colour3 kAabbColour{0.2f, 0.9f, 0.2f};
constexpr Colour3 kShapeColour{0.2f, 0.8f, 1.0f};
constexpr Colour3 kShapeAsleepColour{0.4f, 0.4f, 0.55f}; // dimmed: body is asleep
constexpr Colour3 kContactNormalColour{1.0f, 0.85f, 0.1f};
constexpr Colour3 kContactPointColour{1.0f, 0.2f, 0.2f};

[[nodiscard]]
Vertex lineVertex(Vec3 position, Colour3 colour)
{
    return Vertex{position, colour, Vec3{}, Vec2{}};
}

void imageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageAspectFlags aspect,
                  vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                  vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                  vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::ImageMemoryBarrier2 b{
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = aspect,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}

} // namespace

DebugDraw::DebugDraw(const Device& device, const Swapchain& swapchain, Resources& resources)
    : swapchain_{&swapchain},
      resources_{&resources},
      pipeline_(device, Pipeline::debugLineConfig(kHdrFormat))
{
    vertexBuffers_ = resources_->createMappedVertexBuffers(kMaxDebugLineVertices * sizeof(Vertex));
}

void DebugDraw::addLine(Vec3 a, Vec3 b, Colour3 colour)
{
    lines_.push_back(lineVertex(a, colour));
    lines_.push_back(lineVertex(b, colour));
}

void DebugDraw::addAabb(const AABB& box, Colour3 colour)
{
    const Vec3 lo = box.min;
    const Vec3 hi = box.max;
    const std::array<Vec3, 8> c{{
        {lo.x(), lo.y(), lo.z()},
        {hi.x(), lo.y(), lo.z()},
        {hi.x(), hi.y(), lo.z()},
        {lo.x(), hi.y(), lo.z()},
        {lo.x(), lo.y(), hi.z()},
        {hi.x(), lo.y(), hi.z()},
        {hi.x(), hi.y(), hi.z()},
        {lo.x(), hi.y(), hi.z()},
    }};
    constexpr std::array<std::array<int, 2>, 12> edges{{{{0, 1}},
                                                        {{1, 2}},
                                                        {{2, 3}},
                                                        {{3, 0}},
                                                        {{4, 5}},
                                                        {{5, 6}},
                                                        {{6, 7}},
                                                        {{7, 4}},
                                                        {{0, 4}},
                                                        {{1, 5}},
                                                        {{2, 6}},
                                                        {{3, 7}}}};
    for (const auto& e : edges)
    {
        addLine(c[static_cast<std::size_t>(e[0])], c[static_cast<std::size_t>(e[1])], colour);
    }
}

void DebugDraw::addBox(Vec3 center, Vec3 halfExtents, const Quaternion& orientation, Colour3 colour)
{
    const Mat4 rot = orientation.toMat4();
    auto corner = [&](float sx, float sy, float sz)
    {
        const Vec3 local{sx * halfExtents.x(), sy * halfExtents.y(), sz * halfExtents.z()};
        return center + Vec3{rot * Vec4{local}};
    };
    const std::array<Vec3, 8> c{{corner(-1, -1, -1), corner(1, -1, -1), corner(1, 1, -1),
                                 corner(-1, 1, -1), corner(-1, -1, 1), corner(1, -1, 1),
                                 corner(1, 1, 1), corner(-1, 1, 1)}};
    constexpr std::array<std::array<int, 2>, 12> edges{{{{0, 1}},
                                                        {{1, 2}},
                                                        {{2, 3}},
                                                        {{3, 0}},
                                                        {{4, 5}},
                                                        {{5, 6}},
                                                        {{6, 7}},
                                                        {{7, 4}},
                                                        {{0, 4}},
                                                        {{1, 5}},
                                                        {{2, 6}},
                                                        {{3, 7}}}};
    for (const auto& e : edges)
    {
        addLine(c[static_cast<std::size_t>(e[0])], c[static_cast<std::size_t>(e[1])], colour);
    }
}

void DebugDraw::addSphere(Vec3 center, float radius, Colour3 colour)
{
    // Three orthogonal great circles.
    auto circle = [&](Vec3 axisU, Vec3 axisV)
    {
        Vec3 prev = center + axisU * radius;
        for (int i = 1; i <= kCircleSegments; ++i)
        {
            const float t =
                static_cast<float>(i) / static_cast<float>(kCircleSegments) * 6.2831853f;
            const Vec3 p = center + (axisU * std::cos(t) + axisV * std::sin(t)) * radius;
            addLine(prev, p, colour);
            prev = p;
        }
    };
    circle(Vec3{1, 0, 0}, Vec3{0, 1, 0});
    circle(Vec3{1, 0, 0}, Vec3{0, 0, 1});
    circle(Vec3{0, 1, 0}, Vec3{0, 0, 1});
}

void DebugDraw::addCapsule(Vec3 p0, Vec3 p1, float radius, Colour3 colour)
{
    // End cap circles (oriented to the segment axis) + connecting struts. A v1
    // simplification: full hemisphere arcs land with P1's richer narrowphase.
    Vec3 axis = p1 - p0;
    const float len = axis.magnitude();
    axis = (len > 1e-5f) ? axis * (1.0f / len) : Vec3{0, 1, 0};
    Vec3 u = Vec3::crossProduct(axis, Vec3{1, 0, 0});
    if (u.magnitudeSquared() < 1e-6f)
    {
        u = Vec3::crossProduct(axis, Vec3{0, 0, 1});
    }
    u = Vec3::normalise(u);
    const Vec3 v = Vec3::normalise(Vec3::crossProduct(axis, u));

    auto ring = [&](Vec3 c)
    {
        Vec3 prev = c + u * radius;
        for (int i = 1; i <= kCircleSegments; ++i)
        {
            const float t =
                static_cast<float>(i) / static_cast<float>(kCircleSegments) * 6.2831853f;
            const Vec3 p = c + (u * std::cos(t) + v * std::sin(t)) * radius;
            addLine(prev, p, colour);
            prev = p;
        }
    };
    ring(p0);
    ring(p1);
    addSphere(p0, radius, colour);
    addSphere(p1, radius, colour);
    addLine(p0 + u * radius, p1 + u * radius, colour);
    addLine(p0 - u * radius, p1 - u * radius, colour);
    addLine(p0 + v * radius, p1 + v * radius, colour);
    addLine(p0 - v * radius, p1 - v * radius, colour);
}

void DebugDraw::addContact(const DebugContact& contact)
{
    // Normal segment + a small axis cross at the point.
    constexpr float kNormalLen = 0.4f;
    constexpr float kCross = 0.08f;
    addLine(contact.point, contact.point + contact.normal * kNormalLen, kContactNormalColour);
    addLine(contact.point - Vec3{kCross, 0, 0}, contact.point + Vec3{kCross, 0, 0},
            kContactPointColour);
    addLine(contact.point - Vec3{0, kCross, 0}, contact.point + Vec3{0, kCross, 0},
            kContactPointColour);
    addLine(contact.point - Vec3{0, 0, kCross}, contact.point + Vec3{0, 0, kCross},
            kContactPointColour);
}

void DebugDraw::buildLines(const PhysicsDebugData& data, const RenderTunables& tunables)
{
    lines_.clear();
    if (tunables.debugDrawAabbs)
    {
        for (const AABB& box : data.aabbs)
        {
            addAabb(box, kAabbColour);
        }
    }
    if (tunables.debugDrawColliders)
    {
        for (std::size_t i = 0; i < data.shapes.size(); ++i)
        {
            const ClothCollider& shape = data.shapes[i];
            const bool asleep = i < data.shapesAsleep.size() && data.shapesAsleep[i] != 0;
            const Colour3 colour = asleep ? kShapeAsleepColour : kShapeColour;
            const Vec3 a{shape.a[0], shape.a[1], shape.a[2]};
            const Vec3 b{shape.b[0], shape.b[1], shape.b[2]};
            switch (shape.type)
            {
            case 1: // sphere: a = center, a.w = radius
                addSphere(a, shape.a[3], colour);
                break;
            case 2: // box: a = center, b = halfExtents, c = orientation quat
                addBox(a, b, Quaternion{shape.c[0], shape.c[1], shape.c[2], shape.c[3]}, colour);
                break;
            case 3: // capsule: a = (p0, radius), b = (p1, _)
                addCapsule(a, b, shape.a[3], colour);
                break;
            default: // plane (type 0) has no bounded extent — skip
                break;
            }
        }
    }
    if (tunables.debugDrawContacts)
    {
        for (const DebugContact& contact : data.contacts)
        {
            addContact(contact);
        }
    }
    if (lines_.size() > kMaxDebugLineVertices)
    {
        lines_.resize(kMaxDebugLineVertices);
    }
}

void DebugDraw::record(vk::CommandBuffer cmd, TextureHandle hdrTarget, const Mat4& viewProj,
                       const PhysicsDebugData& data, const RenderTunables& tunables,
                       uint32_t frameIndex)
{
    buildLines(data, tunables);
    if (lines_.empty())
    {
        return;
    }
    std::memcpy(vertexBuffers_.mapped[frameIndex], lines_.data(), lines_.size() * sizeof(Vertex));

    const auto extent = swapchain_->extent();
    vk::Rect2D renderArea{.offset = vk::Offset2D{.x = 0, .y = 0}, .extent = extent};

    // The HDR target rests in ShaderReadOnly after the particle pass (ready for
    // bloom/post sampling); bring it to a colour attachment to draw lines, then
    // restore it so post-process's precondition holds.
    const vk::Image hdrImage = resources_->vulkanImage(hdrTarget);
    imageBarrier(cmd, hdrImage, vk::ImageAspectFlagBits::eColor,
                 vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal,
                 vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite);

    // Scene depth was left shader-read by the particle pass; bring it back to a
    // depth attachment so the lines can (optionally) depth-test against it.
    imageBarrier(cmd, swapchain_->depthImage(), vk::ImageAspectFlagBits::eDepth,
                 vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                 vk::ImageLayout::eDepthStencilAttachmentOptimal,
                 vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
                 vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                 vk::AccessFlagBits2::eDepthStencilAttachmentRead);

    vk::RenderingAttachmentInfo colour{
        .imageView = resources_->vulkanImageView(hdrTarget),
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    vk::RenderingAttachmentInfo depth{
        .imageView = swapchain_->depthView(),
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
    };
    cmd.beginRendering(makeRenderingInfo(renderArea, {&colour, 1}, &depth));
    cmd.setViewport(0, makeFullViewport(extent));
    cmd.setScissor(0, renderArea);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_.pipeline());
    // Dynamic depth test: off = x-ray (draw over geometry), on = occluded.
    cmd.setDepthTestEnable(tunables.debugDepthTest ? vk::True : vk::False);
    DebugLinePushConstants push{viewProj};
    cmd.pushConstants<DebugLinePushConstants>(pipeline_.pipelineLayout(),
                                              vk::ShaderStageFlagBits::eVertex, 0, push);
    cmd.bindVertexBuffers(0, resources_->vulkanBuffer(vertexBuffers_.buffers[frameIndex]),
                          {vk::DeviceSize{0}});
    cmd.draw(static_cast<uint32_t>(lines_.size()), 1, 0, 0);
    cmd.endRendering();

    // Restore the HDR target to ShaderReadOnly for the post-process pass.
    imageBarrier(cmd, hdrImage, vk::ImageAspectFlagBits::eColor,
                 vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite,
                 vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
}

} // namespace fire_engine
