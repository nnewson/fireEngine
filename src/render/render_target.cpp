#include <fire_engine/render/render_target.hpp>

namespace fire_engine
{

vk::RenderingInfo makeRenderingInfo(vk::Rect2D area,
                                    std::span<const vk::RenderingAttachmentInfo> colours,
                                    const vk::RenderingAttachmentInfo* depth)
{
    return vk::RenderingInfo{
        .renderArea = area,
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(colours.size()),
        .pColorAttachments = colours.data(),
        .pDepthAttachment = depth,
    };
}

} // namespace fire_engine
