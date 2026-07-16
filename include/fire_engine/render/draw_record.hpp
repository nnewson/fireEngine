#pragma once

#include <vulkan/vulkan.hpp>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

// Record an indexed draw for `dc`, honouring the indirect sentinel in ONE place so the three VDPM
// draw sites (forward, depth prepass, transmission) can't drift: a non-null `indirectBuffer`
// records `drawIndexedIndirect` from the DrawIndexedIndirectCommand at `indirectOffset`; otherwise
// a direct `drawIndexed(indexCount, ...)`. The explicit stride
// (`sizeof(DrawIndexedIndirectCommand)` — the record layout, proven equal to
// VkDrawIndexedIndirectCommand's size by the static_asserts in resources.cpp) is ignored by Vulkan
// at drawCount == 1 but is ready for multi-command batches. The index buffer must already be bound
// by the caller.
inline void recordIndexedDraw(vk::CommandBuffer cmd, const DrawCommand& dc,
                              const Resources& resources)
{
    if (dc.indirectBuffer != NullBuffer)
    {
        cmd.drawIndexedIndirect(resources.vulkanBuffer(dc.indirectBuffer), dc.indirectOffset, 1,
                                sizeof(DrawIndexedIndirectCommand));
    }
    else
    {
        cmd.drawIndexed(dc.indexCount, 1, 0, 0, 0);
    }
}

} // namespace fire_engine
