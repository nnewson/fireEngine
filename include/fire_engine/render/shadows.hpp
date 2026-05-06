#pragma once

#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/pipeline.hpp>
#include <fire_engine/render/render_pass.hpp>
#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

class Device;

// Per-frame state for one active point shadow caster — the renderer hands one
// of these to recordPass for every point light that earned a shadow slot, so
// the shadow fragment shader can compute linear distance/range against the
// light's world position.
struct PointShadowCaster
{
    Vec3 worldPosition{};
    float range{0.0f};
};

class Shadows
{
public:
    Shadows(const Device& device, Resources& resources);
    ~Shadows() = default;

    Shadows(const Shadows&) = delete;
    Shadows& operator=(const Shadows&) = delete;
    Shadows(Shadows&&) noexcept = default;
    Shadows& operator=(Shadows&&) noexcept = default;

    [[nodiscard]] PipelineHandle pipelineHandle() const noexcept
    {
        return shadowPipelineHandle_;
    }

    void recordPass(vk::CommandBuffer cmd, const std::vector<DrawCommand>& shadowDraws,
                    int activeSpotCasters,
                    std::span<const PointShadowCaster> pointCasters) const;

private:
    const Device* device_{nullptr};
    Resources* resources_{nullptr};
    RenderPass shadowPass_;
    Pipeline shadowPipeline_;
    PipelineHandle shadowPipelineHandle_{NullPipeline};
    TextureHandle shadowMapHandle_{NullTexture};
    TextureHandle shadowColourHandle_{NullTexture};
    TextureHandle spotShadowMapHandle_{NullTexture};
    TextureHandle spotShadowColourHandle_{NullTexture};
    TextureHandle pointShadowMapHandle_{NullTexture};
    TextureHandle pointShadowColourHandle_{NullTexture};
    // Framebuffers for each spot caster (one per layer in the spot 2D-array
    // shadow map). Reuse the existing depth-only render pass — attachment
    // descriptions are identical.
    RenderPass spotShadowPass_;
    // Framebuffers for each (cube, face) of every point caster.
    RenderPass pointShadowPass_;
};

} // namespace fire_engine
