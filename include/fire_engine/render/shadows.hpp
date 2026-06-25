#pragma once

#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/constants.hpp>
#include <fire_engine/render/pipeline.hpp>
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

    // `shadowViewProjs` are the light/cascade matrices indexed by ShadowPushConstants::
    // matrixIndex; when `cullingEnabled` each iteration drops casters outside its
    // frustum.
    void recordPass(vk::CommandBuffer cmd, std::span<const DrawCommand> shadowDraws,
                    std::span<const DrawCommand> worldOnlyShadowDraws,
                    std::span<const DrawCommand> selfShadowDraws, int activeSpotCasters,
                    std::span<const PointShadowCaster> pointCasters,
                    std::span<const Mat4> shadowViewProjs, bool cullingEnabled) const;

private:
    Resources* resources_{nullptr};
    Pipeline shadowPipeline_;
    Pipeline selfShadowFirstPipeline_;
    Pipeline selfShadowSecondPipeline_;
    PipelineHandle shadowPipelineHandle_{NullPipeline};
    PipelineHandle selfShadowFirstPipelineHandle_{NullPipeline};
    PipelineHandle selfShadowSecondPipelineHandle_{NullPipeline};
    TextureHandle shadowMapHandle_{NullTexture};
    TextureHandle worldShadowMapHandle_{NullTexture};
    TextureHandle selfShadowFirstMapHandle_{NullTexture};
    TextureHandle selfShadowMapHandle_{NullTexture};
    TextureHandle spotShadowMapHandle_{NullTexture};
    TextureHandle pointShadowMapHandle_{NullTexture};
};

} // namespace fire_engine
