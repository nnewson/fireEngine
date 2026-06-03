#pragma once

#include <cmath>

#include <fire_engine/input/input_state.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine
{

struct RenderContext;

class Camera
{
public:
    Camera() = default;
    ~Camera() = default;

    Camera(const Camera&) = default;
    Camera& operator=(const Camera&) = default;
    Camera(Camera&&) noexcept = default;
    Camera& operator=(Camera&&) noexcept = default;

    void update(const InputState& input_state, const Transform& transform);

    [[nodiscard]]
    Mat4 render(const RenderContext& ctx, const Mat4& world);

    [[nodiscard]]
    Vec3 localPosition() const noexcept
    {
        return localPosition_;
    }

    void localPosition(Vec3 pos) noexcept
    {
        localPosition_ = pos;
    }

    [[nodiscard]]
    float localYaw() const noexcept
    {
        return localYaw_;
    }

    void localYaw(float yaw) noexcept
    {
        localYaw_ = yaw;
    }

    [[nodiscard]]
    float localPitch() const noexcept
    {
        return localPitch_;
    }

    void localPitch(float pitch) noexcept
    {
        localPitch_ = clampPitch(pitch);
    }

    [[nodiscard]]
    Vec3 localTarget() const noexcept
    {
        return computeTarget(localPosition_, localYaw_, localPitch_);
    }

    [[nodiscard]]
    Vec3 worldPosition() const noexcept
    {
        return worldPosition_;
    }

    [[nodiscard]]
    float worldYaw() const noexcept
    {
        return worldYaw_;
    }

    [[nodiscard]]
    float worldPitch() const noexcept
    {
        return worldPitch_;
    }

    [[nodiscard]]
    Vec3 worldTarget() const noexcept
    {
        return computeTarget(worldPosition_, worldYaw_, worldPitch_);
    }

private:
    [[nodiscard]]
    static Vec3 computeTarget(Vec3 pos, float yaw, float pitch) noexcept
    {
        return {
            pos.x() + std::cos(pitch) * std::cos(yaw),
            pos.y() + std::sin(pitch),
            pos.z() + std::cos(pitch) * std::sin(yaw),
        };
    }

    static float clampPitch(float p) noexcept
    {
        if (p > kCameraMaxPitch)
        {
            return kCameraMaxPitch;
        }
        if (p < -kCameraMaxPitch)
        {
            return -kCameraMaxPitch;
        }
        return p;
    }

    Vec3 localPosition_{2.0f, 2.0f, 2.0f};
    float localYaw_{-2.356f};
    float localPitch_{-0.615f};

    Vec3 worldPosition_{2.0f, 2.0f, 2.0f};
    float worldYaw_{-2.356f};
    float worldPitch_{-0.615f};
};

} // namespace fire_engine
