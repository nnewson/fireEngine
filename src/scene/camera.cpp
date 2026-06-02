#include <fire_engine/scene/node.hpp>

#include <algorithm>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

void Camera::update(const InputState& input_state, const Transform& transform)
{
    const auto& cs = input_state.cameraState();

    // Apply rotation first (from right-mouse drag)
    localYaw_ += cs.deltaYaw();
    localPitch_ = clampPitch(localPitch_ + cs.deltaPitch());

    // Compute camera basis vectors from yaw
    float cosYaw = std::cos(localYaw_);
    float sinYaw = std::sin(localYaw_);
    Vec3 forwardXZ{cosYaw, 0.0f, sinYaw};
    Vec3 right{sinYaw, 0.0f, -cosYaw};

    // WASD and left-mouse drag movement use the camera-relative XZ plane.
    // Vertical keyboard movement is applied directly on world Y.
    const Vec3& delta = cs.deltaPosition();
    localPosition_ =
        localPosition_ + forwardXZ * delta.z() + right * delta.x() + Vec3{0.0f, delta.y(), 0.0f};

    // Scroll wheel zoom (along full 3D view direction)
    float cosPitch = std::cos(localPitch_);
    float sinPitch = std::sin(localPitch_);
    Vec3 forward3D{cosPitch * cosYaw, sinPitch, cosPitch * sinYaw};
    localPosition_ = localPosition_ + forward3D * cs.deltaZoom();

    const Mat4 world = transform.world();
    const Vec3 localTarget = computeTarget(localPosition_, localYaw_, localPitch_);
    worldPosition_ = Vec3{world * Vec4{localPosition_}};

    Vec3 worldForward = static_cast<Vec3>(world * Vec4{localTarget}) - worldPosition_;
    if (worldForward.magnitudeSquared() < float_epsilon)
    {
        worldForward = computeTarget(Vec3{}, localYaw_, localPitch_);
    }
    else
    {
        worldForward.normalise();
    }

    worldYaw_ = std::atan2(worldForward.z(), worldForward.x());
    worldPitch_ = clampPitch(std::asin(std::clamp(worldForward.y(), -1.0f, 1.0f)));
}

Mat4 Camera::render(const RenderContext& /*ctx*/, const Mat4& world)
{
    return world;
}

} // namespace fire_engine
