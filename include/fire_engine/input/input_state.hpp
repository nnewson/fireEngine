#pragma once

#include <fire_engine/input/animation_state.hpp>
#include <fire_engine/input/camera_state.hpp>
#include <fire_engine/input/controller_state.hpp>
#include <fire_engine/input/variant_state.hpp>

namespace fire_engine
{

class InputState
{
public:
    InputState() = default;
    ~InputState() = default;

    InputState(const InputState&) = default;
    InputState& operator=(const InputState&) = default;
    InputState(InputState&&) noexcept = default;
    InputState& operator=(InputState&&) noexcept = default;

    [[nodiscard]] const CameraState& cameraState() const noexcept
    {
        return cameraState_;
    }
    [[nodiscard]] CameraState& cameraState() noexcept
    {
        return cameraState_;
    }
    void cameraState(CameraState cs) noexcept
    {
        cameraState_ = cs;
    }

    [[nodiscard]] const AnimationState& animationState() const noexcept
    {
        return animationState_;
    }
    [[nodiscard]] AnimationState& animationState() noexcept
    {
        return animationState_;
    }
    void animationState(AnimationState as) noexcept
    {
        animationState_ = as;
    }

    [[nodiscard]] const ControllerState& controllerState() const noexcept
    {
        return controllerState_;
    }
    [[nodiscard]] ControllerState& controllerState() noexcept
    {
        return controllerState_;
    }
    void controllerState(ControllerState cs) noexcept
    {
        controllerState_ = cs;
    }

    [[nodiscard]] const VariantState& variantState() const noexcept
    {
        return variantState_;
    }
    [[nodiscard]] VariantState& variantState() noexcept
    {
        return variantState_;
    }
    void variantState(VariantState vs) noexcept
    {
        variantState_ = vs;
    }

    [[nodiscard]] double time() const noexcept
    {
        return cameraState_.time();
    }
    void time(double t) noexcept
    {
        cameraState_.time(t);
        controllerState_.time(t);
    }

    [[nodiscard]] float deltaTime() const noexcept
    {
        return deltaTime_;
    }
    void deltaTime(float dt) noexcept
    {
        deltaTime_ = dt;
    }

    [[nodiscard]] bool overlayTogglePressed() const noexcept
    {
        return overlayTogglePressed_;
    }
    void overlayTogglePressed(bool pressed) noexcept
    {
        overlayTogglePressed_ = pressed;
    }

private:
    CameraState cameraState_{};
    AnimationState animationState_{};
    ControllerState controllerState_{};
    VariantState variantState_{};
    float deltaTime_{0.0f};
    bool overlayTogglePressed_{false};
};

} // namespace fire_engine
