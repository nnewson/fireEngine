#include <fire_engine/input/input.hpp>

namespace fire_engine
{

InputState Input::update(Window& window, float deltaTime, bool suppressMouse, bool suppressKeyboard)
{
    window.pollEvents();

    keyboard_.poll(window);
    mouse_.poll(window);

    // WASD movement (camera-relative: z = forward/back, x = strafe) plus
    // E/F world-space vertical movement on Y.
    Vec3 delta{};

    if (!suppressKeyboard && keyboard_.pressed(Key::W))
    {
        delta.z(delta.z() + speed_ * deltaTime);
    }
    if (!suppressKeyboard && keyboard_.pressed(Key::S))
    {
        delta.z(delta.z() - speed_ * deltaTime);
    }
    if (!suppressKeyboard && keyboard_.pressed(Key::A))
    {
        delta.x(delta.x() + speed_ * deltaTime);
    }
    if (!suppressKeyboard && keyboard_.pressed(Key::D))
    {
        delta.x(delta.x() - speed_ * deltaTime);
    }
    if (!suppressKeyboard && keyboard_.pressed(Key::E))
    {
        delta.y(delta.y() + speed_ * deltaTime);
    }
    if (!suppressKeyboard && keyboard_.pressed(Key::F))
    {
        delta.y(delta.y() - speed_ * deltaTime);
    }

    // Left mouse button drag = camera-relative movement (same as WASD)
    if (!suppressMouse && mouse_.leftButton())
    {
        delta.x(delta.x() + static_cast<float>(mouse_.deltaX()) * panSensitivity_);
        delta.z(delta.z() - static_cast<float>(mouse_.deltaY()) * panSensitivity_);
    }

    CameraState cameraState;
    cameraState.deltaPosition(delta);

    ControllerState controllerState;
    Vec3 controllerDelta{};
    if (!suppressKeyboard && keyboard_.pressed(Key::Left))
    {
        controllerDelta.x(controllerDelta.x() - deltaTime);
    }
    if (!suppressKeyboard && keyboard_.pressed(Key::Right))
    {
        controllerDelta.x(controllerDelta.x() + deltaTime);
    }
    controllerState.deltaPosition(controllerDelta);

    // Right mouse button drag = rotation (yaw/pitch)
    if (!suppressMouse && mouse_.rightButton())
    {
        cameraState.deltaYaw(static_cast<float>(mouse_.deltaX()) * sensitivity_);
        cameraState.deltaPitch(-static_cast<float>(mouse_.deltaY()) * sensitivity_);
    }

    // Scroll wheel = zoom. Always drained (so it doesn't accumulate while the
    // overlay has the mouse), applied only when the camera owns the mouse.
    double scroll = window.consumeScrollDelta();
    if (!suppressMouse && scroll != 0.0)
    {
        cameraState.deltaZoom(static_cast<float>(scroll) * zoomSpeed_);
    }

    InputState state;
    state.deltaTime(deltaTime);
    state.cameraState(cameraState);
    state.controllerState(controllerState);

    if (!suppressKeyboard && keyboard_.pressed(Key::One))
    {
        state.animationState().activeAnimation(0);
    }
    else if (!suppressKeyboard && keyboard_.pressed(Key::Two))
    {
        state.animationState().activeAnimation(1);
    }
    else if (!suppressKeyboard && keyboard_.pressed(Key::Three))
    {
        state.animationState().activeAnimation(2);
    }

    const bool variantKey = !suppressKeyboard && keyboard_.pressed(Key::V);
    if (variantKey && !previousVariantKey_)
    {
        state.variantState().cycleDelta(keyboard_.shift() ? -1 : 1);
    }
    previousVariantKey_ = variantKey;

    return state;
}

} // namespace fire_engine
