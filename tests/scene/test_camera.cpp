#include <fire_engine/scene/camera.hpp>

#include <cmath>

#include <fire_engine/input/input_state.hpp>
#include <fire_engine/scene/transform.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Camera;
using fire_engine::InputState;
using fire_engine::Mat4;
using fire_engine::Quaternion;
using fire_engine::Transform;
using fire_engine::Vec3;

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("CameraConstruction.DefaultPosition", "[CameraConstruction]")
{
    Camera cam;
    CHECK(cam.localPosition().x() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(2.0f).margin(1e-5f));
}

TEST_CASE("CameraConstruction.DefaultYaw", "[CameraConstruction]")
{
    Camera cam;
    CHECK(cam.localYaw() == Catch::Approx(-2.356f).margin(1e-5f));
}

TEST_CASE("CameraConstruction.DefaultPitch", "[CameraConstruction]")
{
    Camera cam;
    CHECK(cam.localPitch() == Catch::Approx(-0.615f).margin(1e-5f));
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("CameraAccessors.SetPosition", "[CameraAccessors]")
{
    Camera cam;
    cam.localPosition({5.0f, 10.0f, -3.0f});
    CHECK(cam.localPosition().x() == Catch::Approx(5.0f).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(10.0f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(-3.0f).margin(1e-5f));
}

TEST_CASE("CameraAccessors.SetYaw", "[CameraAccessors]")
{
    Camera cam;
    cam.localYaw(1.0f);
    CHECK(cam.localYaw() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("CameraAccessors.SetPitch", "[CameraAccessors]")
{
    Camera cam;
    cam.localPitch(0.5f);
    CHECK(cam.localPitch() == Catch::Approx(0.5f).margin(1e-5f));
}

// ==========================================================================
// Pitch clamping
// ==========================================================================

TEST_CASE("CameraPitchClamp.ClampsAboveMax", "[CameraPitchClamp]")
{
    Camera cam;
    cam.localPitch(2.0f);
    CHECK(cam.localPitch() == Catch::Approx(1.5f).margin(1e-5f));
}

TEST_CASE("CameraPitchClamp.ClampsBelowMin", "[CameraPitchClamp]")
{
    Camera cam;
    cam.localPitch(-2.0f);
    CHECK(cam.localPitch() == Catch::Approx(-1.5f).margin(1e-5f));
}

TEST_CASE("CameraPitchClamp.ExactMaxIsAllowed", "[CameraPitchClamp]")
{
    Camera cam;
    cam.localPitch(1.5f);
    CHECK(cam.localPitch() == Catch::Approx(1.5f).margin(1e-5f));
}

TEST_CASE("CameraPitchClamp.ExactMinIsAllowed", "[CameraPitchClamp]")
{
    Camera cam;
    cam.localPitch(-1.5f);
    CHECK(cam.localPitch() == Catch::Approx(-1.5f).margin(1e-5f));
}

TEST_CASE("CameraPitchClamp.WithinRangeUnchanged", "[CameraPitchClamp]")
{
    Camera cam;
    cam.localPitch(0.75f);
    CHECK(cam.localPitch() == Catch::Approx(0.75f).margin(1e-5f));
}

// ==========================================================================
// Local target calculation
// ==========================================================================

TEST_CASE("CameraLocalTarget.ZeroPitchYawLooksAlongPositiveX", "[CameraLocalTarget]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    Vec3 t = cam.localTarget();
    // cos(0)*cos(0) = 1, sin(0) = 0, cos(0)*sin(0) = 0
    CHECK(t.x() == Catch::Approx(1.0f).margin(1e-6f));
    CHECK(t.y() == Catch::Approx(0.0f).margin(1e-6f));
    CHECK(t.z() == Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("CameraLocalTarget.YawRotatesInXZPlane", "[CameraLocalTarget]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localPitch(0.0f);

    // yaw = pi/2 should look along +Z
    cam.localYaw(static_cast<float>(M_PI) / 2.0f);
    Vec3 t = cam.localTarget();
    CHECK(t.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.z() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("CameraLocalTarget.PitchLooksUpward", "[CameraLocalTarget]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);

    // pitch = pi/4 should raise Y component
    cam.localPitch(static_cast<float>(M_PI) / 4.0f);
    Vec3 t = cam.localTarget();
    CHECK(t.y() == Catch::Approx(std::sin(static_cast<float>(M_PI) / 4.0f)).margin(1e-6f));
    CHECK(t.y() > 0.0f);
}

TEST_CASE("CameraLocalTarget.TargetOffsetsFromPosition", "[CameraLocalTarget]")
{
    Camera cam;
    cam.localPosition({10.0f, 20.0f, 30.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    Vec3 t = cam.localTarget();
    // Target should be position + direction
    CHECK(t.x() == Catch::Approx(11.0f).margin(1e-6f));
    CHECK(t.y() == Catch::Approx(20.0f).margin(1e-6f));
    CHECK(t.z() == Catch::Approx(30.0f).margin(1e-6f));
}

TEST_CASE("CameraLocalTarget.NegativeYawLooksAlongNegativeZ", "[CameraLocalTarget]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(-static_cast<float>(M_PI) / 2.0f);
    cam.localPitch(0.0f);

    Vec3 t = cam.localTarget();
    CHECK(t.x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.z() == Catch::Approx(-1.0f).margin(1e-5f));
}

// ==========================================================================
// Copy semantics
// ==========================================================================

TEST_CASE("CameraCopy.CopyConstructCreatesIndependentCopy", "[CameraCopy]")
{
    Camera a;
    a.localPosition({1.0f, 2.0f, 3.0f});
    a.localYaw(0.5f);
    a.localPitch(0.3f);

    Camera b{a};
    CHECK(b.localPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(b.localYaw() == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(b.localPitch() == Catch::Approx(0.3f).margin(1e-5f));

    // Modifying b should not affect a
    b.localYaw(1.0f);
    CHECK(a.localYaw() == Catch::Approx(0.5f).margin(1e-5f));
}

TEST_CASE("CameraCopy.CopyAssignCreatesIndependentCopy", "[CameraCopy]")
{
    Camera a;
    a.localPosition({4.0f, 5.0f, 6.0f});

    Camera b;
    b = a;
    CHECK(b.localPosition().x() == Catch::Approx(4.0f).margin(1e-5f));

    b.localPosition({0.0f, 0.0f, 0.0f});
    CHECK(a.localPosition().x() == Catch::Approx(4.0f).margin(1e-5f));
}

// ==========================================================================
// Move semantics
// ==========================================================================

TEST_CASE("CameraMove.MoveConstructTransfersState", "[CameraMove]")
{
    Camera a;
    a.localPosition({7.0f, 8.0f, 9.0f});
    a.localYaw(1.2f);

    Camera b{std::move(a)};
    CHECK(b.localPosition().x() == Catch::Approx(7.0f).margin(1e-5f));
    CHECK(b.localYaw() == Catch::Approx(1.2f).margin(1e-5f));
}

TEST_CASE("CameraMove.MoveAssignTransfersState", "[CameraMove]")
{
    Camera a;
    a.localPitch(0.8f);

    Camera b;
    b = std::move(a);
    CHECK(b.localPitch() == Catch::Approx(0.8f).margin(1e-5f));
}

// ==========================================================================
// Camera-relative WASD movement
// ==========================================================================

using fire_engine::CameraState;

// ==========================================================================
// World transform composition
// ==========================================================================

TEST_CASE("CameraWorldTransform.ImportedIdentityCameraLooksDownNegativeZ", "[CameraWorldTransform]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(-static_cast<float>(M_PI) / 2.0f);
    cam.localPitch(0.0f);

    Transform t;
    t.position({0.0f, 24.0f, 60.0f});
    t.update(Mat4::identity());

    cam.update(InputState{}, t);

    CHECK(cam.worldPosition().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.worldPosition().y() == Catch::Approx(24.0f).margin(1e-5f));
    CHECK(cam.worldPosition().z() == Catch::Approx(60.0f).margin(1e-5f));
    CHECK(cam.worldTarget().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.worldTarget().y() == Catch::Approx(24.0f).margin(1e-5f));
    CHECK(cam.worldTarget().z() == Catch::Approx(59.0f).margin(1e-5f));
}

TEST_CASE("CameraWorldTransform.ParentTransformContributesToWorldPosition",
          "[CameraWorldTransform]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(-static_cast<float>(M_PI) / 2.0f);
    cam.localPitch(0.0f);

    Transform t;
    t.position({1.0f, 2.0f, 3.0f});
    t.update(Mat4::translate(Vec3{10.0f, 20.0f, 30.0f}));

    cam.update(InputState{}, t);

    CHECK(cam.worldPosition().x() == Catch::Approx(11.0f).margin(1e-5f));
    CHECK(cam.worldPosition().y() == Catch::Approx(22.0f).margin(1e-5f));
    CHECK(cam.worldPosition().z() == Catch::Approx(33.0f).margin(1e-5f));
    CHECK(cam.worldTarget().z() == Catch::Approx(32.0f).margin(1e-5f));
}

TEST_CASE("CameraWorldTransform.NodeRotationRotatesImportedCameraForward", "[CameraWorldTransform]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(-static_cast<float>(M_PI) / 2.0f);
    cam.localPitch(0.0f);

    const float halfAngle = static_cast<float>(M_PI) * 0.25f;
    Transform t;
    t.rotation(Quaternion{0.0f, std::sin(halfAngle), 0.0f, std::cos(halfAngle)});
    t.update(Mat4::identity());

    cam.update(InputState{}, t);

    CHECK(cam.worldTarget().x() == Catch::Approx(-1.0f).margin(1e-5f));
    CHECK(cam.worldTarget().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.worldTarget().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraMovement.ForwardMovesAlongYawDirection", "[CameraMovement]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaPosition({0.0f, 0.0f, 1.0f}); // forward
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    // yaw=0 → forward_xz = (cos(0), 0, sin(0)) = (1, 0, 0)
    CHECK(cam.localPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraMovement.StrafeMovesPerpendicularToYaw", "[CameraMovement]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaPosition({1.0f, 0.0f, 0.0f}); // strafe right
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    // yaw=0 → right = (sin(0), 0, -cos(0)) = (0, 0, -1)
    CHECK(cam.localPosition().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(-1.0f).margin(1e-5f));
}

TEST_CASE("CameraMovement.ForwardAtYaw90MovesAlongPositiveZ", "[CameraMovement]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(static_cast<float>(M_PI) / 2.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaPosition({0.0f, 0.0f, 1.0f}); // forward
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    // yaw=pi/2 → forward_xz = (cos(pi/2), 0, sin(pi/2)) = (0, 0, 1)
    CHECK(cam.localPosition().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("CameraMovement.PositiveYDeltaMovesUpwardInWorldSpace", "[CameraMovement]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaPosition({0.0f, 1.5f, 0.0f});
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    CHECK(cam.localPosition().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(1.5f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraMovement.NegativeYDeltaMovesDownwardInWorldSpace", "[CameraMovement]")
{
    Camera cam;
    cam.localPosition({0.0f, 2.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaPosition({0.0f, -0.75f, 0.0f});
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    CHECK(cam.localPosition().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(1.25f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraMovement.VerticalMovementCombinesWithYawRelativeMovement", "[CameraMovement]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaPosition({1.0f, 2.0f, 1.0f});
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    CHECK(cam.localPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(-1.0f).margin(1e-5f));
}

// ==========================================================================
// Zoom
// ==========================================================================

TEST_CASE("CameraZoom.ZoomMovesAlongViewDirection", "[CameraZoom]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaZoom(2.0f);
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    // yaw=0, pitch=0 → forward_3d = (1, 0, 0)
    CHECK(cam.localPosition().x() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraZoom.ZoomWithPitchMovesVertically", "[CameraZoom]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(static_cast<float>(M_PI) / 4.0f); // 45 degrees up

    InputState state;
    CameraState cs;
    cs.deltaZoom(1.0f);
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    // forward_3d = (cos(pi/4)*cos(0), sin(pi/4), cos(pi/4)*sin(0))
    float c = std::cos(static_cast<float>(M_PI) / 4.0f);
    float s = std::sin(static_cast<float>(M_PI) / 4.0f);
    CHECK(cam.localPosition().x() == Catch::Approx(c).margin(1e-5f));
    CHECK(cam.localPosition().y() == Catch::Approx(s).margin(1e-5f));
    CHECK(cam.localPosition().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraZoom.NegativeZoomMovesBackward", "[CameraZoom]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaZoom(-1.0f);
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    CHECK(cam.localPosition().x() == Catch::Approx(-1.0f).margin(1e-5f));
}

// ==========================================================================
// Rotation via update
// ==========================================================================

TEST_CASE("CameraRotation.DeltaYawAppliedViaUpdate", "[CameraRotation]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaYaw(0.5f);
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    CHECK(cam.localYaw() == Catch::Approx(0.5f).margin(1e-5f));
}

TEST_CASE("CameraRotation.DeltaPitchAppliedViaUpdate", "[CameraRotation]")
{
    Camera cam;
    cam.localPosition({0.0f, 0.0f, 0.0f});
    cam.localYaw(0.0f);
    cam.localPitch(0.0f);

    InputState state;
    CameraState cs;
    cs.deltaPitch(0.3f);
    state.cameraState(cs);

    Transform t;
    cam.update(state, t);

    CHECK(cam.localPitch() == Catch::Approx(0.3f).margin(1e-5f));
}
