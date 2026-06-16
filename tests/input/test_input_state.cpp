#include <fire_engine/input/input_state.hpp>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::AnimationState;
using fire_engine::CameraState;
using fire_engine::ControllerState;
using fire_engine::InputState;
using fire_engine::VariantState;
using fire_engine::Vec3;

// ==========================================================================
// Default Construction
// ==========================================================================

TEST_CASE("InputStateConstruction.DefaultCameraStateIsZero", "[InputStateConstruction]")
{
    InputState s;
    CHECK(s.cameraState().deltaYaw() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(s.cameraState().deltaPitch() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(s.time() == Catch::Approx(0.0).margin(1e-5f));
}

TEST_CASE("InputStateConstruction.DefaultAnimationStateHasNoActive", "[InputStateConstruction]")
{
    InputState s;
    CHECK_FALSE(s.animationState().hasActiveAnimation());
}

TEST_CASE("InputStateConstruction.DefaultControllerStateIsZero", "[InputStateConstruction]")
{
    InputState s;
    CHECK(s.controllerState().deltaPosition().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(s.controllerState().time() == Catch::Approx(0.0).margin(1e-5f));
}

TEST_CASE("InputStateConstruction.DefaultDeltaTimeIsZero", "[InputStateConstruction]")
{
    InputState s;
    CHECK(s.deltaTime() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("InputStateConstruction.DefaultVariantStateHasNoCycleCommand", "[InputStateConstruction]")
{
    InputState s;
    CHECK_FALSE(s.variantState().hasCycleCommand());
    CHECK(s.variantState().cycleDelta() == 0);
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("InputStateAccessors.SetCameraState", "[InputStateAccessors]")
{
    InputState s;
    CameraState cs;
    cs.deltaYaw(1.5f);
    s.cameraState(cs);
    CHECK(s.cameraState().deltaYaw() == Catch::Approx(1.5f).margin(1e-5f));
}

TEST_CASE("InputStateAccessors.SetAnimationState", "[InputStateAccessors]")
{
    InputState s;
    AnimationState as;
    as.activeAnimation(2);
    s.animationState(as);
    CHECK(*s.animationState().activeAnimation() == 2);
}

TEST_CASE("InputStateAccessors.SetControllerState", "[InputStateAccessors]")
{
    InputState s;
    ControllerState cs;
    cs.deltaPosition({1.0f, 0.0f, 0.0f});
    s.controllerState(cs);
    CHECK(s.controllerState().deltaPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("InputStateAccessors.SetVariantState", "[InputStateAccessors]")
{
    InputState s;
    VariantState vs;
    vs.cycleDelta(1);
    s.variantState(vs);
    CHECK(s.variantState().cycleDelta() == 1);
}

TEST_CASE("InputStateAccessors.TimeConvenienceAccessor", "[InputStateAccessors]")
{
    InputState s;
    s.time(42.0);
    CHECK(s.time() == Catch::Approx(42.0).margin(1e-5f));
    CHECK(s.cameraState().time() == Catch::Approx(42.0).margin(1e-5f));
    CHECK(s.controllerState().time() == Catch::Approx(42.0).margin(1e-5f));
}

TEST_CASE("InputStateAccessors.SetDeltaTime", "[InputStateAccessors]")
{
    InputState s;
    s.deltaTime(1.25f);
    CHECK(s.deltaTime() == Catch::Approx(1.25f).margin(1e-5f));
}

TEST_CASE("InputStateAccessors.MutableCameraStateRef", "[InputStateAccessors]")
{
    InputState s;
    s.cameraState().deltaPosition({1.0f, 2.0f, 3.0f});
    CHECK(s.cameraState().deltaPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("InputStateAccessors.MutableAnimationStateRef", "[InputStateAccessors]")
{
    InputState s;
    s.animationState().activeAnimation(1);
    CHECK(*s.animationState().activeAnimation() == 1);
}

TEST_CASE("InputStateAccessors.MutableControllerStateRef", "[InputStateAccessors]")
{
    InputState s;
    s.controllerState().deltaPosition({2.0f, 0.0f, 0.0f});
    CHECK(s.controllerState().deltaPosition().x() == Catch::Approx(2.0f).margin(1e-5f));
}

TEST_CASE("InputStateAccessors.MutableVariantStateRef", "[InputStateAccessors]")
{
    InputState s;
    s.variantState().cycleDelta(-1);
    CHECK(s.variantState().cycleDelta() == -1);
}

// ==========================================================================
// Copy / Move
// ==========================================================================

TEST_CASE("InputStateCopy.CopyConstructCreatesIndependentCopy", "[InputStateCopy]")
{
    InputState a;
    a.time(10.0);
    a.animationState().activeAnimation(5);
    a.controllerState().deltaPosition({1.0f, 0.0f, 0.0f});
    a.variantState().cycleDelta(1);

    InputState b{a};
    CHECK(b.time() == Catch::Approx(10.0).margin(1e-5f));
    CHECK(*b.animationState().activeAnimation() == 5);
    CHECK(b.controllerState().deltaPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(b.variantState().cycleDelta() == 1);

    b.time(20.0);
    CHECK(a.time() == Catch::Approx(10.0).margin(1e-5f));
}

TEST_CASE("InputStateCopy.CopyAssign", "[InputStateCopy]")
{
    InputState a;
    a.time(7.0);
    InputState b;
    b = a;
    CHECK(b.time() == Catch::Approx(7.0).margin(1e-5f));
}

TEST_CASE("InputStateMove.MoveConstructTransfersState", "[InputStateMove]")
{
    InputState a;
    a.time(33.0);
    a.animationState().activeAnimation(2);
    a.controllerState().deltaPosition({3.0f, 0.0f, 0.0f});
    InputState b{std::move(a)};
    CHECK(b.time() == Catch::Approx(33.0).margin(1e-5f));
    CHECK(*b.animationState().activeAnimation() == 2);
    CHECK(b.controllerState().deltaPosition().x() == Catch::Approx(3.0f).margin(1e-5f));
}

// ==========================================================================
// Noexcept
// ==========================================================================

TEST_CASE("InputStateNoexcept.AllOperationsAreNoexcept", "[InputStateNoexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<InputState>);
    static_assert(
        test_traits::has_nothrow_input_state_operations<InputState, CameraState, AnimationState,
                                                        ControllerState, VariantState>);
    static_assert(std::is_nothrow_move_constructible_v<InputState>);
}
