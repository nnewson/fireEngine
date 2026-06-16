#include <fire_engine/input/camera_state.hpp>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::CameraState;
using fire_engine::Vec3;

// ==========================================================================
// Default Construction
// ==========================================================================

TEST_CASE("CameraStateConstruction.DefaultDeltaPositionIsZero", "[CameraStateConstruction]")
{
    CameraState s;
    CHECK(s.deltaPosition().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(s.deltaPosition().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(s.deltaPosition().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraStateConstruction.DefaultDeltaYawIsZero", "[CameraStateConstruction]")
{
    CameraState s;
    CHECK(s.deltaYaw() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraStateConstruction.DefaultDeltaPitchIsZero", "[CameraStateConstruction]")
{
    CameraState s;
    CHECK(s.deltaPitch() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("CameraStateConstruction.DefaultTimeIsZero", "[CameraStateConstruction]")
{
    CameraState s;
    CHECK(s.time() == Catch::Approx(0.0).margin(1e-5f));
}

TEST_CASE("CameraStateConstruction.DefaultDeltaZoomIsZero", "[CameraStateConstruction]")
{
    CameraState s;
    CHECK(s.deltaZoom() == Catch::Approx(0.0f).margin(1e-5f));
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("CameraStateAccessors.SetDeltaPosition", "[CameraStateAccessors]")
{
    CameraState s;
    s.deltaPosition({1.0f, 2.0f, 3.0f});
    CHECK(s.deltaPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(s.deltaPosition().y() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(s.deltaPosition().z() == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("CameraStateAccessors.SetDeltaYaw", "[CameraStateAccessors]")
{
    CameraState s;
    s.deltaYaw(0.5f);
    CHECK(s.deltaYaw() == Catch::Approx(0.5f).margin(1e-5f));
}

TEST_CASE("CameraStateAccessors.SetDeltaPitch", "[CameraStateAccessors]")
{
    CameraState s;
    s.deltaPitch(-0.3f);
    CHECK(s.deltaPitch() == Catch::Approx(-0.3f).margin(1e-5f));
}

TEST_CASE("CameraStateAccessors.SetTime", "[CameraStateAccessors]")
{
    CameraState s;
    s.time(1.5);
    CHECK(s.time() == Catch::Approx(1.5).margin(1e-5f));
}

TEST_CASE("CameraStateAccessors.OverwritePreviousValues", "[CameraStateAccessors]")
{
    CameraState s;
    s.deltaYaw(1.0f);
    s.deltaYaw(2.0f);
    CHECK(s.deltaYaw() == Catch::Approx(2.0f).margin(1e-5f));
}

TEST_CASE("CameraStateAccessors.SetDeltaZoom", "[CameraStateAccessors]")
{
    CameraState s;
    s.deltaZoom(2.5f);
    CHECK(s.deltaZoom() == Catch::Approx(2.5f).margin(1e-5f));
}

// ==========================================================================
// Copy and Move Semantics
// ==========================================================================

TEST_CASE("CameraStateCopy.CopyConstructCreatesIndependentCopy", "[CameraStateCopy]")
{
    CameraState a;
    a.deltaPosition({1.0f, 2.0f, 3.0f});
    a.deltaYaw(0.5f);
    a.deltaPitch(-0.3f);

    CameraState b{a};
    CHECK(b.deltaPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(b.deltaYaw() == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(b.deltaPitch() == Catch::Approx(-0.3f).margin(1e-5f));

    b.deltaYaw(9.0f);
    CHECK(a.deltaYaw() == Catch::Approx(0.5f).margin(1e-5f));
}

TEST_CASE("CameraStateCopy.CopyAssign", "[CameraStateCopy]")
{
    CameraState a;
    a.deltaPosition({4.0f, 5.0f, 6.0f});
    CameraState b;
    b = a;
    CHECK(b.deltaPosition().x() == Catch::Approx(4.0f).margin(1e-5f));
}

TEST_CASE("CameraStateMove.MoveConstructTransfersState", "[CameraStateMove]")
{
    CameraState a;
    a.deltaPosition({7.0f, 8.0f, 9.0f});
    a.deltaYaw(1.5f);

    CameraState b{std::move(a)};
    CHECK(b.deltaPosition().x() == Catch::Approx(7.0f).margin(1e-5f));
    CHECK(b.deltaYaw() == Catch::Approx(1.5f).margin(1e-5f));
}

// ==========================================================================
// Noexcept guarantees
// ==========================================================================

TEST_CASE("CameraStateNoexcept.AllOperationsAreNoexcept", "[CameraStateNoexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<CameraState>);
    static_assert(test_traits::has_nothrow_camera_state_operations<CameraState, Vec3>);
}
