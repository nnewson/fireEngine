#include <fire_engine/input/controller_state.hpp>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::ControllerState;
using fire_engine::Vec3;

// ==========================================================================
// Default Construction
// ==========================================================================

TEST_CASE("ControllerStateConstruction.DefaultDeltaPositionIsZero", "[ControllerStateConstruction]")
{
    ControllerState s;
    CHECK(s.deltaPosition().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(s.deltaPosition().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(s.deltaPosition().z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("ControllerStateConstruction.DefaultTimeIsZero", "[ControllerStateConstruction]")
{
    ControllerState s;
    CHECK(s.time() == Catch::Approx(0.0).margin(1e-5f));
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("ControllerStateAccessors.SetDeltaPosition", "[ControllerStateAccessors]")
{
    ControllerState s;
    s.deltaPosition({1.0f, 2.0f, 3.0f});
    CHECK(s.deltaPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(s.deltaPosition().y() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(s.deltaPosition().z() == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("ControllerStateAccessors.SetTime", "[ControllerStateAccessors]")
{
    ControllerState s;
    s.time(1.5);
    CHECK(s.time() == Catch::Approx(1.5).margin(1e-5f));
}

// ==========================================================================
// Copy and Move Semantics
// ==========================================================================

TEST_CASE("ControllerStateCopy.CopyConstructCreatesIndependentCopy", "[ControllerStateCopy]")
{
    ControllerState a;
    a.deltaPosition({1.0f, 2.0f, 3.0f});
    a.time(4.0);

    ControllerState b{a};
    CHECK(b.deltaPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(b.time() == Catch::Approx(4.0).margin(1e-5f));

    b.deltaPosition({9.0f, 0.0f, 0.0f});
    CHECK(a.deltaPosition().x() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("ControllerStateCopy.CopyAssign", "[ControllerStateCopy]")
{
    ControllerState a;
    a.deltaPosition({4.0f, 5.0f, 6.0f});
    ControllerState b;
    b = a;
    CHECK(b.deltaPosition().x() == Catch::Approx(4.0f).margin(1e-5f));
}

TEST_CASE("ControllerStateMove.MoveConstructTransfersState", "[ControllerStateMove]")
{
    ControllerState a;
    a.deltaPosition({7.0f, 8.0f, 9.0f});
    a.time(2.0);

    ControllerState b{std::move(a)};
    CHECK(b.deltaPosition().x() == Catch::Approx(7.0f).margin(1e-5f));
    CHECK(b.time() == Catch::Approx(2.0).margin(1e-5f));
}

// ==========================================================================
// Noexcept guarantees
// ==========================================================================

TEST_CASE("ControllerStateNoexcept.AllOperationsAreNoexcept", "[ControllerStateNoexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<ControllerState>);
    static_assert(test_traits::has_nothrow_controller_state_operations<ControllerState, Vec3>);
    static_assert(std::is_nothrow_move_constructible_v<ControllerState>);
}
