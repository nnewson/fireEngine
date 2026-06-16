#include <fire_engine/input/variant_state.hpp>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::VariantState;

TEST_CASE("VariantStateConstruction.DefaultHasNoCycleCommand", "[VariantStateConstruction]")
{
    VariantState s;
    CHECK(s.cycleDelta() == 0);
    CHECK_FALSE(s.hasCycleCommand());
}

TEST_CASE("VariantStateAccessors.PositiveCycleRoundTrip", "[VariantStateAccessors]")
{
    VariantState s;
    s.cycleDelta(1);
    CHECK(s.cycleDelta() == 1);
    CHECK(s.hasCycleCommand());
}

TEST_CASE("VariantStateAccessors.NegativeCycleRoundTrip", "[VariantStateAccessors]")
{
    VariantState s;
    s.cycleDelta(-1);
    CHECK(s.cycleDelta() == -1);
    CHECK(s.hasCycleCommand());
}

TEST_CASE("VariantStateNoexcept.OperationsAreNoexcept", "[VariantStateNoexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<VariantState>);
    static_assert(test_traits::has_nothrow_variant_state_operations<VariantState>);
    static_assert(std::is_nothrow_move_constructible_v<VariantState>);
}
