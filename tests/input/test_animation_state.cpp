#include <fire_engine/input/animation_state.hpp>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::AnimationState;

// ==========================================================================
// Default Construction
// ==========================================================================

TEST_CASE("AnimationStateConstruction.DefaultHasNoActiveAnimation", "[AnimationStateConstruction]")
{
    AnimationState s;
    CHECK_FALSE(s.hasActiveAnimation());
    CHECK_FALSE(s.activeAnimation().has_value());
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("AnimationStateAccessors.SetActiveAnimation", "[AnimationStateAccessors]")
{
    AnimationState s;
    s.activeAnimation(2);
    CHECK(s.hasActiveAnimation());
    CHECK(*s.activeAnimation() == 2);
}

TEST_CASE("AnimationStateAccessors.OverwriteActiveAnimation", "[AnimationStateAccessors]")
{
    AnimationState s;
    s.activeAnimation(1);
    s.activeAnimation(0);
    CHECK(*s.activeAnimation() == 0);
}

// ==========================================================================
// Copy / Move
// ==========================================================================

TEST_CASE("AnimationStateCopy.CopyConstructCreatesIndependentCopy", "[AnimationStateCopy]")
{
    AnimationState a;
    a.activeAnimation(3);

    AnimationState b{a};
    CHECK(*b.activeAnimation() == 3);

    b.activeAnimation(5);
    CHECK(*a.activeAnimation() == 3);
}

TEST_CASE("AnimationStateCopy.CopyAssign", "[AnimationStateCopy]")
{
    AnimationState a;
    a.activeAnimation(7);
    AnimationState b;
    b = a;
    CHECK(*b.activeAnimation() == 7);
}

TEST_CASE("AnimationStateMove.MoveConstructTransfersState", "[AnimationStateMove]")
{
    AnimationState a;
    a.activeAnimation(4);
    AnimationState b{std::move(a)};
    CHECK(*b.activeAnimation() == 4);
}

// ==========================================================================
// Noexcept
// ==========================================================================

TEST_CASE("AnimationStateNoexcept.AllOperationsAreNoexcept", "[AnimationStateNoexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<AnimationState>);
    static_assert(test_traits::has_nothrow_animation_state_operations<AnimationState>);
    static_assert(std::is_nothrow_move_constructible_v<AnimationState>);
}
