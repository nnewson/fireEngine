#include <fire_engine/scene/animator.hpp>

#include <cmath>

#include <fire_engine/animation/animation.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Animation;
using fire_engine::Animator;
using fire_engine::InputState;
using fire_engine::Mat4;
using fire_engine::Transform;

// ==========================================================================
// Default Construction
// ==========================================================================

TEST_CASE("Animator.DefaultConstructionHasNullAnimation", "[Animator]")
{
    Animator a;
    CHECK(a.animation() == nullptr);
    CHECK(a.animationCount() == 0);
    CHECK(a.activeAnimation() == 0);
}

// ==========================================================================
// Animation Sampling via Update
// ==========================================================================

TEST_CASE("Animator.UpdateWithNoAnimationsProducesIdentity", "[Animator]")
{
    Animator a;
    InputState state;
    Transform transform;
    state.time(1.0);
    a.update(state, transform);

    Mat4 world = Mat4::identity();
    Mat4 result = a.render(world);

    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            CHECK((result[r, c]) == Catch::Approx((world[r, c])).margin(1e-5f));
        }
    }
}

TEST_CASE("Animator.UpdateWithNoKeyframesProducesIdentity", "[Animator]")
{
    Animator a;
    Animation anim;
    a.addAnimation(&anim);

    InputState state;
    Transform transform;
    state.time(1.0);
    a.update(state, transform);

    Mat4 world = Mat4::identity();
    Mat4 result = a.render(world);

    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            CHECK((result[r, c]) == Catch::Approx((world[r, c])).margin(1e-5f));
        }
    }
}

TEST_CASE("Animator.UpdateSamplesAnimationAtElapsedTime", "[Animator]")
{
    Animator a;
    Animation anim;

    // 90 degrees around Y over 2 seconds
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 0.0f, s45, 0.0f, c45},
    });
    a.addAnimation(&anim);

    InputState state;
    Transform transform;

    // First update at t=10.0 initialises startTime
    state.time(10.0);
    a.update(state, transform);

    // Second update at t=11.0 -> elapsed = 1.0s -> midpoint of animation (45 deg Y)
    state.time(11.0);
    a.update(state, transform);

    Mat4 result = a.render(Mat4::identity());

    float cos45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    float sin45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    CHECK((result[0, 0]) == Catch::Approx(cos45).margin(1e-4f));
    CHECK((result[0, 2]) == Catch::Approx(sin45).margin(1e-4f));
    CHECK((result[2, 0]) == Catch::Approx(-sin45).margin(1e-4f));
    CHECK((result[2, 2]) == Catch::Approx(cos45).margin(1e-4f));
}

TEST_CASE("Animator.RenderAppliesWorldMatrix", "[Animator]")
{
    Animator a;
    Animation anim;
    a.addAnimation(&anim);

    InputState state;
    Transform transform;
    state.time(0.0);
    a.update(state, transform);

    // Scale world by 2
    Mat4 world;
    world[0, 0] = 2.0f;
    world[1, 1] = 2.0f;
    world[2, 2] = 2.0f;
    world[3, 3] = 1.0f;

    Mat4 result = a.render(world);

    // With identity animation, result should equal world
    CHECK((result[0, 0]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((result[1, 1]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((result[2, 2]) == Catch::Approx(2.0f).margin(1e-5f));
}

// ==========================================================================
// Multi-Animation Support
// ==========================================================================

TEST_CASE("Animator.AddAnimationIncreasesCount", "[Animator]")
{
    Animator a;
    Animation anim1, anim2, anim3;
    a.addAnimation(&anim1);
    CHECK(a.animationCount() == 1);
    a.addAnimation(&anim2);
    CHECK(a.animationCount() == 2);
    a.addAnimation(&anim3);
    CHECK(a.animationCount() == 3);
}

TEST_CASE("Animator.AnimationReturnsActiveAnimation", "[Animator]")
{
    Animator a;
    Animation anim1, anim2;
    a.addAnimation(&anim1);
    a.addAnimation(&anim2);

    CHECK(a.animation() == &anim1);
    a.activeAnimation(1);
    CHECK(a.animation() == &anim2);
}

TEST_CASE("Animator.ActiveAnimationDefaultsToZero", "[Animator]")
{
    Animator a;
    Animation anim;
    a.addAnimation(&anim);
    CHECK(a.activeAnimation() == 0);
}

TEST_CASE("Animator.ActiveAnimationIgnoresOutOfRangeIndex", "[Animator]")
{
    Animator a;
    Animation anim1, anim2;
    a.addAnimation(&anim1);
    a.addAnimation(&anim2);

    a.activeAnimation(5);
    CHECK(a.activeAnimation() == 0);
    CHECK(a.animation() == &anim1);
}

TEST_CASE("Animator.SwitchingAnimationResetsTimer", "[Animator]")
{
    Animator a;
    Animation anim1;
    anim1.translationKeyframes({
        {0.0f, {0.0f, 0.0f, 0.0f}},
        {1.0f, {10.0f, 0.0f, 0.0f}},
    });

    Animation anim2;
    anim2.translationKeyframes({
        {0.0f, {0.0f, 0.0f, 0.0f}},
        {1.0f, {0.0f, 20.0f, 0.0f}},
    });

    a.addAnimation(&anim1);
    a.addAnimation(&anim2);

    InputState state;
    Transform transform;

    // Play anim1 for a bit
    state.time(100.0);
    a.update(state, transform);
    state.time(100.5);
    a.update(state, transform);

    // Switch to anim2 — timer should reset
    a.activeAnimation(1);
    state.time(200.0);
    a.update(state, transform);
    state.time(200.5);
    a.update(state, transform);

    Mat4 result = a.render(Mat4::identity());

    // At t=0.5 into anim2, Y translation should be ~10.0
    CHECK((result[1, 3]) == Catch::Approx(10.0f).margin(1e-4f));
    // X translation should be 0 (anim2 doesn't translate X)
    CHECK((result[0, 3]) == Catch::Approx(0.0f).margin(1e-4f));
}

// ==========================================================================
// InputState-Driven Animation Switching
// ==========================================================================

TEST_CASE("Animator.InputStateSwitchesAnimation", "[Animator]")
{
    Animator a;
    Animation anim1;
    anim1.translationKeyframes({
        {0.0f, {0.0f, 0.0f, 0.0f}},
        {1.0f, {10.0f, 0.0f, 0.0f}},
    });
    Animation anim2;
    anim2.translationKeyframes({
        {0.0f, {0.0f, 0.0f, 0.0f}},
        {1.0f, {0.0f, 20.0f, 0.0f}},
    });
    a.addAnimation(&anim1);
    a.addAnimation(&anim2);

    InputState state;
    Transform transform;

    // Start playing anim1
    state.time(0.0);
    a.update(state, transform);

    // Switch to anim2 via InputState
    state.time(1.0);
    state.animationState().activeAnimation(1);
    a.update(state, transform);

    CHECK(a.activeAnimation() == 1);

    // Advance time and sample
    state.time(1.5);
    InputState noSwitch;
    noSwitch.time(1.5);
    a.update(noSwitch, transform);

    Mat4 result = a.render(Mat4::identity());

    // At t=0.5 into anim2, Y translation should be ~10.0
    CHECK((result[1, 3]) == Catch::Approx(10.0f).margin(1e-4f));
    CHECK((result[0, 3]) == Catch::Approx(0.0f).margin(1e-4f));
}

TEST_CASE("Animator.InputStateUsesStableAnimationIds", "[Animator]")
{
    Animator a;

    Animation anim10;
    anim10.translationKeyframes({
        {0.0f, {0.0f, 0.0f, 0.0f}},
        {1.0f, {10.0f, 0.0f, 0.0f}},
    });

    Animation anim20;
    anim20.translationKeyframes({
        {0.0f, {0.0f, 0.0f, 0.0f}},
        {1.0f, {0.0f, 20.0f, 0.0f}},
    });

    a.addAnimation(10, &anim10);
    a.addAnimation(20, &anim20);

    InputState state;
    Transform transform;

    state.time(0.0);
    a.update(state, transform);

    state.time(1.0);
    state.animationState().activeAnimation(20);
    a.update(state, transform);

    CHECK(a.activeAnimation() == 20u);
    CHECK(a.animation() == &anim20);
}

TEST_CASE("Animator.InputStateOutOfRangeIndexIsIgnored", "[Animator]")
{
    Animator a;
    Animation anim1, anim2;
    a.addAnimation(&anim1);
    a.addAnimation(&anim2);

    InputState state;
    Transform transform;

    state.time(0.0);
    a.update(state, transform);

    // Try to switch to index 99
    state.animationState().activeAnimation(99);
    a.update(state, transform);

    CHECK(a.activeAnimation() == 0);
}

TEST_CASE("Animator.InputStateNoSelectionContinuesPlayback", "[Animator]")
{
    Animator a;
    Animation anim;
    anim.translationKeyframes({
        {0.0f, {0.0f, 0.0f, 0.0f}},
        {2.0f, {20.0f, 0.0f, 0.0f}},
    });
    a.addAnimation(&anim);

    InputState state;
    Transform transform;

    // Play for a bit
    state.time(10.0);
    a.update(state, transform);
    state.time(11.0);
    a.update(state, transform);

    Mat4 result = a.render(Mat4::identity());

    // At t=1.0, X translation should be ~10.0
    CHECK((result[0, 3]) == Catch::Approx(10.0f).margin(1e-4f));

    // Continue with no animation selection — should not reset
    InputState state2;
    state2.time(11.5);
    a.update(state2, transform);
    result = a.render(Mat4::identity());

    // At t=1.5, X translation should be ~15.0
    CHECK((result[0, 3]) == Catch::Approx(15.0f).margin(1e-4f));
}

TEST_CASE("Animator.InputStateSameIndexDoesNotResetTimer", "[Animator]")
{
    Animator a;
    Animation anim;
    anim.translationKeyframes({
        {0.0f, {0.0f, 0.0f, 0.0f}},
        {2.0f, {20.0f, 0.0f, 0.0f}},
    });
    a.addAnimation(&anim);

    InputState state;
    Transform transform;

    // Start playing
    state.time(10.0);
    a.update(state, transform);
    state.time(11.0);
    a.update(state, transform);

    // "Press key 1" again — same index 0, should NOT reset
    state.time(11.5);
    state.animationState().activeAnimation(0);
    a.update(state, transform);

    Mat4 result = a.render(Mat4::identity());

    // At t=1.5 elapsed, X translation should be ~15.0 (not reset to 0)
    CHECK((result[0, 3]) == Catch::Approx(15.0f).margin(1e-4f));
}
