#include <fire_engine/scene/animator.hpp>

#include <cmath>

#include <fire_engine/animation/animation.hpp>

#include <gtest/gtest.h>

using fire_engine::Animation;
using fire_engine::Animator;
using fire_engine::InputState;
using fire_engine::Mat4;
using fire_engine::Transform;

// ==========================================================================
// Default Construction
// ==========================================================================

TEST(Animator, DefaultConstructionHasNullAnimation)
{
    Animator a;
    EXPECT_EQ(a.animation(), nullptr);
    EXPECT_EQ(a.animationCount(), 0);
    EXPECT_EQ(a.activeAnimation(), 0);
}

// ==========================================================================
// Animation Sampling via Update
// ==========================================================================

TEST(Animator, UpdateWithNoAnimationsProducesIdentity)
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
            EXPECT_NEAR((result[r, c]), (world[r, c]), 1e-5f);
        }
    }
}

TEST(Animator, UpdateWithNoKeyframesProducesIdentity)
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
            EXPECT_NEAR((result[r, c]), (world[r, c]), 1e-5f);
        }
    }
}

TEST(Animator, UpdateSamplesAnimationAtElapsedTime)
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
    EXPECT_NEAR((result[0, 0]), cos45, 1e-4f);
    EXPECT_NEAR((result[0, 2]), sin45, 1e-4f);
    EXPECT_NEAR((result[2, 0]), -sin45, 1e-4f);
    EXPECT_NEAR((result[2, 2]), cos45, 1e-4f);
}

TEST(Animator, RenderAppliesWorldMatrix)
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
    EXPECT_NEAR((result[0, 0]), 2.0f, 1e-5f);
    EXPECT_NEAR((result[1, 1]), 2.0f, 1e-5f);
    EXPECT_NEAR((result[2, 2]), 2.0f, 1e-5f);
}

// ==========================================================================
// Multi-Animation Support
// ==========================================================================

TEST(Animator, AddAnimationIncreasesCount)
{
    Animator a;
    Animation anim1, anim2, anim3;
    a.addAnimation(&anim1);
    EXPECT_EQ(a.animationCount(), 1);
    a.addAnimation(&anim2);
    EXPECT_EQ(a.animationCount(), 2);
    a.addAnimation(&anim3);
    EXPECT_EQ(a.animationCount(), 3);
}

TEST(Animator, AnimationReturnsActiveAnimation)
{
    Animator a;
    Animation anim1, anim2;
    a.addAnimation(&anim1);
    a.addAnimation(&anim2);

    EXPECT_EQ(a.animation(), &anim1);
    a.activeAnimation(1);
    EXPECT_EQ(a.animation(), &anim2);
}

TEST(Animator, ActiveAnimationDefaultsToZero)
{
    Animator a;
    Animation anim;
    a.addAnimation(&anim);
    EXPECT_EQ(a.activeAnimation(), 0);
}

TEST(Animator, ActiveAnimationIgnoresOutOfRangeIndex)
{
    Animator a;
    Animation anim1, anim2;
    a.addAnimation(&anim1);
    a.addAnimation(&anim2);

    a.activeAnimation(5);
    EXPECT_EQ(a.activeAnimation(), 0);
    EXPECT_EQ(a.animation(), &anim1);
}

TEST(Animator, SwitchingAnimationResetsTimer)
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
    EXPECT_NEAR((result[1, 3]), 10.0f, 1e-4f);
    // X translation should be 0 (anim2 doesn't translate X)
    EXPECT_NEAR((result[0, 3]), 0.0f, 1e-4f);
}

// ==========================================================================
// InputState-Driven Animation Switching
// ==========================================================================

TEST(Animator, InputStateSwitchesAnimation)
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

    EXPECT_EQ(a.activeAnimation(), 1);

    // Advance time and sample
    state.time(1.5);
    InputState noSwitch;
    noSwitch.time(1.5);
    a.update(noSwitch, transform);

    Mat4 result = a.render(Mat4::identity());

    // At t=0.5 into anim2, Y translation should be ~10.0
    EXPECT_NEAR((result[1, 3]), 10.0f, 1e-4f);
    EXPECT_NEAR((result[0, 3]), 0.0f, 1e-4f);
}

TEST(Animator, InputStateUsesStableAnimationIds)
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

    EXPECT_EQ(a.activeAnimation(), 20u);
    EXPECT_EQ(a.animation(), &anim20);
}

TEST(Animator, InputStateOutOfRangeIndexIsIgnored)
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

    EXPECT_EQ(a.activeAnimation(), 0);
}

TEST(Animator, InputStateNoSelectionContinuesPlayback)
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
    EXPECT_NEAR((result[0, 3]), 10.0f, 1e-4f);

    // Continue with no animation selection — should not reset
    InputState state2;
    state2.time(11.5);
    a.update(state2, transform);
    result = a.render(Mat4::identity());

    // At t=1.5, X translation should be ~15.0
    EXPECT_NEAR((result[0, 3]), 15.0f, 1e-4f);
}

TEST(Animator, InputStateSameIndexDoesNotResetTimer)
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
    EXPECT_NEAR((result[0, 3]), 15.0f, 1e-4f);
}
