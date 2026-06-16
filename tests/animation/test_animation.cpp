#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <vector>

#include <fire_engine/animation/animation.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <support/test_traits.hpp>

using fire_engine::Animation;
using fire_engine::Mat4;
using fire_engine::Vec3;

namespace
{

void expectWeightsNear(const std::vector<float>& weights, std::initializer_list<float> expected,
                       float eps = 1e-5f)
{
    REQUIRE(weights.size() == expected.size());

    auto expectedIt = expected.begin();
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        CAPTURE(i);
        CHECK(weights[i] == Catch::Approx(*(expectedIt + i)).margin(eps));
    }
}

} // namespace

TEST_CASE("Animation.DefaultConstructionHasNoKeyframes", "[Animation]")
{
    Animation anim;
    CHECK(anim.rotationKeyframes().empty());
    CHECK(anim.translationKeyframes().empty());
    CHECK(anim.scaleKeyframes().empty());
    CHECK(anim.duration() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Animation.SetKeyframesAndDuration", "[Animation]")
{
    Animation anim;
    std::vector<Animation::RotationKeyframe> kf = {
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.7071f, 0.0f, 0.7071f},
        {2.0f, 0.0f, 1.0f, 0.0f, 0.0f},
    };
    anim.rotationKeyframes(kf);
    CHECK(anim.rotationKeyframes().size() == 3u);
    CHECK(anim.duration() == Catch::Approx(2.0f).margin(1e-5f));
}

TEST_CASE("Animation.SampleEmptyReturnsIdentity", "[Animation]")
{
    Animation anim;
    Mat4 result = anim.sample(0.5f);
    Mat4 identity = Mat4::identity();
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            CHECK((result[r, c]) == Catch::Approx((identity[r, c])).margin(1e-5f));
        }
    }
}

TEST_CASE("Animation.SampleSingleKeyframeReturnsConstant", "[Animation]")
{
    Animation anim;
    // Identity quaternion
    anim.rotationKeyframes({{0.0f, 0.0f, 0.0f, 0.0f, 1.0f}});

    Mat4 result = anim.sample(5.0f);
    Mat4 identity = Mat4::identity();
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            CHECK((result[r, c]) == Catch::Approx((identity[r, c])).margin(1e-5f));
        }
    }
}

TEST_CASE("Animation.SampleAtExactKeyframeTime", "[Animation]")
{
    Animation anim;
    // Keyframe at t=0: identity quaternion (no rotation)
    // Keyframe at t=1: 90 degrees around Y axis -> quat (0, sin(45), 0, cos(45))
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 0.0f, s45, 0.0f, c45},
    });

    // Sample at t=0 should be identity
    Mat4 atZero = anim.sample(0.0f);
    CHECK((atZero[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((atZero[1, 1]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((atZero[2, 2]) == Catch::Approx(1.0f).margin(1e-5f));

    // Sample near end should approach 90-degree Y rotation
    // For 90 deg Y rotation: m[0,0]=0, m[0,2]=1, m[2,0]=-1, m[2,2]=0
    Mat4 atEnd = anim.sample(1.999f);
    CHECK((atEnd[0, 0]) == Catch::Approx(0.0f).margin(2e-2f));
    CHECK((atEnd[0, 2]) == Catch::Approx(1.0f).margin(2e-2f));
    CHECK((atEnd[2, 0]) == Catch::Approx(-1.0f).margin(2e-2f));
    CHECK((atEnd[2, 2]) == Catch::Approx(0.0f).margin(2e-2f));
    CHECK((atEnd[1, 1]) == Catch::Approx(1.0f).margin(2e-2f));
}

TEST_CASE("Animation.SampleInterpolatesBetweenKeyframes", "[Animation]")
{
    Animation anim;
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 0.0f, s45, 0.0f, c45},
    });

    // Sample at t=1 (midpoint) should be ~45 degrees around Y
    Mat4 mid = anim.sample(1.0f);
    // cos(45) ~= 0.7071
    float cos45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    float sin45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    CHECK((mid[0, 0]) == Catch::Approx(cos45).margin(1e-4f));
    CHECK((mid[0, 2]) == Catch::Approx(sin45).margin(1e-4f));
    CHECK((mid[2, 0]) == Catch::Approx(-sin45).margin(1e-4f));
    CHECK((mid[2, 2]) == Catch::Approx(cos45).margin(1e-4f));
}

TEST_CASE("Animation.SampleLoops", "[Animation]")
{
    Animation anim;
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 0.0f, 1.0f, 0.0f, 0.0f},
    });

    // Sample at t=0 and t=2.0 (one full loop) should be the same
    Mat4 atZero = anim.sample(0.0f);
    Mat4 atLoop = anim.sample(2.0f);
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            CHECK((atZero[r, c]) == Catch::Approx((atLoop[r, c])).margin(1e-4f));
        }
    }
}

TEST_CASE("Animation.SampleAndDurationAreNoexcept", "[Animation]")
{
    static_assert(test_traits::has_nothrow_animation_sampling<Animation>);
}

// ==========================================================================
// Translation sampling
// ==========================================================================

TEST_CASE("Animation.EmptyTranslationSamplesToZeroOffset", "[Animation]")
{
    Animation anim;
    // Give it a single rotation keyframe so sample() doesn't short-circuit to identity
    anim.rotationKeyframes({{0.0f, 0.0f, 0.0f, 0.0f, 1.0f}});
    Mat4 m = anim.sample(0.5f);
    // Translation column should be zero
    CHECK((m[0, 3]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((m[1, 3]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((m[2, 3]) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Animation.SingleTranslationKeyframeIsConstant", "[Animation]")
{
    Animation anim;
    anim.translationKeyframes({{0.0f, Vec3{1.0f, 2.0f, 3.0f}}});
    Mat4 m = anim.sample(42.0f);
    CHECK((m[0, 3]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[1, 3]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((m[2, 3]) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("Animation.TranslationInterpolatesLinearly", "[Animation]")
{
    Animation anim;
    anim.translationKeyframes({
        {0.0f, Vec3{0.0f, 0.0f, 0.0f}},
        {2.0f, Vec3{2.0f, 4.0f, 6.0f}},
    });
    Mat4 m = anim.sample(1.0f);
    CHECK((m[0, 3]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[1, 3]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((m[2, 3]) == Catch::Approx(3.0f).margin(1e-5f));
}

// ==========================================================================
// Duration = max across channels
// ==========================================================================

TEST_CASE("Animation.DurationIsMaxAcrossChannels", "[Animation]")
{
    Animation anim;
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
    });
    anim.translationKeyframes({
        {0.0f, Vec3{}},
        {2.0f, Vec3{1.0f, 0.0f, 0.0f}},
    });
    CHECK(anim.duration() == Catch::Approx(2.0f).margin(1e-5f));
}

// ==========================================================================
// Composite T * R
// ==========================================================================

TEST_CASE("Animation.CompositeTranslationAndRotationMidpoint", "[Animation]")
{
    Animation anim;

    // Rotation: 0° -> 90° about Y over 2s
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 0.0f, s45, 0.0f, c45},
    });

    // Translation: (0,0,0) -> (2,0,0) over 2s
    anim.translationKeyframes({
        {0.0f, Vec3{0.0f, 0.0f, 0.0f}},
        {2.0f, Vec3{2.0f, 0.0f, 0.0f}},
    });

    // At t=1s: rotation is 45° about Y, translation is (1,0,0)
    Mat4 m = anim.sample(1.0f);

    // Rotation part (upper-left 3x3) should match 45° Y rotation
    CHECK((m[0, 0]) == Catch::Approx(c45).margin(1e-4f));
    CHECK((m[0, 2]) == Catch::Approx(s45).margin(1e-4f));
    CHECK((m[2, 0]) == Catch::Approx(-s45).margin(1e-4f));
    CHECK((m[2, 2]) == Catch::Approx(c45).margin(1e-4f));
    CHECK((m[1, 1]) == Catch::Approx(1.0f).margin(1e-4f));

    // Translation column should be (1, 0, 0) — T * R leaves translation in last column
    CHECK((m[0, 3]) == Catch::Approx(1.0f).margin(1e-4f));
    CHECK((m[1, 3]) == Catch::Approx(0.0f).margin(1e-4f));
    CHECK((m[2, 3]) == Catch::Approx(0.0f).margin(1e-4f));
}

// ==========================================================================
// Channels with different durations clamp independently
// ==========================================================================

TEST_CASE("Animation.ExplicitDurationOverridesComputedMax", "[Animation]")
{
    Animation anim;
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
    });
    // Computed max would be 1.0f — force it to 5.0f so this animation loops in
    // lockstep with a sibling that runs longer.
    anim.duration(5.0f);
    CHECK(anim.duration() == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("Animation.SampleLoopsOnExplicitDurationAndClampsShortChannel", "[Animation]")
{
    Animation anim;
    // Rotation ends at t=1.0 — 0° to 90° about Y.
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, s45, 0.0f, c45},
    });
    // Force duration to 2.0 — simulates a sibling channel running to t=2.0.
    anim.duration(2.0f);

    // At t=1.5 (past rotation's last key, still within shared duration), rotation
    // must clamp to last key (90° Y), NOT loop early.
    Mat4 m = anim.sample(1.5f);
    CHECK((m[0, 0]) == Catch::Approx(0.0f).margin(1e-4f));
    CHECK((m[0, 2]) == Catch::Approx(1.0f).margin(1e-4f));
    CHECK((m[2, 0]) == Catch::Approx(-1.0f).margin(1e-4f));
    CHECK((m[2, 2]) == Catch::Approx(0.0f).margin(1e-4f));

    // At t=2.5 we wrap to 0.5 on the shared duration — half of rotation → 45° Y.
    Mat4 wrapped = anim.sample(2.5f);
    CHECK((wrapped[0, 0]) == Catch::Approx(c45).margin(1e-4f));
    CHECK((wrapped[0, 2]) == Catch::Approx(s45).margin(1e-4f));
}

TEST_CASE("Animation.RotationClampsToFirstKeyframeWhenSampledBeforeItsStart", "[Animation]")
{
    Animation anim;
    // Rotation channel starts at t=1.0 with identity, goes to 90° about Y at t=2.0 —
    // mirrors BoxAnimated where the rotation window starts mid-animation.
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 0.0f, s45, 0.0f, c45},
    });
    // Give it a translation so the animation's duration extends back to t=0.
    anim.translationKeyframes({
        {0.0f, Vec3{0.0f, 0.0f, 0.0f}},
        {2.0f, Vec3{0.0f, 0.0f, 0.0f}},
    });

    // At t=0.5 (before the rotation's first keyframe), the rotation must clamp
    // to the first keyframe (identity), NOT extrapolate backwards via slerp.
    Mat4 m = anim.sample(0.5f);

    CHECK((m[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[1, 1]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[2, 2]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[0, 2]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((m[2, 0]) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Animation.ShorterChannelClampsWhileLongerInterpolates", "[Animation]")
{
    Animation anim;

    // Rotation ends at t=1: 0° -> 90° about Y
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, s45, 0.0f, c45},
    });

    // Translation ends at t=2: (0,0,0) -> (2,0,0)
    anim.translationKeyframes({
        {0.0f, Vec3{0.0f, 0.0f, 0.0f}},
        {2.0f, Vec3{2.0f, 0.0f, 0.0f}},
    });

    // duration() = 2.0, so t=1.5 is in-range (no wrap)
    Mat4 m = anim.sample(1.5f);

    // Rotation should clamp to its last keyframe: 90° about Y
    // For 90°: m[0,0]=0, m[0,2]=1, m[2,0]=-1, m[2,2]=0
    CHECK((m[0, 0]) == Catch::Approx(0.0f).margin(1e-4f));
    CHECK((m[0, 2]) == Catch::Approx(1.0f).margin(1e-4f));
    CHECK((m[2, 0]) == Catch::Approx(-1.0f).margin(1e-4f));
    CHECK((m[2, 2]) == Catch::Approx(0.0f).margin(1e-4f));

    // Translation should be 75% of the way to (2,0,0) = (1.5, 0, 0)
    CHECK((m[0, 3]) == Catch::Approx(1.5f).margin(1e-4f));
    CHECK((m[1, 3]) == Catch::Approx(0.0f).margin(1e-4f));
    CHECK((m[2, 3]) == Catch::Approx(0.0f).margin(1e-4f));
}

// ==========================================================================
// Scale keyframes
// ==========================================================================

TEST_CASE("Animation.DefaultConstructionHasNoScaleKeyframes", "[Animation]")
{
    Animation anim;
    CHECK(anim.scaleKeyframes().empty());
}

TEST_CASE("Animation.SetScaleKeyframes", "[Animation]")
{
    Animation anim;
    anim.scaleKeyframes({
        {0.0f, Vec3{1.0f, 1.0f, 1.0f}},
        {1.0f, Vec3{2.0f, 2.0f, 2.0f}},
    });
    CHECK(anim.scaleKeyframes().size() == 2u);
}

TEST_CASE("Animation.ScaleOnlyDuration", "[Animation]")
{
    Animation anim;
    anim.scaleKeyframes({
        {0.0f, Vec3{1.0f, 1.0f, 1.0f}},
        {3.0f, Vec3{2.0f, 2.0f, 2.0f}},
    });
    CHECK(anim.duration() == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("Animation.ScaleOnlySingleKeyframe", "[Animation]")
{
    Animation anim;
    anim.scaleKeyframes({{0.0f, Vec3{3.0f, 4.0f, 5.0f}}});

    Mat4 m = anim.sample(0.0f);
    CHECK((m[0, 0]) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK((m[1, 1]) == Catch::Approx(4.0f).margin(1e-5f));
    CHECK((m[2, 2]) == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("Animation.ScaleInterpolationMidpoint", "[Animation]")
{
    Animation anim;
    anim.scaleKeyframes({
        {0.0f, Vec3{1.0f, 1.0f, 1.0f}},
        {2.0f, Vec3{3.0f, 5.0f, 7.0f}},
    });

    Mat4 m = anim.sample(1.0f);
    CHECK((m[0, 0]) == Catch::Approx(2.0f).margin(1e-4f));
    CHECK((m[1, 1]) == Catch::Approx(3.0f).margin(1e-4f));
    CHECK((m[2, 2]) == Catch::Approx(4.0f).margin(1e-4f));
}

TEST_CASE("Animation.ScaleAtEndpoint", "[Animation]")
{
    Animation anim;
    anim.scaleKeyframes({
        {0.0f, Vec3{1.0f, 1.0f, 1.0f}},
        {1.0f, Vec3{2.0f, 3.0f, 4.0f}},
    });

    // At t=0.999 (just before loop wraps), should be nearly at the end values
    Mat4 m = anim.sample(0.999f);
    CHECK((m[0, 0]) == Catch::Approx(2.0f).margin(0.01f));
    CHECK((m[1, 1]) == Catch::Approx(3.0f).margin(0.01f));
    CHECK((m[2, 2]) == Catch::Approx(4.0f).margin(0.01f));
}

TEST_CASE("Animation.NoScaleKeyframesDefaultsToIdentityScale", "[Animation]")
{
    Animation anim;
    anim.translationKeyframes({{0.0f, Vec3{5.0f, 0.0f, 0.0f}}});

    Mat4 m = anim.sample(0.0f);
    // Scale diagonal should be 1.0 (identity scale)
    CHECK((m[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[1, 1]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((m[2, 2]) == Catch::Approx(1.0f).margin(1e-5f));
    // Translation still applied
    CHECK((m[0, 3]) == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("Animation.CombinedTranslationAndScale", "[Animation]")
{
    Animation anim;
    anim.translationKeyframes({{0.0f, Vec3{10.0f, 0.0f, 0.0f}}});
    anim.scaleKeyframes({{0.0f, Vec3{2.0f, 2.0f, 2.0f}}});

    Mat4 m = anim.sample(0.0f);
    // T * R * S: translation in column 3, scale on diagonal
    CHECK((m[0, 3]) == Catch::Approx(10.0f).margin(1e-5f));
    CHECK((m[0, 0]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((m[1, 1]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((m[2, 2]) == Catch::Approx(2.0f).margin(1e-5f));
}

TEST_CASE("Animation.ScaleDurationExtendsOverallDuration", "[Animation]")
{
    Animation anim;
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
    });
    anim.scaleKeyframes({
        {0.0f, Vec3{1.0f, 1.0f, 1.0f}},
        {5.0f, Vec3{2.0f, 2.0f, 2.0f}},
    });
    CHECK(anim.duration() == Catch::Approx(5.0f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// WeightKeyframe tests
// ---------------------------------------------------------------------------

TEST_CASE("WeightKeyframe.DefaultConstructionHasNoWeightKeyframes", "[WeightKeyframe]")
{
    Animation anim;
    CHECK(anim.weightKeyframes().empty());
}

TEST_CASE("WeightKeyframe.SetWeightKeyframes", "[WeightKeyframe]")
{
    Animation anim;
    std::vector<Animation::WeightKeyframe> kf = {
        {0.0f, {0.0f, 0.0f}},
        {1.0f, {1.0f, 0.5f}},
        {2.0f, {0.0f, 1.0f}},
    };
    anim.weightKeyframes(kf);
    CHECK(anim.weightKeyframes().size() == 3u);
}

TEST_CASE("WeightKeyframe.WeightKeyframesContributeToDuration", "[WeightKeyframe]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {3.0f, {1.0f}},
    });
    CHECK(anim.duration() == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("WeightKeyframe.DurationUsesMaxOfAllChannels", "[WeightKeyframe]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {2.0f, {1.0f}},
    });
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {5.0f, 0.0f, 1.0f, 0.0f, 0.0f},
    });
    CHECK(anim.duration() == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("SampleWeights.EmptyKeyframesReturnsZeros", "[SampleWeights]")
{
    Animation anim;
    auto w = anim.sampleWeights(0.5f, 2);
    expectWeightsNear(w, {0.0f, 0.0f});
}

TEST_CASE("SampleWeights.ZeroTargetsReturnsEmpty", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({{0.0f, {1.0f}}});
    auto w = anim.sampleWeights(0.0f, 0);
    CHECK(w.empty());
}

TEST_CASE("SampleWeights.SingleKeyframeReturnsConstant", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({{0.0f, {0.7f, 0.3f}}});
    anim.duration(1.0f);
    auto w = anim.sampleWeights(0.5f, 2);
    expectWeightsNear(w, {0.7f, 0.3f});
}

TEST_CASE("SampleWeights.InterpolatesBetweenKeyframes", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f, 1.0f}},
        {2.0f, {1.0f, 0.0f}},
    });
    anim.duration(2.0f);

    auto w = anim.sampleWeights(1.0f, 2);
    expectWeightsNear(w, {0.5f, 0.5f});
}

TEST_CASE("SampleWeights.InterpolatesAtQuarter", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {4.0f, {1.0f}},
    });
    anim.duration(4.0f);

    auto w = anim.sampleWeights(1.0f, 1);
    expectWeightsNear(w, {0.25f});
}

TEST_CASE("SampleWeights.ClampsAtExactKeyframe", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f, 0.0f}},
        {1.0f, {0.8f, 0.2f}},
        {2.0f, {0.0f, 1.0f}},
    });
    anim.duration(2.0f);

    auto w = anim.sampleWeights(1.0f, 2);
    expectWeightsNear(w, {0.8f, 0.2f});
}

TEST_CASE("SampleWeights.PadsExtraTargetsWithZero", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({{0.0f, {0.5f}}});
    anim.duration(1.0f);
    // Request 3 targets but keyframes only have 1 weight
    auto w = anim.sampleWeights(0.0f, 3);
    expectWeightsNear(w, {0.5f, 0.0f, 0.0f});
}

TEST_CASE("SampleWeights.LoopsAnimation", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {2.0f, {1.0f}},
    });
    anim.duration(2.0f);

    // t=3.0 loops to t=1.0 -> midpoint
    auto w = anim.sampleWeights(3.0f, 1);
    expectWeightsNear(w, {0.5f});
}

TEST_CASE("SampleWeights.NegativeTimeLoops", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {2.0f, {1.0f}},
    });
    anim.duration(2.0f);

    // t=-1.0 loops to t=1.0 -> midpoint.
    auto w = anim.sampleWeights(-1.0f, 1);
    expectWeightsNear(w, {0.5f});
}

TEST_CASE("SampleWeights.ExplicitDurationWraps", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {1.0f, {1.0f}},
    });
    anim.duration(4.0f);

    // t=5.5 loops to t=1.5 on the explicit duration, then clamps after the last key.
    auto w = anim.sampleWeights(5.5f, 1);
    expectWeightsNear(w, {1.0f});
}

TEST_CASE("SampleWeights.InterpolatedMissingTargetsPadWithZero", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {2.0f, {1.0f, 0.5f}},
    });
    anim.duration(2.0f);

    auto w = anim.sampleWeights(1.0f, 3);
    expectWeightsNear(w, {0.5f, 0.25f, 0.0f});
}

// ==========================================================================
// Step interpolation
// ==========================================================================

TEST_CASE("Animation.StepTranslationHoldsPreviousKey", "[Animation]")
{
    Animation anim;
    anim.translationKeyframes({
        {0.0f, Vec3{0.0f, 0.0f, 0.0f}},
        {1.0f, Vec3{10.0f, 0.0f, 0.0f}},
    });
    anim.translationInterpolation(Animation::Interpolation::Step);
    anim.duration(1.0f);

    Mat4 mid = anim.sample(0.5f);
    // Step at alpha > 0 holds left key (0,0,0) -> translation column stays zero.
    CHECK((mid[0, 3]) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Animation.StepScaleHoldsPreviousKey", "[Animation]")
{
    Animation anim;
    anim.scaleKeyframes({
        {0.0f, Vec3{1.0f, 1.0f, 1.0f}},
        {1.0f, Vec3{3.0f, 3.0f, 3.0f}},
    });
    anim.scaleInterpolation(Animation::Interpolation::Step);
    anim.duration(1.0f);

    Mat4 mid = anim.sample(0.5f);
    CHECK((mid[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Animation.StepRotationHoldsPreviousKey", "[Animation]")
{
    Animation anim;
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, s45, 0.0f, c45},
    });
    anim.rotationInterpolation(Animation::Interpolation::Step);
    anim.duration(1.0f);

    Mat4 mid = anim.sample(0.5f);
    // Step held identity rotation -> upper-left 3x3 is identity.
    CHECK((mid[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((mid[2, 2]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((mid[0, 2]) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("SampleWeights.StepHoldsPreviousKey", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {1.0f, {1.0f}},
    });
    anim.weightInterpolation(Animation::Interpolation::Step);
    anim.duration(1.0f);

    auto w = anim.sampleWeights(0.5f, 1);
    expectWeightsNear(w, {0.0f});
}

// ==========================================================================
// CubicSpline interpolation
// ==========================================================================

TEST_CASE("Animation.CubicSplineTranslationMatchesHermiteAtMidpoint", "[Animation]")
{
    Animation anim;
    anim.translationKeyframes({
        {0.0f, Vec3{0.0f, 0.0f, 0.0f}},
        {1.0f, Vec3{2.0f, 0.0f, 0.0f}},
    });
    anim.translationInterpolation(Animation::Interpolation::CubicSpline);
    // dt=1. Hermite with p0=0,p1=2,m0=1,m1=1 at a=0.5 -> x = 1.0
    anim.translationTangents({Vec3{1.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}},
                             {Vec3{1.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}});
    anim.duration(1.0f);

    Mat4 mid = anim.sample(0.5f);
    CHECK((mid[0, 3]) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Animation.CubicSplineTranslationMatchesEndpoints", "[Animation]")
{
    Animation anim;
    anim.translationKeyframes({
        {0.0f, Vec3{0.0f, 0.0f, 0.0f}},
        {1.0f, Vec3{2.0f, 0.0f, 0.0f}},
    });
    anim.translationInterpolation(Animation::Interpolation::CubicSpline);
    anim.translationTangents({Vec3{5.0f, 0.0f, 0.0f}, Vec3{5.0f, 0.0f, 0.0f}},
                             {Vec3{5.0f, 0.0f, 0.0f}, Vec3{5.0f, 0.0f, 0.0f}});
    // Use a larger duration so t=1.0 is not wrapped back to 0 by loopTime.
    anim.duration(2.0f);

    Mat4 start = anim.sample(0.0f);
    Mat4 end = anim.sample(1.0f);
    CHECK((start[0, 3]) == Catch::Approx(0.0f).margin(1e-5f));
    CHECK((end[0, 3]) == Catch::Approx(2.0f).margin(1e-5f));
}

TEST_CASE("Animation.CubicSplineScaleMatchesHermiteAtMidpoint", "[Animation]")
{
    Animation anim;
    anim.scaleKeyframes({
        {0.0f, Vec3{1.0f, 1.0f, 1.0f}},
        {1.0f, Vec3{3.0f, 3.0f, 3.0f}},
    });
    anim.scaleInterpolation(Animation::Interpolation::CubicSpline);
    // dt=1. Hermite with p0=1,p1=3,m0=1,m1=1 at a=0.5 -> 2.0
    anim.scaleTangents({Vec3{1.0f, 1.0f, 1.0f}, Vec3{1.0f, 1.0f, 1.0f}},
                       {Vec3{1.0f, 1.0f, 1.0f}, Vec3{1.0f, 1.0f, 1.0f}});
    anim.duration(1.0f);

    Mat4 mid = anim.sample(0.5f);
    CHECK((mid[0, 0]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((mid[1, 1]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((mid[2, 2]) == Catch::Approx(2.0f).margin(1e-5f));
}

TEST_CASE("Animation.CubicSplineScaleMatchesEndpoints", "[Animation]")
{
    Animation anim;
    anim.scaleKeyframes({
        {0.0f, Vec3{1.0f, 1.0f, 1.0f}},
        {1.0f, Vec3{3.0f, 3.0f, 3.0f}},
    });
    anim.scaleInterpolation(Animation::Interpolation::CubicSpline);
    anim.scaleTangents({Vec3{5.0f, 5.0f, 5.0f}, Vec3{5.0f, 5.0f, 5.0f}},
                       {Vec3{5.0f, 5.0f, 5.0f}, Vec3{5.0f, 5.0f, 5.0f}});
    anim.duration(2.0f);

    Mat4 start = anim.sample(0.0f);
    Mat4 end = anim.sample(1.0f);
    CHECK((start[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((end[0, 0]) == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("Animation.CubicSplineRotationProducesUnitQuaternion", "[Animation]")
{
    Animation anim;
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, s45, 0.0f, c45},
    });
    anim.rotationInterpolation(Animation::Interpolation::CubicSpline);
    anim.rotationTangents({fire_engine::Quaternion{0.0f, 0.0f, 0.0f, 0.0f},
                           fire_engine::Quaternion{0.0f, 0.0f, 0.0f, 0.0f}},
                          {fire_engine::Quaternion{0.0f, 0.0f, 0.0f, 0.0f},
                           fire_engine::Quaternion{0.0f, 0.0f, 0.0f, 0.0f}});
    anim.duration(1.0f);

    // Hermite with zero tangents at midpoint -> 0.5*q0 + 0.5*q1, then normalised.
    // The rotation matrix's top-left column should be a unit vector.
    Mat4 mid = anim.sample(0.5f);
    float col0Len = std::sqrt((mid[0, 0]) * (mid[0, 0]) + (mid[1, 0]) * (mid[1, 0]) +
                              (mid[2, 0]) * (mid[2, 0]));
    CHECK(col0Len == Catch::Approx(1.0f).margin(1e-4f));
}

// ==========================================================================
// CubicSpline weight interpolation
// ==========================================================================

TEST_CASE("SampleWeights.CubicSplineMatchesHermiteAtMidpoint", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {1.0f, {2.0f}},
    });
    anim.weightInterpolation(Animation::Interpolation::CubicSpline);
    // dt=1. Hermite with p0=0,p1=2,m0=1,m1=1 at a=0.5 -> 1.0
    anim.weightTangents({{1.0f}, {1.0f}}, {{1.0f}, {1.0f}});
    anim.duration(1.0f);

    auto w = anim.sampleWeights(0.5f, 1);
    expectWeightsNear(w, {1.0f});
}

TEST_CASE("SampleWeights.CubicSplineMatchesEndpoints", "[SampleWeights]")
{
    Animation anim;
    anim.weightKeyframes({
        {0.0f, {0.0f}},
        {1.0f, {2.0f}},
    });
    anim.weightInterpolation(Animation::Interpolation::CubicSpline);
    anim.weightTangents({{5.0f}, {5.0f}}, {{5.0f}, {5.0f}});
    anim.duration(2.0f);

    auto w0 = anim.sampleWeights(0.0f, 1);
    auto w1 = anim.sampleWeights(1.0f, 1);
    expectWeightsNear(w0, {0.0f});
    expectWeightsNear(w1, {2.0f});
}

// ==========================================================================
// Mixed interpolation modes on the same Animation
// ==========================================================================

TEST_CASE("Animation.MixedStepRotationWithLinearTranslation", "[Animation]")
{
    Animation anim;
    float s45 = std::sin(static_cast<float>(M_PI) / 4.0f);
    float c45 = std::cos(static_cast<float>(M_PI) / 4.0f);
    anim.rotationKeyframes({
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, s45, 0.0f, c45},
    });
    anim.rotationInterpolation(Animation::Interpolation::Step);

    anim.translationKeyframes({
        {0.0f, Vec3{0.0f, 0.0f, 0.0f}},
        {1.0f, Vec3{4.0f, 0.0f, 0.0f}},
    });
    anim.translationInterpolation(Animation::Interpolation::Linear);
    anim.duration(1.0f);

    Mat4 mid = anim.sample(0.5f);

    // Rotation: Step holds identity at midpoint
    CHECK((mid[0, 0]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((mid[2, 2]) == Catch::Approx(1.0f).margin(1e-5f));
    CHECK((mid[0, 2]) == Catch::Approx(0.0f).margin(1e-5f));

    // Translation: Linear interpolates to (2, 0, 0)
    CHECK((mid[0, 3]) == Catch::Approx(2.0f).margin(1e-5f));
    CHECK((mid[1, 3]) == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Animation.PerChannelInterpolationDefaultsToLinear", "[Animation]")
{
    Animation anim;
    CHECK(anim.rotationInterpolation() == Animation::Interpolation::Linear);
    CHECK(anim.translationInterpolation() == Animation::Interpolation::Linear);
    CHECK(anim.scaleInterpolation() == Animation::Interpolation::Linear);
    CHECK(anim.weightInterpolation() == Animation::Interpolation::Linear);
}
