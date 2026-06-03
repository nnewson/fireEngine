#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

class Animation
{
public:
    // glTF 2.0 sampler interpolation modes. Set per channel so a single Animation
    // can mix, e.g., LINEAR rotation with STEP translation — matching how glTF
    // stores interpolation on each AnimationSampler.
    enum class Interpolation
    {
        Linear,
        Step,
        CubicSpline,
    };

    struct RotationKeyframe
    {
        float time{0.0f};
        float qx{0.0f};
        float qy{0.0f};
        float qz{0.0f};
        float qw{1.0f};
    };

    struct TranslationKeyframe
    {
        float time{0.0f};
        Vec3 position{};
    };

    struct ScaleKeyframe
    {
        float time{0.0f};
        Vec3 scale{1.0f, 1.0f, 1.0f};
    };

    struct WeightKeyframe
    {
        float time{0.0f};
        std::vector<float> weights;
    };

    Animation() = default;
    ~Animation() = default;

    Animation(const Animation&) = default;
    Animation& operator=(const Animation&) = default;
    Animation(Animation&&) noexcept = default;
    Animation& operator=(Animation&&) noexcept = default;

    [[nodiscard]] const std::string& name() const noexcept
    {
        return name_;
    }
    void name(std::string n) noexcept
    {
        name_ = std::move(n);
    }

    // Keyframe setters — interpolation mode stays whatever it was (default Linear).
    void rotationKeyframes(std::vector<RotationKeyframe> kf) noexcept
    {
        rotationKeyframes_ = std::move(kf);
    }
    void translationKeyframes(std::vector<TranslationKeyframe> kf) noexcept
    {
        translationKeyframes_ = std::move(kf);
    }
    void scaleKeyframes(std::vector<ScaleKeyframe> kf) noexcept
    {
        scaleKeyframes_ = std::move(kf);
    }
    void weightKeyframes(std::vector<WeightKeyframe> kf) noexcept
    {
        weightKeyframes_ = std::move(kf);
    }

    // Interpolation mode setters.
    void rotationInterpolation(Interpolation m) noexcept
    {
        rotationInterp_ = m;
    }
    void translationInterpolation(Interpolation m) noexcept
    {
        translationInterp_ = m;
    }
    void scaleInterpolation(Interpolation m) noexcept
    {
        scaleInterp_ = m;
    }
    void weightInterpolation(Interpolation m) noexcept
    {
        weightInterp_ = m;
    }

    // CubicSpline tangents — one in/out tangent per keyframe, same index as
    // the corresponding keyframe. Ignored when interpolation != CubicSpline.
    void rotationTangents(std::vector<Quaternion> in, std::vector<Quaternion> out) noexcept
    {
        rotationInTangents_ = std::move(in);
        rotationOutTangents_ = std::move(out);
    }
    void translationTangents(std::vector<Vec3> in, std::vector<Vec3> out) noexcept
    {
        translationInTangents_ = std::move(in);
        translationOutTangents_ = std::move(out);
    }
    void scaleTangents(std::vector<Vec3> in, std::vector<Vec3> out) noexcept
    {
        scaleInTangents_ = std::move(in);
        scaleOutTangents_ = std::move(out);
    }
    void weightTangents(std::vector<std::vector<float>> in,
                        std::vector<std::vector<float>> out) noexcept
    {
        weightInTangents_ = std::move(in);
        weightOutTangents_ = std::move(out);
    }

    [[nodiscard]] const std::vector<RotationKeyframe>& rotationKeyframes() const noexcept
    {
        return rotationKeyframes_;
    }
    [[nodiscard]] const std::vector<TranslationKeyframe>& translationKeyframes() const noexcept
    {
        return translationKeyframes_;
    }
    [[nodiscard]] const std::vector<ScaleKeyframe>& scaleKeyframes() const noexcept
    {
        return scaleKeyframes_;
    }
    [[nodiscard]] const std::vector<WeightKeyframe>& weightKeyframes() const noexcept
    {
        return weightKeyframes_;
    }

    [[nodiscard]] Interpolation rotationInterpolation() const noexcept
    {
        return rotationInterp_;
    }
    [[nodiscard]] Interpolation translationInterpolation() const noexcept
    {
        return translationInterp_;
    }
    [[nodiscard]] Interpolation scaleInterpolation() const noexcept
    {
        return scaleInterp_;
    }
    [[nodiscard]] Interpolation weightInterpolation() const noexcept
    {
        return weightInterp_;
    }

    [[nodiscard]] const std::vector<Quaternion>& rotationInTangents() const noexcept
    {
        return rotationInTangents_;
    }
    [[nodiscard]] const std::vector<Quaternion>& rotationOutTangents() const noexcept
    {
        return rotationOutTangents_;
    }
    [[nodiscard]] const std::vector<Vec3>& translationInTangents() const noexcept
    {
        return translationInTangents_;
    }
    [[nodiscard]] const std::vector<Vec3>& translationOutTangents() const noexcept
    {
        return translationOutTangents_;
    }
    [[nodiscard]] const std::vector<Vec3>& scaleInTangents() const noexcept
    {
        return scaleInTangents_;
    }
    [[nodiscard]] const std::vector<Vec3>& scaleOutTangents() const noexcept
    {
        return scaleOutTangents_;
    }
    [[nodiscard]] const std::vector<std::vector<float>>& weightInTangents() const noexcept
    {
        return weightInTangents_;
    }
    [[nodiscard]] const std::vector<std::vector<float>>& weightOutTangents() const noexcept
    {
        return weightOutTangents_;
    }

    [[nodiscard]] float duration() const noexcept;
    // Explicit loop duration override so channels on the same glTF animation
    // loop in lockstep across its full span. std::nullopt (default) = take the
    // max of the channel keyframes.
    void duration(std::optional<float> d) noexcept
    {
        explicitDuration_ = d;
    }

    [[nodiscard]] Mat4 sample(float t) const noexcept;

    [[nodiscard]] std::vector<float> sampleWeights(float t, std::size_t numTargets) const;

private:
    std::string name_;
    std::vector<RotationKeyframe> rotationKeyframes_;
    std::vector<TranslationKeyframe> translationKeyframes_;
    std::vector<ScaleKeyframe> scaleKeyframes_;
    std::vector<WeightKeyframe> weightKeyframes_;

    Interpolation rotationInterp_{Interpolation::Linear};
    Interpolation translationInterp_{Interpolation::Linear};
    Interpolation scaleInterp_{Interpolation::Linear};
    Interpolation weightInterp_{Interpolation::Linear};

    std::vector<Quaternion> rotationInTangents_;
    std::vector<Quaternion> rotationOutTangents_;
    std::vector<Vec3> translationInTangents_;
    std::vector<Vec3> translationOutTangents_;
    std::vector<Vec3> scaleInTangents_;
    std::vector<Vec3> scaleOutTangents_;
    std::vector<std::vector<float>> weightInTangents_;
    std::vector<std::vector<float>> weightOutTangents_;

    std::optional<float> explicitDuration_;
};

} // namespace fire_engine
