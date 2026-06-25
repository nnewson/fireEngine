#include <fire_engine/animation/animation.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <utility>

namespace fire_engine
{

namespace
{

float loopTime(float t, float dur) noexcept
{
    if (dur <= 0.0f)
    {
        return 0.0f;
    }
    t = std::fmod(t, dur);
    if (t < 0.0f)
    {
        t += dur;
    }
    return t;
}

// Cubic Hermite basis functions. p0/p1 are values at the bracket ends; m0 is
// the out-tangent of the left key, m1 the in-tangent of the right key, both
// pre-scaled by dt per glTF spec §Appendix C.
struct HermiteBasis
{
    float h00, h10, h01, h11;
};

HermiteBasis hermite(float a) noexcept
{
    float a2 = a * a;
    float a3 = a2 * a;
    return {
        2.0f * a3 - 3.0f * a2 + 1.0f,
        a3 - 2.0f * a2 + a,
        -2.0f * a3 + 3.0f * a2,
        a3 - a2,
    };
}

float hermiteScalar(float p0, float m0, float p1, float m1, float a) noexcept
{
    auto h = hermite(a);
    return h.h00 * p0 + h.h10 * m0 + h.h01 * p1 + h.h11 * m1;
}

Vec3 hermiteVec3(const Vec3& p0, const Vec3& m0, const Vec3& p1, const Vec3& m1, float a) noexcept
{
    auto h = hermite(a);
    return {
        h.h00 * p0.x() + h.h10 * m0.x() + h.h01 * p1.x() + h.h11 * m1.x(),
        h.h00 * p0.y() + h.h10 * m0.y() + h.h01 * p1.y() + h.h11 * m1.y(),
        h.h00 * p0.z() + h.h10 * m0.z() + h.h01 * p1.z() + h.h11 * m1.z(),
    };
}

// Find bracketing index i and normalized alpha in [0, 1] such that
// keyframes[i].time <= t < keyframes[i+1].time. Clamps at both ends — pre-first
// or post-last samples take the boundary key value (glTF spec).
template <typename KF>
std::pair<std::size_t, float> findBracket(std::span<const KF> kf, float t) noexcept
{
    if (t <= kf.front().time)
    {
        return {0, 0.0f};
    }

    std::size_t i = 0;
    for (; i < kf.size() - 1; ++i)
    {
        if (t < kf[i + 1].time)
        {
            break;
        }
    }
    if (i >= kf.size() - 1)
    {
        return {kf.size() - 1, 0.0f};
    }

    float dt = kf[i + 1].time - kf[i].time;
    float alpha = (dt > 0.0f) ? (t - kf[i].time) / dt : 0.0f;
    return {i, alpha};
}

// Shared keyframe-channel sampler. The control flow is identical for every TRS
// channel — empty → default, single key → that key, otherwise bracket the time
// and switch on the interpolation mode — so it lives here once. Each channel
// supplies the type-specific pieces as callables:
//
//   valueAt(i)        -> V                  value of keyframe i
//   linear(v0, v1, a) -> V                  Linear blend (lerp / slerp)
//   cubic(i, a)       -> std::optional<V>   Hermite result, or nullopt when the
//                                           channel lacks tangent data (then we
//                                           fall back to linear, not garbage)
template <typename KF, typename V, typename ValueAt, typename Linear, typename Cubic>
V sampleChannel(std::span<const KF> kf, Animation::Interpolation mode, float t, const V& emptyValue,
                ValueAt valueAt, Linear linear, Cubic cubic)
{
    if (kf.empty())
    {
        return emptyValue;
    }
    if (kf.size() == 1)
    {
        return valueAt(0);
    }

    auto [i, alpha] = findBracket(kf, t);
    if (alpha == 0.0f)
    {
        return valueAt(i);
    }

    switch (mode)
    {
    case Animation::Interpolation::Step:
        return valueAt(i);

    case Animation::Interpolation::CubicSpline:
        if (auto result = cubic(i, alpha))
        {
            return *result;
        }
        // Missing tangent data — fall back to linear rather than garbage.
        [[fallthrough]];

    case Animation::Interpolation::Linear:
    default:
        return linear(valueAt(i), valueAt(i + 1), alpha);
    }
}

Quaternion sampleRotation(std::span<const Animation::RotationKeyframe> kf,
                          Animation::Interpolation mode, std::span<const Quaternion> inTans,
                          std::span<const Quaternion> outTans, float t) noexcept
{
    return sampleChannel(
        kf, mode, t, Quaternion::identity(),
        [&](std::size_t i) { return Quaternion{kf[i].qx, kf[i].qy, kf[i].qz, kf[i].qw}; },
        [](const Quaternion& q0, const Quaternion& q1, float a)
        { return Quaternion::slerp(q0, q1, a); },
        [&](std::size_t i, float a) -> std::optional<Quaternion>
        {
            if (i >= outTans.size() || (i + 1) >= inTans.size())
            {
                return std::nullopt;
            }
            float dt = kf[i + 1].time - kf[i].time;
            // glTF spec: evaluate Hermite componentwise on the four quaternion
            // components, scale tangents by dt, then normalise the result.
            Quaternion q0{kf[i].qx, kf[i].qy, kf[i].qz, kf[i].qw};
            Quaternion q1{kf[i + 1].qx, kf[i + 1].qy, kf[i + 1].qz, kf[i + 1].qw};
            const Quaternion& m0 = outTans[i];
            const Quaternion& m1 = inTans[i + 1];
            auto h = hermite(a);
            Quaternion r{
                h.h00 * q0.x() + h.h10 * (m0.x() * dt) + h.h01 * q1.x() + h.h11 * (m1.x() * dt),
                h.h00 * q0.y() + h.h10 * (m0.y() * dt) + h.h01 * q1.y() + h.h11 * (m1.y() * dt),
                h.h00 * q0.z() + h.h10 * (m0.z() * dt) + h.h01 * q1.z() + h.h11 * (m1.z() * dt),
                h.h00 * q0.w() + h.h10 * (m0.w() * dt) + h.h01 * q1.w() + h.h11 * (m1.w() * dt),
            };
            return Quaternion::normalise(r);
        });
}

Vec3 sampleTranslation(std::span<const Animation::TranslationKeyframe> kf,
                       Animation::Interpolation mode, std::span<const Vec3> inTans,
                       std::span<const Vec3> outTans, float t) noexcept
{
    return sampleChannel(
        kf, mode, t, Vec3{}, [&](std::size_t i) { return kf[i].position; },
        [](const Vec3& v0, const Vec3& v1, float a) { return v0 + (v1 - v0) * a; },
        [&](std::size_t i, float a) -> std::optional<Vec3>
        {
            if (i >= outTans.size() || (i + 1) >= inTans.size())
            {
                return std::nullopt;
            }
            float dt = kf[i + 1].time - kf[i].time;
            return hermiteVec3(kf[i].position, outTans[i] * dt, kf[i + 1].position,
                               inTans[i + 1] * dt, a);
        });
}

Vec3 sampleScale(std::span<const Animation::ScaleKeyframe> kf, Animation::Interpolation mode,
                 std::span<const Vec3> inTans, std::span<const Vec3> outTans, float t) noexcept
{
    return sampleChannel(
        kf, mode, t, Vec3{1.0f, 1.0f, 1.0f}, [&](std::size_t i) { return kf[i].scale; },
        [](const Vec3& v0, const Vec3& v1, float a) { return v0 + (v1 - v0) * a; },
        [&](std::size_t i, float a) -> std::optional<Vec3>
        {
            if (i >= outTans.size() || (i + 1) >= inTans.size())
            {
                return std::nullopt;
            }
            float dt = kf[i + 1].time - kf[i].time;
            return hermiteVec3(kf[i].scale, outTans[i] * dt, kf[i + 1].scale, inTans[i + 1] * dt,
                               a);
        });
}

} // namespace

float Animation::duration() const noexcept
{
    if (explicitDuration_)
    {
        return *explicitDuration_;
    }
    float rotDur = rotationKeyframes_.empty() ? 0.0f : rotationKeyframes_.back().time;
    float transDur = translationKeyframes_.empty() ? 0.0f : translationKeyframes_.back().time;
    float scaleDur = scaleKeyframes_.empty() ? 0.0f : scaleKeyframes_.back().time;
    float weightDur = weightKeyframes_.empty() ? 0.0f : weightKeyframes_.back().time;
    return std::max({rotDur, transDur, scaleDur, weightDur});
}

Mat4 Animation::sample(float t) const noexcept
{
    if (rotationKeyframes_.empty() && translationKeyframes_.empty() && scaleKeyframes_.empty())
    {
        return Mat4::identity();
    }

    float looped = loopTime(t, duration());

    Quaternion rotation = sampleRotation(rotationKeyframes_, rotationInterp_, rotationInTangents_,
                                         rotationOutTangents_, looped);
    Vec3 position = sampleTranslation(translationKeyframes_, translationInterp_,
                                      translationInTangents_, translationOutTangents_, looped);
    Vec3 scl =
        sampleScale(scaleKeyframes_, scaleInterp_, scaleInTangents_, scaleOutTangents_, looped);

    return Mat4::translate(position) * rotation.toMat4() * Mat4::scale(scl);
}

std::vector<float> Animation::sampleWeights(float t, std::size_t numTargets) const
{
    if (weightKeyframes_.empty() || numTargets == 0)
    {
        return std::vector<float>(numTargets, 0.0f);
    }

    float looped = loopTime(t, duration());

    if (weightKeyframes_.size() == 1)
    {
        auto result = weightKeyframes_[0].weights;
        result.resize(numTargets, 0.0f);
        return result;
    }

    auto [i, alpha] = findBracket(std::span<const WeightKeyframe>{weightKeyframes_}, looped);

    const auto& w0 = weightKeyframes_[i].weights;
    if (alpha == 0.0f)
    {
        auto result = w0;
        result.resize(numTargets, 0.0f);
        return result;
    }

    const auto& w1 = weightKeyframes_[i + 1].weights;
    std::vector<float> result(numTargets, 0.0f);

    switch (weightInterp_)
    {
    case Interpolation::Step:
        for (std::size_t j = 0; j < numTargets; ++j)
        {
            result[j] = (j < w0.size()) ? w0[j] : 0.0f;
        }
        return result;

    case Interpolation::CubicSpline:
        if (i < weightOutTangents_.size() && (i + 1) < weightInTangents_.size())
        {
            float dt = weightKeyframes_[i + 1].time - weightKeyframes_[i].time;
            const auto& m0 = weightOutTangents_[i];
            const auto& m1 = weightInTangents_[i + 1];
            for (std::size_t j = 0; j < numTargets; ++j)
            {
                float a = (j < w0.size()) ? w0[j] : 0.0f;
                float b = (j < w1.size()) ? w1[j] : 0.0f;
                float ma = (j < m0.size()) ? m0[j] * dt : 0.0f;
                float mb = (j < m1.size()) ? m1[j] * dt : 0.0f;
                result[j] = hermiteScalar(a, ma, b, mb, alpha);
            }
            return result;
        }
        [[fallthrough]];

    case Interpolation::Linear:
    default:
        for (std::size_t j = 0; j < numTargets; ++j)
        {
            float a = (j < w0.size()) ? w0[j] : 0.0f;
            float b = (j < w1.size()) ? w1[j] : 0.0f;
            result[j] = a + (b - a) * alpha;
        }
        return result;
    }
}

} // namespace fire_engine
