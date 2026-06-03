#include <fire_engine/animation/animation.hpp>

#include <algorithm>
#include <cmath>
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
std::pair<std::size_t, float> findBracket(const std::vector<KF>& kf, float t) noexcept
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

Quaternion sampleRotation(const std::vector<Animation::RotationKeyframe>& kf,
                          Animation::Interpolation mode, const std::vector<Quaternion>& inTans,
                          const std::vector<Quaternion>& outTans, float t) noexcept
{
    if (kf.empty())
    {
        return Quaternion::identity();
    }
    if (kf.size() == 1)
    {
        return {kf[0].qx, kf[0].qy, kf[0].qz, kf[0].qw};
    }

    auto [i, alpha] = findBracket(kf, t);
    Quaternion q0{kf[i].qx, kf[i].qy, kf[i].qz, kf[i].qw};
    if (alpha == 0.0f)
    {
        return q0;
    }
    Quaternion q1{kf[i + 1].qx, kf[i + 1].qy, kf[i + 1].qz, kf[i + 1].qw};

    switch (mode)
    {
    case Animation::Interpolation::Step:
        return q0;

    case Animation::Interpolation::CubicSpline:
        if (i < outTans.size() && (i + 1) < inTans.size())
        {
            float dt = kf[i + 1].time - kf[i].time;
            // glTF spec: evaluate Hermite componentwise on the four quaternion
            // components, scale tangents by dt, then normalise the result.
            const Quaternion& m0 = outTans[i];
            const Quaternion& m1 = inTans[i + 1];
            auto h = hermite(alpha);
            Quaternion r{
                h.h00 * q0.x() + h.h10 * (m0.x() * dt) + h.h01 * q1.x() + h.h11 * (m1.x() * dt),
                h.h00 * q0.y() + h.h10 * (m0.y() * dt) + h.h01 * q1.y() + h.h11 * (m1.y() * dt),
                h.h00 * q0.z() + h.h10 * (m0.z() * dt) + h.h01 * q1.z() + h.h11 * (m1.z() * dt),
                h.h00 * q0.w() + h.h10 * (m0.w() * dt) + h.h01 * q1.w() + h.h11 * (m1.w() * dt),
            };
            return Quaternion::normalise(r);
        }
        // Missing tangent data — fall back to slerp rather than producing garbage.
        [[fallthrough]];

    case Animation::Interpolation::Linear:
    default:
        return Quaternion::slerp(q0, q1, alpha);
    }
}

Vec3 sampleTranslation(const std::vector<Animation::TranslationKeyframe>& kf,
                       Animation::Interpolation mode, const std::vector<Vec3>& inTans,
                       const std::vector<Vec3>& outTans, float t) noexcept
{
    if (kf.empty())
    {
        return {};
    }
    if (kf.size() == 1)
    {
        return kf[0].position;
    }

    auto [i, alpha] = findBracket(kf, t);
    if (alpha == 0.0f)
    {
        return kf[i].position;
    }

    switch (mode)
    {
    case Animation::Interpolation::Step:
        return kf[i].position;

    case Animation::Interpolation::CubicSpline:
        if (i < outTans.size() && (i + 1) < inTans.size())
        {
            float dt = kf[i + 1].time - kf[i].time;
            return hermiteVec3(kf[i].position, outTans[i] * dt, kf[i + 1].position,
                               inTans[i + 1] * dt, alpha);
        }
        [[fallthrough]];

    case Animation::Interpolation::Linear:
    default:
        return kf[i].position + (kf[i + 1].position - kf[i].position) * alpha;
    }
}

Vec3 sampleScale(const std::vector<Animation::ScaleKeyframe>& kf, Animation::Interpolation mode,
                 const std::vector<Vec3>& inTans, const std::vector<Vec3>& outTans,
                 float t) noexcept
{
    if (kf.empty())
    {
        return {1.0f, 1.0f, 1.0f};
    }
    if (kf.size() == 1)
    {
        return kf[0].scale;
    }

    auto [i, alpha] = findBracket(kf, t);
    if (alpha == 0.0f)
    {
        return kf[i].scale;
    }

    switch (mode)
    {
    case Animation::Interpolation::Step:
        return kf[i].scale;

    case Animation::Interpolation::CubicSpline:
        if (i < outTans.size() && (i + 1) < inTans.size())
        {
            float dt = kf[i + 1].time - kf[i].time;
            return hermiteVec3(kf[i].scale, outTans[i] * dt, kf[i + 1].scale, inTans[i + 1] * dt,
                               alpha);
        }
        [[fallthrough]];

    case Animation::Interpolation::Linear:
    default:
        return kf[i].scale + (kf[i + 1].scale - kf[i].scale) * alpha;
    }
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

    auto [i, alpha] = findBracket(weightKeyframes_, looped);

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
