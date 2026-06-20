#pragma once

#include <cmath>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

class Quaternion
{
public:
    constexpr Quaternion(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 1.0f) noexcept
        : x_(x),
          y_(y),
          z_(z),
          w_(w)
    {
    }

    ~Quaternion() = default;

    Quaternion(const Quaternion&) = default;
    Quaternion& operator=(const Quaternion&) = default;
    Quaternion(Quaternion&&) noexcept = default;
    Quaternion& operator=(Quaternion&&) noexcept = default;

    [[nodiscard]]
    constexpr float x() const noexcept
    {
        return x_;
    }

    constexpr void x(float x) noexcept
    {
        x_ = x;
    }

    [[nodiscard]]
    constexpr float y() const noexcept
    {
        return y_;
    }

    constexpr void y(float y) noexcept
    {
        y_ = y;
    }

    [[nodiscard]]
    constexpr float z() const noexcept
    {
        return z_;
    }

    constexpr void z(float z) noexcept
    {
        z_ = z;
    }

    [[nodiscard]]
    constexpr float w() const noexcept
    {
        return w_;
    }

    constexpr void w(float w) noexcept
    {
        w_ = w;
    }

    [[nodiscard]]
    static constexpr Quaternion identity() noexcept
    {
        return {};
    }

    [[nodiscard]]
    constexpr Quaternion operator-() const noexcept
    {
        return {-x_, -y_, -z_, -w_};
    }

    // Strict bit-for-bit equality. Two quaternions that differ by a single ULP
    // compare not-equal — use approxEqual when you want tolerance. Note: also
    // strict in the sign of the imaginary parts, so q and -q (which represent
    // the same rotation) compare not-equal.
    [[nodiscard]]
    constexpr bool operator==(const Quaternion& rhs) const noexcept
    {
        return x_ == rhs.x_ && y_ == rhs.y_ && z_ == rhs.z_ && w_ == rhs.w_;
    }

    [[nodiscard]]
    constexpr bool bitwiseEqual(const Quaternion& rhs) const noexcept
    {
        return *this == rhs;
    }

    [[nodiscard]]
    constexpr bool approxEqual(const Quaternion& rhs, float eps = float_epsilon) const noexcept
    {
        const float dx = x_ - rhs.x_;
        const float dy = y_ - rhs.y_;
        const float dz = z_ - rhs.z_;
        const float dw = w_ - rhs.w_;
        return dx <= eps && dx >= -eps && dy <= eps && dy >= -eps && dz <= eps && dz >= -eps &&
               dw <= eps && dw >= -eps;
    }

    [[nodiscard]]
    static constexpr float dotProduct(const Quaternion& a, const Quaternion& b) noexcept
    {
        return a.x_ * b.x_ + a.y_ * b.y_ + a.z_ * b.z_ + a.w_ * b.w_;
    }

    [[nodiscard]]
    constexpr float dotProduct(const Quaternion& rhs) const noexcept
    {
        return Quaternion::dotProduct(*this, rhs);
    }

    [[nodiscard]]
    constexpr float magnitudeSquared() const noexcept
    {
        return x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_;
    }

    [[nodiscard]]
    float magnitude() const noexcept
    {
        return std::sqrt(magnitudeSquared());
    }

    [[nodiscard]]
    static Quaternion normalise(const Quaternion& q) noexcept
    {
        float len = q.magnitude();
        if (len < float_epsilon)
        {
            return Quaternion::identity();
        }
        return {q.x_ / len, q.y_ / len, q.z_ / len, q.w_ / len};
    }

    Quaternion& normalise() noexcept
    {
        *this = Quaternion::normalise(*this);
        return *this;
    }

    // Conjugate ({-x,-y,-z,w}). For a unit quaternion this is also its inverse —
    // the rotation that undoes this one (used to map world vectors into a body's
    // local frame).
    [[nodiscard]]
    constexpr Quaternion conjugate() const noexcept
    {
        return {-x_, -y_, -z_, w_};
    }

    // Rotate a vector by this (unit) quaternion:
    //   v' = v + 2w·(u×v) + 2·u×(u×v),  u = (x,y,z).
    // Cheaper and matrix-free vs building toMat4(); equivalent for unit quats.
    [[nodiscard]]
    constexpr Vec3 rotate(const Vec3& v) const noexcept
    {
        const Vec3 u{x_, y_, z_};
        const Vec3 t = Vec3::crossProduct(u, v) * 2.0f;
        return v + t * w_ + Vec3::crossProduct(u, t);
    }

    // Hamilton product (composition of rotations: apply `rhs`, then `*this`).
    [[nodiscard]]
    constexpr Quaternion operator*(const Quaternion& rhs) const noexcept
    {
        return {
            w_ * rhs.x_ + x_ * rhs.w_ + y_ * rhs.z_ - z_ * rhs.y_,
            w_ * rhs.y_ - x_ * rhs.z_ + y_ * rhs.w_ + z_ * rhs.x_,
            w_ * rhs.z_ + x_ * rhs.y_ - y_ * rhs.x_ + z_ * rhs.w_,
            w_ * rhs.w_ - x_ * rhs.x_ - y_ * rhs.y_ - z_ * rhs.z_,
        };
    }

    // Advance this orientation by an angular velocity `omega` (rad/s) over `dt`
    // using the exponential map: build the incremental rotation Δq from the
    // rotation vector ω·dt, then return normalise(Δq · this). Stable for large
    // steps and unconditionally re-normalised.
    [[nodiscard]]
    Quaternion integrate(const Vec3& omega, float dt) const noexcept
    {
        const Vec3 rotation = omega * dt; // rotation vector this step
        const float angle = rotation.magnitude();
        Quaternion delta;
        if (angle < float_epsilon)
        {
            // Small angle: Δq ≈ {0.5·ω·dt, 1}, normalised below.
            delta = Quaternion{rotation.x() * 0.5f, rotation.y() * 0.5f, rotation.z() * 0.5f, 1.0f};
        }
        else
        {
            const float half = angle * 0.5f;
            const float s = std::sin(half) / angle; // sin(half) applied to axis = rotation/angle
            delta =
                Quaternion{rotation.x() * s, rotation.y() * s, rotation.z() * s, std::cos(half)};
        }
        return Quaternion::normalise(delta * *this);
    }

    // Shortest-arc rotation that maps `from` onto `to`. Inputs are assumed
    // to be unit-length; the formula degenerates if they are not. Handles
    // the antiparallel case by rotating 180° about an axis orthogonal to
    // `from`.
    [[nodiscard]]
    static Quaternion fromVectors(Vec3 from, Vec3 to) noexcept
    {
        // 1e-6 is the smallest threshold that still survives single-precision
        // round-off near ±1: float_epsilon (1e-8) rounds back to 1.0f when
        // subtracted from 1.0f, so it would let antiparallel inputs slip
        // through to the general branch and produce a zero quaternion.
        constexpr float kColinearTolerance = 1e-6f;
        const float d = Vec3::dotProduct(from, to);
        if (d > 1.0f - kColinearTolerance)
        {
            return Quaternion::identity();
        }
        if (d < -1.0f + kColinearTolerance)
        {
            Vec3 axis = Vec3::crossProduct(from, Vec3{1.0f, 0.0f, 0.0f});
            if (axis.magnitudeSquared() < kColinearTolerance)
            {
                axis = Vec3::crossProduct(from, Vec3{0.0f, 1.0f, 0.0f});
            }
            axis.normalise();
            return {axis.x(), axis.y(), axis.z(), 0.0f};
        }
        const Vec3 c = Vec3::crossProduct(from, to);
        return Quaternion::normalise({c.x(), c.y(), c.z(), 1.0f + d});
    }

    [[nodiscard]]
    static Quaternion slerp(const Quaternion& a, const Quaternion& b, float t) noexcept
    {
        float dot = Quaternion::dotProduct(a, b);

        // If the dot product is negative, negate one quaternion to take the shorter path
        Quaternion bCorrected = b;
        if (dot < 0.0f)
        {
            bCorrected = -b;
            dot = -dot;
        }

        // If the inputs are very close, fall back to NLERP to avoid division by zero
        if (dot > 0.9995f)
        {
            Quaternion result{
                a.x_ + t * (bCorrected.x_ - a.x_),
                a.y_ + t * (bCorrected.y_ - a.y_),
                a.z_ + t * (bCorrected.z_ - a.z_),
                a.w_ + t * (bCorrected.w_ - a.w_),
            };
            return Quaternion::normalise(result);
        }

        float theta = std::acos(dot);
        float sinTheta = std::sin(theta);

        float wa = std::sin((1.0f - t) * theta) / sinTheta;
        float wb = std::sin(t * theta) / sinTheta;

        return {
            wa * a.x_ + wb * bCorrected.x_,
            wa * a.y_ + wb * bCorrected.y_,
            wa * a.z_ + wb * bCorrected.z_,
            wa * a.w_ + wb * bCorrected.w_,
        };
    }

    [[nodiscard]]
    Vec3 toEulerXYZ() const noexcept
    {
        // Extrinsic XYZ (equivalently intrinsic ZYX) Tait-Bryan decomposition.
        float sy = 2.0f * (w_ * y_ - z_ * x_);
        if (sy > 1.0f)
        {
            sy = 1.0f;
        }
        if (sy < -1.0f)
        {
            sy = -1.0f;
        }
        float rotX = std::atan2(2.0f * (w_ * x_ + y_ * z_), 1.0f - 2.0f * (x_ * x_ + y_ * y_));
        float rotY = std::asin(sy);
        float rotZ = std::atan2(2.0f * (w_ * z_ + x_ * y_), 1.0f - 2.0f * (y_ * y_ + z_ * z_));
        return {rotX, rotY, rotZ};
    }

    [[nodiscard]]
    Mat4 toMat4() const noexcept
    {
        float xx = x_ * x_;
        float yy = y_ * y_;
        float zz = z_ * z_;
        float xy = x_ * y_;
        float xz = x_ * z_;
        float yz = y_ * z_;
        float wx = w_ * x_;
        float wy = w_ * y_;
        float wz = w_ * z_;

        // Column-major rotation matrix from unit quaternion
        Mat4 m;
        // Column 0
        m[0, 0] = 1.0f - 2.0f * (yy + zz);
        m[1, 0] = 2.0f * (xy + wz);
        m[2, 0] = 2.0f * (xz - wy);
        // Column 1
        m[0, 1] = 2.0f * (xy - wz);
        m[1, 1] = 1.0f - 2.0f * (xx + zz);
        m[2, 1] = 2.0f * (yz + wx);
        // Column 2
        m[0, 2] = 2.0f * (xz + wy);
        m[1, 2] = 2.0f * (yz - wx);
        m[2, 2] = 1.0f - 2.0f * (xx + yy);
        // Column 3
        m[3, 3] = 1.0f;
        return m;
    }

private:
    float x_;
    float y_;
    float z_;
    float w_;
};

} // namespace fire_engine
