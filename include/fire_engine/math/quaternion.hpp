#pragma once

#include <cmath>
#include <limits>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/scalar.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

class Quaternion
{
public:
    constexpr Quaternion() noexcept = default;

    constexpr Quaternion(float x, float y, float z, float w) noexcept
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

    // EXACT component-wise IEEE equality — not bitwise, despite what this used to claim. Two
    // differences matter and both are the float `==` operator's, not ours: `-0.0f` equals `+0.0f`
    // though their bit patterns differ, and a NaN equals nothing at all though its bit pattern is
    // identical to itself. Use `approxEqual` when you want tolerance; if a determinism diagnostic
    // ever needs REAL bit comparison, it has to say so with `std::bit_cast`.
    //
    // Note this is a COMPONENT comparison, so `q` and `-q` are unequal here although they are the
    // same rotation: the rotation-aware question belongs to a rotation type, not to this one.
    [[nodiscard]]
    constexpr bool operator==(const Quaternion& rhs) const noexcept
    {
        return x_ == rhs.x_ && y_ == rhs.y_ && z_ == rhs.z_ && w_ == rhs.w_;
    }

    // Approximate equality, component by component, through the ONE scalar authority
    // (`math/scalar.hpp`), and in its three forms — no argument means both defaults, an explicit
    // tolerance means ABSOLUTE ONLY (so `approxEqual(rhs, 1e-9f)` rejects anything further apart
    // than 1e-9, exactly as it always did), and both arguments mean both terms. An invalid
    // tolerance — negative, NaN or infinite — makes the comparison FALSE rather than being
    // reinterpreted. NaNs compare unequal now, which is the defect this replaced.
    [[nodiscard]]
    constexpr bool approxEqual(const Quaternion& rhs, float eps, float relativeEps) const noexcept
    {
        return almostEqual(x_, rhs.x_, eps, relativeEps) &&
               almostEqual(y_, rhs.y_, eps, relativeEps) &&
               almostEqual(z_, rhs.z_, eps, relativeEps) &&
               almostEqual(w_, rhs.w_, eps, relativeEps);
    }

    [[nodiscard]]
    constexpr bool approxEqual(const Quaternion& rhs, float eps) const noexcept
    {
        // ABSOLUTE ONLY — a stated tolerance is the whole answer.
        return approxEqual(rhs, eps, 0.0f);
    }

    [[nodiscard]]
    constexpr bool approxEqual(const Quaternion& rhs) const noexcept
    {
        return approxEqual(rhs, float_epsilon, float_relative_epsilon);
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
    // Fast path first — see `VecBase::magnitude`. While the sum of squares is finite and positive
    // this is the arithmetic the engine always did, bit for bit; the scaled form below runs only
    // when that sum came back zero, infinite or NaN, which is precisely when it had no answer.
    float magnitude() const noexcept
    {
        const float sumOfSquares = magnitudeSquared();
        // `isfinite` rather than a bare `< infinity` comparison: the two classify identically here
        // (a NaN fails every comparison, an infinity fails the bound, zero fails the first test)
        // and they measured identically too, so the one that says what it means wins.
        // NORMAL, not merely positive. A sum that has gone SUBNORMAL has already lost most of its
        // precision without reaching zero: (3e-23, 3e-23, 0) sums to 2.8e-45, which carries about
        // two significant bits, and `sqrt` of it answers 5.29e-23 against a true 4.24e-23 — a 24.8%
        // error from a fast path that thought it was fine because the sum was finite and positive.
        // Requiring the sum to be at least `float`'s smallest NORMAL value routes that whole region
        // to the scaled form, where the components are rescaled before they are squared and no
        // precision is lost at all. `isfinite` then rules out the top end; a NaN fails both.
        if (sumOfSquares >= std::numeric_limits<float>::min() && std::isfinite(sumOfSquares))
        {
            return std::sqrt(sumOfSquares);
        }
        return scaledMagnitude();
    }

    // A rotation's norm is the number every unit-quaternion assumption rests on — rotate(),
    // slerp(), toMat4() — so the same three answers as the vector types. A non-finite component
    // yields a NaN quaternion (visibly invalid, never laundered into the identity), a magnitude
    // below `float_normalise_cutoff` yields the identity rotation as it always has, and anything
    // else is normalised.
    [[nodiscard]]
    static Quaternion normalise(const Quaternion& q) noexcept
    {
        const float sumOfSquares = q.magnitudeSquared();
        // A NORMAL sum, for the subnormal-precision reason given on `magnitude`.
        if (sumOfSquares >= std::numeric_limits<float>::min() && std::isfinite(sumOfSquares))
        {
            const float length = std::sqrt(sumOfSquares);
            if (length < float_normalise_cutoff)
            {
                return Quaternion::identity();
            }
            return {q.x_ / length, q.y_ / length, q.z_ / length, q.w_ / length};
        }
        return scaledNormalise(q);
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

    // Rotation by `angle` radians about `axis` (assumed unit-length). The standard
    // axis-angle form q = {axis·sin(θ/2), cos(θ/2)}; a zero angle returns identity.
    [[nodiscard]]
    static Quaternion fromAxisAngle(const Vec3& axis, float angle) noexcept
    {
        const float half = angle * 0.5f;
        const float s = std::sin(half);
        return Quaternion{axis.x() * s, axis.y() * s, axis.z() * s, std::cos(half)};
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

    // Extract the rotation from a transform matrix's upper-left 3×3 (assumed
    // orthonormal — strip any scale first). Column-major element access m[row, col];
    // Shepperd's method, picking the largest diagonal term for numerical stability.
    [[nodiscard]]
    static Quaternion fromMatrix(const Mat4& m) noexcept
    {
        const float m00 = m[0, 0];
        const float m11 = m[1, 1];
        const float m22 = m[2, 2];
        const float trace = m00 + m11 + m22;
        Quaternion q;
        if (trace > 0.0f)
        {
            const float s = std::sqrt(trace + 1.0f) * 2.0f; // s = 4w
            q.w(0.25f * s);
            q.x((m[2, 1] - m[1, 2]) / s);
            q.y((m[0, 2] - m[2, 0]) / s);
            q.z((m[1, 0] - m[0, 1]) / s);
        }
        else if (m00 > m11 && m00 > m22)
        {
            const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f; // s = 4x
            q.w((m[2, 1] - m[1, 2]) / s);
            q.x(0.25f * s);
            q.y((m[0, 1] + m[1, 0]) / s);
            q.z((m[0, 2] + m[2, 0]) / s);
        }
        else if (m11 > m22)
        {
            const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f; // s = 4y
            q.w((m[0, 2] - m[2, 0]) / s);
            q.x((m[0, 1] + m[1, 0]) / s);
            q.y(0.25f * s);
            q.z((m[1, 2] + m[2, 1]) / s);
        }
        else
        {
            const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f; // s = 4z
            q.w((m[1, 0] - m[0, 1]) / s);
            q.x((m[0, 2] + m[2, 0]) / s);
            q.y((m[1, 2] + m[2, 1]) / s);
            q.z(0.25f * s);
        }
        return Quaternion::normalise(q);
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
    // THE ROBUST PATH, reached only when the sum of squares was zero, infinite or NaN.
    [[nodiscard]] float scaledMagnitude() const noexcept
    {
        const float components[4]{x_, y_, z_, w_};
        float largest = 0.0f;
        bool anyInfinite = false;
        for (const float component : components)
        {
            if (std::isnan(component))
            {
                return component;
            }
            if (std::isinf(component))
            {
                anyInfinite = true;
                continue;
            }
            const float componentMagnitude = std::fabs(component);
            largest = componentMagnitude > largest ? componentMagnitude : largest;
        }
        if (anyInfinite)
        {
            return std::numeric_limits<float>::infinity();
        }
        if (largest == 0.0f)
        {
            return 0.0f;
        }
        float sumOfScaledSquares = 0.0f;
        for (const float component : components)
        {
            const float scaled = component / largest;
            sumOfScaledSquares += scaled * scaled;
        }
        return largest * std::sqrt(sumOfScaledSquares);
    }

    [[nodiscard]] static Quaternion scaledNormalise(const Quaternion& q) noexcept
    {
        const float components[4]{q.x_, q.y_, q.z_, q.w_};
        float largest = 0.0f;
        for (const float component : components)
        {
            if (!std::isfinite(component))
            {
                const float nan = std::numeric_limits<float>::quiet_NaN();
                return {nan, nan, nan, nan};
            }
            const float componentMagnitude = std::fabs(component);
            largest = componentMagnitude > largest ? componentMagnitude : largest;
        }
        if (largest == 0.0f)
        {
            return Quaternion::identity();
        }
        float sumOfScaledSquares = 0.0f;
        for (const float component : components)
        {
            const float scaled = component / largest;
            sumOfScaledSquares += scaled * scaled;
        }
        const float scaledNorm = std::sqrt(sumOfScaledSquares); // in [1, 2]
        const float length = largest * scaledNorm;
        if (length < float_normalise_cutoff)
        {
            return Quaternion::identity();
        }
        if (std::isfinite(length))
        {
            return {q.x_ / length, q.y_ / length, q.z_ / length, q.w_ / length};
        }
        return {(q.x_ / largest) / scaledNorm, (q.y_ / largest) / scaledNorm,
                (q.z_ / largest) / scaledNorm, (q.w_ / largest) / scaledNorm};
    }

    float x_{0.0f};
    float y_{0.0f};
    float z_{0.0f};
    float w_{1.0f};
};

} // namespace fire_engine
