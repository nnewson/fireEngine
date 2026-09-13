#pragma once

#include <cmath>
#include <limits>
#include <optional>

#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/scalar.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// The conditioning threshold `tryInverse` applies to the NORMALISED determinant — its own named
// constant, because it is a policy about invertibility and not a comparison tolerance.
//
// 1e-9 sits about two orders BELOW float's epsilon (1.19e-7), so it is a permissive default: it
// rejects only what is numerically hopeless and leaves callers with a stricter requirement to say
// so. `makeVdpmViewParams` is one — its cone predicate wants a shape bound far tighter than this,
// and passes its own.
inline constexpr float kInverseConditionTolerance = 1.0e-9f;

// Column-major 3x3 matrix, mirroring Mat4's `[row, col]` accessor and storage
// (`m_[col * 3 + row]`). Used for rotation matrices and (inverse) inertia tensors
// in the rigid-body solver; kept minimal — only what the physics needs.
class Mat3
{
public:
    constexpr Mat3() noexcept
        : m_{}
    {
    }

    ~Mat3() = default;

    Mat3(const Mat3&) = default;
    Mat3& operator=(const Mat3&) = default;
    Mat3(Mat3&&) noexcept = default;
    Mat3& operator=(Mat3&&) noexcept = default;

    [[nodiscard]]
    constexpr float operator[](int row, int col) const noexcept
    {
        return m_[col * 3 + row];
    }

    constexpr float& operator[](int row, int col) noexcept
    {
        return m_[col * 3 + row];
    }

    [[nodiscard]]
    constexpr const float* data() const noexcept
    {
        return m_;
    }

    [[nodiscard]]
    static constexpr Mat3 identity() noexcept
    {
        Mat3 r;
        r.m_[0] = r.m_[4] = r.m_[8] = 1.0f;
        return r;
    }

    // Diagonal matrix from a vector (each component on the main diagonal).
    [[nodiscard]]
    static constexpr Mat3 diagonal(const Vec3& d) noexcept
    {
        Mat3 r;
        r.m_[0] = d.x();
        r.m_[4] = d.y();
        r.m_[8] = d.z();
        return r;
    }

    // Rotation matrix whose columns are the body axes (q rotating the basis
    // vectors). Equivalent to the upper-left 3x3 of `q.toMat4()`.
    [[nodiscard]]
    static constexpr Mat3 fromQuaternion(const Quaternion& q) noexcept
    {
        const Vec3 c0 = q.rotate(Vec3{1.0f, 0.0f, 0.0f});
        const Vec3 c1 = q.rotate(Vec3{0.0f, 1.0f, 0.0f});
        const Vec3 c2 = q.rotate(Vec3{0.0f, 0.0f, 1.0f});
        return fromColumns(c0, c1, c2);
    }

    [[nodiscard]]
    static constexpr Mat3 fromColumns(const Vec3& c0, const Vec3& c1, const Vec3& c2) noexcept
    {
        Mat3 r;
        r.m_[0] = c0.x();
        r.m_[1] = c0.y();
        r.m_[2] = c0.z();
        r.m_[3] = c1.x();
        r.m_[4] = c1.y();
        r.m_[5] = c1.z();
        r.m_[6] = c2.x();
        r.m_[7] = c2.y();
        r.m_[8] = c2.z();
        return r;
    }

    [[nodiscard]]
    constexpr Mat3 transpose() const noexcept
    {
        Mat3 r;
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                r.m_[col * 3 + row] = m_[row * 3 + col];
            }
        }
        return r;
    }

    // DOUBLE, and the return type is the point rather than the intermediates.
    //
    // A float determinant of a tiny transform underflows: a uniform scale of 1e-5 has determinant
    // 1e-15, and a tiny REFLECTED one lands at -0.0f — which reads as non-negative, so a caller
    // deriving orientation from `det >= 0` concludes the transform preserves winding when it
    // reverses it. VDPM does exactly that to fold a reflection into its cone facing, and the wrong
    // answer there enables cone culling with an inverted facing sign: geometry culled from the side
    // that should be visible. The magnitude is a conditioning question and the SIGN is an
    // orientation question, and double keeps both answerable for transforms a float determinant
    // cannot represent at all.
    [[nodiscard]]
    constexpr double determinant() const noexcept
    {
        const double a = m_[0], b = m_[3], c = m_[6]; // row 0
        const double d = m_[1], e = m_[4], f = m_[7]; // row 1
        const double g = m_[2], h = m_[5], i = m_[8]; // row 2
        return a * (e * i - f * h) + b * (f * g - d * i) + c * (d * h - e * g);
    }

    // THE inverse, and the only one: there is no `inverse()` returning a zero matrix any more.
    //
    // The old signature failed by returning `Mat3{}`, which is both a legitimate value and an error
    // report, so a caller could not tell "singular" from "the answer is zero" without checking
    // again — and its threshold was ABSOLUTE (`|det| <= 1e-12`), which rejects transforms that are
    // perfectly well conditioned merely for being small. A uniform scale of 1e-5 has determinant
    // 1e-15 and an exact inverse of 1e5; the old test called it singular and handed back zeros.
    // VDPM had already worked around this with its own scale-invariant predicate, then called the
    // absolute-threshold inverse anyway and got the zero matrix while its own test said "usable".
    //
    // SCALE-INVARIANT, so the question asked is conditioning rather than size: the matrix is
    // normalised by its largest absolute component before the determinant is taken, which puts a
    // uniformly scaled transform and its unit-scale twin on exactly the same footing. `tolerance`
    // is therefore a RELATIVE threshold on that normalised determinant, not a magnitude in the
    // caller's units.
    //
    // Non-finite input is rejected rather than propagated: an inverse built from a NaN is a matrix
    // of NaNs that every later operation quietly spreads.
    [[nodiscard]]
    std::optional<Mat3> tryInverse(float tolerance = kInverseConditionTolerance) const noexcept
    {
        // THE TOLERANCE IS VALIDATED FIRST, and not as a formality: `magnitude > tolerance` is TRUE
        // for a zero determinant against a negative tolerance, so a singular matrix would be
        // accepted and then divided by its own zero. A negative, NaN or infinite threshold is a
        // caller defect, and the answer to it is "no inverse", never "every matrix is invertible".
        if (!std::isfinite(tolerance) || tolerance < 0.0f)
        {
            return std::nullopt;
        }

        double scale = 0.0;
        for (const float value : m_)
        {
            if (!std::isfinite(value))
            {
                return std::nullopt;
            }
            const double magnitude =
                value < 0.0f ? -static_cast<double>(value) : static_cast<double>(value);
            scale = magnitude > scale ? magnitude : scale;
        }
        if (scale == 0.0)
        {
            return std::nullopt; // the zero matrix: singular, and the one case scaling cannot help
        }

        const double inverseScale = 1.0 / scale;
        const double a = m_[0] * inverseScale, b = m_[3] * inverseScale, c = m_[6] * inverseScale;
        const double d = m_[1] * inverseScale, e = m_[4] * inverseScale, f = m_[7] * inverseScale;
        const double g = m_[2] * inverseScale, h = m_[5] * inverseScale, i = m_[8] * inverseScale;
        const double A = e * i - f * h;
        const double B = f * g - d * i;
        const double C = d * h - e * g;
        const double normalisedDet = a * A + b * B + c * C;
        const double magnitude = normalisedDet < 0.0 ? -normalisedDet : normalisedDet;
        if (!(magnitude > static_cast<double>(tolerance)))
        {
            return std::nullopt; // `!(x > t)` so a NaN determinant is a refusal, not an acceptance
        }

        // The normalised inverse, scaled back: inv(s·M) = inv(M)/s.
        const double s = 1.0 / (normalisedDet * scale);
        const double elements[9]{
            A * s,
            B * s,
            C * s,
            (c * h - b * i) * s,
            (a * i - c * g) * s,
            (b * g - a * h) * s,
            (b * f - c * e) * s,
            (c * d - a * f) * s,
            (a * e - b * d) * s,
        };

        // WELL CONDITIONED IS NOT THE SAME AS REPRESENTABLE. A uniform scale of 1e-39 is perfectly
        // conditioned — its normalised determinant is 1 — and its inverse is 1e39, which float
        // cannot hold. Converting anyway would hand back an engaged optional full of infinities
        // (or, for values above float's range, an out-of-range conversion), so the caller would
        // believe it had an inverse and propagate garbage. Every element is checked in double
        // BEFORE any conversion happens, and the answer is "no usable inverse" instead.
        //
        // This is what makes the optional mean "a representable inverse exists", which is the only
        // claim a caller can act on.
        constexpr double kFloatMax = static_cast<double>(std::numeric_limits<float>::max());
        for (const double element : elements)
        {
            const double elementMagnitude = element < 0.0 ? -element : element;
            if (!std::isfinite(element) || elementMagnitude > kFloatMax)
            {
                return std::nullopt;
            }
        }

        Mat3 r;
        r[0, 0] = static_cast<float>(elements[0]);
        r[1, 0] = static_cast<float>(elements[1]);
        r[2, 0] = static_cast<float>(elements[2]);
        r[0, 1] = static_cast<float>(elements[3]);
        r[1, 1] = static_cast<float>(elements[4]);
        r[2, 1] = static_cast<float>(elements[5]);
        r[0, 2] = static_cast<float>(elements[6]);
        r[1, 2] = static_cast<float>(elements[7]);
        r[2, 2] = static_cast<float>(elements[8]);
        return r;
    }

    [[nodiscard]]
    constexpr Mat3 operator*(const Mat3& rhs) const noexcept
    {
        Mat3 r;
        for (int c = 0; c < 3; ++c)
        {
            for (int row = 0; row < 3; ++row)
            {
                for (int k = 0; k < 3; ++k)
                {
                    r.m_[c * 3 + row] += m_[k * 3 + row] * rhs.m_[c * 3 + k];
                }
            }
        }
        return r;
    }

    [[nodiscard]]
    constexpr Vec3 operator*(const Vec3& v) const noexcept
    {
        return {m_[0] * v.x() + m_[3] * v.y() + m_[6] * v.z(),
                m_[1] * v.x() + m_[4] * v.y() + m_[7] * v.z(),
                m_[2] * v.x() + m_[5] * v.y() + m_[8] * v.z()};
    }

    [[nodiscard]]
    constexpr Mat3 operator+(const Mat3& rhs) const noexcept
    {
        Mat3 r;
        for (int i = 0; i < 9; ++i)
        {
            r.m_[i] = m_[i] + rhs.m_[i];
        }
        return r;
    }

    [[nodiscard]]
    constexpr Mat3 operator-(const Mat3& rhs) const noexcept
    {
        Mat3 r;
        for (int i = 0; i < 9; ++i)
        {
            r.m_[i] = m_[i] - rhs.m_[i];
        }
        return r;
    }

    [[nodiscard]]
    constexpr Mat3 operator*(float s) const noexcept
    {
        Mat3 r;
        for (int i = 0; i < 9; ++i)
        {
            r.m_[i] = m_[i] * s;
        }
        return r;
    }

    // Skew-symmetric (cross-product) matrix of `v`: skew(v) * w == v × w. Its transpose
    // is skew(-v); used throughout spatial (6-D) rigid-body algebra for the r× coupling.
    [[nodiscard]]
    static constexpr Mat3 skew(const Vec3& v) noexcept
    {
        Mat3 r;
        r[0, 1] = -v.z();
        r[0, 2] = v.y();
        r[1, 0] = v.z();
        r[1, 2] = -v.x();
        r[2, 0] = -v.y();
        r[2, 1] = v.x();
        return r;
    }

    // EXACT component-wise IEEE equality — not bitwise, despite what this used to claim. Two
    // differences matter and both are the float `==` operator's, not ours: `-0.0f` equals `+0.0f`
    // though their bit patterns differ, and a NaN equals nothing at all though its bit pattern is
    // identical to itself. Use `approxEqual` when you want tolerance; if a determinism diagnostic
    // ever needs REAL bit comparison, it has to say so with `std::bit_cast`.
    [[nodiscard]]
    constexpr bool operator==(const Mat3& rhs) const noexcept
    {
        for (int i = 0; i < 9; ++i)
        {
            if (m_[i] != rhs.m_[i])
            {
                return false;
            }
        }
        return true;
    }

    // Approximate equality, component by component, through the ONE scalar authority
    // (`math/scalar.hpp`), and in its three forms — no argument means both defaults, an explicit
    // tolerance means ABSOLUTE ONLY (so `approxEqual(rhs, 1e-9f)` rejects anything further apart
    // than 1e-9, exactly as it always did), and both arguments mean both terms. An invalid
    // tolerance — negative, NaN or infinite — makes the comparison FALSE rather than being
    // reinterpreted. NaNs compare unequal now, which is the defect this replaced.
    [[nodiscard]]
    constexpr bool approxEqual(const Mat3& rhs, float eps, float relativeEps) const noexcept
    {
        for (int i = 0; i < 9; ++i)
        {
            if (!almostEqual(m_[i], rhs.m_[i], eps, relativeEps))
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]]
    constexpr bool approxEqual(const Mat3& rhs, float eps) const noexcept
    {
        // ABSOLUTE ONLY — a stated tolerance is the whole answer.
        return approxEqual(rhs, eps, 0.0f);
    }

    [[nodiscard]]
    constexpr bool approxEqual(const Mat3& rhs) const noexcept
    {
        return approxEqual(rhs, float_epsilon, float_relative_epsilon);
    }

private:
    float m_[9];
};

} // namespace fire_engine
