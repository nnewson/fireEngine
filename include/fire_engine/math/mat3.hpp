#pragma once

#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

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

    [[nodiscard]]
    constexpr float determinant() const noexcept
    {
        const float a = m_[0], b = m_[3], c = m_[6]; // row 0
        const float d = m_[1], e = m_[4], f = m_[7]; // row 1
        const float g = m_[2], h = m_[5], i = m_[8]; // row 2
        return a * (e * i - f * h) + b * (f * g - d * i) + c * (d * h - e * g);
    }

    // Inverse via the adjugate / determinant. Returns the zero matrix when (near-)singular
    // (|det| <= eps) so callers can detect it and fall back rather than propagate NaNs.
    [[nodiscard]]
    constexpr Mat3 inverse(float eps = 1.0e-12f) const noexcept
    {
        const float a = m_[0], b = m_[3], c = m_[6]; // row 0
        const float d = m_[1], e = m_[4], f = m_[7]; // row 1
        const float g = m_[2], h = m_[5], i = m_[8]; // row 2
        const float A = e * i - f * h;
        const float B = f * g - d * i;
        const float C = d * h - e * g;
        const float det = a * A + b * B + c * C;
        if (det <= eps && det >= -eps)
        {
            return Mat3{};
        }
        const float s = 1.0f / det;
        Mat3 r;
        r[0, 0] = A * s;
        r[0, 1] = (c * h - b * i) * s;
        r[0, 2] = (b * f - c * e) * s;
        r[1, 0] = B * s;
        r[1, 1] = (a * i - c * g) * s;
        r[1, 2] = (c * d - a * f) * s;
        r[2, 0] = C * s;
        r[2, 1] = (b * g - a * h) * s;
        r[2, 2] = (a * e - b * d) * s;
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

    // Strict bit-for-bit equality — use approxEqual for tolerance.
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

    [[nodiscard]]
    constexpr bool approxEqual(const Mat3& rhs, float eps = float_epsilon) const noexcept
    {
        for (int i = 0; i < 9; ++i)
        {
            const float d = m_[i] - rhs.m_[i];
            if (d > eps || d < -eps)
            {
                return false;
            }
        }
        return true;
    }

private:
    float m_[9];
};

} // namespace fire_engine
