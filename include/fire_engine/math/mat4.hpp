#pragma once

#include <cmath>

#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>
#include <fire_engine/math/view_basis.hpp>

namespace fire_engine
{

class Mat4
{
public:
    constexpr Mat4() noexcept
        : m_{}
    {
    }

    ~Mat4() = default;

    Mat4(const Mat4&) = default;
    Mat4& operator=(const Mat4&) = default;
    Mat4(Mat4&&) noexcept = default;
    Mat4& operator=(Mat4&&) noexcept = default;

    [[nodiscard]]
    constexpr float operator[](int row, int col) const noexcept
    {
        return m_[col * 4 + row];
    }

    constexpr float& operator[](int row, int col) noexcept
    {
        return m_[col * 4 + row];
    }

    [[nodiscard]]
    constexpr const float* data() const noexcept
    {
        return m_;
    }

    [[nodiscard]]
    static constexpr Mat4 identity() noexcept
    {
        Mat4 r;
        r.m_[0] = r.m_[5] = r.m_[10] = r.m_[15] = 1.0f;
        return r;
    }

    [[nodiscard]]
    constexpr Mat4 operator*(const Mat4& rhs) const noexcept
    {
        Mat4 r;
        for (int c = 0; c < 4; ++c)
        {
            for (int row = 0; row < 4; ++row)
            {
                for (int k = 0; k < 4; ++k)
                {
                    r.m_[c * 4 + row] += m_[k * 4 + row] * rhs.m_[c * 4 + k];
                }
            }
        }
        return r;
    }

    constexpr Mat4& operator*=(const Mat4& rhs) noexcept
    {
        *this = *this * rhs;
        return *this;
    }

    [[nodiscard]]
    constexpr Vec4 operator*(const Vec4& rhs) const noexcept
    {
        return {m_[0] * rhs.x() + m_[4] * rhs.y() + m_[8] * rhs.z() + m_[12] * rhs.w(),
                m_[1] * rhs.x() + m_[5] * rhs.y() + m_[9] * rhs.z() + m_[13] * rhs.w(),
                m_[2] * rhs.x() + m_[6] * rhs.y() + m_[10] * rhs.z() + m_[14] * rhs.w(),
                m_[3] * rhs.x() + m_[7] * rhs.y() + m_[11] * rhs.z() + m_[15] * rhs.w()};
    }

    [[nodiscard]]
    constexpr bool operator==(const Mat4& rhs) const noexcept
    {
        for (int i = 0; i < 16; ++i)
        {
            if (m_[i] != rhs.m_[i])
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]]
    static Mat4 rotateY(float rad) noexcept
    {
        Mat4 r = identity();
        float c = std::cos(rad), s = std::sin(rad);
        r.m_[0] = c;
        r.m_[8] = s;
        r.m_[2] = -s;
        r.m_[10] = c;
        return r;
    }

    [[nodiscard]]
    static Mat4 rotateX(float rad) noexcept
    {
        Mat4 r = identity();
        float c = std::cos(rad), s = std::sin(rad);
        r.m_[5] = c;
        r.m_[9] = -s;
        r.m_[6] = s;
        r.m_[10] = c;
        return r;
    }

    [[nodiscard]]
    static Mat4 rotateZ(float rad) noexcept
    {
        Mat4 r = identity();
        float c = std::cos(rad), s = std::sin(rad);
        r.m_[0] = c;
        r.m_[4] = -s;
        r.m_[1] = s;
        r.m_[5] = c;
        return r;
    }

    [[nodiscard]]
    static constexpr Mat4 translate(Vec3 v) noexcept
    {
        Mat4 r = identity();
        r.m_[12] = v.x();
        r.m_[13] = v.y();
        r.m_[14] = v.z();
        return r;
    }

    [[nodiscard]]
    static constexpr Mat4 scale(Vec3 v) noexcept
    {
        Mat4 r = identity();
        r.m_[0] = v.x();
        r.m_[5] = v.y();
        r.m_[10] = v.z();
        return r;
    }

    [[nodiscard]]
    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) noexcept
    {
        const ViewBasis basis = makeViewBasis(eye, center, up);
        const float fx = basis.forward.x();
        const float fy = basis.forward.y();
        const float fz = basis.forward.z();
        const float sx = basis.right.x();
        const float sy = basis.right.y();
        const float sz = basis.right.z();
        const float ux = basis.up.x();
        const float uy = basis.up.y();
        const float uz = basis.up.z();
        Mat4 r = identity();
        r.m_[0] = sx;
        r.m_[4] = sy;
        r.m_[8] = sz;
        r.m_[12] = -(sx * eye.x() + sy * eye.y() + sz * eye.z());
        r.m_[1] = ux;
        r.m_[5] = uy;
        r.m_[9] = uz;
        r.m_[13] = -(ux * eye.x() + uy * eye.y() + uz * eye.z());
        r.m_[2] = -fx;
        r.m_[6] = -fy;
        r.m_[10] = -fz;
        r.m_[14] = (fx * eye.x() + fy * eye.y() + fz * eye.z());
        return r;
    }

    [[nodiscard]]
    static Mat4 perspective(float fovRad, float aspect, float near, float far) noexcept
    {
        float tanHalf = std::tan(fovRad / 2.0f);
        Mat4 r;
        r.m_[0] = 1.0f / (aspect * tanHalf);
        r.m_[5] = -1.0f / tanHalf; // Vulkan Y is flipped vs OpenGL
        r.m_[10] = far / (near - far);
        r.m_[11] = -1.0f;
        r.m_[14] = (near * far) / (near - far);
        return r;
    }

    [[nodiscard]]
    static constexpr Mat4 ortho(float left, float right, float bottom, float top, float near,
                                float far) noexcept
    {
        Mat4 r;
        r.m_[0] = 2.0f / (right - left);
        r.m_[5] = -2.0f / (top - bottom); // Vulkan Y flipped
        r.m_[10] = -1.0f / (far - near);  // Vulkan Z in [0,1], view forward is -Z
        r.m_[12] = -(right + left) / (right - left);
        r.m_[13] = (top + bottom) / (top - bottom);
        r.m_[14] = -near / (far - near);
        r.m_[15] = 1.0f;
        return r;
    }

private:
    float m_[16];
};

} // namespace fire_engine
