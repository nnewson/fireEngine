#pragma once

#include <array>

#include <fire_engine/graphics/bounds.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

// A view frustum as six inward-pointing planes (a point p is inside the frustum when
// `normal·p + d >= 0` for all six). Extracted from a view-projection matrix by the
// Gribb-Hartmann method, for Vulkan clip space (depth in [0, 1]). Vulkan-free, so it
// can be unit-tested and lives beside Bounds3 in the graphics layer.
class Frustum
{
public:
    struct Plane
    {
        Vec3 normal{};
        float d{0.0f};
    };

    Frustum() = default;

    // Build from a world→clip matrix `m` (column-major; `m[row, col]`). clip = m·world,
    // and a point is inside when -w<=x<=w, -w<=y<=w, 0<=z<=w — each inequality is one
    // plane formed from the matrix rows.
    [[nodiscard]]
    static Frustum fromViewProj(const Mat4& m) noexcept
    {
        // Rows of m: rowK = (m[k,0], m[k,1], m[k,2], m[k,3]); clip.{x,y,z,w} = row{0,1,2,3}·world.
        // Side/top/bottom planes are row3 ± rowK; near (Vulkan [0,1] depth) is just row2.
        const auto combine = [&m](float sign, int k) -> Plane
        {
            return Plane{
                Vec3{m[3, 0] + sign * m[k, 0], m[3, 1] + sign * m[k, 1], m[3, 2] + sign * m[k, 2]},
                m[3, 3] + sign * m[k, 3]};
        };

        Frustum f;
        f.planes_[0] = combine(1.0f, 0);                                // left:   row3 + row0
        f.planes_[1] = combine(-1.0f, 0);                               // right:  row3 - row0
        f.planes_[2] = combine(1.0f, 1);                                // bottom: row3 + row1
        f.planes_[3] = combine(-1.0f, 1);                               // top:    row3 - row1
        f.planes_[4] = Plane{Vec3{m[2, 0], m[2, 1], m[2, 2]}, m[2, 3]}; // near:   row2
        f.planes_[5] = combine(-1.0f, 2);                               // far:    row3 - row2
        return f;
    }

    // Whether the AABB is (at least partly) inside the frustum. The "positive vertex"
    // test: the box is rejected only when its farthest vertex along a plane normal is
    // still behind that plane, so it never reports a visible box as outside (a box near
    // a frustum edge may be conservatively reported inside). An invalid Bounds3 (e.g.
    // the unbounded skybox) is always visible.
    [[nodiscard]]
    bool intersects(const Bounds3& bounds) const noexcept
    {
        if (!bounds.valid)
        {
            return true;
        }
        for (const Plane& plane : planes_)
        {
            const Vec3 positive{plane.normal.x() >= 0.0f ? bounds.max.x() : bounds.min.x(),
                                plane.normal.y() >= 0.0f ? bounds.max.y() : bounds.min.y(),
                                plane.normal.z() >= 0.0f ? bounds.max.z() : bounds.min.z()};
            if (Vec3::dotProduct(plane.normal, positive) + plane.d < 0.0f)
            {
                return false; // fully behind this plane ⇒ outside
            }
        }
        return true;
    }

private:
    std::array<Plane, 6> planes_{};
};

} // namespace fire_engine
