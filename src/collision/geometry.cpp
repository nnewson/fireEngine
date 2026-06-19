#include <fire_engine/collision/geometry.hpp>

#include <algorithm>

namespace fire_engine
{

namespace
{
constexpr float kEps = 1e-8f;
}

Vec3 closestPointOnSegment(const Vec3& p, const Vec3& a, const Vec3& b) noexcept
{
    const Vec3 ab = b - a;
    const float len2 = ab.magnitudeSquared();
    if (len2 < kEps)
    {
        return a;
    }
    const float t = std::clamp(Vec3::dotProduct(p - a, ab) / len2, 0.0f, 1.0f);
    return a + ab * t;
}

Vec3 closestPointOnObb(const Vec3& p, const WorldBox& box) noexcept
{
    // Into box-local space (conjugate undoes the orientation), clamp to the
    // half-extents, then back to world.
    const Vec3 local = box.orientation.conjugate().rotate(p - box.center);
    const Vec3 clamped{std::clamp(local.x(), -box.halfExtents.x(), box.halfExtents.x()),
                       std::clamp(local.y(), -box.halfExtents.y(), box.halfExtents.y()),
                       std::clamp(local.z(), -box.halfExtents.z(), box.halfExtents.z())};
    return box.center + box.orientation.rotate(clamped);
}

SegmentClosest closestPointsBetweenSegments(const Vec3& p1, const Vec3& q1, const Vec3& p2,
                                            const Vec3& q2) noexcept
{
    const Vec3 d1 = q1 - p1; // direction of segment 1
    const Vec3 d2 = q2 - p2; // direction of segment 2
    const Vec3 r = p1 - p2;
    const float a = d1.magnitudeSquared();
    const float e = d2.magnitudeSquared();
    const float f = Vec3::dotProduct(d2, r);

    float s = 0.0f;
    float t = 0.0f;
    if (a < kEps && e < kEps)
    {
        // Both segments are points.
        return {p1, p2};
    }
    if (a < kEps)
    {
        // Segment 1 is a point.
        t = std::clamp(f / e, 0.0f, 1.0f);
    }
    else
    {
        const float c = Vec3::dotProduct(d1, r);
        if (e < kEps)
        {
            // Segment 2 is a point.
            s = std::clamp(-c / a, 0.0f, 1.0f);
        }
        else
        {
            const float b = Vec3::dotProduct(d1, d2);
            const float denom = a * e - b * b; // always >= 0
            if (denom > kEps)
            {
                s = std::clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            }
            else
            {
                s = 0.0f; // parallel — pick an arbitrary point on segment 1
            }
            t = (b * s + f) / e;
            if (t < 0.0f)
            {
                t = 0.0f;
                s = std::clamp(-c / a, 0.0f, 1.0f);
            }
            else if (t > 1.0f)
            {
                t = 1.0f;
                s = std::clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    return {p1 + d1 * s, p2 + d2 * t};
}

} // namespace fire_engine
