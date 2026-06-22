#include <fire_engine/collision/ray.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace fire_engine
{

namespace
{

constexpr float kParallelEpsilon = 1e-8f;

[[nodiscard]] float component(const Vec3& v, int axis) noexcept
{
    return axis == 0 ? v.x() : (axis == 1 ? v.y() : v.z());
}

// Nearest non-negative root of t² + 2·b·t + c = 0 within [0, maxDistance], or -1 if
// none. `dir` is unit so the quadratic's leading coefficient is 1.
[[nodiscard]] float nearestSphereRoot(float b, float c, float maxDistance) noexcept
{
    const float disc = b * b - c;
    if (disc < 0.0f)
    {
        return -1.0f;
    }
    const float sqrtDisc = std::sqrt(disc);
    float t = -b - sqrtDisc;
    if (t < 0.0f)
    {
        t = -b + sqrtDisc; // origin inside the sphere — take the forward surface
    }
    return (t < 0.0f || t > maxDistance) ? -1.0f : t;
}

} // namespace

bool rayIntersectsAabb(const Ray& ray, const AABB& box, float& tNear) noexcept
{
    float tmin = 0.0f;
    float tmax = ray.maxDistance;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float o = component(ray.origin, axis);
        const float d = component(ray.direction, axis);
        const float lo = component(box.min, axis);
        const float hi = component(box.max, axis);
        if (std::abs(d) < kParallelEpsilon)
        {
            if (o < lo || o > hi)
            {
                return false;
            }
            continue;
        }
        const float inv = 1.0f / d;
        float t1 = (lo - o) * inv;
        float t2 = (hi - o) * inv;
        if (t1 > t2)
        {
            std::swap(t1, t2);
        }
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmin > tmax)
        {
            return false;
        }
    }
    tNear = tmin;
    return true;
}

std::optional<RayHit> rayIntersect(const Ray& ray, const WorldSphere& sphere) noexcept
{
    const Vec3 m = ray.origin - sphere.center;
    const float b = Vec3::dotProduct(m, ray.direction);
    const float c = Vec3::dotProduct(m, m) - sphere.radius * sphere.radius;
    const float t = nearestSphereRoot(b, c, ray.maxDistance);
    if (t < 0.0f)
    {
        return std::nullopt;
    }
    const Vec3 point = ray.origin + ray.direction * t;
    return RayHit{t, point, Vec3::normalise(point - sphere.center)};
}

std::optional<RayHit> rayIntersect(const Ray& ray, const WorldBox& box) noexcept
{
    const std::array<Vec3, 3> axes{box.orientation.rotate(Vec3{1.0f, 0.0f, 0.0f}),
                                   box.orientation.rotate(Vec3{0.0f, 1.0f, 0.0f}),
                                   box.orientation.rotate(Vec3{0.0f, 0.0f, 1.0f})};
    const std::array<float, 3> half{box.halfExtents.x(), box.halfExtents.y(), box.halfExtents.z()};
    const Vec3 p = ray.origin - box.center;

    float tmin = 0.0f;
    float tmax = ray.maxDistance;
    int hitAxis = -1;
    float hitSign = 1.0f;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float o = Vec3::dotProduct(p, axes[static_cast<std::size_t>(axis)]);
        const float d = Vec3::dotProduct(ray.direction, axes[static_cast<std::size_t>(axis)]);
        const float h = half[static_cast<std::size_t>(axis)];
        if (std::abs(d) < kParallelEpsilon)
        {
            if (o < -h || o > h)
            {
                return std::nullopt;
            }
            continue;
        }
        const float inv = 1.0f / d;
        float t1 = (-h - o) * inv;
        float t2 = (h - o) * inv;
        float sign = -1.0f; // entering the -h face
        if (t1 > t2)
        {
            std::swap(t1, t2);
            sign = 1.0f;
        }
        if (t1 > tmin)
        {
            tmin = t1;
            hitAxis = axis;
            hitSign = sign;
        }
        tmax = std::min(tmax, t2);
        if (tmin > tmax)
        {
            return std::nullopt;
        }
    }
    if (hitAxis < 0 || tmin > ray.maxDistance)
    {
        return std::nullopt;
    }
    const Vec3 normal = axes[static_cast<std::size_t>(hitAxis)] * hitSign;
    return RayHit{tmin, ray.origin + ray.direction * tmin, normal};
}

std::optional<RayHit> rayIntersect(const Ray& ray, const WorldCapsule& capsule) noexcept
{
    // Inigo Quilez capsule intersection: solve the infinite-cylinder side, fall back to
    // the endcap spheres when the axial coordinate falls outside the segment.
    const Vec3 ba = capsule.p1 - capsule.p0;
    const Vec3 oa = ray.origin - capsule.p0;
    const float baba = Vec3::dotProduct(ba, ba);
    const float bard = Vec3::dotProduct(ba, ray.direction);
    const float baoa = Vec3::dotProduct(ba, oa);
    const float rdoa = Vec3::dotProduct(ray.direction, oa);
    const float oaoa = Vec3::dotProduct(oa, oa);
    const float r = capsule.radius;

    float bestT = -1.0f;
    Vec3 bestNormal{};

    if (baba > kParallelEpsilon)
    {
        const float a = baba - bard * bard;
        const float b = baba * rdoa - baoa * bard;
        const float c = baba * oaoa - baoa * baoa - r * r * baba;
        const float h = b * b - a * c;
        if (h >= 0.0f && std::abs(a) > kParallelEpsilon)
        {
            const float t = (-b - std::sqrt(h)) / a;
            const float y = baoa + t * bard;
            if (t >= 0.0f && y > 0.0f && y < baba)
            {
                bestT = t;
                const Vec3 onAxis = capsule.p0 + ba * (y / baba);
                bestNormal = Vec3::normalise(ray.origin + ray.direction * t - onAxis);
            }
        }
    }

    // Endcap spheres (also covers a zero-length capsule = sphere).
    const auto testCap = [&](const Vec3& center)
    {
        const Vec3 m = ray.origin - center;
        const float b = Vec3::dotProduct(m, ray.direction);
        const float c = Vec3::dotProduct(m, m) - r * r;
        const float t = nearestSphereRoot(b, c, ray.maxDistance);
        if (t >= 0.0f && (bestT < 0.0f || t < bestT))
        {
            bestT = t;
            bestNormal = Vec3::normalise(ray.origin + ray.direction * t - center);
        }
    };
    testCap(capsule.p0);
    testCap(capsule.p1);

    if (bestT < 0.0f || bestT > ray.maxDistance)
    {
        return std::nullopt;
    }
    return RayHit{bestT, ray.origin + ray.direction * bestT, bestNormal};
}

std::optional<RayHit> rayIntersect(const Ray& ray, const WorldConvex& convex) noexcept
{
    if (convex.vertices.empty() || convex.faces.empty())
    {
        return std::nullopt;
    }

    float tEnter = 0.0f;
    float tExit = ray.maxDistance;
    Vec3 enterNormal{};
    bool haveEnter = false;

    for (const ConvexFace& face : convex.faces)
    {
        if (face.loop.size() < 3)
        {
            continue;
        }
        // World outward normal, recomputed from the world vertices (the loop is
        // CCW-outward, so this cross product points outward).
        const Vec3& a = convex.vertices[static_cast<std::size_t>(face.loop[0])];
        const Vec3& b = convex.vertices[static_cast<std::size_t>(face.loop[1])];
        const Vec3& c = convex.vertices[static_cast<std::size_t>(face.loop[2])];
        const Vec3 normal = Vec3::normalise(Vec3::crossProduct(b - a, c - a));

        const float e0 = Vec3::dotProduct(ray.origin - a, normal);
        const float denom = Vec3::dotProduct(ray.direction, normal);
        if (std::abs(denom) < kParallelEpsilon)
        {
            if (e0 > 0.0f)
            {
                return std::nullopt; // parallel and outside this face
            }
            continue;
        }
        const float t = -e0 / denom;
        if (denom < 0.0f) // ray entering this half-space
        {
            if (t > tEnter)
            {
                tEnter = t;
                enterNormal = normal;
                haveEnter = true;
            }
        }
        else if (t < tExit) // ray exiting this half-space
        {
            tExit = t;
        }
        if (tEnter > tExit)
        {
            return std::nullopt;
        }
    }

    if (tEnter > ray.maxDistance)
    {
        return std::nullopt;
    }
    if (!haveEnter)
    {
        // Origin inside the hull — report a zero-distance hit facing back along the ray.
        return RayHit{0.0f, ray.origin, ray.direction * -1.0f};
    }
    return RayHit{tEnter, ray.origin + ray.direction * tEnter, enterNormal};
}

std::optional<RayHit> rayIntersect(const Ray& ray, const WorldShape& shape) noexcept
{
    return std::visit([&ray](const auto& s) { return rayIntersect(ray, s); }, shape);
}

std::optional<RayHit> rayIntersectTriangle(const Ray& ray, const Vec3& v0, const Vec3& v1,
                                           const Vec3& v2) noexcept
{
    const Vec3 e1 = v1 - v0;
    const Vec3 e2 = v2 - v0;
    const Vec3 pvec = Vec3::crossProduct(ray.direction, e2);
    const float det = Vec3::dotProduct(e1, pvec);
    if (std::abs(det) < kParallelEpsilon)
    {
        return std::nullopt; // ray parallel to the triangle plane
    }
    const float invDet = 1.0f / det;
    const Vec3 tvec = ray.origin - v0;
    const float u = Vec3::dotProduct(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return std::nullopt;
    }
    const Vec3 qvec = Vec3::crossProduct(tvec, e1);
    const float v = Vec3::dotProduct(ray.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return std::nullopt;
    }
    const float t = Vec3::dotProduct(e2, qvec) * invDet;
    if (t < 0.0f || t > ray.maxDistance)
    {
        return std::nullopt;
    }
    Vec3 normal = Vec3::normalise(Vec3::crossProduct(e1, e2));
    if (Vec3::dotProduct(normal, ray.direction) > 0.0f)
    {
        normal = normal * -1.0f; // face the hit back toward the ray
    }
    return RayHit{t, ray.origin + ray.direction * t, normal};
}

} // namespace fire_engine
