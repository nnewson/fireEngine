#include <fire_engine/collision/narrow_phase.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <variant>
#include <vector>

#include <fire_engine/collision/geometry.hpp>
#include <fire_engine/math/constants.hpp>

namespace fire_engine
{

namespace
{

constexpr float kGeomEps = 1e-6f;

[[nodiscard]] float dot(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::dotProduct(a, b);
}

// One-point manifold with the normal pointing b -> a (the convention `collide`
// returns). `point` is the world contact, `pen` the penetration depth.
[[nodiscard]] ContactManifold onePoint(const Vec3& normal, const Vec3& point, float pen) noexcept
{
    ContactManifold m;
    m.normal = normal;
    m.points[0] = {point, pen};
    m.count = 1;
    return m;
}

// Flip a manifold's normal (used when the dispatch evaluated the swapped pair).
[[nodiscard]] ContactManifold flipped(ContactManifold m) noexcept
{
    m.normal = m.normal * -1.0f;
    return m;
}

// ----- sphere / sphere / capsule analytic pairs (normal points b -> a) -------

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldSphere& a,
                                                         const WorldSphere& b) noexcept
{
    const Vec3 d = a.center - b.center; // b -> a
    const float dist = d.magnitude();
    const float rsum = a.radius + b.radius;
    if (dist >= rsum)
    {
        return std::nullopt;
    }
    const Vec3 n = dist > kGeomEps ? d * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    const Vec3 onA = a.center - n * a.radius;
    const Vec3 onB = b.center + n * b.radius;
    return onePoint(n, (onA + onB) * 0.5f, rsum - dist);
}

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldSphere& s,
                                                         const WorldBox& box) noexcept
{
    const Vec3 closest = closestPointOnObb(s.center, box);
    Vec3 d = s.center - closest; // box surface -> sphere centre
    const float dist2 = d.magnitudeSquared();
    if (dist2 >= s.radius * s.radius)
    {
        return std::nullopt;
    }
    Vec3 n;
    float pen;
    if (dist2 > kGeomEps * kGeomEps)
    {
        const float dist = std::sqrt(dist2);
        n = d * (1.0f / dist);
        pen = s.radius - dist;
    }
    else
    {
        // Sphere centre inside the box: push out along the least-penetrated face.
        const Vec3 local = box.orientation.conjugate().rotate(s.center - box.center);
        const float px = box.halfExtents.x() - std::abs(local.x());
        const float py = box.halfExtents.y() - std::abs(local.y());
        const float pz = box.halfExtents.z() - std::abs(local.z());
        Vec3 localN;
        if (px <= py && px <= pz)
        {
            localN = {local.x() < 0.0f ? -1.0f : 1.0f, 0.0f, 0.0f};
            pen = px + s.radius;
        }
        else if (py <= pz)
        {
            localN = {0.0f, local.y() < 0.0f ? -1.0f : 1.0f, 0.0f};
            pen = py + s.radius;
        }
        else
        {
            localN = {0.0f, 0.0f, local.z() < 0.0f ? -1.0f : 1.0f};
            pen = pz + s.radius;
        }
        n = box.orientation.rotate(localN);
    }
    return onePoint(n, closest, pen);
}

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldSphere& s,
                                                         const WorldCapsule& cap) noexcept
{
    const Vec3 closest = closestPointOnSegment(s.center, cap.p0, cap.p1);
    const Vec3 d = s.center - closest;
    const float dist = d.magnitude();
    const float rsum = s.radius + cap.radius;
    if (dist >= rsum)
    {
        return std::nullopt;
    }
    const Vec3 n = dist > kGeomEps ? d * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    const Vec3 onS = s.center - n * s.radius;
    const Vec3 onC = closest + n * cap.radius;
    return onePoint(n, (onS + onC) * 0.5f, rsum - dist);
}

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldCapsule& a,
                                                         const WorldCapsule& b) noexcept
{
    const SegmentClosest cl = closestPointsBetweenSegments(a.p0, a.p1, b.p0, b.p1);
    const Vec3 d = cl.c1 - cl.c2; // b -> a
    const float dist = d.magnitude();
    const float rsum = a.radius + b.radius;
    if (dist >= rsum)
    {
        return std::nullopt;
    }
    const Vec3 n = dist > kGeomEps ? d * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    const Vec3 onA = cl.c1 - n * a.radius;
    const Vec3 onB = cl.c2 + n * b.radius;
    return onePoint(n, (onA + onB) * 0.5f, rsum - dist);
}

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldBox& box,
                                                         const WorldCapsule& cap) noexcept
{
    // Closest point between the capsule segment and the box, by alternating
    // closest-on-segment / closest-on-box a few times (converges quickly), then
    // treat the capsule as a sphere at that segment point. Normal points cap->box
    // (b -> a where a = box).
    Vec3 onSeg = closestPointOnSegment(box.center, cap.p0, cap.p1);
    Vec3 onBox = closestPointOnObb(onSeg, box);
    for (int i = 0; i < 4; ++i)
    {
        onSeg = closestPointOnSegment(onBox, cap.p0, cap.p1);
        onBox = closestPointOnObb(onSeg, box);
    }
    Vec3 d = onBox - onSeg; // cap segment -> box surface
    const float dist = d.magnitude();
    if (dist >= cap.radius)
    {
        return std::nullopt;
    }
    // Normal cap -> box.
    const Vec3 n = dist > kGeomEps ? d * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    return onePoint(n, onBox, cap.radius - dist);
}

// ----- box / box SAT + face clipping (normal points b -> a) ------------------

struct BoxFrame
{
    Vec3 c;
    std::array<Vec3, 3> axis;
    std::array<float, 3> he;
};

[[nodiscard]] BoxFrame frameOf(const WorldBox& b) noexcept
{
    return {b.center,
            {b.orientation.rotate({1.0f, 0.0f, 0.0f}), b.orientation.rotate({0.0f, 1.0f, 0.0f}),
             b.orientation.rotate({0.0f, 0.0f, 1.0f})},
            {b.halfExtents.x(), b.halfExtents.y(), b.halfExtents.z()}};
}

[[nodiscard]] float projRadius(const BoxFrame& f, const Vec3& L) noexcept
{
    return std::abs(dot(L, f.axis[0])) * f.he[0] + std::abs(dot(L, f.axis[1])) * f.he[1] +
           std::abs(dot(L, f.axis[2])) * f.he[2];
}

// Signed separation along unit axis L: >0 separated, <=0 overlapping (pen = -s).
[[nodiscard]] float separation(const BoxFrame& a, const BoxFrame& b, const Vec3& t,
                               const Vec3& L) noexcept
{
    return std::abs(dot(t, L)) - (projRadius(a, L) + projRadius(b, L));
}

// The 4 world vertices of box f's face with outward normal sign*axis[ai].
[[nodiscard]] std::array<Vec3, 4> faceVertices(const BoxFrame& f, int ai, float sign) noexcept
{
    const int u = (ai + 1) % 3;
    const int v = (ai + 2) % 3;
    const Vec3 base = f.c + f.axis[ai] * (sign * f.he[ai]);
    const Vec3 du = f.axis[u] * f.he[u];
    const Vec3 dv = f.axis[v] * f.he[v];
    return {base - du - dv, base + du - dv, base + du + dv, base - du + dv};
}

// Clip polygon against the half-space dot(n, p) <= d (Sutherland-Hodgman).
void clipAgainstPlane(std::vector<Vec3>& poly, const Vec3& n, float d)
{
    std::vector<Vec3> out;
    out.reserve(poly.size() + 1);
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const Vec3& cur = poly[i];
        const Vec3& nxt = poly[(i + 1) % poly.size()];
        const float dc = dot(n, cur) - d;
        const float dn = dot(n, nxt) - d;
        if (dc <= 0.0f)
        {
            out.push_back(cur);
        }
        if ((dc < 0.0f) != (dn < 0.0f))
        {
            const float tt = dc / (dc - dn);
            out.push_back(cur + (nxt - cur) * tt);
        }
    }
    poly.swap(out);
}

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldBox& A,
                                                         const WorldBox& B) noexcept
{
    const BoxFrame a = frameOf(A);
    const BoxFrame b = frameOf(B);
    const Vec3 t = b.c - a.c; // a -> b

    // Best face axis on each box (max separation = least penetration) + best edge.
    float bestFaceSep = -std::numeric_limits<float>::infinity();
    Vec3 bestFaceAxis;
    const BoxFrame* refBox = &a;
    int refAxis = 0;
    for (int box = 0; box < 2; ++box)
    {
        const BoxFrame& f = box == 0 ? a : b;
        for (int i = 0; i < 3; ++i)
        {
            const float s = separation(a, b, t, f.axis[i]);
            if (s > 0.0f)
            {
                return std::nullopt; // separating axis found
            }
            if (s > bestFaceSep)
            {
                bestFaceSep = s;
                bestFaceAxis = f.axis[i];
                refBox = box == 0 ? &a : &b;
                refAxis = i;
            }
        }
    }

    float bestEdgeSep = -std::numeric_limits<float>::infinity();
    Vec3 bestEdgeAxis;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            Vec3 L = Vec3::crossProduct(a.axis[i], b.axis[j]);
            if (L.magnitudeSquared() < kGeomEps)
            {
                continue; // parallel edges — degenerate axis
            }
            L = Vec3::normalise(L);
            const float s = separation(a, b, t, L);
            if (s > 0.0f)
            {
                return std::nullopt;
            }
            if (s > bestEdgeSep)
            {
                bestEdgeSep = s;
                bestEdgeAxis = L;
            }
        }
    }

    // Prefer a face contact unless an edge axis is clearly less penetrating.
    constexpr float kFaceBias = 1e-3f;
    if (bestEdgeSep > bestFaceSep + kFaceBias)
    {
        // Edge-edge: single contact point, normal = edge axis oriented b -> a.
        Vec3 n = bestEdgeAxis;
        if (dot(n, t) > 0.0f) // t is a -> b; we want b -> a
        {
            n = n * -1.0f;
        }
        return onePoint(n, (a.c + b.c) * 0.5f, -bestEdgeSep);
    }

    // Face contact. Reference normal points from the reference box outward toward
    // the other (incident) box.
    const BoxFrame& ref = *refBox;
    const BoxFrame& inc = (refBox == &a) ? b : a;
    const Vec3 refToInc = inc.c - ref.c;
    const float refSign = dot(bestFaceAxis, refToInc) >= 0.0f ? 1.0f : -1.0f;
    const Vec3 refNormal = bestFaceAxis * refSign; // ref -> inc

    // Incident face: the one on `inc` most antiparallel to refNormal.
    int incAxis = 0;
    float incSign = 1.0f;
    float minDotN = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; ++i)
    {
        for (float sgn : {1.0f, -1.0f})
        {
            const float dd = dot(inc.axis[i] * sgn, refNormal);
            if (dd < minDotN)
            {
                minDotN = dd;
                incAxis = i;
                incSign = sgn;
            }
        }
    }

    std::array<Vec3, 4> incFace = faceVertices(inc, incAxis, incSign);
    std::vector<Vec3> poly(incFace.begin(), incFace.end());

    // Clip against the reference face's 4 side planes.
    const int ru = (refAxis + 1) % 3;
    const int rv = (refAxis + 2) % 3;
    for (int sideAxis : {ru, rv})
    {
        for (float sgn : {1.0f, -1.0f})
        {
            const Vec3 sideN = ref.axis[sideAxis] * sgn;
            const float sideD = dot(sideN, ref.c) + ref.he[sideAxis];
            clipAgainstPlane(poly, sideN, sideD);
        }
    }

    // Keep points below the reference face (penetrating); penetration = depth.
    const float refFaceD = dot(refNormal, ref.c) + ref.he[refAxis];
    Vec3 nBA = refNormal; // ref -> inc; convert to b -> a below
    // We want the manifold normal to point b -> a. t = a -> b, so b -> a opposes t.
    if (dot(nBA, b.c - a.c) > 0.0f)
    {
        nBA = nBA * -1.0f;
    }

    ContactManifold m;
    m.normal = nBA;
    for (const Vec3& p : poly)
    {
        const float depth = refFaceD - dot(refNormal, p);
        if (depth < -1e-4f)
        {
            continue; // above the face, not in contact
        }
        if (m.count >= kMaxManifoldPoints)
        {
            // Replace the shallowest if this one is deeper.
            int shallow = 0;
            for (int i = 1; i < m.count; ++i)
            {
                if (m.points[i].penetration < m.points[shallow].penetration)
                {
                    shallow = i;
                }
            }
            if (depth > m.points[shallow].penetration)
            {
                m.points[shallow] = {p, std::max(depth, 0.0f)};
            }
            continue;
        }
        m.points[m.count++] = {p, std::max(depth, 0.0f)};
    }
    if (m.count == 0)
    {
        // Fallback to a single deepest point if clipping produced nothing.
        m.points[0] = {(a.c + b.c) * 0.5f, -bestFaceSep};
        m.count = 1;
    }
    return m;
}

// Swapped overloads: evaluate the canonical pair, then flip the normal.
[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldBox& box,
                                                         const WorldSphere& s) noexcept
{
    auto m = collidePair(s, box);
    return m ? std::optional<ContactManifold>{flipped(*m)} : std::nullopt;
}
[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldCapsule& cap,
                                                         const WorldSphere& s) noexcept
{
    auto m = collidePair(s, cap);
    return m ? std::optional<ContactManifold>{flipped(*m)} : std::nullopt;
}
[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldCapsule& cap,
                                                         const WorldBox& box) noexcept
{
    auto m = collidePair(box, cap);
    return m ? std::optional<ContactManifold>{flipped(*m)} : std::nullopt;
}


[[nodiscard]]
Vec3 axisNormal(Axis axis, float direction) noexcept
{
    switch (axis)
    {
    case Axis::X:
        return {direction, 0.0f, 0.0f};
    case Axis::Y:
        return {0.0f, direction, 0.0f};
    case Axis::Z:
        return {0.0f, 0.0f, direction};
    }

    return {direction, 0.0f, 0.0f};
}

[[nodiscard]]
bool intervalsOverlap(const AABB& lhs, const AABB& rhs, Axis axis) noexcept
{
    return lhs.axisMin(axis) <= rhs.axisMax(axis) && lhs.axisMax(axis) >= rhs.axisMin(axis);
}

[[nodiscard]]
bool aabbOverlap(const AABB& lhs, const AABB& rhs) noexcept
{
    return intervalsOverlap(lhs, rhs, Axis::X) && intervalsOverlap(lhs, rhs, Axis::Y) &&
           intervalsOverlap(lhs, rhs, Axis::Z);
}

[[nodiscard]]
float axisComponent(Vec3 value, Axis axis) noexcept
{
    switch (axis)
    {
    case Axis::X:
        return value.x();
    case Axis::Y:
        return value.y();
    case Axis::Z:
        return value.z();
    }

    return value.x();
}

[[nodiscard]]
Vec3 boundsDelta(const AABB& previous, const AABB& current) noexcept
{
    return {current.min.x() - previous.min.x(), current.min.y() - previous.min.y(),
            current.min.z() - previous.min.z()};
}

[[nodiscard]]
Vec3 startingOverlapNormal(const AABB& movingBounds, const AABB& targetBounds) noexcept
{
    const std::array<Axis, 3> axes{Axis::X, Axis::Y, Axis::Z};
    Axis bestAxis = Axis::X;
    float bestDepth = std::numeric_limits<float>::max();

    for (Axis axis : axes)
    {
        const float depth = std::min(movingBounds.axisMax(axis), targetBounds.axisMax(axis)) -
                            std::max(movingBounds.axisMin(axis), targetBounds.axisMin(axis));
        if (depth < bestDepth)
        {
            bestDepth = depth;
            bestAxis = axis;
        }
    }

    const float movingCentre =
        (movingBounds.axisMin(bestAxis) + movingBounds.axisMax(bestAxis)) * 0.5f;
    const float targetCentre =
        (targetBounds.axisMin(bestAxis) + targetBounds.axisMax(bestAxis)) * 0.5f;
    return axisNormal(bestAxis, movingCentre < targetCentre ? -1.0f : 1.0f);
}

} // namespace

std::optional<ContactManifold> NarrowPhase::collide(const WorldShape& a,
                                                    const WorldShape& b) const noexcept
{
    // 2-arg visit dispatches to the matching collidePair overload (6 canonical +
    // 3 swapped). The result normal points b -> a.
    return std::visit([](const auto& sa, const auto& sb) { return collidePair(sa, sb); }, a, b);
}

std::optional<SweptAabbContact> NarrowPhase::sweptAabb(const Collider& moving,
                                                       const Collider& target) const noexcept
{
    const AABB movingStart = moving.previousWorldBounds();
    const AABB movingEnd = moving.worldBounds();
    const AABB targetStart = target.previousWorldBounds();
    const AABB targetEnd = target.worldBounds();
    const Vec3 relativeDelta =
        boundsDelta(movingStart, movingEnd) - boundsDelta(targetStart, targetEnd);

    if (relativeDelta.magnitudeSquared() < float_epsilon * float_epsilon)
    {
        if (!aabbOverlap(movingEnd, targetEnd))
        {
            return std::nullopt;
        }
        return SweptAabbContact{0.0f, startingOverlapNormal(movingEnd, targetEnd)};
    }

    const std::array<Axis, 3> axes{Axis::X, Axis::Y, Axis::Z};
    float entryTime = -std::numeric_limits<float>::infinity();
    float exitTime = std::numeric_limits<float>::infinity();
    Vec3 normal{};

    for (Axis axis : axes)
    {
        const float delta = axisComponent(relativeDelta, axis);
        float axisEntry = -std::numeric_limits<float>::infinity();
        float axisExit = std::numeric_limits<float>::infinity();
        Vec3 axisEntryNormal{};

        if (std::abs(delta) < float_epsilon)
        {
            if (!intervalsOverlap(movingStart, targetStart, axis))
            {
                return std::nullopt;
            }
        }
        else if (delta > 0.0f)
        {
            axisEntry = (targetStart.axisMin(axis) - movingStart.axisMax(axis)) / delta;
            axisExit = (targetStart.axisMax(axis) - movingStart.axisMin(axis)) / delta;
            axisEntryNormal = axisNormal(axis, -1.0f);
        }
        else
        {
            axisEntry = (targetStart.axisMax(axis) - movingStart.axisMin(axis)) / delta;
            axisExit = (targetStart.axisMin(axis) - movingStart.axisMax(axis)) / delta;
            axisEntryNormal = axisNormal(axis, 1.0f);
        }

        if (axisEntry > entryTime)
        {
            entryTime = axisEntry;
            normal = axisEntryNormal;
        }
        exitTime = std::min(exitTime, axisExit);
    }

    if (entryTime > exitTime || exitTime < 0.0f || entryTime > 1.0f)
    {
        return std::nullopt;
    }

    if (entryTime < 0.0f)
    {
        return SweptAabbContact{0.0f, startingOverlapNormal(movingStart, targetStart)};
    }

    return SweptAabbContact{entryTime, normal};
}

} // namespace fire_engine
