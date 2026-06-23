#include <fire_engine/collision/gjk_epa.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <fire_engine/collision/support.hpp>
#include <fire_engine/math/vec3.hpp>

namespace fire_engine
{

namespace
{

constexpr int kMaxGjkIterations = 32;
constexpr int kMaxEpaIterations = 64;
constexpr float kGjkEps = 1e-10f;
// Relative slack on the GJK convergence test, scaling with the closest feature's squared
// distance so large shapes (whose support dot products carry larger FP noise) still
// converge instead of over-growing the simplex into a false origin enclosure.
constexpr float kGjkRel = 1e-7f;
constexpr float kEpaEps = 1e-5f;

[[nodiscard]] float dot(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::dotProduct(a, b);
}

[[nodiscard]] Vec3 cross(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::crossProduct(a, b);
}

// A vertex of the Minkowski difference A⊖B: `v = sa − sb`, with the supporting
// points on each shape kept so contact witnesses can be recovered by barycentric.
struct SimplexVertex
{
    Vec3 v{};
    Vec3 sa{};
    Vec3 sb{};
};

[[nodiscard]] SimplexVertex minkowskiSupport(const WorldShape& a, const WorldShape& b,
                                             const Vec3& dir) noexcept
{
    const Vec3 sa = supportPoint(a, dir);
    const Vec3 sb = supportPoint(b, dir * -1.0f);
    return {sa - sb, sa, sb};
}

// Barycentric weights over up to 4 simplex vertices for the point on the simplex
// closest to the origin, plus that point. `containsOrigin` is only meaningful for
// the 4-vertex (tetrahedron) case. Vertices with zero weight are dropped by GJK.
struct Closest
{
    Vec3 point{};
    std::array<float, 4> weight{};
    bool containsOrigin{false};
};

// Closest point on segment [a,b] to the origin (barycentric la, lb).
void closestSegment(const Vec3& a, const Vec3& b, float& la, float& lb) noexcept
{
    const Vec3 ab = b - a;
    const float denom = dot(ab, ab);
    float t = denom > kGjkEps ? dot(a * -1.0f, ab) / denom : 0.0f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    la = 1.0f - t;
    lb = t;
}

// Closest point on triangle [a,b,c] to the origin (barycentric u,v,w). Ericson RTCD.
void closestTriangle(const Vec3& a, const Vec3& b, const Vec3& c, float& u, float& v,
                     float& w) noexcept
{
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = a * -1.0f;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        u = 1.0f;
        v = 0.0f;
        w = 0.0f;
        return;
    }
    const Vec3 bp = b * -1.0f;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        u = 0.0f;
        v = 1.0f;
        w = 0.0f;
        return;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float t = d1 / (d1 - d3);
        u = 1.0f - t;
        v = t;
        w = 0.0f;
        return;
    }
    const Vec3 cp = c * -1.0f;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        u = 0.0f;
        v = 0.0f;
        w = 1.0f;
        return;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float t = d2 / (d2 - d6);
        u = 1.0f - t;
        v = 0.0f;
        w = t;
        return;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        u = 0.0f;
        v = 1.0f - t;
        w = t;
        return;
    }
    const float denom = 1.0f / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    u = 1.0f - v - w;
}

[[nodiscard]] bool originOutsidePlane(const Vec3& a, const Vec3& b, const Vec3& c,
                                      const Vec3& d) noexcept
{
    const Vec3 n = cross(b - a, c - a);
    const float signOrigin = dot(n, a * -1.0f); // dot(n, origin - a)
    const float signD = dot(n, d - a);
    return signOrigin * signD < 0.0f; // origin on the opposite side from d
}

// Closest point + barycentric over a simplex of `count` (1..4) Minkowski vertices.
[[nodiscard]] Closest closestOnSimplex(const std::array<SimplexVertex, 4>& s, int count) noexcept
{
    Closest result;
    if (count == 1)
    {
        result.weight[0] = 1.0f;
        result.point = s[0].v;
        return result;
    }
    if (count == 2)
    {
        closestSegment(s[0].v, s[1].v, result.weight[0], result.weight[1]);
        result.point = s[0].v * result.weight[0] + s[1].v * result.weight[1];
        return result;
    }
    if (count == 3)
    {
        closestTriangle(s[0].v, s[1].v, s[2].v, result.weight[0], result.weight[1],
                        result.weight[2]);
        result.point =
            s[0].v * result.weight[0] + s[1].v * result.weight[1] + s[2].v * result.weight[2];
        return result;
    }

    // Tetrahedron: test each face the origin lies outside of, keep the nearest;
    // if outside none, the origin is enclosed.
    const std::array<std::array<int, 4>, 4> faces{{
        {0, 1, 2, 3}, // face abc, opposite d
        {0, 2, 3, 1}, // face acd, opposite b
        {0, 3, 1, 2}, // face adb, opposite c
        {1, 3, 2, 0}, // face bdc, opposite a
    }};

    // A near-coplanar (zero-volume) tetrahedron can't actually enclose the origin, and its
    // per-face "origin outside plane" tests are numerically unreliable (the opposite vertex
    // sits on the plane). This happens with large axis-aligned shapes whose supports all
    // land on one Minkowski face. Treat it as separated: take the closest of all four faces
    // rather than risk a false enclosure → spurious collision.
    const float volume = dot(s[3].v - s[0].v, cross(s[1].v - s[0].v, s[2].v - s[0].v));
    const float edgeScale = (s[1].v - s[0].v).magnitude() * (s[2].v - s[0].v).magnitude() *
                            (s[3].v - s[0].v).magnitude();
    const bool degenerate = std::abs(volume) <= 1e-7f * (edgeScale + 1e-20f);

    float bestSq = std::numeric_limits<float>::max();
    bool anyOutside = false;
    for (const auto& f : faces)
    {
        if (!degenerate && !originOutsidePlane(s[f[0]].v, s[f[1]].v, s[f[2]].v, s[f[3]].v))
        {
            continue;
        }
        anyOutside = true;
        float u = 0.0f;
        float v = 0.0f;
        float w = 0.0f;
        closestTriangle(s[f[0]].v, s[f[1]].v, s[f[2]].v, u, v, w);
        const Vec3 p = s[f[0]].v * u + s[f[1]].v * v + s[f[2]].v * w;
        const float dsq = dot(p, p);
        if (dsq < bestSq)
        {
            bestSq = dsq;
            result.point = p;
            result.weight = {0.0f, 0.0f, 0.0f, 0.0f};
            result.weight[f[0]] = u;
            result.weight[f[1]] = v;
            result.weight[f[2]] = w;
        }
    }
    if (!degenerate && !anyOutside)
    {
        result.containsOrigin = true;
    }
    return result;
}

// ----- EPA: expand the GJK tetrahedron to the penetration normal + depth --------

struct Face
{
    int a{0};
    int b{0};
    int c{0};
    Vec3 normal{}; // outward (away from the origin, which is enclosed)
    float dist{0.0f};
};

[[nodiscard]] Face makeFace(const std::vector<SimplexVertex>& verts, int a, int b, int c) noexcept
{
    Vec3 n = cross(verts[static_cast<std::size_t>(b)].v - verts[static_cast<std::size_t>(a)].v,
                   verts[static_cast<std::size_t>(c)].v - verts[static_cast<std::size_t>(a)].v);
    const float len = n.magnitude();
    n = len > 1e-12f ? n * (1.0f / len) : Vec3{0.0f, 1.0f, 0.0f};
    float d = dot(n, verts[static_cast<std::size_t>(a)].v);
    if (d < 0.0f) // flip to point away from the enclosed origin
    {
        n = n * -1.0f;
        d = -d;
    }
    return {a, b, c, n, d};
}

// Add directed edge (i,j) to the horizon; if its reverse already exists it is an
// interior edge shared by two visible faces and cancels out.
void addHorizonEdge(std::vector<std::pair<int, int>>& horizon, int i, int j) noexcept
{
    for (std::size_t k = 0; k < horizon.size(); ++k)
    {
        if (horizon[k].first == j && horizon[k].second == i)
        {
            horizon[k] = horizon.back();
            horizon.pop_back();
            return;
        }
    }
    horizon.emplace_back(i, j);
}

// Expand the enclosing tetrahedron `tetra` to the closest face of the Minkowski
// polytope → penetration normal/depth + surface witnesses. Normal points b -> a.
[[nodiscard]] ConvexContact epa(const WorldShape& a, const WorldShape& b,
                                const std::array<SimplexVertex, 4>& tetra) noexcept
{
    std::vector<SimplexVertex> verts(tetra.begin(), tetra.end());
    std::vector<Face> faces;
    faces.push_back(makeFace(verts, 0, 1, 2));
    faces.push_back(makeFace(verts, 0, 1, 3));
    faces.push_back(makeFace(verts, 0, 2, 3));
    faces.push_back(makeFace(verts, 1, 2, 3));

    Face closest = faces.front();
    for (int iter = 0; iter < kMaxEpaIterations; ++iter)
    {
        std::size_t ci = 0;
        for (std::size_t i = 1; i < faces.size(); ++i)
        {
            if (faces[i].dist < faces[ci].dist)
            {
                ci = i;
            }
        }
        closest = faces[ci];

        const SimplexVertex p = minkowskiSupport(a, b, closest.normal);
        if (dot(p.v, closest.normal) - closest.dist < kEpaEps)
        {
            break; // converged on the closest face
        }

        const int pi = static_cast<int>(verts.size());
        verts.push_back(p);

        std::vector<std::pair<int, int>> horizon;
        std::vector<Face> kept;
        kept.reserve(faces.size());
        for (const Face& f : faces)
        {
            if (dot(f.normal, p.v - verts[static_cast<std::size_t>(f.a)].v) > kEpaEps)
            {
                addHorizonEdge(horizon, f.a, f.b);
                addHorizonEdge(horizon, f.b, f.c);
                addHorizonEdge(horizon, f.c, f.a);
            }
            else
            {
                kept.push_back(f);
            }
        }
        faces = std::move(kept);
        for (const auto& edge : horizon)
        {
            faces.push_back(makeFace(verts, edge.first, edge.second, pi));
        }
        if (faces.empty())
        {
            break;
        }
    }

    // Barycentric of the origin's projection onto the closest face → witnesses.
    float u = 0.0f;
    float v = 0.0f;
    float w = 0.0f;
    closestTriangle(verts[static_cast<std::size_t>(closest.a)].v,
                    verts[static_cast<std::size_t>(closest.b)].v,
                    verts[static_cast<std::size_t>(closest.c)].v, u, v, w);
    const SimplexVertex& va = verts[static_cast<std::size_t>(closest.a)];
    const SimplexVertex& vb = verts[static_cast<std::size_t>(closest.b)];
    const SimplexVertex& vc = verts[static_cast<std::size_t>(closest.c)];

    ConvexContact contact;
    contact.colliding = true;
    contact.depth = closest.dist;
    // closest.normal is the Minkowski-difference outward normal; A separates from B
    // by moving along -normal, i.e. the b -> a push direction is -normal.
    contact.normal = closest.normal * -1.0f;
    contact.pointA = va.sa * u + vb.sa * v + vc.sa * w;
    contact.pointB = va.sb * u + vb.sb * v + vc.sb * w;
    return contact;
}

// Robust fallback for the degenerate overlap cases EPA can't initialise cleanly
// (e.g. perfectly axis-aligned symmetric boxes, where the origin lands on a simplex
// boundary). Minimum-translation-vector by sampling the world axes — exact for
// axis-aligned boxes; the clean EPA path covers oriented/general convex overlaps.
[[nodiscard]] ConvexContact directionalMtv(const WorldShape& a, const WorldShape& b) noexcept
{
    float bestDepth = std::numeric_limits<float>::max();
    Vec3 bestNormal{0.0f, 1.0f, 0.0f};
    for (const Vec3& d : {Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f}})
    {
        const float maxA = dot(supportPoint(a, d), d);
        const float minA = dot(supportPoint(a, d * -1.0f), d);
        const float maxB = dot(supportPoint(b, d), d);
        const float minB = dot(supportPoint(b, d * -1.0f), d);
        const float penPos = maxB - minA; // overlap resolved by pushing A toward +d
        const float penNeg = maxA - minB; // ... toward -d
        const float pen = penPos < penNeg ? penPos : penNeg;
        if (pen < bestDepth)
        {
            bestDepth = pen;
            bestNormal = penPos < penNeg ? d : d * -1.0f; // b -> a push direction
        }
    }
    ConvexContact contact;
    contact.colliding = true;
    contact.depth = bestDepth > 0.0f ? bestDepth : 0.0f;
    contact.normal = bestNormal;
    contact.pointA = supportPoint(a, bestNormal * -1.0f);
    contact.pointB = supportPoint(b, bestNormal);
    return contact;
}

// Overlap detected but the GJK simplex has fewer than 4 vertices (origin on a
// lower-dimensional feature). Grow it to a non-degenerate tetrahedron enclosing the
// origin by adding support points off the current feature, then run EPA.
[[nodiscard]] ConvexContact completeTetraEpa(const WorldShape& a, const WorldShape& b,
                                             std::array<SimplexVertex, 4> s, int count) noexcept
{
    if (count == 1)
    {
        for (const Vec3& axis : {Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                 Vec3{0.0f, 0.0f, 1.0f}, Vec3{-1.0f, 0.0f, 0.0f}})
        {
            const SimplexVertex p = minkowskiSupport(a, b, axis);
            if ((p.v - s[0].v).magnitudeSquared() > kEpaEps)
            {
                s[1] = p;
                count = 2;
                break;
            }
        }
    }
    if (count == 2)
    {
        const Vec3 ab = s[1].v - s[0].v;
        const Vec3 seed =
            std::abs(ab.x()) <= std::abs(ab.y()) && std::abs(ab.x()) <= std::abs(ab.z())
                ? Vec3{1.0f, 0.0f, 0.0f}
            : std::abs(ab.y()) <= std::abs(ab.z()) ? Vec3{0.0f, 1.0f, 0.0f}
                                                   : Vec3{0.0f, 0.0f, 1.0f};
        const Vec3 dir = cross(ab, seed);
        SimplexVertex p = minkowskiSupport(a, b, dir);
        if (cross(p.v - s[0].v, ab).magnitudeSquared() < kEpaEps)
        {
            p = minkowskiSupport(a, b, dir * -1.0f);
        }
        s[2] = p;
        count = 3;
    }
    if (count == 3)
    {
        const Vec3 n = cross(s[1].v - s[0].v, s[2].v - s[0].v);
        SimplexVertex p = minkowskiSupport(a, b, n);
        if (dot(p.v - s[0].v, n) <= kEpaEps)
        {
            p = minkowskiSupport(a, b, n * -1.0f);
        }
        s[3] = p;
        count = 4;
    }
    if (count < 4)
    {
        return ConvexContact{.colliding = true};
    }
    return epa(a, b, s);
}

} // namespace

ConvexContact gjkEpaContact(const WorldShape& a, const WorldShape& b) noexcept
{
    std::array<SimplexVertex, 4> simplex{};
    int count = 1;
    simplex[0] = minkowskiSupport(a, b, Vec3{1.0f, 0.0f, 0.0f});

    Closest closest = closestOnSimplex(simplex, count);

    // Tightest lower bound on the separation seen so far, plus the direction that achieved
    // it. The simplex-closest magnitude |v| is an *upper* bound on the gap; the support
    // projection dot(w,v)/|v| is a *lower* bound (the separating plane through the closest
    // feature). They coincide on clean convergence, but when the support tie-breaks on a flat
    // face/edge the loop stalls with |v| over-estimating while dot(w,v)/|v| already holds the
    // exact gap — so we report the projection rather than the stale simplex magnitude.
    float bestGap = 0.0f;
    Vec3 bestNormal{0.0f, 1.0f, 0.0f};
    for (int iter = 0; iter < kMaxGjkIterations; ++iter)
    {
        const Vec3 dir = closest.point * -1.0f; // toward the origin
        if (dot(dir, dir) < kGjkEps)
        {
            // Origin on the simplex (touching / shallow overlap). Grow to an
            // enclosing tetrahedron (if degenerate) and run EPA; if EPA can't get a
            // clean depth (degenerate symmetric case), use the directional fallback.
            const ConvexContact r = completeTetraEpa(a, b, simplex, count);
            return r.depth < kEpaEps ? directionalMtv(a, b) : r;
        }

        const SimplexVertex p = minkowskiSupport(a, b, dir);

        // Update the lower-bound separation from this support's projection onto the closest
        // feature's direction. dot(p.v, closest)/|closest| is the signed distance of the
        // separating plane through the closest feature; its max over iterations is the gap.
        const float cmag = closest.point.magnitude();
        if (cmag > 1e-12f)
        {
            const float lb = dot(p.v, closest.point) / cmag;
            if (lb > bestGap)
            {
                bestGap = lb;
                bestNormal = closest.point * (1.0f / cmag);
            }
        }

        // Duplicate support → the closest feature can't be refined further from this
        // direction (the simplex has stalled, e.g. a flat-face/edge support tie-break).
        // Stop and report the lower-bound gap rather than spinning to the iteration cap.
        bool duplicate = false;
        for (int i = 0; i < count; ++i)
        {
            const Vec3 d = p.v - simplex[static_cast<std::size_t>(i)].v;
            if (dot(d, d) <= kGjkRel * (cmag * cmag + 1e-20f))
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
        {
            break;
        }

        // No progress past the current closest feature toward the origin → converged
        // (separated). For overlap the support reaches past the origin, so progress
        // stays positive until the simplex grows to a tetrahedron enclosing it. The
        // tolerance is *relative* to the closest feature's squared distance: with large
        // shapes (e.g. a big floor/wall) the dot products are large and their FP noise
        // would never fall under a fixed absolute epsilon, so GJK would keep iterating,
        // over-grow the simplex, and a degenerate tetrahedron could falsely enclose the
        // origin — reporting a separated pair as colliding.
        const float progress = dot(p.v, dir) - dot(closest.point, dir);
        if (progress < kGjkEps + kGjkRel * dot(closest.point, closest.point))
        {
            break;
        }

        simplex[count++] = p;
        closest = closestOnSimplex(simplex, count);
        if (closest.containsOrigin)
        {
            const ConvexContact r = epa(a, b, simplex); // enclosing tetra → depth/normal
            return r.depth < kEpaEps ? directionalMtv(a, b) : r;
        }

        // Compact the simplex to the contributing vertices (non-zero weight).
        std::array<SimplexVertex, 4> reduced{};
        std::array<float, 4> reducedWeight{};
        int newCount = 0;
        for (int i = 0; i < count; ++i)
        {
            if (closest.weight[static_cast<std::size_t>(i)] > 0.0f)
            {
                reduced[static_cast<std::size_t>(newCount)] = simplex[static_cast<std::size_t>(i)];
                reducedWeight[static_cast<std::size_t>(newCount)] =
                    closest.weight[static_cast<std::size_t>(i)];
                ++newCount;
            }
        }
        simplex = reduced;
        count = newCount;
        closest.weight = reducedWeight;
    }

    // Separated: recover witnesses from the closest feature's barycentric weights.
    Vec3 pointA{};
    Vec3 pointB{};
    for (int i = 0; i < count; ++i)
    {
        const float wgt = closest.weight[static_cast<std::size_t>(i)];
        pointA += simplex[static_cast<std::size_t>(i)].sa * wgt;
        pointB += simplex[static_cast<std::size_t>(i)].sb * wgt;
    }
    const Vec3 delta = pointA - pointB;
    const float dist = delta.magnitude();
    ConvexContact contact;
    contact.colliding = false;
    contact.pointA = pointA;
    contact.pointB = pointB;
    // Prefer the lower-bound projection (exact at the closest feature) over the simplex
    // magnitude, which over-estimates when the support stalled on a flat face/edge. They
    // agree on clean convergence; the projection wins only on the degenerate tie-break.
    if (bestGap > 0.0f)
    {
        contact.depth = bestGap;
        contact.normal = bestNormal;
    }
    else
    {
        contact.depth = dist;
        contact.normal = dist > 1e-6f ? delta * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    }
    return contact;
}

} // namespace fire_engine
