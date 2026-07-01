#include <fire_engine/collision/narrow_phase.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

#include <fire_engine/collision/geometry.hpp>
#include <fire_engine/collision/gjk_epa.hpp>

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

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldSphere& a, const WorldSphere& b,
                                                         float margin) noexcept
{
    const Vec3 d = a.center - b.center; // b -> a
    const float dist = d.magnitude();
    const float rsum = a.radius + b.radius;
    if (dist >= rsum + margin)
    {
        return std::nullopt;
    }
    const Vec3 n = dist > kGeomEps ? d * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    const Vec3 onA = a.center - n * a.radius;
    const Vec3 onB = b.center + n * b.radius;
    return onePoint(n, (onA + onB) * 0.5f, rsum - dist);
}

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldSphere& s, const WorldBox& box,
                                                         float margin) noexcept
{
    const Vec3 closest = closestPointOnObb(s.center, box);
    Vec3 d = s.center - closest; // box surface -> sphere centre
    const float dist2 = d.magnitudeSquared();
    const float reach = s.radius + margin;
    if (dist2 >= reach * reach)
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

[[nodiscard]] std::optional<ContactManifold>
collidePair(const WorldSphere& s, const WorldCapsule& cap, float margin) noexcept
{
    const Vec3 closest = closestPointOnSegment(s.center, cap.p0, cap.p1);
    const Vec3 d = s.center - closest;
    const float dist = d.magnitude();
    const float rsum = s.radius + cap.radius;
    if (dist >= rsum + margin)
    {
        return std::nullopt;
    }
    const Vec3 n = dist > kGeomEps ? d * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    const Vec3 onS = s.center - n * s.radius;
    const Vec3 onC = closest + n * cap.radius;
    return onePoint(n, (onS + onC) * 0.5f, rsum - dist);
}

[[nodiscard]] std::optional<ContactManifold>
collidePair(const WorldCapsule& a, const WorldCapsule& b, float margin) noexcept
{
    const SegmentClosest cl = closestPointsBetweenSegments(a.p0, a.p1, b.p0, b.p1);
    const Vec3 d = cl.c1 - cl.c2; // b -> a
    const float dist = d.magnitude();
    const float rsum = a.radius + b.radius;
    if (dist >= rsum + margin)
    {
        return std::nullopt;
    }
    const Vec3 n = dist > kGeomEps ? d * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    const Vec3 onA = cl.c1 - n * a.radius;
    const Vec3 onB = cl.c2 + n * b.radius;
    return onePoint(n, (onA + onB) * 0.5f, rsum - dist);
}

[[nodiscard]] std::optional<ContactManifold>
collidePair(const WorldBox& box, const WorldCapsule& cap, float margin) noexcept
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
    if (dist >= cap.radius + margin)
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

[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldBox& A, const WorldBox& B,
                                                         float margin) noexcept
{
    const BoxFrame a = frameOf(A);
    const BoxFrame b = frameOf(B);
    const Vec3 t = b.c - a.c; // a -> b

    // Best face axis on each box (max separation = least penetration) + best edge.
    // We scan every axis (no early-out on a separating axis) so the maximum
    // separation is known: it drives both the margin reject and the speculative
    // (separated-within-margin) contact below.
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
            if (s > bestEdgeSep)
            {
                bestEdgeSep = s;
                bestEdgeAxis = L;
            }
        }
    }

    // Separated by more than the margin → no contact. When the best separation is
    // in (0, margin] the boxes are apart but approaching close enough to matter: the
    // face/edge paths below yield a single point with negative penetration (a gap)
    // via the `-bestSep` depth, exactly the speculative contact the solver wants.
    if (std::max(bestFaceSep, bestEdgeSep) > margin)
    {
        return std::nullopt;
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
[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldBox& box, const WorldSphere& s,
                                                         float margin) noexcept
{
    auto m = collidePair(s, box, margin);
    return m ? std::optional<ContactManifold>{flipped(*m)} : std::nullopt;
}
[[nodiscard]] std::optional<ContactManifold>
collidePair(const WorldCapsule& cap, const WorldSphere& s, float margin) noexcept
{
    auto m = collidePair(s, cap, margin);
    return m ? std::optional<ContactManifold>{flipped(*m)} : std::nullopt;
}
[[nodiscard]] std::optional<ContactManifold> collidePair(const WorldCapsule& cap,
                                                         const WorldBox& box, float margin) noexcept
{
    auto m = collidePair(box, cap, margin);
    return m ? std::optional<ContactManifold>{flipped(*m)} : std::nullopt;
}

// ----- Convex (GJK/EPA) contact: any pair where at least one shape is convex -----

// A world-space polygon face: outward normal + ordered boundary loop.
struct PolyFace
{
    Vec3 normal{};
    std::vector<Vec3> verts;
};

[[nodiscard]] std::vector<PolyFace> boxFaces(const WorldBox& box)
{
    const std::array<Vec3, 3> axis{box.orientation.rotate({1.0f, 0.0f, 0.0f}),
                                   box.orientation.rotate({0.0f, 1.0f, 0.0f}),
                                   box.orientation.rotate({0.0f, 0.0f, 1.0f})};
    const std::array<float, 3> he{box.halfExtents.x(), box.halfExtents.y(), box.halfExtents.z()};
    std::vector<PolyFace> faces;
    faces.reserve(6);
    for (int ai = 0; ai < 3; ++ai)
    {
        const int u = (ai + 1) % 3;
        const int v = (ai + 2) % 3;
        for (float sgn : {1.0f, -1.0f})
        {
            const Vec3 base = box.center + axis[ai] * (sgn * he[ai]);
            const Vec3 du = axis[u] * he[u];
            const Vec3 dv = axis[v] * he[v];
            PolyFace f;
            f.normal = axis[ai] * sgn;
            f.verts = {base - du - dv, base + du - dv, base + du + dv, base - du + dv};
            faces.push_back(std::move(f));
        }
    }
    return faces;
}

[[nodiscard]] std::vector<PolyFace> convexFaces(const WorldConvex& hull)
{
    Vec3 centroid{};
    for (const Vec3& v : hull.vertices)
    {
        centroid += v;
    }
    if (!hull.vertices.empty())
    {
        centroid = centroid * (1.0f / static_cast<float>(hull.vertices.size()));
    }

    std::vector<PolyFace> faces;
    faces.reserve(hull.faces.size());
    for (const ConvexFace& cf : hull.faces)
    {
        if (cf.loop.size() < 3)
        {
            continue;
        }
        PolyFace f;
        f.verts.reserve(cf.loop.size());
        Vec3 faceCenter{};
        for (int idx : cf.loop)
        {
            const Vec3& v = hull.vertices[static_cast<std::size_t>(idx)];
            f.verts.push_back(v);
            faceCenter += v;
        }
        faceCenter = faceCenter * (1.0f / static_cast<float>(cf.loop.size()));
        Vec3 n =
            Vec3::normalise(Vec3::crossProduct(f.verts[1] - f.verts[0], f.verts[2] - f.verts[0]));
        if (dot(n, faceCenter - centroid) < 0.0f)
        {
            n = n * -1.0f; // point outward from the hull
        }
        f.normal = n;
        faces.push_back(std::move(f));
    }
    return faces;
}

[[nodiscard]] std::vector<PolyFace> polytopeFaces(const WorldShape& s)
{
    if (const auto* box = std::get_if<WorldBox>(&s))
    {
        return boxFaces(*box);
    }
    if (const auto* hull = std::get_if<WorldConvex>(&s))
    {
        return convexFaces(*hull);
    }
    return {};
}

[[nodiscard]] bool isPolytope(const WorldShape& s) noexcept
{
    return std::holds_alternative<WorldBox>(s) || std::holds_alternative<WorldConvex>(s);
}

[[nodiscard]] int bestAlignedFace(std::span<const PolyFace> faces, const Vec3& dir) noexcept
{
    int best = 0;
    float bestDot = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < faces.size(); ++i)
    {
        const float d = dot(faces[i].normal, dir);
        if (d > bestDot)
        {
            bestDot = d;
            best = static_cast<int>(i);
        }
    }
    return best;
}

// Multi-point manifold for a polytope-polytope contact: clip the incident face
// against the reference face's side planes (reference = whichever face is most
// aligned with the contact normal `n`, which points b -> a). Mirrors the box/box
// face path, generalised to arbitrary convex faces.
[[nodiscard]] ContactManifold clipFaceManifold(std::span<const PolyFace> facesA,
                                               std::span<const PolyFace> facesB, const Vec3& n)
{
    // A's contact face points toward B (≈ -n); B's toward A (≈ +n).
    const int fa = bestAlignedFace(facesA, n * -1.0f);
    const int fb = bestAlignedFace(facesB, n);
    const float alignA =
        facesA.empty() ? -1.0f : dot(facesA[static_cast<std::size_t>(fa)].normal, n * -1.0f);
    const float alignB =
        facesB.empty() ? -1.0f : dot(facesB[static_cast<std::size_t>(fb)].normal, n);

    const std::span<const PolyFace>* incFaces = nullptr;
    const PolyFace* ref = nullptr;
    if (alignA >= alignB)
    {
        ref = &facesA[static_cast<std::size_t>(fa)];
        incFaces = &facesB;
    }
    else
    {
        ref = &facesB[static_cast<std::size_t>(fb)];
        incFaces = &facesA;
    }
    const Vec3 refNormal = ref->normal;
    const PolyFace& incident =
        (*incFaces)[static_cast<std::size_t>(bestAlignedFace(*incFaces, refNormal * -1.0f))];

    // Clip the incident loop against the reference face's edge side-planes.
    std::vector<Vec3> poly = incident.verts;
    Vec3 refCentroid{};
    for (const Vec3& v : ref->verts)
    {
        refCentroid += v;
    }
    refCentroid = refCentroid * (1.0f / static_cast<float>(ref->verts.size()));
    for (std::size_t i = 0; i < ref->verts.size(); ++i)
    {
        const Vec3& v0 = ref->verts[i];
        const Vec3& v1 = ref->verts[(i + 1) % ref->verts.size()];
        Vec3 sideN = Vec3::crossProduct(v1 - v0, refNormal);
        if (dot(sideN, refCentroid - v0) > 0.0f)
        {
            sideN = sideN * -1.0f; // point outward from the face
        }
        clipAgainstPlane(poly, sideN, dot(sideN, v0));
    }

    const float refFaceD = dot(refNormal, ref->verts[0]);
    ContactManifold m;
    m.normal = n; // b -> a
    for (const Vec3& p : poly)
    {
        const float depth = refFaceD - dot(refNormal, p);
        // Keep points within kContactOffset *above* the reference face too (depth down to
        // -kContactOffset), recording the true signed gap as a negative penetration. These
        // near-contact points widen a near-degenerate support (e.g. a tetra landing almost
        // flat) into a small patch; the solver treats the gap points as speculative
        // (brake-only), so they share the friction load without lifting a resting body.
        if (depth < -kContactOffset)
        {
            continue; // beyond the offset above the reference face, not in contact
        }
        if (m.count >= kMaxManifoldPoints)
        {
            int shallow = 0;
            for (int i = 1; i < m.count; ++i)
            {
                if (m.points[static_cast<std::size_t>(i)].penetration <
                    m.points[static_cast<std::size_t>(shallow)].penetration)
                {
                    shallow = i;
                }
            }
            if (depth > m.points[static_cast<std::size_t>(shallow)].penetration)
            {
                m.points[static_cast<std::size_t>(shallow)] = {p, depth};
            }
            continue;
        }
        m.points[static_cast<std::size_t>(m.count++)] = {p, depth};
    }
    return m;
}

// Generic convex-vs-anything contact (any pair where at least one shape is a
// WorldConvex). GJK/EPA over support functions handles every such pair uniformly,
// then a polytope-face clip or single witness point builds the manifold. Normal
// points b -> a; honours the speculative margin (separated within margin → a
// negative-penetration gap contact).
[[nodiscard]] std::optional<ContactManifold> collideConvex(const WorldShape& a, const WorldShape& b,
                                                           float margin)
{
    const ConvexContact c = gjkEpaContact(a, b);
    if (!c.colliding)
    {
        if (c.depth > margin)
        {
            return std::nullopt; // separated beyond the speculative margin
        }
        return onePoint(c.normal, (c.pointA + c.pointB) * 0.5f, -c.depth); // gap contact
    }
    if (isPolytope(a) && isPolytope(b))
    {
        ContactManifold m = clipFaceManifold(polytopeFaces(a), polytopeFaces(b), c.normal);
        if (m.count > 0)
        {
            return m;
        }
    }
    // Curved-involving (or a degenerate clip) → a single point at the contact.
    return onePoint(c.normal, (c.pointA + c.pointB) * 0.5f, c.depth);
}

} // namespace

std::optional<ContactManifold> NarrowPhase::collide(const WorldShape& a, const WorldShape& b,
                                                    float speculativeMargin) const noexcept
{
    // 2-arg visit. Primitive pairs go to the analytic collidePair overloads; any
    // pair involving a convex hull goes through the generic GJK/EPA collideConvex.
    // The result normal points b -> a.
    return std::visit(
        [&a, &b, speculativeMargin](const auto& sa,
                                    const auto& sb) -> std::optional<ContactManifold>
        {
            using A = std::decay_t<decltype(sa)>;
            using B = std::decay_t<decltype(sb)>;
            if constexpr (std::is_same_v<A, WorldConvex> || std::is_same_v<B, WorldConvex>)
            {
                return collideConvex(a, b, speculativeMargin);
            }
            else
            {
                return collidePair(sa, sb, speculativeMargin);
            }
        },
        a, b);
}

} // namespace fire_engine
