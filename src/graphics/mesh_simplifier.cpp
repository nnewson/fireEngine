#include <fire_engine/graphics/mesh_simplifier.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace fire_engine
{

namespace
{

// A point in the attribute space R⁵ = (x, y, z, weighted-u, weighted-v).
using Vec5 = std::array<double, 5>;

[[nodiscard]] double dot5(const Vec5& x, const Vec5& y) noexcept
{
    double s = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        s += x[i] * y[i];
    }
    return s;
}

// Garland–Heckbert attribute error quadric over R⁵. Q(V) is the squared distance from V to a
// triangle's 2-plane in R⁵, so it penalises geometric deviation AND UV stretch in one metric — a
// collapse that moves position off the surface OR distorts the texture parameterisation both cost.
// When every UV is equal (the position-only case) it reduces exactly to the geometric plane
// quadric. Stored as the full quadratic form error(V) = VᵀAV + 2b·V + c.
struct Quadric
{
    std::array<double, 25> a{}; // 5×5, row-major (symmetric)
    std::array<double, 5> b{};
    double c{0.0};

    // Squared distance to the 2-plane through V1 spanned by (V2-V1, V3-V1).
    static Quadric fromTriangle(const Vec5& v1, const Vec5& v2, const Vec5& v3,
                                double weight) noexcept
    {
        auto sub = [](const Vec5& x, const Vec5& y)
        {
            Vec5 r{};
            for (int i = 0; i < 5; ++i)
            {
                r[i] = x[i] - y[i];
            }
            return r;
        };
        auto normalise = [](Vec5 v)
        {
            const double len = std::sqrt(dot5(v, v));
            if (len > 1e-12)
            {
                for (double& e : v)
                {
                    e /= len;
                }
            }
            else
            {
                v = {};
            }
            return v;
        };

        const Vec5 e1 = normalise(sub(v2, v1));
        const Vec5 d2 = sub(v3, v1);
        const double proj = dot5(d2, e1);
        Vec5 e2{};
        for (int i = 0; i < 5; ++i)
        {
            e2[i] = d2[i] - proj * e1[i];
        }
        e2 = normalise(e2);

        Quadric q;
        // A = weight · (I − e1·e1ᵀ − e2·e2ᵀ): projection onto the plane's orthogonal complement.
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                q.a[(i * 5) + j] = weight * ((i == j ? 1.0 : 0.0) - e1[i] * e1[j] - e2[i] * e2[j]);
            }
        }
        for (int i = 0; i < 5; ++i) // b = −A·V1
        {
            double s = 0.0;
            for (int j = 0; j < 5; ++j)
            {
                s += q.a[(i * 5) + j] * v1[j];
            }
            q.b[i] = -s;
        }
        q.c = -dot5(v1, q.b); // c = V1ᵀ·A·V1 = −b·V1
        return q;
    }

    // Position-only plane quadric (UV components zero) — for boundary preservation, which is purely
    // geometric.
    static Quadric fromPositionPlane(double nx, double ny, double nz, double d,
                                     double weight) noexcept
    {
        const Vec5 n{nx, ny, nz, 0.0, 0.0};
        Quadric q;
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                q.a[(i * 5) + j] = weight * n[i] * n[j];
            }
            q.b[i] = weight * d * n[i];
        }
        q.c = weight * d * d;
        return q;
    }

    Quadric& operator+=(const Quadric& o) noexcept
    {
        for (int i = 0; i < 25; ++i)
        {
            a[i] += o.a[i];
        }
        for (int i = 0; i < 5; ++i)
        {
            b[i] += o.b[i];
        }
        c += o.c;
        return *this;
    }

    [[nodiscard]] double eval(const Vec5& v) const noexcept
    {
        double s = c;
        for (int i = 0; i < 5; ++i)
        {
            double av = 0.0;
            for (int j = 0; j < 5; ++j)
            {
                av += a[(i * 5) + j] * v[j];
            }
            s += v[i] * av + 2.0 * b[i] * v[i];
        }
        return s;
    }
};

[[nodiscard]] Vec3 triangleNormal(const Vec3& a, const Vec3& b, const Vec3& c) noexcept
{
    const Vec3 n = Vec3::crossProduct(b - a, c - a);
    const float len = n.magnitude();
    return len > 1e-12f ? n * (1.0f / len) : Vec3{};
}

// Perpendicular distance from p to the plane of triangle abc (0 for a degenerate face). We use
// point-to-*plane*, not point-to-triangle: a collapse retires the two triangles on its edge,
// leaving a small in-plane topological gap, and on a FLAT region the removed vertex sits at the rim
// of that gap — point-to-triangle would read the in-plane gap distance as "deviation" and break the
// flats-stay-zero property the whole metric relies on. The plane extends across the gap, so only
// genuine off-surface curvature contributes. (The plane's under-read at a sharp silhouette/boundary
// is deliberately left to the separate normal/shading channel.)
[[nodiscard]] float pointPlaneDistance(const Vec3& p, const Vec3& a, const Vec3& b,
                                       const Vec3& c) noexcept
{
    const Vec3 n = triangleNormal(a, b, c);
    return std::abs(Vec3::dotProduct(n, p - a));
}

// Distance from point p to triangle abc, closest point clamped to the triangle (Ericson, Real-Time
// Collision Detection §5.1.5). Used only to *select* the nearest one-ring face for the deviation
// channels — it reads 0 for a face p sits in even on a flat region where every plane distance is 0,
// so the nearest face is the one whose plane the attribute interpolation should extrapolate on.
[[nodiscard]] float pointTriangleDistance(const Vec3& p, const Vec3& a, const Vec3& b,
                                          const Vec3& c) noexcept
{
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const float d1 = Vec3::dotProduct(ab, ap);
    const float d2 = Vec3::dotProduct(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        return ap.magnitude();
    }
    const Vec3 bp = p - b;
    const float d3 = Vec3::dotProduct(ab, bp);
    const float d4 = Vec3::dotProduct(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        return bp.magnitude();
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        return (p - (a + ab * (d1 / (d1 - d3)))).magnitude();
    }
    const Vec3 cp = p - c;
    const float d5 = Vec3::dotProduct(ab, cp);
    const float d6 = Vec3::dotProduct(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        return cp.magnitude();
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        return (p - (a + ac * (d2 / (d2 - d6)))).magnitude();
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        return (p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))))).magnitude();
    }
    const float denom = 1.0f / (va + vb + vc);
    return (p - (a + ab * (vb * denom) + ac * (vc * denom))).magnitude();
}

// Unclamped barycentric coordinates of p projected onto the plane of triangle abc (x for a, y for
// b, z for c; they sum to 1 but may fall outside [0,1] when p projects beyond the triangle).
// Unclamped on purpose: interpolating attributes with these keeps a flat, affine-UV region reading
// ~0 across the in-plane gap a collapse leaves — clamping to the triangle would measure that gap,
// exactly the failure point-to-plane fixed for the geometric channel.
[[nodiscard]] Vec3 barycentricOnPlane(const Vec3& p, const Vec3& a, const Vec3& b,
                                      const Vec3& c) noexcept
{
    const Vec3 v0 = b - a;
    const Vec3 v1 = c - a;
    const Vec3 v2 = p - a;
    const float d00 = Vec3::dotProduct(v0, v0);
    const float d01 = Vec3::dotProduct(v0, v1);
    const float d11 = Vec3::dotProduct(v1, v1);
    const float d20 = Vec3::dotProduct(v2, v0);
    const float d21 = Vec3::dotProduct(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-20f)
    {
        return Vec3{1.0f, 0.0f, 0.0f}; // degenerate face — fall back to vertex a
    }
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    return Vec3{1.0f - v - w, v, w};
}

// Angular deviation (radians) between attr[removed] and the barycentric interpolation of
// attr[a],attr[b],attr[c] at removed's spot on the covering face, renormalised (matching how the
// shader interpolates a per-vertex direction across a triangle). 0 when the field is locally
// consistent (a flat region with equal directions); the fan angle when it swings. Zero-length
// inputs — an unpopulated attribute, e.g. a mesh without tangents — read 0. Shared by the
// shading-normal and tangent VDPM channels.
[[nodiscard]] float angularInterpDeviation(const std::vector<Vec3>& attr, std::uint32_t removed,
                                           std::uint32_t a, std::uint32_t b, std::uint32_t c,
                                           const Vec3& bc) noexcept
{
    const Vec3 vi = (attr[a] * bc.x()) + (attr[b] * bc.y()) + (attr[c] * bc.z());
    const float viLen = vi.magnitude();
    const float remLen = attr[removed].magnitude();
    if (viLen <= 1e-6f || remLen <= 1e-6f)
    {
        return 0.0f;
    }
    const float cosA =
        std::clamp(Vec3::dotProduct(vi, attr[removed]) / (viLen * remLen), -1.0f, 1.0f);
    return std::acos(cosA);
}

[[nodiscard]] float wedgeDistance(const Vertex& a, const Vertex& b) noexcept
{
    const Vec2 uv0 = a.texCoord() - b.texCoord();
    const Vec2 uv1 = a.texCoord1() - b.texCoord1();
    const Vec3 normal = a.normal() - b.normal();
    const Vec4 tangent = a.tangent() - b.tangent();
    return Vec2::dotProduct(uv0, uv0) + Vec2::dotProduct(uv1, uv1) +
           0.25f * Vec3::dotProduct(normal, normal) + 0.25f * Vec4::dotProduct(tangent, tangent);
}

[[nodiscard]] std::uint64_t edgeKey(std::uint32_t a, std::uint32_t b) noexcept
{
    const std::uint64_t lo = a < b ? a : b;
    const std::uint64_t hi = a < b ? b : a;
    return (lo << 32) | hi;
}

// Exact-position weld key. glTF duplicates a vertex at every attribute seam; leaving them all split
// shatters the mesh into boundary-locked islands that can't simplify, so collapse connectivity is
// built on position-welded vertices (all wedges at one position merge into a canonical). The per-
// corner UVs are preserved separately at emit time (see emitIndices — nearest-wedge matching).
struct PosKey
{
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    bool operator==(const PosKey&) const noexcept = default;
};

struct PosKeyHash
{
    std::size_t operator()(const PosKey& k) const noexcept
    {
        std::size_t h = k.x;
        h = h * 1000003u ^ k.y;
        h = h * 1000003u ^ k.z;
        return h;
    }
};

[[nodiscard]] PosKey posKey(const Vec3& p) noexcept
{
    return PosKey{std::bit_cast<std::uint32_t>(p.x()), std::bit_cast<std::uint32_t>(p.y()),
                  std::bit_cast<std::uint32_t>(p.z())};
}

// A candidate collapse in the priority queue. `a`/`b` are the vertex roots at push time; `va`/`vb`
// their versions then — a later collapse bumps a root's version, so any stale entry is discarded on
// pop and a fresh one has already been pushed.
struct Edge
{
    double cost;
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t va;
    std::uint32_t vb;
};

struct EdgeGreater
{
    bool operator()(const Edge& lhs, const Edge& rhs) const noexcept
    {
        return lhs.cost > rhs.cost; // min-cost first
    }
};

// The greedy edge-collapse engine. Runs until the live triangle count reaches `targetTris` (or no
// valid collapse remains), recording the ordered collapse stream and emitting the surviving
// triangles as indices into the original vertex array. Subset placement: the kept vertex never
// moves.
class QemRun
{
public:
    QemRun(std::span<const Vertex> vertices, std::span<const uint32_t> indices)
        : vertices_(vertices)
    {
        const std::size_t vertexCount = vertices.size();
        pos_.resize(vertexCount);
        for (std::size_t i = 0; i < vertexCount; ++i)
        {
            pos_[i] = vertices[i].position();
        }
        remap_.resize(vertexCount);
        version_.assign(vertexCount, 0);
        weight_.assign(vertexCount, 0.0);
        quad_.resize(vertexCount);
        vertexTris_.resize(vertexCount);
        deviation_.assign(vertexCount, 0.0f);
        uvDeviation_.assign(vertexCount, 0.0f);
        normalDeviation_.assign(vertexCount, 0.0f);
        tangentDeviation_.assign(vertexCount, 0.0f);
        uv_.resize(vertexCount);
        nrm_.resize(vertexCount);
        tan_.resize(vertexCount);
        weld_.resize(vertexCount);
        canonicalWedges_.resize(vertexCount);
        for (std::uint32_t i = 0; i < vertexCount; ++i)
        {
            remap_[i] = i;
            uv_[i] = vertices[i].texCoord();
            nrm_[i] = vertices[i].normal();
            const Vec4 t = vertices[i].tangent();
            tan_[i] = Vec3{t.x(), t.y(), t.z()};
        }

        // Weld coincident positions: build the collapse topology on the canonical (first) vertex at
        // each position so attribute seams don't lock the mesh into un-collapsible islands. The
        // original per-corner vertices are kept (origTris_) so the emit can restore each corner's
        // own UV from the surviving position's wedges (canonicalWedges_) — see emitIndices.
        std::unordered_map<PosKey, std::uint32_t, PosKeyHash> weldMap;
        weldMap.reserve(vertexCount);
        for (std::uint32_t v = 0; v < vertexCount; ++v)
        {
            weld_[v] = weldMap.try_emplace(posKey(pos_[v]), v).first->second;
            canonicalWedges_[weld_[v]].push_back(v);
        }

        for (std::size_t i = 0; i + 3 <= indices.size(); i += 3)
        {
            origTris_.push_back({indices[i], indices[i + 1], indices[i + 2]});
            tris_.push_back({weld_[indices[i]], weld_[indices[i + 1]], weld_[indices[i + 2]]});
        }
        triAlive_.assign(tris_.size(), 1);
        liveTris_ = tris_.size();

        // Render-chart identity for the VIPM cross-chart collapse veto. A chart is a maximal patch
        // of faces joined across edges that are NOT render-attribute seams; a UV/normal/tangent
        // seam (the two incident faces carry different-attribute wedges for the shared corners)
        // divides charts. Union-find over ORIGINAL vertices (wedges): union the three corners of
        // each face (intra-face), then union the shared corners across every edge whose two
        // incident faces AGREE on attributes. This is attribute-aware — a benign index duplicate
        // with identical attributes (e.g. a fully-shattered mesh) unions back into one chart and
        // still simplifies, while a real seam stays a chart boundary. canonicalCharts_[c] is the
        // sorted set of charts meeting at canonical position c; the collapse loop vetoes any
        // collapse whose survivor lacks a chart the removed vertex carries, since a VIPM geomorph
        // across that missing chart would shear/pop the texture.
        chartRoot_.resize(vertexCount);
        for (std::uint32_t v = 0; v < vertexCount; ++v)
        {
            chartRoot_[v] = v;
        }
        for (const std::array<std::uint32_t, 3>& ot : origTris_)
        {
            chartUnion(ot[0], ot[1]);
            chartUnion(ot[1], ot[2]);
        }
        constexpr float kChartSeamEps = 1.0e-8f; // squared attr distance: identical vs a real split
        std::unordered_map<std::uint64_t, std::array<std::uint32_t, 2>> firstWedge;
        firstWedge.reserve(tris_.size() * 3);
        for (const std::array<std::uint32_t, 3>& ot : origTris_)
        {
            for (int e = 0; e < 3; ++e)
            {
                std::uint32_t wa = ot[e];
                std::uint32_t wb = ot[(e + 1) % 3];
                const std::uint32_t v0 = weld_[wa];
                const std::uint32_t v1 = weld_[wb];
                if (v0 == v1)
                {
                    continue;
                }
                if (v1 <
                    v0) // orient (wa, wb) to (min, max) endpoints so faces compare like-for-like
                {
                    std::swap(wa, wb);
                }
                const std::uint64_t key = edgeKey(v0, v1);
                auto [it, inserted] = firstWedge.try_emplace(key, std::array{wa, wb});
                if (!inserted &&
                    wedgeDistance(vertices_[it->second[0]], vertices_[wa]) <= kChartSeamEps &&
                    wedgeDistance(vertices_[it->second[1]], vertices_[wb]) <= kChartSeamEps)
                {
                    chartUnion(it->second[0], wa); // non-seam edge: the two faces share a chart
                    chartUnion(it->second[1], wb);
                }
            }
        }
        canonicalCharts_.resize(vertexCount);
        for (std::uint32_t v = 0; v < vertexCount; ++v)
        {
            canonicalCharts_[weld_[v]].push_back(chartFind(v));
        }
        for (std::vector<std::uint32_t>& charts : canonicalCharts_)
        {
            std::sort(charts.begin(), charts.end());
            charts.erase(std::unique(charts.begin(), charts.end()), charts.end());
        }

        Vec3 lo = pos_.empty() ? Vec3{} : pos_[0];
        Vec3 hi = lo;
        for (const Vec3& p : pos_)
        {
            lo = Vec3{std::min(lo.x(), p.x()), std::min(lo.y(), p.y()), std::min(lo.z(), p.z())};
            hi = Vec3{std::max(hi.x(), p.x()), std::max(hi.y(), p.y()), std::max(hi.z(), p.z())};
        }
        const Vec3 diag = hi - lo;
        boundsDiagSq_ = static_cast<double>(Vec3::dotProduct(diag, diag));
        // Scale UV (0..1) into the mesh's world extent so a unit of texture stretch weighs
        // comparably to a unit of geometric deviation in the shared R⁵ error metric.
        uvWeight_ = kUvWeightFactor * std::sqrt(boundsDiagSq_);

        buildQuadrics();
        buildInitialEdges();
    }

    void run(std::size_t targetTris)
    {
        // Stop collapsing once even the cheapest remaining collapse costs more than a mesh-scale
        // ceiling, so an un-simplifiable shape (a cube — every collapse folds a face) is refused
        // rather than destroyed to hit the target count. Boundary quadrics push border-wrecking
        // collapses far above this; genuine surface simplification stays well under it.
        const double ceiling = kErrorCeilingFactor * boundsDiagSq_;
        while (liveTris_ > targetTris && !queue_.empty())
        {
            const Edge e = queue_.top();
            queue_.pop();
            const std::uint32_t ra = resolve(e.a);
            const std::uint32_t rb = resolve(e.b);
            if (ra == rb)
            {
                continue; // already merged
            }
            if (e.va != version_[ra] || e.vb != version_[rb])
            {
                continue; // stale — a fresh entry exists
            }

            const Quadric sum = combined(ra, rb);
            const double errA = sum.eval(vec5(ra));
            const double errB = sum.eval(vec5(rb));
            std::uint32_t kept = errA <= errB ? ra : rb;
            std::uint32_t removed = kept == ra ? rb : ra;
            double err = errA <= errB ? errA : errB;

            if (err > ceiling)
            {
                break; // min-heap: nothing cheaper remains
            }
            // Chart veto (VIPM seam preservation): a collapse must not drop a render chart, or the
            // geomorph shears/pops the texture across it. Prefer the error-chosen direction; if it
            // loses a chart, try the reverse (e.g. force an interior vertex INTO its seam rather
            // than the seam into the interior); if both lose a chart (an edge between two different
            // seams), skip the edge entirely.
            if (crossesChart(removed, kept))
            {
                std::swap(kept, removed);
                err = errA <= errB ? errB : errA;
                if (crossesChart(removed, kept))
                {
                    continue;
                }
                // The reversed direction carries the OTHER (larger) endpoint error, which the
                // ceiling check above never saw. Re-check it: a chart-legal-but-over-budget
                // reversal must not sneak past the quality bound (and must not inflate maxError /
                // LOD selection). `continue`, not `break` — the heap is ordered by each edge's
                // *minimum* endpoint cost, so this reversed cost says nothing about the edges still
                // queued.
                if (err > ceiling)
                {
                    continue;
                }
            }
            if (wouldFlip(removed, kept))
            {
                continue; // reject a collapse that inverts an incident triangle
            }

            collapse(kept, removed, err);
        }
    }

    [[nodiscard]] std::vector<uint32_t> emitIndices() const
    {
        std::vector<uint32_t> out;
        out.reserve(liveTris_ * 3);
        for (std::size_t t = 0; t < tris_.size(); ++t)
        {
            if (triAlive_[t] == 0)
            {
                continue;
            }
            // A triangle dies when two corners resolve to the same surviving position.
            const std::uint32_t a = resolveConst(tris_[t][0]);
            const std::uint32_t b = resolveConst(tris_[t][1]);
            const std::uint32_t c = resolveConst(tris_[t][2]);
            if (a == b || b == c || a == c)
            {
                continue;
            }
            // Emit each corner's own render wedge: at the surviving position, pick the native
            // wedge whose UVs / normal / tangent are nearest this corner's original attributes. On
            // a preserved seam (a position with wedges from two charts or normals) each side keeps
            // the closest chart/shading identity; the collapse geometry is shared, the attributes
            // are not smeared across the seam.
            out.push_back(nearestWedge(a, origTris_[t][0]));
            out.push_back(nearestWedge(b, origTris_[t][1]));
            out.push_back(nearestWedge(c, origTris_[t][2]));
        }
        return out;
    }

    // The native wedge at surviving position `canonical` whose render attributes are closest to
    // original vertex `source`.
    [[nodiscard]] std::uint32_t nearestWedge(std::uint32_t canonical, std::uint32_t source) const
    {
        std::uint32_t best = canonical;
        float bestDist = std::numeric_limits<float>::max();
        for (const std::uint32_t w : canonicalWedges_[canonical])
        {
            const float dist = wedgeDistance(vertices_[w], vertices_[source]);
            if (dist < bestDist)
            {
                bestDist = dist;
                best = w;
            }
        }
        return best;
    }

    // Consuming accessor: moves the recorded stream out of a finished run. `&&`-qualified so it can
    // only be called on an rvalue (the callers are done with `run`); this keeps `noexcept` honest —
    // a vector move never allocates, whereas the old by-value *copy* could throw on allocation and
    // so would have terminated under `noexcept`.
    [[nodiscard]] std::vector<MeshCollapse> sequence() && noexcept
    {
        return std::move(sequence_);
    }

    [[nodiscard]] float maxError() const noexcept
    {
        return maxError_;
    }

    [[nodiscard]] std::size_t collapseCount() const noexcept
    {
        return sequence_.size();
    }

private:
    std::uint32_t resolve(std::uint32_t v) noexcept
    {
        while (remap_[v] != v)
        {
            remap_[v] = remap_[remap_[v]]; // path halving
            v = remap_[v];
        }
        return v;
    }

    [[nodiscard]] std::uint32_t resolveConst(std::uint32_t v) const noexcept
    {
        while (remap_[v] != v)
        {
            v = remap_[v];
        }
        return v;
    }

    [[nodiscard]] Quadric combined(std::uint32_t a, std::uint32_t b) const noexcept
    {
        Quadric q = quad_[a];
        q += quad_[b];
        return q;
    }

    void buildQuadrics()
    {
        std::unordered_map<std::uint64_t, int> edgeCount;
        edgeCount.reserve(tris_.size() * 3);

        for (std::size_t ti = 0; ti < tris_.size(); ++ti)
        {
            const std::array<std::uint32_t, 3>& wt = tris_[ti];     // welded (accumulation targets)
            const std::array<std::uint32_t, 3>& ot = origTris_[ti]; // original corners (attributes)
            // R⁵ quadric from the original (position + UV) corners, so it captures this triangle's
            // texture gradient, accumulated onto the position-welded canonical vertices.
            const Quadric face = Quadric::fromTriangle(vec5(ot[0]), vec5(ot[1]), vec5(ot[2]), 1.0);
            for (int i = 0; i < 3; ++i)
            {
                quad_[wt[i]] += face;
                weight_[wt[i]] += 1.0;
                vertexTris_[wt[i]].push_back(static_cast<std::uint32_t>(ti));
                ++edgeCount[edgeKey(wt[i], wt[(i + 1) % 3])];
            }
        }

        // Boundary preservation: an edge in exactly one triangle is a border. Add a heavy
        // (position- only) plane perpendicular to that triangle through the edge, so its endpoints
        // resist leaving the border. (Render-attribute seams are NOT preserved here — a quadric
        // can't separate a legal along-seam collapse from an illegal cross-chart one at any single
        // weight; the cross-chart case is vetoed topologically in the collapse loop instead. See
        // crossesChart / chartOf.)
        for (const std::array<std::uint32_t, 3>& t : tris_)
        {
            const Vec3 n = triangleNormal(pos_[t[0]], pos_[t[1]], pos_[t[2]]);
            for (int e = 0; e < 3; ++e)
            {
                const std::uint32_t v0 = t[e];
                const std::uint32_t v1 = t[(e + 1) % 3];
                if (edgeCount[edgeKey(v0, v1)] != 1)
                {
                    continue;
                }
                const Vec3 dir = pos_[v1] - pos_[v0];
                Vec3 bn = Vec3::crossProduct(dir, n);
                const float len = bn.magnitude();
                if (len <= 1e-12f)
                {
                    continue;
                }
                bn = bn * (1.0f / len);
                const double d = -static_cast<double>(Vec3::dotProduct(bn, pos_[v0]));
                const Quadric border =
                    Quadric::fromPositionPlane(bn.x(), bn.y(), bn.z(), d, kBoundaryWeight);
                quad_[v0] += border;
                quad_[v1] += border;
                // Keep weight_ consistent with the quadric so the RMS error normalises correctly
                // (otherwise a boundary collapse's error is inflated by ×√kBoundaryWeight).
                weight_[v0] += kBoundaryWeight;
                weight_[v1] += kBoundaryWeight;
            }
        }
    }

    void buildInitialEdges()
    {
        std::unordered_map<std::uint64_t, std::uint8_t> seen;
        seen.reserve(tris_.size() * 3);
        for (const auto& t : tris_)
        {
            for (int e = 0; e < 3; ++e)
            {
                const std::uint32_t a = t[e];
                const std::uint32_t b = t[(e + 1) % 3];
                if (seen.emplace(edgeKey(a, b), 1).second)
                {
                    pushEdge(a, b);
                }
            }
        }
    }

    void pushEdge(std::uint32_t a, std::uint32_t b)
    {
        if (a == b)
        {
            return;
        }
        const Quadric sum = combined(a, b);
        const double cost = std::min(sum.eval(vec5(a)), sum.eval(vec5(b)));
        queue_.push(Edge{cost, a, b, version_[a], version_[b]});
    }

    // The attribute-space point (position, weighted UV) for a canonical vertex — the UV is the
    // vertex's representative wedge value; a seam vertex's differing wedges make the accumulated
    // triangle quadrics disagree there, which raises the collapse cost and preserves the seam.
    [[nodiscard]] Vec5 vec5(std::uint32_t v) const noexcept
    {
        return Vec5{pos_[v].x(), pos_[v].y(), pos_[v].z(), uvWeight_ * uv_[v].s(),
                    uvWeight_ * uv_[v].t()};
    }

    // Would collapsing `removed` into `kept` invert any triangle that survives the move?
    [[nodiscard]] bool wouldFlip(std::uint32_t removed, std::uint32_t kept) noexcept
    {
        for (const std::uint32_t t : vertexTris_[removed])
        {
            if (triAlive_[t] == 0)
            {
                continue;
            }
            std::array<std::uint32_t, 3> v{resolve(tris_[t][0]), resolve(tris_[t][1]),
                                           resolve(tris_[t][2])};
            // Triangles on the collapsing edge die (degenerate) — they don't constrain orientation.
            if ((v[0] == removed || v[1] == removed || v[2] == removed) &&
                (v[0] == kept || v[1] == kept || v[2] == kept))
            {
                continue;
            }
            const Vec3 before = triangleNormal(pos_[v[0]], pos_[v[1]], pos_[v[2]]);
            for (auto& idx : v)
            {
                if (idx == removed)
                {
                    idx = kept;
                }
            }
            if (v[0] == v[1] || v[1] == v[2] || v[0] == v[2])
            {
                continue;
            }
            const Vec3 after = triangleNormal(pos_[v[0]], pos_[v[1]], pos_[v[2]]);
            if (Vec3::dotProduct(before, after) < 0.0f)
            {
                return true;
            }
        }
        return false;
    }

    // Union-find over original vertices (wedges) for render-chart identity — see the init.
    [[nodiscard]] std::uint32_t chartFind(std::uint32_t v) noexcept
    {
        while (chartRoot_[v] != v)
        {
            chartRoot_[v] = chartRoot_[chartRoot_[v]]; // path halving
            v = chartRoot_[v];
        }
        return v;
    }

    void chartUnion(std::uint32_t a, std::uint32_t b) noexcept
    {
        const std::uint32_t ra = chartFind(a);
        const std::uint32_t rb = chartFind(b);
        if (ra != rb)
        {
            chartRoot_[std::max(ra, rb)] = std::min(ra, rb); // deterministic root
        }
    }

    // VIPM seam veto: does collapsing `removed` into `kept` drop a render chart? True when
    // `removed` touches a chart the survivor `kept` does not, so a wedge in that chart would have
    // no same-chart survivor to geomorph toward and the texture would shear/pop. Both sets are tiny
    // (1–3 charts).
    [[nodiscard]] bool crossesChart(std::uint32_t removed, std::uint32_t kept) const noexcept
    {
        for (const std::uint32_t chart : canonicalCharts_[removed])
        {
            const std::vector<std::uint32_t>& keptCharts = canonicalCharts_[kept];
            if (std::find(keptCharts.begin(), keptCharts.end(), chart) == keptCharts.end())
            {
                return true;
            }
        }
        return false;
    }

    void collapse(std::uint32_t kept, std::uint32_t removed, double quadricCost)
    {
        // The quadric cost drives collapse *ordering*, but it's a SUM of squared distances over
        // every plane folded in, so it inflates as collapses chain and isn't a geometric deviation.
        // Divide by the accumulated plane weight (unit per face) to get a root-mean-square distance
        // — ~0 for a flat surface (in-plane moves cost nothing), a small fraction of the mesh for a
        // curved one — a meaningful world-space error to project to screen-space pixels.
        const double weight = std::max(1.0, weight_[kept] + weight_[removed]);
        const auto geomError = static_cast<float>(std::sqrt(std::max(0.0, quadricCost) / weight));
        maxError_ = std::max(maxError_, geomError);
        sequence_.push_back(MeshCollapse{kept, removed, pos_[kept], geomError});

        weight_[kept] += weight_[removed];
        quad_[kept] += quad_[removed];
        remap_[removed] = kept;
        ++version_[kept];
        ++version_[removed];

        // Move removed's incident triangles onto kept, kill any that became degenerate.
        auto& keptTris = vertexTris_[kept];
        keptTris.insert(keptTris.end(), vertexTris_[removed].begin(), vertexTris_[removed].end());
        vertexTris_[removed].clear();
        for (const std::uint32_t t : keptTris)
        {
            if (triAlive_[t] == 0)
            {
                continue;
            }
            const std::uint32_t a = resolve(tris_[t][0]);
            const std::uint32_t b = resolve(tris_[t][1]);
            const std::uint32_t c = resolve(tris_[t][2]);
            if (a == b || b == c || a == c)
            {
                triAlive_[t] = 0;
                --liveTris_;
            }
        }

        // VDPM error channels, both conservative screen-space estimates (not Hausdorff bounds):
        // compose the endpoint subtrees' radii with this step's local movement, measured against
        // kept's nearest new one-ring face. Geometric dev = point-to-plane distance (~0 on flats,
        // accumulating on curves). UV dev = max(smooth stretch, seam spread): the smooth term is
        // the removed vertex's UV vs a containing face's unclamped barycentric interpolation (0 for
        // an affine chart, non-zero where the parameterisation stretches); the seam term is the
        // spread between the removed position's welded atlas wedges (0 off a seam, up to a chart's
        // width on one — what makes a seam-crossing collapse read its true cost so VDPM refines it
        // under magnification instead of warping the texture). MeshCollapse::error (R⁵ RMS) stays
        // as-is for discrete/VIPM.
        float dev = 0.0f;
        float uvStep = 0.0f;
        float normalStep = 0.0f;
        float tangentStep = 0.0f;
        float nearestPlane = std::numeric_limits<float>::max();
        float nearestCoverTri = std::numeric_limits<float>::max();
        for (const std::uint32_t t : keptTris)
        {
            if (triAlive_[t] == 0)
            {
                continue;
            }
            const std::uint32_t a = resolve(tris_[t][0]);
            const std::uint32_t b = resolve(tris_[t][1]);
            const std::uint32_t c = resolve(tris_[t][2]);
            if (a == b || b == c || a == c)
            {
                continue;
            }
            // Skip a geometrically degenerate (collinear / zero-area) face: it has distinct indices
            // but a zero normal, so its plane and barycentric are undefined. barycentricOnPlane
            // returns {1,0,0} for it, which snaps the UV interpolation to vertex a and manufactures
            // a spurious step on an otherwise affine (uvStep==0) flat region.
            if (triangleNormal(pos_[a], pos_[b], pos_[c]).magnitudeSquared() < 0.5f)
            {
                continue;
            }
            // Geometry: distance to the nearest one-ring face plane. Point-to-plane, so a flat
            // region reads ~0 across the collapse's in-plane gap and only genuine curvature
            // accumulates.
            nearestPlane = std::min(nearestPlane,
                                    pointPlaneDistance(pos_[removed], pos_[a], pos_[b], pos_[c]));
            // Smooth-stretch UV term: interpolate on the nearest face that actually *contains*
            // removed (barycentric in [0,1]), unclamped on its plane so an affine parameterisation
            // reads exactly 0. We require containment rather than extrapolating on the nearest
            // face: a one-ring often has thin sliver faces, and unclamped barycentric FAR outside a
            // sliver overshoots wildly (measured: a helmet collapse read a UV step of ~70 in a 0..1
            // space). The seam/gap error a containment requirement would miss is recovered by the
            // bounded wedge-spread term after the loop, so we lose nothing and stay
            // well-conditioned.
            const float triDist = pointTriangleDistance(pos_[removed], pos_[a], pos_[b], pos_[c]);
            if (triDist >= nearestCoverTri)
            {
                continue;
            }
            const Vec3 bc = barycentricOnPlane(pos_[removed], pos_[a], pos_[b], pos_[c]);
            constexpr float kInsideEps = 0.01f;
            if (bc.x() < -kInsideEps || bc.y() < -kInsideEps || bc.z() < -kInsideEps)
            {
                continue; // removed is outside this face — don't extrapolate here
            }
            nearestCoverTri = triDist;
            const float ui = (uv_[a].s() * bc.x()) + (uv_[b].s() * bc.y()) + (uv_[c].s() * bc.z());
            const float vi = (uv_[a].t() * bc.x()) + (uv_[b].t() * bc.y()) + (uv_[c].t() * bc.z());
            const float du = uv_[removed].s() - ui;
            const float dv = uv_[removed].t() - vi;
            uvStep = std::sqrt((du * du) + (dv * dv));

            // Shading channels: the same covering face interpolates its three vertex normals — and
            // tangents — to removed's spot; the angle to removed's own direction is the shading
            // error the collapse introduces. A flat region with a consistent frame reads 0; a
            // smooth-shaded curve whose normals/tangents fan reads the fan angle even where the
            // geometry stays coplanar. The tangent frame drives normal-map sampling, so it is a
            // distinct error source from the shading normal (and reads 0 on meshes without
            // tangents).
            normalStep = angularInterpDeviation(nrm_, removed, a, b, c, bc);
            tangentStep = angularInterpDeviation(tan_, removed, a, b, c, bc);
        }
        // Seam/wedge UV term (bounded, interpolation-free): the render restores each corner to its
        // own wedge at emit, so collapsing a position welded from several UV-seam wedges loses up
        // to the spread between those wedges — invisible to the single-representative smooth term
        // above, and the DOMINANT VDPM under-refinement on atlased meshes (a seam-crossing collapse
        // reads "cheap" and warps the texture over a few triangles at full magnification). The max
        // pairwise wedge-UV distance is exactly that lost spread, capped at the atlas's own scale
        // (no extrapolation), so a seam collapse reports its true cost and VDPM refines it once it
        // projects large; a non-seam vertex has one wedge and contributes nothing.
        const std::vector<std::uint32_t>& wedges = canonicalWedges_[removed];
        for (std::size_t i = 0; i < wedges.size(); ++i)
        {
            for (std::size_t j = i + 1; j < wedges.size(); ++j)
            {
                const float du = uv_[wedges[i]].s() - uv_[wedges[j]].s();
                const float dv = uv_[wedges[i]].t() - uv_[wedges[j]].t();
                uvStep = std::max(uvStep, std::sqrt((du * du) + (dv * dv)));
            }
        }
        if (nearestPlane != std::numeric_limits<float>::max())
        {
            dev = nearestPlane;
        }
        deviation_[kept] = std::max(deviation_[kept], deviation_[removed]) + dev;
        // UV accumulates by MAX, not max+add: unlike the geometric deviation (a spatial envelope
        // that genuinely compounds up the tree, so the conservative bound is the running sum), a UV
        // error is a screen-space texture discontinuity — what the eye sees is the single WORST
        // jump in the region, not the sum of every collapse's stretch. Summing per-wedge seam steps
        // (~1.0 each on an atlas) would blow the radius to 100+ in a 0..1 UV space and swamp the
        // budget; the max keeps it in UV's natural range, cleanly separates a seam (~1) from smooth
        // stretch (~0.01), and stays monotone (kept >= removed) for the VDPM front. So a seam
        // region stays refined until its jump is sub-pixel while flat charts coarsen freely.
        uvDeviation_[kept] = std::max({uvDeviation_[kept], uvDeviation_[removed], uvStep});
        normalDeviation_[kept] =
            std::max(normalDeviation_[kept], normalDeviation_[removed]) + normalStep;
        tangentDeviation_[kept] =
            std::max(tangentDeviation_[kept], tangentDeviation_[removed]) + tangentStep;
        sequence_.back().deviationRadius = deviation_[kept];
        sequence_.back().uvDeviationRadius = uvDeviation_[kept];
        sequence_.back().normalDeviationRadius = normalDeviation_[kept];
        sequence_.back().tangentDeviationRadius = tangentDeviation_[kept];

        // Re-cost every edge from kept to its current neighbours.
        std::unordered_map<std::uint32_t, std::uint8_t> neighbours;
        for (const std::uint32_t t : keptTris)
        {
            if (triAlive_[t] == 0)
            {
                continue;
            }
            for (int i = 0; i < 3; ++i)
            {
                const std::uint32_t r = resolve(tris_[t][i]);
                if (r != kept)
                {
                    neighbours.emplace(r, 1);
                }
            }
        }
        for (const auto& entry : neighbours)
        {
            pushEdge(kept, entry.first);
        }
    }

    static constexpr double kBoundaryWeight = 1000.0;
    // Collapse-cost ceiling as a fraction of the mesh's squared bounding-box diagonal. Deliberately
    // generous: coarse LODs legitimately introduce visible per-collapse error, and the runtime
    // screen-space selection is what guarantees on-screen quality (a coarse level is only drawn
    // when its error projects to sub-pixel). The ceiling only exists to refuse a genuinely
    // un-simplifiable shape — a cube, whose every collapse is boundary-weighted ×kBoundaryWeight,
    // far above this.
    static constexpr double kErrorCeilingFactor = 40.0;
    // UV weight as a fraction of the mesh's bounding-box diagonal: how hard the shared R⁵ metric
    // penalises texture stretch relative to geometric deviation. Higher preserves texturing more
    // aggressively (fewer UV-distorting collapses) at some cost to triangle reduction.
    static constexpr double kUvWeightFactor = 0.1;

    std::span<const Vertex> vertices_;
    std::vector<Vec3> pos_;
    std::vector<Vec2> uv_;  // per original vertex, for the wedge-preserving emit
    std::vector<Vec3> nrm_; // per original vertex normal, for the shading-deviation channel
    std::vector<Vec3> tan_; // per original vertex tangent xyz, for the tangent-deviation channel
    std::vector<std::uint32_t> weld_; // original vertex -> canonical position vertex
    std::vector<std::array<std::uint32_t, 3>> origTris_; // original (pre-weld) corners, for UV emit
    std::vector<std::vector<std::uint32_t>> canonicalWedges_; // canonical -> its native wedges
    std::vector<std::uint32_t> chartRoot_;                    // union-find over wedges -> chart id
    std::vector<std::vector<std::uint32_t>> canonicalCharts_; // canonical -> sorted chart ids on it
    std::vector<Quadric> quad_;
    std::vector<std::array<std::uint32_t, 3>> tris_;
    std::vector<std::uint8_t> triAlive_;
    std::vector<std::uint32_t> remap_;
    std::vector<std::uint32_t> version_;
    std::vector<double> weight_;         // accumulated face-plane weight per vertex (for RMS error)
    std::vector<float> deviation_;       // cumulative geometric deviation radius per vertex (VDPM)
    std::vector<float> uvDeviation_;     // cumulative UV deviation radius per vertex (VDPM)
    std::vector<float> normalDeviation_; // cumulative shading-normal deviation (radians) per vertex
    std::vector<float> tangentDeviation_; // cumulative tangent deviation (radians) per vertex
    std::vector<std::vector<std::uint32_t>> vertexTris_;
    std::priority_queue<Edge, std::vector<Edge>, EdgeGreater> queue_;
    std::vector<MeshCollapse> sequence_;
    std::size_t liveTris_{0};
    double boundsDiagSq_{0.0};
    double uvWeight_{0.0};
    float maxError_{0.0f};
};

} // namespace

SimplifiedMesh QuadricSimplifier::simplify(std::span<const Vertex> vertices,
                                           std::span<const uint32_t> indices,
                                           float targetRatio) const
{
    const std::size_t triCount = indices.size() / 3;
    const float ratio = std::clamp(targetRatio, 0.0f, 1.0f);
    const std::size_t targetTris =
        std::max<std::size_t>(1, static_cast<std::size_t>(static_cast<float>(triCount) * ratio));

    QemRun run(vertices, indices);
    run.run(targetTris);
    return SimplifiedMesh{run.emitIndices(), run.maxError()};
}

std::vector<MeshCollapse>
QuadricSimplifier::collapseSequence(std::span<const Vertex> vertices,
                                    std::span<const uint32_t> indices) const
{
    QemRun run(vertices, indices);
    run.run(1); // collapse as far as the mesh allows
    return std::move(run).sequence();
}

ProgressiveMesh QuadricSimplifier::buildProgressive(std::span<const Vertex> vertices,
                                                    std::span<const uint32_t> indices,
                                                    std::span<const float> ratios) const
{
    const std::size_t triCount = indices.size() / 3;
    ProgressiveMesh out;
    out.lods.push_back(
        ProgressiveLod{std::vector<uint32_t>{indices.begin(), indices.end()}, 0.0f, 0});
    if (triCount == 0)
    {
        return out;
    }

    QemRun run(vertices, indices);
    std::size_t previousIndexCount = indices.size();
    for (const float requestedRatio : ratios)
    {
        const float ratio = std::clamp(requestedRatio, 0.0f, 1.0f);
        const std::size_t targetTris = std::max<std::size_t>(
            1, static_cast<std::size_t>(static_cast<float>(triCount) * ratio));
        run.run(targetTris);
        std::vector<uint32_t> lodIndices = run.emitIndices();
        if (lodIndices.empty() || lodIndices.size() >= previousIndexCount)
        {
            continue;
        }
        previousIndexCount = lodIndices.size();
        out.lods.push_back(
            ProgressiveLod{std::move(lodIndices), run.maxError(), run.collapseCount()});
    }

    run.run(1); // keep the full stream available for future/debug consumers.
    out.collapses = std::move(run).sequence();
    return out;
}

} // namespace fire_engine
