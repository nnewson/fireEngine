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
        uv_.resize(vertexCount);
        weld_.resize(vertexCount);
        canonicalWedges_.resize(vertexCount);
        for (std::uint32_t i = 0; i < vertexCount; ++i)
        {
            remap_[i] = i;
            uv_[i] = vertices[i].texCoord();
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
            const std::uint32_t kept = errA <= errB ? ra : rb;
            const std::uint32_t removed = kept == ra ? rb : ra;
            const double err = errA <= errB ? errA : errB;

            if (err > ceiling)
            {
                break; // min-heap: nothing cheaper remains
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

    [[nodiscard]] std::vector<MeshCollapse> sequence() const noexcept
    {
        return sequence_;
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
        // resist leaving the border.
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
    std::vector<Vec2> uv_;            // per original vertex, for the wedge-preserving emit
    std::vector<std::uint32_t> weld_; // original vertex -> canonical position vertex
    std::vector<std::array<std::uint32_t, 3>> origTris_; // original (pre-weld) corners, for UV emit
    std::vector<std::vector<std::uint32_t>> canonicalWedges_; // canonical -> its native wedges
    std::vector<Quadric> quad_;
    std::vector<std::array<std::uint32_t, 3>> tris_;
    std::vector<std::uint8_t> triAlive_;
    std::vector<std::uint32_t> remap_;
    std::vector<std::uint32_t> version_;
    std::vector<double> weight_; // accumulated face-plane weight per vertex (for RMS error)
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
    return run.sequence();
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
    out.collapses = run.sequence();
    return out;
}

} // namespace fire_engine
