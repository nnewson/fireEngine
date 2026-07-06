#include <fire_engine/graphics/mesh_simplifier.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

namespace fire_engine
{

namespace
{

// A symmetric 4x4 error quadric stored as its 10 unique entries (double for accumulation
// stability). Q(v) for the homogeneous point (x,y,z,1) is vᵀ·A·v; A is built from plane p=(a,b,c,d)
// as p·pᵀ so Q(v) is the squared distance from v to that plane.
struct Quadric
{
    // [xx xy xz xw  yy yz yw  zz zw  ww]
    std::array<double, 10> m{};

    static Quadric fromPlane(double a, double b, double c, double d, double weight) noexcept
    {
        Quadric q;
        q.m[0] = weight * a * a;
        q.m[1] = weight * a * b;
        q.m[2] = weight * a * c;
        q.m[3] = weight * a * d;
        q.m[4] = weight * b * b;
        q.m[5] = weight * b * c;
        q.m[6] = weight * b * d;
        q.m[7] = weight * c * c;
        q.m[8] = weight * c * d;
        q.m[9] = weight * d * d;
        return q;
    }

    Quadric& operator+=(const Quadric& o) noexcept
    {
        for (std::size_t i = 0; i < 10; ++i)
        {
            m[i] += o.m[i];
        }
        return *this;
    }

    [[nodiscard]] double eval(const Vec3& v) const noexcept
    {
        const double x = v.x();
        const double y = v.y();
        const double z = v.z();
        return m[0] * x * x + 2.0 * m[1] * x * y + 2.0 * m[2] * x * z + 2.0 * m[3] * x +
               m[4] * y * y + 2.0 * m[5] * y * z + 2.0 * m[6] * y + m[7] * z * z + 2.0 * m[8] * z +
               m[9];
    }
};

[[nodiscard]] Vec3 triangleNormal(const Vec3& a, const Vec3& b, const Vec3& c) noexcept
{
    const Vec3 n = Vec3::crossProduct(b - a, c - a);
    const float len = n.magnitude();
    return len > 1e-12f ? n * (1.0f / len) : Vec3{};
}

[[nodiscard]] std::uint64_t edgeKey(std::uint32_t a, std::uint32_t b) noexcept
{
    const std::uint64_t lo = a < b ? a : b;
    const std::uint64_t hi = a < b ? b : a;
    return (lo << 32) | hi;
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
    {
        const std::size_t vertexCount = vertices.size();
        pos_.resize(vertexCount);
        for (std::size_t i = 0; i < vertexCount; ++i)
        {
            pos_[i] = vertices[i].position();
        }
        remap_.resize(vertexCount);
        version_.assign(vertexCount, 0);
        quad_.resize(vertexCount);
        vertexTris_.resize(vertexCount);
        for (std::uint32_t i = 0; i < vertexCount; ++i)
        {
            remap_[i] = i;
        }

        for (std::size_t i = 0; i + 3 <= indices.size(); i += 3)
        {
            tris_.push_back({indices[i], indices[i + 1], indices[i + 2]});
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
            const double errA = sum.eval(pos_[ra]);
            const double errB = sum.eval(pos_[rb]);
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
            const std::uint32_t a = resolveConst(tris_[t][0]);
            const std::uint32_t b = resolveConst(tris_[t][1]);
            const std::uint32_t c = resolveConst(tris_[t][2]);
            if (a == b || b == c || a == c)
            {
                continue;
            }
            out.push_back(a);
            out.push_back(b);
            out.push_back(c);
        }
        return out;
    }

    [[nodiscard]] std::vector<MeshCollapse> sequence() const noexcept
    {
        return sequence_;
    }

    [[nodiscard]] float maxError() const noexcept
    {
        return maxError_;
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

        for (const auto& t : tris_)
        {
            const Vec3& a = pos_[t[0]];
            const Vec3& b = pos_[t[1]];
            const Vec3& c = pos_[t[2]];
            const Vec3 n = triangleNormal(a, b, c);
            const double d = -static_cast<double>(Vec3::dotProduct(n, a));
            const Quadric face = Quadric::fromPlane(n.x(), n.y(), n.z(), d, 1.0);
            quad_[t[0]] += face;
            quad_[t[1]] += face;
            quad_[t[2]] += face;

            vertexTris_[t[0]].push_back(static_cast<std::uint32_t>(&t - tris_.data()));
            vertexTris_[t[1]].push_back(static_cast<std::uint32_t>(&t - tris_.data()));
            vertexTris_[t[2]].push_back(static_cast<std::uint32_t>(&t - tris_.data()));

            for (int e = 0; e < 3; ++e)
            {
                ++edgeCount[edgeKey(t[e], t[(e + 1) % 3])];
            }
        }

        // Boundary preservation: an edge in exactly one triangle is a border. Add a heavy plane
        // perpendicular to that triangle through the edge, so its endpoints resist leaving the
        // border.
        for (const auto& t : tris_)
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
                    Quadric::fromPlane(bn.x(), bn.y(), bn.z(), d, kBoundaryWeight);
                quad_[v0] += border;
                quad_[v1] += border;
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
        const double cost = std::min(sum.eval(pos_[a]), sum.eval(pos_[b]));
        queue_.push(Edge{cost, a, b, version_[a], version_[b]});
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

    void collapse(std::uint32_t kept, std::uint32_t removed, double err)
    {
        sequence_.push_back(
            MeshCollapse{kept, removed, pos_[kept], static_cast<float>(err > 0.0 ? err : 0.0)});
        maxError_ = std::max(maxError_, static_cast<float>(err > 0.0 ? err : 0.0));

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
    // Collapse-cost ceiling as a fraction of the mesh's squared bounding-box diagonal. Surface
    // simplification stays far below it; boundary-folding collapses (weighted ×kBoundaryWeight) sit
    // far above, so an un-simplifiable shape is refused rather than destroyed.
    static constexpr double kErrorCeilingFactor = 0.25;

    std::vector<Vec3> pos_;
    std::vector<Quadric> quad_;
    std::vector<std::array<std::uint32_t, 3>> tris_;
    std::vector<std::uint8_t> triAlive_;
    std::vector<std::uint32_t> remap_;
    std::vector<std::uint32_t> version_;
    std::vector<std::vector<std::uint32_t>> vertexTris_;
    std::priority_queue<Edge, std::vector<Edge>, EdgeGreater> queue_;
    std::vector<MeshCollapse> sequence_;
    std::size_t liveTris_{0};
    double boundsDiagSq_{0.0};
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

} // namespace fire_engine
