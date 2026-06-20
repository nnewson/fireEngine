#include <fire_engine/core/convex_hull_builder.hpp>

#include <cmath>
#include <utility>
#include <vector>

namespace fire_engine
{

namespace
{

constexpr float kWeldEpsSq = 1e-10f; // squared distance for welding vertices
constexpr float kNormalEps = 0.999f; // dot threshold for coplanar face normals
constexpr float kOffsetEps = 1e-4f;  // plane-offset tolerance for coplanarity

[[nodiscard]] float dot(const Vec3& a, const Vec3& b) noexcept
{
    return Vec3::dotProduct(a, b);
}

struct Triangle
{
    int v0{0};
    int v1{0};
    int v2{0};
    Vec3 normal{};
    float offset{0.0f};
};

struct FaceGroup
{
    Vec3 normal{};
    float offset{0.0f};
    std::vector<int> triangles;
};

// Cancel directed edge (a,b) against an existing (b,a) — interior edge shared by two
// triangles of the group; otherwise record it as a boundary edge.
void addBoundaryEdge(std::vector<std::pair<int, int>>& edges, int a, int b)
{
    for (std::size_t i = 0; i < edges.size(); ++i)
    {
        if (edges[i].first == b && edges[i].second == a)
        {
            edges[i] = edges.back();
            edges.pop_back();
            return;
        }
    }
    edges.emplace_back(a, b);
}

} // namespace

ConvexHullShape buildConvexHull(std::span<const Vec3> positions,
                                std::span<const std::uint32_t> indices)
{
    ConvexHullShape hull;
    if (indices.size() < 3)
    {
        return hull;
    }

    // 1. Weld coincident vertices (collider meshes are small → O(n²) is fine).
    std::vector<int> remap(positions.size(), 0);
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        int found = -1;
        for (std::size_t j = 0; j < hull.vertices.size(); ++j)
        {
            if ((hull.vertices[j] - positions[i]).magnitudeSquared() < kWeldEpsSq)
            {
                found = static_cast<int>(j);
                break;
            }
        }
        if (found < 0)
        {
            found = static_cast<int>(hull.vertices.size());
            hull.vertices.push_back(positions[i]);
        }
        remap[i] = found;
    }

    // 2. Build welded, non-degenerate triangles with their planes.
    std::vector<Triangle> tris;
    for (std::size_t t = 0; t + 2 < indices.size(); t += 3)
    {
        Triangle tri{remap[indices[t]], remap[indices[t + 1]], remap[indices[t + 2]]};
        if (tri.v0 == tri.v1 || tri.v1 == tri.v2 || tri.v0 == tri.v2)
        {
            continue;
        }
        const Vec3& a = hull.vertices[static_cast<std::size_t>(tri.v0)];
        const Vec3& b = hull.vertices[static_cast<std::size_t>(tri.v1)];
        const Vec3& c = hull.vertices[static_cast<std::size_t>(tri.v2)];
        const Vec3 n = Vec3::crossProduct(b - a, c - a);
        if (n.magnitudeSquared() < kWeldEpsSq)
        {
            continue; // degenerate (zero area)
        }
        tri.normal = Vec3::normalise(n);
        tri.offset = dot(tri.normal, a);
        tris.push_back(tri);
    }

    // 3. Group coplanar triangles.
    std::vector<FaceGroup> groups;
    for (std::size_t i = 0; i < tris.size(); ++i)
    {
        FaceGroup* match = nullptr;
        for (FaceGroup& g : groups)
        {
            if (dot(g.normal, tris[i].normal) > kNormalEps &&
                std::abs(g.offset - tris[i].offset) < kOffsetEps)
            {
                match = &g;
                break;
            }
        }
        if (match == nullptr)
        {
            groups.push_back({tris[i].normal, tris[i].offset, {}});
            match = &groups.back();
        }
        match->triangles.push_back(static_cast<int>(i));
    }

    // 4. Per group: cancel interior edges, chain boundary edges into a loop.
    for (const FaceGroup& g : groups)
    {
        std::vector<std::pair<int, int>> edges;
        for (int ti : g.triangles)
        {
            const Triangle& tri = tris[static_cast<std::size_t>(ti)];
            addBoundaryEdge(edges, tri.v0, tri.v1);
            addBoundaryEdge(edges, tri.v1, tri.v2);
            addBoundaryEdge(edges, tri.v2, tri.v0);
        }
        if (edges.size() < 3)
        {
            continue;
        }

        ConvexFace face;
        face.normal = g.normal;
        const int start = edges.front().first;
        int current = start;
        for (std::size_t guard = 0; guard <= edges.size(); ++guard)
        {
            face.loop.push_back(current);
            int next = -1;
            for (const auto& e : edges)
            {
                if (e.first == current)
                {
                    next = e.second;
                    break;
                }
            }
            if (next < 0 || next == start)
            {
                break;
            }
            current = next;
        }
        if (face.loop.size() >= 3)
        {
            hull.faces.push_back(std::move(face));
        }
    }

    return hull;
}

bool isConvex(const ConvexHullShape& hull) noexcept
{
    if (hull.faces.empty())
    {
        return false;
    }
    for (const ConvexFace& face : hull.faces)
    {
        if (face.loop.empty())
        {
            return false;
        }
        const float offset =
            dot(face.normal, hull.vertices[static_cast<std::size_t>(face.loop[0])]);
        for (const Vec3& v : hull.vertices)
        {
            if (dot(face.normal, v) > offset + kOffsetEps)
            {
                return false; // a vertex in front of a face → not convex
            }
        }
    }
    return true;
}

} // namespace fire_engine
