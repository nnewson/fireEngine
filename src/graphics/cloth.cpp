#include <fire_engine/graphics/cloth.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace fire_engine
{

ClothCollider makePlaneCollider(Vec3 normal, float offset)
{
    ClothCollider c;
    c.type = static_cast<int32_t>(ClothColliderType::Plane);
    c.a[0] = normal.x();
    c.a[1] = normal.y();
    c.a[2] = normal.z();
    c.a[3] = offset;
    return c;
}

ClothCollider makeSphereCollider(Vec3 center, float radius)
{
    ClothCollider c;
    c.type = static_cast<int32_t>(ClothColliderType::Sphere);
    c.a[0] = center.x();
    c.a[1] = center.y();
    c.a[2] = center.z();
    c.a[3] = radius;
    return c;
}

ClothCollider makeBoxCollider(Vec3 center, Vec3 halfExtents, Quaternion orientation)
{
    ClothCollider c;
    c.type = static_cast<int32_t>(ClothColliderType::Box);
    c.a[0] = center.x();
    c.a[1] = center.y();
    c.a[2] = center.z();
    c.b[0] = halfExtents.x();
    c.b[1] = halfExtents.y();
    c.b[2] = halfExtents.z();
    c.c[0] = orientation.x();
    c.c[1] = orientation.y();
    c.c[2] = orientation.z();
    c.c[3] = orientation.w();
    return c;
}

ClothCollider makeCapsuleCollider(Vec3 p0, Vec3 p1, float radius)
{
    ClothCollider c;
    c.type = static_cast<int32_t>(ClothColliderType::Capsule);
    c.a[0] = p0.x();
    c.a[1] = p0.y();
    c.a[2] = p0.z();
    c.a[3] = radius;
    c.b[0] = p1.x();
    c.b[1] = p1.y();
    c.b[2] = p1.z();
    return c;
}

namespace
{

// Append a distance constraint between particles a and b, with rest length taken
// from their initial positions.
void addConstraint(ClothMesh& mesh, uint32_t a, uint32_t b, float compliance)
{
    const float rest = (mesh.particles[b].position - mesh.particles[a].position).magnitude();
    mesh.constraints.push_back(
        ClothConstraint{.a = a, .b = b, .restLength = rest, .compliance = compliance});
}

} // namespace

void colourConstraints(std::vector<ClothConstraint>& constraints,
                       std::vector<uint32_t>& colourRanges, uint32_t particleCount)
{
    const std::size_t n = constraints.size();
    std::vector<uint32_t> colour(n, 0);
    // Colours already claimed by each particle's incident constraints.
    std::vector<std::unordered_set<uint32_t>> particleColours(particleCount);

    uint32_t numColours = 0;
    for (std::size_t e = 0; e < n; ++e)
    {
        const ClothConstraint& c = constraints[e];
        uint32_t col = 0;
        while (particleColours[c.a].contains(col) || particleColours[c.b].contains(col))
        {
            ++col;
        }
        colour[e] = col;
        particleColours[c.a].insert(col);
        particleColours[c.b].insert(col);
        numColours = std::max(numColours, col + 1);
    }

    // Counting sort the constraints by colour into CSR order.
    colourRanges.assign(numColours + 1, 0);
    for (std::size_t e = 0; e < n; ++e)
    {
        ++colourRanges[colour[e] + 1];
    }
    for (uint32_t c = 0; c < numColours; ++c)
    {
        colourRanges[c + 1] += colourRanges[c];
    }

    std::vector<ClothConstraint> sorted(n);
    std::vector<uint32_t> cursor(colourRanges.begin(), colourRanges.end() - 1);
    for (std::size_t e = 0; e < n; ++e)
    {
        sorted[cursor[colour[e]]++] = constraints[e];
    }
    constraints = std::move(sorted);
}

ClothMesh makeGridCloth(const ClothGridParams& params)
{
    ClothMesh mesh;

    const uint32_t nx = params.resX;
    const uint32_t nz = params.resZ;
    const uint32_t count = nx * nz;
    const float spacing = params.spacing;
    // Uniform inverse mass for free particles. Only the *ratio* between particles
    // matters to the constraint solve (and it's identical for a uniform cloth), so
    // 1.0 keeps compliance intuitive; pinned particles get 0 below. (`mass` is
    // reserved for future non-uniform cloths.)
    const float invMass = params.mass > 0.0f ? 1.0f : 0.0f;

    const float originX = -0.5f * static_cast<float>(nx - 1) * spacing;
    const float originZ = -0.5f * static_cast<float>(nz - 1) * spacing;

    auto idx = [nx](uint32_t i, uint32_t j) { return j * nx + i; };

    mesh.particles.reserve(count);
    mesh.vertices.reserve(count);
    for (uint32_t j = 0; j < nz; ++j)
    {
        for (uint32_t i = 0; i < nx; ++i)
        {
            const Vec3 pos{originX + static_cast<float>(i) * spacing + params.origin.x(),
                           params.origin.y(),
                           originZ + static_cast<float>(j) * spacing + params.origin.z()};
            mesh.particles.push_back(ClothParticle{.position = pos, .invMass = invMass});

            const Vec2 uv{static_cast<float>(i) / static_cast<float>(nx - 1),
                          static_cast<float>(j) / static_cast<float>(nz - 1)};
            mesh.vertices.emplace_back(pos, Colour3{1.0f, 1.0f, 1.0f}, Vec3{0.0f, 1.0f, 0.0f}, uv);
        }
    }

    if (params.pinTopCorners && count > 0)
    {
        mesh.particles[idx(0, nz - 1)].invMass = 0.0f;
        mesh.particles[idx(nx - 1, nz - 1)].invMass = 0.0f;
    }

    // Structural: horizontal + vertical neighbours.
    for (uint32_t j = 0; j < nz; ++j)
    {
        for (uint32_t i = 0; i < nx; ++i)
        {
            if (i + 1 < nx)
            {
                addConstraint(mesh, idx(i, j), idx(i + 1, j), params.structuralCompliance);
            }
            if (j + 1 < nz)
            {
                addConstraint(mesh, idx(i, j), idx(i, j + 1), params.structuralCompliance);
            }
        }
    }
    // Shear: both diagonals of each cell.
    for (uint32_t j = 0; j + 1 < nz; ++j)
    {
        for (uint32_t i = 0; i + 1 < nx; ++i)
        {
            addConstraint(mesh, idx(i, j), idx(i + 1, j + 1), params.shearCompliance);
            addConstraint(mesh, idx(i + 1, j), idx(i, j + 1), params.shearCompliance);
        }
    }
    // Bend: skip-one neighbours (softer; keeps folds from collapsing).
    for (uint32_t j = 0; j < nz; ++j)
    {
        for (uint32_t i = 0; i < nx; ++i)
        {
            if (i + 2 < nx)
            {
                addConstraint(mesh, idx(i, j), idx(i + 2, j), params.bendCompliance);
            }
            if (j + 2 < nz)
            {
                addConstraint(mesh, idx(i, j), idx(i, j + 2), params.bendCompliance);
            }
        }
    }

    // Two triangles per grid cell (CCW from +Y).
    mesh.indices.reserve((nx - 1) * (nz - 1) * 6);
    for (uint32_t j = 0; j + 1 < nz; ++j)
    {
        for (uint32_t i = 0; i + 1 < nx; ++i)
        {
            const uint32_t p00 = idx(i, j);
            const uint32_t p10 = idx(i + 1, j);
            const uint32_t p01 = idx(i, j + 1);
            const uint32_t p11 = idx(i + 1, j + 1);
            mesh.indices.insert(mesh.indices.end(), {p00, p11, p10, p00, p01, p11});
        }
    }

    colourConstraints(mesh.constraints, mesh.colourRanges, count);
    buildNormalAdjacency(mesh.indices, count, mesh.normalAdjOffsets, mesh.normalAdjTris);
    return mesh;
}

void buildNormalAdjacency(std::span<const uint32_t> indices, uint32_t particleCount,
                          std::vector<uint32_t>& offsets, std::vector<uint32_t>& tris)
{
    const std::size_t triCount = indices.size() / 3;

    // Counting sort triangle references into CSR order keyed by vertex.
    offsets.assign(particleCount + 1, 0);
    for (std::size_t t = 0; t < triCount; ++t)
    {
        for (uint32_t k = 0; k < 3; ++k)
        {
            ++offsets[indices[t * 3 + k] + 1];
        }
    }
    for (uint32_t v = 0; v < particleCount; ++v)
    {
        offsets[v + 1] += offsets[v];
    }

    tris.assign(offsets[particleCount], 0);
    std::vector<uint32_t> cursor(offsets.begin(), offsets.end() - 1);
    for (std::size_t t = 0; t < triCount; ++t)
    {
        for (uint32_t k = 0; k < 3; ++k)
        {
            tris[cursor[indices[t * 3 + k]]++] = static_cast<uint32_t>(t);
        }
    }
}

namespace
{

// Weld vertices whose positions coincide (within an epsilon grid) into shared
// particles, so a glTF mesh's duplicated seam/UV-split vertices simulate as one
// point and don't tear. Returns a remap from original vertex index → particle
// index; `particlePositions` is filled with one representative position each.
std::vector<uint32_t> weldPositions(std::span<const Vertex> vertices,
                                    std::vector<Vec3>& particlePositions, float epsilon)
{
    const float inv = 1.0f / epsilon;
    auto key = [inv](const Vec3& p)
    {
        return std::tuple<int64_t, int64_t, int64_t>{
            static_cast<int64_t>(std::llround(p.x() * inv)),
            static_cast<int64_t>(std::llround(p.y() * inv)),
            static_cast<int64_t>(std::llround(p.z() * inv))};
    };

    struct KeyHash
    {
        std::size_t operator()(const std::tuple<int64_t, int64_t, int64_t>& k) const noexcept
        {
            const auto h = std::hash<int64_t>{};
            return h(std::get<0>(k)) * 73856093ULL ^ h(std::get<1>(k)) * 19349663ULL ^
                   h(std::get<2>(k)) * 83492791ULL;
        }
    };

    std::unordered_map<std::tuple<int64_t, int64_t, int64_t>, uint32_t, KeyHash> seen;
    std::vector<uint32_t> remap(vertices.size(), 0);
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        const Vec3 p = vertices[i].position();
        const auto k = key(p);
        auto it = seen.find(k);
        if (it == seen.end())
        {
            const auto particle = static_cast<uint32_t>(particlePositions.size());
            seen.emplace(k, particle);
            particlePositions.push_back(p);
            remap[i] = particle;
        }
        else
        {
            remap[i] = it->second;
        }
    }
    return remap;
}

} // namespace

ClothMesh makeClothFromMesh(std::span<const Vertex> vertices, std::span<const uint32_t> indices,
                            const ClothMeshParams& params)
{
    ClothMesh mesh;

    // Weld coincident positions into particles. Epsilon is a fraction of the mesh
    // size so it scales with the asset rather than assuming metre-scale geometry.
    Vec3 lo{vertices.empty() ? Vec3{} : vertices[0].position()};
    Vec3 hi{lo};
    for (const Vertex& v : vertices)
    {
        const Vec3 p = v.position();
        lo = Vec3{std::min(lo.x(), p.x()), std::min(lo.y(), p.y()), std::min(lo.z(), p.z())};
        hi = Vec3{std::max(hi.x(), p.x()), std::max(hi.y(), p.y()), std::max(hi.z(), p.z())};
    }
    const Vec3 extent = hi - lo;
    const float diag = extent.magnitude();
    const float epsilon = std::max(diag * 1.0e-5f, 1.0e-6f);

    std::vector<Vec3> particlePositions;
    const std::vector<uint32_t> remap = weldPositions(vertices, particlePositions, epsilon);
    const auto particleCount = static_cast<uint32_t>(particlePositions.size());

    mesh.particles.reserve(particleCount);
    for (const Vec3& p : particlePositions)
    {
        mesh.particles.push_back(ClothParticle{.position = p, .invMass = 1.0f});
    }

    // Render vertices keep their original attributes; only their positions will be
    // driven by the solver. Indices are remapped onto welded particles so the
    // simulation, normal adjacency, and the render mesh all share one index space.
    mesh.vertices.assign(vertices.begin(), vertices.end());
    mesh.indices.reserve(indices.size());
    for (uint32_t i : indices)
    {
        mesh.indices.push_back(remap[i]);
    }

    // Structural constraints: one per unique mesh edge. Bend constraints: for each
    // interior edge shared by two triangles, link the two opposite ("wing")
    // vertices. Both keyed on ordered particle-index pairs to dedupe.
    auto edgeKey = [](uint32_t a, uint32_t b)
    { return (static_cast<uint64_t>(std::min(a, b)) << 32) | std::max(a, b); };

    std::unordered_map<uint64_t, uint32_t> edgeOpposite; // edge → its first triangle's wing vertex
    std::unordered_set<uint64_t> structural;
    std::unordered_set<uint64_t> bend;

    auto addStructural = [&](uint32_t a, uint32_t b)
    {
        if (a != b && structural.insert(edgeKey(a, b)).second)
        {
            const float rest =
                (mesh.particles[b].position - mesh.particles[a].position).magnitude();
            mesh.constraints.push_back(ClothConstraint{
                .a = a, .b = b, .restLength = rest, .compliance = params.structuralCompliance});
        }
    };
    auto addBend = [&](uint32_t a, uint32_t b)
    {
        if (a != b && bend.insert(edgeKey(a, b)).second)
        {
            const float rest =
                (mesh.particles[b].position - mesh.particles[a].position).magnitude();
            mesh.constraints.push_back(ClothConstraint{
                .a = a, .b = b, .restLength = rest, .compliance = params.bendCompliance});
        }
    };

    const std::size_t triCount = mesh.indices.size() / 3;
    for (std::size_t t = 0; t < triCount; ++t)
    {
        const uint32_t v[3] = {mesh.indices[t * 3], mesh.indices[t * 3 + 1],
                               mesh.indices[t * 3 + 2]};
        for (uint32_t e = 0; e < 3; ++e)
        {
            const uint32_t a = v[e];
            const uint32_t b = v[(e + 1) % 3];
            const uint32_t wing = v[(e + 2) % 3]; // the vertex opposite this edge
            addStructural(a, b);

            const uint64_t k = edgeKey(a, b);
            auto it = edgeOpposite.find(k);
            if (it == edgeOpposite.end())
            {
                edgeOpposite.emplace(k, wing);
            }
            else
            {
                addBend(it->second, wing); // second triangle on this edge: link the wings
            }
        }
    }

    if (params.pin != ClothMeshParams::Pin::None && particleCount > 0)
    {
        // Resolve pins against local bounds: the max-Z edge, or its two extreme-X
        // corners. Tolerances are a fraction of the extent so welded grids hit.
        const float zMax = hi.z();
        const float zTol = std::max(extent.z() * 1.0e-3f, epsilon);
        const float xTol = std::max(extent.x() * 1.0e-3f, epsilon);
        for (uint32_t p = 0; p < particleCount; ++p)
        {
            const Vec3& pos = mesh.particles[p].position;
            if (pos.z() < zMax - zTol)
            {
                continue;
            }
            const bool corner = pos.x() <= lo.x() + xTol || pos.x() >= hi.x() - xTol;
            if (params.pin == ClothMeshParams::Pin::TopEdge ||
                (params.pin == ClothMeshParams::Pin::TopCorners && corner))
            {
                mesh.particles[p].invMass = 0.0f;
            }
        }
    }

    colourConstraints(mesh.constraints, mesh.colourRanges, particleCount);
    buildNormalAdjacency(mesh.indices, particleCount, mesh.normalAdjOffsets, mesh.normalAdjTris);
    return mesh;
}

} // namespace fire_engine
