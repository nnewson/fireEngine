#include <fire_engine/graphics/cloth.hpp>

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
    mesh.resX = params.resX;
    mesh.resZ = params.resZ;

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

    const float compliance = params.compliance;
    // Structural: horizontal + vertical neighbours.
    for (uint32_t j = 0; j < nz; ++j)
    {
        for (uint32_t i = 0; i < nx; ++i)
        {
            if (i + 1 < nx)
            {
                addConstraint(mesh, idx(i, j), idx(i + 1, j), compliance);
            }
            if (j + 1 < nz)
            {
                addConstraint(mesh, idx(i, j), idx(i, j + 1), compliance);
            }
        }
    }
    // Shear: both diagonals of each cell.
    for (uint32_t j = 0; j + 1 < nz; ++j)
    {
        for (uint32_t i = 0; i + 1 < nx; ++i)
        {
            addConstraint(mesh, idx(i, j), idx(i + 1, j + 1), compliance);
            addConstraint(mesh, idx(i + 1, j), idx(i, j + 1), compliance);
        }
    }
    // Bend: skip-one neighbours (softer; keeps folds from collapsing).
    for (uint32_t j = 0; j < nz; ++j)
    {
        for (uint32_t i = 0; i < nx; ++i)
        {
            if (i + 2 < nx)
            {
                addConstraint(mesh, idx(i, j), idx(i + 2, j), compliance);
            }
            if (j + 2 < nz)
            {
                addConstraint(mesh, idx(i, j), idx(i, j + 2), compliance);
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
    return mesh;
}

} // namespace fire_engine
