#include <fire_engine/core/tangent_generator.hpp>

#include <cmath>

#include <fire_engine/math/constants.hpp>

namespace fire_engine
{

namespace
{

constexpr float tangentEpsilon = 1e-8f;

[[nodiscard]] TangentGenerationResult failure(const std::string& reason)
{
    TangentGenerationResult result;
    result.reason = reason;
    return result;
}

[[nodiscard]] Vec4 fallbackTangentFromNormal(const Vec3& normal)
{
    Vec3 up = std::abs(normal.z()) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{0.0f, 1.0f, 0.0f};
    Vec3 tangent = Vec3::crossProduct(up, normal);
    if (tangent.magnitudeSquared() <= tangentEpsilon)
    {
        up = Vec3{1.0f, 0.0f, 0.0f};
        tangent = Vec3::crossProduct(up, normal);
    }
    tangent = Vec3::normalise(tangent);
    return Vec4{tangent.x(), tangent.y(), tangent.z(), 1.0f};
}

} // namespace

TangentGenerationResult TangentGenerator::generate(std::span<const Vec3> positions,
                                                   std::span<const Vec3> normals,
                                                   std::span<const Vec2> texcoords,
                                                   std::span<const uint32_t> indices)
{
    if (positions.empty())
    {
        return failure("missing positions");
    }
    if (normals.size() != positions.size())
    {
        return failure("missing normals");
    }
    if (texcoords.size() != positions.size())
    {
        return failure("missing UVs");
    }
    if (indices.empty())
    {
        return failure("missing triangle indices");
    }
    if ((indices.size() % 3u) != 0u)
    {
        return failure("non-triangle primitive");
    }

    std::vector<Vec3> tan1(positions.size());
    std::vector<Vec3> tan2(positions.size());
    std::vector<bool> contributed(positions.size(), false);
    std::size_t validTriangleCount = 0;

    for (std::size_t i = 0; i < indices.size(); i += 3)
    {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i + 1];
        uint32_t i2 = indices[i + 2];

        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size())
        {
            return failure("index out of range");
        }

        const Vec3& p0 = positions[i0];
        const Vec3& p1 = positions[i1];
        const Vec3& p2 = positions[i2];

        const Vec2& uv0 = texcoords[i0];
        const Vec2& uv1 = texcoords[i1];
        const Vec2& uv2 = texcoords[i2];

        Vec3 edge1 = p1 - p0;
        Vec3 edge2 = p2 - p0;
        Vec2 deltaUv1 = uv1 - uv0;
        Vec2 deltaUv2 = uv2 - uv0;

        float determinant = deltaUv1.s() * deltaUv2.t() - deltaUv1.t() * deltaUv2.s();
        if (std::abs(determinant) <= tangentEpsilon)
        {
            continue;
        }

        float invDeterminant = 1.0f / determinant;
        Vec3 sdir = (edge1 * deltaUv2.t() - edge2 * deltaUv1.t()) * invDeterminant;
        Vec3 tdir = (edge2 * deltaUv1.s() - edge1 * deltaUv2.s()) * invDeterminant;

        if (sdir.magnitudeSquared() <= tangentEpsilon || tdir.magnitudeSquared() <= tangentEpsilon)
        {
            continue;
        }

        tan1[i0] += sdir;
        tan1[i1] += sdir;
        tan1[i2] += sdir;
        tan2[i0] += tdir;
        tan2[i1] += tdir;
        tan2[i2] += tdir;
        contributed[i0] = true;
        contributed[i1] = true;
        contributed[i2] = true;
        ++validTriangleCount;
    }

    if (validTriangleCount == 0)
    {
        return failure("degenerate UV mapping");
    }

    TangentGenerationResult result;
    result.succeeded = true;
    result.tangents.resize(positions.size());

    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        Vec3 normal = Vec3::normalise(normals[i]);
        if (normal.magnitudeSquared() <= tangentEpsilon)
        {
            return failure("missing normals");
        }

        if (!contributed[i])
        {
            result.tangents[i] = fallbackTangentFromNormal(normal);
            continue;
        }

        Vec3 tangent = tan1[i] - normal * Vec3::dotProduct(normal, tan1[i]);
        if (tangent.magnitudeSquared() <= tangentEpsilon)
        {
            result.tangents[i] = fallbackTangentFromNormal(normal);
            continue;
        }

        tangent = Vec3::normalise(tangent);
        float handedness =
            Vec3::dotProduct(Vec3::crossProduct(normal, tangent), tan2[i]) < 0.0f ? -1.0f : 1.0f;

        result.tangents[i] = Vec4{tangent.x(), tangent.y(), tangent.z(), handedness};
    }

    return result;
}

} // namespace fire_engine
