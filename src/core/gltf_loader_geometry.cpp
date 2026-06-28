#include <fire_engine/core/gltf_loader.hpp>

#include <fire_engine/render/resources.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{
namespace
{

const char* primitiveTypeName(fastgltf::PrimitiveType type)
{
    switch (type)
    {
    case fastgltf::PrimitiveType::Points:
        return "Points";
    case fastgltf::PrimitiveType::Lines:
        return "Lines";
    case fastgltf::PrimitiveType::LineLoop:
        return "LineLoop";
    case fastgltf::PrimitiveType::LineStrip:
        return "LineStrip";
    case fastgltf::PrimitiveType::Triangles:
        return "Triangles";
    case fastgltf::PrimitiveType::TriangleStrip:
        return "TriangleStrip";
    case fastgltf::PrimitiveType::TriangleFan:
        return "TriangleFan";
    }
    return "Unknown";
}

template <typename Value>
std::vector<Value> readAccessorValues(const fastgltf::Asset& asset, std::size_t accessorIndex)
{
    const auto& accessor = asset.accessors[accessorIndex];
    std::vector<Value> values(accessor.count);
    fastgltf::iterateAccessorWithIndex<Value>(asset, accessor, [&](Value value, std::size_t index)
                                              { values[index] = value; });
    return values;
}

std::vector<std::uint32_t> readPrimitiveIndices(const fastgltf::Asset& asset,
                                                const fastgltf::Primitive& primitive,
                                                std::size_t vertexCount)
{
    std::vector<std::uint32_t> indices;
    if (primitive.indicesAccessor.has_value())
    {
        const auto& indexAccessor = asset.accessors[primitive.indicesAccessor.value()];
        indices.reserve(indexAccessor.count);
        fastgltf::iterateAccessor<std::uint32_t>(asset, indexAccessor, [&](std::uint32_t index)
                                                 { indices.push_back(index); });
        return indices;
    }

    indices.reserve(vertexCount);
    for (std::size_t i = 0; i < vertexCount; ++i)
    {
        indices.push_back(static_cast<std::uint32_t>(i));
    }
    return indices;
}

struct PrimitiveGeometryData
{
    std::vector<fastgltf::math::fvec3> positions;
    std::vector<fastgltf::math::fvec3> normals;
    std::vector<fastgltf::math::fvec2> texcoords;
    std::vector<fastgltf::math::fvec2> texcoords1;
    std::vector<std::array<std::uint32_t, 4>> joints;
    std::vector<fastgltf::math::fvec4> weights;
    std::vector<fastgltf::math::fvec4> sourceTangents;
    std::vector<fastgltf::math::fvec4> colours;
    std::vector<std::uint32_t> indices;
    bool hasSourceTangents{false};
};

PrimitiveGeometryData readPrimitiveGeometryData(const fastgltf::Asset& asset,
                                                const fastgltf::Primitive& primitive)
{
    const auto* posAttr = primitive.findAttribute("POSITION");
    const auto* normAttr = primitive.findAttribute("NORMAL");
    const auto* uvAttr = primitive.findAttribute("TEXCOORD_0");
    const auto* uv1Attr = primitive.findAttribute("TEXCOORD_1");
    const auto* colourAttr = primitive.findAttribute("COLOR_0");
    const auto* jointsAttr = primitive.findAttribute("JOINTS_0");
    const auto* weightsAttr = primitive.findAttribute("WEIGHTS_0");
    const auto* tangentAttr = primitive.findAttribute("TANGENT");

    if (posAttr == primitive.attributes.end())
    {
        throw std::runtime_error("glTF primitive missing POSITION attribute");
    }

    PrimitiveGeometryData data;
    data.positions = readAccessorValues<fastgltf::math::fvec3>(asset, posAttr->accessorIndex);

    if (normAttr != primitive.attributes.end())
    {
        data.normals = readAccessorValues<fastgltf::math::fvec3>(asset, normAttr->accessorIndex);
    }
    if (uvAttr != primitive.attributes.end())
    {
        data.texcoords = readAccessorValues<fastgltf::math::fvec2>(asset, uvAttr->accessorIndex);
    }
    if (uv1Attr != primitive.attributes.end())
    {
        data.texcoords1 = readAccessorValues<fastgltf::math::fvec2>(asset, uv1Attr->accessorIndex);
    }
    if (jointsAttr != primitive.attributes.end())
    {
        const auto& jointsAccessor = asset.accessors[jointsAttr->accessorIndex];
        data.joints.resize(jointsAccessor.count);
        fastgltf::iterateAccessorWithIndex<fastgltf::math::u16vec4>(
            asset, jointsAccessor, [&](fastgltf::math::u16vec4 joints, std::size_t index)
            { data.joints[index] = {joints.x(), joints.y(), joints.z(), joints.w()}; });
    }
    if (weightsAttr != primitive.attributes.end())
    {
        data.weights = readAccessorValues<fastgltf::math::fvec4>(asset, weightsAttr->accessorIndex);
    }
    if (tangentAttr != primitive.attributes.end())
    {
        data.hasSourceTangents = true;
        data.sourceTangents =
            readAccessorValues<fastgltf::math::fvec4>(asset, tangentAttr->accessorIndex);
    }
    if (colourAttr != primitive.attributes.end())
    {
        data.colours = readAccessorValues<fastgltf::math::fvec4>(asset, colourAttr->accessorIndex);
    }

    data.indices = readPrimitiveIndices(asset, primitive, data.positions.size());
    return data;
}

void generateMissingNormals(PrimitiveGeometryData& data)
{
    if (!data.normals.empty() || data.positions.empty())
    {
        return;
    }

    std::vector<Vec3> enginePositions;
    enginePositions.reserve(data.positions.size());
    for (const auto& position : data.positions)
    {
        enginePositions.emplace_back(position.x(), position.y(), position.z());
    }

    const auto generated = GltfLoader::generateSmoothNormals(enginePositions, data.indices);
    data.normals.resize(data.positions.size());
    for (std::size_t i = 0; i < generated.size(); ++i)
    {
        data.normals[i] = {generated[i].x(), generated[i].y(), generated[i].z()};
    }
}

std::vector<Vec4> buildTangents(const PrimitiveGeometryData& data, bool needsTangents,
                                TangentGenerationResult& tangentResult)
{
    std::vector<Vec4> tangents;
    if (data.hasSourceTangents)
    {
        tangents.reserve(data.sourceTangents.size());
        for (const auto& tangent : data.sourceTangents)
        {
            tangents.emplace_back(tangent.x(), tangent.y(), tangent.z(), tangent.w());
        }
        tangentResult.succeeded = true;
        return tangents;
    }

    if (!needsTangents)
    {
        return tangents;
    }

    std::vector<Vec3> positionData;
    positionData.reserve(data.positions.size());
    for (const auto& position : data.positions)
    {
        positionData.emplace_back(position.x(), position.y(), position.z());
    }

    std::vector<Vec3> normalData;
    normalData.reserve(data.normals.size());
    for (const auto& normal : data.normals)
    {
        normalData.emplace_back(normal.x(), normal.y(), normal.z());
    }

    std::vector<Vec2> texcoordData;
    texcoordData.reserve(data.texcoords.size());
    for (const auto& texcoord : data.texcoords)
    {
        texcoordData.emplace_back(texcoord.x(), texcoord.y());
    }

    tangentResult =
        TangentGenerator::generate(positionData, normalData, texcoordData, data.indices);
    if (tangentResult.succeeded)
    {
        tangents = tangentResult.tangents;
    }
    return tangents;
}

std::vector<Vertex> buildVertices(const PrimitiveGeometryData& data, std::span<const Vec4> tangents)
{
    std::vector<Vertex> vertices;
    vertices.reserve(data.positions.size());
    for (std::size_t i = 0; i < data.positions.size(); ++i)
    {
        Vec3 position{data.positions[i].x(), data.positions[i].y(), data.positions[i].z()};
        Vec3 normal = (i < data.normals.size())
                          ? Vec3{data.normals[i].x(), data.normals[i].y(), data.normals[i].z()}
                          : Vec3{0.0f, 1.0f, 0.0f};
        float u = (i < data.texcoords.size()) ? data.texcoords[i].x() : 0.0f;
        float v = (i < data.texcoords.size()) ? data.texcoords[i].y() : 0.0f;

        Colour3 vertexColour{1.0f, 1.0f, 1.0f};
        if (i < data.colours.size())
        {
            vertexColour = Colour3{data.colours[i].x(), data.colours[i].y(), data.colours[i].z()};
        }

        Joints4 joints{};
        if (i < data.joints.size())
        {
            joints =
                Joints4{data.joints[i][0], data.joints[i][1], data.joints[i][2], data.joints[i][3]};
        }

        Vec4 weights{};
        if (i < data.weights.size())
        {
            weights = Vec4{data.weights[i].x(), data.weights[i].y(), data.weights[i].z(),
                           data.weights[i].w()};
        }

        Vec4 tangent{0.0f, 0.0f, 0.0f, 1.0f};
        if (i < tangents.size())
        {
            tangent = tangents[i];
        }

        const float u1 = (i < data.texcoords1.size()) ? data.texcoords1[i].x() : u;
        const float v1 = (i < data.texcoords1.size()) ? data.texcoords1[i].y() : v;

        Vertex vertex{position, vertexColour, normal, Vec2{u, v}, joints, weights, tangent};
        vertex.texCoord1(Vec2{u1, v1});
        vertices.push_back(vertex);
    }
    return vertices;
}

void loadMorphTargets(const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                      std::size_t vertexCount, Geometry& geometry)
{
    if (primitive.targets.empty())
    {
        return;
    }

    std::vector<std::vector<Vec3>> morphPositions;
    std::vector<std::vector<Vec3>> morphNormals;
    std::vector<std::vector<Vec3>> morphTangents;

    for (const auto& target : primitive.targets)
    {
        std::vector<Vec3> targetPos(vertexCount);
        std::vector<Vec3> targetNorm(vertexCount);
        std::vector<Vec3> targetTang(vertexCount);

        for (const auto& attr : target)
        {
            const auto& accessor = asset.accessors[attr.accessorIndex];
            if (attr.name == "POSITION")
            {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                    asset, accessor, [&](fastgltf::math::fvec3 position, std::size_t index)
                    { targetPos[index] = Vec3{position.x(), position.y(), position.z()}; });
            }
            else if (attr.name == "NORMAL")
            {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                    asset, accessor, [&](fastgltf::math::fvec3 normal, std::size_t index)
                    { targetNorm[index] = Vec3{normal.x(), normal.y(), normal.z()}; });
            }
            else if (attr.name == "TANGENT")
            {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                    asset, accessor, [&](fastgltf::math::fvec3 tangent, std::size_t index)
                    { targetTang[index] = Vec3{tangent.x(), tangent.y(), tangent.z()}; });
            }
        }

        morphPositions.push_back(std::move(targetPos));
        morphNormals.push_back(std::move(targetNorm));
        morphTangents.push_back(std::move(targetTang));
    }

    geometry.morphPositions(std::move(morphPositions));
    geometry.morphNormals(std::move(morphNormals));
    geometry.morphTangents(std::move(morphTangents));
}

} // namespace

std::vector<Vec3> GltfLoader::generateSmoothNormals(std::span<const Vec3> positions,
                                                    std::span<const uint32_t> indices)
{
    std::vector<Vec3> normals(positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    for (std::size_t k = 0; k + 2 < indices.size(); k += 3)
    {
        const auto i0 = indices[k];
        const auto i1 = indices[k + 1];
        const auto i2 = indices[k + 2];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size())
        {
            continue;
        }
        const Vec3& a = positions[i0];
        const Vec3& b = positions[i1];
        const Vec3& c = positions[i2];
        const Vec3 e1{b.x() - a.x(), b.y() - a.y(), b.z() - a.z()};
        const Vec3 e2{c.x() - a.x(), c.y() - a.y(), c.z() - a.z()};
        const Vec3 face = Vec3::crossProduct(e1, e2);
        normals[i0] += face;
        normals[i1] += face;
        normals[i2] += face;
    }
    for (auto& normal : normals)
    {
        if (normal.magnitudeSquared() > 1e-16f)
        {
            normal = Vec3::normalise(normal);
        }
        else
        {
            normal = Vec3{0.0f, 1.0f, 0.0f};
        }
    }
    return normals;
}

TangentGenerationResult GltfLoader::loadGeometry(const fastgltf::Asset& asset,
                                                 const fastgltf::Primitive& primitive,
                                                 bool needsTangents, Resources& resources,
                                                 Assets& assets, std::size_t geoIdx)
{
    auto& geometry = assets.geometry(geoIdx);
    if (geometry.loaded())
    {
        return {};
    }

    if (!isSupportedPrimitiveType(primitive.type))
    {
        std::clog << "Skipping glTF primitive with unsupported mode: "
                  << primitiveTypeName(primitive.type)
                  << " (only Triangles is currently rendered).\n";
        return {};
    }

    auto data = readPrimitiveGeometryData(asset, primitive);
    generateMissingNormals(data);

    TangentGenerationResult tangentResult;
    auto tangents = buildTangents(data, needsTangents, tangentResult);
    geometry.vertices(buildVertices(data, tangents));
    geometry.indices(std::move(data.indices));
    loadMorphTargets(asset, primitive, data.positions.size(), geometry);

    geometry.load(resources);
    return tangentResult;
}

} // namespace fire_engine
