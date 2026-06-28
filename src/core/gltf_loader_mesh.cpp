#include <fire_engine/core/gltf_loader.hpp>

#include <fire_engine/core/convex_hull_builder.hpp>
#include <fire_engine/render/resources.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <simdjson.h>

#include <fire_engine/animation/animation.hpp>
#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/ktx_image.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/object.hpp>
#include <fire_engine/graphics/sampler_settings.hpp>
#include <fire_engine/graphics/skin.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/scene/animator.hpp>
#include <fire_engine/scene/camera.hpp>
#include <fire_engine/scene/empty.hpp>
#include <fire_engine/scene/light.hpp>
#include <fire_engine/scene/mesh.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{
bool GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType type) noexcept
{
    return type == fastgltf::PrimitiveType::Triangles;
}

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
} // namespace

static WrapMode toWrapMode(fastgltf::Wrap w)
{
    switch (w)
    {
    case fastgltf::Wrap::MirroredRepeat:
        return WrapMode::MirroredRepeat;
    case fastgltf::Wrap::ClampToEdge:
        return WrapMode::ClampToEdge;
    default:
        return WrapMode::Repeat;
    }
}

static FilterMode toFilterMode(fastgltf::Filter f)
{
    switch (f)
    {
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return FilterMode::Nearest;
    default:
        return FilterMode::Linear;
    }
}

static SamplerSettings extractSamplerSettings(const fastgltf::Asset& asset,
                                              std::size_t textureIndex)
{
    SamplerSettings settings;
    const auto& tex = asset.textures[textureIndex];
    if (tex.samplerIndex.has_value())
    {
        const auto& sampler = asset.samplers[tex.samplerIndex.value()];
        settings.wrapS = toWrapMode(sampler.wrapS);
        settings.wrapT = toWrapMode(sampler.wrapT);
        if (sampler.magFilter.has_value())
        {
            settings.magFilter = toFilterMode(sampler.magFilter.value());
        }
        if (sampler.minFilter.has_value())
        {
            settings.minFilter = toFilterMode(sampler.minFilter.value());
        }
    }
    return settings;
}

static std::vector<std::byte> readFileBytes(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> bytes(size);
    if (size > 0)
    {
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    }
    if (!file && size > 0)
    {
        throw std::runtime_error("Failed to read file: " + path.string());
    }

    return bytes;
}

static std::vector<std::byte> sliceBytes(std::span<const std::byte> bytes, std::size_t offset,
                                         std::size_t length, const std::string& label)
{
    if (offset > bytes.size() || offset + length > bytes.size())
    {
        throw std::runtime_error("Out-of-range byte slice for " + label);
    }

    return {bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + length)};
}

static std::vector<std::byte> loadDataSourceBytes(const fastgltf::Asset& asset,
                                                  const fastgltf::DataSource& source,
                                                  const std::string& baseDir,
                                                  const std::string& label)
{
    if (auto* uri = std::get_if<fastgltf::sources::URI>(&source))
    {
        if (!uri->uri.isLocalPath())
        {
            throw std::runtime_error("Unsupported non-local URI for " + label + ": " +
                                     std::string(uri->uri.string()));
        }

        auto bytes = readFileBytes(std::filesystem::path(baseDir) / uri->uri.fspath());
        return sliceBytes(bytes, uri->fileByteOffset, bytes.size() - uri->fileByteOffset, label);
    }
    if (auto* array = std::get_if<fastgltf::sources::Array>(&source))
    {
        return {array->bytes.begin(), array->bytes.end()};
    }
    if (auto* byteView = std::get_if<fastgltf::sources::ByteView>(&source))
    {
        return {byteView->bytes.begin(), byteView->bytes.end()};
    }
    if (auto* bufferView = std::get_if<fastgltf::sources::BufferView>(&source))
    {
        const auto& view = asset.bufferViews[bufferView->bufferViewIndex];
        auto bytes =
            loadDataSourceBytes(asset, asset.buffers[view.bufferIndex].data, baseDir, label);
        return sliceBytes(bytes, view.byteOffset, view.byteLength, label);
    }
    if (auto* vector = std::get_if<fastgltf::sources::Vector>(&source))
    {
        return {vector->bytes.begin(), vector->bytes.end()};
    }
    if (std::holds_alternative<fastgltf::sources::CustomBuffer>(source))
    {
        throw std::runtime_error("Unsupported custom buffer source for " + label);
    }
    if (std::holds_alternative<fastgltf::sources::Fallback>(source))
    {
        throw std::runtime_error("Unsupported fallback source for " + label);
    }

    throw std::runtime_error("Unsupported data source for " + label);
}

// ---------------------------------------------------------------------------
// Mesh loading helpers
// ---------------------------------------------------------------------------

namespace
{
[[nodiscard]]
std::optional<Vec3> boundsArrayVec3(const fastgltf::AccessorBoundsArray& bounds)
{
    if (bounds.size() < 3)
    {
        return std::nullopt;
    }

    if (bounds.isType<double>())
    {
        return Vec3{static_cast<float>(bounds.get<double>(0)),
                    static_cast<float>(bounds.get<double>(1)),
                    static_cast<float>(bounds.get<double>(2))};
    }
    if (bounds.isType<std::int64_t>())
    {
        return Vec3{static_cast<float>(bounds.get<std::int64_t>(0)),
                    static_cast<float>(bounds.get<std::int64_t>(1)),
                    static_cast<float>(bounds.get<std::int64_t>(2))};
    }

    return std::nullopt;
}

[[nodiscard]]
AABB mergeBounds(const AABB& lhs, const AABB& rhs) noexcept
{
    return AABB{
        Vec3{std::min(lhs.min.x(), rhs.min.x()), std::min(lhs.min.y(), rhs.min.y()),
             std::min(lhs.min.z(), rhs.min.z())},
        Vec3{std::max(lhs.max.x(), rhs.max.x()), std::max(lhs.max.y(), rhs.max.y()),
             std::max(lhs.max.z(), rhs.max.z())},
    };
}
} // namespace

std::optional<AABB> GltfLoader::primitiveBounds(const fastgltf::Asset& asset,
                                                const fastgltf::Primitive& primitive)
{
    if (!isSupportedPrimitiveType(primitive.type))
    {
        return std::nullopt;
    }

    const auto* posAttr = primitive.findAttribute("POSITION");
    if (posAttr == primitive.attributes.end())
    {
        return std::nullopt;
    }

    const auto& posAccessor = asset.accessors[posAttr->accessorIndex];
    if (posAccessor.min.has_value() && posAccessor.max.has_value())
    {
        auto min = boundsArrayVec3(*posAccessor.min);
        auto max = boundsArrayVec3(*posAccessor.max);
        if (min && max)
        {
            return AABB{*min, *max};
        }
    }

    std::optional<AABB> bounds;
    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
        asset, posAccessor,
        [&bounds](fastgltf::math::fvec3 pos, std::size_t)
        {
            const Vec3 point{pos.x(), pos.y(), pos.z()};
            if (!bounds.has_value())
            {
                bounds = AABB{point, point};
                return;
            }

            bounds->min.x(std::min(bounds->min.x(), point.x()));
            bounds->min.y(std::min(bounds->min.y(), point.y()));
            bounds->min.z(std::min(bounds->min.z(), point.z()));
            bounds->max.x(std::max(bounds->max.x(), point.x()));
            bounds->max.y(std::max(bounds->max.y(), point.y()));
            bounds->max.z(std::max(bounds->max.z(), point.z()));
        });

    return bounds;
}

std::optional<AABB> GltfLoader::meshBounds(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh)
{
    std::optional<AABB> bounds;
    for (const auto& primitive : mesh.primitives)
    {
        auto primitiveBox = primitiveBounds(asset, primitive);
        if (!primitiveBox.has_value())
        {
            continue;
        }

        bounds = bounds ? mergeBounds(*bounds, *primitiveBox) : primitiveBox;
    }

    return bounds;
}

namespace
{

// Gather a mesh's POSITION vertices + triangle indices across its supported primitives
// (re-basing each primitive's indices into the combined vertex list).
void gatherMeshGeometry(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh,
                        std::vector<Vec3>& positions, std::vector<std::uint32_t>& indices)
{
    for (const auto& primitive : mesh.primitives)
    {
        if (!GltfLoader::isSupportedPrimitiveType(primitive.type))
        {
            continue;
        }
        const auto* posAttr = primitive.findAttribute("POSITION");
        if (posAttr == primitive.attributes.end())
        {
            continue;
        }

        const auto base = static_cast<std::uint32_t>(positions.size());
        const auto& posAccessor = asset.accessors[posAttr->accessorIndex];
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
            asset, posAccessor, [&](fastgltf::math::fvec3 p, std::size_t)
            { positions.push_back({p.x(), p.y(), p.z()}); });

        if (primitive.indicesAccessor.has_value())
        {
            const auto& indexAccessor = asset.accessors[primitive.indicesAccessor.value()];
            fastgltf::iterateAccessor<std::uint32_t>(asset, indexAccessor, [&](std::uint32_t idx)
                                                     { indices.push_back(base + idx); });
        }
        else
        {
            for (std::size_t i = 0; i < posAccessor.count; ++i)
            {
                indices.push_back(base + static_cast<std::uint32_t>(i));
            }
        }
    }
}

} // namespace

ConvexHullShape GltfLoader::meshConvexHull(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh)
{
    std::vector<Vec3> positions;
    std::vector<std::uint32_t> indices;
    gatherMeshGeometry(asset, mesh, positions, indices);
    return buildConvexHull(positions, indices);
}

StaticMeshShape GltfLoader::meshTriangles(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh)
{
    StaticMeshShape result;
    gatherMeshGeometry(asset, mesh, result.vertices, result.indices);
    return result;
}

namespace
{

struct ImageSourceData
{
    std::optional<std::filesystem::path> path;
    std::vector<std::byte> bytes;
    std::string label;
};

ImageSourceData resolveImageSourceData(const fastgltf::Asset& asset, std::size_t imageIndex,
                                       const std::string& baseDir)
{
    const auto& image = asset.images[imageIndex];
    ImageSourceData result{.label = "image[" + std::to_string(imageIndex) + "]"};

    if (auto* uri = std::get_if<fastgltf::sources::URI>(&image.data);
        uri != nullptr && uri->uri.isLocalPath() && !uri->uri.isDataUri() &&
        uri->fileByteOffset == 0)
    {
        result.path = std::filesystem::path(baseDir) / uri->uri.fspath();
        return result;
    }

    result.bytes = loadDataSourceBytes(asset, image.data, baseDir, result.label);
    if (result.bytes.empty())
    {
        throw std::runtime_error("Image data is empty for " + result.label);
    }
    return result;
}

} // namespace

Image GltfLoader::loadImage(const fastgltf::Asset& asset, std::size_t imageIndex,
                            const std::string& baseDir)
{
    const ImageSourceData source = resolveImageSourceData(asset, imageIndex, baseDir);
    if (source.path.has_value())
    {
        return Image::load_from_file(source.path->string());
    }

    return Image::load_from_memory(reinterpret_cast<const uint8_t*>(source.bytes.data()),
                                   source.bytes.size(), source.label);
}

KtxImage GltfLoader::loadKtxImage(const fastgltf::Asset& asset, std::size_t imageIndex,
                                  const std::string& baseDir)
{
    const ImageSourceData source = resolveImageSourceData(asset, imageIndex, baseDir);
    if (source.path.has_value())
    {
        return KtxImage::load_from_file(source.path->string());
    }

    return KtxImage::load_from_memory(reinterpret_cast<const uint8_t*>(source.bytes.data()),
                                      source.bytes.size(), source.label);
}

namespace
{

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

const Texture* GltfLoader::resolveTextureIndex(const fastgltf::Asset& asset,
                                               std::size_t textureIndex, const std::string& baseDir,
                                               Resources& resources, Assets& assets,
                                               TextureEncoding encoding)
{
    auto& tex = assets.texture(textureIndex);
    if (!tex.loaded())
    {
        auto settings = extractSamplerSettings(asset, textureIndex);
        const auto& gltfTexture = asset.textures[textureIndex];

        if (gltfTexture.basisuImageIndex.has_value())
        {
            auto image = loadKtxImage(asset, gltfTexture.basisuImageIndex.value(), baseDir);
            tex = Texture::load_from_ktx_image(std::move(image), resources, settings, encoding);
        }
        else if (gltfTexture.imageIndex.has_value())
        {
            auto image = loadImage(asset, gltfTexture.imageIndex.value(), baseDir);
            tex = Texture::load_from_image(image, resources, settings, encoding);
        }
        else
        {
            throw std::runtime_error("Texture[" + std::to_string(textureIndex) +
                                     "] does not reference an image source");
        }
    }

    return &tex;
}

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
        // cross(b - a, c - a). Magnitude is 2 × triangle area, so the
        // un-normalised cross naturally area-weights the accumulation —
        // larger triangles dominate at shared vertices, which is what we want.
        const Vec3 e1{b.x() - a.x(), b.y() - a.y(), b.z() - a.z()};
        const Vec3 e2{c.x() - a.x(), c.y() - a.y(), c.z() - a.z()};
        const Vec3 face = Vec3::crossProduct(e1, e2);
        normals[i0] += face;
        normals[i1] += face;
        normals[i2] += face;
    }
    for (auto& n : normals)
    {
        if (n.magnitudeSquared() > 1e-16f)
        {
            n = Vec3::normalise(n);
        }
        else
        {
            // Vertex unreferenced by any triangle (or degenerate fan). The
            // up-pointing fallback is harmless because the vertex isn't drawn.
            n = Vec3{0.0f, 1.0f, 0.0f};
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

namespace
{
struct ResolvedMaterialTextures
{
    const Texture* baseColour{nullptr};
    const Texture* emissive{nullptr};
    const Texture* normal{nullptr};
    const Texture* metallicRoughness{nullptr};
    const Texture* occlusion{nullptr};
    const Texture* transmission{nullptr};
    const Texture* clearcoat{nullptr};
    const Texture* clearcoatRoughness{nullptr};
    const Texture* clearcoatNormal{nullptr};
    const Texture* thickness{nullptr};
};

UvTransform readUvTransform(const fastgltf::TextureInfo& info) noexcept
{
    UvTransform transform;
    if (info.transform)
    {
        const auto& source = *info.transform;
        transform.offsetX = static_cast<float>(source.uvOffset.x());
        transform.offsetY = static_cast<float>(source.uvOffset.y());
        transform.scaleX = static_cast<float>(source.uvScale.x());
        transform.scaleY = static_cast<float>(source.uvScale.y());
        transform.rotation = static_cast<float>(source.rotation);
    }
    return transform;
}

void applyTextureSlotUv(Material& material, MaterialTextureSlot slot,
                        const fastgltf::TextureInfo& info) noexcept
{
    material.texture(slot).texCoord = static_cast<int>(info.texCoordIndex);
    material.texture(slot).transform = readUvTransform(info);
}

void applyBaseMaterialFields(Material& material, const fastgltf::Material& gltfMat)
{
    material.name(std::string(gltfMat.name));

    const auto& pbr = gltfMat.pbrData;
    material.baseColor({static_cast<float>(pbr.baseColorFactor.x()),
                        static_cast<float>(pbr.baseColorFactor.y()),
                        static_cast<float>(pbr.baseColorFactor.z())});
    material.alpha(static_cast<float>(pbr.baseColorFactor.w()));
    material.metallic(static_cast<float>(pbr.metallicFactor));
    material.roughness(static_cast<float>(pbr.roughnessFactor));

    // KHR_materials_emissive_strength: multiplies emissiveFactor at load time.
    const float emissiveStrength = static_cast<float>(gltfMat.emissiveStrength);
    material.emissive({static_cast<float>(gltfMat.emissiveFactor.x()) * emissiveStrength,
                       static_cast<float>(gltfMat.emissiveFactor.y()) * emissiveStrength,
                       static_cast<float>(gltfMat.emissiveFactor.z()) * emissiveStrength});

    if (gltfMat.normalTexture.has_value())
    {
        material.normalScale(static_cast<float>(gltfMat.normalTexture.value().scale));
    }
    if (gltfMat.occlusionTexture.has_value())
    {
        material.occlusionStrength(static_cast<float>(gltfMat.occlusionTexture.value().strength));
    }
}

void applyBaseTextureSlotUv(Material& material, const fastgltf::Material& gltfMat)
{
    using Slot = MaterialTextureSlot;
    if (gltfMat.pbrData.baseColorTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::BaseColour, gltfMat.pbrData.baseColorTexture.value());
    }
    if (gltfMat.emissiveTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::Emissive, gltfMat.emissiveTexture.value());
    }
    if (gltfMat.normalTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::Normal, gltfMat.normalTexture.value());
    }
    if (gltfMat.pbrData.metallicRoughnessTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::MetallicRoughness,
                           gltfMat.pbrData.metallicRoughnessTexture.value());
    }
    if (gltfMat.occlusionTexture.has_value())
    {
        applyTextureSlotUv(material, Slot::Occlusion, gltfMat.occlusionTexture.value());
    }
}

void applyTransmission(Material& material, const fastgltf::Material& gltfMat)
{
    const float ior = static_cast<float>(gltfMat.ior);
    if (gltfMat.transmission != nullptr || ior != 1.5f)
    {
        TransmissionParams transmission;
        transmission.ior = ior;
        if (gltfMat.transmission != nullptr)
        {
            transmission.factor = static_cast<float>(gltfMat.transmission->transmissionFactor);
        }
        material.transmission(transmission);
    }
    if (gltfMat.transmission != nullptr && gltfMat.transmission->transmissionTexture.has_value())
    {
        applyTextureSlotUv(material, MaterialTextureSlot::Transmission,
                           gltfMat.transmission->transmissionTexture.value());
    }
}

void applyClearcoat(Material& material, const fastgltf::Material& gltfMat)
{
    if (gltfMat.clearcoat == nullptr)
    {
        return;
    }

    const auto& source = *gltfMat.clearcoat;
    ClearcoatParams clearcoat;
    clearcoat.factor = static_cast<float>(source.clearcoatFactor);
    clearcoat.roughness = static_cast<float>(source.clearcoatRoughnessFactor);
    if (source.clearcoatTexture.has_value())
    {
        applyTextureSlotUv(material, MaterialTextureSlot::Clearcoat,
                           source.clearcoatTexture.value());
    }
    if (source.clearcoatRoughnessTexture.has_value())
    {
        applyTextureSlotUv(material, MaterialTextureSlot::ClearcoatRoughness,
                           source.clearcoatRoughnessTexture.value());
    }
    if (source.clearcoatNormalTexture.has_value())
    {
        const auto& info = source.clearcoatNormalTexture.value();
        applyTextureSlotUv(material, MaterialTextureSlot::ClearcoatNormal, info);
        clearcoat.normalScale = static_cast<float>(info.scale);
    }
    material.clearcoat(clearcoat);
}

void applyVolume(Material& material, const fastgltf::Material& gltfMat)
{
    if (gltfMat.volume == nullptr)
    {
        return;
    }

    const auto& source = *gltfMat.volume;
    VolumeParams volume;
    volume.thicknessFactor = static_cast<float>(source.thicknessFactor);
    volume.attenuationColor = Colour3{static_cast<float>(source.attenuationColor.x()),
                                      static_cast<float>(source.attenuationColor.y()),
                                      static_cast<float>(source.attenuationColor.z())};
    volume.attenuationDistance = static_cast<float>(source.attenuationDistance);
    material.volume(volume);
    if (source.thicknessTexture.has_value())
    {
        applyTextureSlotUv(material, MaterialTextureSlot::Thickness,
                           source.thicknessTexture.value());
    }
}

void applyAlphaFields(Material& material, const fastgltf::Material& gltfMat) noexcept
{
    switch (gltfMat.alphaMode)
    {
    case fastgltf::AlphaMode::Opaque:
        material.alphaMode(AlphaMode::Opaque);
        break;
    case fastgltf::AlphaMode::Mask:
        material.alphaMode(AlphaMode::Mask);
        break;
    case fastgltf::AlphaMode::Blend:
        material.alphaMode(AlphaMode::Blend);
        break;
    }
    material.alphaCutoff(static_cast<float>(gltfMat.alphaCutoff));
    material.doubleSided(gltfMat.doubleSided);
}

bool materialNeedsTangents(const fastgltf::Asset& asset, std::optional<std::size_t> materialIndex)
{
    if (!materialIndex.has_value())
    {
        return false;
    }

    const auto& material = asset.materials[materialIndex.value()];
    return material.normalTexture.has_value() ||
           (material.clearcoat != nullptr &&
            material.clearcoat->clearcoatNormalTexture.has_value());
}

void warnSkippedTangentTexture(std::string_view meshName, std::size_t primIdx,
                               std::optional<std::size_t> variantIndex,
                               std::string_view textureName,
                               const TangentGenerationResult& tangentResult)
{
    std::cerr << "Warning: Skipping tangent-space " << textureName << " for " << meshName
              << " primitive " << primIdx;
    if (variantIndex.has_value())
    {
        std::cerr << " variant " << variantIndex.value();
    }
    std::cerr << ": " << tangentResult.reason << '\n';
}

void applyResolvedMaterialTextures(Material& material, const ResolvedMaterialTextures& textures,
                                   const TangentGenerationResult& tangentResult,
                                   std::string_view meshName, std::size_t primIdx,
                                   std::optional<std::size_t> variantIndex = std::nullopt)
{
    using Slot = MaterialTextureSlot;
    if (textures.baseColour != nullptr)
    {
        material.texture(Slot::BaseColour).texture = textures.baseColour;
    }
    if (textures.emissive != nullptr)
    {
        material.texture(Slot::Emissive).texture = textures.emissive;
    }
    if (textures.metallicRoughness != nullptr)
    {
        material.texture(Slot::MetallicRoughness).texture = textures.metallicRoughness;
    }
    if (textures.occlusion != nullptr)
    {
        material.texture(Slot::Occlusion).texture = textures.occlusion;
    }
    if (textures.transmission != nullptr)
    {
        material.texture(Slot::Transmission).texture = textures.transmission;
    }
    if (textures.clearcoat != nullptr)
    {
        material.texture(Slot::Clearcoat).texture = textures.clearcoat;
    }
    if (textures.clearcoatRoughness != nullptr)
    {
        material.texture(Slot::ClearcoatRoughness).texture = textures.clearcoatRoughness;
    }
    if (textures.thickness != nullptr)
    {
        material.texture(Slot::Thickness).texture = textures.thickness;
    }
    if (textures.normal != nullptr)
    {
        if (tangentResult.succeeded)
        {
            material.texture(Slot::Normal).texture = textures.normal;
        }
        else
        {
            warnSkippedTangentTexture(meshName, primIdx, variantIndex, "normal map", tangentResult);
        }
    }
    if (textures.clearcoatNormal != nullptr)
    {
        if (tangentResult.succeeded)
        {
            material.texture(Slot::ClearcoatNormal).texture = textures.clearcoatNormal;
        }
        else
        {
            warnSkippedTangentTexture(meshName, primIdx, variantIndex, "clearcoat normal map",
                                      tangentResult);
        }
    }
}

template <typename ResolveTexture>
ResolvedMaterialTextures resolveMaterialTextures(std::optional<std::size_t> materialIndex,
                                                 const fastgltf::Asset& asset,
                                                 ResolveTexture&& resolveTexture)
{
    ResolvedMaterialTextures result;
    if (!materialIndex.has_value())
    {
        return result;
    }

    const auto& material = asset.materials[materialIndex.value()];
    if (material.pbrData.baseColorTexture.has_value())
    {
        result.baseColour =
            resolveTexture(material.pbrData.baseColorTexture->textureIndex, TextureEncoding::Srgb);
    }
    if (material.emissiveTexture.has_value())
    {
        result.emissive =
            resolveTexture(material.emissiveTexture->textureIndex, TextureEncoding::Srgb);
    }
    if (material.normalTexture.has_value())
    {
        result.normal =
            resolveTexture(material.normalTexture->textureIndex, TextureEncoding::Linear);
    }
    if (material.pbrData.metallicRoughnessTexture.has_value())
    {
        result.metallicRoughness = resolveTexture(
            material.pbrData.metallicRoughnessTexture->textureIndex, TextureEncoding::Linear);
    }
    if (material.occlusionTexture.has_value())
    {
        result.occlusion =
            resolveTexture(material.occlusionTexture->textureIndex, TextureEncoding::Linear);
    }
    if (material.transmission != nullptr && material.transmission->transmissionTexture.has_value())
    {
        result.transmission = resolveTexture(
            material.transmission->transmissionTexture->textureIndex, TextureEncoding::Linear);
    }
    if (material.clearcoat != nullptr)
    {
        const auto& clearcoat = *material.clearcoat;
        if (clearcoat.clearcoatTexture.has_value())
        {
            result.clearcoat =
                resolveTexture(clearcoat.clearcoatTexture->textureIndex, TextureEncoding::Linear);
        }
        if (clearcoat.clearcoatRoughnessTexture.has_value())
        {
            result.clearcoatRoughness = resolveTexture(
                clearcoat.clearcoatRoughnessTexture->textureIndex, TextureEncoding::Linear);
        }
        if (clearcoat.clearcoatNormalTexture.has_value())
        {
            result.clearcoatNormal = resolveTexture(clearcoat.clearcoatNormalTexture->textureIndex,
                                                    TextureEncoding::Linear);
        }
    }
    if (material.volume != nullptr && material.volume->thicknessTexture.has_value())
    {
        result.thickness = resolveTexture(material.volume->thicknessTexture->textureIndex,
                                          TextureEncoding::Linear);
    }
    return result;
}
} // namespace

Object GltfLoader::loadMesh(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh,
                            const std::string& baseDir, Resources& resources, Assets& assets,
                            std::size_t meshIndex)
{
    std::size_t geoStartIdx = 0;
    for (std::size_t m = 0; m < meshIndex; ++m)
    {
        geoStartIdx += asset.meshes[m].primitives.size();
    }

    Object object;

    for (std::size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
    {
        const auto& primitive = mesh.primitives[primIdx];
        const auto baseMaterialIndex = primitive.materialIndex;
        auto materialData = loadMaterial(asset, baseMaterialIndex);
        const std::string meshName =
            mesh.name.empty() ? "mesh[" + std::to_string(meshIndex) + "]" : std::string(mesh.name);

        auto resolveTexture = [&](std::size_t textureIndex, TextureEncoding encoding)
        { return resolveTextureIndex(asset, textureIndex, baseDir, resources, assets, encoding); };
        const ResolvedMaterialTextures baseTextures =
            resolveMaterialTextures(baseMaterialIndex, asset, resolveTexture);

        std::size_t geoIdx = geoStartIdx + primIdx;
        bool needsTangents = materialNeedsTangents(asset, baseMaterialIndex);
        for (const auto& mappedMaterialIndex : primitive.mappings)
        {
            needsTangents = needsTangents || materialNeedsTangents(asset, mappedMaterialIndex);
        }
        auto tangentResult =
            loadGeometry(asset, primitive, needsTangents, resources, assets, geoIdx);

        applyResolvedMaterialTextures(materialData, baseTextures, tangentResult, meshName, primIdx);

        Material* matPtr = &assets.addMaterial(std::move(materialData));
        assets.geometry(geoIdx).material(matPtr);
        object.addGeometry(assets.geometry(geoIdx));

        for (std::size_t variantIndex = 0; variantIndex < primitive.mappings.size(); ++variantIndex)
        {
            const auto mappedMaterialIndex = primitive.mappings[variantIndex];
            if (!mappedMaterialIndex.has_value())
            {
                continue;
            }

            auto variantMaterial = loadMaterial(asset, mappedMaterialIndex);
            const ResolvedMaterialTextures variantTextures =
                resolveMaterialTextures(mappedMaterialIndex, asset, resolveTexture);
            applyResolvedMaterialTextures(variantMaterial, variantTextures, tangentResult, meshName,
                                          primIdx, variantIndex);

            Material* variantMatPtr = &assets.addMaterial(std::move(variantMaterial));
            object.addVariantMaterial(primIdx, variantIndex, variantMatPtr);
        }
    }

    object.load(resources);
    return object;
}

Material GltfLoader::loadMaterial(const fastgltf::Asset& asset,
                                  std::optional<std::size_t> materialIndex)
{
    Material material;
    if (!materialIndex.has_value())
    {
        return material;
    }

    const auto& gltfMat = asset.materials[materialIndex.value()];
    applyBaseMaterialFields(material, gltfMat);
    applyBaseTextureSlotUv(material, gltfMat);
    applyTransmission(material, gltfMat);
    applyClearcoat(material, gltfMat);
    applyVolume(material, gltfMat);
    material.unlit(gltfMat.unlit);
    applyAlphaFields(material, gltfMat);

    return material;
}

} // namespace fire_engine
