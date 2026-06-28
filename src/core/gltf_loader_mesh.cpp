#include <fire_engine/core/gltf_loader.hpp>

#include <fire_engine/core/convex_hull_builder.hpp>
#include <fire_engine/render/resources.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/object.hpp>
#include <fire_engine/graphics/texture.hpp>

namespace fire_engine
{
bool GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType type) noexcept
{
    return type == fastgltf::PrimitiveType::Triangles;
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

std::size_t firstGeometryIndexForMesh(const fastgltf::Asset& asset, std::size_t meshIndex)
{
    std::size_t geometryIndex = 0;
    for (std::size_t i = 0; i < meshIndex; ++i)
    {
        geometryIndex += asset.meshes[i].primitives.size();
    }
    return geometryIndex;
}

std::string meshDisplayName(const fastgltf::Mesh& mesh, std::size_t meshIndex)
{
    return mesh.name.empty() ? "mesh[" + std::to_string(meshIndex) + "]" : std::string(mesh.name);
}

bool primitiveNeedsTangents(const fastgltf::Asset& asset, const fastgltf::Primitive& primitive)
{
    bool needsTangents = materialNeedsTangents(asset, primitive.materialIndex);
    for (const auto& mappedMaterialIndex : primitive.mappings)
    {
        needsTangents = needsTangents || materialNeedsTangents(asset, mappedMaterialIndex);
    }
    return needsTangents;
}
} // namespace

Object GltfLoader::loadMesh(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh,
                            const std::string& baseDir, Resources& resources, Assets& assets,
                            std::size_t meshIndex)
{
    Object object;
    const std::size_t geoStartIdx = firstGeometryIndexForMesh(asset, meshIndex);
    const std::string meshName = meshDisplayName(mesh, meshIndex);

    auto resolveTexture = [&](std::size_t textureIndex, TextureEncoding encoding)
    { return resolveTextureIndex(asset, textureIndex, baseDir, resources, assets, encoding); };

    for (std::size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
    {
        const auto& primitive = mesh.primitives[primIdx];
        const auto baseMaterialIndex = primitive.materialIndex;
        std::size_t geoIdx = geoStartIdx + primIdx;

        auto tangentResult = loadGeometry(
            asset, primitive, primitiveNeedsTangents(asset, primitive), resources, assets, geoIdx);

        auto loadMaterialWithTextures = [&](std::optional<std::size_t> materialIndex,
                                            std::optional<std::size_t> variantIndex = std::nullopt)
        {
            auto material = loadMaterial(asset, materialIndex);
            const ResolvedMaterialTextures textures =
                resolveMaterialTextures(materialIndex, asset, resolveTexture);
            applyResolvedMaterialTextures(material, textures, tangentResult, meshName, primIdx,
                                          variantIndex);
            return material;
        };

        auto materialData = loadMaterialWithTextures(baseMaterialIndex);
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

            auto variantMaterial = loadMaterialWithTextures(mappedMaterialIndex, variantIndex);
            Material* variantMatPtr = &assets.addMaterial(std::move(variantMaterial));
            object.addVariantMaterial(primIdx, variantIndex, variantMatPtr);
        }
    }

    object.load(resources);
    return object;
}

} // namespace fire_engine
