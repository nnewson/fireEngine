#include <fire_engine/core/gltf_loader.hpp>

#include <fire_engine/render/resources.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/image.hpp>
#include <fire_engine/graphics/ktx_image.hpp>
#include <fire_engine/graphics/sampler_settings.hpp>
#include <fire_engine/graphics/texture.hpp>

namespace fire_engine
{
namespace
{

WrapMode toWrapMode(fastgltf::Wrap wrap)
{
    switch (wrap)
    {
    case fastgltf::Wrap::MirroredRepeat:
        return WrapMode::MirroredRepeat;
    case fastgltf::Wrap::ClampToEdge:
        return WrapMode::ClampToEdge;
    default:
        return WrapMode::Repeat;
    }
}

FilterMode toFilterMode(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return FilterMode::Nearest;
    default:
        return FilterMode::Linear;
    }
}

SamplerSettings extractSamplerSettings(const fastgltf::Asset& asset, std::size_t textureIndex)
{
    SamplerSettings settings;
    const auto& texture = asset.textures[textureIndex];
    if (texture.samplerIndex.has_value())
    {
        const auto& sampler = asset.samplers[texture.samplerIndex.value()];
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

std::vector<std::byte> readFileBytes(const std::filesystem::path& path)
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

std::vector<std::byte> sliceBytes(std::span<const std::byte> bytes, std::size_t offset,
                                  std::size_t length, const std::string& label)
{
    if (offset > bytes.size() || offset + length > bytes.size())
    {
        throw std::runtime_error("Out-of-range byte slice for " + label);
    }

    return {bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + length)};
}

std::vector<std::byte> loadDataSourceBytes(const fastgltf::Asset& asset,
                                           const fastgltf::DataSource& source,
                                           const std::string& baseDir, const std::string& label)
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

const Texture* GltfLoader::resolveTextureIndex(const fastgltf::Asset& asset,
                                               std::size_t textureIndex, const std::string& baseDir,
                                               Resources& resources, Assets& assets,
                                               TextureEncoding encoding)
{
    auto& texture = assets.texture(textureIndex);
    if (!texture.loaded())
    {
        auto settings = extractSamplerSettings(asset, textureIndex);
        const auto& gltfTexture = asset.textures[textureIndex];

        if (gltfTexture.basisuImageIndex.has_value())
        {
            auto image = loadKtxImage(asset, gltfTexture.basisuImageIndex.value(), baseDir);
            texture = Texture::load_from_ktx_image(std::move(image), resources, settings, encoding);
        }
        else if (gltfTexture.imageIndex.has_value())
        {
            auto image = loadImage(asset, gltfTexture.imageIndex.value(), baseDir);
            texture = Texture::load_from_image(image, resources, settings, encoding);
        }
        else
        {
            throw std::runtime_error("Texture[" + std::to_string(textureIndex) +
                                     "] does not reference an image source");
        }
    }

    return &texture;
}

} // namespace fire_engine
