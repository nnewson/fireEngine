#include <fire_engine/core/gltf_loader.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <simdjson.h>

#include <fire_engine/graphics/assets.hpp>

namespace fire_engine
{
namespace
{
struct SupportedExtension
{
    std::string_view name;
    fastgltf::Extensions flag;
};

constexpr auto kSupportedExtensions = std::to_array<SupportedExtension>({
    {std::string_view{"KHR_materials_emissive_strength"},
     fastgltf::Extensions::KHR_materials_emissive_strength},
    {std::string_view{"KHR_texture_transform"}, fastgltf::Extensions::KHR_texture_transform},
    {std::string_view{"KHR_texture_basisu"}, fastgltf::Extensions::KHR_texture_basisu},
    {std::string_view{"KHR_materials_variants"}, fastgltf::Extensions::KHR_materials_variants},
    {std::string_view{"KHR_materials_unlit"}, fastgltf::Extensions::KHR_materials_unlit},
    {std::string_view{"KHR_lights_punctual"}, fastgltf::Extensions::KHR_lights_punctual},
    {std::string_view{"KHR_materials_transmission"},
     fastgltf::Extensions::KHR_materials_transmission},
    {std::string_view{"KHR_materials_ior"}, fastgltf::Extensions::KHR_materials_ior},
    {std::string_view{"KHR_materials_clearcoat"}, fastgltf::Extensions::KHR_materials_clearcoat},
    {std::string_view{"KHR_materials_volume"}, fastgltf::Extensions::KHR_materials_volume},
});

constexpr fastgltf::Extensions supportedExtensionMask() noexcept
{
    fastgltf::Extensions mask = fastgltf::Extensions::None;
    for (const auto& extension : kSupportedExtensions)
    {
        mask |= extension.flag;
    }
    return mask;
}

struct ExtrasParseState
{
    std::unordered_set<std::size_t>* controllableNodeIndices{nullptr};
    std::unordered_map<std::size_t, GltfLoader::PhysicsConfig>* physicsNodeConfigs{nullptr};
    std::unordered_map<std::size_t, ClothMeshParams>* clothNodeConfigs{nullptr};
    std::unordered_map<std::size_t, RagdollParams>* ragdollNodeConfigs{nullptr};
};
} // namespace

void GltfLoader::ensureSupportedExtensions(std::span<const std::string_view> required)
{
    std::vector<std::string_view> unsupported;
    for (const auto& ext : required)
    {
        bool found = false;
        for (const auto& supported : kSupportedExtensions)
        {
            if (ext == supported.name)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            unsupported.push_back(ext);
        }
    }
    if (unsupported.empty())
    {
        return;
    }
    std::string msg = "glTF asset requires unsupported extension(s):";
    for (const auto& ext : unsupported)
    {
        msg += " ";
        msg.append(ext);
    }
    throw std::runtime_error(msg);
}

void parseNodeExtras(simdjson::dom::object* extras, std::size_t objectIndex,
                     fastgltf::Category objectType, void* userPointer)
{
    if (objectType != fastgltf::Category::Nodes || userPointer == nullptr)
    {
        return;
    }

    auto* state = static_cast<ExtrasParseState*>(userPointer);
    if (state->controllableNodeIndices != nullptr && GltfLoader::nodeExtrasControllable(extras))
    {
        state->controllableNodeIndices->insert(objectIndex);
    }

    if (state->physicsNodeConfigs != nullptr)
    {
        auto physics = GltfLoader::nodeExtrasPhysics(extras);
        if (physics.has_value())
        {
            state->physicsNodeConfigs->insert_or_assign(objectIndex, physics.value());
        }
    }

    if (state->clothNodeConfigs != nullptr)
    {
        auto cloth = GltfLoader::nodeExtrasCloth(extras);
        if (cloth.has_value())
        {
            state->clothNodeConfigs->insert_or_assign(objectIndex, cloth.value());
        }
    }

    if (state->ragdollNodeConfigs != nullptr)
    {
        auto ragdoll = GltfLoader::nodeExtrasRagdoll(extras);
        if (ragdoll.has_value())
        {
            state->ragdollNodeConfigs->insert_or_assign(objectIndex, ragdoll.value());
        }
    }
}

// ---------------------------------------------------------------------------
// Asset parsing and setup
// ---------------------------------------------------------------------------

fastgltf::Expected<fastgltf::Asset>
GltfLoader::parseAsset(const std::filesystem::path& gltfPath,
                       std::unordered_set<std::size_t>* controllableNodeIndices,
                       std::unordered_map<std::size_t, PhysicsConfig>* physicsNodeConfigs,
                       std::unordered_map<std::size_t, ClothMeshParams>* clothNodeConfigs,
                       std::unordered_map<std::size_t, RagdollParams>* ragdollNodeConfigs)
{
    // fastgltf only parses extension data when the extension is enabled here.
    // Without the opt-in, extension fields silently stay at their defaults.
    fastgltf::Parser parser(supportedExtensionMask());
    ExtrasParseState extrasState{controllableNodeIndices, physicsNodeConfigs, clothNodeConfigs,
                                 ragdollNodeConfigs};
    if (controllableNodeIndices != nullptr || physicsNodeConfigs != nullptr ||
        clothNodeConfigs != nullptr || ragdollNodeConfigs != nullptr)
    {
        if (controllableNodeIndices != nullptr)
        {
            controllableNodeIndices->clear();
        }
        if (physicsNodeConfigs != nullptr)
        {
            physicsNodeConfigs->clear();
        }
        if (clothNodeConfigs != nullptr)
        {
            clothNodeConfigs->clear();
        }
        if (ragdollNodeConfigs != nullptr)
        {
            ragdollNodeConfigs->clear();
        }
        parser.setUserPointer(&extrasState);
        parser.setExtrasParseCallback(&parseNodeExtras);
    }

    auto dataResult = fastgltf::GltfDataBuffer::FromPath(gltfPath);
    if (dataResult.error() != fastgltf::Error::None)
    {
        throw std::runtime_error("Failed to read glTF file: " +
                                 std::string(fastgltf::getErrorMessage(dataResult.error())));
    }

    auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages;
    auto result = parser.loadGltf(dataResult.get(), gltfPath.parent_path(), options);
    if (result.error() != fastgltf::Error::None)
    {
        throw std::runtime_error("Failed to load glTF: " +
                                 std::string(fastgltf::getErrorMessage(result.error())));
    }

    return result;
}

void GltfLoader::presizeAssets(const fastgltf::Asset& asset, Assets& assets)
{
    assets.resizeTextures(std::max<std::size_t>(asset.textures.size(), 1));
    assets.resizeMaterials(std::max<std::size_t>(asset.materials.size(), 1));

    std::size_t totalPrimitives = 0;
    for (const auto& m : asset.meshes)
    {
        totalPrimitives += m.primitives.size();
    }
    assets.resizeGeometries(totalPrimitives);
    assets.resizeSkins(asset.skins.size());

    // Count (glTF animation, node) pairs for Animation slots
    std::size_t animSlotCount = 0;
    for (std::size_t ai = 0; ai < asset.animations.size(); ++ai)
    {
        std::unordered_set<std::size_t> nodesInAnim;
        for (const auto& channel : asset.animations[ai].channels)
        {
            if (channel.nodeIndex.has_value())
            {
                nodesInAnim.insert(channel.nodeIndex.value());
            }
        }
        animSlotCount += nodesInAnim.size();
    }
    assets.resizeAnimations(animSlotCount);
}

} // namespace fire_engine
