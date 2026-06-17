#include <fire_engine/core/gltf_loader.hpp>

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
}

// ---------------------------------------------------------------------------
// Asset parsing and setup
// ---------------------------------------------------------------------------

namespace
{

// Reads typed scalar/vector fields out of a node-extras object (`extras.Physics`
// or `extras.Cloth`), throwing a section-prefixed error on a type mismatch.
class ExtrasReader
{
public:
    ExtrasReader(simdjson::dom::object& object, std::string_view section) noexcept
        : object_{object},
          section_{section}
    {
    }

    [[nodiscard]] float readFloat(std::string_view key, float fallback,
                                  std::string_view label) const
    {
        auto element = object_.at_key(key);
        if (element.error() == simdjson::NO_SUCH_FIELD)
        {
            return fallback;
        }

        double value = 0.0;
        if (element.get(value) != simdjson::SUCCESS)
        {
            throw std::runtime_error(prefix(label) + " must be a number");
        }
        return static_cast<float>(value);
    }

    [[nodiscard]] std::uint32_t readUint(std::string_view key, std::uint32_t fallback,
                                         std::string_view label) const
    {
        auto element = object_.at_key(key);
        if (element.error() == simdjson::NO_SUCH_FIELD)
        {
            return fallback;
        }

        std::uint64_t value = 0;
        if (element.get(value) != simdjson::SUCCESS ||
            value > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error(prefix(label) + " must be an unsigned 32-bit integer");
        }
        return static_cast<std::uint32_t>(value);
    }

    [[nodiscard]] Vec3 readVec3(std::string_view key, Vec3 fallback, std::string_view label) const
    {
        auto element = object_.at_key(key);
        if (element.error() == simdjson::NO_SUCH_FIELD)
        {
            return fallback;
        }

        simdjson::dom::array array;
        if (element.get_array().get(array) != simdjson::SUCCESS)
        {
            throw std::runtime_error(prefix(label) + " must be an array of three numbers");
        }

        std::array<float, 3> values{};
        std::size_t i = 0;
        for (auto valueElement : array)
        {
            if (i >= values.size())
            {
                throw std::runtime_error(prefix(label) + " must contain exactly three numbers");
            }

            double value = 0.0;
            if (valueElement.get(value) != simdjson::SUCCESS)
            {
                throw std::runtime_error(prefix(label) + " must contain only numbers");
            }
            values[i] = static_cast<float>(value);
            ++i;
        }

        if (i != values.size())
        {
            throw std::runtime_error(prefix(label) + " must contain exactly three numbers");
        }

        return {values[0], values[1], values[2]};
    }

private:
    [[nodiscard]] std::string prefix(std::string_view label) const
    {
        return "glTF " + std::string(section_) + " " + std::string(label);
    }

    simdjson::dom::object& object_;
    std::string_view section_;
};

} // namespace

bool GltfLoader::nodeExtrasControllable(simdjson::dom::object* extras) noexcept
{
    if (extras == nullptr)
    {
        return false;
    }

    auto controllable = extras->at_key("Controllable");
    bool enabled = false;
    return controllable.get(enabled) == simdjson::SUCCESS && enabled;
}

std::optional<GltfLoader::PhysicsConfig>
GltfLoader::nodeExtrasPhysics(simdjson::dom::object* extras)
{
    if (extras == nullptr)
    {
        return std::nullopt;
    }

    simdjson::dom::object physicsObject;
    auto physicsElement = extras->at_key("Physics");
    if (physicsElement.error() == simdjson::NO_SUCH_FIELD)
    {
        return std::nullopt;
    }
    if (physicsElement.get_object().get(physicsObject) != simdjson::SUCCESS)
    {
        throw std::runtime_error("glTF Physics extras must be an object");
    }

    const ExtrasReader reader{physicsObject, "Physics"};
    PhysicsConfig config;
    auto bodyTypeElement = physicsObject.at_key("BodyType");
    if (bodyTypeElement.error() != simdjson::NO_SUCH_FIELD)
    {
        std::string_view bodyType;
        if (bodyTypeElement.get(bodyType) != simdjson::SUCCESS)
        {
            throw std::runtime_error("glTF Physics BodyType must be a string");
        }

        if (bodyType == "Static")
        {
            config.bodyType = PhysicsBodyType::Static;
        }
        else if (bodyType == "Kinematic")
        {
            config.bodyType = PhysicsBodyType::Kinematic;
        }
        else if (bodyType == "Dynamic")
        {
            config.bodyType = PhysicsBodyType::Dynamic;
        }
        else
        {
            throw std::runtime_error("glTF Physics BodyType must be Static, Kinematic, or Dynamic");
        }
    }

    config.layer = reader.readUint("Layer", config.layer, "Layer");
    config.mask = reader.readUint("Mask", config.mask, "Mask");
    config.velocity = reader.readVec3("Velocity", config.velocity, "Velocity");
    config.mass = reader.readFloat("Mass", config.mass, "Mass");
    config.restitution = reader.readFloat("Restitution", config.restitution, "Restitution");
    config.friction = reader.readFloat("Friction", config.friction, "Friction");
    config.gravityScale = reader.readFloat("GravityScale", config.gravityScale, "GravityScale");

    auto shapeElement = physicsObject.at_key("Shape");
    if (shapeElement.error() != simdjson::NO_SUCH_FIELD)
    {
        std::string_view shape;
        if (shapeElement.get(shape) != simdjson::SUCCESS)
        {
            throw std::runtime_error("glTF Physics Shape must be a string");
        }

        const Vec3 center = reader.readVec3("Center", {}, "Center");
        if (shape == "Box")
        {
            config.shape =
                BoxShape{reader.readVec3("HalfExtents", {0.5f, 0.5f, 0.5f}, "HalfExtents"), center};
        }
        else if (shape == "Sphere")
        {
            config.shape = SphereShape{reader.readFloat("Radius", 0.5f, "Radius"), center};
        }
        else if (shape == "Capsule")
        {
            config.shape = CapsuleShape{reader.readFloat("Radius", 0.5f, "Radius"),
                                        reader.readFloat("HalfHeight", 0.5f, "HalfHeight"), center};
        }
        else
        {
            throw std::runtime_error("glTF Physics Shape must be Box, Sphere, or Capsule");
        }
    }

    return config;
}

std::optional<ClothMeshParams> GltfLoader::nodeExtrasCloth(simdjson::dom::object* extras)
{
    if (extras == nullptr)
    {
        return std::nullopt;
    }

    simdjson::dom::object clothObject;
    auto clothElement = extras->at_key("Cloth");
    if (clothElement.error() == simdjson::NO_SUCH_FIELD)
    {
        return std::nullopt;
    }
    if (clothElement.get_object().get(clothObject) != simdjson::SUCCESS)
    {
        throw std::runtime_error("glTF Cloth extras must be an object");
    }

    const ExtrasReader reader{clothObject, "Cloth"};
    ClothMeshParams params;
    params.structuralCompliance =
        reader.readFloat("Compliance", params.structuralCompliance, "Compliance");
    params.bendCompliance =
        reader.readFloat("BendCompliance", params.bendCompliance, "BendCompliance");

    auto pinElement = clothObject.at_key("Pin");
    if (pinElement.error() != simdjson::NO_SUCH_FIELD)
    {
        std::string_view pin;
        if (pinElement.get(pin) != simdjson::SUCCESS)
        {
            throw std::runtime_error("glTF Cloth Pin must be a string");
        }
        if (pin == "None")
        {
            params.pin = ClothMeshParams::Pin::None;
        }
        else if (pin == "TopCorners")
        {
            params.pin = ClothMeshParams::Pin::TopCorners;
        }
        else if (pin == "TopEdge")
        {
            params.pin = ClothMeshParams::Pin::TopEdge;
        }
        else
        {
            throw std::runtime_error("glTF Cloth Pin must be None, TopCorners, or TopEdge");
        }
    }

    return params;
}

fastgltf::Expected<fastgltf::Asset>
GltfLoader::parseAsset(const std::filesystem::path& gltfPath,
                       std::unordered_set<std::size_t>* controllableNodeIndices,
                       std::unordered_map<std::size_t, PhysicsConfig>* physicsNodeConfigs,
                       std::unordered_map<std::size_t, ClothMeshParams>* clothNodeConfigs)
{
    // fastgltf only parses extension data when the extension is enabled here.
    // Without the opt-in, extension fields silently stay at their defaults.
    fastgltf::Parser parser(supportedExtensionMask());
    ExtrasParseState extrasState{controllableNodeIndices, physicsNodeConfigs, clothNodeConfigs};
    if (controllableNodeIndices != nullptr || physicsNodeConfigs != nullptr ||
        clothNodeConfigs != nullptr)
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
