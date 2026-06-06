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
};
} // namespace

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

// ---------------------------------------------------------------------------
// Sampler helpers
// ---------------------------------------------------------------------------

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

static std::vector<std::byte> sliceBytes(const std::vector<std::byte>& bytes, std::size_t offset,
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

static std::size_t countMorphTargets(const fastgltf::Mesh& mesh)
{
    if (mesh.primitives.empty() || mesh.primitives[0].targets.empty())
    {
        return 0;
    }
    return mesh.primitives[0].targets.size();
}

static std::vector<float> initialMorphWeights(const fastgltf::Mesh& mesh,
                                              std::size_t numMorphTargets)
{
    std::vector<float> weights(numMorphTargets, 0.0f);
    for (std::size_t w = 0; w < mesh.weights.size() && w < numMorphTargets; ++w)
    {
        weights[w] = mesh.weights[w];
    }
    return weights;
}

static Light::Type toLightType(fastgltf::LightType t) noexcept
{
    switch (t)
    {
    case fastgltf::LightType::Point:
        return Light::Type::Point;
    case fastgltf::LightType::Spot:
        return Light::Type::Spot;
    case fastgltf::LightType::Directional:
    default:
        return Light::Type::Directional;
    }
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
}

void applyControllable(std::size_t nodeIndex,
                       const std::unordered_set<std::size_t>& controllableNodeIndices, Node& node)
{
    if (controllableNodeIndices.contains(nodeIndex))
    {
        node.emplaceControllable();
    }
}

void applyPhysicsConfig(std::size_t nodeIndex,
                        const std::unordered_map<std::size_t, GltfLoader::PhysicsConfig>& configs,
                        const fastgltf::Asset& asset, const fastgltf::Mesh& mesh, Node& node,
                        PhysicsWorld& physics)
{
    auto it = configs.find(nodeIndex);
    if (it == configs.end())
    {
        return;
    }

    Transform transform = node.transform();
    transform.update(Mat4::identity());

    PhysicsBodyDesc bodyDesc;
    bodyDesc.type = it->second.bodyType;
    bodyDesc.position = transform.position();
    bodyDesc.rotation = transform.rotation();
    bodyDesc.scale = transform.scale();
    bodyDesc.linearVelocity = it->second.velocity;
    bodyDesc.mass = it->second.mass;
    bodyDesc.gravityScale = it->second.gravityScale;
    bodyDesc.material = PhysicsMaterial{it->second.restitution, it->second.friction};

    PhysicsBodyHandle bodyHandle = physics.createBody(bodyDesc);
    node.physicsBodyHandle(bodyHandle);

    ColliderDesc colliderDesc;
    colliderDesc.collisionLayer = it->second.layer;
    colliderDesc.collisionMask = it->second.mask;
    colliderDesc.material = bodyDesc.material;
    if (it->second.shape.has_value())
    {
        colliderDesc.shape = it->second.shape.value();
    }
    else if (auto bounds = GltfLoader::meshBounds(asset, mesh); bounds.has_value())
    {
        colliderDesc.shape = AabbShape{bounds.value()};
    }
    node.physicsColliderHandle(physics.createCollider(bodyHandle, colliderDesc));
}

void validatePhysicsTarget(
    std::size_t nodeIndex, const std::unordered_set<std::size_t>& controllableNodeIndices,
    const std::unordered_map<std::size_t, GltfLoader::PhysicsConfig>& physicsNodeConfigs,
    const fastgltf::Node& gltfNode)
{
    const auto config = physicsNodeConfigs.find(nodeIndex);
    if (config == physicsNodeConfigs.end())
    {
        return;
    }

    if (config->second.bodyType == PhysicsBodyType::Dynamic &&
        controllableNodeIndices.contains(nodeIndex))
    {
        throw std::runtime_error("glTF node '" + std::string(gltfNode.name) +
                                 "' cannot be both dynamic physics and Controllable");
    }
    if (!gltfNode.meshIndex.has_value())
    {
        throw std::runtime_error("glTF node '" + std::string(gltfNode.name) +
                                 "' has Physics extras but no mesh");
    }
}

// KHR_lights_punctual: attach a Light component to `node` if the glTF node
// references a light. Only fires when the variant is still Empty — Light has
// to share the Components variant slot, so a node already carrying a Mesh /
// Animator / Camera skips with a warning. Cone angles only matter for Spot.
static void applyLight(const fastgltf::Asset& asset, const fastgltf::Node& gltfNode, Node& node)
{
    if (!gltfNode.lightIndex.has_value())
    {
        return;
    }
    if (!std::holds_alternative<Empty>(node.component()))
    {
        std::clog << "glTF: skipping KHR_lights_punctual on node '" << node.name()
                  << "' — node already has a non-Empty component (mesh/animator)\n";
        return;
    }
    const auto& gl = asset.lights[gltfNode.lightIndex.value()];
    auto& light = node.component().emplace<Light>();
    light.type(toLightType(gl.type));
    light.colour(Colour3{gl.color.x(), gl.color.y(), gl.color.z()});
    light.intensity(static_cast<float>(gl.intensity));
    if (gl.range.has_value())
    {
        light.range(static_cast<float>(gl.range.value()));
    }
    if (gl.outerConeAngle.has_value())
    {
        light.outerConeRad(static_cast<float>(gl.outerConeAngle.value()));
    }
    if (gl.innerConeAngle.has_value())
    {
        light.innerConeRad(static_cast<float>(gl.innerConeAngle.value()));
    }
}

// ---------------------------------------------------------------------------
// Node helpers
// ---------------------------------------------------------------------------

void GltfLoader::applyTRS(const fastgltf::Node& gltfNode, Node& node)
{
    if (auto* trs = std::get_if<fastgltf::TRS>(&gltfNode.transform))
    {
        node.transform().position(
            {trs->translation.x(), trs->translation.y(), trs->translation.z()});
        node.transform().rotation(
            {trs->rotation.x(), trs->rotation.y(), trs->rotation.z(), trs->rotation.w()});
        node.transform().scale({trs->scale.x(), trs->scale.y(), trs->scale.z()});
    }
    else if (auto* mat = std::get_if<fastgltf::math::fmat4x4>(&gltfNode.transform))
    {
        fastgltf::math::fvec3 scale;
        fastgltf::math::fquat rotation;
        fastgltf::math::fvec3 translation;
        fastgltf::math::decomposeTransformMatrix(*mat, scale, rotation, translation);

        node.transform().position({translation.x(), translation.y(), translation.z()});
        node.transform().rotation({rotation.x(), rotation.y(), rotation.z(), rotation.w()});
        node.transform().scale({scale.x(), scale.y(), scale.z()});
    }
}

void GltfLoader::applyRestTRS(const fastgltf::Node& gltfNode, Animation& anim)
{
    fastgltf::math::fvec3 t{0.0f, 0.0f, 0.0f};
    fastgltf::math::fquat r{0.0f, 0.0f, 0.0f, 1.0f};
    fastgltf::math::fvec3 s{1.0f, 1.0f, 1.0f};

    if (auto* trs = std::get_if<fastgltf::TRS>(&gltfNode.transform))
    {
        t = trs->translation;
        r = trs->rotation;
        s = trs->scale;
    }
    else if (auto* mat = std::get_if<fastgltf::math::fmat4x4>(&gltfNode.transform))
    {
        fastgltf::math::decomposeTransformMatrix(*mat, s, r, t);
    }

    if (anim.translationKeyframes().empty())
    {
        anim.translationKeyframes({{0.0f, {t.x(), t.y(), t.z()}}});
    }
    if (anim.rotationKeyframes().empty())
    {
        anim.rotationKeyframes({{0.0f, r.x(), r.y(), r.z(), r.w()}});
    }
    if (anim.scaleKeyframes().empty())
    {
        anim.scaleKeyframes({{0.0f, {s.x(), s.y(), s.z()}}});
    }
}

std::string GltfLoader::descendantMeshName(const fastgltf::Asset& asset,
                                           const fastgltf::Node& gltfNode)
{
    if (gltfNode.meshIndex.has_value())
    {
        const auto& name = asset.meshes[gltfNode.meshIndex.value()].name;
        if (!name.empty())
        {
            return std::string(name);
        }
    }
    for (auto childIndex : gltfNode.children)
    {
        auto result = descendantMeshName(asset, asset.nodes[childIndex]);
        if (!result.empty())
        {
            return result;
        }
    }
    return {};
}

std::string GltfLoader::nodeName(const fastgltf::Asset& asset, const fastgltf::Node& gltfNode)
{
    if (!gltfNode.name.empty())
    {
        return std::string(gltfNode.name);
    }
    auto meshName = descendantMeshName(asset, gltfNode);
    if (!meshName.empty())
    {
        return meshName;
    }
    return "Node";
}

Node& GltfLoader::attachCamera(Node& node, Node*& activeCamera)
{
    auto configureCamera = [](Camera& camera)
    {
        camera.localPosition({0.0f, 0.0f, 0.0f});
        camera.localYaw(-pi / 2.0f);
        camera.localPitch(0.0f);
    };

    Node* cameraNode = &node;
    if (std::holds_alternative<Empty>(node.component()))
    {
        configureCamera(node.component().emplace<Camera>());
    }
    else
    {
        auto child = std::make_unique<Node>(node.name() + "_Camera");
        cameraNode = &node.addChild(std::move(child));
        configureCamera(cameraNode->component().emplace<Camera>());
    }

    if (activeCamera == nullptr)
    {
        activeCamera = cameraNode;
    }

    return *cameraNode;
}

// ---------------------------------------------------------------------------
// Asset parsing and setup
// ---------------------------------------------------------------------------

namespace
{

class PhysicsExtrasReader
{
public:
    explicit PhysicsExtrasReader(simdjson::dom::object& object) noexcept
        : object_{object}
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
            throw std::runtime_error("glTF Physics " + std::string(label) + " must be a number");
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
            throw std::runtime_error("glTF Physics " + std::string(label) +
                                     " must be an unsigned 32-bit integer");
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
            throw std::runtime_error("glTF Physics " + std::string(label) +
                                     " must be an array of three numbers");
        }

        std::array<float, 3> values{};
        std::size_t i = 0;
        for (auto valueElement : array)
        {
            if (i >= values.size())
            {
                throw std::runtime_error("glTF Physics " + std::string(label) +
                                         " must contain exactly three numbers");
            }

            double value = 0.0;
            if (valueElement.get(value) != simdjson::SUCCESS)
            {
                throw std::runtime_error("glTF Physics " + std::string(label) +
                                         " must contain only numbers");
            }
            values[i] = static_cast<float>(value);
            ++i;
        }

        if (i != values.size())
        {
            throw std::runtime_error("glTF Physics " + std::string(label) +
                                     " must contain exactly three numbers");
        }

        return {values[0], values[1], values[2]};
    }

private:
    simdjson::dom::object& object_;
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

    const PhysicsExtrasReader reader{physicsObject};
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

fastgltf::Expected<fastgltf::Asset>
GltfLoader::parseAsset(const std::filesystem::path& gltfPath,
                       std::unordered_set<std::size_t>* controllableNodeIndices,
                       std::unordered_map<std::size_t, PhysicsConfig>* physicsNodeConfigs)
{
    // fastgltf only parses extension data when the extension is enabled here.
    // Without the opt-in, extension fields silently stay at their defaults.
    fastgltf::Parser parser(supportedExtensionMask());
    ExtrasParseState extrasState{controllableNodeIndices, physicsNodeConfigs};
    if (controllableNodeIndices != nullptr || physicsNodeConfigs != nullptr)
    {
        if (controllableNodeIndices != nullptr)
        {
            controllableNodeIndices->clear();
        }
        if (physicsNodeConfigs != nullptr)
        {
            physicsNodeConfigs->clear();
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

// ---------------------------------------------------------------------------
// Scene and node loading
// ---------------------------------------------------------------------------

Node* GltfLoader::loadScene(const std::string& path, SceneGraph& scene, Resources& resources,
                            Assets& assets, PhysicsWorld& physics)
{
    auto gltfPath = std::filesystem::path(path);
    std::unordered_set<std::size_t> controllableNodeIndices;
    std::unordered_map<std::size_t, PhysicsConfig> physicsNodeConfigs;
    auto result = parseAsset(gltfPath, &controllableNodeIndices, &physicsNodeConfigs);
    auto& asset = result.get();

    // fastgltf stores extensionsRequired in a pmr-allocated string vector.
    // Lift to string_view for the helper's portable signature.
    std::vector<std::string_view> requiredViews;
    requiredViews.reserve(asset.extensionsRequired.size());
    for (const auto& ext : asset.extensionsRequired)
    {
        requiredViews.emplace_back(ext);
    }
    ensureSupportedExtensions(requiredViews);

    presizeAssets(asset, assets);

    std::size_t sceneIndex = asset.defaultScene.has_value() ? asset.defaultScene.value() : 0;
    if (sceneIndex >= asset.scenes.size())
    {
        throw std::runtime_error("glTF scene index out of range");
    }

    const auto& gltfScene = asset.scenes[sceneIndex];
    std::string baseDir = gltfPath.parent_path().string();
    NodeMap nodeMap;
    MeshMap meshMap;
    std::size_t nextAnimSlot = 0;
    Node* activeCamera = nullptr;
    for (auto nodeIndex : gltfScene.nodeIndices)
    {
        auto rootNode = std::make_unique<Node>(nodeName(asset, asset.nodes[nodeIndex]));
        auto& rootRef = scene.addNode(std::move(rootNode));
        nodeMap[nodeIndex] = &rootRef;
        applyControllable(nodeIndex, controllableNodeIndices, rootRef);

        if (nodeHasAnimation(asset, nodeIndex))
        {
            configureAnimatedNode(asset, nodeIndex, rootRef, baseDir, resources, assets, nodeMap,
                                  meshMap, nextAnimSlot, activeCamera, controllableNodeIndices,
                                  physicsNodeConfigs, physics);
        }
        else
        {
            loadNode(asset, nodeIndex, rootRef, baseDir, resources, assets, nodeMap, meshMap,
                     nextAnimSlot, activeCamera, controllableNodeIndices, physicsNodeConfigs,
                     physics);
        }
    }

    // Resolve skins after the full scene graph is built
    applySkins(asset, nodeMap, meshMap, assets);

    return activeCamera;
}

Mesh& GltfLoader::attachMeshToNode(
    const fastgltf::Asset& asset, std::size_t nodeIndex, std::size_t meshIndex, Node& meshNode,
    Node& physicsNode, const std::string& baseDir, Resources& resources, Assets& assets,
    MeshMap& meshMap, const std::unordered_map<std::size_t, PhysicsConfig>& physicsNodeConfigs,
    PhysicsWorld& physics)
{
    const auto& gltfMesh = asset.meshes[meshIndex];
    auto object = loadMesh(asset, gltfMesh, baseDir, resources, assets, meshIndex);
    meshNode.component().emplace<Mesh>(std::move(object));

    auto& mesh = std::get<Mesh>(meshNode.component());
    mesh.variantNames(asset.materialVariants);
    applyPhysicsConfig(nodeIndex, physicsNodeConfigs, asset, gltfMesh, physicsNode, physics);
    meshMap[nodeIndex] = &mesh;
    return mesh;
}

void GltfLoader::configureAnimatedNode(
    const fastgltf::Asset& asset, std::size_t nodeIndex, Node& node, const std::string& baseDir,
    Resources& resources, Assets& assets, NodeMap& nodeMap, MeshMap& meshMap,
    std::size_t& nextAnimSlot, Node*& activeCamera,
    const std::unordered_set<std::size_t>& controllableNodeIndices,
    const std::unordered_map<std::size_t, PhysicsConfig>& physicsNodeConfigs, PhysicsWorld& physics)
{
    const auto& gltfNode = asset.nodes[nodeIndex];
    validatePhysicsTarget(nodeIndex, controllableNodeIndices, physicsNodeConfigs, gltfNode);

    // Determine morph target count from the mesh (if any)
    std::size_t numMorphTargets = 0;
    if (gltfNode.meshIndex.has_value())
    {
        const auto& gltfMesh = asset.meshes[gltfNode.meshIndex.value()];
        numMorphTargets = countMorphTargets(gltfMesh);
    }

    // Check if this node has transform vs weight animation channels
    bool hasTransformAnim = false;
    bool hasWeightAnim = false;
    for (const auto& anim : asset.animations)
    {
        for (const auto& channel : anim.channels)
        {
            if (!channel.nodeIndex.has_value() || channel.nodeIndex.value() != nodeIndex)
            {
                continue;
            }
            if (channel.path == fastgltf::AnimationPath::Rotation ||
                channel.path == fastgltf::AnimationPath::Translation ||
                channel.path == fastgltf::AnimationPath::Scale)
            {
                hasTransformAnim = true;
            }
            else if (channel.path == fastgltf::AnimationPath::Weights)
            {
                hasWeightAnim = true;
            }
        }
    }

    // Only apply rest TRS when animation won't drive it. glTF animation channels
    // replace (not compose with) the node's base TRS, so applying rest on top of
    // the animator's sampled matrix double-transforms the node.
    if (!hasTransformAnim)
    {
        applyTRS(gltfNode, node);
    }

    // KHR_lights_punctual on this node: only attaches if the node won't
    // otherwise be holding a Mesh / Animator (helper guards via variant
    // alternative check). Animated nodes lose the light with a warning —
    // rare in practice and worth documenting once, not redesigning for.
    applyLight(asset, gltfNode, node);

    // Load each glTF animation as a separate Animation object for this node
    std::vector<std::pair<std::size_t, Animation*>> nodeAnimations;
    for (std::size_t ai = 0; ai < asset.animations.size(); ++ai)
    {
        bool touchesNode = false;
        for (const auto& channel : asset.animations[ai].channels)
        {
            if (channel.nodeIndex.has_value() && channel.nodeIndex.value() == nodeIndex)
            {
                touchesNode = true;
                break;
            }
        }
        if (!touchesNode)
        {
            continue;
        }

        auto& la = assets.animation(nextAnimSlot);
        ++nextAnimSlot;
        loadAnimation(asset, ai, nodeIndex, la, numMorphTargets);
        applyRestTRS(gltfNode, la);
        la.name(std::string(asset.animations[ai].name));
        nodeAnimations.push_back({ai, &la});
    }

    if (hasTransformAnim)
    {
        // Node gets an Animator for transform; mesh goes on a child node
        node.component().emplace<Animator>();
        auto& animator = std::get<Animator>(node.component());
        for (const auto& [animId, anim] : nodeAnimations)
        {
            animator.addAnimation(animId, anim);
        }

        if (gltfNode.meshIndex.has_value())
        {
            const auto& gltfMesh = asset.meshes[gltfNode.meshIndex.value()];
            std::string meshName = gltfMesh.name.empty() ? std::string(gltfNode.name) + "_Mesh"
                                                         : std::string(gltfMesh.name);
            auto meshNode = std::make_unique<Node>(std::move(meshName));
            auto& meshRef = node.addChild(std::move(meshNode));
            Mesh& mesh =
                attachMeshToNode(asset, nodeIndex, gltfNode.meshIndex.value(), meshRef, node,
                                 baseDir, resources, assets, meshMap, physicsNodeConfigs, physics);

            if (hasWeightAnim)
            {
                for (const auto& [animId, anim] : nodeAnimations)
                {
                    mesh.addMorphAnimation(animId, anim);
                }
                mesh.initialMorphWeights(std::vector<float>(numMorphTargets, 0.0f));
            }
        }
    }
    else if (gltfNode.meshIndex.has_value())
    {
        // Only weight animation -- mesh goes directly on this node
        const auto& gltfMesh = asset.meshes[gltfNode.meshIndex.value()];
        Mesh& mesh =
            attachMeshToNode(asset, nodeIndex, gltfNode.meshIndex.value(), node, node, baseDir,
                             resources, assets, meshMap, physicsNodeConfigs, physics);

        if (hasWeightAnim)
        {
            for (const auto& [animId, anim] : nodeAnimations)
            {
                mesh.addMorphAnimation(animId, anim);
            }
        }

        // Apply initial weights from glTF mesh
        mesh.initialMorphWeights(initialMorphWeights(gltfMesh, numMorphTargets));
    }

    if (gltfNode.cameraIndex.has_value())
    {
        attachCamera(node, activeCamera);
    }

    for (auto childIndex : gltfNode.children)
    {
        auto childNode = std::make_unique<Node>(nodeName(asset, asset.nodes[childIndex]));
        auto& childRef = node.addChild(std::move(childNode));
        nodeMap[childIndex] = &childRef;
        applyControllable(childIndex, controllableNodeIndices, childRef);

        if (nodeHasAnimation(asset, childIndex))
        {
            configureAnimatedNode(asset, childIndex, childRef, baseDir, resources, assets, nodeMap,
                                  meshMap, nextAnimSlot, activeCamera, controllableNodeIndices,
                                  physicsNodeConfigs, physics);
        }
        else
        {
            loadNode(asset, childIndex, childRef, baseDir, resources, assets, nodeMap, meshMap,
                     nextAnimSlot, activeCamera, controllableNodeIndices, physicsNodeConfigs,
                     physics);
        }
    }
}

void GltfLoader::loadNode(const fastgltf::Asset& asset, std::size_t nodeIndex, Node& node,
                          const std::string& baseDir, Resources& resources, Assets& assets,
                          NodeMap& nodeMap, MeshMap& meshMap, std::size_t& nextAnimSlot,
                          Node*& activeCamera,
                          const std::unordered_set<std::size_t>& controllableNodeIndices,
                          const std::unordered_map<std::size_t, PhysicsConfig>& physicsNodeConfigs,
                          PhysicsWorld& physics)
{
    const auto& gltfNode = asset.nodes[nodeIndex];
    validatePhysicsTarget(nodeIndex, controllableNodeIndices, physicsNodeConfigs, gltfNode);
    applyTRS(gltfNode, node);

    applyLight(asset, gltfNode, node);

    if (gltfNode.meshIndex.has_value())
    {
        const auto& gltfMesh = asset.meshes[gltfNode.meshIndex.value()];
        Mesh& mesh =
            attachMeshToNode(asset, nodeIndex, gltfNode.meshIndex.value(), node, node, baseDir,
                             resources, assets, meshMap, physicsNodeConfigs, physics);

        // Static meshes with morph targets still honour mesh.weights (e.g.
        // MorphPrimitivesTest). Without this, weights stay at zero and the
        // base geometry renders unmorphed.
        std::size_t numMorphTargets = countMorphTargets(gltfMesh);
        if (numMorphTargets > 0)
        {
            mesh.initialMorphWeights(initialMorphWeights(gltfMesh, numMorphTargets));
        }
    }

    if (gltfNode.cameraIndex.has_value())
    {
        attachCamera(node, activeCamera);
    }

    for (auto childIndex : gltfNode.children)
    {
        auto childNode = std::make_unique<Node>(nodeName(asset, asset.nodes[childIndex]));
        auto& childRef = node.addChild(std::move(childNode));
        nodeMap[childIndex] = &childRef;
        applyControllable(childIndex, controllableNodeIndices, childRef);

        if (nodeHasAnimation(asset, childIndex))
        {
            configureAnimatedNode(asset, childIndex, childRef, baseDir, resources, assets, nodeMap,
                                  meshMap, nextAnimSlot, activeCamera, controllableNodeIndices,
                                  physicsNodeConfigs, physics);
        }
        else
        {
            loadNode(asset, childIndex, childRef, baseDir, resources, assets, nodeMap, meshMap,
                     nextAnimSlot, activeCamera, controllableNodeIndices, physicsNodeConfigs,
                     physics);
        }
    }
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
        auto min = boundsArrayVec3(posAccessor.min.value());
        auto max = boundsArrayVec3(posAccessor.max.value());
        if (min.has_value() && max.has_value())
        {
            return AABB{min.value(), max.value()};
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

        bounds =
            bounds.has_value() ? mergeBounds(bounds.value(), primitiveBox.value()) : primitiveBox;
    }

    return bounds;
}

Image GltfLoader::loadImage(const fastgltf::Asset& asset, std::size_t imageIndex,
                            const std::string& baseDir)
{
    const auto& image = asset.images[imageIndex];

    if (auto* uri = std::get_if<fastgltf::sources::URI>(&image.data);
        uri != nullptr && uri->uri.isLocalPath() && !uri->uri.isDataUri() &&
        uri->fileByteOffset == 0)
    {
        return Image::load_from_file((std::filesystem::path(baseDir) / uri->uri.fspath()).string());
    }

    auto bytes = loadDataSourceBytes(asset, image.data, baseDir,
                                     "image[" + std::to_string(imageIndex) + "]");
    if (bytes.empty())
    {
        throw std::runtime_error("Image data is empty for image[" + std::to_string(imageIndex) +
                                 "]");
    }

    return Image::load_from_memory(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(),
                                   "image[" + std::to_string(imageIndex) + "]");
}

KtxImage GltfLoader::loadKtxImage(const fastgltf::Asset& asset, std::size_t imageIndex,
                                  const std::string& baseDir)
{
    const auto& image = asset.images[imageIndex];

    if (auto* uri = std::get_if<fastgltf::sources::URI>(&image.data);
        uri != nullptr && uri->uri.isLocalPath() && !uri->uri.isDataUri() &&
        uri->fileByteOffset == 0)
    {
        return KtxImage::load_from_file(
            (std::filesystem::path(baseDir) / uri->uri.fspath()).string());
    }

    auto bytes = loadDataSourceBytes(asset, image.data, baseDir,
                                     "image[" + std::to_string(imageIndex) + "]");
    if (bytes.empty())
    {
        throw std::runtime_error("Image data is empty for image[" + std::to_string(imageIndex) +
                                 "]");
    }

    return KtxImage::load_from_memory(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(),
                                      "image[" + std::to_string(imageIndex) + "]");
}

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

    const auto& posAccessor = asset.accessors[posAttr->accessorIndex];
    std::vector<fastgltf::math::fvec3> positions(posAccessor.count);
    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
        asset, posAccessor,
        [&](fastgltf::math::fvec3 pos, std::size_t idx) { positions[idx] = pos; });

    std::vector<fastgltf::math::fvec3> normals;
    if (normAttr != primitive.attributes.end())
    {
        const auto& normAccessor = asset.accessors[normAttr->accessorIndex];
        normals.resize(normAccessor.count);
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
            asset, normAccessor,
            [&](fastgltf::math::fvec3 n, std::size_t idx) { normals[idx] = n; });
    }

    std::vector<fastgltf::math::fvec2> texcoords;
    if (uvAttr != primitive.attributes.end())
    {
        const auto& uvAccessor = asset.accessors[uvAttr->accessorIndex];
        texcoords.resize(uvAccessor.count);
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(
            asset, uvAccessor,
            [&](fastgltf::math::fvec2 uv, std::size_t idx) { texcoords[idx] = uv; });
    }

    // Optional second UV set. Stage B will let materials sample either set
    // per texture slot; for now we just round-trip the data into the GPU
    // buffer so the shader path can pick it up later.
    std::vector<fastgltf::math::fvec2> texcoords1;
    if (uv1Attr != primitive.attributes.end())
    {
        const auto& uv1Accessor = asset.accessors[uv1Attr->accessorIndex];
        texcoords1.resize(uv1Accessor.count);
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(
            asset, uv1Accessor,
            [&](fastgltf::math::fvec2 uv, std::size_t idx) { texcoords1[idx] = uv; });
    }

    // Joint indices — may be uint8 or uint16 in glTF, stored as uint32
    std::vector<std::array<uint32_t, 4>> joints;
    if (jointsAttr != primitive.attributes.end())
    {
        const auto& jointsAccessor = asset.accessors[jointsAttr->accessorIndex];
        joints.resize(jointsAccessor.count);
        fastgltf::iterateAccessorWithIndex<fastgltf::math::u16vec4>(
            asset, jointsAccessor, [&](fastgltf::math::u16vec4 j, std::size_t idx)
            { joints[idx] = {j.x(), j.y(), j.z(), j.w()}; });
    }

    std::vector<fastgltf::math::fvec4> weights;
    if (weightsAttr != primitive.attributes.end())
    {
        const auto& weightsAccessor = asset.accessors[weightsAttr->accessorIndex];
        weights.resize(weightsAccessor.count);
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
            asset, weightsAccessor,
            [&](fastgltf::math::fvec4 w, std::size_t idx) { weights[idx] = w; });
    }

    std::vector<fastgltf::math::fvec4> sourceTangents;
    if (tangentAttr != primitive.attributes.end())
    {
        const auto& tangentAccessor = asset.accessors[tangentAttr->accessorIndex];
        sourceTangents.resize(tangentAccessor.count);
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
            asset, tangentAccessor,
            [&](fastgltf::math::fvec4 t, std::size_t idx) { sourceTangents[idx] = t; });
    }

    std::vector<fastgltf::math::fvec4> colours;
    if (colourAttr != primitive.attributes.end())
    {
        const auto& colourAccessor = asset.accessors[colourAttr->accessorIndex];
        colours.resize(colourAccessor.count);
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
            asset, colourAccessor,
            [&](fastgltf::math::fvec4 c, std::size_t idx) { colours[idx] = c; });
    }

    std::vector<uint32_t> idxs;
    if (primitive.indicesAccessor.has_value())
    {
        const auto& indexAccessor = asset.accessors[primitive.indicesAccessor.value()];
        idxs.reserve(indexAccessor.count);
        fastgltf::iterateAccessor<std::uint32_t>(asset, indexAccessor,
                                                 [&](std::uint32_t idx) { idxs.push_back(idx); });
    }
    else
    {
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            idxs.push_back(static_cast<uint32_t>(i));
        }
    }

    // glTF 2.0 spec: when NORMAL is absent the implementation must compute
    // normals. Smooth (area-weighted) variant — matches glTF reference viewer
    // output and avoids the ugly faceted look on curved meshes like Fox.
    if (normals.empty() && !positions.empty())
    {
        std::vector<Vec3> enginePositions;
        enginePositions.reserve(positions.size());
        for (const auto& p : positions)
        {
            enginePositions.emplace_back(p.x(), p.y(), p.z());
        }
        const auto generated = GltfLoader::generateSmoothNormals(enginePositions, idxs);
        normals.resize(positions.size());
        for (std::size_t i = 0; i < generated.size(); ++i)
        {
            normals[i] = {generated[i].x(), generated[i].y(), generated[i].z()};
        }
    }

    std::vector<Vec4> tangents;
    TangentGenerationResult tangentResult;
    if (tangentAttr != primitive.attributes.end())
    {
        tangents.reserve(sourceTangents.size());
        for (const auto& tangent : sourceTangents)
        {
            tangents.emplace_back(tangent.x(), tangent.y(), tangent.z(), tangent.w());
        }
        tangentResult.succeeded = true;
    }
    else if (needsTangents)
    {
        std::vector<Vec3> positionData;
        positionData.reserve(positions.size());
        for (const auto& position : positions)
        {
            positionData.emplace_back(position.x(), position.y(), position.z());
        }

        std::vector<Vec3> normalData;
        normalData.reserve(normals.size());
        for (const auto& normal : normals)
        {
            normalData.emplace_back(normal.x(), normal.y(), normal.z());
        }

        std::vector<Vec2> texcoordData;
        texcoordData.reserve(texcoords.size());
        for (const auto& texcoord : texcoords)
        {
            texcoordData.emplace_back(texcoord.x(), texcoord.y());
        }

        tangentResult = TangentGenerator::generate(positionData, normalData, texcoordData, idxs);
        if (tangentResult.succeeded)
        {
            tangents = tangentResult.tangents;
        }
    }

    std::vector<Vertex> verts;
    verts.reserve(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        Vec3 pos{positions[i].x(), positions[i].y(), positions[i].z()};
        Vec3 norm = (i < normals.size()) ? Vec3{normals[i].x(), normals[i].y(), normals[i].z()}
                                         : Vec3{0.0f, 1.0f, 0.0f};
        float u = (i < texcoords.size()) ? texcoords[i].x() : 0.0f;
        float v = (i < texcoords.size()) ? texcoords[i].y() : 0.0f;

        Colour3 vertexColour{1.0f, 1.0f, 1.0f};
        if (i < colours.size())
        {
            vertexColour = Colour3{colours[i].x(), colours[i].y(), colours[i].z()};
        }

        Joints4 jt{};
        if (i < joints.size())
        {
            jt = Joints4{joints[i][0], joints[i][1], joints[i][2], joints[i][3]};
        }

        Vec4 wt{};
        if (i < weights.size())
        {
            wt = Vec4{weights[i].x(), weights[i].y(), weights[i].z(), weights[i].w()};
        }

        Vec4 tang{0.0f, 0.0f, 0.0f, 1.0f};
        if (i < tangents.size())
        {
            tang = tangents[i];
        }

        // Second UV set falls back to the first when the source mesh only
        // has TEXCOORD_0 — keeps the shader path uniform.
        const float u1 = (i < texcoords1.size()) ? texcoords1[i].x() : u;
        const float v1 = (i < texcoords1.size()) ? texcoords1[i].y() : v;

        Vertex vertex{pos, vertexColour, norm, Vec2{u, v}, jt, wt, tang};
        vertex.texCoord1(Vec2{u1, v1});
        verts.push_back(vertex);
    }
    geometry.vertices(std::move(verts));
    geometry.indices(std::move(idxs));

    // Load morph targets
    if (!primitive.targets.empty())
    {
        std::vector<std::vector<Vec3>> morphPositions;
        std::vector<std::vector<Vec3>> morphNormals;
        std::vector<std::vector<Vec3>> morphTangents;

        for (std::size_t ti = 0; ti < primitive.targets.size(); ++ti)
        {
            const auto& target = primitive.targets[ti];
            std::vector<Vec3> targetPos(positions.size());
            std::vector<Vec3> targetNorm(positions.size());
            std::vector<Vec3> targetTang(positions.size());

            for (const auto& attr : target)
            {
                const auto& accessor = asset.accessors[attr.accessorIndex];
                if (attr.name == "POSITION")
                {
                    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                        asset, accessor, [&](fastgltf::math::fvec3 p, std::size_t idx)
                        { targetPos[idx] = Vec3{p.x(), p.y(), p.z()}; });
                }
                else if (attr.name == "NORMAL")
                {
                    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                        asset, accessor, [&](fastgltf::math::fvec3 n, std::size_t idx)
                        { targetNorm[idx] = Vec3{n.x(), n.y(), n.z()}; });
                }
                else if (attr.name == "TANGENT")
                {
                    // glTF spec: morph TANGENT deltas are vec3 (xyz only —
                    // handedness in the base tangent's .w stays unchanged).
                    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                        asset, accessor, [&](fastgltf::math::fvec3 t, std::size_t idx)
                        { targetTang[idx] = Vec3{t.x(), t.y(), t.z()}; });
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

        auto resolveTextures = [&](std::optional<std::size_t> materialIndex)
        {
            ResolvedMaterialTextures result;
            if (!materialIndex.has_value())
            {
                return result;
            }

            const auto& gltfMat = asset.materials[materialIndex.value()];
            if (gltfMat.pbrData.baseColorTexture.has_value())
            {
                result.baseColour =
                    resolveTextureIndex(asset, gltfMat.pbrData.baseColorTexture->textureIndex,
                                        baseDir, resources, assets, TextureEncoding::Srgb);
            }
            if (gltfMat.emissiveTexture.has_value())
            {
                result.emissive =
                    resolveTextureIndex(asset, gltfMat.emissiveTexture->textureIndex, baseDir,
                                        resources, assets, TextureEncoding::Srgb);
            }
            if (gltfMat.normalTexture.has_value())
            {
                result.normal =
                    resolveTextureIndex(asset, gltfMat.normalTexture->textureIndex, baseDir,
                                        resources, assets, TextureEncoding::Linear);
            }
            if (gltfMat.pbrData.metallicRoughnessTexture.has_value())
            {
                result.metallicRoughness = resolveTextureIndex(
                    asset, gltfMat.pbrData.metallicRoughnessTexture->textureIndex, baseDir,
                    resources, assets, TextureEncoding::Linear);
            }
            if (gltfMat.occlusionTexture.has_value())
            {
                result.occlusion =
                    resolveTextureIndex(asset, gltfMat.occlusionTexture->textureIndex, baseDir,
                                        resources, assets, TextureEncoding::Linear);
            }
            if (gltfMat.transmission != nullptr &&
                gltfMat.transmission->transmissionTexture.has_value())
            {
                result.transmission = resolveTextureIndex(
                    asset, gltfMat.transmission->transmissionTexture->textureIndex, baseDir,
                    resources, assets, TextureEncoding::Linear);
            }
            if (gltfMat.clearcoat != nullptr)
            {
                const auto& clearcoat = *gltfMat.clearcoat;
                if (clearcoat.clearcoatTexture.has_value())
                {
                    result.clearcoat =
                        resolveTextureIndex(asset, clearcoat.clearcoatTexture->textureIndex,
                                            baseDir, resources, assets, TextureEncoding::Linear);
                }
                if (clearcoat.clearcoatRoughnessTexture.has_value())
                {
                    result.clearcoatRoughness = resolveTextureIndex(
                        asset, clearcoat.clearcoatRoughnessTexture->textureIndex, baseDir,
                        resources, assets, TextureEncoding::Linear);
                }
                if (clearcoat.clearcoatNormalTexture.has_value())
                {
                    result.clearcoatNormal =
                        resolveTextureIndex(asset, clearcoat.clearcoatNormalTexture->textureIndex,
                                            baseDir, resources, assets, TextureEncoding::Linear);
                }
            }
            if (gltfMat.volume != nullptr && gltfMat.volume->thicknessTexture.has_value())
            {
                result.thickness =
                    resolveTextureIndex(asset, gltfMat.volume->thicknessTexture->textureIndex,
                                        baseDir, resources, assets, TextureEncoding::Linear);
            }
            return result;
        };

        const ResolvedMaterialTextures baseTextures = resolveTextures(baseMaterialIndex);

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
            const ResolvedMaterialTextures variantTextures = resolveTextures(mappedMaterialIndex);
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

    if (materialIndex.has_value())
    {
        const auto& gltfMat = asset.materials[materialIndex.value()];
        material.name(std::string(gltfMat.name));

        const auto& pbr = gltfMat.pbrData;
        material.baseColor({static_cast<float>(pbr.baseColorFactor.x()),
                            static_cast<float>(pbr.baseColorFactor.y()),
                            static_cast<float>(pbr.baseColorFactor.z())});
        material.alpha(static_cast<float>(pbr.baseColorFactor.w()));
        material.metallic(static_cast<float>(pbr.metallicFactor));
        material.roughness(static_cast<float>(pbr.roughnessFactor));

        // KHR_materials_emissive_strength: multiplies emissiveFactor at load
        // time. Default 1.0 = unchanged. Authored values >1.0 produce HDR
        // emissives that read correctly through the bloom chain.
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
            material.occlusionStrength(
                static_cast<float>(gltfMat.occlusionTexture.value().strength));
        }

        // KHR_texture_transform per-slot UV transform reader. fastgltf parses
        // the extension into TextureInfo::transform when the parser opts in;
        // absent → identity transform → no behaviour change for old assets.
        auto readUvTransform = [](const fastgltf::TextureInfo& info) noexcept -> UvTransform
        {
            UvTransform t;
            if (info.transform)
            {
                const auto& src = *info.transform;
                t.offsetX = static_cast<float>(src.uvOffset.x());
                t.offsetY = static_cast<float>(src.uvOffset.y());
                t.scaleX = static_cast<float>(src.uvScale.x());
                t.scaleY = static_cast<float>(src.uvScale.y());
                t.rotation = static_cast<float>(src.rotation);
            }
            return t;
        };

        // Per-slot UV-set index + KHR_texture_transform. glTF allows each
        // texture to point at TEXCOORD_0 or TEXCOORD_1 (default 0). The shader
        // picks the stream via material.texCoordIndices and applies the transform.
        using Slot = MaterialTextureSlot;
        auto loadSlotUv = [&](Slot slot, const fastgltf::TextureInfo& info)
        {
            material.texture(slot).texCoord = static_cast<int>(info.texCoordIndex);
            material.texture(slot).transform = readUvTransform(info);
        };
        if (gltfMat.pbrData.baseColorTexture.has_value())
        {
            loadSlotUv(Slot::BaseColour, gltfMat.pbrData.baseColorTexture.value());
        }
        if (gltfMat.emissiveTexture.has_value())
        {
            loadSlotUv(Slot::Emissive, gltfMat.emissiveTexture.value());
        }
        if (gltfMat.normalTexture.has_value())
        {
            loadSlotUv(Slot::Normal, gltfMat.normalTexture.value());
        }
        if (gltfMat.pbrData.metallicRoughnessTexture.has_value())
        {
            loadSlotUv(Slot::MetallicRoughness, gltfMat.pbrData.metallicRoughnessTexture.value());
        }
        if (gltfMat.occlusionTexture.has_value())
        {
            loadSlotUv(Slot::Occlusion, gltfMat.occlusionTexture.value());
        }

        // KHR_materials_transmission + KHR_materials_ior, grouped into the
        // optional Transmission block. Engage it when transmission is authored
        // or ior differs from the spec default (1.5) — so an ior-only material
        // still carries its value; otherwise leave it unset and the defaults
        // (factor 0 / ior 1.5) apply. The transmission texture lives in the slot.
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
        if (gltfMat.transmission != nullptr &&
            gltfMat.transmission->transmissionTexture.has_value())
        {
            loadSlotUv(Slot::Transmission, gltfMat.transmission->transmissionTexture.value());
        }

        // KHR_materials_clearcoat. fastgltf surfaces clearcoat via a
        // unique_ptr<MaterialClearcoat>; null when the extension isn't on the
        // asset. clearcoatNormalTexture carries its own scale (NormalTextureInfo).
        if (gltfMat.clearcoat != nullptr)
        {
            const auto& cc = *gltfMat.clearcoat;
            ClearcoatParams clearcoat;
            clearcoat.factor = static_cast<float>(cc.clearcoatFactor);
            clearcoat.roughness = static_cast<float>(cc.clearcoatRoughnessFactor);
            if (cc.clearcoatTexture.has_value())
            {
                loadSlotUv(Slot::Clearcoat, cc.clearcoatTexture.value());
            }
            if (cc.clearcoatRoughnessTexture.has_value())
            {
                loadSlotUv(Slot::ClearcoatRoughness, cc.clearcoatRoughnessTexture.value());
            }
            if (cc.clearcoatNormalTexture.has_value())
            {
                const auto& info = cc.clearcoatNormalTexture.value();
                loadSlotUv(Slot::ClearcoatNormal, info);
                clearcoat.normalScale = static_cast<float>(info.scale);
            }
            material.clearcoat(clearcoat);
        }

        // KHR_materials_volume. Beer-Lambert absorption + thickness-driven
        // refraction sampling. Absent (volume == nullptr) leaves the optional
        // unset → thicknessFactor 0 and attenuationDistance +inf at UBO time,
        // collapsing to F3 thin-surface behaviour at sample time.
        if (gltfMat.volume != nullptr)
        {
            const auto& vol = *gltfMat.volume;
            VolumeParams volume;
            volume.thicknessFactor = static_cast<float>(vol.thicknessFactor);
            volume.attenuationColor = Colour3{static_cast<float>(vol.attenuationColor.x()),
                                              static_cast<float>(vol.attenuationColor.y()),
                                              static_cast<float>(vol.attenuationColor.z())};
            volume.attenuationDistance = static_cast<float>(vol.attenuationDistance);
            material.volume(volume);
            if (vol.thicknessTexture.has_value())
            {
                loadSlotUv(Slot::Thickness, vol.thicknessTexture.value());
            }
        }

        // KHR_materials_unlit. fastgltf surfaces this as a plain bool that
        // stays false unless the extension is present + enabled on the parser.
        material.unlit(gltfMat.unlit);

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

    return material;
}

// ---------------------------------------------------------------------------
// Animation helpers
// ---------------------------------------------------------------------------

bool GltfLoader::nodeHasAnimation(const fastgltf::Asset& asset, std::size_t nodeIndex)
{
    for (const auto& anim : asset.animations)
    {
        for (const auto& channel : anim.channels)
        {
            if (channel.nodeIndex.has_value() && channel.nodeIndex.value() == nodeIndex)
            {
                return true;
            }
        }
    }
    return false;
}

float GltfLoader::computeSharedDuration(const fastgltf::Asset& asset, std::size_t gltfAnimIndex)
{
    float sharedDuration = 0.0f;

    const auto& anim = asset.animations[gltfAnimIndex];
    for (const auto& channel : anim.channels)
    {
        const auto& sampler = anim.samplers[channel.samplerIndex];
        const auto& inputAccessor = asset.accessors[sampler.inputAccessor];
        fastgltf::iterateAccessor<float>(asset, inputAccessor, [&](float t)
                                         { sharedDuration = std::max(sharedDuration, t); });
    }

    return sharedDuration;
}

Animation::Interpolation GltfLoader::mapInterpolation(fastgltf::AnimationInterpolation m)
{
    switch (m)
    {
    case fastgltf::AnimationInterpolation::Step:
        return Animation::Interpolation::Step;
    case fastgltf::AnimationInterpolation::CubicSpline:
        return Animation::Interpolation::CubicSpline;
    case fastgltf::AnimationInterpolation::Linear:
    default:
        return Animation::Interpolation::Linear;
    }
}

namespace
{

// glTF CUBICSPLINE output layout: per keyframe, three elements packed as
// [in_tangent, value, out_tangent]. For non-CubicSpline modes there is a single
// value per keyframe. Returns the stride (3 for CubicSpline, 1 otherwise).
constexpr std::size_t outputStride(Animation::Interpolation m) noexcept
{
    return m == Animation::Interpolation::CubicSpline ? 3 : 1;
}

} // namespace

void GltfLoader::loadRotationChannel(const fastgltf::Asset& asset,
                                     const fastgltf::AnimationSampler& sampler, Animation& anim)
{
    const auto interp = mapInterpolation(sampler.interpolation);
    const auto& inputAccessor = asset.accessors[sampler.inputAccessor];
    const auto& outputAccessor = asset.accessors[sampler.outputAccessor];

    std::vector<float> times(inputAccessor.count);
    fastgltf::iterateAccessorWithIndex<float>(asset, inputAccessor,
                                              [&](float t, std::size_t idx) { times[idx] = t; });

    std::vector<fastgltf::math::fvec4> quats(outputAccessor.count);
    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
        asset, outputAccessor, [&](fastgltf::math::fvec4 q, std::size_t idx) { quats[idx] = q; });

    const std::size_t stride = outputStride(interp);
    const std::size_t valueOffset = (interp == Animation::Interpolation::CubicSpline) ? 1 : 0;

    std::vector<Animation::RotationKeyframe> kf;
    kf.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
    {
        std::size_t vi = i * stride + valueOffset;
        if (vi >= quats.size())
        {
            break;
        }
        kf.push_back({times[i], quats[vi].x(), quats[vi].y(), quats[vi].z(), quats[vi].w()});
    }

    anim.rotationKeyframes(std::move(kf));
    anim.rotationInterpolation(interp);

    if (interp == Animation::Interpolation::CubicSpline)
    {
        std::vector<Quaternion> in;
        std::vector<Quaternion> out;
        in.reserve(times.size());
        out.reserve(times.size());
        for (std::size_t i = 0; i < times.size(); ++i)
        {
            std::size_t inIdx = i * 3;
            std::size_t outIdx = i * 3 + 2;
            if (outIdx >= quats.size())
            {
                break;
            }
            in.push_back(
                Quaternion{quats[inIdx].x(), quats[inIdx].y(), quats[inIdx].z(), quats[inIdx].w()});
            out.push_back(Quaternion{quats[outIdx].x(), quats[outIdx].y(), quats[outIdx].z(),
                                     quats[outIdx].w()});
        }
        anim.rotationTangents(std::move(in), std::move(out));
    }
}

void GltfLoader::loadTranslationChannel(const fastgltf::Asset& asset,
                                        const fastgltf::AnimationSampler& sampler, Animation& anim)
{
    const auto interp = mapInterpolation(sampler.interpolation);
    const auto& inputAccessor = asset.accessors[sampler.inputAccessor];
    const auto& outputAccessor = asset.accessors[sampler.outputAccessor];

    std::vector<float> times(inputAccessor.count);
    fastgltf::iterateAccessorWithIndex<float>(asset, inputAccessor,
                                              [&](float t, std::size_t idx) { times[idx] = t; });

    std::vector<fastgltf::math::fvec3> positions(outputAccessor.count);
    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
        asset, outputAccessor,
        [&](fastgltf::math::fvec3 p, std::size_t idx) { positions[idx] = p; });

    const std::size_t stride = outputStride(interp);
    const std::size_t valueOffset = (interp == Animation::Interpolation::CubicSpline) ? 1 : 0;

    std::vector<Animation::TranslationKeyframe> kf;
    kf.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
    {
        std::size_t vi = i * stride + valueOffset;
        if (vi >= positions.size())
        {
            break;
        }
        kf.push_back({times[i], Vec3{positions[vi].x(), positions[vi].y(), positions[vi].z()}});
    }

    anim.translationKeyframes(std::move(kf));
    anim.translationInterpolation(interp);

    if (interp == Animation::Interpolation::CubicSpline)
    {
        std::vector<Vec3> in;
        std::vector<Vec3> out;
        in.reserve(times.size());
        out.reserve(times.size());
        for (std::size_t i = 0; i < times.size(); ++i)
        {
            std::size_t inIdx = i * 3;
            std::size_t outIdx = i * 3 + 2;
            if (outIdx >= positions.size())
            {
                break;
            }
            in.push_back(Vec3{positions[inIdx].x(), positions[inIdx].y(), positions[inIdx].z()});
            out.push_back(
                Vec3{positions[outIdx].x(), positions[outIdx].y(), positions[outIdx].z()});
        }
        anim.translationTangents(std::move(in), std::move(out));
    }
}

void GltfLoader::loadScaleChannel(const fastgltf::Asset& asset,
                                  const fastgltf::AnimationSampler& sampler, Animation& anim)
{
    const auto interp = mapInterpolation(sampler.interpolation);
    const auto& inputAccessor = asset.accessors[sampler.inputAccessor];
    const auto& outputAccessor = asset.accessors[sampler.outputAccessor];

    std::vector<float> times(inputAccessor.count);
    fastgltf::iterateAccessorWithIndex<float>(asset, inputAccessor,
                                              [&](float t, std::size_t idx) { times[idx] = t; });

    std::vector<fastgltf::math::fvec3> scales(outputAccessor.count);
    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
        asset, outputAccessor, [&](fastgltf::math::fvec3 s, std::size_t idx) { scales[idx] = s; });

    const std::size_t stride = outputStride(interp);
    const std::size_t valueOffset = (interp == Animation::Interpolation::CubicSpline) ? 1 : 0;

    std::vector<Animation::ScaleKeyframe> kf;
    kf.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
    {
        std::size_t vi = i * stride + valueOffset;
        if (vi >= scales.size())
        {
            break;
        }
        kf.push_back({times[i], Vec3{scales[vi].x(), scales[vi].y(), scales[vi].z()}});
    }

    anim.scaleKeyframes(std::move(kf));
    anim.scaleInterpolation(interp);

    if (interp == Animation::Interpolation::CubicSpline)
    {
        std::vector<Vec3> in;
        std::vector<Vec3> out;
        in.reserve(times.size());
        out.reserve(times.size());
        for (std::size_t i = 0; i < times.size(); ++i)
        {
            std::size_t inIdx = i * 3;
            std::size_t outIdx = i * 3 + 2;
            if (outIdx >= scales.size())
            {
                break;
            }
            in.push_back(Vec3{scales[inIdx].x(), scales[inIdx].y(), scales[inIdx].z()});
            out.push_back(Vec3{scales[outIdx].x(), scales[outIdx].y(), scales[outIdx].z()});
        }
        anim.scaleTangents(std::move(in), std::move(out));
    }
}

void GltfLoader::loadWeightChannel(const fastgltf::Asset& asset,
                                   const fastgltf::AnimationSampler& sampler, Animation& anim,
                                   std::size_t numTargets)
{
    const auto interp = mapInterpolation(sampler.interpolation);
    const auto& inputAccessor = asset.accessors[sampler.inputAccessor];
    const auto& outputAccessor = asset.accessors[sampler.outputAccessor];

    std::vector<float> times(inputAccessor.count);
    fastgltf::iterateAccessorWithIndex<float>(asset, inputAccessor,
                                              [&](float t, std::size_t idx) { times[idx] = t; });

    // Output is a flat array of scalars. Linear/Step: numKeyframes * numTargets.
    // CubicSpline: numKeyframes * 3 * numTargets (in_tan, value, out_tan per key).
    std::vector<float> allWeights(outputAccessor.count);
    fastgltf::iterateAccessorWithIndex<float>(asset, outputAccessor, [&](float w, std::size_t idx)
                                              { allWeights[idx] = w; });

    const std::size_t stride = outputStride(interp);
    const std::size_t valueOffset = (interp == Animation::Interpolation::CubicSpline) ? 1 : 0;

    std::vector<Animation::WeightKeyframe> kf;
    kf.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
    {
        Animation::WeightKeyframe wkf;
        wkf.time = times[i];
        wkf.weights.resize(numTargets, 0.0f);
        std::size_t base = (i * stride + valueOffset) * numTargets;
        for (std::size_t w = 0; w < numTargets; ++w)
        {
            std::size_t idx = base + w;
            if (idx < allWeights.size())
            {
                wkf.weights[w] = allWeights[idx];
            }
        }
        kf.push_back(std::move(wkf));
    }

    anim.weightKeyframes(std::move(kf));
    anim.weightInterpolation(interp);

    if (interp == Animation::Interpolation::CubicSpline)
    {
        std::vector<std::vector<float>> in;
        std::vector<std::vector<float>> out;
        in.reserve(times.size());
        out.reserve(times.size());
        for (std::size_t i = 0; i < times.size(); ++i)
        {
            std::vector<float> inV(numTargets, 0.0f);
            std::vector<float> outV(numTargets, 0.0f);
            std::size_t inBase = (i * 3) * numTargets;
            std::size_t outBase = (i * 3 + 2) * numTargets;
            for (std::size_t w = 0; w < numTargets; ++w)
            {
                if (inBase + w < allWeights.size())
                {
                    inV[w] = allWeights[inBase + w];
                }
                if (outBase + w < allWeights.size())
                {
                    outV[w] = allWeights[outBase + w];
                }
            }
            in.push_back(std::move(inV));
            out.push_back(std::move(outV));
        }
        anim.weightTangents(std::move(in), std::move(out));
    }
}

void GltfLoader::loadAnimation(const fastgltf::Asset& asset, std::size_t gltfAnimIndex,
                               std::size_t nodeIndex, Animation& la, std::size_t numMorphTargets)
{
    float sharedDuration = computeSharedDuration(asset, gltfAnimIndex);

    const auto& anim = asset.animations[gltfAnimIndex];
    for (const auto& channel : anim.channels)
    {
        if (!channel.nodeIndex.has_value() || channel.nodeIndex.value() != nodeIndex)
        {
            continue;
        }

        const auto& sampler = anim.samplers[channel.samplerIndex];

        if (channel.path == fastgltf::AnimationPath::Rotation)
        {
            loadRotationChannel(asset, sampler, la);
        }
        else if (channel.path == fastgltf::AnimationPath::Translation)
        {
            loadTranslationChannel(asset, sampler, la);
        }
        else if (channel.path == fastgltf::AnimationPath::Scale)
        {
            loadScaleChannel(asset, sampler, la);
        }
        else if (channel.path == fastgltf::AnimationPath::Weights && numMorphTargets > 0)
        {
            loadWeightChannel(asset, sampler, la, numMorphTargets);
        }
    }

    la.duration(sharedDuration);
}

// ---------------------------------------------------------------------------
// Skin loading helpers
// ---------------------------------------------------------------------------

void GltfLoader::loadSkin(const fastgltf::Asset& asset, std::size_t skinIndex,
                          const NodeMap& nodeMap, Assets& assets)
{
    const auto& gltfSkin = asset.skins[skinIndex];
    auto& skin = assets.skin(skinIndex);

    if (!gltfSkin.name.empty())
    {
        skin.name(std::string(gltfSkin.name));
    }

    // Read inverse bind matrices (one per joint)
    std::vector<Mat4> inverseBindMatrices(gltfSkin.joints.size(), Mat4::identity());
    if (gltfSkin.inverseBindMatrices.has_value())
    {
        const auto& accessor = asset.accessors[gltfSkin.inverseBindMatrices.value()];
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fmat4x4>(
            asset, accessor,
            [&](const fastgltf::math::fmat4x4& m, std::size_t idx)
            {
                Mat4 mat;
                for (int col = 0; col < 4; ++col)
                {
                    for (int row = 0; row < 4; ++row)
                    {
                        mat[row, col] = m.col(col)[row];
                    }
                }
                inverseBindMatrices[idx] = mat;
            });
    }

    for (std::size_t i = 0; i < gltfSkin.joints.size(); ++i)
    {
        auto jointNodeIndex = gltfSkin.joints[i];
        auto it = nodeMap.find(jointNodeIndex);
        if (it == nodeMap.end())
        {
            throw std::runtime_error("Skin joint references unknown node index " +
                                     std::to_string(jointNodeIndex));
        }
        skin.addJoint(it->second, inverseBindMatrices[i]);
    }
}

void GltfLoader::applySkins(const fastgltf::Asset& asset, const NodeMap& nodeMap,
                            const MeshMap& meshMap, Assets& assets)
{
    for (const auto& [nodeIndex, nodePtr] : nodeMap)
    {
        const auto& gltfNode = asset.nodes[nodeIndex];
        if (!gltfNode.skinIndex.has_value())
        {
            continue;
        }

        auto skinIndex = gltfNode.skinIndex.value();
        if (assets.skin(skinIndex).empty())
        {
            loadSkin(asset, skinIndex, nodeMap, assets);
        }

        auto meshIt = meshMap.find(nodeIndex);
        if (meshIt != meshMap.end())
        {
            meshIt->second->skin(&assets.skin(skinIndex));
        }
    }
}

} // namespace fire_engine
