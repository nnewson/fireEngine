#include <fire_engine/core/gltf_loader.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <simdjson.h>

#include <fire_engine/math/quaternion.hpp>

namespace fire_engine
{
namespace
{

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

    [[nodiscard]] bool readBool(std::string_view key, bool fallback, std::string_view label) const
    {
        auto element = object_.at_key(key);
        if (element.error() == simdjson::NO_SUCH_FIELD)
        {
            return fallback;
        }

        bool value = false;
        if (element.get(value) != simdjson::SUCCESS)
        {
            throw std::runtime_error(prefix(label) + " must be a boolean");
        }
        return value;
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

    [[nodiscard]] Quaternion readQuat(std::string_view key, std::string_view label) const
    {
        auto element = object_.at_key(key);
        if (element.error() == simdjson::NO_SUCH_FIELD)
        {
            return Quaternion::identity();
        }

        simdjson::dom::array array;
        if (element.get_array().get(array) != simdjson::SUCCESS)
        {
            throw std::runtime_error(prefix(label) + " must be an array of four numbers [x,y,z,w]");
        }

        std::array<float, 4> values{};
        std::size_t i = 0;
        for (auto valueElement : array)
        {
            if (i >= values.size())
            {
                throw std::runtime_error(prefix(label) + " must contain exactly four numbers");
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
            throw std::runtime_error(prefix(label) + " must contain exactly four numbers");
        }
        return {values[0], values[1], values[2], values[3]};
    }

private:
    [[nodiscard]] std::string prefix(std::string_view label) const
    {
        return "glTF " + std::string(section_) + " " + std::string(label);
    }

    simdjson::dom::object& object_;
    std::string_view section_;
};

[[nodiscard]] std::optional<ColliderShape> parsePrimitiveShape(std::string_view shape,
                                                               const ExtrasReader& reader)
{
    const Vec3 center = reader.readVec3("Center", {}, "Center");
    if (shape == "Box")
    {
        return BoxShape{reader.readVec3("HalfExtents", {0.5f, 0.5f, 0.5f}, "HalfExtents"), center};
    }
    if (shape == "Sphere")
    {
        return SphereShape{reader.readFloat("Radius", 0.5f, "Radius"), center};
    }
    if (shape == "Capsule")
    {
        return CapsuleShape{reader.readFloat("Radius", 0.5f, "Radius"),
                            reader.readFloat("HalfHeight", 0.5f, "HalfHeight"), center};
    }
    return std::nullopt;
}

[[nodiscard]] std::vector<CompoundChild> parseCompoundChildren(simdjson::dom::object& physicsObject)
{
    auto childrenElement = physicsObject.at_key("Children");
    simdjson::dom::array childrenArray;
    if (childrenElement.error() == simdjson::NO_SUCH_FIELD ||
        childrenElement.get_array().get(childrenArray) != simdjson::SUCCESS)
    {
        throw std::runtime_error("glTF Physics Compound requires a Children array");
    }

    std::vector<CompoundChild> children;
    for (auto childElement : childrenArray)
    {
        simdjson::dom::object childObject;
        if (childElement.get_object().get(childObject) != simdjson::SUCCESS)
        {
            throw std::runtime_error("glTF Physics Compound child must be an object");
        }
        const ExtrasReader childReader{childObject, "Physics Compound child"};

        std::string_view shape;
        if (childObject.at_key("Shape").get(shape) != simdjson::SUCCESS)
        {
            throw std::runtime_error("glTF Physics Compound child requires a Shape string");
        }
        auto primitive = parsePrimitiveShape(shape, childReader);
        if (!primitive.has_value())
        {
            throw std::runtime_error(
                "glTF Physics Compound child Shape must be Box, Sphere, or Capsule");
        }

        CompoundChild child;
        child.shape = std::move(primitive.value());
        child.localPosition = childReader.readVec3("Position", {}, "Position");
        child.localRotation = childReader.readQuat("Rotation", "Rotation");
        child.material = PhysicsMaterial{childReader.readFloat("Restitution", 1.0f, "Restitution"),
                                         childReader.readFloat("Friction", 0.0f, "Friction")};
        children.push_back(std::move(child));
    }
    return children;
}

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
    config.isTrigger = reader.readBool("IsTrigger", config.isTrigger, "IsTrigger");
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

        if (auto primitive = parsePrimitiveShape(shape, reader))
        {
            config.shape = std::move(primitive.value());
        }
        else if (shape == "ConvexHull")
        {
            config.convexHullFromMesh = true;
        }
        else if (shape == "Mesh")
        {
            config.staticMeshFromMesh = true;
        }
        else if (shape == "Compound")
        {
            config.compoundChildren = parseCompoundChildren(physicsObject);
        }
        else
        {
            throw std::runtime_error(
                "glTF Physics Shape must be Box, Sphere, Capsule, ConvexHull, Mesh, or Compound");
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

std::optional<RagdollParams> GltfLoader::nodeExtrasRagdoll(simdjson::dom::object* extras)
{
    if (extras == nullptr)
    {
        return std::nullopt;
    }

    simdjson::dom::object ragdollObject;
    auto ragdollElement = extras->at_key("Ragdoll");
    if (ragdollElement.error() == simdjson::NO_SUCH_FIELD)
    {
        return std::nullopt;
    }
    if (ragdollElement.get_object().get(ragdollObject) != simdjson::SUCCESS)
    {
        throw std::runtime_error("glTF Ragdoll extras must be an object");
    }

    const ExtrasReader reader{ragdollObject, "Ragdoll"};
    RagdollParams params;
    params.mass = reader.readFloat("Mass", params.mass, "Mass");
    params.radius = reader.readFloat("Radius", params.radius, "Radius");
    params.defaultBoneLength =
        reader.readFloat("BoneLength", params.defaultBoneLength, "BoneLength");
    params.swingLimit = reader.readFloat("SwingLimit", params.swingLimit, "SwingLimit");
    params.twistLimit = reader.readFloat("TwistLimit", params.twistLimit, "TwistLimit");

    auto coneTwistElement = ragdollObject.at_key("ConeTwist");
    if (coneTwistElement.error() != simdjson::NO_SUCH_FIELD)
    {
        bool coneTwist = false;
        if (coneTwistElement.get(coneTwist) != simdjson::SUCCESS)
        {
            throw std::runtime_error("glTF Ragdoll ConeTwist must be a boolean");
        }
        params.coneTwist = coneTwist;
    }

    return params;
}

} // namespace fire_engine
