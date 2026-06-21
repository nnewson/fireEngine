#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <simdjson.h>

#include <fire_engine/core/gltf_loader.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/transform.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::AlphaMode;
using fire_engine::ClearcoatParams;
using fire_engine::ClothMeshParams;
using fire_engine::GltfLoader;
using fire_engine::Mat4;
using fire_engine::Material;
using fire_engine::MaterialTextureSlot;
using fire_engine::Node;
using fire_engine::PhysicsBodyType;
using fire_engine::SphereShape;
using fire_engine::TransmissionParams;
using fire_engine::Vec3;

// Mirrors the occlusion-strength translation performed inside
// GltfLoader::loadMaterial so it can be exercised without a GPU.
static void applyOcclusionStrength(const fastgltf::Material& gltfMat, Material& material)
{
    if (gltfMat.occlusionTexture.has_value())
    {
        material.occlusionStrength(static_cast<float>(gltfMat.occlusionTexture.value().strength));
    }
}

// Mirrors KHR_materials_emissive_strength translation: emissiveFactor is
// scaled by emissiveStrength at load time. Default emissiveStrength = 1.0.
static void applyEmissiveStrength(const fastgltf::Material& gltfMat, Material& material)
{
    const float strength = static_cast<float>(gltfMat.emissiveStrength);
    material.emissive({static_cast<float>(gltfMat.emissiveFactor.x()) * strength,
                       static_cast<float>(gltfMat.emissiveFactor.y()) * strength,
                       static_cast<float>(gltfMat.emissiveFactor.z()) * strength});
}

// Mirrors KHR_texture_transform translation: when a TextureInfo carries the
// extension, copy offset/scale/rotation onto the matching Material slot.
static fire_engine::UvTransform readUvTransform(const fastgltf::TextureInfo& info)
{
    fire_engine::UvTransform t;
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
}

// Mirrors per-slot texCoord-index translation: each glTF TextureInfo names a
// texCoord index (0 = TEXCOORD_0 default, 1 = TEXCOORD_1).
static void applyTexCoordIndices(const fastgltf::Material& gltfMat, Material& material)
{
    using Slot = MaterialTextureSlot;
    if (gltfMat.pbrData.baseColorTexture.has_value())
    {
        material.texture(Slot::BaseColour).texCoord =
            static_cast<int>(gltfMat.pbrData.baseColorTexture.value().texCoordIndex);
    }
    if (gltfMat.emissiveTexture.has_value())
    {
        material.texture(Slot::Emissive).texCoord =
            static_cast<int>(gltfMat.emissiveTexture.value().texCoordIndex);
    }
    if (gltfMat.normalTexture.has_value())
    {
        material.texture(Slot::Normal).texCoord =
            static_cast<int>(gltfMat.normalTexture.value().texCoordIndex);
    }
    if (gltfMat.pbrData.metallicRoughnessTexture.has_value())
    {
        material.texture(Slot::MetallicRoughness).texCoord =
            static_cast<int>(gltfMat.pbrData.metallicRoughnessTexture.value().texCoordIndex);
    }
    if (gltfMat.occlusionTexture.has_value())
    {
        material.texture(Slot::Occlusion).texCoord =
            static_cast<int>(gltfMat.occlusionTexture.value().texCoordIndex);
    }
}

static fastgltf::Asset parseRealGltfAsset(const std::filesystem::path& gltfPath)
{
    constexpr fastgltf::Extensions enabledExtensions =
        fastgltf::Extensions::KHR_materials_emissive_strength |
        fastgltf::Extensions::KHR_texture_transform | fastgltf::Extensions::KHR_texture_basisu |
        fastgltf::Extensions::KHR_materials_variants | fastgltf::Extensions::KHR_materials_unlit |
        fastgltf::Extensions::KHR_lights_punctual |
        fastgltf::Extensions::KHR_materials_transmission | fastgltf::Extensions::KHR_materials_ior |
        fastgltf::Extensions::KHR_materials_clearcoat | fastgltf::Extensions::KHR_materials_volume;

    fastgltf::Parser parser(enabledExtensions);
    auto dataResult = fastgltf::GltfDataBuffer::FromPath(gltfPath);
    CHECK(dataResult.error() == fastgltf::Error::None);
    if (dataResult.error() != fastgltf::Error::None)
    {
        return fastgltf::Asset{};
    }

    auto result = parser.loadGltf(dataResult.get(), gltfPath.parent_path(),
                                  fastgltf::Options::LoadExternalBuffers |
                                      fastgltf::Options::LoadExternalImages);
    CHECK(result.error() == fastgltf::Error::None);
    if (result.error() != fastgltf::Error::None)
    {
        return fastgltf::Asset{};
    }

    return std::move(result.get());
}

static std::vector<std::size_t> referencedTextureIndices(const fastgltf::Material& material)
{
    std::vector<std::size_t> indices;

    if (material.pbrData.baseColorTexture.has_value())
    {
        indices.push_back(material.pbrData.baseColorTexture->textureIndex);
    }
    if (material.emissiveTexture.has_value())
    {
        indices.push_back(material.emissiveTexture->textureIndex);
    }
    if (material.normalTexture.has_value())
    {
        indices.push_back(material.normalTexture->textureIndex);
    }
    if (material.pbrData.metallicRoughnessTexture.has_value())
    {
        indices.push_back(material.pbrData.metallicRoughnessTexture->textureIndex);
    }
    if (material.occlusionTexture.has_value())
    {
        indices.push_back(material.occlusionTexture->textureIndex);
    }
    if (material.transmission != nullptr && material.transmission->transmissionTexture.has_value())
    {
        indices.push_back(material.transmission->transmissionTexture->textureIndex);
    }
    if (material.clearcoat != nullptr)
    {
        if (material.clearcoat->clearcoatTexture.has_value())
        {
            indices.push_back(material.clearcoat->clearcoatTexture->textureIndex);
        }
        if (material.clearcoat->clearcoatRoughnessTexture.has_value())
        {
            indices.push_back(material.clearcoat->clearcoatRoughnessTexture->textureIndex);
        }
        if (material.clearcoat->clearcoatNormalTexture.has_value())
        {
            indices.push_back(material.clearcoat->clearcoatNormalTexture->textureIndex);
        }
    }
    if (material.volume != nullptr && material.volume->thicknessTexture.has_value())
    {
        indices.push_back(material.volume->thicknessTexture->textureIndex);
    }

    return indices;
}

// Mirrors the alpha-mode translation performed inside
// GltfLoader::loadMaterial so the translation can be exercised without
// needing a GPU-backed Resources object.
static void applyAlphaFields(const fastgltf::Material& gltfMat, Material& material)
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

static bool nodeExtrasControllableFromJson(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string{json}};
    auto doc = parser.parse(padded);
    simdjson::dom::object extras;
    CHECK(doc.get_object().get(extras) == simdjson::SUCCESS);
    return GltfLoader::nodeExtrasControllable(&extras);
}

static std::optional<GltfLoader::PhysicsConfig> nodeExtrasPhysicsFromJson(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string{json}};
    auto doc = parser.parse(padded);
    simdjson::dom::object extras;
    CHECK(doc.get_object().get(extras) == simdjson::SUCCESS);
    return GltfLoader::nodeExtrasPhysics(&extras);
}

static std::optional<ClothMeshParams> nodeExtrasClothFromJson(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string{json}};
    auto doc = parser.parse(padded);
    simdjson::dom::object extras;
    CHECK(doc.get_object().get(extras) == simdjson::SUCCESS);
    return GltfLoader::nodeExtrasCloth(&extras);
}

static std::optional<fire_engine::RagdollParams> nodeExtrasRagdollFromJson(std::string_view json)
{
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string{json}};
    auto doc = parser.parse(padded);
    simdjson::dom::object extras;
    CHECK(doc.get_object().get(extras) == simdjson::SUCCESS);
    return GltfLoader::nodeExtrasRagdoll(&extras);
}

// Applies a fastgltf matrix to a Node's Transform via decomposition,
// mirroring the logic in GltfLoader::applyTRS for the matrix branch.
static void applyMatrix(const fastgltf::math::fmat4x4& mat, Node& node)
{
    fastgltf::math::fvec3 scale;
    fastgltf::math::fquat rotation;
    fastgltf::math::fvec3 translation;
    fastgltf::math::decomposeTransformMatrix(mat, scale, rotation, translation);

    node.transform().position({translation.x(), translation.y(), translation.z()});
    node.transform().rotation({rotation.x(), rotation.y(), rotation.z(), rotation.w()});
    node.transform().scale({scale.x(), scale.y(), scale.z()});
}

// ==========================================================================
// Matrix decomposition via applyMatrix (mirrors GltfLoader::applyTRS)
// ==========================================================================

TEST_CASE("MatrixDecomposition.IdentityMatrixGivesDefaultTransform", "[MatrixDecomposition]")
{
    fastgltf::math::fmat4x4 mat{};
    mat.col(0) = {1, 0, 0, 0};
    mat.col(1) = {0, 1, 0, 0};
    mat.col(2) = {0, 0, 1, 0};
    mat.col(3) = {0, 0, 0, 1};

    Node node("test");
    applyMatrix(mat, node);

    CHECK(node.transform().position().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().scale().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(node.transform().scale().y() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(node.transform().scale().z() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("MatrixDecomposition.PureTranslation", "[MatrixDecomposition]")
{
    fastgltf::math::fmat4x4 mat{};
    mat.col(0) = {1, 0, 0, 0};
    mat.col(1) = {0, 1, 0, 0};
    mat.col(2) = {0, 0, 1, 0};
    mat.col(3) = {5, 10, 15, 1};

    Node node("test");
    applyMatrix(mat, node);

    CHECK(node.transform().position().x() == Catch::Approx(5.0f).margin(1e-5f));
    CHECK(node.transform().position().y() == Catch::Approx(10.0f).margin(1e-5f));
    CHECK(node.transform().position().z() == Catch::Approx(15.0f).margin(1e-5f));
    CHECK(node.transform().scale().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(node.transform().scale().y() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(node.transform().scale().z() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("MatrixDecomposition.PureScale", "[MatrixDecomposition]")
{
    fastgltf::math::fmat4x4 mat{};
    mat.col(0) = {2, 0, 0, 0};
    mat.col(1) = {0, 3, 0, 0};
    mat.col(2) = {0, 0, 4, 0};
    mat.col(3) = {0, 0, 0, 1};

    Node node("test");
    applyMatrix(mat, node);

    CHECK(node.transform().position().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().scale().x() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(node.transform().scale().y() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(node.transform().scale().z() == Catch::Approx(4.0f).margin(1e-5f));
}

TEST_CASE("MatrixDecomposition.TranslationAndScale", "[MatrixDecomposition]")
{
    fastgltf::math::fmat4x4 mat{};
    mat.col(0) = {2, 0, 0, 0};
    mat.col(1) = {0, 3, 0, 0};
    mat.col(2) = {0, 0, 4, 0};
    mat.col(3) = {1, 2, 3, 1};

    Node node("test");
    applyMatrix(mat, node);

    CHECK(node.transform().position().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(node.transform().position().y() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(node.transform().position().z() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(node.transform().scale().x() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(node.transform().scale().y() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(node.transform().scale().z() == Catch::Approx(4.0f).margin(1e-5f));
}

TEST_CASE("MatrixDecomposition.ZUpRotationMatrix", "[MatrixDecomposition]")
{
    // From RiggedSimple.gltf Node 0 "Z_UP":
    // [1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1] (column-major)
    fastgltf::math::fmat4x4 mat{};
    mat.col(0) = {1, 0, 0, 0};
    mat.col(1) = {0, 0, -1, 0};
    mat.col(2) = {0, 1, 0, 0};
    mat.col(3) = {0, 0, 0, 1};

    Node node("Z_UP");
    applyMatrix(mat, node);

    CHECK(node.transform().position().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().scale().x() == Catch::Approx(1.0f).margin(1e-4f));
    CHECK(node.transform().scale().y() == Catch::Approx(1.0f).margin(1e-4f));
    CHECK(node.transform().scale().z() == Catch::Approx(1.0f).margin(1e-4f));

    // Rotation should be non-identity (this is a 90° rotation around X)
    auto rot = node.transform().rotation();
    CHECK((std::abs(rot.x()) > 0.01f || std::abs(rot.y()) > 0.01f || std::abs(rot.z()) > 0.01f));
}

TEST_CASE("MatrixDecomposition.ArmatureRotationMatrix", "[MatrixDecomposition]")
{
    // From RiggedSimple.gltf Node 1 "Armature":
    // [-4.37e-08,-1,0,0, 1,-4.37e-08,0,0, 0,0,1,0, 0,0,0,1]
    float eps = -4.371139894487897e-08f;
    fastgltf::math::fmat4x4 mat{};
    mat.col(0) = {eps, 1, 0, 0};
    mat.col(1) = {-1, eps, 0, 0};
    mat.col(2) = {0, 0, 1, 0};
    mat.col(3) = {0, 0, 0, 1};

    Node node("Armature");
    applyMatrix(mat, node);

    CHECK(node.transform().position().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().scale().x() == Catch::Approx(1.0f).margin(1e-4f));
    CHECK(node.transform().scale().y() == Catch::Approx(1.0f).margin(1e-4f));
    CHECK(node.transform().scale().z() == Catch::Approx(1.0f).margin(1e-4f));
}

TEST_CASE("MatrixDecomposition.BoneTranslationMatrix", "[MatrixDecomposition]")
{
    // From RiggedSimple.gltf Node 3 "Bone":
    // [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,-1.36e-07,-4.18,1]
    fastgltf::math::fmat4x4 mat{};
    mat.col(0) = {1, 0, 0, 0};
    mat.col(1) = {0, 1, 0, 0};
    mat.col(2) = {0, 0, 1, 0};
    mat.col(3) = {0, -1.3597299641787688e-07f, -4.1803297996521f, 1};

    Node node("Bone");
    applyMatrix(mat, node);

    CHECK(node.transform().position().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(node.transform().position().y() ==
          Catch::Approx(-1.3597299641787688e-07f).margin(1e-10f));
    CHECK(node.transform().position().z() == Catch::Approx(-4.1803297996521f).margin(1e-3f));
    CHECK(node.transform().scale().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(node.transform().scale().y() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(node.transform().scale().z() == Catch::Approx(1.0f).margin(1e-5f));
}

// ==========================================================================
// Verify Transform::update produces correct world matrix after decomposition
// ==========================================================================

TEST_CASE("MatrixDecomposition.UpdateProducesCorrectWorldTranslation", "[MatrixDecomposition]")
{
    fastgltf::math::fmat4x4 mat{};
    mat.col(0) = {1, 0, 0, 0};
    mat.col(1) = {0, 1, 0, 0};
    mat.col(2) = {0, 0, 1, 0};
    mat.col(3) = {3, 7, 11, 1};

    Node node("test");
    applyMatrix(mat, node);
    node.transform().update(Mat4::identity());

    CHECK((node.transform().world()[0, 3]) == Catch::Approx(3.0f).margin(1e-5f));
    CHECK((node.transform().world()[1, 3]) == Catch::Approx(7.0f).margin(1e-5f));
    CHECK((node.transform().world()[2, 3]) == Catch::Approx(11.0f).margin(1e-5f));
}

// ==========================================================================
// TRS rotation round-trip — the loader must store the glTF quaternion
// verbatim on the Node's Transform, so the rendered rotation matches the
// source asset without passing through an intermediate Euler representation.
// ==========================================================================

TEST_CASE("TRSRotation.DecalBlendQuaternionRoundTrip", "[TRSRotation]")
{
    // AlphaBlendModeTest DecalBlend/DecalOpaque: pure X-axis rotation of ~-56°.
    // Applying the quaternion directly must land the rotation on the X axis,
    // not permute it onto Z as the old Euler path did.
    fastgltf::Node gltf;
    fastgltf::TRS trsInit{};
    trsInit.translation = fastgltf::math::fvec3{0.0f, 0.0f, 0.4090209901332855f};
    trsInit.rotation =
        fastgltf::math::fquat{-0.47185850143432617f, 0.0f, 0.0f, 0.88167440891265869f};
    trsInit.scale = fastgltf::math::fvec3{1.0f, 1.0f, 1.0f};
    gltf.transform = trsInit;

    Node node("DecalBlend");
    // Mirror the TRS branch of GltfLoader::applyTRS inline so this test
    // exercises the stored-quaternion contract without needing access to the
    // private static.
    auto* trs = std::get_if<fastgltf::TRS>(&gltf.transform);
    REQUIRE(trs != nullptr);
    node.transform().position({trs->translation.x(), trs->translation.y(), trs->translation.z()});
    node.transform().rotation(
        {trs->rotation.x(), trs->rotation.y(), trs->rotation.z(), trs->rotation.w()});
    node.transform().scale({trs->scale.x(), trs->scale.y(), trs->scale.z()});

    auto rot = node.transform().rotation();
    CHECK(rot.x() == Catch::Approx(-0.47185850143432617f).margin(1e-5f));
    CHECK(rot.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(rot.z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(rot.w() == Catch::Approx(0.88167440891265869f).margin(1e-5f));

    // Extrinsic-XYZ Euler extraction must place the rotation on the X axis.
    auto e = rot.toEulerXYZ();
    CHECK(e.x() == Catch::Approx(-0.98279f).margin(1e-4f));
    CHECK(e.y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(e.z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("GltfFixture.MinimalTriangleFixtureParses", "[GltfFixture]")
{
    auto gltfPath = std::filesystem::path("test_assets/minimal_triangle.gltf");

    fastgltf::Parser parser;
    auto dataResult = fastgltf::GltfDataBuffer::FromPath(gltfPath);
    REQUIRE(dataResult.error() == fastgltf::Error::None);

    auto result =
        parser.loadGltf(dataResult.get(), gltfPath.parent_path(), fastgltf::Options::None);
    REQUIRE(result.error() == fastgltf::Error::None);

    const auto& asset = result.get();
    REQUIRE(asset.scenes.size() == 1u);
    REQUIRE(asset.nodes.size() == 1u);
    REQUIRE(asset.meshes.size() == 1u);
    REQUIRE(asset.meshes[0].primitives.size() == 1u);
    const auto& primitive = asset.meshes[0].primitives[0];
    CHECK(primitive.indicesAccessor.has_value() == false);
    const auto position = primitive.findAttribute("POSITION");
    REQUIRE(position != primitive.attributes.end());
    CHECK(position->accessorIndex == 0u);
    REQUIRE(asset.accessors.size() == 1u);
    CHECK(asset.accessors[0].count == 3u);
}

TEST_CASE("GltfNodeExtras.ControllableTrueIsEnabled", "[GltfNodeExtras]")
{
    CHECK(nodeExtrasControllableFromJson(R"({"Controllable":true})"));
}

TEST_CASE("GltfNodeExtras.ControllableFalseIsIgnored", "[GltfNodeExtras]")
{
    CHECK_FALSE(nodeExtrasControllableFromJson(R"({"Controllable":false})"));
}

TEST_CASE("GltfNodeExtras.MissingControllableIsIgnored", "[GltfNodeExtras]")
{
    CHECK_FALSE(nodeExtrasControllableFromJson(R"({"Physics":{"Layer":1}})"));
}

TEST_CASE("GltfNodeExtras.NonBooleanControllableIsIgnored", "[GltfNodeExtras]")
{
    CHECK_FALSE(nodeExtrasControllableFromJson(R"({"Controllable":"true"})"));
}

TEST_CASE("GltfNodeExtras.MissingPhysicsIsIgnored", "[GltfNodeExtras]")
{
    auto config = nodeExtrasPhysicsFromJson(R"({"Controllable":true})");
    CHECK_FALSE(config.has_value());
}

TEST_CASE("GltfNodeExtras.PhysicsRigidBodyFieldsAreParsed", "[GltfNodeExtras]")
{
    auto config = nodeExtrasPhysicsFromJson(
        R"({"Physics":{"BodyType":"Dynamic","Layer":1,"Mask":10,"Velocity":[1.0,0.5,-2.0],"Mass":2.5,"Restitution":0.25,"Friction":0.75,"GravityScale":0.0}})");

    REQUIRE(config.has_value());
    CHECK(config->bodyType == PhysicsBodyType::Dynamic);
    CHECK(config->layer == 1u);
    CHECK(config->mask == 10u);
    CHECK(config->velocity == Vec3(1.0f, 0.5f, -2.0f));
    CHECK(config->mass == Catch::Approx(2.5f).margin(1e-5f));
    CHECK(config->restitution == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(config->friction == Catch::Approx(0.75f).margin(1e-5f));
    CHECK(config->gravityScale == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("GltfNodeExtras.PhysicsShapeFieldsAreParsed", "[GltfNodeExtras]")
{
    auto config = nodeExtrasPhysicsFromJson(
        R"({"Physics":{"Shape":"Sphere","Radius":3.0,"Center":[1,2,3]}})");

    REQUIRE(config.has_value());
    REQUIRE(config->shape.has_value());
    REQUIRE(std::holds_alternative<SphereShape>(config->shape.value()));
    const auto shape = std::get<SphereShape>(config->shape.value());
    CHECK(shape.radius == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(shape.center == Vec3(1.0f, 2.0f, 3.0f));
}

TEST_CASE("GltfNodeExtras.InvalidPhysicsObjectThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasPhysicsFromJson(R"({"Physics":true})"), std::runtime_error);
}

TEST_CASE("GltfNodeExtras.InvalidPhysicsBodyTypeThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasPhysicsFromJson(R"({"Physics":{"BodyType":"Actor"}})"),
                    std::runtime_error);
}

TEST_CASE("GltfNodeExtras.OutOfRangePhysicsMaskThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasPhysicsFromJson(R"({"Physics":{"Mask":4294967296}})"),
                    std::runtime_error);
}

TEST_CASE("GltfNodeExtras.VelocityWithWrongCountThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasPhysicsFromJson(R"({"Physics":{"Velocity":[1.0,0.0]}})"),
                    std::runtime_error);
    CHECK_THROWS_AS(nodeExtrasPhysicsFromJson(R"({"Physics":{"Velocity":[1.0,0.0,0.0,0.0]}})"),
                    std::runtime_error);
}

TEST_CASE("GltfNodeExtras.MissingClothIsIgnored", "[GltfNodeExtras]")
{
    auto config = nodeExtrasClothFromJson(R"({"Physics":{"BodyType":"Static"}})");
    CHECK_FALSE(config.has_value());
}

TEST_CASE("GltfNodeExtras.ClothFieldsAreParsed", "[GltfNodeExtras]")
{
    auto config = nodeExtrasClothFromJson(
        R"({"Cloth":{"Pin":"TopEdge","Compliance":0.001,"BendCompliance":0.0002}})");

    REQUIRE(config.has_value());
    CHECK(config->pin == ClothMeshParams::Pin::TopEdge);
    CHECK(config->structuralCompliance == Catch::Approx(0.001f).margin(1e-7f));
    CHECK(config->bendCompliance == Catch::Approx(0.0002f).margin(1e-7f));
}

TEST_CASE("GltfNodeExtras.ClothDefaultsWhenFieldsOmitted", "[GltfNodeExtras]")
{
    auto config = nodeExtrasClothFromJson(R"({"Cloth":{}})");

    REQUIRE(config.has_value());
    CHECK(config->pin == ClothMeshParams::Pin::None);
    CHECK(config->structuralCompliance == Catch::Approx(0.0f).margin(1e-7f));
    // Bend defaults to the soft authored value, not zero.
    CHECK(config->bendCompliance > 0.0f);
}

TEST_CASE("GltfNodeExtras.ClothPinValuesParse", "[GltfNodeExtras]")
{
    CHECK(nodeExtrasClothFromJson(R"({"Cloth":{"Pin":"None"}})")->pin ==
          ClothMeshParams::Pin::None);
    CHECK(nodeExtrasClothFromJson(R"({"Cloth":{"Pin":"TopCorners"}})")->pin ==
          ClothMeshParams::Pin::TopCorners);
}

TEST_CASE("GltfNodeExtras.InvalidClothObjectThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasClothFromJson(R"({"Cloth":true})"), std::runtime_error);
}

TEST_CASE("GltfNodeExtras.InvalidClothPinThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasClothFromJson(R"({"Cloth":{"Pin":"Middle"}})"), std::runtime_error);
}

TEST_CASE("GltfNodeExtras.NoRagdollExtrasReturnsNullopt", "[GltfNodeExtras]")
{
    CHECK_FALSE(nodeExtrasRagdollFromJson(R"({"Physics":{"BodyType":"Dynamic"}})").has_value());
}

TEST_CASE("GltfNodeExtras.EmptyRagdollUsesDefaults", "[GltfNodeExtras]")
{
    const auto params = nodeExtrasRagdollFromJson(R"({"Ragdoll":{}})");
    REQUIRE(params.has_value());
    const fire_engine::RagdollParams defaults;
    CHECK(params->mass == defaults.mass);
    CHECK(params->radius == defaults.radius);
    CHECK(params->coneTwist == defaults.coneTwist);
    CHECK(params->swingLimit == defaults.swingLimit);
}

TEST_CASE("GltfNodeExtras.RagdollFieldsParse", "[GltfNodeExtras]")
{
    const auto params = nodeExtrasRagdollFromJson(
        R"({"Ragdoll":{"Mass":2.5,"Radius":0.1,"BoneLength":0.3,"ConeTwist":false,)"
        R"("SwingLimit":0.9,"TwistLimit":0.6}})");
    REQUIRE(params.has_value());
    CHECK(params->mass == Catch::Approx(2.5f));
    CHECK(params->radius == Catch::Approx(0.1f));
    CHECK(params->defaultBoneLength == Catch::Approx(0.3f));
    CHECK_FALSE(params->coneTwist);
    CHECK(params->swingLimit == Catch::Approx(0.9f));
    CHECK(params->twistLimit == Catch::Approx(0.6f));
}

TEST_CASE("GltfNodeExtras.InvalidRagdollObjectThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasRagdollFromJson(R"({"Ragdoll":true})"), std::runtime_error);
}

TEST_CASE("GltfNodeExtras.NonNumericRagdollMassThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasRagdollFromJson(R"({"Ragdoll":{"Mass":"heavy"}})"),
                    std::runtime_error);
}

TEST_CASE("GltfNodeExtras.NonNumericVelocityThrows", "[GltfNodeExtras]")
{
    CHECK_THROWS_AS(nodeExtrasPhysicsFromJson(R"({"Physics":{"Velocity":[1.0,"0",0.0]}})"),
                    std::runtime_error);
}

TEST_CASE("GltfMeshBounds.UsesPositionAccessorMinMax", "[GltfMeshBounds]")
{
    auto gltfPath = std::filesystem::path("test_assets/minimal_triangle.gltf");

    fastgltf::Parser parser;
    auto dataResult = fastgltf::GltfDataBuffer::FromPath(gltfPath);
    REQUIRE(dataResult.error() == fastgltf::Error::None);

    auto result =
        parser.loadGltf(dataResult.get(), gltfPath.parent_path(), fastgltf::Options::None);
    REQUIRE(result.error() == fastgltf::Error::None);

    const auto& asset = result.get();
    REQUIRE(asset.meshes.size() == 1u);
    auto bounds = GltfLoader::meshBounds(asset, asset.meshes[0]);
    REQUIRE(bounds.has_value());
    CHECK(bounds->min == Vec3(0.0f, 0.0f, 0.0f));
    CHECK(bounds->max == Vec3(1.0f, 1.0f, 0.0f));
}

TEST_CASE("GltfMeshBounds.FallsBackToScanningPositionsWhenAccessorBoundsAreAbsent",
          "[GltfMeshBounds]")
{
    auto gltfPath = std::filesystem::path("test_assets/minimal_triangle.gltf");

    fastgltf::Parser parser;
    auto dataResult = fastgltf::GltfDataBuffer::FromPath(gltfPath);
    REQUIRE(dataResult.error() == fastgltf::Error::None);

    auto result = parser.loadGltf(dataResult.get(), gltfPath.parent_path(),
                                  fastgltf::Options::LoadExternalBuffers);
    REQUIRE(result.error() == fastgltf::Error::None);

    auto& asset = result.get();
    REQUIRE(asset.accessors.size() == 1u);
    REQUIRE(asset.meshes.size() == 1u);
    asset.accessors[0].min.reset();
    asset.accessors[0].max.reset();

    auto bounds = GltfLoader::meshBounds(asset, asset.meshes[0]);
    REQUIRE(bounds.has_value());
    CHECK(bounds->min == Vec3(0.0f, 0.0f, 0.0f));
    CHECK(bounds->max == Vec3(1.0f, 1.0f, 0.0f));
}

TEST_CASE("GltfMeshBounds.UnsupportedPrimitiveDoesNotCreateBounds", "[GltfMeshBounds]")
{
    auto gltfPath = std::filesystem::path("test_assets/minimal_triangle.gltf");

    fastgltf::Parser parser;
    auto dataResult = fastgltf::GltfDataBuffer::FromPath(gltfPath);
    REQUIRE(dataResult.error() == fastgltf::Error::None);

    auto result =
        parser.loadGltf(dataResult.get(), gltfPath.parent_path(), fastgltf::Options::None);
    REQUIRE(result.error() == fastgltf::Error::None);

    auto& asset = result.get();
    REQUIRE(asset.meshes.size() == 1u);
    REQUIRE(asset.meshes[0].primitives.size() == 1u);
    asset.meshes[0].primitives[0].type = fastgltf::PrimitiveType::Lines;

    auto bounds = GltfLoader::meshBounds(asset, asset.meshes[0]);
    CHECK_FALSE(bounds.has_value());
}

TEST_CASE("GltfMeshBounds.MissingPositionAttributeDoesNotCreateBounds", "[GltfMeshBounds]")
{
    auto gltfPath = std::filesystem::path("test_assets/minimal_triangle.gltf");

    fastgltf::Parser parser;
    auto dataResult = fastgltf::GltfDataBuffer::FromPath(gltfPath);
    REQUIRE(dataResult.error() == fastgltf::Error::None);

    auto result =
        parser.loadGltf(dataResult.get(), gltfPath.parent_path(), fastgltf::Options::None);
    REQUIRE(result.error() == fastgltf::Error::None);

    auto& asset = result.get();
    REQUIRE(asset.meshes.size() == 1u);
    REQUIRE(asset.meshes[0].primitives.size() == 1u);
    asset.meshes[0].primitives[0].attributes.clear();

    auto bounds = GltfLoader::meshBounds(asset, asset.meshes[0]);
    CHECK_FALSE(bounds.has_value());
}

// ==========================================================================
// Smooth-normal generation — runs in the loader when a glTF primitive omits
// the NORMAL attribute (Fox.gltf, etc.). Verifies the algorithm directly so
// we don't need a GPU-backed Resources to exercise it.
// ==========================================================================

TEST_CASE("GenerateSmoothNormals.SingleTriangleProducesFaceNormal", "[GenerateSmoothNormals]")
{
    // Triangle in XY plane, CCW when viewed from +Z. Face normal is +Z.
    std::vector<Vec3> positions{
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    };
    std::vector<uint32_t> indices{0, 1, 2};

    const auto normals = GltfLoader::generateSmoothNormals(positions, indices);
    REQUIRE(normals.size() == 3u);
    for (const auto& n : normals)
    {
        CHECK(n.x() == Catch::Approx(0.0f).margin(1e-5f));
        CHECK(n.y() == Catch::Approx(0.0f).margin(1e-5f));
        CHECK(n.z() == Catch::Approx(1.0f).margin(1e-5f));
    }
}

TEST_CASE("GenerateSmoothNormals.SharedVertexAreaWeightedAverage", "[GenerateSmoothNormals]")
{
    // Two triangles sharing vertex 0 at the origin. Both normals are +Z, so
    // the shared vertex's accumulated normal is also +Z, unit length.
    std::vector<Vec3> positions{
        {0.0f, 0.0f, 0.0f}, // shared
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f},
    };
    std::vector<uint32_t> indices{
        0, 1, 2, // CCW from +Z, normal +Z
        0, 2, 3, // CCW from +Z, normal +Z
    };

    const auto normals = GltfLoader::generateSmoothNormals(positions, indices);
    REQUIRE(normals.size() == 4u);
    CHECK(normals[0].z() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(std::sqrt(normals[0].x() * normals[0].x() + normals[0].y() * normals[0].y() +
                    normals[0].z() * normals[0].z()) == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("GenerateSmoothNormals.UnreferencedVertexFallsBackToUp", "[GenerateSmoothNormals]")
{
    // Three real triangle verts plus a stray fourth that no triangle uses.
    // The stray's accumulated normal is zero; the function must not divide
    // by zero and instead emits the documented up-pointing fallback.
    std::vector<Vec3> positions{
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {99.0f, 99.0f, 99.0f}, // unreferenced
    };
    std::vector<uint32_t> indices{0, 1, 2};

    const auto normals = GltfLoader::generateSmoothNormals(positions, indices);
    REQUIRE(normals.size() == 4u);
    CHECK(normals[3].x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(normals[3].y() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(normals[3].z() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("GenerateSmoothNormals.OutOfRangeIndicesAreSkipped", "[GenerateSmoothNormals]")
{
    // Malformed mesh: an index references a vertex that doesn't exist.
    // Function must skip the bad triangle without UB or crash; remaining
    // triangle still contributes.
    std::vector<Vec3> positions{
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    };
    std::vector<uint32_t> indices{
        0, 1, 2,  // good
        0, 1, 99, // bad — index 99 out of range
    };

    const auto normals = GltfLoader::generateSmoothNormals(positions, indices);
    REQUIRE(normals.size() == 3u);
    for (const auto& n : normals)
    {
        CHECK(n.z() == Catch::Approx(1.0f).margin(1e-5f));
    }
}

// ==========================================================================
// Alpha fields — glTF alphaMode / alphaCutoff / doubleSided must land on the
// engine's Material verbatim so the renderer can route to the right pipeline.
// ==========================================================================

TEST_CASE("MaterialAlphaFields.OpaqueIsDefault", "[MaterialAlphaFields]")
{
    fastgltf::Material gltfMat{};
    Material material;
    applyAlphaFields(gltfMat, material);
    CHECK(material.alphaMode() == AlphaMode::Opaque);
    CHECK_FALSE(material.doubleSided());
    // fastgltf default alphaCutoff is 0.5f per the glTF spec.
    CHECK(material.alphaCutoff() == Catch::Approx(0.5f).margin(1e-5f));
}

TEST_CASE("MaterialAlphaFields.MaskWithCustomCutoff", "[MaterialAlphaFields]")
{
    fastgltf::Material gltfMat{};
    gltfMat.alphaMode = fastgltf::AlphaMode::Mask;
    gltfMat.alphaCutoff = 0.25f;
    gltfMat.doubleSided = true;
    Material material;
    applyAlphaFields(gltfMat, material);
    CHECK(material.alphaMode() == AlphaMode::Mask);
    CHECK(material.alphaCutoff() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(material.doubleSided());
}

TEST_CASE("MaterialAlphaFields.BlendMapsThrough", "[MaterialAlphaFields]")
{
    fastgltf::Material gltfMat{};
    gltfMat.alphaMode = fastgltf::AlphaMode::Blend;
    gltfMat.doubleSided = true;
    Material material;
    applyAlphaFields(gltfMat, material);
    CHECK(material.alphaMode() == AlphaMode::Blend);
    CHECK(material.doubleSided());
}

// ==========================================================================
// Occlusion strength — glTF spec default is 1.0; explicit values must round
// trip through GltfLoader::loadMaterial onto Material::occlusionStrength().
// ==========================================================================

TEST_CASE("MaterialOcclusionStrength.DefaultsToOne", "[MaterialOcclusionStrength]")
{
    Material material;
    CHECK(material.occlusionStrength() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("MaterialOcclusionStrength.AbsentTextureLeavesStrengthUnchanged",
          "[MaterialOcclusionStrength]")
{
    fastgltf::Material gltfMat{};
    Material material;
    material.occlusionStrength(0.42f);
    applyOcclusionStrength(gltfMat, material);
    CHECK(material.occlusionStrength() == Catch::Approx(0.42f).margin(1e-5f));
}

TEST_CASE("MaterialOcclusionStrength.ExplicitStrengthRoundTrips", "[MaterialOcclusionStrength]")
{
    fastgltf::Material gltfMat{};
    gltfMat.occlusionTexture.emplace().strength = 0.5f;
    Material material;
    applyOcclusionStrength(gltfMat, material);
    CHECK(material.occlusionStrength() == Catch::Approx(0.5f).margin(1e-5f));
}

// ==========================================================================
// KHR_materials_emissive_strength — emissiveFactor is multiplied by the
// extension's strength scalar at load time so HDR emissives reach the bloom
// chain at the authored magnitude.
// ==========================================================================

TEST_CASE("EmissiveStrength.DefaultStrengthIsIdentity", "[EmissiveStrength]")
{
    fastgltf::Material gltfMat{};
    gltfMat.emissiveFactor = {0.5f, 0.25f, 0.125f};
    // emissiveStrength defaults to 1.0 per the extension spec.
    Material material;
    applyEmissiveStrength(gltfMat, material);
    CHECK(material.emissive().r() == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(material.emissive().g() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(material.emissive().b() == Catch::Approx(0.125f).margin(1e-5f));
}

TEST_CASE("EmissiveStrength.ExplicitStrengthScalesEmissiveFactor", "[EmissiveStrength]")
{
    fastgltf::Material gltfMat{};
    gltfMat.emissiveFactor = {1.0f, 0.5f, 0.25f};
    gltfMat.emissiveStrength = 4.0f;
    Material material;
    applyEmissiveStrength(gltfMat, material);
    CHECK(material.emissive().r() == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(material.emissive().g() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(material.emissive().b() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("EmissiveStrength.ZeroStrengthZeroesEmission", "[EmissiveStrength]")
{
    fastgltf::Material gltfMat{};
    gltfMat.emissiveFactor = {1.0f, 1.0f, 1.0f};
    gltfMat.emissiveStrength = 0.0f;
    Material material;
    applyEmissiveStrength(gltfMat, material);
    CHECK(material.emissive().r() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(material.emissive().g() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(material.emissive().b() == Catch::Approx(0.0f).margin(1e-5f));
}

// ==========================================================================
// extensionsRequired — spec MUST refuse to load assets needing extensions we
// don't support. Helper is exposed on GltfLoader so we can hit it without a
// full asset.
// ==========================================================================

TEST_CASE("EnsureSupportedExtensions.EmptyVectorAccepts", "[EnsureSupportedExtensions]")
{
    std::vector<std::string_view> required;
    CHECK_NOTHROW(GltfLoader::ensureSupportedExtensions(required));
}

TEST_CASE("EnsureSupportedExtensions.SupportedExtensionAccepts", "[EnsureSupportedExtensions]")
{
    std::vector<std::string_view> required{"KHR_materials_emissive_strength"};
    CHECK_NOTHROW(GltfLoader::ensureSupportedExtensions(required));
}

TEST_CASE("EnsureSupportedExtensions.KtxBasisuAccepts", "[EnsureSupportedExtensions]")
{
    std::vector<std::string_view> required{"KHR_texture_basisu"};
    CHECK_NOTHROW(GltfLoader::ensureSupportedExtensions(required));
}

TEST_CASE("EnsureSupportedExtensions.MaterialsVariantsAccepts", "[EnsureSupportedExtensions]")
{
    std::vector<std::string_view> required{"KHR_materials_variants"};
    CHECK_NOTHROW(GltfLoader::ensureSupportedExtensions(required));
}

TEST_CASE("EnsureSupportedExtensions.UnsupportedExtensionThrows", "[EnsureSupportedExtensions]")
{
    std::vector<std::string_view> required{"KHR_draco_mesh_compression"};
    try
    {
        GltfLoader::ensureSupportedExtensions(required);
        FAIL("expected ensureSupportedExtensions to throw");
    }
    catch (const std::runtime_error& e)
    {
        CHECK(std::string(e.what()).find("KHR_draco_mesh_compression") != std::string::npos);
    }
}

TEST_CASE("EnsureSupportedExtensions.MixedThrowsListingOnlyUnsupported",
          "[EnsureSupportedExtensions]")
{
    std::vector<std::string_view> required{"KHR_materials_emissive_strength",
                                           "KHR_mesh_quantization", "KHR_materials_variants"};
    try
    {
        GltfLoader::ensureSupportedExtensions(required);
        FAIL("expected ensureSupportedExtensions to throw");
    }
    catch (const std::runtime_error& e)
    {
        const std::string what(e.what());
        CHECK(what.find("KHR_mesh_quantization") != std::string::npos);
        // Supported extensions must not be listed in the unsupported message.
        CHECK(what.find("KHR_materials_variants") == std::string::npos);
        CHECK(what.find("KHR_materials_emissive_strength") == std::string::npos);
    }
}

TEST_CASE("EnsureSupportedExtensions.ErrorListsEveryUnsupportedExtension",
          "[EnsureSupportedExtensions]")
{
    std::vector<std::string_view> required{"KHR_mesh_quantization", "KHR_draco_mesh_compression"};
    try
    {
        GltfLoader::ensureSupportedExtensions(required);
        FAIL("expected ensureSupportedExtensions to throw");
    }
    catch (const std::runtime_error& e)
    {
        const std::string what(e.what());
        CHECK(what.find("KHR_mesh_quantization") != std::string::npos);
        CHECK(what.find("KHR_draco_mesh_compression") != std::string::npos);
    }
}

TEST_CASE("EnsureSupportedExtensions.StainedGlassLampKtxAndVariantsExtensionsNowAccept",
          "[EnsureSupportedExtensions]")
{
    REQUIRE(std::filesystem::exists("StainedGlassLampKTX/StainedGlassLamp.gltf"));

    std::ifstream file("StainedGlassLampKTX/StainedGlassLamp.gltf");
    REQUIRE(file.is_open());
    const std::string json((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);

    std::vector<std::string_view> required;
    std::vector<std::string_view> used;
    simdjson::dom::array requiredExtensions;
    REQUIRE(doc["extensionsRequired"].get(requiredExtensions) == simdjson::SUCCESS);
    for (const auto extValue : requiredExtensions)
    {
        std::string_view ext;
        REQUIRE(extValue.get(ext) == simdjson::SUCCESS);
        required.emplace_back(ext);
    }

    simdjson::dom::array usedExtensions;
    REQUIRE(doc["extensionsUsed"].get(usedExtensions) == simdjson::SUCCESS);
    for (const auto extValue : usedExtensions)
    {
        std::string_view ext;
        REQUIRE(extValue.get(ext) == simdjson::SUCCESS);
        used.emplace_back(ext);
    }

    bool hasBasisuTexture = false;
    simdjson::dom::array textures;
    REQUIRE(doc["textures"].get(textures) == simdjson::SUCCESS);
    for (const auto textureValue : textures)
    {
        simdjson::dom::object textureObject;
        REQUIRE(textureValue.get(textureObject) == simdjson::SUCCESS);

        simdjson::dom::element extensionsElement;
        if (textureObject["extensions"].get(extensionsElement) != simdjson::SUCCESS)
        {
            continue;
        }

        simdjson::dom::element basisuElement;
        if (extensionsElement["KHR_texture_basisu"].get(basisuElement) == simdjson::SUCCESS)
        {
            hasBasisuTexture = true;
            break;
        }
    }
    REQUIRE(hasBasisuTexture);
    REQUIRE(std::find(required.begin(), required.end(), std::string_view{"KHR_texture_basisu"}) !=
            required.end());
    REQUIRE(std::find(used.begin(), used.end(), std::string_view{"KHR_materials_variants"}) !=
            used.end());

    CHECK_NOTHROW(GltfLoader::ensureSupportedExtensions(required));
}

TEST_CASE("ParseAssetKtxBasisu.StainedGlassLampTexturesUseBasisuSources", "[ParseAssetKtxBasisu]")
{
    const auto asset =
        parseRealGltfAsset(std::filesystem::path("StainedGlassLampKTX") / "StainedGlassLamp.gltf");

    REQUIRE_FALSE(asset.textures.empty());
    for (const auto& texture : asset.textures)
    {
        CHECK(texture.basisuImageIndex.has_value());
        CHECK_FALSE(texture.imageIndex.has_value());
    }
}

TEST_CASE("ParseAssetKtxBasisu.StainedGlassLampMaterialsReferenceBasisuTextures",
          "[ParseAssetKtxBasisu]")
{
    const auto asset =
        parseRealGltfAsset(std::filesystem::path("StainedGlassLampKTX") / "StainedGlassLamp.gltf");

    REQUIRE_FALSE(asset.materials.empty());
    bool sawReferencedTexture = false;
    for (const auto& material : asset.materials)
    {
        for (const std::size_t textureIndex : referencedTextureIndices(material))
        {
            REQUIRE(textureIndex < asset.textures.size());
            CHECK(asset.textures[textureIndex].basisuImageIndex.has_value());
            CHECK_FALSE(asset.textures[textureIndex].imageIndex.has_value());
            sawReferencedTexture = true;
        }
    }
    CHECK(sawReferencedTexture);
}

TEST_CASE("ParseAssetImageSources.TextureSettingsTestUsesLegacyImageIndices",
          "[ParseAssetImageSources]")
{
    const auto asset = parseRealGltfAsset(std::filesystem::path("TextureSettingsTest") /
                                          "TextureSettingsTest.gltf");

    REQUIRE_FALSE(asset.textures.empty());
    for (const auto& texture : asset.textures)
    {
        CHECK(texture.imageIndex.has_value());
        CHECK_FALSE(texture.basisuImageIndex.has_value());
    }
}

// ==========================================================================
// Primitive mode — vertex layout / index buffer assume triangles. Anything
// else gets skipped (with a warning at load time).
// ==========================================================================

TEST_CASE("SupportedPrimitiveType.TrianglesIsSupported", "[SupportedPrimitiveType]")
{
    CHECK(GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType::Triangles));
}

TEST_CASE("SupportedPrimitiveType.AllOtherModesAreSkipped", "[SupportedPrimitiveType]")
{
    CHECK_FALSE(GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType::Points));
    CHECK_FALSE(GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType::Lines));
    CHECK_FALSE(GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType::LineLoop));
    CHECK_FALSE(GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType::LineStrip));
    CHECK_FALSE(GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType::TriangleStrip));
    CHECK_FALSE(GltfLoader::isSupportedPrimitiveType(fastgltf::PrimitiveType::TriangleFan));
}

// ==========================================================================
// Per-slot UV-set indices — each glTF TextureInfo names a `texCoord` index
// that the loader must record on Material so the shader samples the right
// vertex stream.
// ==========================================================================

TEST_CASE("MaterialTexCoordIndices.AbsentTexturesLeaveDefaultsZero", "[MaterialTexCoordIndices]")
{
    fastgltf::Material gltfMat{};
    Material material;
    applyTexCoordIndices(gltfMat, material);
    CHECK(material.texture(MaterialTextureSlot::BaseColour).texCoord == 0);
    CHECK(material.texture(MaterialTextureSlot::Emissive).texCoord == 0);
    CHECK(material.texture(MaterialTextureSlot::Normal).texCoord == 0);
    CHECK(material.texture(MaterialTextureSlot::MetallicRoughness).texCoord == 0);
    CHECK(material.texture(MaterialTextureSlot::Occlusion).texCoord == 0);
}

TEST_CASE("MaterialTexCoordIndices.ExplicitTexCoordOnesRoundTrip", "[MaterialTexCoordIndices]")
{
    fastgltf::Material gltfMat{};
    gltfMat.pbrData.baseColorTexture.emplace().texCoordIndex = 1;
    gltfMat.emissiveTexture.emplace().texCoordIndex = 1;
    gltfMat.normalTexture.emplace().texCoordIndex = 1;
    gltfMat.pbrData.metallicRoughnessTexture.emplace().texCoordIndex = 1;
    gltfMat.occlusionTexture.emplace().texCoordIndex = 1;
    Material material;
    applyTexCoordIndices(gltfMat, material);
    CHECK(material.texture(MaterialTextureSlot::BaseColour).texCoord == 1);
    CHECK(material.texture(MaterialTextureSlot::Emissive).texCoord == 1);
    CHECK(material.texture(MaterialTextureSlot::Normal).texCoord == 1);
    CHECK(material.texture(MaterialTextureSlot::MetallicRoughness).texCoord == 1);
    CHECK(material.texture(MaterialTextureSlot::Occlusion).texCoord == 1);
}

TEST_CASE("MaterialTexCoordIndices.MixedSlotsRoundTrip", "[MaterialTexCoordIndices]")
{
    // Common real-world layout: occlusion baked onto its own UV set.
    fastgltf::Material gltfMat{};
    gltfMat.pbrData.baseColorTexture.emplace().texCoordIndex = 0;
    gltfMat.occlusionTexture.emplace().texCoordIndex = 1;
    Material material;
    applyTexCoordIndices(gltfMat, material);
    CHECK(material.texture(MaterialTextureSlot::BaseColour).texCoord == 0);
    CHECK(material.texture(MaterialTextureSlot::Occlusion).texCoord == 1);
}

// ==========================================================================
// KHR_texture_transform — when present, fastgltf parses uvOffset/uvScale/
// rotation onto TextureInfo; loader copies them onto the per-slot Material
// UvTransform. Absent → identity.
// ==========================================================================

TEST_CASE("UvTransformDefault.FieldsAreIdentity", "[UvTransformDefault]")
{
    fire_engine::UvTransform t;
    CHECK(t.offsetX == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.offsetY == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.scaleX == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(t.scaleY == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(t.rotation == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("UvTransformLoader.AbsentExtensionGivesIdentity", "[UvTransformLoader]")
{
    fastgltf::TextureInfo info{};
    auto t = readUvTransform(info);
    CHECK(t.offsetX == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(t.scaleX == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(t.rotation == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("UvTransformLoader.ExplicitTransformRoundTrips", "[UvTransformLoader]")
{
    fastgltf::TextureInfo info{};
    info.transform = std::make_unique<fastgltf::TextureTransform>();
    info.transform->uvOffset = {0.25f, 0.5f};
    info.transform->uvScale = {2.0f, 3.0f};
    info.transform->rotation = 1.5f;
    auto t = readUvTransform(info);
    CHECK(t.offsetX == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(t.offsetY == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(t.scaleX == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(t.scaleY == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(t.rotation == Catch::Approx(1.5f).margin(1e-5f));
}

// ==========================================================================
// KHR_materials_unlit — bool round-trips through fastgltf onto Material.
// Default false; renderer's lighting path is the same as before.
// ==========================================================================

TEST_CASE("MaterialUnlit.DefaultIsLit", "[MaterialUnlit]")
{
    Material material;
    CHECK_FALSE(material.unlit());
}

TEST_CASE("MaterialUnlit.GltfFlagPropagates", "[MaterialUnlit]")
{
    fastgltf::Material gltfMat{};
    gltfMat.unlit = true;
    Material material;
    material.unlit(gltfMat.unlit);
    CHECK(material.unlit());
}

TEST_CASE("MaterialUnlit.AbsentFlagLeavesLit", "[MaterialUnlit]")
{
    fastgltf::Material gltfMat{};
    Material material;
    material.unlit(gltfMat.unlit);
    CHECK_FALSE(material.unlit());
}

// ==========================================================================
// Vertex colour extraction (mirrors GltfLoader::extractPrimitive)
// ==========================================================================

// Mirrors the per-vertex colour logic: when COLOR_0 data is present, use it;
// when absent, default to white (1,1,1) per the glTF spec.
static fire_engine::Colour3 extractVertexColour(const std::vector<fastgltf::math::fvec4>& colors,
                                                std::size_t index)
{
    if (index < colors.size())
    {
        return {colors[index].x(), colors[index].y(), colors[index].z()};
    }
    return {1.0f, 1.0f, 1.0f};
}

TEST_CASE("VertexColourExtraction.DefaultsToWhiteWhenAbsent", "[VertexColourExtraction]")
{
    std::vector<fastgltf::math::fvec4> colors;
    auto c = extractVertexColour(colors, 0);
    CHECK(c.r() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(c.g() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(c.b() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("VertexColourExtraction.UsesProvidedColour", "[VertexColourExtraction]")
{
    std::vector<fastgltf::math::fvec4> colors;
    colors.push_back(fastgltf::math::fvec4{0.2f, 0.4f, 0.6f, 1.0f});
    auto c = extractVertexColour(colors, 0);
    CHECK(c.r() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(c.g() == Catch::Approx(0.4f).margin(1e-5f));
    CHECK(c.b() == Catch::Approx(0.6f).margin(1e-5f));
}

TEST_CASE("VertexColourExtraction.IndexOutOfRangeFallsBackToWhite", "[VertexColourExtraction]")
{
    std::vector<fastgltf::math::fvec4> colors;
    colors.push_back(fastgltf::math::fvec4{1.0f, 0.0f, 0.0f, 1.0f});
    auto c = extractVertexColour(colors, 5);
    CHECK(c.r() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(c.g() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(c.b() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("VertexColourExtraction.IgnoresAlphaChannel", "[VertexColourExtraction]")
{
    std::vector<fastgltf::math::fvec4> colors;
    colors.push_back(fastgltf::math::fvec4{0.1f, 0.2f, 0.3f, 0.5f});
    auto c = extractVertexColour(colors, 0);
    CHECK(c.r() == Catch::Approx(0.1f).margin(1e-5f));
    CHECK(c.g() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(c.b() == Catch::Approx(0.3f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// KHR_materials_ior — fastgltf surfaces gltfMat.ior with default 1.5f when
// the extension is absent (matches the spec); the loader copies it through.
// ---------------------------------------------------------------------------

// Mirrors the loader: ior is grouped into the optional Transmission block,
// engaged when transmission is authored or ior differs from the spec default.
static void applyIor(const fastgltf::Material& gltfMat, Material& material)
{
    const float ior = static_cast<float>(gltfMat.ior);
    if (ior != 1.5f)
    {
        material.transmission(TransmissionParams{0.0f, ior});
    }
}

TEST_CASE("LoadMaterialIor.DefaultIsFifteen", "[LoadMaterialIor]")
{
    fastgltf::Material gltfMat;
    Material material;
    applyIor(gltfMat, material);
    CHECK_FALSE(material.transmission().has_value());
    CHECK(material.transmission().value_or(TransmissionParams{}).ior ==
          Catch::Approx(1.5f).margin(1e-5f));
}

TEST_CASE("LoadMaterialIor.AuthoredValuePropagates", "[LoadMaterialIor]")
{
    fastgltf::Material gltfMat;
    gltfMat.ior = 2.42f; // diamond
    Material material;
    applyIor(gltfMat, material);
    REQUIRE(material.transmission().has_value());
    CHECK(material.transmission()->ior == Catch::Approx(2.42f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// KHR_materials_clearcoat — fastgltf surfaces clearcoat through a
// unique_ptr<MaterialClearcoat>. Null when the extension is absent.
// ---------------------------------------------------------------------------

static void applyClearcoat(const fastgltf::Material& gltfMat, Material& material)
{
    if (gltfMat.clearcoat == nullptr)
    {
        return;
    }
    const auto& cc = *gltfMat.clearcoat;
    ClearcoatParams clearcoat;
    clearcoat.factor = static_cast<float>(cc.clearcoatFactor);
    clearcoat.roughness = static_cast<float>(cc.clearcoatRoughnessFactor);
    if (cc.clearcoatNormalTexture.has_value())
    {
        clearcoat.normalScale = static_cast<float>(cc.clearcoatNormalTexture->scale);
    }
    material.clearcoat(clearcoat);
}

TEST_CASE("LoadMaterialClearcoat.AbsentExtensionLeavesDefaults", "[LoadMaterialClearcoat]")
{
    fastgltf::Material gltfMat;
    Material material;
    applyClearcoat(gltfMat, material);
    CHECK_FALSE(material.clearcoat().has_value());
    const ClearcoatParams defaults = material.clearcoat().value_or(ClearcoatParams{});
    CHECK(defaults.factor == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(defaults.roughness == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(defaults.normalScale == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("LoadMaterialClearcoat.AuthoredFactorAndRoughnessPropagate", "[LoadMaterialClearcoat]")
{
    fastgltf::Material gltfMat;
    gltfMat.clearcoat = std::make_unique<fastgltf::MaterialClearcoat>();
    gltfMat.clearcoat->clearcoatFactor = 0.75f;
    gltfMat.clearcoat->clearcoatRoughnessFactor = 0.2f;
    Material material;
    applyClearcoat(gltfMat, material);
    REQUIRE(material.clearcoat().has_value());
    CHECK(material.clearcoat()->factor == Catch::Approx(0.75f).margin(1e-5f));
    CHECK(material.clearcoat()->roughness == Catch::Approx(0.2f).margin(1e-5f));
}

TEST_CASE("LoadMaterialClearcoat.NormalScalePropagates", "[LoadMaterialClearcoat]")
{
    fastgltf::Material gltfMat;
    gltfMat.clearcoat = std::make_unique<fastgltf::MaterialClearcoat>();
    auto& info = gltfMat.clearcoat->clearcoatNormalTexture.emplace();
    info.textureIndex = 0;
    info.scale = 0.4f;
    Material material;
    applyClearcoat(gltfMat, material);
    REQUIRE(material.clearcoat().has_value());
    CHECK(material.clearcoat()->normalScale == Catch::Approx(0.4f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// KHR_materials_volume — fastgltf surfaces volume via unique_ptr<MaterialVolume>;
// null when the extension is absent.
// ---------------------------------------------------------------------------

static void applyVolume(const fastgltf::Material& gltfMat, fire_engine::Material& material)
{
    if (gltfMat.volume == nullptr)
    {
        return;
    }
    const auto& vol = *gltfMat.volume;
    fire_engine::VolumeParams volume;
    volume.thicknessFactor = static_cast<float>(vol.thicknessFactor);
    volume.attenuationColor = fire_engine::Colour3{static_cast<float>(vol.attenuationColor.x()),
                                                   static_cast<float>(vol.attenuationColor.y()),
                                                   static_cast<float>(vol.attenuationColor.z())};
    volume.attenuationDistance = static_cast<float>(vol.attenuationDistance);
    material.volume(volume);
}

TEST_CASE("LoadMaterialVolume.AbsentExtensionLeavesDefaults", "[LoadMaterialVolume]")
{
    fastgltf::Material gltfMat;
    Material material;
    applyVolume(gltfMat, material);
    CHECK_FALSE(material.volume().has_value());
    const fire_engine::VolumeParams defaults =
        material.volume().value_or(fire_engine::VolumeParams{});
    CHECK(defaults.thicknessFactor == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(std::isinf(defaults.attenuationDistance));
}

TEST_CASE("LoadMaterialVolume.ThicknessFactorPropagates", "[LoadMaterialVolume]")
{
    fastgltf::Material gltfMat;
    gltfMat.volume = std::make_unique<fastgltf::MaterialVolume>();
    gltfMat.volume->thicknessFactor = 0.5f;
    Material material;
    applyVolume(gltfMat, material);
    REQUIRE(material.volume().has_value());
    CHECK(material.volume()->thicknessFactor == Catch::Approx(0.5f).margin(1e-5f));
}

TEST_CASE("LoadMaterialVolume.AttenuationColorAndDistancePropagate", "[LoadMaterialVolume]")
{
    fastgltf::Material gltfMat;
    gltfMat.volume = std::make_unique<fastgltf::MaterialVolume>();
    gltfMat.volume->attenuationColor = fastgltf::math::nvec3{0.3f, 0.5f, 0.7f};
    gltfMat.volume->attenuationDistance = 4.0f;
    Material material;
    applyVolume(gltfMat, material);
    REQUIRE(material.volume().has_value());
    CHECK(material.volume()->attenuationColor == fire_engine::Colour3(0.3f, 0.5f, 0.7f));
    CHECK(material.volume()->attenuationDistance == Catch::Approx(4.0f).margin(1e-5f));
}
