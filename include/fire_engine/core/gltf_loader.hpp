#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fastgltf/core.hpp>

#include <fire_engine/animation/animation.hpp>
#include <fire_engine/collision/collider.hpp>
#include <fire_engine/core/tangent_generator.hpp>
#include <fire_engine/graphics/cloth.hpp>
#include <fire_engine/graphics/image.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/collider_shape.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/ragdoll.hpp>

namespace simdjson::dom
{
class object;
} // namespace simdjson::dom

namespace fire_engine
{

class Animator;
class Assets;
class Geometry;
class Animation;
class Material;
class Node;
class Object;
class Resources;
class SceneGraph;
class Skin;
class Texture;
enum class TextureEncoding : std::uint8_t;
class Mesh;
class KtxImage;

class GltfLoader
{
public:
    GltfLoader() = delete;

    // A cloth authored on a glTF node via `extras.Cloth`. The loader builds the
    // node's mesh as a simulated cloth (storage vertex buffer) and hands one of
    // these back per cloth node so the caller can register it with the solver —
    // `geometry` is the (Assets-owned) loaded geometry whose vertex buffer the
    // solver writes; `mesh` is the CPU cloth built from it.
    struct ClothRegistration
    {
        ClothMesh mesh;
        Geometry* geometry{nullptr};
    };

    // Returns the first imported glTF camera node, or nullptr when the scene
    // has no authored camera. The node owns an engine Camera component. When
    // `clothRegistrations` is non-null, any `extras.Cloth` nodes are appended for
    // the caller to register with the soft-body solver. When `ragdolls` is non-null,
    // any `extras.Ragdoll` node auto-builds a Ragdoll from its skin's joints (bodies
    // + joints created in `physics`, activated), appended for the caller to retain.
    static Node* loadScene(const std::string& path, SceneGraph& scene, Resources& resources,
                           Assets& assets, PhysicsWorld& physics,
                           std::vector<ClothRegistration>* clothRegistrations = nullptr,
                           std::vector<Ragdoll>* ragdolls = nullptr);

    // Synthesises per-vertex normals from a triangle mesh when the source
    // glTF lacks the NORMAL attribute. Smooth (area-weighted accumulate-and-
    // normalize) so curved meshes look right; the spec's "flat normals"
    // wording would require de-indexing and produces visibly worse results
    // on real assets like Fox.gltf.
    [[nodiscard]] static std::vector<Vec3> generateSmoothNormals(std::span<const Vec3> positions,
                                                                 std::span<const uint32_t> indices);

    // glTF 2.0 §3.2: implementations MUST refuse to load assets whose
    // `extensionsRequired` lists anything they don't support. Throws a
    // std::runtime_error naming each unsupported extension. Anything not
    // declared `required` (only `used`) is informational and ignored here.
    // Span of string_view so the call site can pass either fastgltf's
    // pmr-allocated strings or a plain std::vector from a unit test.
    static void ensureSupportedExtensions(std::span<const std::string_view> required);

    // Triangles only. Other primitive modes (POINTS / LINES / *_STRIP /
    // *_FAN) would need different vertex layout and index handling — we skip
    // the primitive with a warning rather than render garbage.
    [[nodiscard]] static bool isSupportedPrimitiveType(fastgltf::PrimitiveType type) noexcept;

    using NodeMap = std::unordered_map<std::size_t, Node*>;
    using MeshMap = std::unordered_map<std::size_t, Mesh*>;

    struct PhysicsConfig
    {
        PhysicsBodyType bodyType{PhysicsBodyType::Static};
        std::uint32_t layer{1U};
        std::uint32_t mask{~0U};
        Vec3 velocity{};
        float mass{1.0f};
        float restitution{1.0f};
        float friction{0.0f};
        float gravityScale{1.0f};
        std::optional<ColliderShape> shape;
        // `Shape: "ConvexHull"` / `"Mesh"` defer to the node mesh (built in
        // applyPhysicsConfig), since the geometry isn't available when parsing extras.
        bool convexHullFromMesh{false};
        bool staticMeshFromMesh{false};
        // `Shape: "Compound"` children (each an offset primitive). Non-empty selects a
        // compound collider over the single `shape`.
        std::vector<CompoundChild> compoundChildren;
    };

    // CPU-only mesh bounds for collision setup. Prefers POSITION accessor
    // min/max when present, falling back to scanning POSITION data.
    [[nodiscard]]
    static std::optional<AABB> meshBounds(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh);

    // Convex hull collider built from a mesh's welded POSITION vertices + triangles.
    [[nodiscard]]
    static ConvexHullShape meshConvexHull(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh);

    // Static triangle-mesh collider geometry (raw POSITION vertices + triangle indices)
    // from a node mesh — for `Shape: "Mesh"`.
    [[nodiscard]]
    static StaticMeshShape meshTriangles(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh);

    [[nodiscard]]
    static bool nodeExtrasControllable(simdjson::dom::object* extras) noexcept;

    [[nodiscard]]
    static std::optional<PhysicsConfig> nodeExtrasPhysics(simdjson::dom::object* extras);

    // `extras.Cloth` authoring parameters. Compliance/BendCompliance are numbers;
    // Pin is one of "None" / "TopCorners" / "TopEdge". Returns nullopt when the
    // node carries no Cloth extras.
    [[nodiscard]]
    static std::optional<ClothMeshParams> nodeExtrasCloth(simdjson::dom::object* extras);

    // `extras.Ragdoll` authoring parameters on a skinned node. All fields optional
    // (sensible defaults from RagdollParams); presence of the "Ragdoll" object is
    // what flags the node. ConeTwist is a bool; the rest are numbers.
    [[nodiscard]]
    static std::optional<RagdollParams> nodeExtrasRagdoll(simdjson::dom::object* extras);

private:
    // Asset parsing and setup
    [[nodiscard]]
    static fastgltf::Expected<fastgltf::Asset>
    parseAsset(const std::filesystem::path& gltfPath,
               std::unordered_set<std::size_t>* controllableNodeIndices = nullptr,
               std::unordered_map<std::size_t, PhysicsConfig>* physicsNodeConfigs = nullptr,
               std::unordered_map<std::size_t, ClothMeshParams>* clothNodeConfigs = nullptr,
               std::unordered_map<std::size_t, RagdollParams>* ragdollNodeConfigs = nullptr);

    static void presizeAssets(const fastgltf::Asset& asset, Assets& assets);

    // Node helpers
    static void applyTRS(const fastgltf::Node& gltfNode, Node& node);

    [[nodiscard]]
    static std::string descendantMeshName(const fastgltf::Asset& asset,
                                          const fastgltf::Node& gltfNode);

    [[nodiscard]]
    static std::string nodeName(const fastgltf::Asset& asset, const fastgltf::Node& gltfNode);

    static Node& attachCamera(Node& node, Node*& activeCamera);

    static void
    configureAnimatedNode(const fastgltf::Asset& asset, std::size_t nodeIndex, Node& node,
                          const std::string& baseDir, Resources& resources, Assets& assets,
                          NodeMap& nodeMap, MeshMap& meshMap, std::size_t& nextAnimSlot,
                          Node*& activeCamera,
                          const std::unordered_set<std::size_t>& controllableNodeIndices,
                          const std::unordered_map<std::size_t, PhysicsConfig>& physicsNodeConfigs,
                          PhysicsWorld& physics);

    static void loadNode(const fastgltf::Asset& asset, std::size_t nodeIndex, Node& parentNode,
                         const std::string& baseDir, Resources& resources, Assets& assets,
                         NodeMap& nodeMap, MeshMap& meshMap, std::size_t& nextAnimSlot,
                         Node*& activeCamera,
                         const std::unordered_set<std::size_t>& controllableNodeIndices,
                         const std::unordered_map<std::size_t, PhysicsConfig>& physicsNodeConfigs,
                         PhysicsWorld& physics);

    static Mesh&
    attachMeshToNode(const fastgltf::Asset& asset, std::size_t nodeIndex, std::size_t meshIndex,
                     Node& meshNode, Node& physicsNode, const std::string& baseDir,
                     Resources& resources, Assets& assets, MeshMap& meshMap,
                     const std::unordered_map<std::size_t, PhysicsConfig>& physicsNodeConfigs,
                     PhysicsWorld& physics);

    // Skin loading
    static void loadSkin(const fastgltf::Asset& asset, std::size_t skinIndex,
                         const NodeMap& nodeMap, Assets& assets);

    static void applySkins(const fastgltf::Asset& asset, const NodeMap& nodeMap,
                           const MeshMap& meshMap, Assets& assets);

    // Mesh loading
    [[nodiscard]]
    static std::optional<AABB> primitiveBounds(const fastgltf::Asset& asset,
                                               const fastgltf::Primitive& primitive);

    [[nodiscard]]
    static Object loadMesh(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh,
                           const std::string& baseDir, Resources& resources, Assets& assets,
                           std::size_t meshIndex);

    [[nodiscard]]
    static TangentGenerationResult
    loadGeometry(const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                 bool needsTangents, Resources& resources, Assets& assets, std::size_t geoIdx);

    [[nodiscard]]
    static Material loadMaterial(const fastgltf::Asset& asset,
                                 std::optional<std::size_t> materialIndex);

    [[nodiscard]]
    static Image loadImage(const fastgltf::Asset& asset, std::size_t imageIndex,
                           const std::string& baseDir);

    [[nodiscard]]
    static KtxImage loadKtxImage(const fastgltf::Asset& asset, std::size_t imageIndex,
                                 const std::string& baseDir);

    [[nodiscard]]
    static const Texture* resolveTextureIndex(const fastgltf::Asset& asset,
                                              std::size_t textureIndex, const std::string& baseDir,
                                              Resources& resources, Assets& assets,
                                              TextureEncoding encoding);

    // Animation
    static void applyRestTRS(const fastgltf::Node& gltfNode, Animation& anim);

    [[nodiscard]]
    static bool nodeHasAnimation(const fastgltf::Asset& asset, std::size_t nodeIndex);

    static void loadAnimation(const fastgltf::Asset& asset, std::size_t gltfAnimIndex,
                              std::size_t nodeIndex, Animation& anim,
                              std::size_t numMorphTargets = 0);

    [[nodiscard]]
    static float computeSharedDuration(const fastgltf::Asset& asset, std::size_t gltfAnimIndex);

    [[nodiscard]]
    static Animation::Interpolation mapInterpolation(fastgltf::AnimationInterpolation m);

    static void loadRotationChannel(const fastgltf::Asset& asset,
                                    const fastgltf::AnimationSampler& sampler, Animation& anim);

    static void loadTranslationChannel(const fastgltf::Asset& asset,
                                       const fastgltf::AnimationSampler& sampler, Animation& anim);

    static void loadScaleChannel(const fastgltf::Asset& asset,
                                 const fastgltf::AnimationSampler& sampler, Animation& anim);

    static void loadWeightChannel(const fastgltf::Asset& asset,
                                  const fastgltf::AnimationSampler& sampler, Animation& anim,
                                  std::size_t numTargets);
};

} // namespace fire_engine
