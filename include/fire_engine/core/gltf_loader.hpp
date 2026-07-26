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
class VdpmGpuRegistry;
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
    // `vdpmRegistry`, when non-null, registers every loaded geometry's forest + instance front with
    // the GPU-driven VDPM backend (Stage B5b); null loads the scene CPU-only.
    static Node* loadScene(const std::string& path, SceneGraph& scene, Resources& resources,
                           Assets& assets, PhysicsWorld& physics,
                           std::vector<ClothRegistration>* clothRegistrations = nullptr,
                           std::vector<Ragdoll>* ragdolls = nullptr,
                           VdpmGpuRegistry* vdpmRegistry = nullptr);

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
        bool isTrigger{false};
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

    // `extras.Shadow` authoring on a mesh node. Today the block carries one key,
    // `Casts` (bool): false makes the node's geometry a shadow RECEIVER only, the
    // authored equivalent of `Geometry::castsShadow(false)`. A large flat receiver that
    // casts writes its own depth into every cascade and self-shadows the whole surface,
    // so a floor authored in glTF needs this the same way the built-in `-f` plane does.
    // Returns nullopt when the node carries no Shadow extras. `Receives` is deliberately
    // NOT accepted: nothing implements it, and an authoring key the engine ignores is
    // worse than no key at all.
    [[nodiscard]]
    static std::optional<bool> nodeExtrasShadowCasts(simdjson::dom::object* extras);

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
               std::unordered_map<std::size_t, RagdollParams>* ragdollNodeConfigs = nullptr,
               std::unordered_map<std::size_t, bool>* shadowCastsNodes = nullptr);

    static void presizeAssets(const fastgltf::Asset& asset, Assets& assets);

    struct GltfLoadContext
    {
        const fastgltf::Asset& asset;
        std::string baseDir;
        Resources& resources;
        Assets& assets;
        PhysicsWorld& physics;
        std::unordered_set<std::size_t> controllableNodeIndices;
        std::unordered_map<std::size_t, PhysicsConfig> physicsNodeConfigs;
        std::unordered_map<std::size_t, ClothMeshParams> clothNodeConfigs;
        std::unordered_map<std::size_t, RagdollParams> ragdollNodeConfigs;
        // `extras.Shadow.Casts` per node index; absent means the default (casts).
        std::unordered_map<std::size_t, bool> shadowCastsNodes;
        NodeMap nodeMap;
        MeshMap meshMap;
        std::size_t nextAnimSlot{0};
        Node* activeCamera{nullptr};
        // GPU-driven VDPM registration seam (Stage B5b), threaded from loadScene. Non-null enrolls
        // each loaded geometry's forest + each instance's front with the GPU backend; null keeps
        // the scene on the CPU front. Set after construction (not a ctor arg), default null.
        VdpmGpuRegistry* vdpmRegistry{nullptr};

        // Constructor to centralize initialization and avoid missing-field-initializers warnings.
        GltfLoadContext(const fastgltf::Asset& assetRef, std::string baseDir_,
                        Resources& resourcesRef, Assets& assetsRef, PhysicsWorld& physicsRef,
                        std::unordered_set<std::size_t> controllableNodeIndices_ = {},
                        std::unordered_map<std::size_t, PhysicsConfig> physicsNodeConfigs_ = {},
                        std::unordered_map<std::size_t, ClothMeshParams> clothNodeConfigs_ = {},
                        std::unordered_map<std::size_t, RagdollParams> ragdollNodeConfigs_ = {},
                        std::unordered_map<std::size_t, bool> shadowCastsNodes_ = {})
            : asset(assetRef),
              baseDir(std::move(baseDir_)),
              resources(resourcesRef),
              assets(assetsRef),
              physics(physicsRef),
              controllableNodeIndices(std::move(controllableNodeIndices_)),
              physicsNodeConfigs(std::move(physicsNodeConfigs_)),
              clothNodeConfigs(std::move(clothNodeConfigs_)),
              ragdollNodeConfigs(std::move(ragdollNodeConfigs_)),
              shadowCastsNodes(std::move(shadowCastsNodes_)),
              nodeMap(),
              meshMap(),
              nextAnimSlot(0),
              activeCamera(nullptr)
        {
        }
    };

    class GltfSceneBuilder
    {
    public:
        explicit GltfSceneBuilder(GltfLoadContext context);

        Node* build(SceneGraph& scene, std::vector<ClothRegistration>* clothRegistrations,
                    std::vector<Ragdoll>* ragdolls);

    private:
        GltfLoadContext context_;

        void loadRootNode(SceneGraph& scene, std::size_t nodeIndex);
        void configureAnimatedNode(std::size_t nodeIndex, Node& node);
        void loadNode(std::size_t nodeIndex, Node& node);
        Mesh& attachMeshToNode(std::size_t nodeIndex, std::size_t meshIndex, Node& meshNode,
                               Node& physicsNode);
        void validatePhysicsTarget(std::size_t nodeIndex, const fastgltf::Node& gltfNode) const;
        void applyPhysicsConfig(std::size_t nodeIndex, const fastgltf::Mesh& mesh, Node& node);

        void loadSkin(std::size_t skinIndex);
        void applySkins();

        [[nodiscard]]
        // `castsShadow` comes from the attaching node's `extras.Shadow.Casts` (default true)
        // and is recorded on THIS instance's Object bindings, not on the shared Geometry — so
        // two nodes instancing one mesh can disagree and both are honoured.
        Object loadMesh(const fastgltf::Mesh& mesh, std::size_t meshIndex, bool castsShadow);

        [[nodiscard]]
        TangentGenerationResult loadGeometry(const fastgltf::Primitive& primitive,
                                             bool needsTangents, std::size_t geoIdx);

        [[nodiscard]]
        Image loadImage(std::size_t imageIndex);

        [[nodiscard]]
        KtxImage loadKtxImage(std::size_t imageIndex);

        [[nodiscard]]
        const Texture* resolveTextureIndex(std::size_t textureIndex, TextureEncoding encoding);

        void loadAnimation(std::size_t gltfAnimIndex, std::size_t nodeIndex, Animation& anim,
                           std::size_t numMorphTargets = 0);

        Node& attachCamera(Node& node);
    };

    // Node helpers
    static void applyControllable(std::size_t nodeIndex,
                                  const std::unordered_set<std::size_t>& controllableNodeIndices,
                                  Node& node);

    static void applyTRS(const fastgltf::Node& gltfNode, Node& node);

    [[nodiscard]]
    static std::string descendantMeshName(const fastgltf::Asset& asset,
                                          const fastgltf::Node& gltfNode);

    [[nodiscard]]
    static std::string nodeName(const fastgltf::Asset& asset, const fastgltf::Node& gltfNode);

    // Mesh loading
    [[nodiscard]]
    static std::optional<AABB> primitiveBounds(const fastgltf::Asset& asset,
                                               const fastgltf::Primitive& primitive);

    [[nodiscard]]
    static Material loadMaterial(const fastgltf::Asset& asset,
                                 std::optional<std::size_t> materialIndex);

    // Animation
    static void applyRestTRS(const fastgltf::Node& gltfNode, Animation& anim);

    [[nodiscard]]
    static bool nodeHasAnimation(const fastgltf::Asset& asset, std::size_t nodeIndex);

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
