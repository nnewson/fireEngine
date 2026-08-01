#include <fire_engine/core/gltf_loader.hpp>
#include <fire_engine/core/node_component_layout.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/types.hpp>

#include <fire_engine/animation/animation.hpp>
#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/object.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/physics/physics_world.hpp>
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

std::size_t countMorphTargets(const fastgltf::Mesh& mesh)
{
    if (mesh.primitives.empty() || mesh.primitives[0].targets.empty())
    {
        return 0;
    }
    return mesh.primitives[0].targets.size();
}

std::vector<float> initialMorphWeights(const fastgltf::Mesh& mesh, std::size_t numMorphTargets)
{
    std::vector<float> weights(numMorphTargets, 0.0f);
    for (std::size_t w = 0; w < mesh.weights.size() && w < numMorphTargets; ++w)
    {
        weights[w] = mesh.weights[w];
    }
    return weights;
}

Light::Type toLightType(fastgltf::LightType t) noexcept
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

// Every direct emplacement goes through this. `materializeNodeComponentLayout` hands out targets
// that are Empty by construction, so a non-Empty one means the layout and the attach sites have
// come apart — two payloads believing they own the same node. Terminal rather than a warning,
// because the failure it replaces was silent: the second emplacement simply destroyed the first,
// and a scene rendered plausibly with a component nobody could find again.
void requireEmptyComponent(const Node& node, std::string_view what)
{
    if (!std::holds_alternative<Empty>(node.component()))
    {
        throw std::runtime_error(
            std::format("glTF node '{}': cannot attach {} — the target already holds {}",
                        node.name(), what, componentName(node.component())));
    }
}

// KHR_lights_punctual: attach the node's light to the target the layout chose. Placement is not
// decided here and must not be — see `core/node_component_layout.hpp` for why the previous
// order-dependent version destroyed lights without a word.
void attachLight(const fastgltf::Asset& asset, const fastgltf::Node& gltfNode, Node& target)
{
    if (!gltfNode.lightIndex.has_value())
    {
        return;
    }
    requireEmptyComponent(target, "light");
    const auto& gl = asset.lights[gltfNode.lightIndex.value()];
    auto& light = target.component().emplace<Light>();
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

} // namespace

void GltfLoader::applyControllable(std::size_t nodeIndex,
                                   const std::unordered_set<std::size_t>& controllableNodeIndices,
                                   Node& node)
{
    if (controllableNodeIndices.contains(nodeIndex))
    {
        node.emplaceControllable();
    }
}

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

void GltfLoader::GltfSceneBuilder::validatePhysicsTarget(std::size_t nodeIndex,
                                                         const fastgltf::Node& gltfNode) const
{
    const auto config = context_.physicsNodeConfigs.find(nodeIndex);
    if (config == context_.physicsNodeConfigs.end())
    {
        return;
    }

    if (config->second.bodyType == PhysicsBodyType::Dynamic &&
        context_.controllableNodeIndices.contains(nodeIndex))
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

void GltfLoader::GltfSceneBuilder::applyPhysicsConfig(std::size_t nodeIndex,
                                                      const fastgltf::Mesh& mesh, Node& node)
{
    auto it = context_.physicsNodeConfigs.find(nodeIndex);
    if (it == context_.physicsNodeConfigs.end())
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

    PhysicsBodyHandle bodyHandle = context_.physics.createBody(bodyDesc);
    node.physicsBodyHandle(bodyHandle);

    // Compound: one child collider per authored child (aggregate mass properties).
    if (!it->second.compoundChildren.empty())
    {
        node.physicsColliderHandle(context_.physics.createCompoundCollider(
            bodyHandle, it->second.compoundChildren, it->second.layer, it->second.mask));
        return;
    }
    // Static triangle mesh from the node geometry.
    if (it->second.staticMeshFromMesh)
    {
        node.physicsColliderHandle(context_.physics.createMeshCollider(
            bodyHandle, GltfLoader::meshTriangles(context_.asset, mesh),
            PhysicsMaterial{it->second.restitution, it->second.friction}, it->second.layer,
            it->second.mask));
        return;
    }

    ColliderDesc colliderDesc;
    colliderDesc.collisionLayer = it->second.layer;
    colliderDesc.collisionMask = it->second.mask;
    colliderDesc.isTrigger = it->second.isTrigger;
    colliderDesc.material = bodyDesc.material;
    if (const auto& shape = it->second.shape; shape.has_value())
    {
        colliderDesc.shape = *shape;
    }
    else if (it->second.convexHullFromMesh)
    {
        ConvexHullShape hull = GltfLoader::meshConvexHull(context_.asset, mesh);
        if (!hull.faces.empty())
        {
            colliderDesc.shape = std::move(hull);
        }
        else if (auto bounds = GltfLoader::meshBounds(context_.asset, mesh); bounds.has_value())
        {
            colliderDesc.shape = AabbShape{bounds.value()}; // degenerate hull -> AABB
        }
    }
    else if (auto bounds = GltfLoader::meshBounds(context_.asset, mesh); bounds.has_value())
    {
        colliderDesc.shape = AabbShape{bounds.value()};
    }
    node.physicsColliderHandle(context_.physics.createCollider(bodyHandle, colliderDesc));
}

// The camera goes on the node the LAYOUT chose — this function no longer inspects the variant to
// decide for itself. That inspection is exactly how placement drifted from the rule: each attach
// site read the current state and reached its own conclusion, so the order of the calls became the
// real policy.
Node& GltfLoader::GltfSceneBuilder::attachCamera(Node& target)
{
    requireEmptyComponent(target, "camera");
    Camera& camera = target.component().emplace<Camera>();
    camera.localPosition({0.0f, 0.0f, 0.0f});
    camera.localYaw(-pi / 2.0f);
    camera.localPitch(0.0f);

    if (context_.activeCamera == nullptr)
    {
        context_.activeCamera = &target;
    }

    return target;
}

Mesh& GltfLoader::GltfSceneBuilder::attachMeshToNode(std::size_t nodeIndex, std::size_t meshIndex,
                                                     Node& meshNode, Node& physicsNode)
{
    const auto& gltfMesh = context_.asset.meshes[meshIndex];
    // `extras.Shadow.Casts` is authored per NODE and stays per node: it rides into this
    // instance's Object binding, never onto the shared Geometry. Two nodes instancing one
    // mesh with opposite settings are both honoured.
    const auto authored = context_.shadowCastsNodes.find(nodeIndex);
    const bool castsShadow = authored == context_.shadowCastsNodes.end() ? true : authored->second;
    auto object = loadMesh(gltfMesh, meshIndex, castsShadow);
    meshNode.component().emplace<Mesh>(std::move(object));

    auto& mesh = std::get<Mesh>(meshNode.component());
    mesh.variantNames(context_.asset.materialVariants);
    applyPhysicsConfig(nodeIndex, gltfMesh, physicsNode);
    context_.meshMap[nodeIndex] = &mesh;
    return mesh;
}

void GltfLoader::GltfSceneBuilder::loadRootNode(SceneGraph& scene, std::size_t nodeIndex)
{
    const fastgltf::Asset& asset = context_.asset;
    auto rootNode = std::make_unique<Node>(nodeName(asset, asset.nodes[nodeIndex]));
    auto& rootRef = scene.addNode(std::move(rootNode));
    context_.nodeMap[nodeIndex] = &rootRef;
    applyControllable(nodeIndex, context_.controllableNodeIndices, rootRef);

    if (nodeHasAnimation(asset, nodeIndex))
    {
        configureAnimatedNode(nodeIndex, rootRef);
    }
    else
    {
        loadNode(nodeIndex, rootRef);
    }
}

void GltfLoader::GltfSceneBuilder::configureAnimatedNode(std::size_t nodeIndex, Node& node)
{
    const fastgltf::Asset& asset = context_.asset;
    Assets& assets = context_.assets;
    const auto& gltfNode = asset.nodes[nodeIndex];
    validatePhysicsTarget(nodeIndex, gltfNode);

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
            if (!channel.nodeIndex || *channel.nodeIndex != nodeIndex)
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

    // Load each glTF animation as a separate Animation object for this node
    std::vector<std::pair<std::size_t, Animation*>> nodeAnimations;
    for (std::size_t ai = 0; ai < asset.animations.size(); ++ai)
    {
        bool touchesNode = false;
        for (const auto& channel : asset.animations[ai].channels)
        {
            if (channel.nodeIndex && *channel.nodeIndex == nodeIndex)
            {
                touchesNode = true;
                break;
            }
        }
        if (!touchesNode)
        {
            continue;
        }

        auto& la = assets.animation(context_.nextAnimSlot);
        ++context_.nextAnimSlot;
        loadAnimation(ai, nodeIndex, la, numMorphTargets);
        applyRestTRS(gltfNode, la);
        la.name(std::string(asset.animations[ai].name));
        nodeAnimations.emplace_back(ai, &la);
    }

    // ONE decision, taken before anything is attached: which node holds what. Every attach below
    // consumes a target rather than inspecting the current variant, so no reordering of these calls
    // can change placement — the failure mode that silently destroyed lights.
    const NodeComponentLayout layout =
        planNodeComponents(hasTransformAnim, gltfNode.meshIndex.has_value(),
                           gltfNode.lightIndex.has_value(), gltfNode.cameraIndex.has_value());
    const std::string meshChildName =
        gltfNode.meshIndex.has_value() && !asset.meshes[gltfNode.meshIndex.value()].name.empty()
            ? std::string(asset.meshes[gltfNode.meshIndex.value()].name)
            : std::string{};
    const NodeComponentTargets targets =
        materializeNodeComponentLayout(node, layout, meshChildName);

    if (layout.primary == NodeComponentLayout::Primary::Animator)
    {
        requireEmptyComponent(node, "animator");
        node.component().emplace<Animator>();
        auto& animator = std::get<Animator>(node.component());
        for (const auto& [animId, anim] : nodeAnimations)
        {
            animator.addAnimation(animId, anim);
        }
    }

    if (targets.mesh != nullptr)
    {
        // Physics stays on the TRANSFORM owner, which is this node whether or not the mesh moved to
        // a child of it.
        Mesh& mesh = attachMeshToNode(nodeIndex, gltfNode.meshIndex.value(), *targets.mesh, node);
        if (hasWeightAnim)
        {
            for (const auto& [animId, anim] : nodeAnimations)
            {
                mesh.addMorphAnimation(animId, anim);
            }
        }
        // A transform-animated node starts its morph weights at zero (the animation drives them);
        // a weight-only-animated node keeps the glTF's authored weights.
        mesh.initialMorphWeights(
            hasTransformAnim
                ? std::vector<float>(numMorphTargets, 0.0f)
                : initialMorphWeights(asset.meshes[gltfNode.meshIndex.value()], numMorphTargets));
    }

    if (targets.light != nullptr)
    {
        attachLight(asset, gltfNode, *targets.light);
    }

    if (targets.camera != nullptr)
    {
        attachCamera(*targets.camera);
    }

    for (auto childIndex : gltfNode.children)
    {
        auto childNode = std::make_unique<Node>(nodeName(asset, asset.nodes[childIndex]));
        auto& childRef = node.addChild(std::move(childNode));
        context_.nodeMap[childIndex] = &childRef;
        applyControllable(childIndex, context_.controllableNodeIndices, childRef);

        if (nodeHasAnimation(asset, childIndex))
        {
            configureAnimatedNode(childIndex, childRef);
        }
        else
        {
            loadNode(childIndex, childRef);
        }
    }
}

void GltfLoader::GltfSceneBuilder::loadNode(std::size_t nodeIndex, Node& node)
{
    const fastgltf::Asset& asset = context_.asset;
    const auto& gltfNode = asset.nodes[nodeIndex];
    validatePhysicsTarget(nodeIndex, gltfNode);
    applyTRS(gltfNode, node);

    // Same one decision as the animated path, with no transform animation in play.
    const NodeComponentLayout layout =
        planNodeComponents(false, gltfNode.meshIndex.has_value(), gltfNode.lightIndex.has_value(),
                           gltfNode.cameraIndex.has_value());
    const std::string meshChildName =
        gltfNode.meshIndex.has_value() && !asset.meshes[gltfNode.meshIndex.value()].name.empty()
            ? std::string(asset.meshes[gltfNode.meshIndex.value()].name)
            : std::string{};
    const NodeComponentTargets targets =
        materializeNodeComponentLayout(node, layout, meshChildName);

    if (targets.mesh != nullptr)
    {
        const auto& gltfMesh = asset.meshes[gltfNode.meshIndex.value()];
        Mesh& mesh = attachMeshToNode(nodeIndex, gltfNode.meshIndex.value(), *targets.mesh, node);

        // Static meshes with morph targets still honour mesh.weights (e.g.
        // MorphPrimitivesTest). Without this, weights stay at zero and the
        // base geometry renders unmorphed.
        std::size_t numMorphTargets = countMorphTargets(gltfMesh);
        if (numMorphTargets > 0)
        {
            mesh.initialMorphWeights(initialMorphWeights(gltfMesh, numMorphTargets));
        }
    }

    if (targets.light != nullptr)
    {
        attachLight(asset, gltfNode, *targets.light);
    }

    if (targets.camera != nullptr)
    {
        attachCamera(*targets.camera);
    }

    for (auto childIndex : gltfNode.children)
    {
        auto childNode = std::make_unique<Node>(nodeName(asset, asset.nodes[childIndex]));
        auto& childRef = node.addChild(std::move(childNode));
        context_.nodeMap[childIndex] = &childRef;
        applyControllable(childIndex, context_.controllableNodeIndices, childRef);

        if (nodeHasAnimation(asset, childIndex))
        {
            configureAnimatedNode(childIndex, childRef);
        }
        else
        {
            loadNode(childIndex, childRef);
        }
    }
}

} // namespace fire_engine
