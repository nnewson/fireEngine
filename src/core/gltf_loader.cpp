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

    // Compound: one child collider per authored child (aggregate mass properties).
    if (!it->second.compoundChildren.empty())
    {
        node.physicsColliderHandle(physics.createCompoundCollider(
            bodyHandle, it->second.compoundChildren, it->second.layer, it->second.mask));
        return;
    }
    // Static triangle mesh from the node geometry.
    if (it->second.staticMeshFromMesh)
    {
        node.physicsColliderHandle(
            physics.createMeshCollider(bodyHandle, GltfLoader::meshTriangles(asset, mesh),
                                       PhysicsMaterial{it->second.restitution, it->second.friction},
                                       it->second.layer, it->second.mask));
        return;
    }

    ColliderDesc colliderDesc;
    colliderDesc.collisionLayer = it->second.layer;
    colliderDesc.collisionMask = it->second.mask;
    colliderDesc.material = bodyDesc.material;
    if (it->second.shape.has_value())
    {
        colliderDesc.shape = it->second.shape.value();
    }
    else if (it->second.convexHullFromMesh)
    {
        ConvexHullShape hull = GltfLoader::meshConvexHull(asset, mesh);
        if (!hull.faces.empty())
        {
            colliderDesc.shape = std::move(hull);
        }
        else if (auto bounds = GltfLoader::meshBounds(asset, mesh); bounds.has_value())
        {
            colliderDesc.shape = AabbShape{bounds.value()}; // degenerate hull → AABB
        }
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
// Scene and node loading
// ---------------------------------------------------------------------------

Node* GltfLoader::loadScene(const std::string& path, SceneGraph& scene, Resources& resources,
                            Assets& assets, PhysicsWorld& physics,
                            std::vector<ClothRegistration>* clothRegistrations,
                            std::vector<Ragdoll>* ragdolls)
{
    auto gltfPath = std::filesystem::path(path);
    std::unordered_set<std::size_t> controllableNodeIndices;
    std::unordered_map<std::size_t, PhysicsConfig> physicsNodeConfigs;
    std::unordered_map<std::size_t, ClothMeshParams> clothNodeConfigs;
    std::unordered_map<std::size_t, RagdollParams> ragdollNodeConfigs;
    auto result = parseAsset(gltfPath, &controllableNodeIndices, &physicsNodeConfigs,
                             clothRegistrations != nullptr ? &clothNodeConfigs : nullptr,
                             ragdolls != nullptr ? &ragdollNodeConfigs : nullptr);
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

    // Cloth nodes: resolve each to its mesh's first-primitive geometry index and
    // flag that geometry for a storage vertex buffer *before* the graph build loads
    // it (so the solver can write it in place). The registration is filled in after
    // the build, once the geometry's CPU vertices/indices + GPU buffer exist.
    auto firstGeometryIndex = [&asset](std::size_t meshIndex)
    {
        std::size_t geoIdx = 0;
        for (std::size_t m = 0; m < meshIndex; ++m)
        {
            geoIdx += asset.meshes[m].primitives.size();
        }
        return geoIdx;
    };
    std::vector<std::pair<std::size_t, ClothMeshParams>> clothGeometries; // (geoIdx, params)
    for (const auto& [nodeIndex, params] : clothNodeConfigs)
    {
        const auto& gltfNode = asset.nodes[nodeIndex];
        if (!gltfNode.meshIndex.has_value())
        {
            std::clog << "glTF: node '" << nodeName(asset, gltfNode)
                      << "' has Cloth extras but no mesh; ignoring.\n";
            continue;
        }
        const std::size_t geoIdx = firstGeometryIndex(gltfNode.meshIndex.value());
        assets.geometry(geoIdx).storageVertices(true);
        clothGeometries.emplace_back(geoIdx, params);
    }

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

    // Build a cloth from each flagged geometry now that it's loaded (CPU vertices +
    // indices retained, storage vertex buffer allocated). The caller registers
    // these with the soft-body solver.
    for (const auto& [geoIdx, params] : clothGeometries)
    {
        const Geometry& geometry = assets.geometry(geoIdx);
        ClothRegistration reg{makeClothFromMesh(geometry.vertices(), geometry.indices(), params),
                              &assets.geometry(geoIdx)};
        clothRegistrations->push_back(std::move(reg));
    }

    // Auto-build a ragdoll from each `extras.Ragdoll` node's skin. Resolve once so
    // the bones carry their bind-pose composed-world (the ragdoll seeds bodies from
    // it); the per-frame update() recomputes it afterwards.
    if (!ragdollNodeConfigs.empty())
    {
        scene.resolve();
        for (const auto& [nodeIndex, params] : ragdollNodeConfigs)
        {
            const auto& gltfNode = asset.nodes[nodeIndex];
            if (!gltfNode.skinIndex.has_value())
            {
                std::clog << "glTF: node '" << nodeName(asset, gltfNode)
                          << "' has Ragdoll extras but no skin; ignoring.\n";
                continue;
            }
            const auto& gltfSkin = asset.skins[gltfNode.skinIndex.value()];
            std::vector<Node*> bones;
            bones.reserve(gltfSkin.joints.size());
            for (const auto jointNodeIndex : gltfSkin.joints)
            {
                const auto it = nodeMap.find(jointNodeIndex);
                if (it != nodeMap.end())
                {
                    bones.push_back(it->second);
                }
            }
            if (bones.empty())
            {
                continue;
            }
            Ragdoll ragdoll = Ragdoll::make(physics, bones, params);
            ragdoll.activate();
            ragdolls->push_back(std::move(ragdoll));
        }
    }

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

} // namespace fire_engine
