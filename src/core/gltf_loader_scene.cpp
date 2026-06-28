#include <fire_engine/core/gltf_loader.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/ragdoll.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{
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

    const std::size_t sceneIndex = asset.defaultScene.value_or(0);
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

} // namespace fire_engine
