#include <fire_engine/core/gltf_loader.hpp>

#include <filesystem>
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

#include <fire_engine/core/log.hpp>
#include <fire_engine/graphics/assets.hpp>
#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/ragdoll.hpp>
#include <fire_engine/scene/scene_graph.hpp>

namespace fire_engine
{
GltfLoader::GltfSceneBuilder::GltfSceneBuilder(GltfLoadContext context)
    : context_{std::move(context)}
{
}

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

    GltfLoadContext context(asset, gltfPath.parent_path().string(), resources, assets, physics,
                            std::move(controllableNodeIndices), std::move(physicsNodeConfigs),
                            std::move(clothNodeConfigs), std::move(ragdollNodeConfigs));
    GltfSceneBuilder builder{std::move(context)};
    return builder.build(scene, clothRegistrations, ragdolls);
}

Node* GltfLoader::GltfSceneBuilder::build(SceneGraph& scene,
                                          std::vector<ClothRegistration>* clothRegistrations,
                                          std::vector<Ragdoll>* ragdolls)
{
    const fastgltf::Asset& asset = context_.asset;
    Assets& assets = context_.assets;

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
    for (const auto& [nodeIndex, params] : context_.clothNodeConfigs)
    {
        const auto& gltfNode = asset.nodes[nodeIndex];
        if (!gltfNode.meshIndex.has_value())
        {
            log::warn(log::category::gltf, "node '{}' has Cloth extras but no mesh; ignoring.",
                      nodeName(asset, gltfNode));
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
    for (auto nodeIndex : gltfScene.nodeIndices)
    {
        loadRootNode(scene, nodeIndex);
    }

    // Resolve skins after the full scene graph is built
    applySkins();

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
    if (!context_.ragdollNodeConfigs.empty())
    {
        for (const auto& [nodeIndex, params] : context_.ragdollNodeConfigs)
        {
            const auto& gltfNode = asset.nodes[nodeIndex];
            if (!gltfNode.skinIndex.has_value())
            {
                log::warn(log::category::gltf,
                          "node '{}' has Ragdoll extras but no skin; ignoring.",
                          nodeName(asset, gltfNode));
                continue;
            }
            const auto& gltfSkin = asset.skins[gltfNode.skinIndex.value()];
            std::vector<Node*> bones;
            bones.reserve(gltfSkin.joints.size());
            for (const auto jointNodeIndex : gltfSkin.joints)
            {
                const auto it = context_.nodeMap.find(jointNodeIndex);
                if (it != context_.nodeMap.end())
                {
                    // Seed the ragdoll from the BIND pose: an *animated* skeleton's joint nodes
                    // aren't at their bind transform at load time (the animation hasn't been
                    // evaluated — they sit at identity), so composing them would collapse every
                    // bone onto the armature origin and build a degenerate ragdoll. Reset each
                    // joint to its glTF (bind) TRS first.
                    applyTRS(asset.nodes[jointNodeIndex], *it->second);
                    bones.push_back(it->second);
                }
            }
            if (bones.empty())
            {
                continue;
            }
            scene.resolve(); // compose the just-restored bind pose before seeding the ragdoll
            Ragdoll ragdoll = params.articulated
                                  ? Ragdoll::makeArticulated(context_.physics, bones, params)
                                  : Ragdoll::make(context_.physics, bones, params);
            ragdoll.activate();
            log::info(log::category::gltf, "built {} ragdoll for node '{}' with {} bones",
                      params.articulated ? "articulated" : "maximal", nodeName(asset, gltfNode),
                      bones.size());
            ragdolls->push_back(std::move(ragdoll));
        }
    }

    return context_.activeCamera;
}

} // namespace fire_engine
