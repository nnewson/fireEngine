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
