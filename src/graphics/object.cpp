#include <fire_engine/graphics/object.hpp>

#include <cstring>

#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/material_binding.hpp>
#include <fire_engine/graphics/skin.hpp>
#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/vec4.hpp>
#include <fire_engine/math/view_basis.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

[[nodiscard]]
Resources::FallbackTextureKind fallbackForSlot(MaterialTextureSlot slot) noexcept
{
    using Kind = Resources::FallbackTextureKind;
    switch (slot)
    {
    case MaterialTextureSlot::BaseColour:
    case MaterialTextureSlot::Clearcoat:
    case MaterialTextureSlot::ClearcoatRoughness:
    case MaterialTextureSlot::Thickness:
        return Kind::BaseColour;
    case MaterialTextureSlot::Emissive:
        return Kind::Emissive;
    case MaterialTextureSlot::Normal:
    case MaterialTextureSlot::ClearcoatNormal:
        return Kind::Normal;
    case MaterialTextureSlot::MetallicRoughness:
        return Kind::MetallicRoughness;
    case MaterialTextureSlot::Occlusion:
    case MaterialTextureSlot::Transmission:
        return Kind::Occlusion;
    }

    return Kind::BaseColour;
}

[[nodiscard]]
std::vector<float> packMorphTargetDeltas(const Geometry& geometry)
{
    const auto numTargets = geometry.morphTargetCount();
    const auto numVerts = geometry.vertices().size();
    const std::size_t totalEntries = numTargets * numVerts * 3;
    std::vector<float> ssboData(totalEntries * 4, 0.0f);
    float* dst = ssboData.data();

    auto writeDeltas = [&](const std::vector<std::vector<Vec3>>& src)
    {
        for (std::size_t t = 0; t < numTargets; ++t)
        {
            for (std::size_t v = 0; v < numVerts; ++v)
            {
                const Vec3& delta = (t < src.size() && v < src[t].size()) ? src[t][v] : Vec3{};
                *dst++ = delta.x();
                *dst++ = delta.y();
                *dst++ = delta.z();
                *dst++ = 0.0f;
            }
        }
    };

    writeDeltas(geometry.morphPositions());
    writeDeltas(geometry.morphNormals());
    writeDeltas(geometry.morphTangents());
    return ssboData;
}

[[nodiscard]]
Vec3 skinnedPosition(const Vertex& vertex, Vec3 position, const std::vector<Mat4>& joints) noexcept
{
    const Joints4 jointIds = vertex.joints();
    const Vec4 weights = vertex.weights();

    Vec3 result{};
    float totalWeight = 0.0f;
    auto addJoint = [&](uint32_t jointId, float weight)
    {
        if (weight <= 0.0f || jointId >= joints.size())
        {
            return;
        }
        result += static_cast<Vec3>(joints[jointId] * Vec4{position}) * weight;
        totalWeight += weight;
    };

    addJoint(jointIds.j0(), weights.x());
    addJoint(jointIds.j1(), weights.y());
    addJoint(jointIds.j2(), weights.z());
    addJoint(jointIds.j3(), weights.w());
    if (totalWeight <= 0.0f)
    {
        return position;
    }
    return result;
}

} // namespace

void Object::addGeometry(const Geometry& geometry)
{
    auto& binding = bindings_.emplace_back();
    binding.geometry = &geometry;
    binding.shadowGeometry = &geometry;
    binding.defaultMaterial = &geometry.material();
    binding.activeMaterial = &geometry.material();
}

void Object::shadowGeometry(std::size_t geometryIndex, const Geometry* geometry) noexcept
{
    if (geometryIndex >= bindings_.size())
    {
        return;
    }
    bindings_[geometryIndex].shadowGeometry =
        geometry != nullptr ? geometry : bindings_[geometryIndex].geometry;
}

void Object::addVariantMaterial(std::size_t geometryIndex, std::size_t variantIndex,
                                const Material* material)
{
    if (geometryIndex >= bindings_.size() || material == nullptr)
    {
        return;
    }

    auto& binding = bindings_[geometryIndex];
    if (binding.variantMaterials.size() <= variantIndex)
    {
        binding.variantMaterials.resize(variantIndex + 1, nullptr);
    }
    binding.variantMaterials[variantIndex] = material;
}

void Object::load(Resources& resources)
{
    resources_ = &resources;
    if (objectId_ == 0)
    {
        objectId_ = resources.allocateObjectId();
    }

    const ObjectDescriptorRequest req = createForwardBindings(resources);
    createShadowBindings(resources, req);
}

ObjectDescriptorRequest Object::createForwardBindings(Resources& resources)
{
    // Create shared uniform buffers (model/view/proj)
    auto uniformSet = resources.createMappedUniformBuffers(sizeof(UniformBufferObject));
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        uniformMapped_[i] = uniformSet.mapped[i];
    }

    Resources::ObjectDescriptorRequest req;
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        req.uniformBufs[i] = uniformSet.buffers[i];
    }

    for (auto& binding : bindings_)
    {
        GeometryDescriptorInfo geoInfo;

        // Material buffers
        MaterialUBO matUbo = toMaterialUBO(*binding.activeMaterial);
        auto matSet = resources.createMappedUniformBuffers(sizeof(MaterialUBO));
        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            binding.materialMapped[i] = matSet.mapped[i];
            geoInfo.materialBufs[i] = matSet.buffers[i];
            std::memcpy(matSet.mapped[i], &matUbo, sizeof(matUbo));
        }

        // Skin buffers
        SkinUBO skinUbo{};
        for (std::size_t j = 0; j < kMaxJoints; ++j)
        {
            skinUbo.joints[j] = Mat4::identity();
        }
        auto skinSet = resources.createMappedUniformBuffers(sizeof(SkinUBO));
        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            binding.skinMapped[i] = skinSet.mapped[i];
            geoInfo.skinBufs[i] = skinSet.buffers[i];
            std::memcpy(skinSet.mapped[i], &skinUbo, sizeof(skinUbo));
        }

        // Morph UBO buffers
        MorphUBO morphUbo{};
        auto morphUboSet = resources.createMappedUniformBuffers(sizeof(MorphUBO));
        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            binding.morphUboMapped[i] = morphUboSet.mapped[i];
            geoInfo.morphUboBufs[i] = morphUboSet.buffers[i];
            std::memcpy(morphUboSet.mapped[i], &morphUbo, sizeof(morphUbo));
        }

        // Morph SSBO
        auto numTargets = binding.geometry->morphTargetCount();
        auto numVerts = binding.geometry->vertices().size();

        if (numTargets == 0 || numVerts == 0)
        {
            // Minimal dummy SSBO
            float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            auto ssboSet = resources.createMappedStorageBuffer(sizeof(zeros), zeros);
            geoInfo.morphSsbo = ssboSet.buffers[0];
            geoInfo.morphSsboSize = 0;
        }
        else
        {
            std::vector<float> ssboData = packMorphTargetDeltas(*binding.geometry);
            const std::size_t ssboSize = ssboData.size() * sizeof(float);
            auto ssboSet = resources.createMappedStorageBuffer(ssboSize, ssboData.data());
            geoInfo.morphSsbo = ssboSet.buffers[0];
            geoInfo.morphSsboSize = ssboSize;
        }

        // Texture handle — use a 1x1 white dummy when material has no texture
        applyMaterialTextures(geoInfo, *binding.activeMaterial, resources);

        req.geometries.push_back(geoInfo);
    }

    auto descResult = resources.descriptors().createObjectDescriptors(req);
    for (std::size_t g = 0; g < bindings_.size(); ++g)
    {
        bindings_[g].descSets = descResult.descSets[g];
    }

    return req;
}

void Object::createShadowBindings(Resources& resources, const ObjectDescriptorRequest& req)
{
    // Shadow descriptor sets reuse the forward skin / morph / morphSsbo
    // buffers allocated above — no duplicate uploads — plus a new per-geometry
    // ShadowUBO buffer carrying model + per-cascade lightViewProj[4] + hasSkin.
    Resources::ShadowDescriptorRequest shadowReq;
    shadowReq.geometries.reserve(bindings_.size());
    for (std::size_t g = 0; g < bindings_.size(); ++g)
    {
        auto& binding = bindings_[g];
        Resources::ShadowGeometryDescriptorInfo shadowInfo;

        ShadowUBO initialShadow{};
        initialShadow.model = Mat4::identity();
        for (Mat4& m : initialShadow.lightViewProj)
        {
            m = Mat4::identity();
        }
        auto shadowSet = resources.createMappedUniformBuffers(sizeof(ShadowUBO));
        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            binding.shadowMapped[i] = shadowSet.mapped[i];
            shadowInfo.shadowUboBufs[i] = shadowSet.buffers[i];
            std::memcpy(shadowSet.mapped[i], &initialShadow, sizeof(initialShadow));
            shadowInfo.skinBufs[i] = req.geometries[g].skinBufs[i];
            shadowInfo.morphUboBufs[i] = req.geometries[g].morphUboBufs[i];
        }
        shadowInfo.morphSsbo = req.geometries[g].morphSsbo;
        shadowInfo.morphSsboSize = req.geometries[g].morphSsboSize;

        shadowReq.geometries.push_back(shadowInfo);
    }

    auto shadowDescResult = resources.descriptors().createShadowDescriptors(shadowReq);
    for (std::size_t g = 0; g < bindings_.size(); ++g)
    {
        bindings_[g].shadowDescSets = shadowDescResult.descSets[g];
    }
}

void Object::activeVariant(std::optional<std::size_t> variantIndex)
{
    for (auto& binding : bindings_)
    {
        const Material* nextMaterial = binding.defaultMaterial;
        if (variantIndex.has_value() && variantIndex.value() < binding.variantMaterials.size() &&
            binding.variantMaterials[variantIndex.value()] != nullptr)
        {
            nextMaterial = binding.variantMaterials[variantIndex.value()];
        }

        if (nextMaterial == binding.activeMaterial)
        {
            continue;
        }

        binding.activeMaterial = nextMaterial;
        binding.descriptorDirty.fill(true);
    }
}

bool Object::hasVariant(std::size_t variantIndex) const noexcept
{
    for (const auto& binding : bindings_)
    {
        if (variantIndex < binding.variantMaterials.size() &&
            binding.variantMaterials[variantIndex] != nullptr)
        {
            return true;
        }
    }
    return false;
}

bool Object::wouldChangeVariant(std::optional<std::size_t> variantIndex) const noexcept
{
    for (const auto& binding : bindings_)
    {
        const Material* candidate = binding.defaultMaterial;
        if (variantIndex.has_value() && variantIndex.value() < binding.variantMaterials.size() &&
            binding.variantMaterials[variantIndex.value()] != nullptr)
        {
            candidate = binding.variantMaterials[variantIndex.value()];
        }

        if (!materialsEquivalent(*candidate, *binding.activeMaterial))
        {
            return true;
        }
    }

    return false;
}

void Object::applyMaterialTextures(GeometryDescriptorInfo& geoInfo, const Material& mat,
                                   Resources& resources)
{
    geoInfo.materialTextures = materialTextureHandles(mat);
    for (const MaterialTextureBinding& binding : materialTextureBindings)
    {
        TextureHandle& handle = geoInfo.materialTextures[slotIndex(binding.slot)];
        if (handle == NullTexture)
        {
            handle = resources.fallbackTexture(fallbackForSlot(binding.slot));
        }
    }
}

void Object::updateSkin()
{
    if (skin_ != nullptr && !skin_->empty())
    {
        skin_->updateJointMatrices();
    }
}

Bounds3 Object::computeShadowBounds(const std::vector<Mat4>& jointMatrices, bool hasSkin,
                                    const Mat4& world) const noexcept
{
    Bounds3 bounds;
    for (const auto& binding : bindings_)
    {
        const Geometry* geometry =
            binding.shadowGeometry != nullptr ? binding.shadowGeometry : binding.geometry;
        if (geometry == nullptr)
        {
            continue;
        }

        const auto& vertices = geometry->vertices();
        const auto& morphPositions = geometry->morphPositions();
        for (std::size_t v = 0; v < vertices.size(); ++v)
        {
            Vec3 position = vertices[v].position();
            for (std::size_t target = 0;
                 target < morphPositions.size() && target < morphWeights_.size(); ++target)
            {
                if (v < morphPositions[target].size())
                {
                    position += morphPositions[target][v] * morphWeights_[target];
                }
            }

            Vec3 worldPosition = hasSkin ? skinnedPosition(vertices[v], position, jointMatrices)
                                         : static_cast<Vec3>(world * Vec4{position});
            bounds.expand(worldPosition);
        }
    }
    return bounds;
}

std::vector<DrawCommand> Object::render(const FrameInfo& frame, const Mat4& world,
                                        const Mat4& previousWorld)
{
    const bool hasSkin = skin_ != nullptr && !skin_->empty();

    std::vector<Mat4> emptyJointMatrices;
    const std::vector<Mat4>& jointMatrices =
        hasSkin ? skin_->cachedJointMatrices() : emptyJointMatrices;

    writeForwardUniforms(frame, world, previousWorld, hasSkin, jointMatrices);
    writeShadowUniforms(frame, world, hasSkin);

    const Bounds3 shadowBounds = computeShadowBounds(jointMatrices, hasSkin, world);
    return buildDrawCommands(frame, world, hasSkin, shadowBounds);
}

void Object::writeForwardUniforms(const FrameInfo& frame, const Mat4& world,
                                  const Mat4& previousWorld, bool hasSkin,
                                  const std::vector<Mat4>& jointMatrices)
{
    // Shared per-object UBO. view/proj are computed once per frame and carried
    // on FrameInfo (see RenderContext::frameInfo), not recomputed here.
    UniformBufferObject ubo{};
    ubo.model = world;
    ubo.view = frame.view;
    ubo.proj = frame.proj;
    ubo.cameraPos[0] = frame.cameraPosition.x();
    ubo.cameraPos[1] = frame.cameraPosition.y();
    ubo.cameraPos[2] = frame.cameraPosition.z();
    ubo.cameraPos[3] = 0.0f;
    ubo.hasSkin = hasSkin ? 1 : 0;
    // Motion-vector inputs (TAA): previous model + jitter-free view-projections.
    ubo.previousModel = previousWorld;
    ubo.currentViewProj = frame.currentViewProj;
    ubo.previousViewProj = frame.previousViewProj;
    std::memcpy(uniformMapped_[frame.currentFrame], &ubo, sizeof(ubo));

    if (hasSkin)
    {
        SkinUBO skinUbo{};
        for (std::size_t j = 0; j < jointMatrices.size() && j < kMaxJoints; ++j)
        {
            skinUbo.joints[j] = jointMatrices[j];
        }
        for (auto& binding : bindings_)
        {
            std::memcpy(binding.skinMapped[frame.currentFrame], &skinUbo, sizeof(skinUbo));
        }
    }

    for (auto& binding : bindings_)
    {
        MaterialUBO matUbo = toMaterialUBO(*binding.activeMaterial);
        std::memcpy(binding.materialMapped[frame.currentFrame], &matUbo, sizeof(matUbo));

        if (binding.descriptorDirty[frame.currentFrame] && resources_ != nullptr)
        {
            GeometryDescriptorInfo geoInfo;
            applyMaterialTextures(geoInfo, *binding.activeMaterial, *resources_);
            resources_->descriptors().updateObjectGeometryTextures(
                binding.descSets[frame.currentFrame], geoInfo);
            binding.descriptorDirty[frame.currentFrame] = false;
        }

        auto numTargets = binding.geometry->morphTargetCount();
        MorphUBO morphUbo{};
        if (numTargets > 0 && !morphWeights_.empty())
        {
            morphUbo.hasMorph = 1;
            morphUbo.morphTargetCount = static_cast<int>(numTargets);
            morphUbo.vertexCount = static_cast<int>(binding.geometry->vertices().size());
            for (std::size_t w = 0;
                 w < morphWeights_.size() && w < static_cast<std::size_t>(kMaxMorphTargets); ++w)
            {
                morphUbo.weights[w] = morphWeights_[w];
            }
        }
        std::memcpy(binding.morphUboMapped[frame.currentFrame], &morphUbo, sizeof(morphUbo));
    }
}

void Object::writeShadowUniforms(const FrameInfo& frame, const Mat4& world, bool hasSkin)
{
    // Shadow UBO (model + per-cascade lightViewProj[4] + hasSkin). The renderer
    // buckets by pipeline so shadow draws replay inside the shadow pass and
    // forward draws inside the forward pass.
    ShadowUBO shadowData{};
    shadowData.model = world;
    for (std::size_t i = 0; i < frame.shadowViewProjs.size(); ++i)
    {
        shadowData.lightViewProj[i] = frame.shadowViewProjs[i];
    }
    shadowData.hasSkin = hasSkin ? 1 : 0;
    for (auto& binding : bindings_)
    {
        std::memcpy(binding.shadowMapped[frame.currentFrame], &shadowData, sizeof(shadowData));
    }
}

std::vector<DrawCommand> Object::buildDrawCommands(const FrameInfo& frame, const Mat4& world,
                                                   bool hasSkin, const Bounds3& shadowBounds) const
{
    // Camera forward used to project draw centroids for back-to-front sort of
    // blend draws. Each mesh instance is taken as its world-translation origin
    // — fine for the scenes the engine renders today (flat decals etc.); a
    // future AABB-based centroid would be the natural upgrade.
    const Vec3 forwardVec = makeViewBasis(frame.cameraPosition, frame.cameraTarget).forward;

    std::vector<DrawCommand> commands;
    commands.reserve(bindings_.size() * 2);
    for (const auto& binding : bindings_)
    {
        const Material& mat = *binding.activeMaterial;
        PipelineHandle pipe{};
        if (mat.alphaMode() == AlphaMode::Blend)
        {
            pipe = frame.pipelines.blend;
        }
        else if (mat.doubleSided())
        {
            pipe = frame.pipelines.opaqueDoubleSided;
        }
        else
        {
            pipe = frame.pipelines.opaque;
        }

        Vec3 centroid{world[0, 3], world[1, 3], world[2, 3]};
        float depth = Vec3::dotProduct(forwardVec, centroid - frame.cameraPosition);

        DrawCommand cmd;
        cmd.vertexBuffer = binding.geometry->vertexBuffer();
        cmd.indexBuffer = binding.geometry->indexBuffer();
        cmd.indexCount = binding.geometry->indexCount();
        cmd.indexType = binding.geometry->indexType();
        cmd.descriptorSet = binding.descSets[frame.currentFrame];
        cmd.pipeline = pipe;
        cmd.sortDepth = depth;
        cmd.objectId = objectId_;
        cmd.hasSkin = hasSkin;
        cmd.shadowBounds = shadowBounds;
        // KHR_materials_transmission F3: defer this draw to the second forward
        // sub-pass so its fragment shader can sample the post-opaque HDR
        // target via screen-space refraction.
        const bool hasTransmissionFactor = mat.transmission() && mat.transmission()->factor > 0.0f;
        cmd.transmissive =
            hasTransmissionFactor || mat.texture(MaterialTextureSlot::Transmission).has();
        commands.push_back(cmd);

        if (binding.geometry->castsShadow() && frame.shadowPipeline != NullPipeline &&
            binding.shadowDescSets[frame.currentFrame] != NullDescriptorSet)
        {
            DrawCommand shadowCmd = cmd;
            const Geometry* shadowGeometry =
                binding.shadowGeometry != nullptr ? binding.shadowGeometry : binding.geometry;
            shadowCmd.vertexBuffer = shadowGeometry->vertexBuffer();
            shadowCmd.indexBuffer = shadowGeometry->indexBuffer();
            shadowCmd.indexCount = shadowGeometry->indexCount();
            shadowCmd.indexType = shadowGeometry->indexType();
            shadowCmd.pipeline = frame.shadowPipeline;
            shadowCmd.descriptorSet = binding.shadowDescSets[frame.currentFrame];
            shadowCmd.sortDepth = 0.0f;
            commands.push_back(shadowCmd);
        }
    }
    return commands;
}

} // namespace fire_engine
