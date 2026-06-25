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
std::vector<float> packMorphTargetDeltas(const Geometry& geometry)
{
    const auto numTargets = geometry.morphTargetCount();
    const auto numVerts = geometry.vertices().size();
    const std::size_t totalEntries = numTargets * numVerts * 3;
    std::vector<float> ssboData(totalEntries * 4, 0.0f);
    float* dst = ssboData.data();

    auto writeDeltas = [&](std::span<const std::vector<Vec3>> src)
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
Vec3 skinnedPosition(const Vertex& vertex, Vec3 position, std::span<const Mat4> joints) noexcept
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

    createForwardBindings(resources);
    createShadowBindings(resources);
}

void Object::createForwardBindings(Resources& resources)
{
    // Shared per-object UBO (model/view/proj), pushed as forward set-0 binding 0
    // per draw via VK_KHR_push_descriptor — no per-object descriptor set.
    auto uniformSet = resources.createMappedUniformBuffers(sizeof(UniformBufferObject));
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        uniformMapped_[i] = uniformSet.mapped[i];
        uniformBufs_[i] = uniformSet.buffers[i];
    }

    for (auto& binding : bindings_)
    {
        // Material data (textures + scalars) is bindless now (global set 2);
        // registered lazily in buildDrawCommands. No per-object material UBO.

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
            binding.skinBufs[i] = skinSet.buffers[i];
            std::memcpy(skinSet.mapped[i], &skinUbo, sizeof(skinUbo));
        }

        // Morph UBO buffers
        MorphUBO morphUbo{};
        auto morphUboSet = resources.createMappedUniformBuffers(sizeof(MorphUBO));
        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            binding.morphUboMapped[i] = morphUboSet.mapped[i];
            binding.morphUboBufs[i] = morphUboSet.buffers[i];
            std::memcpy(morphUboSet.mapped[i], &morphUbo, sizeof(morphUbo));
        }

        // Morph SSBO (exactly sized; the push descriptor binds it WholeSize). A
        // minimal dummy keeps the binding valid for geometry without morph
        // targets.
        auto numTargets = binding.geometry->morphTargetCount();
        auto numVerts = binding.geometry->vertices().size();
        if (numTargets == 0 || numVerts == 0)
        {
            float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            auto ssboSet = resources.createMappedStorageBuffer(sizeof(zeros), zeros);
            binding.morphSsbo = ssboSet.buffers[0];
        }
        else
        {
            std::vector<float> ssboData = packMorphTargetDeltas(*binding.geometry);
            const std::size_t ssboSize = ssboData.size() * sizeof(float);
            auto ssboSet = resources.createMappedStorageBuffer(ssboSize, ssboData.data());
            binding.morphSsbo = ssboSet.buffers[0];
        }
    }
}

void Object::createShadowBindings(Resources& resources)
{
    // Per-object ShadowUBO (model + per-cascade lightViewProj[4] + hasSkin),
    // pushed as shadow set-0 binding 0 per draw. The skin / morph / morphSsbo
    // buffers allocated by createForwardBindings are reused for the shadow draw —
    // no duplicate uploads — and the shared self-shadow image+sampler (bindings
    // 4/5) are pushed from Resources by the shadow pass.
    for (auto& binding : bindings_)
    {
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
            binding.shadowBufs[i] = shadowSet.buffers[i];
            std::memcpy(shadowSet.mapped[i], &initialShadow, sizeof(initialShadow));
        }
    }
}

void Object::activeVariant(std::optional<std::size_t> variantIndex)
{
    for (auto& binding : bindings_)
    {
        const Material* nextMaterial = binding.defaultMaterial;
        if (variantIndex && *variantIndex < binding.variantMaterials.size() &&
            binding.variantMaterials[*variantIndex] != nullptr)
        {
            nextMaterial = binding.variantMaterials[*variantIndex];
        }

        if (nextMaterial == binding.activeMaterial)
        {
            continue;
        }

        binding.activeMaterial = nextMaterial;
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
        if (variantIndex && *variantIndex < binding.variantMaterials.size() &&
            binding.variantMaterials[*variantIndex] != nullptr)
        {
            candidate = binding.variantMaterials[*variantIndex];
        }

        if (!materialsEquivalent(*candidate, *binding.activeMaterial))
        {
            return true;
        }
    }

    return false;
}

void Object::updateSkin()
{
    if (skin_ != nullptr && !skin_->empty())
    {
        skin_->updateJointMatrices();
    }
}

const Bounds3& Object::localBounds() const noexcept
{
    if (!localBounds_.has_value())
    {
        Bounds3 bounds;
        for (const auto& binding : bindings_)
        {
            const Geometry* geometry =
                binding.geometry != nullptr ? binding.geometry : binding.shadowGeometry;
            if (geometry == nullptr)
            {
                continue;
            }
            for (const auto& vertex : geometry->vertices())
            {
                bounds.expand(vertex.position());
            }
        }
        localBounds_ = bounds;
    }
    return localBounds_.value();
}

Bounds3 Object::computeShadowBounds(std::span<const Mat4> jointMatrices, bool hasSkin,
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
                                  std::span<const Mat4> jointMatrices)
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
        // Material data is bindless (global set 2 materials[] SSBO); a draw selects
        // its material by index (buildDrawCommands → registerMaterial), so a variant
        // switch is just a different index — nothing to write per object here.
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
        // Opaque and double-sided materials share the opaque forward pipeline;
        // the cull-mode difference is carried on the draw (doubleSided) and set
        // via dynamic state. BLEND materials use the dedicated blend pipeline.
        const bool isBlend = mat.alphaMode() == AlphaMode::Blend;
        const PipelineHandle pipe = isBlend ? frame.pipelines.blend : frame.pipelines.opaque;

        Vec3 centroid{world[0, 3], world[1, 3], world[2, 3]};
        float depth = Vec3::dotProduct(forwardVec, centroid - frame.cameraPosition);

        DrawCommand cmd;
        cmd.vertexBuffer = binding.geometry->vertexBuffer();
        cmd.indexBuffer = binding.geometry->indexBuffer();
        cmd.indexCount = binding.geometry->indexCount();
        cmd.indexType = binding.geometry->indexType();
        // Forward set 0 is pushed inline at draw time, not bound — carry the
        // buffer handles instead of a descriptor set.
        cmd.frameUbo = uniformBufs_[frame.currentFrame];
        cmd.skinUbo = binding.skinBufs[frame.currentFrame];
        cmd.morphUbo = binding.morphUboBufs[frame.currentFrame];
        cmd.morphSsbo = binding.morphSsbo;
        cmd.pipeline = pipe;
        cmd.doubleSided = mat.doubleSided();
        cmd.sortDepth = depth;
        cmd.objectId = objectId_;
        cmd.hasSkin = hasSkin;
        cmd.shadowBounds = shadowBounds;
        // Bindless material index (idempotent registration — first sight assigns a
        // slot in the global materials[] SSBO; cached thereafter).
        cmd.materialIndex = resources_ != nullptr ? resources_->registerMaterial(mat) : 0;
        // KHR_materials_transmission F3: defer this draw to the second forward
        // sub-pass so its fragment shader can sample the post-opaque HDR
        // target via screen-space refraction.
        const bool hasTransmissionFactor = mat.transmission() && mat.transmission()->factor > 0.0f;
        cmd.transmissive =
            hasTransmissionFactor || mat.texture(MaterialTextureSlot::Transmission).has();
        commands.push_back(cmd);

        if (binding.geometry->castsShadow() && frame.shadowPipeline != NullPipeline &&
            binding.shadowBufs[frame.currentFrame] != NullBuffer)
        {
            DrawCommand shadowCmd = cmd;
            const Geometry* shadowGeometry =
                binding.shadowGeometry != nullptr ? binding.shadowGeometry : binding.geometry;
            shadowCmd.vertexBuffer = shadowGeometry->vertexBuffer();
            shadowCmd.indexBuffer = shadowGeometry->indexBuffer();
            shadowCmd.indexCount = shadowGeometry->indexCount();
            shadowCmd.indexType = shadowGeometry->indexType();
            shadowCmd.pipeline = frame.shadowPipeline;
            // Shadow set 0 is pushed inline per draw: the ShadowUBO here plus the
            // skin/morph/morphSsbo handles copied from the forward cmd above.
            shadowCmd.shadowUbo = binding.shadowBufs[frame.currentFrame];
            shadowCmd.sortDepth = 0.0f;
            commands.push_back(shadowCmd);
        }
    }
    return commands;
}

} // namespace fire_engine
