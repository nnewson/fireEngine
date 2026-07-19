#include <fire_engine/graphics/mapped_buffer.hpp>
#include <fire_engine/graphics/object.hpp>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <fire_engine/graphics/geometry.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/material_binding.hpp>
#include <fire_engine/graphics/skin.hpp>
#include <fire_engine/graphics/vdpm_gpu_registry.hpp>
#include <fire_engine/graphics/vdpm_material.hpp>
#include <fire_engine/graphics/vipm.hpp>
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

void Object::load(Resources& resources, VdpmGpuRegistry* registry)
{
    resources_ = &resources;
    if (objectId_ == 0)
    {
        objectId_ = resources.allocateObjectId();
    }

    createForwardBindings(resources, registry);
    createShadowBindings(resources);
}

void Object::createForwardBindings(Resources& resources, VdpmGpuRegistry* registry)
{
    // Shared per-object UBO (model/view/proj), pushed as forward set-0 binding 0
    // per draw via VK_KHR_push_descriptor — no per-object descriptor set.
    auto uniformSet = resources.createMappedUniformBuffers(sizeof(ObjectUBO));
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
        for (auto& joint : skinUbo.joints)
        {
            joint = Mat4::identity();
        }
        auto skinSet = resources.createMappedUniformBuffers(sizeof(SkinUBO));
        auto prevSkinSet = resources.createMappedUniformBuffers(sizeof(SkinUBO));
        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            binding.skinMapped[i] = skinSet.mapped[i];
            binding.skinBufs[i] = skinSet.buffers[i];
            writeMapped(skinSet.mapped[i], skinUbo);
            // Previous-frame joints — identity until the first skinned frame writes real ones.
            binding.prevSkinMapped[i] = prevSkinSet.mapped[i];
            binding.prevSkinBufs[i] = prevSkinSet.buffers[i];
            writeMapped(prevSkinSet.mapped[i], skinUbo);
        }

        // Morph UBO buffers
        MorphUBO morphUbo{};
        auto morphUboSet = resources.createMappedUniformBuffers(sizeof(MorphUBO));
        for (int i = 0; i < kMaxFramesInFlight; ++i)
        {
            binding.morphUboMapped[i] = morphUboSet.mapped[i];
            binding.morphUboBufs[i] = morphUboSet.buffers[i];
            writeMapped(morphUboSet.mapped[i], morphUbo);
        }

        // Morph SSBO (exactly sized; the push descriptor binds it WholeSize). A
        // minimal dummy keeps the binding valid for geometry without morph
        // targets.
        auto numTargets = binding.geometry->morphTargetCount();
        auto numVerts = binding.geometry->vertices().size();
        if (numTargets == 0 || numVerts == 0)
        {
            float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            binding.morphSsbo = resources.createSharedStorageBuffer(sizeof(zeros), zeros);
        }
        else
        {
            std::vector<float> ssboData = packMorphTargetDeltas(*binding.geometry);
            const std::size_t ssboSize = ssboData.size() * sizeof(float);
            binding.morphSsbo = resources.createSharedStorageBuffer(ssboSize, ssboData.data());
        }

        // VIPM geomorph buffer (Continuous LOD): the geometry's per-vertex morph data when it has
        // coarser levels; otherwise a minimal dummy keeps the push-descriptor binding valid (such
        // meshes get morphFactor 0, so the vertex shader never reads it).
        if (binding.geometry->hasVipmData())
        {
            binding.vipmBuffer = binding.geometry->morphBuffer();
        }
        else
        {
            float vipmZeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            binding.vipmBuffer = resources.createSharedStorageBuffer(sizeof(vipmZeros), vipmZeros);
        }

        // VDPM (View-dependent LOD): build the per-instance active front + its per-frame dynamic
        // index buffers (sized to the finest index buffer, the max the front can emit) for static
        // meshes carrying a collapse stream.
        if (binding.geometry->hasVdpmData())
        {
            binding.vdpmFront =
                ActiveFront::build(binding.geometry->vertices(), binding.geometry->indices(),
                                   binding.geometry->collapses());
            const std::size_t maxBytes = binding.geometry->indices().size() * sizeof(uint32_t);
            auto indexSet = resources.createMappedIndexBuffers(maxBytes);
            // One indirect command per instance per frame, CPU-written from the emit count (Stage
            // A).
            auto indirectSet =
                resources.createMappedIndirectBuffers(sizeof(DrawIndexedIndirectCommand));
            for (int i = 0; i < kMaxFramesInFlight; ++i)
            {
                binding.vdpmIndexBufs[i] = indexSet.buffers[i];
                binding.vdpmIndexMapped[i] = indexSet.mapped[i];
                binding.vdpmIndirectBufs[i] = indirectSet.buffers[i];
                binding.vdpmIndirectMapped[i] = indirectSet.mapped[i];
            }

            // GPU-driven VDPM (Stage B5b): create this instance's GPU front over the geometry's
            // registered mesh. In B5b-1 it runs alongside the CPU front above (the compute is a
            // shadow run; the CPU output is still drawn). A null registry, an unsupported device,
            // or a geometry the backend rejected leaves the geometry's mesh handle null, so
            // createFront returns NullVdpmFront and the instance stays CPU-only.
            if (registry != nullptr)
            {
                binding.vdpmGpuFront = registry->createFront(binding.geometry->vdpmMeshHandle());
            }
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
            writeMapped(shadowSet.mapped[i], initialShadow);
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
    return *localBounds_;
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

bool Object::vdpmGpuDrives(const FrameInfo& frame, const GeometryBindings& binding)
{
    if (!frame.vdpmGpuBackend || binding.vdpmGpuFront == NullVdpmFront)
    {
        return false;
    }
    // Backend on + a live front ⇒ the sink MUST be present (the Renderer sets them together). A
    // null sink here would let writeForwardUniforms skip the CPU emit while buildDrawCommands falls
    // back to the (now unwritten) CPU buffers — a construction bug, not a runtime condition to
    // tolerate.
    if (frame.vdpmRequestSink == nullptr)
    {
        throw std::logic_error(
            "VDPM GPU backend enabled but FrameInfo carries no request sink (the Renderer must set "
            "vdpmGpuBackend and vdpmRequestSink together)");
    }
    return true;
}

void Object::addVdpmRepairCounts(uint32_t& foldovers, uint32_t& coverage) const
{
    for (const auto& binding : bindings_)
    {
        // Only CPU-driven fronts that actually ran this frame contribute; a GPU-backed instance's
        // CPU counters are stale (its lifecycle was skipped), so suppress them until B5c's delayed
        // GPU diagnostics.
        if (binding.vdpmFront && binding.vdpmCpuRanThisFrame)
        {
            foldovers += binding.vdpmFront->foldoversRepaired();
            coverage += binding.vdpmFront->coverageRepaired();
        }
    }
}

void Object::addVdpmChannelStats(VdpmChannelStats& out) const
{
    for (const auto& binding : bindings_)
    {
        // As addVdpmRepairCounts: skip GPU-backed instances whose CPU front didn't run this frame.
        if (binding.vdpmFront && binding.vdpmCpuRanThisFrame)
        {
            const ActiveFront::ChannelStats& cs = binding.vdpmFront->channelStats();
            // Triggers accumulate across instances (a scene total); the max ratios MAX-reduce (the
            // hardest any single instance pushed the channel — summing them would be meaningless).
            out.geometryTriggers += cs.geometryTriggers;
            out.uvTriggers += cs.uvTriggers;
            out.normalTriggers += cs.normalTriggers;
            out.tangentTriggers += cs.tangentTriggers;
            out.maxGeometryRatio = std::max(out.maxGeometryRatio, cs.maxGeometryRatio);
            out.maxUvRatio = std::max(out.maxUvRatio, cs.maxUvRatio);
            out.maxNormalRatio = std::max(out.maxNormalRatio, cs.maxNormalRatio);
            out.maxTangentRatio = std::max(out.maxTangentRatio, cs.maxTangentRatio);
        }
    }
}

void Object::writeForwardUniforms(const FrameInfo& frame, const Mat4& world,
                                  const Mat4& previousWorld, bool hasSkin,
                                  std::span<const Mat4> jointMatrices)
{
    // Per-object UBO — model + hasSkin + previousModel only. The camera (view/proj/cameraPos/
    // view-projections) is per-frame data written once by the Renderer into the set-1 CameraUBO,
    // not duplicated here per object. Skip the mapped write when this frame slot already holds the
    // same world/previousWorld/hasSkin (a static object): after the first write per slot, a still
    // object does no per-frame UBO upload. previousWorld changes every frame for a moving object,
    // so it always rewrites — exactly when it must.
    const std::uint32_t slot = frame.currentFrame;
    const int skinFlag = hasSkin ? 1 : 0;
    if (lastWorld_[slot] != world || lastPreviousWorld_[slot] != previousWorld ||
        lastHasSkin_[slot] != skinFlag)
    {
        ObjectUBO ubo{};
        ubo.model = world;
        ubo.hasSkin = skinFlag;
        ubo.previousModel = previousWorld;
        writeMapped(uniformMapped_[slot], ubo);
        lastWorld_[slot] = world;
        lastPreviousWorld_[slot] = previousWorld;
        lastHasSkin_[slot] = skinFlag;
    }

    if (hasSkin)
    {
        SkinUBO skinUbo{};
        for (std::size_t j = 0; j < jointMatrices.size() && j < kMaxJoints; ++j)
        {
            skinUbo.joints[j] = jointMatrices[j];
        }
        // Previous-frame joints for the motion vector: last frame's, or this frame's on the very
        // first frame (zero deformation velocity until there is a real previous). This mirrors how
        // previousModel works for rigid nodes — previous = the immediately preceding rendered
        // frame.
        const std::span<const Mat4> prevJoints =
            previousJointMatrices_.empty() ? jointMatrices
                                           : std::span<const Mat4>{previousJointMatrices_};
        SkinUBO prevSkinUbo{};
        for (std::size_t j = 0; j < prevJoints.size() && j < kMaxJoints; ++j)
        {
            prevSkinUbo.joints[j] = prevJoints[j];
        }
        for (auto& binding : bindings_)
        {
            writeMapped(binding.skinMapped[frame.currentFrame], skinUbo);
            writeMapped(binding.prevSkinMapped[frame.currentFrame], prevSkinUbo);
        }
        previousJointMatrices_.assign(jointMatrices.begin(), jointMatrices.end());
    }

    for (auto& binding : bindings_)
    {
        // Reset the CPU-VDPM-ran flag for this binding update; it is set true only immediately
        // before the CPU front lifecycle runs below. When the GPU backend drives this instance (or
        // it isn't a VDPM draw) the CPU front is skipped and the flag stays false, so the overlay
        // suppresses this binding's now-stale CPU repair/channel stats (real GPU diagnostics: B5c).
        binding.vdpmCpuRanThisFrame = false;

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

        // VIPM geomorph scalars (Continuous LOD, forward pass only — shadow keeps discrete). Same
        // distance/level basis as the discrete selectLod in buildDrawCommands, so topology + morph
        // agree and the swap lands as morphFactor reaches 1.
        if (frame.lodMode == LodMode::Continuous && frame.lodEnabled &&
            binding.geometry->hasVipmData() && binding.geometry->lods().size() > 1)
        {
            const Vec3 centroid{world[0, 3], world[1, 3], world[2, 3]};
            const float distance = (centroid - frame.cameraPosition).magnitude();
            const VipmSelection sel =
                selectVipm(binding.geometry->lods(), distance, std::abs(frame.proj[1, 1]),
                           static_cast<float>(frame.viewportHeight), frame.lodPixelErrorBudget);
            morphUbo.morphFactor = sel.morphFactor;
            morphUbo.vipmTargetLevel = static_cast<int>(sel.targetLevel);
        }

        // VDPM (View-dependent LOD, forward pass — shadow keeps discrete): refine this instance's
        // active front for the camera and rebuild its dynamic index buffer. buildDrawCommands then
        // points the draw at the freshly-uploaded index set. SKIPPED when the GPU backend drives
        // this instance (backend selected + a live GPU front): the GPU runs the whole score →
        // refine → repair → emit lifecycle in compute (VdpmGpuManager), so the ~per-instance CPU
        // cost is retired. A per-mesh fallback instance (ineligible mesh → NullVdpmFront) keeps the
        // CPU path.
        if (frame.lodMode == LodMode::ViewDependent && frame.lodEnabled && binding.vdpmFront &&
            !vdpmGpuDrives(frame, binding))
        {
            binding.vdpmCpuRanThisFrame = true;
            // Material-aware channel scales: disable the channels this material can't show (unlit →
            // no shading/tangent; no normal map → no tangent frame; no textures → no UV) and
            // tighten the normal channel for glossy materials. Derived here at refine time so the
            // collapse stream stays material-agnostic (activeMaterial is always set — see
            // createForwardBindings).
            const VdpmChannelScales vs = vdpmChannelScales(*binding.activeMaterial);
            // Back-face cone suppression is only valid when the draw actually culls back-faces: a
            // double-sided or blended material renders its back-faces, so their refinement must NOT
            // be suppressed (matches the renderer's cull state — blend pipeline is CullMode::eNone,
            // opaque double-sided sets dynamic cull none).
            const Material& mat = *binding.activeMaterial;
            const bool rasterBackfaceCulling =
                mat.alphaMode() != AlphaMode::Blend && !mat.doubleSided();
            binding.vdpmFront->refineForView(binding.geometry->vertices(), world,
                                             frame.cameraPosition, std::abs(frame.proj[1, 1]),
                                             static_cast<float>(frame.viewportHeight),
                                             frame.lodPixelErrorBudget, kVdpmSilhouetteBoost,
                                             rasterBackfaceCulling, vs.uv, vs.normal, vs.tangent);
            // Joint foldover+coverage repair (MANDATORY before emission): closes the foldover and
            // silhouette-coverage holes a selective front leaves, to a joint fixed point (each can
            // reopen the other). Uses the JITTER-FREE currentViewProj, not the TAA-jittered
            // frame.proj — the sub-pixel jitter would shift the coverage test ±0.5px each frame and
            // thrash the front (a borderline hole flickering with no camera motion). Same cull
            // policy as refineForView: a double-sided/blended draw shows its back-faces, so they
            // need coverage too.
            binding.vdpmFront->repairFront(
                binding.geometry->vertices(), world, frame.cameraPosition, frame.currentViewProj,
                static_cast<float>(frame.viewportWidth), static_cast<float>(frame.viewportHeight),
                rasterBackfaceCulling);
            binding.vdpmFront->emitActiveIndices(
                binding.geometry->vertices(), binding.geometry->indices(), binding.vdpmEmitScratch);
            const std::vector<uint32_t>& idx = binding.vdpmEmitScratch;
            binding.vdpmIndexCount = static_cast<uint32_t>(idx.size());
            writeMapped(binding.vdpmIndexMapped[frame.currentFrame], idx.data(),
                        idx.size() * sizeof(uint32_t));
            // Record the indirect draw command from the freshly-emitted count. EVERY field is set
            // (no stale carry-over): one instance, from the start of the index buffer. Written
            // after the fence has freed this frame slot (persistent mapping) and before submission,
            // so the submit makes the host write available to the draw — no explicit barrier (the
            // dynamic index buffer above relies on the identical guarantee). Stage B5 replaces this
            // CPU write with a compute shader + a compute→indirect-read barrier.
            const DrawIndexedIndirectCommand indirectCmd{.indexCount = binding.vdpmIndexCount,
                                                         .instanceCount = 1,
                                                         .firstIndex = 0,
                                                         .vertexOffset = 0,
                                                         .firstInstance = 0};
            writeMapped(binding.vdpmIndirectMapped[frame.currentFrame], indirectCmd);
        }
        writeMapped(binding.morphUboMapped[frame.currentFrame], morphUbo);
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
        writeMapped(binding.shadowMapped[frame.currentFrame], shadowData);
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
        // View-dependent LOD: draw the per-instance active front's dynamic index set (refined +
        // uploaded in writeForwardUniforms this frame). Takes precedence over discrete/VIPM.
        if (frame.lodMode == LodMode::ViewDependent && frame.lodEnabled && binding.vdpmFront)
        {
            if (vdpmGpuDrives(frame, binding))
            {
                // GPU-driven VDPM (Stage B5b-2): tag the forward draw with its front handle and
                // queue a work request; the renderer harvests the tag from the camera-visible
                // forward buckets (a shadow-only instance's forward command never survives the
                // camera cull, so its front's compute is not recorded), records the deduped
                // requests, and RESOLVES indexBuffer/indirectBuffer to the GPU-emitted ring for
                // this frame after collection. The GPU-emitted index stream is ALWAYS uint32,
                // independent of the source geometry's index representation. The CPU vdpm* buffers
                // are not written (the CPU front was skipped in writeForwardUniforms), so
                // indexCount is GPU-only — left 0 for the triangle overlay until B5c's delayed
                // diagnostics.
                cmd.vdpmGpuFront = binding.vdpmGpuFront;
                cmd.indexType = DrawIndexType::UInt32;
                cmd.indexCount = 0;
                const Material& vmat = *binding.activeMaterial;
                const VdpmChannelScales vscales = vdpmChannelScales(vmat);
                const bool cullBackfaces =
                    vmat.alphaMode() != AlphaMode::Blend && !vmat.doubleSided();
                frame.vdpmRequestSink->push_back(
                    VdpmWorkRequest{.front = binding.vdpmGpuFront,
                                    .world = world,
                                    .uvScale = vscales.uv,
                                    .normalScale = vscales.normal,
                                    .tangentScale = vscales.tangent,
                                    .rasterBackfaceCulling = cullBackfaces});
            }
            else
            {
                // CPU front (per-mesh fallback, or GPU backend off): draw the per-instance active
                // front's dynamic index set refined + uploaded in writeForwardUniforms this frame.
                cmd.indexBuffer = binding.vdpmIndexBufs[frame.currentFrame];
                cmd.indexCount = binding.vdpmIndexCount;
                // Draw indirect: the renderer reads the count from the GPU buffer (Stage A).
                // indexCount stays set above for the triangle overlay. Offset 0 — one command per
                // instance buffer.
                cmd.indirectBuffer = binding.vdpmIndirectBufs[frame.currentFrame];
                cmd.indirectOffset = 0;
            }
        }
        // Discrete LOD: swap in a coarser index set (same vertex buffer) for distant/small static
        // meshes, chosen so the level's geometric error stays within the pixel budget.
        else if (frame.lodEnabled && binding.geometry->lods().size() > 1)
        {
            const float distance = (centroid - frame.cameraPosition).magnitude();
            const std::size_t level =
                selectLod(binding.geometry->lods(), distance, std::abs(frame.proj[1, 1]),
                          static_cast<float>(frame.viewportHeight), frame.lodPixelErrorBudget);
            cmd.indexBuffer = binding.geometry->lods()[level].indexBuffer;
            cmd.indexCount = binding.geometry->lods()[level].indexCount;
            cmd.lodLevel = static_cast<uint32_t>(level);
        }
        // Forward set 0 is pushed inline at draw time, not bound — carry the
        // buffer handles instead of a descriptor set.
        cmd.frameUbo = uniformBufs_[frame.currentFrame];
        cmd.cameraUbo = frame.cameraUbo;
        cmd.skinUbo = binding.skinBufs[frame.currentFrame];
        cmd.prevSkinUbo = binding.prevSkinBufs[frame.currentFrame];
        cmd.morphUbo = binding.morphUboBufs[frame.currentFrame];
        cmd.morphSsbo = binding.morphSsbo;
        cmd.vipmBuffer = binding.vipmBuffer;
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
        const auto& transmission = mat.transmission();
        const bool hasTransmissionFactor = transmission.has_value() && transmission->factor > 0.0f;
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
            // Shadows keep discrete LOD and draw directly — clear the VDPM indirect handle the copy
            // inherited from the forward command, or the "non-null selects indirect" invariant
            // would point the shadow draw at the forward index count. Clear the GPU-front handle
            // too: VDPM is forward-only, so a shadow command must never carry a front (a future
            // generic front resolver would otherwise pick it up from the shadow bucket).
            shadowCmd.indirectBuffer = NullBuffer;
            shadowCmd.indirectOffset = 0;
            shadowCmd.vdpmGpuFront = NullVdpmFront;
            // Shadows tolerate a coarser LOD than the main view (silhouette detail matters less).
            if (frame.lodEnabled && shadowGeometry->lods().size() > 1)
            {
                const float distance = (centroid - frame.cameraPosition).magnitude();
                const std::size_t level =
                    selectLod(shadowGeometry->lods(), distance, std::abs(frame.proj[1, 1]),
                              static_cast<float>(frame.viewportHeight),
                              frame.lodPixelErrorBudget * kShadowLodBias);
                shadowCmd.indexBuffer = shadowGeometry->lods()[level].indexBuffer;
                shadowCmd.indexCount = shadowGeometry->lods()[level].indexCount;
            }
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
