#include <fire_engine/render/shadows.hpp>

#include <format>
#include <stdexcept>
#include <string_view>

#include <fire_engine/render/device.hpp>
#include <fire_engine/render/render_target.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/viewport.hpp>

namespace fire_engine
{

namespace
{

// One-subresource (single mip, single array layer) layout transition through
// synchronization2. Dynamic rendering does no implicit attachment transitions,
// so each shadow iteration brackets its draw with these.
void imageLayerBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageAspectFlags aspect,
                       uint32_t layer, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                       vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                       vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::ImageMemoryBarrier2 b{
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = aspect,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = layer,
                                                      .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}

// A diagnostic row the recorder is about to rasterise into that the plan never claimed, or that
// holds a different logical view. Terminal because the alternative is a row whose counters are the
// sum of two unrelated views under one of their names — worse than no measurement, because it reads
// like one.
[[noreturn]] void contradictoryShadowViewRow(std::string_view group, std::size_t slot)
{
    throw std::runtime_error(
        std::format("shadow view row {} slot {} was not claimed by this frame's plan, or is "
                    "claimed by a different logical view",
                    group, slot));
}

void recordShadowDrawBucket(vk::CommandBuffer cmd, const PreparedShadowLayer& layer,
                            const Resources& resources, ShadowPipelinePair pipelines,
                            const ShadowPushConstants& viewConstants,
                            const ShadowLogicalViewId& logicalId, ShadowViewGroup group,
                            std::size_t slot, ShadowFrameStats& stats)
{
    // One raster pass, counted where the GPU work is: this call brackets a real depth image. The
    // row was CLAIMED during preparation, and the identity is checked against that claim rather
    // than re-claimed — otherwise "some view claimed this row" would be enough, and rasterising
    // view B into view A's row would still read plausibly under A's name.
    ShadowViewStats& viewStats = stats.view(group, slot);
    if (!viewStats.beginRasterPass(logicalId))
    {
        contradictoryShadowViewRow(toString(group), slot);
    }

    // The pipeline currently bound, so the two fragment paths can interleave freely within one
    // iteration: a layer's draws are one list, and splitting it by material would either reorder
    // the draws or walk it twice. NullPipeline means "nothing bound yet in this iteration".
    PipelineHandle boundPipeline = NullPipeline;
    for (const PreparedShadowDraw& draw : layer.draws)
    {
        // SH-05: the fragment path this caster needs, from the classification preparation recorded
        // — the same value the comparison holds, so the path that rasterises a cached map and the
        // path described by its content cannot differ.
        const PipelineHandle pipelineHandle = pipelines.forCaster(draw.alpha);
        const bool pipelineChanged = pipelineHandle != boundPipeline;
        if (pipelineChanged)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             resources.vulkanPipeline(pipelineHandle));
            boundPipeline = pipelineHandle;
        }
        const vk::PipelineLayout layout = resources.vulkanPipelineLayout(pipelineHandle);
        // Cull mode is dynamic on every shadow pipeline (SH-05), so it is set per draw and not once
        // per pass: within one layer a single-sided and a double-sided caster need different
        // answers, and the pipeline carries none. The EFFECTIVE answer was resolved in preparation
        // — this is only its Vulkan spelling.
        cmd.setCullMode(shadowCullMode(draw.cull));
        if (draw.vertexBuffer != NullBuffer)
        {
            cmd.bindVertexBuffers(0, resources.vulkanBuffer(draw.vertexBuffer),
                                  {vk::DeviceSize{0}});
        }

        vk::IndexType indexType = draw.indexType == DrawIndexType::UInt32 ? vk::IndexType::eUint32
                                                                          : vk::IndexType::eUint16;
        // The RESOLVED buffer, never a command's: a shadow command carries none, and the level that
        // chose this one was decided per view during preparation.
        cmd.bindIndexBuffer(resources.vulkanBuffer(draw.indexBuffer), 0, indexType);

        // Shadow set 0 is pushed inline (core 1.4 push descriptors) — no allocated
        // per-object descriptor set, mirroring the forward pass.
        pushShadowObjectDescriptors(cmd, resources, layout, draw.shadowUbo, draw.skinUbo,
                                    draw.morphUbo, draw.morphSsbo);
        if (pipelineChanged)
        {
            // Bindless materials (set 2) for the masked fragment path. Bound AFTER the push
            // descriptors that establish set 0, matching the ordering the forward pass documents
            // (Vulkan layout compatibility preserves set 0, and this order also avoids a validation
            // -layer first-use state-tracking defect). Bound for the opaque path too: all four
            // shadow pipelines share one layout, so this costs one call per pipeline change and
            // removes any question of which path had the set.
            //
            // The SET itself is the one Resources allocated from the FORWARD pipeline's bindless
            // layout — there is a single global materials set, not one per pipeline. That is legal
            // because the layouts are IDENTICALLY DEFINED (same bindings, same binding flags, from
            // Pipeline::createBindlessDescriptorSetLayout), which Vulkan treats as the same layout
            // for compatibility. Change those bindings and every pipeline that opts in changes with
            // them, which is exactly the coupling we want here.
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 2,
                                   resources.bindlessDescriptorSet(), {});
        }
        // Per draw, because materialIndex varies per draw: the view's constants supply everything
        // else. One struct assembled in one place, so the masked path cannot read a material index
        // that belongs to the previously recorded caster.
        ShadowPushConstants pc = viewConstants;
        pc.materialIndex = draw.materialIndex;
        cmd.pushConstants<ShadowPushConstants>(
            layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.drawIndexed(draw.indexCount, 1, 0, 0, 0);
    }
}

} // namespace

Shadows::Shadows(const Device& device, Resources& resources)
    : resources_{&resources},
      shadowPipeline_(device, Pipeline::shadowConfig()),
      shadowMaskedPipeline_(device, Pipeline::shadowMaskedConfig()),
      selfShadowSecondPipeline_(device, Pipeline::selfShadowSecondConfig()),
      selfShadowSecondMaskedPipeline_(device, Pipeline::selfShadowSecondMaskedConfig())
{
    // Four pipelines for the four shadow material modes the plan names, not eight: the ALPHA half
    // (opaque / masked) needs a different fragment shader and so a different pipeline, while the
    // SIDEDNESS half is dynamic cull state on all of them. The self-shadow FIRST layer needs no
    // pipeline of its own any more — it is the main pair recorded with AllFaces.
    shadowPipelines_ = ShadowPipelinePair{
        .opaque = resources_->registerPipeline(shadowPipeline_.pipeline(),
                                               shadowPipeline_.pipelineLayout()),
        .masked = resources_->registerPipeline(shadowMaskedPipeline_.pipeline(),
                                               shadowMaskedPipeline_.pipelineLayout()),
    };
    selfShadowSecondPipelines_ = ShadowPipelinePair{
        .opaque = resources_->registerPipeline(selfShadowSecondPipeline_.pipeline(),
                                               selfShadowSecondPipeline_.pipelineLayout()),
        .masked = resources_->registerPipeline(selfShadowSecondMaskedPipeline_.pipeline(),
                                               selfShadowSecondMaskedPipeline_.pipelineLayout()),
    };

    // Dynamic rendering needs no framebuffers: recordPass binds each per-layer
    // depth view straight into vk::RenderingInfo (depth-only — no colour
    // attachment). We retain only the texture handles; per-layer views are
    // fetched from Resources at record time. Every shadow depth image is created
    // already transitioned to DepthStencilReadOnlyOptimal across all layers
    // (createShadowMap), the resting layout the forward sampler expects and the
    // per-iteration barriers cycle through.
    shadowMapHandle_ = resources_->createShadowMap(kShadowMapExtent, kShadowCascadeCount);
    worldShadowMapHandle_ = resources_->createShadowMap(kShadowMapExtent, kShadowCascadeCount);

    selfShadowFirstMapHandle_ =
        resources_->createShadowMap(kSkinnedSelfShadowMapExtent, kMaxSkinnedSelfShadowCasters);
    selfShadowMapHandle_ =
        resources_->createShadowMap(kSkinnedSelfShadowMapExtent, kMaxSkinnedSelfShadowCasters);

    // Spot casters share a 2D-array depth image, one layer per caster.
    spotShadowMapHandle_ = resources_->createShadowMap(kSpotShadowMapExtent, kMaxSpotShadowCasters);

    // Point casters: one cubemap-array depth image, kCubeFaceCount faces per caster. Layout
    // `kCubeFaceCount * cube + face` matches Resources::vulkanPointShadowFaceView and the flat
    // point-view slot the diagnostics and the view set use (`shadowPointViewSlot`).
    pointShadowMapHandle_ =
        resources_->createPointShadowMap(kPointShadowMapExtent, kMaxPointShadowCasters);

    auto& shared = resources_->sharedTextures();
    shared.shadowMap = shadowMapHandle_;
    shared.worldShadowMap = worldShadowMapHandle_;
    shared.selfShadowFirstMap = selfShadowFirstMapHandle_;
    shared.selfShadowMap = selfShadowMapHandle_;
    shared.spotShadowMap = spotShadowMapHandle_;
    shared.pointShadowMap = pointShadowMapHandle_;
    // Debug ShadowDepth view samples the CSM depth map directly (raw depth ==
    // the gl_FragCoord.z the old throwaway colour attachment stored).
    shared.shadowDebugImage = shadowMapHandle_;
}

// Which physical depth image a prepared layer fills, and which fragment paths rasterise it.
//
// The image and the pipeline pair are both functions of (family, layer kind), so they are resolved
// TOGETHER: a self-shadow second layer bound to the first layer's image would rasterise the
// dual-depth rejection into the map it is supposed to be sampling, and every counter would still
// read correctly.
Shadows::LayerTarget Shadows::layerTarget(ShadowViewGroup group, std::size_t slot,
                                          ShadowLayerKind kind) const
{
    const auto layer = static_cast<uint32_t>(slot);
    switch (group)
    {
    case ShadowViewGroup::Cascade:
        return {resources_->vulkanImage(shadowMapHandle_),
                resources_->vulkanShadowMapLayerView(shadowMapHandle_, layer), layer,
                shadowPipelines_};
    case ShadowViewGroup::WorldOnly:
        return {resources_->vulkanImage(worldShadowMapHandle_),
                resources_->vulkanShadowMapLayerView(worldShadowMapHandle_, layer), layer,
                shadowPipelines_};
    case ShadowViewGroup::Self:
        // The FIRST layer captures whatever the light sees first (the main pair, recorded with an
        // all-faces policy since SH-05 made cull mode dynamic); the SECOND samples that image and
        // discards the surface it already recorded, which needs its own fragment shaders.
        if (kind == ShadowLayerKind::SelfSecondDepth)
        {
            return {resources_->vulkanImage(selfShadowMapHandle_),
                    resources_->vulkanShadowMapLayerView(selfShadowMapHandle_, layer), layer,
                    selfShadowSecondPipelines_};
        }
        return {resources_->vulkanImage(selfShadowFirstMapHandle_),
                resources_->vulkanShadowMapLayerView(selfShadowFirstMapHandle_, layer), layer,
                shadowPipelines_};
    case ShadowViewGroup::Spot:
        return {resources_->vulkanImage(spotShadowMapHandle_),
                resources_->vulkanShadowMapLayerView(spotShadowMapHandle_, layer), layer,
                shadowPipelines_};
    case ShadowViewGroup::Point:
    {
        // ONE derivation feeding the attachment view and the depth layer: the point family's flat
        // slot IS `6 * cube + face`, so the cube and the face come back out of it rather than being
        // carried alongside and trusted to agree.
        const auto faces = static_cast<std::size_t>(kCubeFaceCount);
        return {resources_->vulkanImage(pointShadowMapHandle_),
                resources_->vulkanPointShadowFaceView(pointShadowMapHandle_,
                                                      static_cast<uint32_t>(slot / faces),
                                                      static_cast<uint32_t>(slot % faces)),
                layer, shadowPipelines_};
    }
    case ShadowViewGroup::Count:
        break;
    }
    // Unreachable for a real group; the switch is exhaustive over the families the plan indexes.
    throw std::runtime_error(
        std::format("shadow layer target asked for group {}", static_cast<int>(group)));
}

void Shadows::recordPass(vk::CommandBuffer cmd, const ShadowFramePlan& plan,
                         ShadowFrameStats& stats, const GpuProfiler& profiler,
                         uint32_t frameIndex) const
{
    // Nothing to record at all — `--no-shadows`, a scene with no light any family is fitted to, or
    // (once the residency store lands) a frame in which every view's map was reused. Returning here
    // is what makes that OBSERVABLE: no draws, no clears, no timestamps, so every shadow row in the
    // diagnostics and every shadow group in the GPU timings reads zero.
    if (plan.recordsNothing())
    {
        return;
    }
    // Bottom-to-bottom group timing: both boundaries are bottom-of-pipe stamps, so adjacent
    // sub-millisecond groups cannot overlap and inflate each other the way top-to-bottom spans do.
    // Every pass in the engine stamps this way now — begin() itself is bottom-of-pipe — so this is
    // no longer a shadow-only convention, just the convention. A group that records nothing leaves
    // its two stamps unwritten and reports 0, rather than an empty span that reads as a small real
    // cost and inflates the frame total. An active family with zero prepared draws is still timed —
    // its clears and layout barriers are real GPU work.
    //
    // `recording` gates the body AND the stamps together, which is the point: a family this frame
    // does not record must draw nothing, clear nothing and time nothing, and one gate is what makes
    // those three the same answer. It is the PLAN's answer — `records()` is true only if some view
    // of the family has work — so a family whose maps were all reused opens no span around nothing.
    const auto timeGroup = [&](ProfilePass pass, bool recording, auto&& body)
    {
        if (!recording)
        {
            return;
        }
        profiler.begin(cmd, frameIndex, pass);
        body();
        profiler.end(cmd, frameIndex, pass);
    };

    const vk::ClearValue depthClear{.depthStencil =
                                        vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0}};

    // Renders one prepared layer with depth-only dynamic rendering. Depth rests in
    // DepthStencilReadOnlyOptimal (the forward-sampler layout) between frames, so we cycle it
    // ReadOnly → Attachment → ReadOnly. No colour attachment: current MoltenVK commits depth-only
    // stores under dynamic rendering.
    const auto recordLayer = [&](ShadowViewGroup group, std::size_t slot,
                                 const PreparedShadowView& view, const PreparedShadowLayer& layer,
                                 const ShadowPushConstants& pc)
    {
        const LayerTarget target = layerTarget(group, slot, layer.kind);
        const uint32_t extent = view.extent();
        vk::Viewport vp = makeFullViewport(static_cast<float>(extent), static_cast<float>(extent));
        vk::Rect2D scissor{
            .offset = vk::Offset2D{.x = 0, .y = 0},
            .extent = vk::Extent2D{.width = extent, .height = extent},
        };

        imageLayerBarrier(cmd, target.image, vk::ImageAspectFlagBits::eDepth, target.layer,
                          vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal,
                          vk::PipelineStageFlagBits2::eFragmentShader,
                          vk::AccessFlagBits2::eShaderRead,
                          vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                          vk::AccessFlagBits2::eDepthStencilAttachmentWrite);

        vk::RenderingAttachmentInfo depth{
            .imageView = target.view,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = depthClear,
        };
        cmd.beginRendering(makeRenderingInfo(scissor, {}, &depth));
        cmd.setViewport(0, vp);
        cmd.setScissor(0, scissor);
        // From the PREPARED view, which is also what the comparison holds: a bias changed without
        // the cache seeing it would keep a map whose depth was written under different rules.
        cmd.setDepthBias(view.depthBiasConstant(), 0.0f, view.depthBiasSlope());
        // The push constants are pushed PER DRAW inside the bucket (SH-05 added a per-draw
        // materialIndex to the block), so `pc` travels in as this view's part of them rather than
        // being pushed here — one struct, assembled in one place, instead of a view-level push a
        // per-draw push would then have to agree with.
        recordShadowDrawBucket(cmd, layer, *resources_, target.pipelines, pc, view.logicalId(),
                               group, slot, stats);
        cmd.endRendering();

        imageLayerBarrier(cmd, target.image, vk::ImageAspectFlagBits::eDepth, target.layer,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal,
                          vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                          vk::PipelineStageFlagBits2::eLateFragmentTests,
                          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                          vk::PipelineStageFlagBits2::eFragmentShader,
                          vk::AccessFlagBits2::eShaderRead);
    };

    // One family: every slot the plan says RECORDS, in slot order, each of its layers in the order
    // preparation built them. A slot that is reused or absent is not touched at all — no barrier,
    // no clear, no draw — which is the whole point of the disposition.
    const auto recordFamily = [&](ShadowViewGroup group)
    {
        for (std::size_t slot = 0; slot < shadowViewSlotCount(group); ++slot)
        {
            if (!shadowViewRecords(plan.disposition(group, slot)))
            {
                continue;
            }
            const PreparedShadowView* view = plan.view(group, slot);
            if (view == nullptr)
            {
                // A slot that records must carry content — `ShadowFramePlan::add` refuses any
                // other combination — so this is a contradiction inside the plan rather than a
                // frame to degrade through.
                throw std::runtime_error(
                    std::format("shadow view {} slot {} records but carries no prepared content",
                                toString(group), slot));
            }
            // ONE assembly of the view's constants, shared by its layers. Every field comes from
            // the prepared view: the matrix it rasterises with, how its fragments store depth, and
            // — for a point face — the light that depth is measured against. `selfShadowSlot` is
            // the physical slot itself, which is what the second depth layer samples the first
            // layer's image with.
            ShadowPushConstants pc{};
            pc.lightViewProj = view->viewProj();
            pc.radialDepth = shadowRadialDepthFlag(view->depthMode());
            pc.lightPosRange[0] = view->lightPosition().x();
            pc.lightPosRange[1] = view->lightPosition().y();
            pc.lightPosRange[2] = view->lightPosition().z();
            pc.lightPosRange[3] = view->lightRange();
            pc.selfShadowSlot = group == ShadowViewGroup::Self ? static_cast<int>(slot) : -1;

            for (const PreparedShadowLayer& layer : view->layers())
            {
                recordLayer(group, slot, *view, layer, pc);
            }
        }
    };

    // The main CSM and the world-only CSM are recorded as CONTIGUOUS groups rather than interleaved
    // per cascade. Nothing depends on the interleaving (each layer is independently barriered), and
    // grouping them is what lets each family carry one bottom-to-bottom timestamp boundary in the
    // per-group GPU timing — interleaved, the two families' costs could not be separated at all.
    //
    // The world-only CSM exists so skinned receivers can sample a cascade without their own
    // geometry; with no skinned draw this frame no world-only view is prepared, so the duplicate
    // 4-cascade render is skipped entirely — and the receiver is TOLD (the WORLD_ONLY bit), rather
    // than left to rely on nothing sampling it.
    for (std::size_t g = 0; g < kShadowViewGroupCount; ++g)
    {
        const auto group = static_cast<ShadowViewGroup>(g);
        timeGroup(shadowProfilePass(group), plan.records(group), [&] { recordFamily(group); });
    }
}

} // namespace fire_engine
