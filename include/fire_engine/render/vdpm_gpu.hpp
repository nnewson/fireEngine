#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/graphics/vertex.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/compute_pipeline.hpp>
#include <fire_engine/render/resources.hpp>
#include <fire_engine/render/ubo.hpp>
#include <fire_engine/render/vdpm_scan.hpp>

namespace fire_engine
{

struct DependencyDag; // graphics/vdpm_parallel.hpp — the forest's refine-dependency DAG (B3
                      // topology)

// CPU → GPU packing for the VDPM GPU-front scoring ABI (Stage B1). These convert the Vulkan-free
// scoring authority (VdpmViewParams / VertexSplit, graphics/vdpm.hpp) into the std430 SSBO images
// (render/ubo.hpp) the compute shader reads. Keeping the conversion here — a thin, tested field
// copy — is what stops the oracle, the uploader, and the shader from drifting into three different
// scorings. (The score pipeline + dispatch live in vdpm_gpu.cpp alongside these.)

// Static per-split metric record. parent/child become IDs into the canonical-position buffer.
[[nodiscard]] VdpmSplitGpu packVdpmSplit(const VertexSplit& split) noexcept;

// One canonical vertex's object-space position (padded to a vec4).
[[nodiscard]] VdpmPositionGpu packVdpmPosition(const Vec3& position) noexcept;

// Per-instance params: the std430 image of `view` plus the three buffer_reference device addresses
// and the split count. `worldLinear` is written as three padded vec4 columns (a GLSL mat3).
[[nodiscard]] VdpmScoreParams packVdpmScoreParams(const VdpmViewParams& view,
                                                  std::uint64_t splitsAddress,
                                                  std::uint64_t positionsAddress,
                                                  std::uint64_t outputsAddress,
                                                  std::uint32_t splitCount) noexcept;

// The score compute pipeline: `shaders/vdpm_score.comp`. No descriptor sets (everything is reached
// by buffer_reference); one push-constant range carrying the params block's device address.
[[nodiscard]] ComputePipelineConfig vdpmScorePipelineConfig();

// The four VDPM GPU emit pipeline configs (Stage B2): ancestor resolution, per-face survival,
// stable scatter, and the one-invocation finalize. Each is descriptor-free (all buffers reached by
// buffer_reference) with a single push-constant range carrying its ABI push block. The exclusive
// scan between survival and scatter is the shared VdpmScan primitive.
[[nodiscard]] ComputePipelineConfig vdpmAncestorPipelineConfig();
[[nodiscard]] ComputePipelineConfig vdpmSurvivalPipelineConfig();
[[nodiscard]] ComputePipelineConfig vdpmScatterPipelineConfig();
[[nodiscard]] ComputePipelineConfig vdpmEmitFinalizePipelineConfig();

// The four VDPM GPU refine/coarsen pipeline configs (Stage B3): mark (seed required), close
// (requirement closure over the DAG), refine (rank-ordered apply), coarsen (rank-ordered collapse).
// Descriptor-free; each carries its ABI push block.
[[nodiscard]] ComputePipelineConfig vdpmMarkPipelineConfig();
[[nodiscard]] ComputePipelineConfig vdpmClosePipelineConfig();
[[nodiscard]] ComputePipelineConfig vdpmRefinePipelineConfig();
[[nodiscard]] ComputePipelineConfig vdpmCoarsenPipelineConfig();

// The VDPM GPU repair pipeline configs (Stage B4): the per-face foldover/coverage detector and the
// full-detail fallback seed. (The ancestor resolve reuses vdpmAncestorPipelineConfig.)
[[nodiscard]] ComputePipelineConfig vdpmRepairDetectPipelineConfig();
[[nodiscard]] ComputePipelineConfig vdpmRepairFallbackPipelineConfig();

// The persistent repair kernel config (perf arc, Stage 2): `shaders/vdpm_repair_kernel.comp` runs
// the whole repair fixpoint for one front in ONE workgroup. Descriptor-free; push =
// VdpmRepairKernelPush (job-array address + count).
[[nodiscard]] ComputePipelineConfig vdpmRepairKernelPipelineConfig();

// A rank's contiguous range in `splitsByRank` — the CPU recorder issues one dispatch per rank over
// `[offset, offset + count)`, and an array of these is uploaded device-local for the persistent
// repair kernel (binding.rankRangesAddress). The shared GPU ABI struct lives in ubo.hpp (the
// project invariant for every host↔shader struct); RankRange is the one authority used both places.
using RankRange = VdpmRankRangeGpu;

// Validate a rank-range list as a contiguous partition of `[0, splitCount)`: rank 0 at offset 0,
// each rank starting where the previous ended, the last ending exactly at splitCount. Throws
// std::runtime_error on a gap, overlap, wrong terminal count, or a count that overflows (the
// running offset accumulates in 64-bit). Pure + Vulkan-free — run at the mesh-build boundary BEFORE
// any GPU upload, and unit-tested directly.
void validateVdpmRankRanges(std::span<const RankRange> ranges, std::uint32_t splitCount);

// The IMMUTABLE GPU binding of a VdpmGpuMesh — device addresses + counts, copied into a front so
// the front never holds a pointer into a (movable) mesh object. The emit block (indices, weld,
// removalParent, wedge choices) is populated ONLY by the full build; a score-only build leaves
// `hasEmitData == false` and those addresses null.
struct VdpmGpuMeshBinding
{
    std::uint64_t splitsAddress{0};
    std::uint64_t positionsAddress{0};
    std::uint32_t splitCount{0};

    // Emit static data (full build only).
    std::uint64_t indicesAddress{0};       // 3 * faceCount original corner indices
    std::uint64_t weldAddress{0};          // per original vertex: its canonical vertex
    std::uint64_t removalParentAddress{0}; // per canonical: one removal-parent step (root == self)
    std::uint64_t wedgeChoicesAddress{0};  // CSR restored wedge per (original vertex, depth)
    std::uint64_t wedgeOffsetsAddress{0};  // per original vertex: CSR start into wedgeChoices
    std::uint32_t faceCount{0};            // finest face count (= indices / 3)
    std::uint32_t vertexCount{0};          // canonical vertex count (= forest.vertexCount)
    std::uint32_t maxDepth{0};             // deepest removal-parent chain (the ancestor loop bound)
    bool hasEmitData{false};

    // Canonical finest faces (Stage B4 repair) — `canonicalFaces(weld, indices)`, the per-face
    // detector's input. Full build only. `finestFaceCount` post-weld faces (degenerate faces
    // dropped, so <= faceCount).
    std::uint64_t finestFacesAddress{0}; // 3 * finestFaceCount canonical corner indices
    std::uint64_t removingSplitAddress{
        0}; // per canonical: the split that removes it (kNoSplit=root)
    std::uint32_t finestFaceCount{0};

    // Refine/coarsen front topology (Stage B3) — built from the forest's dependency DAG in the base
    // build (so every mesh carries it); null/zero only for an empty (zero-split) forest.
    std::uint64_t frontSplitsAddress{
        0}; // VdpmFrontSplitGpu per split (vertices + dependency splits)
    std::uint64_t splitsByRankAddress{0}; // uint32 per split, packed by ascending rank
    std::uint64_t rankRangesAddress{0};   // RankRange[maxRank+1] {offset,count} into splitsByRank
    std::uint32_t maxRank{0};             // rank passes = maxRank + 1
};

// STATIC per-geometry GPU data — the per-split metric records + the canonical-vertex positions,
// device-local and uploaded ONCE. Shared by every instance of the geometry (never duplicated per
// instance); the later repair/emit stages reuse the same positions. parent/child IDs in the split
// records index the positions array.
class VdpmGpuMesh
{
public:
    // Score-only build: `vertices` is the ORIGINAL vertex array (canonical IDs index into it);
    // `forest` supplies the splits. THE VALIDATION BOUNDARY: runs `validateForest` and checks
    // `vertices.size() == forest.vertexCount` (throws std::runtime_error otherwise) before anything
    // is walked or uploaded — a shader must never see a malformed forest. Both buffers uploaded
    // device-local. The binding carries no emit data (`hasEmitData == false`).
    [[nodiscard]] static VdpmGpuMesh build(Resources& resources, std::span<const Vertex> vertices,
                                           const VertexForest& forest);

    // Full build (score + emit, Stage B2): additionally uploads the static emit geometry — the
    // finest face indices, the position weld (`weldByPosition(vertices)`, derived here so it can't
    // disagree), the collapsed removal-parent chain, and the precomputed wedge-choice CSR. Mirrors
    // the score-only validation boundary and further checks the index buffer: `indices.size() % 3
    // == 0`, every index in `[0, vertexCount)`, and the finest index count fits 32-bit (throws
    // std::runtime_error otherwise). `hasEmitData == true` on the returned binding.
    [[nodiscard]] static VdpmGpuMesh build(Resources& resources, std::span<const Vertex> vertices,
                                           std::span<const std::uint32_t> indices,
                                           const VertexForest& forest);

    // GPU-eligibility gate for the B5 backend selector: whether a mesh of these dimensions can be
    // driven on this device without any static compute dispatch (score / ancestor / survival /
    // scatter / scan) exceeding the 1-D group-count cap — so the selector can fall back to the CPU
    // front BEFORE any GPU allocation, rather than fault at frame-record time. `build` enforces the
    // SAME bound (throwing) as part of its validation boundary, before it allocates anything; the
    // record-time checks in recordScore/recordEmit stay as defence-in-depth. Group counts are
    // computed in 64-bit, so an oversized forest/index count can't wrap. noexcept — a pure query.
    [[nodiscard]] static bool fitsComputeDispatchLimits(const Resources& resources,
                                                        const VertexForest& forest,
                                                        std::size_t indexCount) noexcept;

    [[nodiscard]] VdpmGpuMeshBinding binding() const noexcept
    {
        return binding_;
    }
    [[nodiscard]] std::uint64_t splitsAddress() const noexcept
    {
        return splitsAddress_;
    }
    [[nodiscard]] std::uint64_t positionsAddress() const noexcept
    {
        return positionsAddress_;
    }
    [[nodiscard]] std::uint32_t splitCount() const noexcept
    {
        return splitCount_;
    }
    // The per-rank dispatch ranges (Stage B3), copied into a front at build so it holds no pointer
    // into this (movable) mesh. Empty for a zero-split forest.
    [[nodiscard]] std::span<const RankRange> rankRanges() const noexcept
    {
        return rankRanges_;
    }
    // The coarsest front's initial `active` flags (roots = 1, others = 0), uploaded into a front's
    // persistent state at build. Per original/canonical vertex.
    [[nodiscard]] std::span<const std::uint32_t> initialActive() const noexcept
    {
        return initialActive_;
    }

private:
    // Pack + upload the static splits/positions + front topology (from the caller-built `dag`) and
    // fill the score portion of the binding. Assumes the forest is ALREADY validated and the DAG
    // ALREADY built: the caller runs every validation + throwing CPU derivation (incl.
    // buildDependencyDag) BEFORE the first upload, so malformed input orphans no GPU-resource
    // entries.
    static void uploadScoreData(VdpmGpuMesh& mesh, Resources& resources,
                                std::span<const Vertex> vertices, const VertexForest& forest,
                                const DependencyDag& dag);

    BufferHandle splits_{NullBuffer};
    BufferHandle positions_{NullBuffer};
    std::uint64_t splitsAddress_{0};
    std::uint64_t positionsAddress_{0};
    std::uint32_t splitCount_{0};

    // Emit static data (full build only); null/empty for a score-only build.
    BufferHandle indices_{NullBuffer};
    BufferHandle weld_{NullBuffer};
    BufferHandle removalParent_{NullBuffer};
    BufferHandle wedgeChoices_{NullBuffer};
    BufferHandle wedgeOffsets_{NullBuffer};
    BufferHandle finestFaces_{NullBuffer};   // canonical faces (B4 repair detector input)
    BufferHandle removingSplit_{NullBuffer}; // per canonical: its removing split (B4 mark target)

    // Refine/coarsen front topology (Stage B3), built in the base build; null/empty for a
    // zero-split forest. `rankRanges_`/`initialActive_` are CPU-side (copied into each front at
    // build).
    BufferHandle frontSplits_{NullBuffer};
    BufferHandle splitsByRank_{NullBuffer};
    BufferHandle rankRangesBuffer_{NullBuffer}; // device-local RankRange[] (persistent kernel ABI)
    std::vector<RankRange> rankRanges_;
    std::vector<std::uint32_t> initialActive_;

    VdpmGpuMeshBinding binding_{}; // fully assembled at build; the front copies it verbatim
};

// The five emit pipelines built once and reused across instances (Stage B2): the four emit passes +
// the shared exclusive-scan primitive. Descriptor-free; owns no per-instance state. Non-movable
// (the pipelines and scan hold pointers to `device`); construct one per device.
class VdpmEmitPipelines
{
public:
    explicit VdpmEmitPipelines(const Device& device)
        : ancestor_(device, vdpmAncestorPipelineConfig()),
          survival_(device, vdpmSurvivalPipelineConfig()),
          scatter_(device, vdpmScatterPipelineConfig()),
          finalize_(device, vdpmEmitFinalizePipelineConfig()),
          scan_(device)
    {
    }
    VdpmEmitPipelines(const VdpmEmitPipelines&) = delete;
    VdpmEmitPipelines& operator=(const VdpmEmitPipelines&) = delete;
    VdpmEmitPipelines(VdpmEmitPipelines&&) = delete;
    VdpmEmitPipelines& operator=(VdpmEmitPipelines&&) = delete;
    ~VdpmEmitPipelines() = default;

    [[nodiscard]] const ComputePipeline& ancestor() const noexcept
    {
        return ancestor_;
    }
    [[nodiscard]] const ComputePipeline& survival() const noexcept
    {
        return survival_;
    }
    [[nodiscard]] const ComputePipeline& scatter() const noexcept
    {
        return scatter_;
    }
    [[nodiscard]] const ComputePipeline& finalize() const noexcept
    {
        return finalize_;
    }
    [[nodiscard]] const VdpmScan& scan() const noexcept
    {
        return scan_;
    }

private:
    ComputePipeline ancestor_;
    ComputePipeline survival_;
    ComputePipeline scatter_;
    ComputePipeline finalize_;
    VdpmScan scan_;
};

// The four refine/coarsen pipelines built once and reused across instances (Stage B3). Descriptor-
// free; non-movable (the pipelines hold pointers to `device`); construct one per device.
class VdpmRefinePipelines
{
public:
    explicit VdpmRefinePipelines(const Device& device)
        : mark_(device, vdpmMarkPipelineConfig()),
          close_(device, vdpmClosePipelineConfig()),
          refine_(device, vdpmRefinePipelineConfig()),
          coarsen_(device, vdpmCoarsenPipelineConfig())
    {
    }
    VdpmRefinePipelines(const VdpmRefinePipelines&) = delete;
    VdpmRefinePipelines& operator=(const VdpmRefinePipelines&) = delete;
    VdpmRefinePipelines(VdpmRefinePipelines&&) = delete;
    VdpmRefinePipelines& operator=(VdpmRefinePipelines&&) = delete;
    ~VdpmRefinePipelines() = default;

    [[nodiscard]] const ComputePipeline& mark() const noexcept
    {
        return mark_;
    }
    [[nodiscard]] const ComputePipeline& close() const noexcept
    {
        return close_;
    }
    [[nodiscard]] const ComputePipeline& refine() const noexcept
    {
        return refine_;
    }
    [[nodiscard]] const ComputePipeline& coarsen() const noexcept
    {
        return coarsen_;
    }

private:
    ComputePipeline mark_;
    ComputePipeline close_;
    ComputePipeline refine_;
    ComputePipeline coarsen_;
};

// The VDPM GPU repair pipelines built once and reused across instances (Stage B4): the shared
// ancestor resolve + the per-face detector + the full-detail fallback. Descriptor-free; non-movable
// (the pipelines hold pointers to `device`); construct one per device.
class VdpmRepairPipelines
{
public:
    explicit VdpmRepairPipelines(const Device& device)
        : ancestor_(device, vdpmAncestorPipelineConfig()),
          detect_(device, vdpmRepairDetectPipelineConfig()),
          fallback_(device, vdpmRepairFallbackPipelineConfig())
    {
    }
    VdpmRepairPipelines(const VdpmRepairPipelines&) = delete;
    VdpmRepairPipelines& operator=(const VdpmRepairPipelines&) = delete;
    VdpmRepairPipelines(VdpmRepairPipelines&&) = delete;
    VdpmRepairPipelines& operator=(VdpmRepairPipelines&&) = delete;
    ~VdpmRepairPipelines() = default;

    [[nodiscard]] const ComputePipeline& ancestor() const noexcept
    {
        return ancestor_;
    }
    [[nodiscard]] const ComputePipeline& detect() const noexcept
    {
        return detect_;
    }
    [[nodiscard]] const ComputePipeline& fallback() const noexcept
    {
        return fallback_;
    }

private:
    ComputePipeline ancestor_;
    ComputePipeline detect_;
    ComputePipeline fallback_;
};

// The persistent repair kernel (perf arc, Stage 2): one workgroup per front runs the whole repair
// fixpoint in a single dispatch, replacing the ~1000-command recorder. Descriptor-free; non-movable
// (holds a device pointer). Capability-gated INDEPENDENTLY of the rest of the GPU front — the
// manager builds it only when `deviceSupported`, else it keeps the multi-dispatch recorder.
class VdpmRepairKernel
{
public:
    // The kernel's fixed workgroup size (local_size_x in vdpm_repair_kernel.comp).
    static constexpr std::uint32_t kLocalSize = 256;

    // Throws std::runtime_error if !deviceSupported. The check runs BEFORE the pipeline is
    // constructed (the public ctor delegates through requireSupported, whose result is evaluated
    // first), so a caller that forgot deviceSupported() fails loudly here — not at pipeline
    // creation.
    explicit VdpmRepairKernel(const Device& device);

    VdpmRepairKernel(const VdpmRepairKernel&) = delete;
    VdpmRepairKernel& operator=(const VdpmRepairKernel&) = delete;
    VdpmRepairKernel(VdpmRepairKernel&&) = delete;
    VdpmRepairKernel& operator=(VdpmRepairKernel&&) = delete;
    ~VdpmRepairKernel() = default;

    // True when the device can run the kernel: a workgroup of kLocalSize invocations (size[0] +
    // invocations) plus the shared-memory reduction word. The 1-D workgroup COUNT cap (relevant
    // once Stage 4 dispatches N fronts) is checked per-dispatch.
    [[nodiscard]] static bool deviceSupported(const Device& device);

    [[nodiscard]] const ComputePipeline& pipeline() const noexcept
    {
        return pipeline_;
    }

private:
    // Token proving deviceSupported ran before the pipeline member is constructed (the delegating
    // ctor evaluates requireSupported first).
    struct Checked
    {
    };
    [[nodiscard]] static Checked requireSupported(const Device& device);
    VdpmRepairKernel(const Device& device, Checked);

    ComputePipeline pipeline_;
};

// PER-INSTANCE GPU front state — the per-split score/backface output + the per-frame mapped params
// block, and (full build only) the emit workspace + resident emitted-index buffer. References its
// shared VdpmGpuMesh.
class VdpmGpuFront
{
public:
    // Score-only front (no emit workspace). Matches a score-only VdpmGpuMesh; `recordEmit` throws.
    [[nodiscard]] static VdpmGpuFront build(Resources& resources, const VdpmGpuMesh& mesh);

    // Full front: additionally allocates the emit workspace (ancestor id/depth, survival, scan
    // output + scratch, counters, the host-visible uploaded `active` buffer, and the resident
    // emitted-index buffer). Requires `mesh.binding().hasEmitData` (throws std::logic_error
    // otherwise). Single-buffered — the per-frame-in-flight ring is a Stage B5 concern.
    [[nodiscard]] static VdpmGpuFront buildWithEmit(Resources& resources, const VdpmGpuMesh& mesh);

    // Refine/coarsen front (Stage B3): the persistent front STATE
    // (active/refined/dependents/required — device-resident uint32), the uploaded per-split score
    // input, the invariant-failure flags, and the per-rank dispatch ranges copied from the mesh.
    // State is initialized to the coarsest front (roots active, nothing refined, dependents 0).
    // Does NOT allocate the score-output/params or the emit workspace (B3 uploads scores directly).
    // Copies the mesh's rank ranges (no pointer into it). When the mesh carries repair data (a full
    // build) it also allocates the B4 repair scratch; `withClassificationReadback` additionally
    // allocates the per-face classification diagnostic buffer that `recordDetectClassify` needs —
    // OFF by default so a PRODUCTION front spends no per-face memory on a test-only readback.
    [[nodiscard]] static VdpmGpuFront buildWithFront(Resources& resources, const VdpmGpuMesh& mesh,
                                                     bool withClassificationReadback = false);

    // The PRODUCTION runtime front (Stage B5): the full lifecycle — scoring + persistent
    // refine/coarsen state + repair scratch + emit workspace — in ONE front, with the draw-consumed
    // outputs (emitted indices, indirect command, counters) and host-written repair params RINGED
    // per frame-in-flight (persistent front per instance, transient output per frame slot).
    // Requires the full mesh (emit/repair data). `recordFrame` drives it; the renderer binds
    // `emittedIndicesBuffer(frameIndex)` + `emittedIndirectBuffer(frameIndex)` for the indirect
    // draw.
    [[nodiscard]] static VdpmGpuFront buildRuntime(Resources& resources, const VdpmGpuMesh& mesh);

    // Record ONE full GPU frame for a runtime front (Stage B5): score(view) → apply(scored) →
    // repair → emit into frame slot `frameIndex`, chaining GPU buffers with no host round-trip
    // (only the score view params + repair params are host-written, into their ring slots). Records
    // NO consumer barrier after emit — the caller (renderer) adds the compute→(index-read +
    // indirect-read) barrier before the draw. Throws std::logic_error on a non-runtime front.
    void recordFrame(vk::CommandBuffer cmd, const ComputePipeline& scorePipeline,
                     const VdpmRefinePipelines& refinePipelines,
                     const VdpmRepairPipelines& repairPipelines,
                     const VdpmEmitPipelines& emitPipelines, Resources& resources,
                     std::uint32_t frameIndex, const VdpmViewParams& scoreView,
                     const VdpmRepairParams& repairParams, float pixelBudget, float coarsenBudget,
                     std::uint32_t repairRoundBudget);

    // Record ONLY the score dispatch for frame `frameIndex` (writes that slot's mapped params from
    // `view`, pushes its address, dispatches ceil(splitCount / 64)). NO barriers — the consumer
    // owns synchronisation (the harness a compute→transfer→host readback; later stages
    // compute→compute). A zero-split front records nothing.
    void recordScore(vk::CommandBuffer cmd, const ComputePipeline& pipeline,
                     std::uint32_t frameIndex, const VdpmViewParams& view);

    // Record the full emit for a settled front (Stage B2): upload `active` (per canonical, 0/1) to
    // the host-visible workspace, then record ancestor → survival → scan → (scatter + finalize),
    // with a compute→compute barrier between each stage and after the scan before BOTH consumers.
    // Clears the counters buffer once (fillBuffer, an eClear→compute barrier) so ancestor's atomic,
    // the scan's total, and finalize's index count all start defined — no CPU readback.
    //
    // CONSUMER SYNC CONTRACT: records NO consumer barrier; the caller synchronises the reads. The
    // emitted-index buffer (`emittedIndicesBuffer`) is COMPUTE-written (the scatter), so a consumer
    // barrier from it uses srcStage eComputeShader. The counters buffer (`countersBuffer`) is
    // MIXED: it is compute-written by the ancestor pass (counters[0]), the scan (counters[1]), and
    // finalize (counters[2]) — the scatter does NOT touch it — but on a valid front counters[0],
    // and on a zero-face mesh counters[1], are only CLEAR-written by the fillBuffer (no compute
    // write reaches them). So a counters consumer barrier's source scope MUST include
    // eClear|eComputeShader / eTransferWrite|eShaderStorageWrite, not compute alone, or the
    // clear-written values race the read. (This bit the B2 readback harness in review.)
    //
    // `active.size()` must equal the mesh's canonical vertex count; the dispatches are validated
    // against the device's max 1-D group count. Throws std::logic_error on a score-only front.
    void recordEmit(vk::CommandBuffer cmd, const VdpmEmitPipelines& pipelines, Resources& resources,
                    std::span<const std::uint32_t> active);

    // Like recordEmit but reads the front's OWN live refine/coarsen state (`activeState`) instead
    // of a CPU-uploaded active span (Stage B5, the active-state→emit seam), and writes the emitted
    // indices / counters / indirect command into frame slot `frameIndex` of the transient RING (so
    // a draw can consume slot N while slot N+1 is computed). Requires a runtime front
    // (buildRuntime).
    void recordEmitFromFront(vk::CommandBuffer cmd, const VdpmEmitPipelines& pipelines,
                             Resources& resources, std::uint32_t frameIndex);

    // Record one refine/coarsen frame (Stage B3): upload `scores` (one VdpmScoreOut per split),
    // clear the failure flags, then mark → recordCloseAndRefineRequired → coarsen. Mirrors
    // `ParallelFront::applyView` exactly, as rank-ordered dispatches. The persistent front STATE is
    // mutated in place (score→refine/coarsen decisions matched integer-exact to the CPU model). The
    // shared recorder owns its own boundary barriers, so both this and the future B4 repair round
    // are safe by construction. Records NO consumer barrier after the final coarsen — the caller
    // synchronises the state read-back. `scores.size()` must equal splitCount; the per-rank
    // dispatches are validated against the device cap (defence-in-depth; `build` already rejected
    // an ineligible mesh). Throws std::logic_error on a non-front build. A zero-split front records
    // nothing.
    void recordApplyView(vk::CommandBuffer cmd, const VdpmRefinePipelines& pipelines,
                         Resources& resources, std::span<const VdpmScoreOut> scores,
                         float pixelBudget, float coarsenBudget);

    // Like recordApplyView but reads the front's OWN GPU score output (from a prior recordScore) as
    // the per-split scores — no host upload (Stage B5, the score→apply seam). The caller records
    // recordScore then a compute→compute barrier then this. Same barriers/contract as
    // recordApplyView.
    void recordApplyScoredView(vk::CommandBuffer cmd, const VdpmRefinePipelines& pipelines,
                               Resources& resources, float pixelBudget, float coarsenBudget);

    // Record the GPU foldover ∪ coverage repair to a fixpoint (Stage B4) — the snapshot analogue of
    // `ParallelFront::repairFront`, run AFTER recordApplyView has settled the front. Each round:
    // clear `required`, resolve ancestors (shared pass, live front), DETECT per finest face (mark
    // each violation's inactive-corner removing split), then recordCloseAndRefineRequired. After
    // `roundBudget` snapshot rounds it runs ONE final detect and a **full-detail fallback**: if a
    // violation still remains (no CPU readback — the fallback reads the GPU anyMarked flag), it
    // seeds every unrefined split and refines to full detail (always hole-free), recording
    // `repairControl[2]`. So normal fronts converge economically; a pathological convergence stays
    // correct. The repair control buffer `{anyMarked, ancestorFailure, fallbackFired, pad}` is
    // SEPARATE from B3's failFlags (clearing it here can't erase a B3 refine/underflow failure).
    //
    // SYNC: records a LEADING compute→(compute|clear) barrier (applyView's coarsen has no trailing
    // one) and, per round, orders the prior close's `required` write → the clear → the next detect
    // (compute→eClear→compute). Records NO consumer barrier after the final refine — the caller
    // synchronises the state read-back. Uploads `params`. Throws std::logic_error without repair
    // state (needs a full-mesh `buildWithFront`).
    void recordRepair(vk::CommandBuffer cmd, const VdpmRefinePipelines& refinePipelines,
                      const VdpmRepairPipelines& repairPipelines, Resources& resources,
                      const VdpmRepairParams& params, std::uint32_t roundBudget);

    // Pack this front's complete repair job (perf arc, Stage 2) for frame slot `frameIndex` under
    // `roundBudget` — the single assembly authority Stage 4 reuses to fill its N-job manager array.
    // Points `paramsAddress` at the frame slot's repair-params ring (the caller uploads the params
    // there first). Requires a runtime front with repair data (buildRuntime full build); throws
    // std::logic_error otherwise. `roundBudget` must be <= the allocated roundHistory capacity.
    [[nodiscard]] VdpmRepairJobGpu makeRepairJob(std::uint32_t frameIndex,
                                                 std::uint32_t roundBudget) const;

    // Record the persistent-kernel repair for frame slot `frameIndex`: upload `params`, pack +
    // upload the 1-element job, a leading apply/coarsen→repair compute barrier (recordFrame owns
    // the cross-frame lifecycle barrier), then ONE dispatch of ONE workgroup. Records NO consumer
    // barrier — the caller synchronises the state read-back. The whole repair fixpoint is this one
    // dispatch. Throws std::logic_error without a runtime repair front, or if roundBudget exceeds
    // the history capacity.
    void recordRepairKernel(vk::CommandBuffer cmd, const VdpmRepairKernel& kernel,
                            std::uint32_t frameIndex, const VdpmRepairParams& params,
                            std::uint32_t roundBudget);

    // TEST-ONLY (Stage B4): record a SINGLE ancestor + detect against the current settled front,
    // writing each finest face's packed classification (`kVdpmDetect*` bits) to
    // `repairClassificationBuffer` — so a harness can cross-check the GPU classifier against the
    // CPU `detail::` classifiers per face per BRANCH (not just the aggregate zero-violation
    // result). Does NOT close/refine (the front is unchanged); marks `required` as a side effect.
    // Clears repair control + required first (leading barrier). Throws without repair state.
    void recordDetectClassify(vk::CommandBuffer cmd, const VdpmRepairPipelines& repairPipelines,
                              Resources& resources, const VdpmRepairParams& params);

    // The device-local score output (one VdpmScoreOut per split), created with eTransferSrc so a
    // test can copy it back.
    [[nodiscard]] BufferHandle outputBuffer() const noexcept
    {
        return output_;
    }
    // The persistent refine/coarsen state buffers (uint32; Stage B3) — all eTransferSrc so the
    // harness can read them back and cross-check the CPU model + `validateFrontInvariants`.
    [[nodiscard]] BufferHandle activeStateBuffer() const noexcept
    {
        return activeState_;
    }
    [[nodiscard]] BufferHandle refinedStateBuffer() const noexcept
    {
        return refinedState_;
    }
    [[nodiscard]] BufferHandle dependentsStateBuffer() const noexcept
    {
        return dependentsState_;
    }
    // The 2-uint invariant-failure flags [refineFailure, dependentsUnderflow]; must read back as 0.
    [[nodiscard]] BufferHandle failFlagsBuffer() const noexcept
    {
        return failFlags_;
    }
    // The 4-uint repair-control buffer [anyMarked, ancestorFailure, fallbackFired, pad] (Stage B4);
    // eTransferSrc so the harness reads back the ancestor-failure + fallback-fired diagnostics.
    [[nodiscard]] BufferHandle repairControlBuffer() const noexcept
    {
        return repairControl_;
    }
    // Per-round convergence history (perf instrumentation): one uint per bounded repair round, each
    // atomic-OR'd with that round's detect `anyMarked` (via address redirection — the SAME atomic
    // the round already recorded, just aimed at its own slot; the accumulated bounded value in
    // repairControl[0] was unused, only the final detect drives the fallback). Cleared once per
    // recordFrame. Reads as a marked-then-clean prefix; a marked round after a clean one (view
    // fixed) would flag a repair/sync bug. Empty on isolated (non-runtime) fronts. eTransferSrc for
    // the delayed readback.
    [[nodiscard]] BufferHandle roundHistoryBuffer() const noexcept
    {
        return roundHistory_;
    }
    // Per-face packed classification from recordDetectClassify (test only); one uint per finest
    // face.
    [[nodiscard]] BufferHandle repairClassificationBuffer() const noexcept
    {
        return repairClassification_;
    }
    // The resident emitted-index buffer (3 * survivingFaces uint32; full build only) — storage +
    // BDA
    // + eIndexBuffer (so B5 binds it as the draw's index source) + eTransferSrc (harness readback).
    // The valid length is `counters[2]`.
    [[nodiscard]] BufferHandle emittedIndicesBuffer() const noexcept
    {
        return emittedIndices_;
    }
    // The 3-uint counters buffer [ancestorFailures, survivingFaces, emittedIndexCount];
    // eTransferSrc so the harness can read the failure + index counts back.
    [[nodiscard]] BufferHandle countersBuffer() const noexcept
    {
        return counters_;
    }
    // The GPU-written draw indirect command (laid out as graphics::DrawIndexedIndirectCommand) —
    // eIndirectBuffer (B5 draw) + eTransferSrc (harness readback). The finalize fills all 5 words.
    [[nodiscard]] BufferHandle emittedIndirectBuffer() const noexcept
    {
        return emittedIndirect_;
    }

    // Runtime-front (buildRuntime) ring-slot accessors: the draw-consumed outputs for frame slot
    // `frameIndex`. The renderer binds these for the indirect draw; the harness reads them back.
    [[nodiscard]] BufferHandle emittedIndicesBuffer(std::uint32_t frameIndex) const noexcept
    {
        return frameOutputs_[frameIndex].emittedIndices;
    }
    [[nodiscard]] BufferHandle emittedIndirectBuffer(std::uint32_t frameIndex) const noexcept
    {
        return frameOutputs_[frameIndex].indirect;
    }
    [[nodiscard]] BufferHandle countersBuffer(std::uint32_t frameIndex) const noexcept
    {
        return frameOutputs_[frameIndex].counters;
    }
    [[nodiscard]] std::uint32_t splitCount() const noexcept
    {
        return binding_.splitCount;
    }
    [[nodiscard]] std::uint32_t faceCount() const noexcept
    {
        return binding_.faceCount;
    }

    // Rank count R = maxRank + 1 = the number of per-rank dispatch waves the refine/coarsen/repair
    // passes fan into (0 for a zero-split front, which records nothing). The dominant driver of the
    // per-frame command count — see analyticComputeCost.
    [[nodiscard]] std::uint32_t rankCount() const noexcept
    {
        return static_cast<std::uint32_t>(rankRanges_.size());
    }

    // Analytic per-frame compute-dispatch + pipeline-barrier count for a front with `rankCount`
    // ranks under repair round budget `roundBudget` (perf instrumentation, B5b-perf). Mirrors the
    // recorder structure exactly: score (1 dispatch) + apply mark/close/refine/coarsen (3R+1
    // dispatches, 3R+2 barriers) + the bounded repair — B rounds of reset→ancestor→detect→close→
    // refine ((2R+2) dispatches, (2R+4) barriers each) plus a final detect + fallback + a full
    // close/refine ((2R+3) dispatches, (2R+7) barriers) — + emit (~7 dispatches, 4 barriers) + the
    // frame lifecycle barrier (1). So dispatches = B(2R+2) + 5R + 12, barriers = B(2R+4) + 5R + 16.
    // The ~94%-of-work repair term B(2R+2) is why an early-out on convergence is the target.
    // Returns {0,0} for a zero-split front. KEEP IN SYNC with recordFrame's recorders if they
    // change.
    struct ComputeCost
    {
        std::uint32_t dispatches{0};
        std::uint32_t barriers{0};
    };
    [[nodiscard]] static constexpr ComputeCost
    analyticComputeCost(std::uint32_t rankCount, std::uint32_t roundBudget) noexcept
    {
        if (rankCount == 0)
        {
            return {};
        }
        const std::uint32_t r = rankCount;
        const std::uint32_t b = roundBudget;
        return {.dispatches = b * (2 * r + 2) + 5 * r + 12,
                .barriers = b * (2 * r + 4) + 5 * r + 16};
    }

private:
    VdpmGpuMeshBinding binding_{};        // copied at build — no pointer into a movable mesh
    std::uint32_t maxWorkGroupCountX_{0}; // device cap on a 1-D dispatch's group count
    BufferHandle output_{NullBuffer};
    std::array<std::span<std::byte>, kMaxFramesInFlight> paramsMapped_{};
    std::array<std::uint64_t, kMaxFramesInFlight> paramsAddress_{};
    std::uint64_t outputAddress_{0};

    // Emit workspace (full build only; single-buffered for Stage B2). `active` is host-visible +
    // BDA (uploaded per emit); the rest are device-local + BDA. `counters` + `emittedIndices` carry
    // eTransferSrc for readback. `scanScratch` holds the exclusive scan's per-level partial sums
    // (empty when faceCount <= one block).
    bool hasEmit_{false};
    std::span<std::byte> activeMapped_{};
    std::uint64_t activeAddress_{0};
    BufferHandle ancestorId_{NullBuffer};
    BufferHandle ancestorDepth_{NullBuffer};
    BufferHandle survive_{NullBuffer};
    BufferHandle outSlot_{NullBuffer};
    BufferHandle counters_{NullBuffer};
    BufferHandle emittedIndices_{NullBuffer};
    BufferHandle emittedIndirect_{NullBuffer}; // GPU-written draw indirect command (B5)
    std::uint64_t emittedIndirectAddress_{0};
    std::uint64_t ancestorIdAddress_{0};
    std::uint64_t ancestorDepthAddress_{0};
    std::uint64_t surviveAddress_{0};
    std::uint64_t outSlotAddress_{0};
    std::uint64_t countersAddress_{0};
    std::uint64_t emittedIndicesAddress_{0};
    std::vector<BufferHandle> scanScratch_;
    std::vector<std::uint64_t> scanScratchAddress_;

    // Refine/coarsen front (Stage B3). Persistent device-resident state (uint32, readback-enabled);
    // `scores` is host-visible + BDA (uploaded per applyView); `failFlags` is device-local +
    // readback
    // + transfer-dst (cleared each frame). `rankRanges_` is copied from the mesh.
    bool hasFront_{false};
    BufferHandle activeState_{NullBuffer};
    BufferHandle refinedState_{NullBuffer};
    BufferHandle dependentsState_{NullBuffer};
    BufferHandle requiredState_{NullBuffer};
    BufferHandle failFlags_{NullBuffer};
    std::uint64_t activeStateAddress_{0};
    std::uint64_t refinedStateAddress_{0};
    std::uint64_t dependentsStateAddress_{0};
    std::uint64_t requiredStateAddress_{0};
    std::uint64_t failFlagsAddress_{0};
    std::span<std::byte> scoresMapped_{};
    std::uint64_t scoresAddress_{0};
    std::vector<RankRange> rankRanges_;

    // Repair fixpoint (Stage B4; allocated by buildWithFront only when the mesh carries the
    // emit/repair data — finest faces + removingSplit). `ancestorId/Depth` are the per-round
    // ancestor cache; the host-visible `repairParams` is uploaded per repair; `repairControl` is
    // device-local + readback + transfer-dst, SEPARATE from failFlags_.
    bool hasRepair_{false};
    BufferHandle repairAncestorId_{NullBuffer};
    BufferHandle repairAncestorDepth_{NullBuffer};
    BufferHandle repairControl_{NullBuffer};
    BufferHandle repairClassification_{NullBuffer}; // per finest face, test-only readback
    std::uint64_t repairAncestorIdAddress_{0};
    std::uint64_t repairAncestorDepthAddress_{0};
    std::uint64_t repairControlAddress_{0};
    std::uint64_t repairClassificationAddress_{0};
    std::span<std::byte> repairParamsMapped_{};
    std::uint64_t repairParamsAddress_{0};
    // Per-round convergence history (perf instrumentation, runtime front only).
    // kVdpmGpuRepairRounds slots; NullBuffer/0 on isolated fronts, which fall back to writing
    // anyMarked into repairControl (no per-round capture).
    BufferHandle roundHistory_{NullBuffer};
    std::uint64_t roundHistoryAddress_{0};

    // The shared close+refine recorder (Stage B3), reused by B4's repair round. Owns its boundary
    // barriers: a leading seed-write→closure-read barrier, barriers between close ranks and before
    // refine, barriers between refine ranks, and a trailing refine-write→consumer-read barrier — so
    // every caller (mark→…→coarsen, or detect→…→next-detect) is synchronised by construction.
    void recordCloseAndRefineRequired(vk::CommandBuffer cmd, const VdpmRefinePipelines& pipelines);

    // The shared applyView body (mark → close+refine → coarsen) reading `scoresAddress` — the
    // common recorder both recordApplyView (host-uploaded scores) and recordApplyScoredView (the
    // GPU score output) delegate to, so a raw device address is never exposed publicly.
    void recordApplyViewImpl(vk::CommandBuffer cmd, const VdpmRefinePipelines& pipelines,
                             Resources& resources, std::uint64_t scoresAddress, float pixelBudget,
                             float coarsenBudget);

    // One frame slot's transient, draw-consumed emit outputs (ringed in a runtime front).
    struct FrameOutput
    {
        BufferHandle emittedIndices{NullBuffer};
        BufferHandle counters{NullBuffer};
        BufferHandle indirect{NullBuffer};
        std::uint64_t emittedIndicesAddress{0};
        std::uint64_t countersAddress{0};
        std::uint64_t indirectAddress{0};
    };
    // The shared emit body reading `activeAddress` and writing `out` — the common recorder both
    // recordEmit (CPU-uploaded active, single output) and recordEmitFromFront (live front state,
    // ring slot) delegate to.
    void recordEmitImpl(vk::CommandBuffer cmd, const VdpmEmitPipelines& pipelines,
                        Resources& resources, std::uint64_t activeAddress, const FrameOutput& out);

    // The shared repair body reading the repair params at `paramsAddress` — recordRepair (single
    // params buffer) and recordFrame (ring slot) delegate to it.
    void recordRepairImpl(vk::CommandBuffer cmd, const VdpmRefinePipelines& refinePipelines,
                          const VdpmRepairPipelines& repairPipelines, Resources& resources,
                          std::uint64_t paramsAddress, std::uint32_t roundBudget);

    // Runtime front (Stage B5): the draw-consumed outputs ringed per frame-in-flight + the
    // host-written repair params ringed (persistent front, transient output). `hasRuntime_` implies
    // hasFront_ + hasEmit_ + hasRepair_.
    bool hasRuntime_{false};
    std::array<FrameOutput, kMaxFramesInFlight> frameOutputs_{};
    std::array<std::span<std::byte>, kMaxFramesInFlight> repairParamsRing_{};
    std::array<std::uint64_t, kMaxFramesInFlight> repairParamsRingAddress_{};
    // Persistent-kernel job ring (perf arc, Stage 2): one host-visible VdpmRepairJobGpu per frame
    // slot, packed + uploaded per repair and read by the kernel via its device address.
    std::array<std::span<std::byte>, kMaxFramesInFlight> jobRing_{};
    std::array<std::uint64_t, kMaxFramesInFlight> jobRingAddress_{};
};

} // namespace fire_engine
