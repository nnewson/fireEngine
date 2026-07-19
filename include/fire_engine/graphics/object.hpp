#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/graphics/renderable_scene.hpp>
#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

class Geometry;
class Material;
class Resources;
class Skin;
class VdpmGpuRegistry;

class Object
{
public:
    Object() = default;
    ~Object() = default;

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&) noexcept = default;
    Object& operator=(Object&&) noexcept = default;

    void addGeometry(const Geometry& geometry);
    void shadowGeometry(std::size_t geometryIndex, const Geometry* geometry) noexcept;
    void addVariantMaterial(std::size_t geometryIndex, std::size_t variantIndex,
                            const Material* material);
    // `registry`, when non-null, creates a per-instance GPU-driven VDPM front over each geometry's
    // registered mesh (Stage B5b). Null (CPU backend / unsupported device) or a geometry with no
    // GPU mesh handle leaves the binding's vdpmGpuFront == NullVdpmFront and the instance CPU-only.
    // No default — every caller states its registration choice (pass Renderer::vdpmRegistry()).
    void load(Resources& resources, VdpmGpuRegistry* registry);
    void activeVariant(std::optional<std::size_t> variantIndex);
    [[nodiscard]] bool hasVariant(std::size_t variantIndex) const noexcept;
    [[nodiscard]] bool wouldChangeVariant(std::optional<std::size_t> variantIndex) const noexcept;

    void skin(Skin* s) noexcept
    {
        skin_ = s;
    }
    [[nodiscard]] const Skin* skin() const noexcept
    {
        return skin_;
    }

    void updateSkin();

    void morphWeights(std::span<const float> weights)
    {
        morphWeights_.assign(weights.begin(), weights.end());
    }

    [[nodiscard]]
    std::vector<DrawCommand> render(const FrameInfo& frame, const Mat4& world,
                                    const Mat4& previousWorld);

    // Add this frame's VDPM repair work (vertices each pass pulled back in) across this object's
    // active fronts to the running totals. Valid after render() this frame; a diagnostic surfaced
    // in the overlay so a repair-count regression is visible.
    void addVdpmRepairCounts(uint32_t& foldovers, uint32_t& coverage) const;
    void addVdpmChannelStats(VdpmChannelStats& out) const;

    // Local-space (bind-pose) AABB over the geometry vertices, cached on first use.
    // Used for frustum culling: a rigid object's world bound is this transformed by its
    // node's world matrix (exact). Deformable objects are not culled by it (see below).
    [[nodiscard]] const Bounds3& localBounds() const noexcept;

    // True when the rendered geometry deforms beyond its bind pose (skin or morph) — its
    // local bound under-covers the deformed mesh, so the coarse scene cull never skips it
    // (it is always drawn, then the precise per-draw cull uses its exact world bounds).
    [[nodiscard]] bool deformable() const noexcept
    {
        return skin_ != nullptr || !morphWeights_.empty();
    }

private:
    struct GeometryBindings
    {
        const Geometry* geometry{nullptr};
        const Geometry* shadowGeometry{nullptr};
        const Material* defaultMaterial{nullptr};
        const Material* activeMaterial{nullptr};
        std::vector<const Material*> variantMaterials;

        std::array<std::span<std::byte>, kMaxFramesInFlight> skinMapped{};
        std::array<std::span<std::byte>, kMaxFramesInFlight> morphUboMapped{};
        std::array<std::span<std::byte>, kMaxFramesInFlight> shadowMapped{};
        // Forward set-0 buffer handles (pushed inline per draw — no descriptor
        // set). Frame UBO is object-wide (Object::uniformBufs_); these are
        // per-geometry.
        std::array<BufferHandle, kMaxFramesInFlight> skinBufs{NullBuffer, NullBuffer};
        // Previous-frame joint matrices for skinned-mesh TAA motion vectors (forward set-0
        // PrevSkin).
        std::array<std::span<std::byte>, kMaxFramesInFlight> prevSkinMapped{};
        std::array<BufferHandle, kMaxFramesInFlight> prevSkinBufs{NullBuffer, NullBuffer};
        std::array<BufferHandle, kMaxFramesInFlight> morphUboBufs{NullBuffer, NullBuffer};
        BufferHandle morphSsbo{NullBuffer};
        // VIPM geomorph buffer (Continuous LOD): the geometry's per-vertex morph data, or a dummy.
        BufferHandle vipmBuffer{NullBuffer};
        // VDPM (View-dependent LOD): the per-instance active front + its per-frame dynamic index
        // buffers, rebuilt each frame from the camera when LodMode::ViewDependent. The front is
        // absent for non-VDPM meshes (small/deformable/no collapse stream). vdpmIndexCount is the
        // current frame's emitted index count.
        std::optional<ActiveFront> vdpmFront;
        // GPU-driven VDPM front handle for this instance (Stage B5b), created at load when a
        // registry is supplied and the geometry has a GPU mesh. In B5b-1 it runs alongside the CPU
        // vdpmFront (the compute is a shadow run; the CPU output above is still drawn).
        // NullVdpmFront ⇒ CPU only.
        VdpmFrontHandle vdpmGpuFront{NullVdpmFront};
        std::array<BufferHandle, kMaxFramesInFlight> vdpmIndexBufs{NullBuffer, NullBuffer};
        std::array<std::span<std::byte>, kMaxFramesInFlight> vdpmIndexMapped{};
        uint32_t vdpmIndexCount{0};
        // Per-frame host-visible indirect-command buffers (one DrawIndexedIndirectCommand each),
        // CPU-written from vdpmIndexCount so the VDPM draw records drawIndexedIndirect (Stage A of
        // the GPU-driven front; a compute shader writes them in Stage B5).
        std::array<BufferHandle, kMaxFramesInFlight> vdpmIndirectBufs{NullBuffer, NullBuffer};
        std::array<std::span<std::byte>, kMaxFramesInFlight> vdpmIndirectMapped{};
        // Reused emit buffer so the per-frame VDPM emission allocates nothing steady-state.
        std::vector<uint32_t> vdpmEmitScratch;
        // Per-object ShadowUBO buffer handles (shadow set-0 binding 0, pushed
        // inline per draw — no descriptor set). skin/morph/morphSsbo above are
        // reused for the shadow draw.
        std::array<BufferHandle, kMaxFramesInFlight> shadowBufs{NullBuffer, NullBuffer};
    };

    [[nodiscard]] Bounds3 computeShadowBounds(std::span<const Mat4> jointMatrices, bool hasSkin,
                                              const Mat4& world) const noexcept;

    // load() phases: createForwardBindings allocates the per-geometry vertex-stage
    // buffers; createShadowBindings allocates the per-object ShadowUBO buffers.
    // Both forward and shadow set 0 are pushed inline per draw (no descriptor
    // sets); the shadow draw reuses the forward skin/morph/morphSsbo buffers.
    void createForwardBindings(Resources& resources, VdpmGpuRegistry* registry);
    void createShadowBindings(Resources& resources);

    // render() phases: write the per-frame UBOs (shared/skin/material/morph),
    // write the shadow UBO, then assemble forward + shadow draw commands.
    void writeForwardUniforms(const FrameInfo& frame, const Mat4& world, const Mat4& previousWorld,
                              bool hasSkin, std::span<const Mat4> jointMatrices);
    void writeShadowUniforms(const FrameInfo& frame, const Mat4& world, bool hasSkin);
    [[nodiscard]] std::vector<DrawCommand> buildDrawCommands(const FrameInfo& frame,
                                                             const Mat4& world, bool hasSkin,
                                                             const Bounds3& shadowBounds) const;

    Skin* skin_{nullptr};
    std::vector<float> morphWeights_;
    Resources* resources_{nullptr};
    uint32_t objectId_{0};

    std::array<std::span<std::byte>, kMaxFramesInFlight> uniformMapped_{};
    // Per-object UBO buffer handles (pushed as forward set-0 binding 0 per draw). Holds model /
    // hasSkin / previousModel only — the camera lives in the per-frame CameraUBO (set-0 binding
    // 29).
    std::array<BufferHandle, kMaxFramesInFlight> uniformBufs_{NullBuffer, NullBuffer};
    // Last values written into each frame slot's ObjectUBO, so a static object (unchanged world +
    // previous world) skips the mapped re-upload. Default-zero worlds never match a real (affine)
    // transform, so the first frame per slot always writes. Mat4 == is exact/bitwise — intended
    // here.
    std::array<Mat4, kMaxFramesInFlight> lastWorld_{};
    std::array<Mat4, kMaxFramesInFlight> lastPreviousWorld_{};
    std::array<int, kMaxFramesInFlight> lastHasSkin_{};
    // Last frame's joint matrices, so a skinned mesh's TAA motion vector reprojects each vertex
    // from where it actually was (per-vertex deformation velocity), not just camera motion. Empty
    // until the first skinned frame — then previous == current (zero deformation velocity on frame
    // one).
    std::vector<Mat4> previousJointMatrices_;

    std::vector<GeometryBindings> bindings_;
    // Lazily computed local-space AABB over the geometry vertices (see localBounds()).
    mutable std::optional<Bounds3> localBounds_;
};

} // namespace fire_engine
