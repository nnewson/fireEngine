#pragma once

#include <array>
#include <optional>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/frame_info.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/gpu_limits.hpp>
#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

class Geometry;
class Material;
class Resources;
class Skin;

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
    void load(Resources& resources);
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

    void morphWeights(const std::vector<float>& weights) noexcept
    {
        morphWeights_ = weights;
    }

    [[nodiscard]]
    std::vector<DrawCommand> render(const FrameInfo& frame, const Mat4& world,
                                    const Mat4& previousWorld);

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

        std::array<MappedMemory, kMaxFramesInFlight> skinMapped{};
        std::array<MappedMemory, kMaxFramesInFlight> morphUboMapped{};
        std::array<MappedMemory, kMaxFramesInFlight> shadowMapped{};
        // Forward set-0 buffer handles (pushed inline per draw — no descriptor
        // set). Frame UBO is object-wide (Object::uniformBufs_); these are
        // per-geometry.
        std::array<BufferHandle, kMaxFramesInFlight> skinBufs{NullBuffer, NullBuffer};
        std::array<BufferHandle, kMaxFramesInFlight> morphUboBufs{NullBuffer, NullBuffer};
        BufferHandle morphSsbo{NullBuffer};
        // Per-object ShadowUBO buffer handles (shadow set-0 binding 0, pushed
        // inline per draw — no descriptor set). skin/morph/morphSsbo above are
        // reused for the shadow draw.
        std::array<BufferHandle, kMaxFramesInFlight> shadowBufs{NullBuffer, NullBuffer};
    };

    [[nodiscard]] Bounds3 computeShadowBounds(const std::vector<Mat4>& jointMatrices, bool hasSkin,
                                              const Mat4& world) const noexcept;

    // load() phases: createForwardBindings allocates the per-geometry vertex-stage
    // buffers; createShadowBindings allocates the per-object ShadowUBO buffers.
    // Both forward and shadow set 0 are pushed inline per draw (no descriptor
    // sets); the shadow draw reuses the forward skin/morph/morphSsbo buffers.
    void createForwardBindings(Resources& resources);
    void createShadowBindings(Resources& resources);

    // render() phases: write the per-frame UBOs (shared/skin/material/morph),
    // write the shadow UBO, then assemble forward + shadow draw commands.
    void writeForwardUniforms(const FrameInfo& frame, const Mat4& world, const Mat4& previousWorld,
                              bool hasSkin, const std::vector<Mat4>& jointMatrices);
    void writeShadowUniforms(const FrameInfo& frame, const Mat4& world, bool hasSkin);
    [[nodiscard]] std::vector<DrawCommand> buildDrawCommands(const FrameInfo& frame,
                                                             const Mat4& world, bool hasSkin,
                                                             const Bounds3& shadowBounds) const;

    Skin* skin_{nullptr};
    std::vector<float> morphWeights_;
    Resources* resources_{nullptr};
    uint32_t objectId_{0};

    std::array<MappedMemory, kMaxFramesInFlight> uniformMapped_{};
    // Shared frame UBO buffer handles (pushed as forward set-0 binding 0 per draw).
    std::array<BufferHandle, kMaxFramesInFlight> uniformBufs_{NullBuffer, NullBuffer};

    std::vector<GeometryBindings> bindings_;
    // Lazily computed local-space AABB over the geometry vertices (see localBounds()).
    mutable std::optional<Bounds3> localBounds_;
};

} // namespace fire_engine
