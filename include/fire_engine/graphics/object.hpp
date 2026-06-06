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
struct GeometryDescriptorInfo;
struct ObjectDescriptorRequest;

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
    std::vector<DrawCommand> render(const FrameInfo& frame, const Mat4& world);

private:
    struct GeometryBindings
    {
        const Geometry* geometry{nullptr};
        const Geometry* shadowGeometry{nullptr};
        const Material* defaultMaterial{nullptr};
        const Material* activeMaterial{nullptr};
        std::vector<const Material*> variantMaterials;
        std::array<bool, kMaxFramesInFlight> descriptorDirty{false, false};

        std::array<MappedMemory, kMaxFramesInFlight> materialMapped{};
        std::array<MappedMemory, kMaxFramesInFlight> skinMapped{};
        std::array<MappedMemory, kMaxFramesInFlight> morphUboMapped{};
        std::array<MappedMemory, kMaxFramesInFlight> shadowMapped{};
        std::array<DescriptorSetHandle, kMaxFramesInFlight> descSets{NullDescriptorSet,
                                                                     NullDescriptorSet};
        std::array<DescriptorSetHandle, kMaxFramesInFlight> shadowDescSets{NullDescriptorSet,
                                                                           NullDescriptorSet};
    };

    static void applyMaterialTextures(GeometryDescriptorInfo& geoInfo, const Material& mat,
                                      Resources& resources);
    [[nodiscard]] Bounds3 computeShadowBounds(const std::vector<Mat4>& jointMatrices, bool hasSkin,
                                              const Mat4& world) const noexcept;

    // load() phases: forward (set 0) descriptors return the per-geometry buffer
    // handles in the request; shadow descriptors reuse those buffers.
    [[nodiscard]] ObjectDescriptorRequest createForwardBindings(Resources& resources);
    void createShadowBindings(Resources& resources, const ObjectDescriptorRequest& req);

    // render() phases: write the per-frame UBOs (shared/skin/material/morph),
    // write the shadow UBO, then assemble forward + shadow draw commands.
    void writeForwardUniforms(const FrameInfo& frame, const Mat4& world, bool hasSkin,
                              const std::vector<Mat4>& jointMatrices);
    void writeShadowUniforms(const FrameInfo& frame, const Mat4& world, bool hasSkin);
    [[nodiscard]] std::vector<DrawCommand> buildDrawCommands(const FrameInfo& frame,
                                                             const Mat4& world, bool hasSkin,
                                                             const Bounds3& shadowBounds) const;

    Skin* skin_{nullptr};
    std::vector<float> morphWeights_;
    Resources* resources_{nullptr};
    uint32_t objectId_{0};

    std::array<MappedMemory, kMaxFramesInFlight> uniformMapped_{};

    std::vector<GeometryBindings> bindings_;
};

} // namespace fire_engine
