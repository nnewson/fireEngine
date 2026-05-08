#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace fire_engine
{

enum class ForwardBinding : std::uint32_t
{
    Frame = 0,
    Material = 1,
    BaseColourTexture = 2,
    Skin = 3,
    Morph = 4,
    MorphTargets = 5,
    EmissiveTexture = 6,
    NormalTexture = 7,
    MetallicRoughnessTexture = 8,
    OcclusionTexture = 9,
    // Shadow images are bound as plain sampled images (no sampler attached) so
    // a single comparison sampler can be reused across CSM, spot and point
    // shadow maps. Combining each into its own CombinedImageSampler would blow
    // past Apple's 16-samplers-per-stage limit.
    ShadowMap = 10,
    Light = 11,
    IrradianceMap = 12,
    PrefilteredMap = 13,
    BrdfLut = 14,
    ShadowCompareSampler = 15,
    TransmissionTexture = 16,
    ClearcoatTexture = 17,
    ClearcoatRoughnessTexture = 18,
    ClearcoatNormalTexture = 19,
    SceneColour = 20,
    ThicknessTexture = 21,
    SpotShadowMap = 22,
    PointShadowMap = 23,
    ShadowDebugSampler = 24,
    ShadowDebugImage = 25,
    WorldShadowMap = 26,
    SelfShadowMap = 27,
};

enum class ShadowBinding : std::uint32_t
{
    Shadow = 0,
    Skin = 1,
    Morph = 2,
    MorphTargets = 3,
    SelfShadowFirstMap = 4,
    SelfShadowDepthSampler = 5,
};

enum class SkyboxBinding : std::uint32_t
{
    Skybox = 0,
    Cubemap = 1,
    Light = 2,
};

enum class PostProcessBinding : std::uint32_t
{
    HdrInput = 0,
    BloomInput = 1,
};

enum class MaterialTextureSlot : std::size_t
{
    BaseColour,
    Emissive,
    Normal,
    MetallicRoughness,
    Occlusion,
    Transmission,
    Clearcoat,
    ClearcoatRoughness,
    ClearcoatNormal,
    Thickness,
};

inline constexpr std::size_t materialTextureSlotCount{10};

struct MaterialTextureBinding
{
    MaterialTextureSlot slot;
    ForwardBinding binding;
};

inline constexpr std::array<MaterialTextureBinding, materialTextureSlotCount>
    materialTextureBindings{{
        {MaterialTextureSlot::BaseColour, ForwardBinding::BaseColourTexture},
        {MaterialTextureSlot::Emissive, ForwardBinding::EmissiveTexture},
        {MaterialTextureSlot::Normal, ForwardBinding::NormalTexture},
        {MaterialTextureSlot::MetallicRoughness, ForwardBinding::MetallicRoughnessTexture},
        {MaterialTextureSlot::Occlusion, ForwardBinding::OcclusionTexture},
        {MaterialTextureSlot::Transmission, ForwardBinding::TransmissionTexture},
        {MaterialTextureSlot::Clearcoat, ForwardBinding::ClearcoatTexture},
        {MaterialTextureSlot::ClearcoatRoughness, ForwardBinding::ClearcoatRoughnessTexture},
        {MaterialTextureSlot::ClearcoatNormal, ForwardBinding::ClearcoatNormalTexture},
        {MaterialTextureSlot::Thickness, ForwardBinding::ThicknessTexture},
    }};

[[nodiscard]]
constexpr std::size_t slotIndex(MaterialTextureSlot slot) noexcept
{
    return static_cast<std::size_t>(slot);
}

[[nodiscard]]
constexpr std::uint32_t bindingIndex(ForwardBinding binding) noexcept
{
    return static_cast<std::uint32_t>(binding);
}

[[nodiscard]]
constexpr std::uint32_t bindingIndex(ShadowBinding binding) noexcept
{
    return static_cast<std::uint32_t>(binding);
}

[[nodiscard]]
constexpr std::uint32_t bindingIndex(SkyboxBinding binding) noexcept
{
    return static_cast<std::uint32_t>(binding);
}

[[nodiscard]]
constexpr std::uint32_t bindingIndex(PostProcessBinding binding) noexcept
{
    return static_cast<std::uint32_t>(binding);
}

} // namespace fire_engine
