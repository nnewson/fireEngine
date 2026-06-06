#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <fire_engine/graphics/colour3.hpp>

namespace fire_engine
{

class Texture;

enum class AlphaMode : uint8_t
{
    Opaque,
    Mask,
    Blend,
};

// The texture slots a material can carry, in the canonical order the render
// layer packs them into MaterialUBO::uv[] and the per-object descriptor set.
// Lives here (graphics) so material code can size slot arrays without reaching
// into render/; the slot-to-descriptor-binding mapping stays render-side.
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

[[nodiscard]]
constexpr std::size_t slotIndex(MaterialTextureSlot slot) noexcept
{
    return static_cast<std::size_t>(slot);
}

// KHR_texture_transform per-slot UV transform. Default = identity (offset 0,
// scale 1, rotation 0) means "no transform" — matches the spec's behaviour
// when the extension is absent on a TextureInfo.
struct UvTransform
{
    float offsetX{0.0f};
    float offsetY{0.0f};
    float scaleX{1.0f};
    float scaleY{1.0f};
    float rotation{0.0f};
};

// One material texture binding: the texture (null when the slot is unused), the
// glTF UV set it samples (TEXCOORD_0 / TEXCOORD_1), and its KHR_texture_transform.
// Replaces the old per-slot texture/texCoord/UvTransform getter triples.
struct TextureSlot
{
    const Texture* texture{nullptr};
    int texCoord{0};
    UvTransform transform{};

    [[nodiscard]] bool has() const noexcept
    {
        return texture != nullptr;
    }
};

// KHR_materials_clearcoat scalars. Present (engaged optional) only when the
// asset authored the extension. The three clearcoat textures live in the
// material texture-slot array (Clearcoat / ClearcoatRoughness / ClearcoatNormal).
struct ClearcoatParams
{
    float factor{0.0f};
    float roughness{0.0f};
    float normalScale{1.0f};
};

// KHR_materials_transmission + KHR_materials_ior. ior defaults to 1.5 (common
// dielectric). The transmission texture lives in the Transmission slot. Named
// *Params to avoid clashing with the render-layer Transmission pass class.
struct TransmissionParams
{
    float factor{0.0f};
    float ior{1.5f};
};

// KHR_materials_volume. attenuationDistance defaults to +infinity (no Beer-
// Lambert tinting). The thickness texture lives in the Thickness slot.
struct VolumeParams
{
    float thicknessFactor{0.0f};
    Colour3 attenuationColor{1.0f, 1.0f, 1.0f};
    float attenuationDistance{std::numeric_limits<float>::infinity()};
};

class Material
{
public:
    Material() = default;
    ~Material() = default;

    Material(const Material&) = default;
    Material& operator=(const Material&) = default;
    Material(Material&&) noexcept = default;
    Material& operator=(Material&&) noexcept = default;

    [[nodiscard]] const std::string& name() const noexcept
    {
        return name_;
    }
    void name(const std::string& name)
    {
        name_ = name;
    }

    // --- Texture slots --------------------------------------------------
    // One accessor for every material texture: the bound Texture, its UV-set
    // index, and its KHR_texture_transform. Index with MaterialTextureSlot.
    [[nodiscard]] const TextureSlot& texture(MaterialTextureSlot slot) const noexcept
    {
        return textures_[slotIndex(slot)];
    }
    [[nodiscard]] TextureSlot& texture(MaterialTextureSlot slot) noexcept
    {
        return textures_[slotIndex(slot)];
    }

    // --- Core PBR -------------------------------------------------------
    [[nodiscard]] Colour3 baseColor() const noexcept
    {
        return baseColor_;
    }
    void baseColor(Colour3 c) noexcept
    {
        baseColor_ = c;
    }

    [[nodiscard]] Colour3 emissive() const noexcept
    {
        return emissive_;
    }
    void emissive(Colour3 c) noexcept
    {
        emissive_ = c;
    }

    [[nodiscard]] float roughness() const noexcept
    {
        return roughness_;
    }
    void roughness(float v) noexcept
    {
        roughness_ = v;
    }

    [[nodiscard]] float metallic() const noexcept
    {
        return metallic_;
    }
    void metallic(float v) noexcept
    {
        metallic_ = v;
    }

    [[nodiscard]] float alpha() const noexcept
    {
        return alpha_;
    }
    void alpha(float v) noexcept
    {
        alpha_ = v;
    }

    [[nodiscard]] float normalScale() const noexcept
    {
        return normalScale_;
    }
    void normalScale(float v) noexcept
    {
        normalScale_ = v;
    }

    // glTF: shader output = lerp(colour, colour * sampledAO, strength).
    // Spec default is 1.0 (full occlusion contribution).
    [[nodiscard]] float occlusionStrength() const noexcept
    {
        return occlusionStrength_;
    }
    void occlusionStrength(float v) noexcept
    {
        occlusionStrength_ = v;
    }

    [[nodiscard]] AlphaMode alphaMode() const noexcept
    {
        return alphaMode_;
    }
    void alphaMode(AlphaMode m) noexcept
    {
        alphaMode_ = m;
    }

    [[nodiscard]] float alphaCutoff() const noexcept
    {
        return alphaCutoff_;
    }
    void alphaCutoff(float v) noexcept
    {
        alphaCutoff_ = v;
    }

    [[nodiscard]] bool doubleSided() const noexcept
    {
        return doubleSided_;
    }
    void doubleSided(bool v) noexcept
    {
        doubleSided_ = v;
    }

    // KHR_materials_unlit. When true, the renderer skips BRDF/IBL/shadow and
    // outputs the (textured) base colour directly. Used for skybox geometry,
    // foliage cards, UI quads, decals — anything authored without lighting.
    [[nodiscard]] bool unlit() const noexcept
    {
        return unlit_;
    }
    void unlit(bool v) noexcept
    {
        unlit_ = v;
    }

    // --- Optional extension blocks --------------------------------------
    // Engaged only when the asset authored the matching glTF extension; absent
    // == default behaviour. Consumers building the UBO can use value_or({}).
    [[nodiscard]] const std::optional<ClearcoatParams>& clearcoat() const noexcept
    {
        return clearcoat_;
    }
    void clearcoat(ClearcoatParams c) noexcept
    {
        clearcoat_ = c;
    }

    [[nodiscard]] const std::optional<TransmissionParams>& transmission() const noexcept
    {
        return transmission_;
    }
    void transmission(TransmissionParams t) noexcept
    {
        transmission_ = t;
    }

    [[nodiscard]] const std::optional<VolumeParams>& volume() const noexcept
    {
        return volume_;
    }
    void volume(VolumeParams v) noexcept
    {
        volume_ = v;
    }

private:
    std::string name_;
    Colour3 baseColor_{};
    Colour3 emissive_{};
    float roughness_{0.0f};
    float metallic_{0.0f};
    float alpha_{1.0f};
    float normalScale_{1.0f};
    float occlusionStrength_{1.0f};
    AlphaMode alphaMode_{AlphaMode::Opaque};
    float alphaCutoff_{0.5f};
    bool doubleSided_{false};
    bool unlit_{false};

    std::array<TextureSlot, materialTextureSlotCount> textures_{};

    std::optional<ClearcoatParams> clearcoat_;
    std::optional<TransmissionParams> transmission_;
    std::optional<VolumeParams> volume_;
};

} // namespace fire_engine
