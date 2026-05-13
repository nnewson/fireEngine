#pragma once

#include <cstdint>
#include <limits>
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

class Material
{
public:
    Material() = default;
    ~Material() = default;

    Material(const Material&) = default;
    Material& operator=(const Material&) = default;
    Material(Material&&) noexcept = default;
    Material& operator=(Material&&) noexcept = default;

    [[nodiscard]] const Texture& texture() const noexcept
    {
        return *texture_;
    }
    void texture(const Texture* t) noexcept
    {
        texture_ = t;
    }
    [[nodiscard]] bool hasTexture() const noexcept
    {
        return texture_ != nullptr;
    }

    [[nodiscard]] const Texture& emissiveTexture() const noexcept
    {
        return *emissiveTexture_;
    }
    void emissiveTexture(const Texture* t) noexcept
    {
        emissiveTexture_ = t;
    }
    [[nodiscard]] bool hasEmissiveTexture() const noexcept
    {
        return emissiveTexture_ != nullptr;
    }

    [[nodiscard]] const Texture& normalTexture() const noexcept
    {
        return *normalTexture_;
    }
    void normalTexture(const Texture* t) noexcept
    {
        normalTexture_ = t;
    }
    [[nodiscard]] bool hasNormalTexture() const noexcept
    {
        return normalTexture_ != nullptr;
    }

    [[nodiscard]] const Texture& metallicRoughnessTexture() const noexcept
    {
        return *metallicRoughnessTexture_;
    }
    void metallicRoughnessTexture(const Texture* t) noexcept
    {
        metallicRoughnessTexture_ = t;
    }
    [[nodiscard]] bool hasMetallicRoughnessTexture() const noexcept
    {
        return metallicRoughnessTexture_ != nullptr;
    }

    [[nodiscard]] const Texture& occlusionTexture() const noexcept
    {
        return *occlusionTexture_;
    }
    void occlusionTexture(const Texture* t) noexcept
    {
        occlusionTexture_ = t;
    }
    [[nodiscard]] bool hasOcclusionTexture() const noexcept
    {
        return occlusionTexture_ != nullptr;
    }

    [[nodiscard]] const Texture& transmissionTexture() const noexcept
    {
        return *transmissionTexture_;
    }
    void transmissionTexture(const Texture* t) noexcept
    {
        transmissionTexture_ = t;
    }
    [[nodiscard]] bool hasTransmissionTexture() const noexcept
    {
        return transmissionTexture_ != nullptr;
    }

    // KHR_materials_clearcoat — thin lacquer layer over the base BRDF. Three
    // optional textures (factor R, roughness G, normal RGB), each with their
    // own UV-set + UvTransform. Defaults: factor 0 (clearcoat absent),
    // roughness 0, normalScale 1.
    [[nodiscard]] const Texture& clearcoatTexture() const noexcept
    {
        return *clearcoatTexture_;
    }
    void clearcoatTexture(const Texture* t) noexcept
    {
        clearcoatTexture_ = t;
    }
    [[nodiscard]] bool hasClearcoatTexture() const noexcept
    {
        return clearcoatTexture_ != nullptr;
    }

    [[nodiscard]] const Texture& clearcoatRoughnessTexture() const noexcept
    {
        return *clearcoatRoughnessTexture_;
    }
    void clearcoatRoughnessTexture(const Texture* t) noexcept
    {
        clearcoatRoughnessTexture_ = t;
    }
    [[nodiscard]] bool hasClearcoatRoughnessTexture() const noexcept
    {
        return clearcoatRoughnessTexture_ != nullptr;
    }

    [[nodiscard]] const Texture& clearcoatNormalTexture() const noexcept
    {
        return *clearcoatNormalTexture_;
    }
    void clearcoatNormalTexture(const Texture* t) noexcept
    {
        clearcoatNormalTexture_ = t;
    }
    [[nodiscard]] bool hasClearcoatNormalTexture() const noexcept
    {
        return clearcoatNormalTexture_ != nullptr;
    }

    [[nodiscard]] float clearcoatFactor() const noexcept
    {
        return clearcoatFactor_;
    }
    void clearcoatFactor(float v) noexcept
    {
        clearcoatFactor_ = v;
    }
    [[nodiscard]] float clearcoatRoughness() const noexcept
    {
        return clearcoatRoughness_;
    }
    void clearcoatRoughness(float v) noexcept
    {
        clearcoatRoughness_ = v;
    }
    [[nodiscard]] float clearcoatNormalScale() const noexcept
    {
        return clearcoatNormalScale_;
    }
    void clearcoatNormalScale(float v) noexcept
    {
        clearcoatNormalScale_ = v;
    }

    [[nodiscard]] int clearcoatTexCoord() const noexcept
    {
        return clearcoatTexCoord_;
    }
    void clearcoatTexCoord(int v) noexcept
    {
        clearcoatTexCoord_ = v;
    }
    [[nodiscard]] int clearcoatRoughnessTexCoord() const noexcept
    {
        return clearcoatRoughnessTexCoord_;
    }
    void clearcoatRoughnessTexCoord(int v) noexcept
    {
        clearcoatRoughnessTexCoord_ = v;
    }
    [[nodiscard]] int clearcoatNormalTexCoord() const noexcept
    {
        return clearcoatNormalTexCoord_;
    }
    void clearcoatNormalTexCoord(int v) noexcept
    {
        clearcoatNormalTexCoord_ = v;
    }

    [[nodiscard]] UvTransform clearcoatUvTransform() const noexcept
    {
        return clearcoatUvTransform_;
    }
    void clearcoatUvTransform(UvTransform t) noexcept
    {
        clearcoatUvTransform_ = t;
    }
    [[nodiscard]] UvTransform clearcoatRoughnessUvTransform() const noexcept
    {
        return clearcoatRoughnessUvTransform_;
    }
    void clearcoatRoughnessUvTransform(UvTransform t) noexcept
    {
        clearcoatRoughnessUvTransform_ = t;
    }
    [[nodiscard]] UvTransform clearcoatNormalUvTransform() const noexcept
    {
        return clearcoatNormalUvTransform_;
    }
    void clearcoatNormalUvTransform(UvTransform t) noexcept
    {
        clearcoatNormalUvTransform_ = t;
    }

    [[nodiscard]] const std::string& name() const noexcept
    {
        return name_;
    }
    void name(const std::string& name)
    {
        name_ = name;
    }

    [[nodiscard]] Colour3 diffuse() const noexcept
    {
        return diffuse_;
    }
    void diffuse(Colour3 c) noexcept
    {
        diffuse_ = c;
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

    // glTF allows each material texture to point at TEXCOORD_0 or TEXCOORD_1.
    // Defaults are 0 — matches glTF spec when omitted. B3 wires the loader to
    // honour the source asset's per-slot index; for now nothing populates
    // anything other than 0, so behaviour is unchanged.
    [[nodiscard]] int baseColorTexCoord() const noexcept
    {
        return baseColorTexCoord_;
    }
    void baseColorTexCoord(int v) noexcept
    {
        baseColorTexCoord_ = v;
    }
    [[nodiscard]] int emissiveTexCoord() const noexcept
    {
        return emissiveTexCoord_;
    }
    void emissiveTexCoord(int v) noexcept
    {
        emissiveTexCoord_ = v;
    }
    [[nodiscard]] int normalTexCoord() const noexcept
    {
        return normalTexCoord_;
    }
    void normalTexCoord(int v) noexcept
    {
        normalTexCoord_ = v;
    }
    [[nodiscard]] int metallicRoughnessTexCoord() const noexcept
    {
        return metallicRoughnessTexCoord_;
    }
    void metallicRoughnessTexCoord(int v) noexcept
    {
        metallicRoughnessTexCoord_ = v;
    }
    [[nodiscard]] int occlusionTexCoord() const noexcept
    {
        return occlusionTexCoord_;
    }
    void occlusionTexCoord(int v) noexcept
    {
        occlusionTexCoord_ = v;
    }
    [[nodiscard]] int transmissionTexCoord() const noexcept
    {
        return transmissionTexCoord_;
    }
    void transmissionTexCoord(int v) noexcept
    {
        transmissionTexCoord_ = v;
    }

    // KHR_texture_transform per-slot UV transforms. Identity by default.
    [[nodiscard]] UvTransform baseColorUvTransform() const noexcept
    {
        return baseColorUvTransform_;
    }
    void baseColorUvTransform(UvTransform t) noexcept
    {
        baseColorUvTransform_ = t;
    }
    [[nodiscard]] UvTransform emissiveUvTransform() const noexcept
    {
        return emissiveUvTransform_;
    }
    void emissiveUvTransform(UvTransform t) noexcept
    {
        emissiveUvTransform_ = t;
    }
    [[nodiscard]] UvTransform normalUvTransform() const noexcept
    {
        return normalUvTransform_;
    }
    void normalUvTransform(UvTransform t) noexcept
    {
        normalUvTransform_ = t;
    }
    [[nodiscard]] UvTransform metallicRoughnessUvTransform() const noexcept
    {
        return metallicRoughnessUvTransform_;
    }
    void metallicRoughnessUvTransform(UvTransform t) noexcept
    {
        metallicRoughnessUvTransform_ = t;
    }
    [[nodiscard]] UvTransform occlusionUvTransform() const noexcept
    {
        return occlusionUvTransform_;
    }
    void occlusionUvTransform(UvTransform t) noexcept
    {
        occlusionUvTransform_ = t;
    }
    [[nodiscard]] UvTransform transmissionUvTransform() const noexcept
    {
        return transmissionUvTransform_;
    }
    void transmissionUvTransform(UvTransform t) noexcept
    {
        transmissionUvTransform_ = t;
    }

    // KHR_materials_transmission. transmissionFactor scales the optional
    // transmissionTexture (red channel). Default is 0 (no transmission) — when
    // both factor and texture are absent the slot stays inert. Spec says the
    // sampled R is gamma-encoded? No — the texture is linear. Encoding handled
    // at load time by passing TextureEncoding::Linear.
    [[nodiscard]] float transmissionFactor() const noexcept
    {
        return transmissionFactor_;
    }
    void transmissionFactor(float v) noexcept
    {
        transmissionFactor_ = v;
    }

    // KHR_materials_ior — index of refraction. Default 1.5 matches the glTF
    // spec for common dielectrics; water 1.33, plastic 1.46, sapphire 1.77,
    // diamond 2.42, etc. Only consulted by the transmission lobe in the
    // fragment shader, so non-transmissive materials are unaffected.
    [[nodiscard]] float ior() const noexcept
    {
        return ior_;
    }
    void ior(float v) noexcept
    {
        ior_ = v;
    }

    // KHR_materials_volume — thicknessFactor + attenuation. Together with
    // KHR_materials_transmission they control how light refracts through and
    // is absorbed by the volume. Defaults: factor 0 (no thickness, surface
    // is thin and refraction collapses to F3 entry-point sampling),
    // attenuationColor (1,1,1) and attenuationDistance +infinity (no Beer-
    // Lambert tinting). Thickness texture's G channel multiplies the factor.
    [[nodiscard]] float thicknessFactor() const noexcept
    {
        return thicknessFactor_;
    }
    void thicknessFactor(float v) noexcept
    {
        thicknessFactor_ = v;
    }
    [[nodiscard]] Colour3 attenuationColor() const noexcept
    {
        return attenuationColor_;
    }
    void attenuationColor(Colour3 c) noexcept
    {
        attenuationColor_ = c;
    }
    [[nodiscard]] float attenuationDistance() const noexcept
    {
        return attenuationDistance_;
    }
    void attenuationDistance(float v) noexcept
    {
        attenuationDistance_ = v;
    }
    [[nodiscard]] const Texture& thicknessTexture() const noexcept
    {
        return *thicknessTexture_;
    }
    void thicknessTexture(const Texture* t) noexcept
    {
        thicknessTexture_ = t;
    }
    [[nodiscard]] bool hasThicknessTexture() const noexcept
    {
        return thicknessTexture_ != nullptr;
    }
    [[nodiscard]] int thicknessTexCoord() const noexcept
    {
        return thicknessTexCoord_;
    }
    void thicknessTexCoord(int v) noexcept
    {
        thicknessTexCoord_ = v;
    }
    [[nodiscard]] UvTransform thicknessUvTransform() const noexcept
    {
        return thicknessUvTransform_;
    }
    void thicknessUvTransform(UvTransform t) noexcept
    {
        thicknessUvTransform_ = t;
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

private:
    std::string name_;
    Colour3 diffuse_{};
    Colour3 emissive_{};
    float roughness_{0.0f};
    float metallic_{0.0f};
    float alpha_{1.0f};
    float normalScale_{1.0f};
    float occlusionStrength_{1.0f};
    int baseColorTexCoord_{0};
    int emissiveTexCoord_{0};
    int normalTexCoord_{0};
    int metallicRoughnessTexCoord_{0};
    int occlusionTexCoord_{0};
    int transmissionTexCoord_{0};
    UvTransform baseColorUvTransform_{};
    UvTransform emissiveUvTransform_{};
    UvTransform normalUvTransform_{};
    UvTransform metallicRoughnessUvTransform_{};
    UvTransform occlusionUvTransform_{};
    UvTransform transmissionUvTransform_{};
    AlphaMode alphaMode_{AlphaMode::Opaque};
    float alphaCutoff_{0.5f};
    float transmissionFactor_{0.0f};
    float ior_{1.5f};
    float clearcoatFactor_{0.0f};
    float clearcoatRoughness_{0.0f};
    float clearcoatNormalScale_{1.0f};
    float thicknessFactor_{0.0f};
    Colour3 attenuationColor_{1.0f, 1.0f, 1.0f};
    float attenuationDistance_{std::numeric_limits<float>::infinity()};
    int thicknessTexCoord_{0};
    UvTransform thicknessUvTransform_{};
    int clearcoatTexCoord_{0};
    int clearcoatRoughnessTexCoord_{0};
    int clearcoatNormalTexCoord_{0};
    UvTransform clearcoatUvTransform_{};
    UvTransform clearcoatRoughnessUvTransform_{};
    UvTransform clearcoatNormalUvTransform_{};
    bool doubleSided_{false};
    bool unlit_{false};

    const Texture* texture_{nullptr};
    const Texture* emissiveTexture_{nullptr};
    const Texture* normalTexture_{nullptr};
    const Texture* metallicRoughnessTexture_{nullptr};
    const Texture* occlusionTexture_{nullptr};
    const Texture* transmissionTexture_{nullptr};
    const Texture* clearcoatTexture_{nullptr};
    const Texture* clearcoatRoughnessTexture_{nullptr};
    const Texture* clearcoatNormalTexture_{nullptr};
    const Texture* thicknessTexture_{nullptr};
};

} // namespace fire_engine
