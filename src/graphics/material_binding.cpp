#include <fire_engine/graphics/material_binding.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include <fire_engine/core/log.hpp>

#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/render/descriptor_bindings.hpp>
#include <fire_engine/render/ubo.hpp>

namespace fire_engine
{

namespace
{

// GPU-side sentinel for "no Beer–Lambert attenuation". Large enough that
// exp(-d / distance) ≈ 1 across any plausible scene depth, but finite so the
// shader can branch on it without an infinity guard. The CPU-side spec default
// is std::numeric_limits<float>::infinity(); this constant is the value we
// shove into the UBO when CPU says "no attenuation".
inline constexpr float kMaxAttenuationDistance = 1.0e6f;

[[nodiscard]]
bool sameTextureSlot(const TextureSlot& a, const TextureSlot& b) noexcept
{
    if (a.has() != b.has())
    {
        return false;
    }
    if (!a.has())
    {
        return true;
    }
    return a.texture->handle() == b.texture->handle();
}

// The ALPHA-RANGE INVARIANT of the GPU material, enforced here because this is the one seam every
// producer (the glTF loader, procedural materials, variant materials) passes through on its way to
// the shaders. `Material` accepts any float — it is a plain value type — so the guarantee has to be
// made where the value becomes GPU truth, not hoped for at each call site.
//
// glTF pins both ranges: `baseColorFactor.a` is [0,1], and `alphaCutoff` has a spec minimum of 0.
// Out-of-range values are normalised to the spec's reading of them rather than rejected — this is
// authored colour data, not a metric. The two clamps are NOT the same kind of change:
//
//   * the CUTOFF clamp is behaviour-preserving. A negative cutoff already discarded nothing in
//     `shader.frag` (any alpha >= 0 clears it), so mapping it to 0 keeps exactly that;
//   * the ALPHA clamp deliberately CHANGES behaviour for invalid input, and that is the point. A
//     negative alpha used to discard in `shader.frag` — an OPAQUE surface silently vanishing on a
//     value nobody meant, since a non-MASK material packs cutoff 0 and the test still runs. Clamped
//     to 0 the fragment is kept. That is glTF-spec normalisation of nonsense, not preservation of
//     it.
//
// Two passes depend on the invariant holding:
//
//   * `shader.frag` applies `alpha < alphaCutoff` to EVERY material, per the above;
//   * `depth_prepass.frag` SKIPS that test when the packed cutoff is 0, which is only equivalent to
//     running it while alpha >= 0. Without this invariant the prepass would keep a fragment the
//     forward pass discards, leaving a depth-only occluder — the exact class of divergence the
//     cutout-aware prepass exists to remove.
//
// These two are PURE and noexcept, and deliberately do not log. `toMaterialUBO` is reachable from
// `materialsEquivalent`, which `Object::wouldChangeVariant` and `Mesh::isSelectableVariantState`
// call from `noexcept` query functions — a throw out of formatting or allocation inside a warning
// would terminate the process. The diagnostic lives in `warnOnMaterialAlphaRangeIssues` below,
// which a non-noexcept caller invokes ONCE per material (see Resources::registerMaterial); that
// also stops a variant-selection query from re-emitting the same warning every frame.
[[nodiscard]]
float packedAlpha(float alpha) noexcept
{
    if (!std::isfinite(alpha))
    {
        return 1.0f; // opaque: the safe reading of a value that names no coverage at all
    }
    return std::clamp(alpha, 0.0f, 1.0f);
}

[[nodiscard]]
float packedAlphaCutoff(float cutoff) noexcept
{
    if (!std::isfinite(cutoff))
    {
        return 0.0f; // discards nothing, which is what a non-finite threshold cannot ask for
    }
    return std::max(cutoff, 0.0f);
}

void writeUv(UvXform& dst, const UvTransform& transform) noexcept
{
    dst.offsetScale[0] = transform.offsetX;
    dst.offsetScale[1] = transform.offsetY;
    dst.offsetScale[2] = transform.scaleX;
    dst.offsetScale[3] = transform.scaleY;
    dst.rotation = transform.rotation;
}

} // namespace

MaterialAlphaRangeIssues materialAlphaRangeIssues(const Material& mat) noexcept
{
    // The cutoff is only consulted for MASK — every other mode packs 0 regardless of what was
    // authored, so an out-of-range cutoff on an OPAQUE material is not an issue with anything.
    const bool cutoffMatters = mat.alphaMode() == AlphaMode::Mask;
    return MaterialAlphaRangeIssues{
        .alpha = packedAlpha(mat.alpha()) != mat.alpha(),
        .cutoff = cutoffMatters && packedAlphaCutoff(mat.alphaCutoff()) != mat.alphaCutoff(),
    };
}

void warnOnMaterialAlphaRangeIssues(const Material& mat)
{
    const MaterialAlphaRangeIssues issues = materialAlphaRangeIssues(mat);
    if (issues.alpha)
    {
        log::warn(
            log::category::general,
            "material base-colour alpha {} is outside glTF's [0,1] (or not finite); packing {}",
            mat.alpha(), packedAlpha(mat.alpha()));
    }
    if (issues.cutoff)
    {
        log::warn(
            log::category::general,
            "material alphaCutoff {} is negative or not finite; packing {} (discards nothing)",
            mat.alphaCutoff(), packedAlphaCutoff(mat.alphaCutoff()));
    }
}

MaterialUBO toMaterialUBO(const Material& mat)
{
    MaterialUBO ubo{};
    ubo.diffuseAlpha[0] = mat.baseColor().r();
    ubo.diffuseAlpha[1] = mat.baseColor().g();
    ubo.diffuseAlpha[2] = mat.baseColor().b();
    ubo.diffuseAlpha[3] = packedAlpha(mat.alpha());
    ubo.emissiveRoughness[0] = mat.emissive().r();
    ubo.emissiveRoughness[1] = mat.emissive().g();
    ubo.emissiveRoughness[2] = mat.emissive().b();
    ubo.emissiveRoughness[3] = mat.roughness();
    ubo.materialParams[0] = mat.metallic();
    ubo.materialParams[1] = mat.normalScale();
    // Cutoff reaches the GPU only for MASK — every other mode packs 0, which is what makes the
    // shared cutout test inert rather than wrong for them.
    ubo.materialParams[2] =
        mat.alphaMode() == AlphaMode::Mask ? packedAlphaCutoff(mat.alphaCutoff()) : 0.0f;
    ubo.materialParams[3] = mat.occlusionStrength();
    using Slot = MaterialTextureSlot;
    ubo.textureFlags[0] = mat.texture(Slot::BaseColour).has() ? 1 : 0;
    ubo.textureFlags[1] = mat.texture(Slot::Emissive).has() ? 1 : 0;
    ubo.textureFlags[2] = mat.texture(Slot::Normal).has() ? 1 : 0;
    ubo.textureFlags[3] = mat.texture(Slot::MetallicRoughness).has() ? 1 : 0;
    ubo.extraFlags[0] = mat.texture(Slot::Occlusion).has() ? 1 : 0;
    ubo.extraFlags[1] = mat.texture(Slot::Occlusion).texCoord;
    ubo.extraFlags[2] = mat.unlit() ? 1 : 0;
    ubo.texCoordIndices[0] = mat.texture(Slot::BaseColour).texCoord;
    ubo.texCoordIndices[1] = mat.texture(Slot::Emissive).texCoord;
    ubo.texCoordIndices[2] = mat.texture(Slot::Normal).texCoord;
    ubo.texCoordIndices[3] = mat.texture(Slot::MetallicRoughness).texCoord;

    // Pack every slot's KHR_texture_transform + bindless texture index, indexed by
    // MaterialTextureSlot. The index is the texture's handle value (its slot in the
    // global set-2 textures[] array); 0 when the slot has no texture (read only
    // where the matching present-flag is set, so the value is don't-care there).
    for (std::size_t i = 0; i < materialTextureSlotCount; ++i)
    {
        const TextureSlot& slot = mat.texture(static_cast<Slot>(i));
        writeUv(ubo.uv[i], slot.transform);
        // The shader indexes the bindless textures[] array by the handle's index bits only
        // (the generation is validation metadata, not part of the array position).
        ubo.textureIndex[i] =
            slot.has() ? static_cast<int32_t>(handleIndex(slot.texture->handle())) : 0;
    }

    // Optional extension blocks: value_or({}) reproduces the old always-present
    // defaults (transmission factor 0 / ior 1.5, clearcoat 0, thickness 0, etc.)
    // so the packed UBO is unchanged for materials without the extension.
    const TransmissionParams tr = mat.transmission().value_or(TransmissionParams{});
    ubo.transmissionParams[0] = tr.factor;
    ubo.transmissionParams[1] = mat.texture(Slot::Transmission).has() ? 1.0f : 0.0f;
    ubo.transmissionParams[2] = static_cast<float>(mat.texture(Slot::Transmission).texCoord);
    ubo.transmissionParams[3] = tr.ior;

    const ClearcoatParams cc = mat.clearcoat().value_or(ClearcoatParams{});
    ubo.clearcoatParams[0] = cc.factor;
    ubo.clearcoatParams[1] = cc.roughness;
    ubo.clearcoatParams[2] = cc.normalScale;
    ubo.clearcoatFlags[0] = mat.texture(Slot::Clearcoat).has() ? 1.0f : 0.0f;
    ubo.clearcoatFlags[1] = mat.texture(Slot::ClearcoatRoughness).has() ? 1.0f : 0.0f;
    ubo.clearcoatFlags[2] = mat.texture(Slot::ClearcoatNormal).has() ? 1.0f : 0.0f;
    ubo.clearcoatTexCoords[0] = static_cast<float>(mat.texture(Slot::Clearcoat).texCoord);
    ubo.clearcoatTexCoords[1] = static_cast<float>(mat.texture(Slot::ClearcoatRoughness).texCoord);
    ubo.clearcoatTexCoords[2] = static_cast<float>(mat.texture(Slot::ClearcoatNormal).texCoord);

    const VolumeParams vol = mat.volume().value_or(VolumeParams{});
    ubo.volumeParams[0] = vol.thicknessFactor;
    ubo.volumeParams[1] = mat.texture(Slot::Thickness).has() ? 1.0f : 0.0f;
    ubo.volumeParams[2] = static_cast<float>(mat.texture(Slot::Thickness).texCoord);
    // volumeParams[3] is reserved; the thickness UV rotation is part of the
    // unified uv[Thickness] slot written by the loop above.
    ubo.attenuation[0] = vol.attenuationColor.r();
    ubo.attenuation[1] = vol.attenuationColor.g();
    ubo.attenuation[2] = vol.attenuationColor.b();

    const float attenuationDistance = vol.attenuationDistance;
    ubo.attenuation[3] = attenuationDistance <= 0.0f || !std::isfinite(attenuationDistance)
                             ? kMaxAttenuationDistance
                             : attenuationDistance;
    return ubo;
}

bool materialsEquivalent(const Material& lhs, const Material& rhs)
{
    const MaterialUBO lhsUbo = toMaterialUBO(lhs);
    const MaterialUBO rhsUbo = toMaterialUBO(rhs);
    // Deliberate byte-image compare: two materials are equivalent iff they serialize to identical
    // GPU UBO bytes. toMaterialUBO value-initialises (`{}`) so padding is deterministically zero;
    // the float bit-compare (not value-compare) is intended — a ±0.0/NaN difference correctly
    // counts as a distinct material. The static_assert keeps memcmp well-defined if the UBO
    // changes.
    static_assert(std::is_trivially_copyable_v<MaterialUBO>);
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison): deliberate byte-image compare (above).
    if (std::memcmp(&lhsUbo, &rhsUbo, sizeof(MaterialUBO)) != 0)
    {
        return false;
    }

    for (std::size_t i = 0; i < materialTextureSlotCount; ++i)
    {
        const auto slot = static_cast<MaterialTextureSlot>(i);
        if (!sameTextureSlot(lhs.texture(slot), rhs.texture(slot)))
        {
            return false;
        }
    }
    return true;
}

MaterialTextureHandles materialTextureHandles(const Material& material) noexcept
{
    MaterialTextureHandles handles{};
    handles.fill(NullTexture);

    for (std::size_t i = 0; i < materialTextureSlotCount; ++i)
    {
        const TextureSlot& slot = material.texture(static_cast<MaterialTextureSlot>(i));
        if (slot.has())
        {
            handles[i] = slot.texture->handle();
        }
    }

    return handles;
}

} // namespace fire_engine
