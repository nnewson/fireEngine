#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/material_binding.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/render/ubo.hpp>

using namespace fire_engine;

using Slot = MaterialTextureSlot;

TEST_CASE("Material.DefaultConstructionUsesPbrDefaults", "[Material]")
{
    Material mat;

    CHECK(mat.name().empty());
    CHECK_FALSE(mat.texture(Slot::BaseColour).has());
    CHECK_FALSE(mat.texture(Slot::Emissive).has());
    CHECK_FALSE(mat.texture(Slot::Normal).has());
    CHECK_FALSE(mat.texture(Slot::MetallicRoughness).has());
    CHECK_FALSE(mat.texture(Slot::Occlusion).has());

    CHECK(mat.baseColor().r() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(mat.baseColor().g() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(mat.baseColor().b() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(mat.emissive().r() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(mat.roughness() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(mat.metallic() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(mat.alpha() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(mat.normalScale() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(mat.alphaMode() == AlphaMode::Opaque);
    CHECK(mat.alphaCutoff() == Catch::Approx(0.5f).margin(1e-5f));
    CHECK_FALSE(mat.doubleSided());
}

TEST_CASE("Material.DefaultExtensionBlocksAreAbsent", "[Material]")
{
    Material mat;
    CHECK_FALSE(mat.transmission().has_value());
    CHECK_FALSE(mat.clearcoat().has_value());
    CHECK_FALSE(mat.volume().has_value());
}

TEST_CASE("Material.SetAndGetCorePbrFields", "[Material]")
{
    Material mat;

    mat.name("helmet");
    mat.baseColor({0.8f, 0.6f, 0.4f});
    mat.emissive({0.1f, 0.2f, 0.3f});
    mat.roughness(0.75f);
    mat.metallic(0.5f);
    mat.alpha(0.25f);
    mat.normalScale(0.9f);
    mat.alphaMode(AlphaMode::Blend);
    mat.alphaCutoff(0.2f);
    mat.doubleSided(true);

    CHECK(mat.name() == "helmet");
    CHECK(mat.baseColor().r() == Catch::Approx(0.8f).margin(1e-5f));
    CHECK(mat.baseColor().g() == Catch::Approx(0.6f).margin(1e-5f));
    CHECK(mat.baseColor().b() == Catch::Approx(0.4f).margin(1e-5f));
    CHECK(mat.emissive().r() == Catch::Approx(0.1f).margin(1e-5f));
    CHECK(mat.emissive().g() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(mat.emissive().b() == Catch::Approx(0.3f).margin(1e-5f));
    CHECK(mat.roughness() == Catch::Approx(0.75f).margin(1e-5f));
    CHECK(mat.metallic() == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(mat.alpha() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(mat.normalScale() == Catch::Approx(0.9f).margin(1e-5f));
    CHECK(mat.alphaMode() == AlphaMode::Blend);
    CHECK(mat.alphaCutoff() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(mat.doubleSided());
}

TEST_CASE("Material.TextureSlotPointersRoundTrip", "[Material]")
{
    Material mat;
    Texture base;
    Texture emissive;
    Texture normal;
    Texture mr;
    Texture occlusion;
    Texture transmission;

    mat.texture(Slot::BaseColour).texture = &base;
    mat.texture(Slot::Emissive).texture = &emissive;
    mat.texture(Slot::Normal).texture = &normal;
    mat.texture(Slot::MetallicRoughness).texture = &mr;
    mat.texture(Slot::Occlusion).texture = &occlusion;
    mat.texture(Slot::Transmission).texture = &transmission;

    CHECK(mat.texture(Slot::BaseColour).has());
    CHECK(mat.texture(Slot::Emissive).has());
    CHECK(mat.texture(Slot::Normal).has());
    CHECK(mat.texture(Slot::MetallicRoughness).has());
    CHECK(mat.texture(Slot::Occlusion).has());
    CHECK(mat.texture(Slot::Transmission).has());
    CHECK(mat.texture(Slot::BaseColour).texture == &base);
}

TEST_CASE("Material.TextureSlotTexCoordAndTransformRoundTrip", "[Material]")
{
    Material mat;
    mat.texture(Slot::Normal).texCoord = 1;
    mat.texture(Slot::Normal).transform = UvTransform{0.5f, 0.25f, 2.0f, 3.0f, 0.4f};

    CHECK(mat.texture(Slot::Normal).texCoord == 1);
    CHECK(mat.texture(Slot::Normal).transform.offsetX == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(mat.texture(Slot::Normal).transform.scaleY == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(mat.texture(Slot::Normal).transform.rotation == Catch::Approx(0.4f).margin(1e-5f));
}

TEST_CASE("MaterialBinding.ToMaterialUboPacksCoreFields", "[MaterialBinding]")
{
    Material mat;
    mat.baseColor({0.8f, 0.6f, 0.4f});
    mat.emissive({0.1f, 0.2f, 0.3f});
    mat.roughness(0.75f);
    mat.metallic(0.5f);
    mat.alpha(0.25f);
    mat.normalScale(0.9f);
    mat.alphaMode(AlphaMode::Mask);
    mat.alphaCutoff(0.2f);
    mat.occlusionStrength(0.7f);

    const MaterialUBO ubo = toMaterialUBO(mat);

    CHECK(ubo.diffuseAlpha[0] == Catch::Approx(0.8f).margin(1e-5f));
    CHECK(ubo.diffuseAlpha[3] == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(ubo.emissiveRoughness[2] == Catch::Approx(0.3f).margin(1e-5f));
    CHECK(ubo.emissiveRoughness[3] == Catch::Approx(0.75f).margin(1e-5f));
    CHECK(ubo.materialParams[0] == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(ubo.materialParams[1] == Catch::Approx(0.9f).margin(1e-5f));
    CHECK(ubo.materialParams[2] == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(ubo.materialParams[3] == Catch::Approx(0.7f).margin(1e-5f));
}

TEST_CASE("MaterialBinding.PackingEnforcesTheAlphaRangeInvariant", "[MaterialBinding]")
{
    // The invariant two depth-writing passes rest on. `shader.frag` applies `alpha < alphaCutoff`
    // to EVERY material (a non-MASK one packs cutoff 0), while `depth_prepass.frag` SKIPS that test
    // when the packed cutoff is 0 — equivalent only while alpha >= 0. If an out-of-range value
    // reached the GPU the two would disagree, and a fragment the forward pass discards but the
    // prepass keeps leaves a depth-only occluder: invisible geometry that still hides what is
    // behind it.
    //
    // `Material` is a plain value type that accepts any float, so this seam is where the guarantee
    // is made. The two clamps differ in kind: the CUTOFF clamp is behaviour-preserving (a negative
    // cutoff already discarded nothing in the forward pass), while the ALPHA clamp deliberately
    // CHANGES behaviour for invalid input — a negative alpha used to discard, and clamped to 0 the
    // fragment is kept. That is glTF-spec normalisation of a value nobody meant, not preservation.
    SECTION("base-colour alpha is clamped into glTF's [0,1]")
    {
        Material mat;
        mat.alpha(-0.25f);
        CHECK(toMaterialUBO(mat).diffuseAlpha[3] == Catch::Approx(0.0f).margin(1e-5f));
        mat.alpha(4.0f);
        CHECK(toMaterialUBO(mat).diffuseAlpha[3] == Catch::Approx(1.0f).margin(1e-5f));
    }
    SECTION("a non-finite alpha packs opaque")
    {
        // Every non-finite value, not just NaN: negative infinity would otherwise fall through the
        // range path to 0 and read as fully transparent, which is a different answer from "this
        // value names no coverage at all".
        Material mat;
        for (const float bad :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity()})
        {
            mat.alpha(bad);
            CHECK(toMaterialUBO(mat).diffuseAlpha[3] == Catch::Approx(1.0f).margin(1e-5f));
        }
    }
    SECTION("a negative cutoff packs 0, which is what it already meant")
    {
        Material mat;
        mat.alphaMode(AlphaMode::Mask);
        mat.alphaCutoff(-0.5f);
        CHECK(toMaterialUBO(mat).materialParams[2] == Catch::Approx(0.0f).margin(1e-5f));
    }
    SECTION("a non-finite cutoff packs 0")
    {
        Material mat;
        mat.alphaMode(AlphaMode::Mask);
        for (const float bad :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity()})
        {
            mat.alphaCutoff(bad);
            CHECK(toMaterialUBO(mat).materialParams[2] == Catch::Approx(0.0f).margin(1e-5f));
        }
    }
    SECTION("the specific divergence that motivated this: MASK, cutoff 0, negative alpha")
    {
        // Pre-enforcement this pair made the forward pass discard (-0.25 < 0) and the prepass keep
        // (its gate saw cutoff 0 and skipped the test). Packed, the pair can no longer express it:
        // alpha is 0 and the cutoff is 0, so NEITHER pass discards.
        Material mat;
        mat.alphaMode(AlphaMode::Mask);
        mat.alphaCutoff(0.0f);
        mat.alpha(-0.25f);
        const MaterialUBO ubo = toMaterialUBO(mat);
        CHECK(ubo.diffuseAlpha[3] == Catch::Approx(0.0f).margin(1e-5f));
        CHECK(ubo.materialParams[2] == Catch::Approx(0.0f).margin(1e-5f));
        CHECK_FALSE(ubo.diffuseAlpha[3] < ubo.materialParams[2]);
    }
    SECTION("in-range values are packed untouched")
    {
        Material mat;
        mat.alphaMode(AlphaMode::Mask);
        mat.alpha(0.25f);
        mat.alphaCutoff(0.75f);
        const MaterialUBO ubo = toMaterialUBO(mat);
        CHECK(ubo.diffuseAlpha[3] == Catch::Approx(0.25f).margin(1e-5f));
        CHECK(ubo.materialParams[2] == Catch::Approx(0.75f).margin(1e-5f));
    }
}

TEST_CASE("MaterialBinding.AlphaRangeIssuesAreReportableWithoutPacking", "[MaterialBinding]")
{
    // The diagnostic's DECISION, pinned pure. The warning itself sits behind this and is emitted
    // once per material from Resources::registerMaterial, because `toMaterialUBO` is reachable from
    // `Object::wouldChangeVariant` — a noexcept query — where a throw out of log formatting would
    // terminate the process. Keeping the decision separate from the reporting is what makes it
    // testable at all.
    SECTION("a well-formed material reports nothing")
    {
        Material mat;
        mat.alphaMode(AlphaMode::Mask);
        mat.alpha(0.5f);
        mat.alphaCutoff(0.5f);
        const MaterialAlphaRangeIssues issues = materialAlphaRangeIssues(mat);
        CHECK_FALSE(issues.alpha);
        CHECK_FALSE(issues.cutoff);
        CHECK_FALSE(issues.any());
    }
    SECTION("each field is reported independently")
    {
        Material alphaBad;
        alphaBad.alpha(2.0f);
        CHECK(materialAlphaRangeIssues(alphaBad).alpha);
        CHECK_FALSE(materialAlphaRangeIssues(alphaBad).cutoff);

        Material cutoffBad;
        cutoffBad.alphaMode(AlphaMode::Mask);
        cutoffBad.alphaCutoff(-1.0f);
        CHECK(materialAlphaRangeIssues(cutoffBad).cutoff);
        CHECK_FALSE(materialAlphaRangeIssues(cutoffBad).alpha);
    }
    SECTION("a cutoff nobody consults is not an issue")
    {
        // Only MASK packs a cutoff at all; on any other mode the authored value is discarded, so
        // reporting it would send someone hunting for a problem that cannot reach a shader.
        Material mat;
        mat.alphaMode(AlphaMode::Opaque);
        mat.alphaCutoff(-1.0f);
        CHECK_FALSE(materialAlphaRangeIssues(mat).any());
    }
}

TEST_CASE("MaterialBinding.ToMaterialUboUsesExtensionDefaultsWhenAbsent", "[MaterialBinding]")
{
    // Absent optional blocks must pack the same defaults the old always-present
    // members produced: transmission factor 0 / ior 1.5, clearcoat 0, etc.
    const MaterialUBO ubo = toMaterialUBO(Material{});
    CHECK(ubo.transmissionParams[0] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.transmissionParams[3] == Catch::Approx(1.5f).margin(1e-5f));
    CHECK(ubo.clearcoatParams[0] == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(ubo.clearcoatParams[2] == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(ubo.volumeParams[0] == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("MaterialBinding.MissingTextureHandlesPackAsNull", "[MaterialBinding]")
{
    const auto handles = materialTextureHandles(Material{});

    for (TextureHandle handle : handles)
    {
        CHECK(handle == NullTexture);
    }
}

TEST_CASE("MaterialBinding.AbsentBindlessTextureIndicesAreZero", "[MaterialBinding]")
{
    // With no textures, every bindless index packs as 0 (don't-care: the matching
    // present-flag is 0, so the shader never reads it).
    const MaterialUBO ubo = toMaterialUBO(Material{});
    for (std::size_t i = 0; i < materialTextureSlotCount; ++i)
    {
        CHECK(ubo.textureIndex[i] == 0);
    }
}

TEST_CASE("MaterialBinding.BindlessTextureIndexComesFromTheSlotHandle", "[MaterialBinding]")
{
    // toMaterialUBO writes each slot's bindless index from its texture handle's *index bits*
    // (the generation is validation metadata, not an array position — CR-12), and only for the
    // slots that carry a texture.
    Texture tex;
    Material mat;
    mat.texture(Slot::Normal).texture = &tex;

    const MaterialUBO ubo = toMaterialUBO(mat);
    const auto expected = static_cast<int32_t>(handleIndex(tex.handle()));
    CHECK(ubo.textureIndex[slotIndex(Slot::Normal)] == expected);
    // Untouched slots stay 0.
    CHECK(ubo.textureIndex[slotIndex(Slot::BaseColour)] == 0);
    CHECK(ubo.textureIndex[slotIndex(Slot::Emissive)] == 0);
}

TEST_CASE("Material.TransmissionBlockRoundTrip", "[Material]")
{
    Material mat;
    CHECK_FALSE(mat.transmission().has_value());

    mat.transmission(TransmissionParams{0.65f, 1.33f});
    REQUIRE(mat.transmission().has_value());
    CHECK(mat.transmission()->factor == Catch::Approx(0.65f).margin(1e-5f));
    CHECK(mat.transmission()->ior == Catch::Approx(1.33f).margin(1e-5f));
}

TEST_CASE("Material.TransmissionDefaultsMatchSpec", "[Material]")
{
    // Spec defaults when the optional is unset: factor 0, ior 1.5.
    const TransmissionParams defaults = Material{}.transmission().value_or(TransmissionParams{});
    CHECK(defaults.factor == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(defaults.ior == Catch::Approx(1.5f).margin(1e-5f));
}

TEST_CASE("Material.TransmissionTextureSlotRoundTrip", "[Material]")
{
    Material mat;
    CHECK(mat.texture(Slot::Transmission).texCoord == 0);
    mat.texture(Slot::Transmission).texCoord = 1;
    mat.texture(Slot::Transmission).transform = UvTransform{0.5f, 0.25f, 2.0f, 3.0f, 0.4f};
    CHECK(mat.texture(Slot::Transmission).texCoord == 1);
    CHECK(mat.texture(Slot::Transmission).transform.offsetX == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(mat.texture(Slot::Transmission).transform.rotation == Catch::Approx(0.4f).margin(1e-5f));
}

TEST_CASE("Material.MoveConstructionPreservesCoreFields", "[Material]")
{
    Material original;
    original.name("moved");
    original.baseColor({0.2f, 0.4f, 0.6f});
    original.emissive({0.7f, 0.1f, 0.0f});
    original.roughness(0.8f);
    original.metallic(0.9f);
    original.alpha(0.3f);
    original.normalScale(0.6f);
    original.clearcoat(ClearcoatParams{0.8f, 0.25f, 0.6f});

    Material moved(std::move(original));

    CHECK(moved.name() == "moved");
    CHECK(moved.baseColor().r() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(moved.emissive().r() == Catch::Approx(0.7f).margin(1e-5f));
    CHECK(moved.roughness() == Catch::Approx(0.8f).margin(1e-5f));
    CHECK(moved.metallic() == Catch::Approx(0.9f).margin(1e-5f));
    CHECK(moved.alpha() == Catch::Approx(0.3f).margin(1e-5f));
    CHECK(moved.normalScale() == Catch::Approx(0.6f).margin(1e-5f));
    REQUIRE(moved.clearcoat().has_value());
    CHECK(moved.clearcoat()->factor == Catch::Approx(0.8f).margin(1e-5f));
}

TEST_CASE("Material.ClearcoatBlockRoundTrips", "[Material]")
{
    Material mat;
    CHECK_FALSE(mat.clearcoat().has_value());

    mat.clearcoat(ClearcoatParams{0.8f, 0.25f, 0.6f});
    REQUIRE(mat.clearcoat().has_value());
    CHECK(mat.clearcoat()->factor == Catch::Approx(0.8f).margin(1e-5f));
    CHECK(mat.clearcoat()->roughness == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(mat.clearcoat()->normalScale == Catch::Approx(0.6f).margin(1e-5f));
}

TEST_CASE("Material.ClearcoatDefaultsMatchSpec", "[Material]")
{
    const ClearcoatParams defaults = Material{}.clearcoat().value_or(ClearcoatParams{});
    CHECK(defaults.factor == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(defaults.roughness == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(defaults.normalScale == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("Material.ClearcoatTextureSlotsRoundTrip", "[Material]")
{
    Material mat;
    Texture tFactor;
    Texture tRough;
    Texture tNormal;
    mat.texture(Slot::Clearcoat).texture = &tFactor;
    mat.texture(Slot::ClearcoatRoughness).texture = &tRough;
    mat.texture(Slot::ClearcoatNormal).texture = &tNormal;
    mat.texture(Slot::Clearcoat).texCoord = 1;
    mat.texture(Slot::ClearcoatNormal).transform = UvTransform{0.5f, 0.6f, 1.5f, 2.5f, 1.5f};

    CHECK(mat.texture(Slot::Clearcoat).has());
    CHECK(mat.texture(Slot::ClearcoatRoughness).has());
    CHECK(mat.texture(Slot::ClearcoatNormal).has());
    CHECK(mat.texture(Slot::Clearcoat).texCoord == 1);
    CHECK(mat.texture(Slot::ClearcoatNormal).transform.rotation ==
          Catch::Approx(1.5f).margin(1e-5f));
}

TEST_CASE("Material.VolumeBlockRoundTrips", "[Material]")
{
    Material mat;
    CHECK_FALSE(mat.volume().has_value());

    mat.volume(VolumeParams{0.7f, Colour3(0.2f, 0.6f, 0.4f), 2.5f});
    REQUIRE(mat.volume().has_value());
    CHECK(mat.volume()->thicknessFactor == Catch::Approx(0.7f).margin(1e-5f));
    CHECK(mat.volume()->attenuationColor == Colour3(0.2f, 0.6f, 0.4f));
    CHECK(mat.volume()->attenuationDistance == Catch::Approx(2.5f).margin(1e-5f));
}

TEST_CASE("Material.VolumeDefaultsMatchSpec", "[Material]")
{
    const VolumeParams defaults = Material{}.volume().value_or(VolumeParams{});
    CHECK(defaults.thicknessFactor == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(defaults.attenuationColor == Colour3(1.0f, 1.0f, 1.0f));
    CHECK(std::isinf(defaults.attenuationDistance));
}

TEST_CASE("Material.ThicknessTextureSlotRoundTrip", "[Material]")
{
    Material mat;
    Texture t;
    mat.texture(Slot::Thickness).texture = &t;
    mat.texture(Slot::Thickness).texCoord = 1;
    CHECK(mat.texture(Slot::Thickness).has());
    CHECK(mat.texture(Slot::Thickness).texCoord == 1);
}
