#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/shadow_caster_alpha.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// SH-05: classifying a shadow caster's fragment coverage.
//
// The resolver's response to a Masked classification (full detail, and a reason that says why) is
// pinned in test_shadow_lod_resolver.cpp; the shadow pass' response (the masked fragment path, and
// the cull mode a double-sided caster needs) is pinned in tests/render/test_pipeline_config.cpp.
// These cases pin the classification ITSELF, on the same argument SH-04 made: proving that a Masked
// request falls back leaves unanswered whether anything ever classifies a cutout as Masked.
// ---------------------------------------------------------------------------

TEST_CASE("ShadowCasterAlpha.AnOpaqueMaterialCastsEveryRasterisedFragment", "[ShadowLodResolver]")
{
    Material mat;
    mat.alphaMode(AlphaMode::Opaque);
    CHECK(shadowCasterAlpha(mat) == ShadowCasterAlpha::Opaque);
}

TEST_CASE("ShadowCasterAlpha.AMaskMaterialIsClassifiedMasked", "[ShadowLodResolver]")
{
    Material mat;
    mat.alphaMode(AlphaMode::Mask);
    CHECK(shadowCasterAlpha(mat) == ShadowCasterAlpha::Masked);
}

TEST_CASE("ShadowCasterAlpha.MaskingIsTheDeclaredModeNotThePresenceOfATexture",
          "[ShadowLodResolver]")
{
    // The trap this avoids: "MASK plus a base-colour texture" looks like the only case worth the
    // fragment path, but a MASK material with no texture still tests its base-colour FACTOR's alpha
    // against its cutoff — and if that fails, the surface draws nothing while a shadow classified
    // Opaque would keep occluding. Shadow and forward have to agree about a surface that isn't
    // there.
    Material mat;
    mat.alphaMode(AlphaMode::Mask);
    mat.alphaCutoff(0.5f);
    mat.alpha(0.25f); // below the cutoff: the surface draws nothing at all
    REQUIRE(!mat.texture(MaterialTextureSlot::BaseColour).has());
    CHECK(shadowCasterAlpha(mat) == ShadowCasterAlpha::Masked);
}

TEST_CASE("ShadowCasterAlpha.BlendIsDeliberatelyNotACutout", "[ShadowLodResolver]")
{
    // BLEND shadow semantics (opaque silhouette / dithered / transmittance) is an open design
    // decision, and SH-05 must not settle it by side effect. Routing BLEND through the cutout path
    // would be inert anyway — the material authority publishes alphaCutoff 0 for it, so nothing
    // could discard — so the only effect would be a texture fetch per shadow fragment and a caster
    // needlessly pinned to full detail.
    Material mat;
    mat.alphaMode(AlphaMode::Blend);
    mat.alphaCutoff(0.5f); // authored but irrelevant to BLEND, exactly as in the forward pass
    CHECK(shadowCasterAlpha(mat) == ShadowCasterAlpha::Opaque);
}

TEST_CASE("ShadowCasterAlpha.ADefaultMaterialCastsSolid", "[ShadowLodResolver]")
{
    // glTF's default alpha mode is OPAQUE, so the common case must not pay the masked path. This is
    // the counterweight to the request's pessimistic default: the CLASSIFIER is precise, and the
    // default only applies where nobody classified anything.
    const Material mat;
    CHECK(shadowCasterAlpha(mat) == ShadowCasterAlpha::Opaque);
}
