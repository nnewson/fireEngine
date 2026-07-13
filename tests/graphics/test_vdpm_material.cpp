#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/lod.hpp>
#include <fire_engine/graphics/material.hpp>
#include <fire_engine/graphics/texture.hpp>
#include <fire_engine/graphics/vdpm_material.hpp>

using namespace fire_engine;

namespace
{
// A lit material with a fully-rough surface and no textures: every channel sits at its default
// until a rule modulates it (rough ⇒ no gloss boost, so the normal scale is exactly
// kVdpmNormalScale).
Material plainMaterial()
{
    Material m;
    m.roughness(1.0f);
    return m;
}
} // namespace

TEST_CASE("vdpmChannelScales: an untextured lit material disables UV and tangent only", "[vdpm]")
{
    const VdpmChannelScales s = vdpmChannelScales(plainMaterial());
    CHECK(s.uv == 0.0f);                 // nothing samples UV
    CHECK(s.tangent == 0.0f);            // no normal map ⇒ the tangent frame is never sampled
    CHECK(s.normal == kVdpmNormalScale); // lit ⇒ shading normal stays (rough ⇒ no boost)
}

TEST_CASE("vdpmChannelScales: a normal-mapped material keeps the tangent + UV channels", "[vdpm]")
{
    const Texture tex;
    Material m = plainMaterial();
    m.texture(MaterialTextureSlot::Normal).texture = &tex;
    const VdpmChannelScales s = vdpmChannelScales(m);
    CHECK(s.tangent == kVdpmTangentScale); // a normal map samples the tangent frame
    CHECK(s.uv == kVdpmUvScale);           // and samples UV
    CHECK(s.normal == kVdpmNormalScale);
}

TEST_CASE("vdpmChannelScales: a clearcoat normal map alone keeps the tangent channel", "[vdpm]")
{
    const Texture tex;
    Material m = plainMaterial();
    m.clearcoat(ClearcoatParams{});
    m.texture(MaterialTextureSlot::ClearcoatNormal).texture = &tex;
    CHECK(vdpmChannelScales(m).tangent == kVdpmTangentScale);
}

TEST_CASE("vdpmChannelScales: an unlit material disables normal and tangent", "[vdpm]")
{
    const Texture tex;
    Material m = plainMaterial();
    m.unlit(true);
    m.texture(MaterialTextureSlot::Normal).texture = &tex; // even with a normal map present...
    const VdpmChannelScales s = vdpmChannelScales(m);
    CHECK(s.normal == 0.0f); // ...unlit never shades, so neither frame matters
    CHECK(s.tangent == 0.0f);
}

TEST_CASE("vdpmChannelScales: gloss tightens the shading-normal channel", "[vdpm]")
{
    const Texture tex;
    Material glossy = plainMaterial();
    glossy.roughness(0.0f);                                         // mirror-smooth
    glossy.texture(MaterialTextureSlot::BaseColour).texture = &tex; // keep UV on to isolate gloss
    CHECK(vdpmChannelScales(glossy).normal ==
          Catch::Approx(kVdpmNormalScale * kVdpmGlossyNormalBoost));

    Material rough = plainMaterial();
    rough.texture(MaterialTextureSlot::BaseColour).texture = &tex;
    CHECK(vdpmChannelScales(rough).normal == Catch::Approx(kVdpmNormalScale)); // rough ⇒ no boost
}
