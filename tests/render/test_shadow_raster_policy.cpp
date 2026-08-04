#include <catch2/catch_test_macros.hpp>

#include <fire_engine/render/shadows.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// SH-05: the two mappings the shadow pass rasterises through.
//
// Both are pure, and both are the kind of code a reviewer cannot catch by reading a green test
// suite: swap the two arms of `forCaster` and every cutout in the scene casts a solid quad; swap
// `AllFaces` and `BackFacesOnly` and the dual-depth self-shadow layer records the wrong surface.
// Nothing else in the suite exercises them — the pipeline-config tests pin which shader each
// pipeline carries and that cull mode is dynamic, which is a different claim entirely. So they are
// tested here, exhaustively over the enum, rather than trusted because the recorder reads
// correctly.
// ---------------------------------------------------------------------------

namespace
{

// Distinct, non-null handles: a pair whose members compared equal would make every `forCaster`
// assertion below pass no matter which arm it returned.
constexpr PipelineHandle kOpaquePipeline = makeHandle<PipelineHandle>(7, 1);
constexpr PipelineHandle kMaskedPipeline = makeHandle<PipelineHandle>(8, 1);

constexpr ShadowPipelinePair kPair{.opaque = kOpaquePipeline, .masked = kMaskedPipeline};

} // namespace

TEST_CASE("ShadowPipelinePair.EachClassificationSelectsItsOwnFragmentPath", "[ShadowRasterPolicy]")
{
    static_assert(kOpaquePipeline != kMaskedPipeline,
                  "the fixture must be able to tell them apart");

    CHECK(kPair.forCaster(ShadowCasterAlpha::Opaque) == kOpaquePipeline);
    CHECK(kPair.forCaster(ShadowCasterAlpha::Masked) == kMaskedPipeline);
}

TEST_CASE("ShadowPipelinePair.AnUnclassifiedCasterTakesTheMaskedPath", "[ShadowRasterPolicy]")
{
    // The pessimistic default, pinned at the RASTER site and not only at the resolver: a producer
    // that never classified its caster must not get the solid-silhouette answer. This reads the
    // default off the request itself, which is the single place the classification is stored — if
    // that default is ever flipped to Opaque, this fails alongside the resolver's equivalent case.
    const ShadowGeometryRequest unclassified{};
    CHECK(kPair.forCaster(unclassified.alpha) == kMaskedPipeline);
}

TEST_CASE("ShadowCullMode.PerCasterFollowsTheMaterialsSidedness", "[ShadowRasterPolicy]")
{
    // THE fix in SH-05's second half. A double-sided caster culls nothing: the pass used to fix
    // `cullMode = eFront`, so a sheet authored face-on to the light had its only faces culled and
    // cast no shadow at all. A single-sided caster keeps front-culling, which is what keeps the
    // recorded depth on its back faces and receiver acne off.
    CHECK(shadowCullMode(ShadowFaceCull::PerCaster, /*casterIsDoubleSided=*/true) ==
          vk::CullModeFlagBits::eNone);
    CHECK(shadowCullMode(ShadowFaceCull::PerCaster, /*casterIsDoubleSided=*/false) ==
          vk::CullModeFlagBits::eFront);
}

TEST_CASE("ShadowCullMode.SelfShadowLayersIgnoreTheMaterialsSidedness", "[ShadowRasterPolicy]")
{
    // The two self-shadow layers are a PASS policy, not a caster property: the first captures
    // whatever surface the light sees first (cull nothing), the second rasterises back faces only
    // (cull front) so the dual-depth rejection is well-founded rather than a coin-flip on marginal
    // fragments. A double-sided caster must not change either — that is what makes the second
    // layer's depth reliably behind the first's.
    for (const bool doubleSided : {false, true})
    {
        CHECK(shadowCullMode(ShadowFaceCull::AllFaces, doubleSided) == vk::CullModeFlagBits::eNone);
        CHECK(shadowCullMode(ShadowFaceCull::BackFacesOnly, doubleSided) ==
              vk::CullModeFlagBits::eFront);
    }
}

TEST_CASE("ShadowCullMode.TheThreePoliciesAreNotInterchangeable", "[ShadowRasterPolicy]")
{
    // Guards the whole file against becoming vacuous: if two policies ever answered alike for every
    // input, the cases above would keep passing while the distinction they exist to protect had
    // quietly gone. `AllFaces` and `BackFacesOnly` differ by construction; `PerCaster` must differ
    // from each on at least one sidedness.
    CHECK(shadowCullMode(ShadowFaceCull::AllFaces, false) !=
          shadowCullMode(ShadowFaceCull::BackFacesOnly, false));
    CHECK(shadowCullMode(ShadowFaceCull::PerCaster, false) !=
          shadowCullMode(ShadowFaceCull::AllFaces, false));
    CHECK(shadowCullMode(ShadowFaceCull::PerCaster, true) !=
          shadowCullMode(ShadowFaceCull::BackFacesOnly, true));
}
