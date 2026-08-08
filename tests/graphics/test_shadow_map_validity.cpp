#include <fire_engine/graphics/shadow_map_validity.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace fire_engine;

namespace
{

constexpr auto kCascades = static_cast<std::size_t>(kShadowCascadeCount);
constexpr auto kFaces = static_cast<std::size_t>(kCubeFaceCount);

// A frame with everything present, so each test can spoil exactly one thing and attribute the
// result to it.
ShadowMapValidityInputs everything()
{
    return ShadowMapValidityInputs{
        .shadowsDisabled = false,
        .primaryDirectionalLight = true,
        .activeCascadeViews = kCascades,
        .activeWorldOnlyViews = kCascades,
        .activeSelfViews = 2,
        .activeSpotViews = 1,
        .activePointViews = kFaces,
    };
}

} // namespace

TEST_CASE("a complete frame validates every family", "[ShadowMapValidity]")
{
    const ShadowMapValidity validity = shadowMapValidity(everything());
    CHECK(validity.cascades);
    CHECK(validity.worldOnly);
    CHECK(validity.self);
    CHECK(validity.spot);
    CHECK(validity.point);
    CHECK_FALSE(validity.none());
}

TEST_CASE("disabling shadows clears every family", "[ShadowMapValidity]")
{
    // Not "clears the directional ones": --no-shadows means the pass records nothing at all, and
    // `none()` is what the pass returns on.
    ShadowMapValidityInputs inputs = everything();
    inputs.shadowsDisabled = true;
    const ShadowMapValidity validity = shadowMapValidity(inputs);
    CHECK(validity == ShadowMapValidity{});
    CHECK(validity.none());
    CHECK(validity.packedMask() == 0);
}

TEST_CASE("the directional families need a real primary directional light", "[ShadowMapValidity]")
{
    // The cascades are FITTED whether or not a sun exists — the views must stay well-formed — so
    // the active counts alone cannot distinguish a real fit from one against the fallback
    // direction. Only this flag can, which is why validity asks for it rather than for a matrix.
    ShadowMapValidityInputs inputs = everything();
    inputs.primaryDirectionalLight = false;
    const ShadowMapValidity validity = shadowMapValidity(inputs);
    CHECK_FALSE(validity.cascades);
    CHECK_FALSE(validity.worldOnly);
    CHECK_FALSE(validity.self);
    // Punctual maps are fitted to their own lights and are unaffected.
    CHECK(validity.spot);
    CHECK(validity.point);
}

TEST_CASE("a partial cascade set is invalid, not partially valid", "[ShadowMapValidity]")
{
    // A fragment picks its cascade layer from its view depth. Three fitted cascades out of four is
    // a family with a hole, and sampling the hole reads another frame's depth.
    for (std::size_t active = 0; active < kCascades; ++active)
    {
        ShadowMapValidityInputs inputs = everything();
        inputs.activeCascadeViews = active;
        CHECK_FALSE(shadowMapValidity(inputs).cascades);
    }
    ShadowMapValidityInputs complete = everything();
    complete.activeCascadeViews = kCascades;
    CHECK(shadowMapValidity(complete).cascades);
}

TEST_CASE("a partial world-only set is invalid too", "[ShadowMapValidity]")
{
    ShadowMapValidityInputs inputs = everything();
    inputs.activeWorldOnlyViews = kCascades - 1;
    const ShadowMapValidity validity = shadowMapValidity(inputs);
    CHECK_FALSE(validity.worldOnly);
    // Independent families: the main CSM is untouched by the world-only twin's absence, which is
    // the ordinary case for a frame with no skinned draw.
    CHECK(validity.cascades);
}

TEST_CASE("point validity requires whole cubes", "[ShadowMapValidity]")
{
    // The six faces are installed atomically (ShadowRenderViewSet::setPointLight). A remainder
    // means that invariant was bypassed upstream; half a cube is a light whose shadow depends on
    // which way the receiver happens to face, so the family reports invalid rather than rendering
    // the faces it has.
    for (std::size_t partial = 1; partial < kFaces; ++partial)
    {
        ShadowMapValidityInputs inputs = everything();
        inputs.activePointViews = partial;
        CHECK_FALSE(shadowMapValidity(inputs).point);
    }
    ShadowMapValidityInputs twoCubes = everything();
    twoCubes.activePointViews = 2 * kFaces;
    CHECK(shadowMapValidity(twoCubes).point);

    ShadowMapValidityInputs cubeAndAHalf = everything();
    cubeAndAHalf.activePointViews = kFaces + 3;
    CHECK_FALSE(shadowMapValidity(cubeAndAHalf).point);
}

TEST_CASE("self and spot families are per-slot, so any active slot validates them",
          "[ShadowMapValidity]")
{
    // Unlike the cascades, these are addressed by an index the draw or the light carries: a slot
    // that was never assigned is never sampled, so one active slot is a valid family rather than a
    // partial one.
    ShadowMapValidityInputs oneEach = everything();
    oneEach.activeSelfViews = 1;
    oneEach.activeSpotViews = 1;
    CHECK(shadowMapValidity(oneEach).self);
    CHECK(shadowMapValidity(oneEach).spot);

    ShadowMapValidityInputs none = everything();
    none.activeSelfViews = 0;
    none.activeSpotViews = 0;
    CHECK_FALSE(shadowMapValidity(none).self);
    CHECK_FALSE(shadowMapValidity(none).spot);
}

TEST_CASE("an empty frame records nothing", "[ShadowMapValidity]")
{
    const ShadowMapValidity validity = shadowMapValidity(ShadowMapValidityInputs{});
    CHECK(validity.none());
    CHECK(validity.packedMask() == 0);
}

TEST_CASE("the packed mask uses the bits the shader reads", "[ShadowMapValidity]")
{
    // The values come from shaders/gpu_limits.glsl through gpu_limits.hpp — the same declarations
    // the receiver compiles against. Pinning them here as well would be a transcription; what this
    // checks is that each family maps to its OWN bit and to no other.
    const std::int32_t cascades =
        ShadowMapValidity{
            .cascades = true, .worldOnly = false, .self = false, .spot = false, .point = false}
            .packedMask();
    const std::int32_t worldOnly =
        ShadowMapValidity{
            .cascades = false, .worldOnly = true, .self = false, .spot = false, .point = false}
            .packedMask();
    const std::int32_t self =
        ShadowMapValidity{
            .cascades = false, .worldOnly = false, .self = true, .spot = false, .point = false}
            .packedMask();
    const std::int32_t spot =
        ShadowMapValidity{
            .cascades = false, .worldOnly = false, .self = false, .spot = true, .point = false}
            .packedMask();
    const std::int32_t point =
        ShadowMapValidity{
            .cascades = false, .worldOnly = false, .self = false, .spot = false, .point = true}
            .packedMask();

    CHECK(cascades == shader_limits::SHADOW_MAP_VALID_CASCADES);
    CHECK(worldOnly == shader_limits::SHADOW_MAP_VALID_WORLD_ONLY);
    CHECK(self == shader_limits::SHADOW_MAP_VALID_SELF);
    CHECK(spot == shader_limits::SHADOW_MAP_VALID_SPOT);
    CHECK(point == shader_limits::SHADOW_MAP_VALID_POINT);

    // Distinct, non-overlapping bits: an OR of all five must be recoverable term by term.
    const std::int32_t all = cascades | worldOnly | self | spot | point;
    CHECK(shadowMapValidity(everything()).packedMask() == all);
    CHECK((cascades & worldOnly & self & spot & point) == 0);
    CHECK(cascades + worldOnly + self + spot + point == all);
}
