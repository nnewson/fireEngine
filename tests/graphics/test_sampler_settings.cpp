#include <fastgltf/types.hpp>

#include <fire_engine/graphics/sampler_settings.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::FilterMode;
using fire_engine::SamplerSettings;
using fire_engine::WrapMode;

// ==========================================================================
// SamplerSettings defaults
// ==========================================================================

TEST_CASE("SamplerSettings.DefaultsToRepeatAndLinear", "[SamplerSettings]")
{
    SamplerSettings s;
    CHECK(s.wrapS == WrapMode::Repeat);
    CHECK(s.wrapT == WrapMode::Repeat);
    CHECK(s.magFilter == FilterMode::Linear);
    CHECK(s.minFilter == FilterMode::Linear);
}

TEST_CASE("SamplerSettings.EqualityOperator", "[SamplerSettings]")
{
    SamplerSettings a;
    SamplerSettings b;
    CHECK(a == b);

    b.wrapS = WrapMode::ClampToEdge;
    CHECK(a != b);
}

TEST_CASE("SamplerSettings.AllWrapModesDistinct", "[SamplerSettings]")
{
    CHECK(WrapMode::Repeat != WrapMode::MirroredRepeat);
    CHECK(WrapMode::Repeat != WrapMode::ClampToEdge);
    CHECK(WrapMode::MirroredRepeat != WrapMode::ClampToEdge);
}

TEST_CASE("SamplerSettings.AllFilterModesDistinct", "[SamplerSettings]")
{
    CHECK(FilterMode::Nearest != FilterMode::Linear);
}

// ==========================================================================
// Wrap/Filter mapping (mirrors GltfLoader helper logic)
// ==========================================================================

static WrapMode toWrapMode(fastgltf::Wrap w)
{
    switch (w)
    {
    case fastgltf::Wrap::MirroredRepeat:
        return WrapMode::MirroredRepeat;
    case fastgltf::Wrap::ClampToEdge:
        return WrapMode::ClampToEdge;
    default:
        return WrapMode::Repeat;
    }
}

static FilterMode toFilterMode(fastgltf::Filter f)
{
    switch (f)
    {
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return FilterMode::Nearest;
    default:
        return FilterMode::Linear;
    }
}

TEST_CASE("SamplerMapping.WrapRepeat", "[SamplerMapping]")
{
    CHECK(toWrapMode(fastgltf::Wrap::Repeat) == WrapMode::Repeat);
}

TEST_CASE("SamplerMapping.WrapMirroredRepeat", "[SamplerMapping]")
{
    CHECK(toWrapMode(fastgltf::Wrap::MirroredRepeat) == WrapMode::MirroredRepeat);
}

TEST_CASE("SamplerMapping.WrapClampToEdge", "[SamplerMapping]")
{
    CHECK(toWrapMode(fastgltf::Wrap::ClampToEdge) == WrapMode::ClampToEdge);
}

TEST_CASE("SamplerMapping.FilterNearest", "[SamplerMapping]")
{
    CHECK(toFilterMode(fastgltf::Filter::Nearest) == FilterMode::Nearest);
}

TEST_CASE("SamplerMapping.FilterLinear", "[SamplerMapping]")
{
    CHECK(toFilterMode(fastgltf::Filter::Linear) == FilterMode::Linear);
}

TEST_CASE("SamplerMapping.FilterNearestMipMapVariants", "[SamplerMapping]")
{
    CHECK(toFilterMode(fastgltf::Filter::NearestMipMapNearest) == FilterMode::Nearest);
    CHECK(toFilterMode(fastgltf::Filter::NearestMipMapLinear) == FilterMode::Nearest);
}

TEST_CASE("SamplerMapping.FilterLinearMipMapVariants", "[SamplerMapping]")
{
    CHECK(toFilterMode(fastgltf::Filter::LinearMipMapNearest) == FilterMode::Linear);
    CHECK(toFilterMode(fastgltf::Filter::LinearMipMapLinear) == FilterMode::Linear);
}
