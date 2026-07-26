#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/render/descriptors.hpp>
#include <fire_engine/render/ubo.hpp>

using fire_engine::DrawCommand;
using fire_engine::ForwardPushConstants;
using fire_engine::makeForwardPushConstants;

// ---------------------------------------------------------------------------
// makeForwardPushConstants — the DrawCommand → fragment push-constant mapping.
//
// Headless: the packer touches no Vulkan object, only the two plain structs.
//
// Every field is checked against a DELIBERATELY NON-DEFAULT value. The bug this
// function was extracted to kill (the transmission recorder silently dropping
// lodLevel) would have survived any test that packed a default-constructed
// command, because the omitted field's correct value was also its default.
// ---------------------------------------------------------------------------

TEST_CASE("Descriptors.ForwardPushConstantsCarryEveryDrawCommandField", "[Descriptors]")
{
    DrawCommand dc{};
    dc.selfShadowSlot = 3;
    dc.materialIndex = 17u;
    dc.lodLevel = 2u;
    dc.shadowLodLevel = 1u;

    const ForwardPushConstants pc = makeForwardPushConstants(dc);

    CHECK(pc.selfShadowSlot == 3);
    CHECK(pc.materialIndex == 17u);
    CHECK(pc.lodLevel == 2u);
    CHECK(pc.shadowLodLevel == 1u);
}

TEST_CASE("Descriptors.ForwardPushConstantsCarryTheNoShadowLodSentinel", "[Descriptors]")
{
    // A non-casting mesh must reach the shader as "no level", not as level 0 — the ShadowLod tint
    // greys it out instead of painting it full-detail green.
    DrawCommand dc{};
    CHECK(dc.shadowLodLevel == fire_engine::kNoShadowLod);
    CHECK(makeForwardPushConstants(dc).shadowLodLevel == fire_engine::kNoShadowLod);

    // The two levels are independent: a mesh drawn at forward level 0 can still cast a coarser
    // shadow, which is exactly the pairing the debug views are read against each other for.
    dc.lodLevel = 0u;
    dc.shadowLodLevel = 3u;
    const ForwardPushConstants pc = makeForwardPushConstants(dc);
    CHECK(pc.lodLevel == 0u);
    CHECK(pc.shadowLodLevel == 3u);
}

TEST_CASE("Descriptors.ForwardPushConstantsPreserveTheNoSelfShadowSentinel", "[Descriptors]")
{
    // -1 means "this draw has no self-shadow layer"; the shader branches on it, so it must survive
    // the packing rather than being clamped or reinterpreted as an unsigned slot.
    DrawCommand dc{};
    dc.selfShadowSlot = -1;

    CHECK(makeForwardPushConstants(dc).selfShadowSlot == -1);
}

TEST_CASE("Descriptors.ForwardPushConstantsAreAPureFunctionOfTheCommand", "[Descriptors]")
{
    // No hidden state: the same command packs identically however many times it is recorded, and
    // two commands differing only in LOD produce push constants differing only in lodLevel.
    DrawCommand a{};
    a.selfShadowSlot = 1;
    a.materialIndex = 9u;
    a.lodLevel = 0u;

    DrawCommand b = a;
    b.lodLevel = 4u;

    const ForwardPushConstants first = makeForwardPushConstants(a);
    const ForwardPushConstants again = makeForwardPushConstants(a);
    const ForwardPushConstants other = makeForwardPushConstants(b);

    CHECK(first.selfShadowSlot == again.selfShadowSlot);
    CHECK(first.materialIndex == again.materialIndex);
    CHECK(first.lodLevel == again.lodLevel);

    CHECK(other.selfShadowSlot == first.selfShadowSlot);
    CHECK(other.materialIndex == first.materialIndex);
    CHECK(other.lodLevel == 4u);
}
