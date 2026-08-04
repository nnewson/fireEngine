#include <catch2/catch_test_macros.hpp>

#include <fire_engine/graphics/shadow_caster_bounds_frame.hpp>

using fire_engine::Bounds3;
using fire_engine::ShadowCasterBounds;
using fire_engine::ShadowCasterBoundsFrame;
using fire_engine::ShadowCasterBoundsKind;
using fire_engine::ShadowCasterGeneration;
using fire_engine::ShadowCasterId;
using fire_engine::Vec3;

namespace
{

[[nodiscard]] Bounds3 boxAt(float x)
{
    Bounds3 b{};
    b.expand(Vec3{x - 1.0f, -1.0f, -1.0f});
    b.expand(Vec3{x + 1.0f, 1.0f, 1.0f});
    return b;
}

[[nodiscard]] ShadowCasterBounds
caster(std::uint32_t id, float x, ShadowCasterBoundsKind kind = ShadowCasterBoundsKind::Exact,
       ShadowCasterGeneration generation = ShadowCasterGeneration::First)
{
    return ShadowCasterBounds{.world = boxAt(x),
                              .casterId = static_cast<ShadowCasterId>(id),
                              .generation = generation,
                              .kind = kind};
}

} // namespace

// The frame is the SH-06 authority on caster bounds: computed once per frame, read by the fit, the
// draw build and the diagnostics. Everything below is about that word "once" — the failure it
// replaces was a second, independent computation that produced a looser answer.
TEST_CASE("ShadowCasterBoundsFrame.KeepsEachCasterSeparateAndFindable", "[ShadowCasterBounds]")
{
    ShadowCasterBoundsFrame frame;
    frame.reset();
    frame.add(caster(7, 10.0f));
    frame.add(caster(9, -4.0f));

    REQUIRE(frame.size() == 2u);
    // Disjoint bindings stay disjoint. An object-wide union — what the draw path used to build —
    // would have handed both of these a box spanning from -5 to 11, containing space neither caster
    // occupies, and the depth fit would then be looser than the geometry justifies.
    const auto& first =
        frame.require(static_cast<ShadowCasterId>(7), ShadowCasterGeneration::First);
    const auto& second =
        frame.require(static_cast<ShadowCasterId>(9), ShadowCasterGeneration::First);
    CHECK(first.world.min.x() == 9.0f);
    CHECK(first.world.max.x() == 11.0f);
    CHECK(second.world.min.x() == -5.0f);
    CHECK(second.world.max.x() == -3.0f);
}

TEST_CASE("ShadowCasterBoundsFrame.GenerationIsPartOfTheIdentity", "[ShadowCasterBounds]")
{
    ShadowCasterBoundsFrame frame;
    frame.reset();
    frame.add(caster(3, 0.0f, ShadowCasterBoundsKind::Exact, ShadowCasterGeneration::First));
    // Same slot, next generation: a DIFFERENT caster, and it must not find the previous one's box.
    const auto nextGeneration = static_cast<ShadowCasterGeneration>(
        static_cast<std::uint32_t>(ShadowCasterGeneration::First) + 1);
    frame.add(caster(3, 20.0f, ShadowCasterBoundsKind::Exact, nextGeneration));

    CHECK(frame.size() == 2u);
    CHECK(frame.require(static_cast<ShadowCasterId>(3), ShadowCasterGeneration::First)
              .world.max.x() == 1.0f);
    CHECK(frame.require(static_cast<ShadowCasterId>(3), nextGeneration).world.max.x() == 21.0f);
}

// The key is a real pair, not two values shifted into one integer. Both halves are 64-bit, so any
// packing loses information — and a collision here is not a slow lookup, it is one caster silently
// receiving another's bounds and being fitted and culled against them.
TEST_CASE("ShadowCasterBoundsFrame.DistinctIdentitiesNeverCollide", "[ShadowCasterBounds]")
{
    ShadowCasterBoundsFrame frame;
    frame.reset();

    // The adversarial pair for a `(id << 32) | generation` packing: both collapse to
    // 0x0000'0003'0000'0000 under it — (2 << 32) | 2^32 and (3 << 32) | 0.
    const auto highGeneration = static_cast<ShadowCasterGeneration>(1ULL << 32);
    frame.add(caster(2, 1.0f, ShadowCasterBoundsKind::Exact, highGeneration));
    CHECK_NOTHROW(frame.add(caster(3, 50.0f)));
    REQUIRE(frame.size() == 2u);
    CHECK(frame.require(static_cast<ShadowCasterId>(2), highGeneration).world.max.x() == 2.0f);
    CHECK(frame.require(static_cast<ShadowCasterId>(3), ShadowCasterGeneration::First)
              .world.max.x() == 51.0f);

    // And the top half of an id must survive: two ids differing only above bit 32 are different
    // casters. A 32-bit shift would have truncated both to zero.
    frame.reset();
    const auto highIdA = static_cast<ShadowCasterId>(1ULL << 33);
    const auto highIdB = static_cast<ShadowCasterId>(1ULL << 34);
    ShadowCasterBounds a = caster(1, 5.0f);
    a.casterId = highIdA;
    ShadowCasterBounds b = caster(1, 90.0f);
    b.casterId = highIdB;
    frame.add(a);
    CHECK_NOTHROW(frame.add(b));
    REQUIRE(frame.size() == 2u);
    CHECK(frame.require(highIdA, ShadowCasterGeneration::First).world.max.x() == 6.0f);
    CHECK(frame.require(highIdB, ShadowCasterGeneration::First).world.max.x() == 91.0f);
}

TEST_CASE("ShadowCasterBoundsFrame.RejectsAmbiguityRatherThanResolvingIt", "[ShadowCasterBounds]")
{
    ShadowCasterBoundsFrame frame;
    frame.reset();
    frame.add(caster(5, 0.0f));

    SECTION("a duplicate key is terminal")
    {
        // Two bindings claiming one identity means the shadow state — hysteresis, drawn history,
        // and now bounds — is shared by casters that are not the same caster.
        CHECK_THROWS(frame.add(caster(5, 50.0f)));
    }
    SECTION("an invalid caster id is terminal")
    {
        ShadowCasterBounds nameless = caster(5, 0.0f);
        nameless.casterId = ShadowCasterId::Invalid;
        CHECK_THROWS(frame.add(nameless));
    }
    SECTION("a missing key is terminal, not an empty box")
    {
        // The dangerous alternative: returning a default Bounds3 would place the caster at the
        // origin, where it would be fitted and culled against geometry it has nothing to do with.
        CHECK_THROWS(frame.require(static_cast<ShadowCasterId>(6), ShadowCasterGeneration::First));
        // `find` is the non-terminal form, for diagnostics that may legitimately ask.
        CHECK(frame.find(static_cast<ShadowCasterId>(6), ShadowCasterGeneration::First) == nullptr);
    }
}

TEST_CASE("ShadowCasterBoundsFrame.RecordsCastersWhoseBoundsAreInvalid", "[ShadowCasterBounds]")
{
    // A casting binding with no vertices still gets an entry, so the recorded set matches the set
    // of shadow draws exactly and a lookup miss always means a real disagreement. Consumers skip
    // invalid bounds explicitly rather than finding them absent.
    ShadowCasterBoundsFrame frame;
    frame.reset();
    ShadowCasterBounds empty = caster(11, 0.0f);
    empty.world = Bounds3{};
    frame.add(empty);

    REQUIRE(frame.size() == 1u);
    const auto& recorded =
        frame.require(static_cast<ShadowCasterId>(11), ShadowCasterGeneration::First);
    CHECK_FALSE(recorded.world.valid);
}

TEST_CASE("ShadowCasterBoundsFrame.ResetEndsTheFrame", "[ShadowCasterBounds]")
{
    ShadowCasterBoundsFrame frame;
    frame.reset();
    frame.add(caster(1, 0.0f));
    frame.reset();

    CHECK(frame.empty());
    CHECK(frame.find(static_cast<ShadowCasterId>(1), ShadowCasterGeneration::First) == nullptr);
    // And the same id may be recorded again — a frame is not a registry, it is one frame.
    CHECK_NOTHROW(frame.add(caster(1, 4.0f)));
}
