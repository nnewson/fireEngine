#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include <fire_engine/graphics/texture.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST_CASE("Texture.DefaultNotLoaded", "[Texture]")
{
    Texture tex;
    CHECK_FALSE(tex.loaded());
}

TEST_CASE("Texture.DefaultHandleIsNull", "[Texture]")
{
    Texture tex;
    CHECK(tex.handle() == NullTexture);
}

TEST_CASE("Texture.DefaultEncodingIsSrgb", "[Texture]")
{
    Texture tex;
    CHECK(tex.encoding() == TextureEncoding::Srgb);
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

TEST_CASE("Texture.IsNonCopyable", "[Texture]")
{
    static_assert(!std::is_copy_constructible_v<Texture>);
    static_assert(!std::is_copy_assignable_v<Texture>);
}

TEST_CASE("Texture.IsNothrowMovable", "[Texture]")
{
    static_assert(std::is_nothrow_move_constructible_v<Texture>);
    static_assert(std::is_nothrow_move_assignable_v<Texture>);
}

TEST_CASE("Texture.MoveConstructionTransfersState", "[Texture]")
{
    Texture original;
    Texture moved(std::move(original));
    CHECK_FALSE(moved.loaded());
    CHECK(moved.handle() == NullTexture);
}

TEST_CASE("Texture.MoveAssignmentTransfersState", "[Texture]")
{
    Texture original;
    Texture target;
    target = std::move(original);
    CHECK_FALSE(target.loaded());
}
