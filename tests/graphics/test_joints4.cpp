#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fire_engine/graphics/joints4.hpp>
#include <support/test_traits.hpp>

#include <cstdint>

using fire_engine::Joints4;

// Construction

TEST_CASE("Joints4Construction.DefaultConstruction", "[Joints4Construction]")
{
    constexpr Joints4 j;
    CHECK(j.j0() == 0u);
    CHECK(j.j1() == 0u);
    CHECK(j.j2() == 0u);
    CHECK(j.j3() == 0u);
}

TEST_CASE("Joints4Construction.ParameterisedConstruction", "[Joints4Construction]")
{
    constexpr Joints4 j{1, 2, 3, 4};
    CHECK(j.j0() == 1u);
    CHECK(j.j1() == 2u);
    CHECK(j.j2() == 3u);
    CHECK(j.j3() == 4u);
}

TEST_CASE("Joints4Construction.PartialConstruction", "[Joints4Construction]")
{
    constexpr Joints4 j{10, 20};
    CHECK(j.j0() == 10u);
    CHECK(j.j1() == 20u);
    CHECK(j.j2() == 0u);
    CHECK(j.j3() == 0u);
}

// Accessors

TEST_CASE("Joints4Accessors.SetJ0", "[Joints4Accessors]")
{
    Joints4 j;
    j.j0(42);
    CHECK(j.j0() == 42u);
}

TEST_CASE("Joints4Accessors.SetJ1", "[Joints4Accessors]")
{
    Joints4 j;
    j.j1(42);
    CHECK(j.j1() == 42u);
}

TEST_CASE("Joints4Accessors.SetJ2", "[Joints4Accessors]")
{
    Joints4 j;
    j.j2(42);
    CHECK(j.j2() == 42u);
}

TEST_CASE("Joints4Accessors.SetJ3", "[Joints4Accessors]")
{
    Joints4 j;
    j.j3(42);
    CHECK(j.j3() == 42u);
}

// Equality

TEST_CASE("Joints4Equality.Equal", "[Joints4Equality]")
{
    constexpr Joints4 a{1, 2, 3, 4};
    constexpr Joints4 b{1, 2, 3, 4};
    CHECK(a == b);
}

TEST_CASE("Joints4Equality.NotEqual", "[Joints4Equality]")
{
    constexpr Joints4 a{1, 2, 3, 4};
    constexpr Joints4 b{1, 2, 3, 5};
    CHECK(a != b);
}

TEST_CASE("Joints4Equality.DefaultsEqual", "[Joints4Equality]")
{
    constexpr Joints4 a;
    constexpr Joints4 b;
    CHECK(a == b);
}

// Copy and Move

TEST_CASE("Joints4Copy.CopyConstruction", "[Joints4Copy]")
{
    constexpr Joints4 original{10, 20, 30, 40};
    constexpr Joints4 copy(original);
    CHECK(copy == original);
}

TEST_CASE("Joints4Copy.CopyAssignment", "[Joints4Copy]")
{
    Joints4 a{10, 20, 30, 40};
    Joints4 b;
    b = a;
    CHECK(b == a);
}

TEST_CASE("Joints4Move.MoveConstruction", "[Joints4Move]")
{
    Joints4 original{10, 20, 30, 40};
    Joints4 moved(std::move(original));
    CHECK(moved == (Joints4{10, 20, 30, 40}));
}

TEST_CASE("Joints4Move.MoveAssignment", "[Joints4Move]")
{
    Joints4 original{10, 20, 30, 40};
    Joints4 target;
    target = std::move(original);
    CHECK(target == (Joints4{10, 20, 30, 40}));
}

// Noexcept

TEST_CASE("Joints4Noexcept.ConstructionIsNoexcept", "[Joints4Noexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<Joints4>);
    static_assert(
        test_traits::nothrow_constructible_from_v<Joints4, uint32_t, uint32_t, uint32_t, uint32_t>);
}

TEST_CASE("Joints4Noexcept.MoveIsNoexcept", "[Joints4Noexcept]")
{
    static_assert(std::is_nothrow_move_constructible_v<Joints4>);
    static_assert(std::is_nothrow_move_assignable_v<Joints4>);
}

TEST_CASE("Joints4Noexcept.GettersAreNoexcept", "[Joints4Noexcept]")
{
    static_assert(test_traits::has_nothrow_joints4_getters<Joints4>);
}

TEST_CASE("Joints4Noexcept.SettersAreNoexcept", "[Joints4Noexcept]")
{
    static_assert(test_traits::has_nothrow_joints4_setters<Joints4>);
}

// Constexpr

TEST_CASE("Joints4Constexpr.ConstexprConstruction", "[Joints4Constexpr]")
{
    constexpr Joints4 j{5, 10, 15, 20};
    static_assert(j.j0() == 5);
    static_assert(j.j1() == 10);
    static_assert(j.j2() == 15);
    static_assert(j.j3() == 20);
}

TEST_CASE("Joints4Constexpr.ConstexprEquality", "[Joints4Constexpr]")
{
    constexpr Joints4 a{1, 2, 3, 4};
    constexpr Joints4 b{1, 2, 3, 4};
    static_assert(a == b);
}

// Layout

TEST_CASE("Joints4Layout.SizeMatchesRawArray", "[Joints4Layout]")
{
    static_assert(sizeof(Joints4) == sizeof(uint32_t) * 4);
}
