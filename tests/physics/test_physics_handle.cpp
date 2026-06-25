#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/collision/collider_id.hpp>
#include <fire_engine/physics/physics_handle.hpp>

using fire_engine::ColliderId;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsColliderHandle;
using fire_engine::PhysicsConstraintHandle;

TEST_CASE("PhysicsHandle.DefaultIsInvalid", "[PhysicsHandle]")
{
    constexpr PhysicsBodyHandle handle;
    CHECK_FALSE(handle.valid());
    CHECK(handle.value() == 0U);
}

TEST_CASE("PhysicsHandle.NonZeroIsValid", "[PhysicsHandle]")
{
    constexpr PhysicsColliderHandle handle{7U};
    CHECK(handle.valid());
    CHECK(handle.value() == 7U);
}

TEST_CASE("PhysicsHandle.EqualityComparesValue", "[PhysicsHandle]")
{
    constexpr PhysicsConstraintHandle a{3U};
    constexpr PhysicsConstraintHandle b{3U};
    constexpr PhysicsConstraintHandle c{4U};

    CHECK(a == b);
    CHECK(a != c);
    static_assert(a == b);
    static_assert(a != c);
}

TEST_CASE("PhysicsHandle.ComparisonsAreConstexpr", "[PhysicsHandle]")
{
    constexpr PhysicsBodyHandle handle{1U};
    static_assert(handle.valid());
    static_assert(handle.value() == 1U);
}

TEST_CASE("ColliderId.DefaultedComparisonOrdersByValue", "[ColliderId]")
{
    constexpr ColliderId first{3U};
    constexpr ColliderId same{3U};
    constexpr ColliderId second{4U};

    static_assert(first == same);
    static_assert(first != second);
    static_assert(first < second);
    static_assert(second > first);

    CHECK(first == same);
    CHECK(first != second);
    CHECK(first < second);
    CHECK(second > first);
}
