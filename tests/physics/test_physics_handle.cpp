#include <gtest/gtest.h>

#include <fire_engine/physics/physics_handle.hpp>

using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsColliderHandle;
using fire_engine::PhysicsConstraintHandle;

TEST(PhysicsHandle, DefaultIsInvalid)
{
    constexpr PhysicsBodyHandle handle;
    EXPECT_FALSE(handle.valid());
    EXPECT_EQ(handle.value(), 0U);
}

TEST(PhysicsHandle, NonZeroIsValid)
{
    constexpr PhysicsColliderHandle handle{7U};
    EXPECT_TRUE(handle.valid());
    EXPECT_EQ(handle.value(), 7U);
}

TEST(PhysicsHandle, EqualityComparesValue)
{
    constexpr PhysicsConstraintHandle a{3U};
    constexpr PhysicsConstraintHandle b{3U};
    constexpr PhysicsConstraintHandle c{4U};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    static_assert(a == b);
    static_assert(a != c);
}

TEST(PhysicsHandle, ComparisonsAreConstexpr)
{
    constexpr PhysicsBodyHandle handle{1U};
    static_assert(handle.valid());
    static_assert(handle.value() == 1U);
}
