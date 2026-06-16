#include <fire_engine/core/system.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::SystemT;

namespace
{

struct MockBackend
{
    static inline bool initCalled = false;
    static inline bool destroyCalled = false;
    static inline double mockTime = 0.0;

    static void init()
    {
        initCalled = true;
    }

    static void destroy()
    {
        destroyCalled = true;
    }

    static double getTime()
    {
        return mockTime;
    }

    static void reset()
    {
        initCalled = false;
        destroyCalled = false;
        mockTime = 0.0;
    }
};

using MockSystem = SystemT<MockBackend>;

} // namespace

// ==========================================================================
// Delegation
// ==========================================================================

TEST_CASE("SystemT.InitDelegatesToBackend", "[SystemT]")
{
    MockBackend::reset();
    CHECK_FALSE(MockBackend::initCalled);
    MockSystem::init();
    CHECK(MockBackend::initCalled);
}

TEST_CASE("SystemT.DestroyDelegatesToBackend", "[SystemT]")
{
    MockBackend::reset();
    CHECK_FALSE(MockBackend::destroyCalled);
    MockSystem::destroy();
    CHECK(MockBackend::destroyCalled);
}

TEST_CASE("SystemT.GetTimeDelegatesToBackend", "[SystemT]")
{
    MockBackend::reset();
    MockBackend::mockTime = 42.5;
    CHECK(MockSystem::getTime() == Catch::Approx(42.5).margin(1e-5f));
}

TEST_CASE("SystemT.GetTimeReflectsChanges", "[SystemT]")
{
    MockBackend::reset();
    MockBackend::mockTime = 1.0;
    CHECK(MockSystem::getTime() == Catch::Approx(1.0).margin(1e-5f));
    MockBackend::mockTime = 2.0;
    CHECK(MockSystem::getTime() == Catch::Approx(2.0).margin(1e-5f));
}

// ==========================================================================
// Non-instantiable
// ==========================================================================

TEST_CASE("SystemT.IsNotInstantiable", "[SystemT]")
{
    static_assert(!std::is_default_constructible_v<MockSystem>);
}
