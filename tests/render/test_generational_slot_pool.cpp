#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/render/generational_slot_pool.hpp>

using namespace fire_engine;

TEST_CASE("GenerationalSlotPool.AcquireGrowsSequentiallyFromZero", "[GenerationalSlotPool]")
{
    GenerationalSlotPool pool;
    const auto a = pool.acquire();
    const auto b = pool.acquire();
    const auto c = pool.acquire();

    CHECK(a.index == 0u);
    CHECK(b.index == 1u);
    CHECK(c.index == 2u);
    CHECK(a.generation == 0u);
    CHECK(pool.slotCount() == 3u);
    CHECK(pool.valid(a.index, a.generation));
    CHECK(pool.valid(b.index, b.generation));
}

TEST_CASE("GenerationalSlotPool.ReleaseRecyclesTheSlotAndBumpsGeneration", "[GenerationalSlotPool]")
{
    GenerationalSlotPool pool;
    const auto first = pool.acquire(); // index 0, gen 0
    pool.release(first.index);

    const auto reused = pool.acquire();
    // The freed index comes back, with an incremented generation.
    CHECK(reused.index == first.index);
    CHECK(reused.generation == first.generation + 1);
    // The table did not grow — the whole point of the free-list (CR-17).
    CHECK(pool.slotCount() == 1u);
}

TEST_CASE("GenerationalSlotPool.StaleHandleToRecycledSlotIsInvalid", "[GenerationalSlotPool]")
{
    GenerationalSlotPool pool;
    const auto original = pool.acquire();
    REQUIRE(pool.valid(original.index, original.generation));

    pool.release(original.index);
    // Released: the old (index, generation) no longer validates.
    CHECK_FALSE(pool.valid(original.index, original.generation));

    const auto reused = pool.acquire();
    // After reuse the slot is live again, but only for the new generation — the original
    // handle stays invalid rather than silently aliasing the new occupant.
    CHECK(pool.valid(reused.index, reused.generation));
    CHECK_FALSE(pool.valid(original.index, original.generation));
}

TEST_CASE("GenerationalSlotPool.RepeatedReleaseAcquireKeepsCountBounded", "[GenerationalSlotPool]")
{
    // Models the resize path: a fixed set of render-target slots released and recreated many
    // times must not grow the table without bound.
    GenerationalSlotPool pool;
    constexpr uint32_t kTargets = 8;
    std::vector<uint32_t> live;
    for (uint32_t i = 0; i < kTargets; ++i)
    {
        live.push_back(pool.acquire().index);
    }
    REQUIRE(pool.slotCount() == kTargets);

    for (int resize = 0; resize < 100; ++resize)
    {
        for (uint32_t index : live)
        {
            pool.release(index);
        }
        live.clear();
        for (uint32_t i = 0; i < kTargets; ++i)
        {
            live.push_back(pool.acquire().index);
        }
    }
    // 100 resizes later the table is still exactly the working-set size.
    CHECK(pool.slotCount() == kTargets);
}

TEST_CASE("GenerationalSlotPool.GenerationWrapsWithinEightBits", "[GenerationalSlotPool]")
{
    // Generation is masked to 8 bits; after 256 release/acquire cycles it wraps to 0. (A wrap
    // can only alias a handle that survived 256 recycles of the same slot unused — acceptable.)
    GenerationalSlotPool pool;
    auto slot = pool.acquire();
    for (int i = 0; i < 256; ++i)
    {
        pool.release(slot.index);
        slot = pool.acquire();
    }
    CHECK(slot.index == 0u);
    CHECK(slot.generation == 0u);
    CHECK(pool.slotCount() == 1u);
}
