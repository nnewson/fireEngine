#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <fire_engine/collision/sweep_and_prune_broad_phase.hpp>
#include <fire_engine/physics/physics_constants.hpp>
#include <fire_engine/physics/physics_world.hpp>

#include <catch2/catch_test_macros.hpp>

#include <support/state_hash.hpp>

// Determinism harness: the fixed-step solver must be a pure function of its
// initial state. These tests replay a scripted scene and assert (a) two runs are
// bit-identical, (b) free-fall integration matches the closed form, and (c) the
// end-state hash matches a recorded golden value (a tripwire for solver changes).

using namespace fire_engine;

namespace
{

constexpr float kFixedDt = 1.0f / 60.0f;
constexpr int kSteps = 120;

// World gravity default (PhysicsWorld::gravity_) — kept in sync for the
// closed-form free-fall check below.
constexpr float kGravityY = -9.81f;

// A fixed scene: a static floor + three dynamic boxes dropped with lateral
// velocity so they fall, bounce, and interact — exercising integrate →
// broadphase → narrowphase → response. Returns body handles in a stable order.
std::vector<PhysicsBodyHandle> buildScene(PhysicsWorld& world)
{
    std::vector<PhysicsBodyHandle> handles;

    auto add = [&](PhysicsBodyType type, Vec3 pos, Vec3 vel, const ColliderShape& shape)
    {
        PhysicsBodyDesc desc;
        desc.type = type;
        desc.position = pos;
        desc.linearVelocity = vel;
        desc.gravityScale = (type == PhysicsBodyType::Dynamic) ? 1.0f : 0.0f;
        desc.material = PhysicsMaterial{.restitution = 0.5f, .friction = 0.0f};
        const PhysicsBodyHandle body = world.createBody(desc);
        ColliderDesc collider;
        collider.shape = shape;
        [[maybe_unused]] const auto c = world.createCollider(body, collider);
        handles.push_back(body);
    };

    add(PhysicsBodyType::Static, {0.0f, 0.0f, 0.0f}, {},
        AabbShape{{Vec3{-10.0f, -1.0f, -10.0f}, Vec3{10.0f, 0.0f, 10.0f}}});
    add(PhysicsBodyType::Dynamic, {0.0f, 5.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        BoxShape{Vec3{0.5f, 0.5f, 0.5f}, {}});
    add(PhysicsBodyType::Dynamic, {2.0f, 6.0f, 0.0f}, {-1.0f, 0.0f, 0.0f},
        BoxShape{Vec3{0.5f, 0.5f, 0.5f}, {}});
    add(PhysicsBodyType::Dynamic, {-2.0f, 4.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
        BoxShape{Vec3{0.5f, 0.5f, 0.5f}, {}});
    return handles;
}

std::uint64_t simulate(PhysicsWorld& world, std::span<const PhysicsBodyHandle> handles, int steps)
{
    for (int i = 0; i < steps; ++i)
    {
        world.step(kFixedDt);
    }
    return test::hashBodyState(world, handles);
}

[[nodiscard]] constexpr std::uint64_t goldenHash() noexcept
{
    // Re-baselined for P9.6 (mid-step manifold refresh): the falling boxes exceed the
    // kSubstepRefreshRotation gate at impact, so their manifolds are re-collided mid-step
    // and the end state intentionally changed.
#if defined(__linux__) && defined(__x86_64__)
    // Recorded via the local Docker CI replica (tools/ci/run-local-ci.sh) — a platform's
    // golden must be recorded on that platform.
    return 0x640fc158fb6ac02bULL;
#else
    return 0x2b31386354735d8bULL;
#endif
}

} // namespace

TEST_CASE("Determinism.ReplayIsBitIdentical", "[Determinism]")
{
    // Two independent runs of the same scripted scene must produce bit-identical
    // state. This is the regression guard: if a future change introduces an
    // unordered-map iteration, an unstable sort, or order-dependent FP, this fails.
    PhysicsWorld a;
    const auto handlesA = buildScene(a);
    const std::uint64_t hashA = simulate(a, handlesA, kSteps);

    PhysicsWorld b;
    const auto handlesB = buildScene(b);
    const std::uint64_t hashB = simulate(b, handlesB, kSteps);

    CHECK(hashA == hashB);
}

TEST_CASE("Determinism.FreeFallMatchesClosedForm", "[Determinism]")
{
    // A single dynamic body with no collider free-falls under semi-implicit Euler.
    // The TGS solver substeps gravity, so over n fixed steps there are m = n·kSubstepCount
    // sub-integrations at h = dt/kSubstepCount: v_m = m·g·h (= n·g·dt, unchanged) and
    // y_m = y0 + h²·g·m(m+1)/2 (slightly less fall than the single-step form, as finer
    // integration is more accurate). A physical sanity check independent of the hash.
    PhysicsWorld world;
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = {0.0f, 100.0f, 0.0f};
    desc.gravityScale = 1.0f;
    const PhysicsBodyHandle body = world.createBody(desc);

    constexpr int n = 30;
    for (int i = 0; i < n; ++i)
    {
        world.step(kFixedDt);
    }

    constexpr int m = n * kSubstepCount;
    constexpr float h = kFixedDt / static_cast<float>(kSubstepCount);
    const float expectedY = 100.0f + h * h * kGravityY * static_cast<float>(m * (m + 1)) / 2.0f;
    const float expectedVy = static_cast<float>(n) * kGravityY * kFixedDt;

    const auto transform = world.bodyTransform(body);
    REQUIRE(transform.has_value());
    CHECK(transform->position().approxEqual(Vec3{0.0f, expectedY, 0.0f}, 1e-3f));
    CHECK(world.body(body)->linearVelocity().approxEqual(Vec3{0.0f, expectedVy, 0.0f}, 1e-3f));
}

TEST_CASE("Determinism.BroadphasesAgree", "[Determinism]")
{
    // Both broadphase implementations honour the same overlap+filter contract, so the
    // same scripted scene must end in the identical state regardless of which one finds
    // the pairs. This exercises the BroadPhase interface through PhysicsWorld (not just
    // the standalone impl tests) and guards against a broadphase that drops or mis-emits
    // pairs. Relies on PhysicsWorld::contacts() canonicalising pair order so the
    // order-dependent solver is independent of each broadphase's internal ordering.
    PhysicsWorld tree; // default = DynamicAabbTreeBroadPhase
    const auto handlesTree = buildScene(tree);
    const std::uint64_t hashTree = simulate(tree, handlesTree, kSteps);

    PhysicsWorld sap{std::make_unique<SweepAndPruneBroadPhase>()};
    const auto handlesSap = buildScene(sap);
    const std::uint64_t hashSap = simulate(sap, handlesSap, kSteps);

    CHECK(hashTree == hashSap);
}

TEST_CASE("Determinism.GoldenHash", "[Determinism]")
{
    // Behaviour tripwire: this hash captures the exact end-state of buildScene
    // after kSteps. It changes whenever the solver math changes — update it
    // INTENTIONALLY (and review why) when that happens, never reflexively.
    // Re-baselined for the P9.2 TGS soft-step solver (substepped gravity, soft contacts
    // + restitution at the true impact velocity) and the inelastic default material.
    //
    // The hash is raw float bits. Replays are bit-identical on a given platform, but
    // macOS/arm64 and Linux/x86_64 differ by a few last-bit contact-solver operations
    // after the first box/box impact, so each supported CI platform records its own
    // intentional golden.
    constexpr std::uint64_t kGoldenHash = goldenHash();

    PhysicsWorld world;
    const auto handles = buildScene(world);
    const std::uint64_t hash = simulate(world, handles, kSteps);

    INFO("actual end-state hash: 0x" << std::hex << hash);
    CHECK(hash == kGoldenHash);
}
