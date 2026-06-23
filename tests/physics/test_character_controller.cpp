#include <fire_engine/physics/character_controller.hpp>

#include <algorithm>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/physics/physics_world.hpp>

using Catch::Approx;
using fire_engine::CharacterController;
using fire_engine::CharacterControllerConfig;
using fire_engine::CharacterMoveResult;
using fire_engine::ColliderDesc;
using fire_engine::PhysicsBodyDesc;
using fire_engine::PhysicsBodyHandle;
using fire_engine::PhysicsBodyType;
using fire_engine::PhysicsWorld;
using fire_engine::Quaternion;
using fire_engine::Vec3;
using BoxShape = fire_engine::BoxShape;

namespace
{

// A static box. Rotation is set before the collider is created so its world bounds /
// world shape are composed from the rotated transform.
void addStaticBox(PhysicsWorld& world, Vec3 position, Vec3 halfExtents,
                  Quaternion rotation = Quaternion::identity())
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    desc.position = position;
    desc.rotation = rotation;
    const PhysicsBodyHandle body = world.createBody(desc);
    (void)world.createCollider(body, ColliderDesc{.shape = BoxShape{halfExtents, Vec3{}}});
}

// Capsule-centre rest height on a floor whose top is at `floorTop`, for the default config
// (radius 0.35, height 1.8 → half-segment 0.55; bottom = centre − 0.9).
constexpr float kDefaultBottomOffset = 0.9f;

} // namespace

TEST_CASE("CharacterController.GroundedOnFloorSnaps", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f}); // top at y = 0.5

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.7f, 0.0f}};
    const CharacterMoveResult r = cc.move(world, {0.0f, -0.1f, 0.0f});

    CHECK(r.grounded);
    CHECK(r.groundNormal.y() == Approx(1.0f).margin(1e-2f));
    // Snapped to rest: bottom on the floor top (0.5) → centre 1.4.
    CHECK(r.position.y() == Approx(0.5f + kDefaultBottomOffset).margin(0.05f));
}

TEST_CASE("CharacterController.AirborneIsNotGrounded", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f});

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 10.0f, 0.0f}};
    const CharacterMoveResult r = cc.move(world, {0.0f, -0.1f, 0.0f});
    CHECK_FALSE(r.grounded);
}

TEST_CASE("CharacterController.BlockedByWall", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {2.0f, 0.0f, 0.0f}, {0.5f, 5.0f, 5.0f}); // near face at x = 1.5

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.0f, 0.0f}};
    (void)cc.move(world, {3.0f, 0.0f, 0.0f}); // walk hard into the wall

    // Stops with its radius (+ skin) short of the wall face, nowhere near x = 3.
    CHECK(cc.position().x() < 1.2f);
    CHECK(cc.position().x() > 1.0f);
}

TEST_CASE("CharacterController.SlidesAlongWall", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {2.0f, 0.0f, 0.0f}, {0.5f, 5.0f, 5.0f}); // wall facing -x at x = 1.5

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.0f, 0.0f}};
    (void)cc.move(world, {3.0f, 0.0f, 1.0f}); // into the wall, angled along +z

    CHECK(cc.position().x() < 1.2f); // blocked on x
    CHECK(cc.position().z() > 0.8f); // slid along z
}

TEST_CASE("CharacterController.StepsUpLowLedge", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f}); // floor, top 0.5
    addStaticBox(world, {2.0f, 0.5f, 0.0f}, {1.0f, 0.15f, 2.0f});  // ledge top at 0.5+0.15+0.5

    // Start grounded on the floor in front of the ledge; walk onto it (a handful of
    // moves — too many would walk off the far edge of the ledge).
    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.4f, 0.0f}};
    CharacterMoveResult r;
    for (int i = 0; i < 6; ++i)
    {
        r = cc.move(world, Vec3{0.3f, 0.0f, 0.0f} + Vec3{0.0f, -0.05f, 0.0f});
    }
    // Mounted the ledge: on top of it (x within [1,3]) and risen ~0.15 above the floor
    // rest height (1.4 → ~1.55).
    CHECK(r.grounded);
    CHECK(cc.position().x() > 1.4f);
    CHECK(cc.position().y() > 1.5f);
}

TEST_CASE("CharacterController.StepsUpLowLedgeAtWalkSpeed", "[CharacterController]")
{
    // The demo cadence: 3 m/s at 60 fps = 0.05 per frame, far less than the capsule radius
    // (0.35). A single lifted walk advances the centre only 0.05, so mounting a step takes a
    // run of frames during which the rounded capsule must *rest on the step's top edge* rather
    // than slide off it. (The existing StepsUpLowLedge test moves 0.3/frame, which mounts in
    // one lifted walk and so never exercises this.)
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f}); // floor, top 0.5
    // A wide ledge (near face x = 1) so the slow climb has room to mount and stay on top
    // rather than walking off a short top.
    addStaticBox(world, {4.0f, 0.65f, 0.0f}, {3.0f, 0.15f, 2.0f}); // top 0.8 (0.3 ≤ stepOffset)

    CharacterController cc{CharacterControllerConfig{},
                           Vec3{0.0f, 1.4f, 0.0f}}; // grounded on floor
    CharacterMoveResult r;
    bool grounded = true;
    for (int i = 0; i < 120; ++i)
    {
        // Gravity only while airborne (as the demo driver does); 0.05 forward per frame.
        const float vertical = grounded ? 0.0f : -0.05f;
        r = cc.move(world, Vec3{0.05f, vertical, 0.0f});
        grounded = r.grounded;
    }
    // Mounted the ledge: up onto it (x past the near face at 1.0) and risen ~0.3 to its rest
    // height (top 0.8 → centre 1.7).
    CHECK(r.grounded);
    CHECK(cc.position().x() > 1.2f);
    CHECK(cc.position().y() > 1.6f);
}

namespace
{

// Result of replaying the demo patrol over a course: the x/y extents reached, the longest run
// of frames with no net movement (a permanent wedge → never reaches the far extents), and the
// longest run of frames making no real *forward* progress while grounded (a temporary stall —
// e.g. catching on a stair edge — that a freeze metric alone would miss).
struct PatrolStats
{
    float minX{1e9f};
    float maxX{-1e9f};
    float maxY{-1e9f};
    int maxFrozenStreak{0};
    int maxStallStreak{0};
};

// Replays the demo's updateCharacter (gravity only while airborne, fixed-speed walk) for
// `seconds` at 60 fps. Turnaround mode: if `farBound > nearBound`, bounds-based like the demo
// (reverse only on the flat ends, so the only low-progress frames are real stair stalls);
// otherwise a *timed* 3 s flip (reverses at arbitrary points — a harsher no-wedge stress, but
// its wall pauses make the stall metric meaningless, so the timed form ignores it).
PatrolStats runPatrol(PhysicsWorld& world, Vec3 start, float seconds, float nearBound = 0.0f,
                      float farBound = 0.0f)
{
    constexpr float dt = 1.0f / 60.0f;
    constexpr float speed = 3.0f;
    constexpr float gravity = -9.8f;
    constexpr float period = 3.0f;
    const bool bounds = farBound > nearBound;
    const float stallThreshold = 0.4f * speed * dt; // <40% of walk speed = not really moving

    CharacterController cc{CharacterControllerConfig{}, start};
    PatrolStats s;
    Vec3 dir{1.0f, 0.0f, 0.0f};
    float timer = 0.0f;
    float vvel = 0.0f;
    bool grounded = true;
    Vec3 last = start;
    int frozen = 0;
    int stall = 0;

    const int frames = static_cast<int>(seconds / dt);
    for (int f = 0; f < frames; ++f)
    {
        if (bounds)
        {
            const float x = cc.position().x();
            if (x > farBound)
            {
                dir = Vec3{-1.0f, 0.0f, 0.0f};
            }
            else if (x < nearBound)
            {
                dir = Vec3{1.0f, 0.0f, 0.0f};
            }
        }
        else
        {
            timer += dt;
            if (timer >= period)
            {
                dir = dir * -1.0f;
                timer = 0.0f;
            }
        }
        if (grounded)
        {
            vvel = 0.0f;
        }
        else
        {
            vvel += gravity * dt;
        }
        const Vec3 disp = dir * (speed * dt) + Vec3{0.0f, grounded ? 0.0f : vvel * dt, 0.0f};
        const CharacterMoveResult r = cc.move(world, disp);

        if ((r.position - last).magnitude() < 1e-4f)
        {
            ++frozen;
        }
        else
        {
            frozen = 0;
        }
        s.maxFrozenStreak = std::max(s.maxFrozenStreak, frozen);

        // Forward progress along the walk direction. Only meaningful in bounds mode (timed mode
        // legitimately pauses against walls). Track it only once past the initial settle.
        if (bounds && f > 30)
        {
            const float forward = (r.position.x() - last.x()) * dir.x();
            stall = forward < stallThreshold && grounded ? stall + 1 : 0;
            s.maxStallStreak = std::max(s.maxStallStreak, stall);
        }

        grounded = r.grounded;
        last = r.position;
        if (f > 30) // skip the initial settle
        {
            s.minX = std::min(s.minX, r.position.x());
            s.maxX = std::max(s.maxX, r.position.x());
            s.maxY = std::max(s.maxY, r.position.y());
        }
    }
    return s;
}

} // namespace

TEST_CASE("CharacterController.PatrolsStepPyramidWithoutWedging", "[CharacterController]")
{
    // A step pyramid the patrol must climb up and down at walk speed, walled at both ends on
    // the floor, with the demo's bounds-based turnaround (reverse only on the flat ends). It
    // must traverse the whole course (no permanent wedge) AND keep moving over the steps (no
    // multi-frame stall catching on a stair edge).
    PhysicsWorld world;
    addStaticBox(world, {5.0f, -0.5f, 0.0f}, {10.0f, 0.5f, 5.0f}); // floor, top 0
    // Up the pyramid (each step rises 0.3 ≤ stepOffset 0.35).
    addStaticBox(world, {3.0f, 0.15f, 0.0f}, {0.5f, 0.15f, 3.0f}); // top 0.3
    addStaticBox(world, {4.0f, 0.30f, 0.0f}, {0.5f, 0.30f, 3.0f}); // top 0.6
    addStaticBox(world, {5.5f, 0.45f, 0.0f}, {1.0f, 0.45f, 3.0f}); // peak, top 0.9
    addStaticBox(world, {7.0f, 0.30f, 0.0f}, {0.5f, 0.30f, 3.0f}); // top 0.6
    addStaticBox(world, {8.0f, 0.15f, 0.0f}, {0.5f, 0.15f, 3.0f}); // top 0.3
    addStaticBox(world, {0.0f, 1.0f, 0.0f}, {0.4f, 1.5f, 3.0f});   // near wall
    addStaticBox(world, {11.0f, 1.0f, 0.0f}, {0.4f, 1.5f, 3.0f});  // far wall

    // Bounds-based turnaround on the flat floor past the pyramid (x≈2.5–8.5).
    const PatrolStats s = runPatrol(world, Vec3{1.5f, 0.9f, 0.0f}, 30.0f, 1.5f, 9.5f);

    INFO("minX=" << s.minX << " maxX=" << s.maxX << " maxY=" << s.maxY
                 << " maxFrozen=" << s.maxFrozenStreak << " maxStall=" << s.maxStallStreak);
    // Crossed the whole pyramid (the descending steps end at x = 8.5) and returned to the near
    // side, climbing the peak each way — a permanent wedge can do none of these.
    CHECK(s.maxX > 8.5f);
    CHECK(s.minX < 2.0f);
    CHECK(s.maxY > 1.7f); // peak top 0.9 → capsule centre ~1.8
    // And never caught on a stair edge for long: a step's rounded-capsule edge crossing is
    // ~radius/speed ≈ 0.12 s; a real wedge/stutter runs much longer.
    CHECK(s.maxStallStreak < 18); // < 0.3 s
}

TEST_CASE("CharacterController.PatrolsRampToWalledPlatformWithoutWedging", "[CharacterController]")
{
    // The elevated wall/ground corner case: the patrol walks up a ramp onto a raised platform
    // and into a wall standing on that platform. Previously the capsule jammed ungrounded in
    // the concave wall/platform corner and froze; here it must reach the wall, turn around on
    // the timer, and come back — so it never racks up a long frozen streak.
    PhysicsWorld world;
    addStaticBox(world, {5.0f, -0.5f, 0.0f}, {10.0f, 0.5f, 5.0f}); // floor, top 0
    const float a = 0.349f; // 20° ramp, well within slope limit
    const Quaternion tilt =
        Quaternion::fromVectors(Vec3{0.0f, 1.0f, 0.0f}, Vec3{-std::sin(a), std::cos(a), 0.0f});
    addStaticBox(world, {4.6f, 0.5f, 0.0f}, {1.8f, 0.1f, 3.0f}, tilt); // ramp rising toward +x
    addStaticBox(world, {8.0f, 0.55f, 0.0f}, {1.2f, 0.55f, 3.0f});     // platform, top 1.1
    addStaticBox(world, {0.0f, 1.5f, 0.0f}, {0.4f, 2.0f, 3.0f});       // near wall
    addStaticBox(world, {10.0f, 1.5f, 0.0f},
                 {0.4f, 2.0f, 3.0f}); // far wall, meets the platform top

    const PatrolStats s = runPatrol(world, Vec3{1.5f, 0.9f, 0.0f}, 40.0f);

    INFO("minX=" << s.minX << " maxX=" << s.maxX << " maxY=" << s.maxY
                 << " maxFrozen=" << s.maxFrozenStreak);
    CHECK(s.maxY > 1.9f);           // climbed onto the platform (top 1.1 → centre ~2.0)
    CHECK(s.maxX > 8.5f);           // reached the far wall on the platform
    CHECK(s.minX < 2.0f);           // and returned to the near side
    CHECK(s.maxFrozenStreak < 240); // no permanent wedge (a wall pause is ≤ the 3 s flip = 180)
}

TEST_CASE("CharacterController.WalksUpGentleRamp", "[CharacterController]")
{
    PhysicsWorld world;
    // ~30° ramp: tilt the box up axis toward (-sin, cos, 0) so +x is uphill.
    const float a = 0.5236f; // 30°
    const Quaternion tilt =
        Quaternion::fromVectors(Vec3{0.0f, 1.0f, 0.0f}, Vec3{-std::sin(a), std::cos(a), 0.0f});
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {6.0f, 0.25f, 6.0f}, tilt);

    // Start just above the ramp surface near the centre and settle briefly (gravity
    // slides a capsule down a tilted ramp, so a long settle would walk it off the edge).
    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.4f, 0.0f}};
    CharacterMoveResult r;
    for (int i = 0; i < 15; ++i)
    {
        r = cc.move(world, {0.0f, -0.1f, 0.0f});
    }
    REQUIRE(r.grounded);

    // Uphill = opposite the ground normal's horizontal lean.
    Vec3 uphill = Vec3{-r.groundNormal.x(), 0.0f, -r.groundNormal.z()};
    uphill = Vec3::normalise(uphill);
    const float startY = cc.position().y();
    for (int i = 0; i < 40; ++i)
    {
        (void)cc.move(world, uphill * 0.05f + Vec3{0.0f, -0.02f, 0.0f});
    }
    CHECK(cc.position().y() > startY + 0.1f); // climbed
}

TEST_CASE("CharacterController.SteepSlopeIsNotClimbed", "[CharacterController]")
{
    PhysicsWorld world;
    addStaticBox(world, {0.0f, 0.0f, 0.0f}, {10.0f, 0.5f, 10.0f}); // floor
    // ~70° face: too steep to walk up (normal.y ≈ 0.34 < maxSlopeCosine 0.64).
    const float a = 1.2217f; // 70°
    const Quaternion tilt =
        Quaternion::fromVectors(Vec3{0.0f, 1.0f, 0.0f}, Vec3{-std::sin(a), std::cos(a), 0.0f});
    addStaticBox(world, {3.0f, 2.0f, 0.0f}, {1.5f, 0.25f, 4.0f}, tilt);

    CharacterController cc{CharacterControllerConfig{}, Vec3{0.0f, 1.4f, 0.0f}};
    const float startY = cc.position().y();
    for (int i = 0; i < 40; ++i)
    {
        (void)cc.move(world, Vec3{0.2f, 0.0f, 0.0f} + Vec3{0.0f, -0.05f, 0.0f});
    }
    // It walked into the slope but did not climb it.
    CHECK(cc.position().y() < startY + 0.2f);
}
