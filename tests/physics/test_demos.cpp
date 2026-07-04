// Headless replay tests for the P8 physics demonstration scenes
// (assets/physics_demos/*.gltf). The .gltf assets are authored for *visual*
// verification, but the GUI can't be asserted in CI; these tests rebuild an
// equivalent PhysicsWorld with the same authored numbers, step the fixed-step
// solver, and assert the labelled behaviour actually happens (and stays a
// regression guard). They mirror the world-building pattern in
// test_physics_determinism.cpp.
//
// The authored numbers here are the shared source of truth with
// assets/physics_demos/generate.py — keep the two in sync when a demo changes.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include <fire_engine/collision/ray.hpp>
#include <fire_engine/core/convex_hull_builder.hpp>
#include <fire_engine/core/gltf_loader.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/physics/collider_shape.hpp>
#include <fire_engine/physics/physics_world.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace fire_engine;

namespace
{

constexpr float kFixedDt = 1.0f / 60.0f;

// A wide thin Static box whose top face sits at y = 0 (matches Scene.static_floor).
PhysicsBodyHandle addStaticFloor(PhysicsWorld& world, float halfXz = 5.0f, float friction = 0.5f,
                                 float thickness = 0.25f)
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    desc.position = {0.0f, -thickness, 0.0f};
    // Restitution 0: contact restitution combines as max(a, b), so the floor must
    // not impose bounce — the per-body restitution stays the controlling value.
    desc.material = PhysicsMaterial{.restitution = 0.0f, .friction = friction};
    const PhysicsBodyHandle body = world.createBody(desc);
    ColliderDesc collider;
    collider.shape = BoxShape{Vec3{halfXz, thickness, halfXz}, {}};
    collider.material = desc.material;
    static_cast<void>(world.createCollider(body, collider));
    return body;
}

PhysicsBodyHandle addDynamicBox(PhysicsWorld& world, Vec3 pos, Vec3 halfExtents, float restitution,
                                float friction, Vec3 velocity = {})
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = pos;
    desc.linearVelocity = velocity;
    desc.gravityScale = 1.0f;
    desc.mass = 1.0f;
    desc.material = PhysicsMaterial{.restitution = restitution, .friction = friction};
    const PhysicsBodyHandle body = world.createBody(desc);
    ColliderDesc collider;
    collider.shape = BoxShape{halfExtents, {}};
    collider.material = desc.material;
    static_cast<void>(world.createCollider(body, collider));
    return body;
}

PhysicsBodyHandle addDynamicSphere(PhysicsWorld& world, Vec3 pos, float radius, float restitution,
                                   float friction)
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = pos;
    desc.gravityScale = 1.0f;
    desc.mass = 1.0f;
    desc.material = PhysicsMaterial{.restitution = restitution, .friction = friction};
    const PhysicsBodyHandle body = world.createBody(desc);
    ColliderDesc collider;
    collider.shape = SphereShape{radius, {}};
    collider.material = desc.material;
    static_cast<void>(world.createCollider(body, collider));
    return body;
}

// A box (Static or Dynamic) at an explicit orientation — used for the tilted ramp
// and the boxes resting flush on it.
PhysicsBodyHandle addOrientedBox(PhysicsWorld& world, PhysicsBodyType type, Vec3 pos,
                                 Quaternion rotation, Vec3 halfExtents, float friction)
{
    PhysicsBodyDesc desc;
    desc.type = type;
    desc.position = pos;
    desc.rotation = rotation;
    desc.gravityScale = (type == PhysicsBodyType::Dynamic) ? 1.0f : 0.0f;
    desc.mass = 1.0f;
    desc.material = PhysicsMaterial{.restitution = 0.0f, .friction = friction};
    const PhysicsBodyHandle body = world.createBody(desc);
    ColliderDesc collider;
    collider.shape = BoxShape{halfExtents, {}};
    collider.material = desc.material;
    static_cast<void>(world.createCollider(body, collider));
    return body;
}

// Regular tetrahedron hull (matches tetrahedron_geometry in generate.py).
ConvexHullShape tetraHull(float s)
{
    const std::vector<Vec3> verts{{s, s, s}, {s, -s, -s}, {-s, s, -s}, {-s, -s, s}};
    const std::vector<std::uint32_t> idx{0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2};
    return buildConvexHull(verts, idx);
}

// Axis-angle quaternion (matches generate.py quat_axis_angle).
Quaternion axisAngle(Vec3 axis, float angle)
{
    const Vec3 a = Vec3::normalise(axis);
    const float s = std::sin(angle * 0.5f);
    return Quaternion{a.x() * s, a.y() * s, a.z() * s, std::cos(angle * 0.5f)};
}

PhysicsBodyHandle addConvexHull(PhysicsWorld& world, Vec3 pos, Quaternion rotation,
                                const ConvexHullShape& hull, float friction)
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = pos;
    desc.rotation = rotation;
    desc.gravityScale = 1.0f;
    desc.mass = 1.0f;
    desc.material = PhysicsMaterial{.restitution = 0.0f, .friction = friction};
    const PhysicsBodyHandle body = world.createBody(desc);
    ColliderDesc collider;
    collider.shape = hull;
    collider.material = desc.material;
    static_cast<void>(world.createCollider(body, collider));
    return body;
}

// The SleepDemo striker: a low-friction box that slides in along the floor (under
// gravity), bumps the stack awake, then friction stops it against the stack.
PhysicsBodyHandle addStriker(PhysicsWorld& world, Vec3 pos, Vec3 velocity)
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = pos;
    desc.linearVelocity = velocity;
    desc.gravityScale = 1.0f;
    desc.mass = 2.0f;
    desc.material = PhysicsMaterial{.restitution = 0.1f, .friction = 0.1f};
    const PhysicsBodyHandle body = world.createBody(desc);
    ColliderDesc collider;
    collider.shape = BoxShape{Vec3{0.5f, 0.5f, 0.5f}, {}};
    collider.material = desc.material;
    static_cast<void>(world.createCollider(body, collider));
    return body;
}

// The trapezoidal valley mesh (matches _VALLEY_VERTS/_VALLEY_TRIS in generate.py).
StaticMeshShape valleyMesh()
{
    StaticMeshShape m;
    m.vertices = {{-7.0f, 1.5f, -5.0f}, {7.0f, 1.5f, -5.0f}, {-7.0f, 0.0f, -3.0f},
                  {7.0f, 0.0f, -3.0f},  {-7.0f, 0.0f, 3.0f}, {7.0f, 0.0f, 3.0f},
                  {-7.0f, 1.5f, 5.0f},  {7.0f, 1.5f, 5.0f}};
    m.indices = {2, 3, 0, 0, 3, 1, 4, 5, 2, 2, 5, 3, 5, 4, 6, 5, 6, 7};
    return m;
}

PhysicsBodyHandle addStaticMesh(PhysicsWorld& world, const StaticMeshShape& mesh, float friction)
{
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Static;
    const PhysicsBodyHandle body = world.createBody(desc);
    static_cast<void>(world.createMeshCollider(
        body, mesh, PhysicsMaterial{.restitution = 0.0f, .friction = friction}));
    return body;
}

CompoundChild boxChild(Vec3 halfExtents, Vec3 position)
{
    CompoundChild child;
    child.shape = BoxShape{halfExtents, Vec3{}};
    child.localPosition = position;
    return child;
}

void step(PhysicsWorld& world, int steps)
{
    for (int i = 0; i < steps; ++i)
    {
        world.step(kFixedDt);
    }
}

// Peak height a sphere of the given restitution reaches after its first bounce
// (drops from y = 2 onto the floor at y = 0; rest height = radius = 0.5). A taller
// drop saturates: the speculative-margin CCD brakes the fast approach and suppresses
// the bounce, so y = 2 keeps the spheres in the clean restitution regime.
float reboundApex(float restitution)
{
    PhysicsWorld world;
    addStaticFloor(world);
    const PhysicsBodyHandle sphere =
        addDynamicSphere(world, {0.0f, 2.0f, 0.0f}, 0.5f, restitution, 0.3f);
    bool contacted = false;
    float apex = 0.0f;
    for (int i = 0; i < 300; ++i)
    {
        world.step(kFixedDt);
        const float y = world.bodyTransform(sphere)->position().y();
        if (!contacted && y <= 0.56f) // reached the floor
        {
            contacted = true;
        }
        if (contacted)
        {
            apex = std::max(apex, y);
        }
    }
    return apex;
}

} // namespace

TEST_CASE("Demos.FallRest.BoxComesToRestOnFloor", "[Demos][slow]")
{
    // FallRestDemo.gltf: a single Dynamic box dropped onto a Static floor should
    // fall, settle, and come to rest with its bottom face on the floor (centre at
    // y = half-extent) and effectively zero velocity. This is the end-to-end smoke
    // test for the whole author -> simulate pipeline.
    PhysicsWorld world;
    addStaticFloor(world);
    const PhysicsBodyHandle box =
        addDynamicBox(world, {0.0f, 2.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, 0.0f, 0.5f);

    step(world, 240); // 4 s — falls ~1.5 m and settles

    const float restY = world.bodyTransform(box)->position().y();
    const Vec3 vel = world.body(box)->linearVelocity();

    CHECK(restY == Catch::Approx(0.5f).margin(0.05f));
    CHECK(std::abs(vel.x()) < 0.05f);
    CHECK(std::abs(vel.y()) < 0.05f);
    CHECK(std::abs(vel.z()) < 0.05f);
}

TEST_CASE("Demos.Restitution.HigherRestitutionBouncesHigher", "[Demos][slow]")
{
    // RestitutionDemo.gltf: three spheres dropped from the same height bounce to
    // rebound heights ordered by restitution (rebound ~ restitution^2 * drop). The
    // restitution-0 sphere does not bounce; the 0.9 sphere bounces lively.
    const float apex00 = reboundApex(0.0f);
    const float apex05 = reboundApex(0.5f);
    const float apex09 = reboundApex(0.9f);

    // No real bounce for restitution 0 (stays near its rest height ~0.5).
    CHECK(apex00 < 0.62f);
    // Strictly ordered, with clear separation between each.
    CHECK(apex05 > apex00 + 0.12f);
    CHECK(apex09 > apex05 + 0.25f);
    // The lively sphere bounces well clear of the floor.
    CHECK(apex09 > 1.0f);
}

TEST_CASE("Demos.Friction.HighFrictionStaysLowFrictionSlides", "[Demos][slow]")
{
    // FrictionRampDemo.gltf: two boxes on a 25-degree ramp. Combined friction is
    // sqrt(a*b); the box slides when the slope exceeds atan(mu). The sticky box
    // (mu 1.0) holds; the slippery box (mu 0.02) slides down the slope.
    PhysicsWorld world;
    addStaticFloor(world, 8.0f, 0.9f); // rough floor to stop the slippery box
    const float angle = 25.0f * std::numbers::pi_v<float> / 180.0f;
    const Quaternion rampRot{0.0f, 0.0f, std::sin(angle * 0.5f), std::cos(angle * 0.5f)};
    const Vec3 rampHalf{4.0f, 0.15f, 2.0f};
    const Vec3 rampPos{0.0f, 2.5f, 0.0f};
    addOrientedBox(world, PhysicsBodyType::Static, rampPos, rampRot, rampHalf, 1.0f);

    const Vec3 boxHalf{0.4f, 0.4f, 0.4f};
    const float topY = rampHalf.y() + boxHalf.y() + 0.02f;
    auto surfaceWorld = [&](Vec3 local) { return rampPos + rampRot.rotate(local); };

    const Vec3 stickyStart = surfaceWorld({2.0f, topY, -1.0f});
    const Vec3 slipperyStart = surfaceWorld({2.0f, topY, 1.0f});
    const PhysicsBodyHandle sticky =
        addOrientedBox(world, PhysicsBodyType::Dynamic, stickyStart, rampRot, boxHalf, 1.0f);
    const PhysicsBodyHandle slippery =
        addOrientedBox(world, PhysicsBodyType::Dynamic, slipperyStart, rampRot, boxHalf, 0.08f);

    step(world, 300); // 5 s — slide off, fall, settle on the floor

    // Down-slope direction in world space (ramp local -x).
    const Vec3 downSlope = rampRot.rotate(Vec3{-1.0f, 0.0f, 0.0f});
    const auto slid = [&](PhysicsBodyHandle h, Vec3 start)
    {
        const Vec3 delta = world.bodyTransform(h)->position() - start;
        return Vec3::dotProduct(delta, downSlope);
    };

    CHECK(slid(sticky, stickyStart) < 0.2f);     // holds position on the ramp
    CHECK(slid(slippery, slipperyStart) > 1.0f); // slid well down off the ramp

    // The slippery box grinds to a halt resting on the floor (box half 0.4), not
    // sliding off its far edge (half extent 8).
    const Vec3 slipPos = world.bodyTransform(slippery)->position();
    const Vec3 slipVel = world.body(slippery)->linearVelocity();
    CHECK(slipPos.y() == Catch::Approx(0.4f).margin(0.15f));
    CHECK(std::abs(slipPos.x()) < 7.5f);
    CHECK(Vec3::dotProduct(slipVel, slipVel) < 0.09f); // |v| < 0.3
}

TEST_CASE("Demos.Stack.SettlesAndStaysStill", "[Demos][slow]")
{
    // StackDemo.gltf: a 5-high tower of boxes dropped with small gaps settles into a
    // resting stack at centres 0.5, 1.5, 2.5, 3.5, 4.5 and stays still (no collapse, no
    // drift) rather than buzzing apart. Five, not three: the TGS soft-step solver (P9.2)
    // propagates the settle down the stack through its substeps + relax pass and quiesces
    // a tall tower cleanly, where the old fixed-iteration solver shuffled for seconds and
    // diverged taller (so this demo was capped at three until P9).
    constexpr int n = 5;
    PhysicsWorld world;
    addStaticFloor(world, 6.0f);
    std::array<PhysicsBodyHandle, n> boxes{};
    for (int i = 0; i < n; ++i)
    {
        boxes[static_cast<std::size_t>(i)] =
            addDynamicBox(world, {0.0f, 0.55f + static_cast<float>(i) * 1.05f, 0.0f},
                          {0.5f, 0.5f, 0.5f}, 0.0f, 0.5f);
    }

    step(world, 240); // 4 s — fall, settle, sleep

    for (std::size_t k = 0; k < boxes.size(); ++k)
    {
        const Vec3 pos = world.bodyTransform(boxes[k])->position();
        const Vec3 vel = world.body(boxes[k])->linearVelocity();
        // Each box rests at its tier height and did not topple sideways.
        CHECK(pos.y() == Catch::Approx(0.5f + static_cast<float>(k)).margin(0.12f));
        CHECK(std::abs(pos.x()) < 0.2f);
        CHECK(std::abs(pos.z()) < 0.2f);
        CHECK(Vec3::dotProduct(vel, vel) < 0.0025f); // |v| < 0.05 — fully settled
    }
}

TEST_CASE("Demos.Topple.TallBoxTopplesOntoSide", "[Demos][slow]")
{
    // ToppleDemo.gltf: a tall box (half 0.3 x 1.0 x 0.3) tilted 30 deg — past its
    // ~16.7 deg balance angle — topples onto its long side (local +y ends up
    // horizontal) and comes to rest. The P3 rotational-dynamics headline.
    PhysicsWorld world;
    addStaticFloor(world, 6.0f, 0.6f);
    const float tilt = 30.0f * std::numbers::pi_v<float> / 180.0f;
    const Quaternion rotation{0.0f, 0.0f, std::sin(tilt * 0.5f), std::cos(tilt * 0.5f)};
    const PhysicsBodyHandle box = addOrientedBox(
        world, PhysicsBodyType::Dynamic, {0.0f, 2.0f, 0.0f}, rotation, {0.3f, 1.0f, 0.3f}, 0.6f);

    step(world, 360); // 6 s

    const auto t = world.bodyTransform(box);
    const Vec3 up = t->rotation().rotate(Vec3{0.0f, 1.0f, 0.0f});
    CHECK(std::abs(up.y()) < 0.2f); // local +y now ~horizontal → toppled onto its side
    CHECK(world.body(box)->angularVelocity().approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 0.1f));
}

TEST_CASE("Demos.ConvexHull.PileSettlesAtRest", "[Demos][slow]")
{
    // ConvexHullDemo.gltf: tetrahedra (built as ConvexHullShape from their mesh)
    // dropped onto the floor tumble through the GJK/EPA convex narrowphase, land on a
    // face, and come to rest — finite, settled, sitting on the floor (not exploded or
    // sunk through). Spread out in x so they rest mostly side by side.
    PhysicsWorld world;
    addStaticFloor(world, 6.0f);
    const ConvexHullShape hull = tetraHull(0.6f);
    const std::array<PhysicsBodyHandle, 3> tetra{
        addConvexHull(world, {-1.6f, 2.0f, 0.2f}, axisAngle({1.0f, 0.0f, 0.0f}, 0.3f), hull, 0.5f),
        addConvexHull(world, {0.0f, 3.0f, -0.2f}, axisAngle({0.0f, 0.0f, 1.0f}, 0.5f), hull, 0.5f),
        addConvexHull(world, {1.6f, 2.4f, 0.3f}, axisAngle({1.0f, 1.0f, 0.0f}, 0.4f), hull, 0.5f),
    };

    step(world, 480); // 8 s — tumble, land, settle

    for (const PhysicsBodyHandle h : tetra)
    {
        const Vec3 pos = world.bodyTransform(h)->position();
        const Vec3 vel = world.body(h)->linearVelocity();
        CHECK(std::isfinite(pos.x()));
        CHECK(std::isfinite(pos.y()));
        CHECK(std::isfinite(pos.z()));
        CHECK(pos.y() > 0.15f);                    // resting on the floor, not sunk through
        CHECK(pos.y() < 1.0f);                     // not perched on a tall pile / exploded
        CHECK(Vec3::dotProduct(vel, vel) < 0.04f); // |v| < 0.2 — settled
    }
}

TEST_CASE("Demos.Sleep.StackSleepsThenWakesOnImpact", "[Demos][slow]")
{
    // SleepDemo.gltf: a 3-box stack settles and the island sleeps; a striker slides
    // in along the floor, wakes it on impact (~step 103), then friction stops the
    // striker against the stack so the whole scene ends asleep on the floor. Verifies
    // the full sleep -> wake -> re-sleep cycle, including the striker coming to rest.
    PhysicsWorld world;
    addStaticFloor(world, 8.0f);
    std::array<PhysicsBodyHandle, 3> stack{};
    for (int i = 0; i < 3; ++i)
    {
        stack[static_cast<std::size_t>(i)] =
            addDynamicBox(world, {0.0f, 0.55f + static_cast<float>(i) * 1.05f, 0.0f},
                          {0.5f, 0.5f, 0.5f}, 0.0f, 0.5f);
    }
    const PhysicsBodyHandle striker = addStriker(world, {-8.0f, 0.5f, 0.0f}, {6.0f, 0.0f, 0.0f});

    // Before the striker arrives, the stack has settled and slept (striker still moving).
    step(world, 90);
    for (const PhysicsBodyHandle h : stack)
    {
        CHECK(world.sleeping(h));
    }
    CHECK_FALSE(world.sleeping(striker));

    // The striker bumps the stack awake.
    step(world, 60); // -> step 150
    bool anyAwake = false;
    for (const PhysicsBodyHandle h : stack)
    {
        anyAwake = anyAwake || !world.sleeping(h);
    }
    CHECK(anyAwake);

    // Clean end state: the disturbance damps out and everything — stack and striker —
    // comes to rest and sleeps on the floor.
    step(world, 450); // -> step 600
    for (const PhysicsBodyHandle h : stack)
    {
        CHECK(world.sleeping(h));
    }
    CHECK(world.sleeping(striker));
}

TEST_CASE("Demos.StaticMesh.BodiesSettleInValley", "[Demos][slow]")
{
    // StaticMeshDemo.gltf: boxes + a sphere dropped onto a triangulated valley (a
    // Static triangle-mesh collider, not a box) land on the flat bottom (y = 0) and
    // settle — proving contacts against the mesh's actual triangles.
    PhysicsWorld world;
    addStaticMesh(world, valleyMesh(), 0.6f);
    const std::array<PhysicsBodyHandle, 3> bodies{
        addDynamicBox(world, {-3.0f, 1.3f, -1.5f}, {0.4f, 0.4f, 0.4f}, 0.0f, 0.5f),
        addDynamicBox(world, {3.0f, 1.3f, 1.5f}, {0.4f, 0.4f, 0.4f}, 0.0f, 0.5f),
        addDynamicSphere(world, {0.0f, 1.3f, 0.0f}, 0.4f, 0.1f, 0.4f),
    };

    step(world, 600); // 10 s — drop onto the mesh surface and settle

    for (const PhysicsBodyHandle h : bodies)
    {
        const Vec3 pos = world.bodyTransform(h)->position();
        const Vec3 vel = world.body(h)->linearVelocity();
        CHECK(std::isfinite(pos.y()));
        CHECK(pos.y() == Catch::Approx(0.4f).margin(0.12f)); // resting on the mesh (half 0.4)
        CHECK(std::abs(pos.z()) < 3.0f);                     // on the flat bottom
        CHECK(Vec3::dotProduct(vel, vel) < 0.04f);           // |v| < 0.2 — at rest
    }
}

TEST_CASE("Demos.Compound.LShapeRestsOnFloor", "[Demos][slow]")
{
    // CompoundDemo.gltf: an L-shaped compound (bar + upright) has its centre of mass
    // offset toward the corner (engine-aggregated, volume-weighted), so it rests
    // stably flat on its bar instead of tipping.
    PhysicsWorld world;
    addStaticFloor(world, 6.0f);
    PhysicsBodyDesc desc;
    desc.type = PhysicsBodyType::Dynamic;
    desc.position = {0.0f, 2.0f, 0.0f};
    desc.gravityScale = 1.0f;
    desc.mass = 3.0f;
    desc.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
    const PhysicsBodyHandle body = world.createBody(desc);
    const std::vector<CompoundChild> children{
        boxChild({1.2f, 0.4f, 0.4f}, {0.0f, 0.0f, 0.0f}),
        boxChild({0.4f, 1.0f, 0.4f}, {-0.8f, 1.0f, 0.0f}),
    };
    static_cast<void>(world.createCompoundCollider(body, children));

    // The aggregated COM is offset toward the corner (−x, +y), not the body origin.
    const Vec3 com = world.body(body)->centerOfMassLocal();
    CHECK(com.x() < -0.2f);
    CHECK(com.y() > 0.2f);

    step(world, 300); // 5 s — fall and settle on the bar

    const auto t = world.bodyTransform(body).value();
    CHECK(t.position().y() == Catch::Approx(0.4f).margin(0.08f));  // bar (half 0.4) on the floor
    CHECK(t.rotation().approxEqual(Quaternion::identity(), 0.1f)); // upright, did not tip
    CHECK(world.sleeping(body));
}

TEST_CASE("Demos.Query.RaycastAndOverlapFindBodies", "[Demos][slow]")
{
    // The -q query-probe demo casts raycasts (a rotating fan) and an overlap sphere at
    // a ring of static bodies. This verifies the underlying queries: a ray hits the
    // body in its path (at the surface) and misses through a gap, and overlapSphere
    // finds the bodies within range.
    PhysicsWorld world;
    const auto addStaticSphere = [&](Vec3 center, float radius)
    {
        PhysicsBodyDesc desc;
        desc.type = PhysicsBodyType::Static;
        desc.position = center;
        const PhysicsBodyHandle body = world.createBody(desc);
        static_cast<void>(
            world.createCollider(body, ColliderDesc{.shape = SphereShape{radius, {}}}));
    };
    addStaticSphere({4.0f, 0.7f, 0.0f}, 0.5f); // +x
    addStaticSphere({0.0f, 0.7f, 4.0f}, 0.5f); // +z

    const Vec3 origin{0.0f, 0.7f, 0.0f};
    // A ray toward +x hits the first sphere at its near surface (x = 4 − 0.5 = 3.5).
    const auto hit = world.raycast(Ray{origin, {1.0f, 0.0f, 0.0f}, 8.0f});
    REQUIRE(hit.has_value());
    CHECK(hit->distance == Catch::Approx(3.5f).margin(0.05f));
    CHECK(hit->point.x() == Catch::Approx(3.5f).margin(0.05f));
    // A ray into a gap (−x) misses.
    CHECK_FALSE(world.raycast(Ray{origin, {-1.0f, 0.0f, 0.0f}, 8.0f}).has_value());
    // overlapSphere finds both bodies when its radius reaches the ring, none when tiny.
    CHECK(world.overlapSphere(origin, 5.0f).size() == 2);
    CHECK(world.overlapSphere(origin, 1.0f).empty());
}

// Diagnostic probe (hidden): replay ConvexHullDemo from the ACTUAL gltf (fastgltf TRS +
// GltfLoader::meshConvexHull — the loader's own hull path, bit-identical inputs) and trace the
// orange tetra (Tetra0)'s yaw / |w| / up-axis / sleep per step, to explain the reported
// "snap rotation about Y just before sleeping".
TEST_CASE("Demos.ConvexHull.Tetra0YawProbe", "[.][ConvexProbe]")
{
    fastgltf::Parser parser;
    auto data = fastgltf::GltfDataBuffer::FromPath("../assets/physics_demos/ConvexHullDemo.gltf");
    REQUIRE(data.error() == fastgltf::Error::None);
    auto result = parser.loadGltf(data.get(), "../assets/physics_demos",
                                  fastgltf::Options::LoadExternalBuffers);
    REQUIRE(result.error() == fastgltf::Error::None);
    const fastgltf::Asset& asset = result.get();

    PhysicsWorld world;
    addStaticFloor(world, 6.0f, 0.5f); // node 0: authored Static box, top at y=0, friction .5

    std::vector<PhysicsBodyHandle> tetras;
    for (std::size_t ni = 1; ni <= 3; ++ni) // Tetra0..2
    {
        const fastgltf::Node& n = asset.nodes[ni];
        const auto* trs = std::get_if<fastgltf::TRS>(&n.transform);
        REQUIRE(trs != nullptr);
        PhysicsBodyDesc desc;
        desc.type = PhysicsBodyType::Dynamic;
        desc.position = {trs->translation.x(), trs->translation.y(), trs->translation.z()};
        desc.rotation = {trs->rotation.x(), trs->rotation.y(), trs->rotation.z(),
                         trs->rotation.w()};
        desc.mass = 1.0f;
        desc.gravityScale = 1.0f;
        desc.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
        const PhysicsBodyHandle body = world.createBody(desc);
        ColliderDesc collider;
        collider.shape = GltfLoader::meshConvexHull(asset, asset.meshes[n.meshIndex.value()]);
        collider.material = desc.material;
        static_cast<void>(world.createCollider(body, collider));
        tetras.push_back(body);
    }

    const PhysicsBodyHandle orange = tetras[0]; // Tetra0 = Mat1 (orange)
    float prevYaw = 0.0f;
    bool prevAsleep = false;
    Quaternion prevQ = Quaternion::identity();
    float twistAccum = 0.0f;
    for (int i = 0; i < 900; ++i)
    {
        world.step(kFixedDt);
        const auto t = world.bodyTransform(orange);
        REQUIRE(t.has_value());
        const Quaternion q = t->rotation();
        const Vec3 xw = q.rotate(Vec3{1.0f, 0.0f, 0.0f});
        const Vec3 up = q.rotate(Vec3{0.0f, 1.0f, 0.0f});
        const float yaw = std::atan2(xw.z(), xw.x()) * 180.0f / std::numbers::pi_v<float>;
        const float w = world.body(orange)->angularVelocity().magnitude();
        const bool asleep = world.sleeping(orange);
        float dyaw = yaw - prevYaw;
        if (dyaw > 180.0f)
        {
            dyaw -= 360.0f;
        }
        if (dyaw < -180.0f)
        {
            dyaw += 360.0f;
        }
        // Contacts on Tetra0 this step (position near the body's XZ, i.e. around x=-1.6).
        int contacts = 0;
        for (const DebugContact& c : world.debugContacts())
        {
            if (std::abs(c.point.x() - t->position().x()) < 0.8f &&
                std::abs(c.point.z() - t->position().z()) < 0.8f)
            {
                ++contacts;
            }
        }
        const float wy = world.body(orange)->angularVelocity().y();
        // True rotation about world Y this step: twist decomposition of the delta rotation.
        const Quaternion dq = Quaternion::normalise(q * prevQ.conjugate());
        const float twistDeg =
            2.0f * std::atan2(dq.y(), dq.w()) * 180.0f / std::numbers::pi_v<float>;
        if (i > 40)
        {
            twistAccum += twistDeg;
        }
        prevQ = q;
        // Print the settle window in full detail, plus periodic lines and the sleep transition.
        if ((i >= 40 && i <= 70) || i % 30 == 0 || asleep != prevAsleep)
        {
            std::printf("[t0] step %3d yaw=%8.2f dyaw=%7.2f twist=%7.2f acc=%7.2f |w|=%6.3f "
                        "wy=%7.3f upY=%5.2f y=%6.3f c=%d%s\n",
                        i, static_cast<double>(yaw), static_cast<double>(dyaw),
                        static_cast<double>(twistDeg), static_cast<double>(twistAccum),
                        static_cast<double>(w), static_cast<double>(wy),
                        static_cast<double>(up.y()), static_cast<double>(t->position().y()),
                        contacts, asleep ? "  <SLEEP>" : "");
        }
        prevYaw = yaw;
        prevAsleep = asleep;
    }
}

// P9.6 gate: the ConvexHullDemo tetras' settle must not "snap" about the vertical. Before the
// mid-step manifold refresh, the orange tetra's final tip-onto-face slammed down at ~2.9 rad/s
// onto a step-start (stale) manifold and picked up 10.5 deg of world-Y rotation inside a single
// 16 ms step; with the refresh the same landing measures ~0.5 deg/step. The 3-degree bound
// separates the failure mode (~10 deg) from healthy behaviour (<1 deg) with cross-platform
// headroom (the settle is chaotic; macOS/arm64 and Linux/x86_64 trajectories diverge in FP
// last-bits). Replays the ACTUAL gltf (fastgltf TRS + the loader's own hull path) so it tests
// what the app runs; the hidden [ConvexProbe] variant prints the full per-step trace.
// The scene provably exercises the refresh path (its bodies exceed kSubstepRefreshRotation at
// impact), so the bit-identical replay check also guards the refresh path's determinism.
TEST_CASE("Demos.ConvexHull.TetraSettleTwistBounded", "[Demos]")
{
    fastgltf::Parser parser;
    auto data = fastgltf::GltfDataBuffer::FromPath("../assets/physics_demos/ConvexHullDemo.gltf");
    REQUIRE(data.error() == fastgltf::Error::None);
    auto result = parser.loadGltf(data.get(), "../assets/physics_demos",
                                  fastgltf::Options::LoadExternalBuffers);
    REQUIRE(result.error() == fastgltf::Error::None);
    const fastgltf::Asset& asset = result.get();

    const auto buildWorld = [&](PhysicsWorld& world)
    {
        addStaticFloor(world, 6.0f, 0.5f);
        std::vector<PhysicsBodyHandle> tetras;
        for (std::size_t ni = 1; ni <= 3; ++ni)
        {
            const fastgltf::Node& n = asset.nodes[ni];
            const auto* trs = std::get_if<fastgltf::TRS>(&n.transform);
            REQUIRE(trs != nullptr);
            PhysicsBodyDesc desc;
            desc.type = PhysicsBodyType::Dynamic;
            desc.position = {trs->translation.x(), trs->translation.y(), trs->translation.z()};
            desc.rotation = {trs->rotation.x(), trs->rotation.y(), trs->rotation.z(),
                             trs->rotation.w()};
            desc.mass = 1.0f;
            desc.gravityScale = 1.0f;
            desc.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
            const PhysicsBodyHandle body = world.createBody(desc);
            ColliderDesc collider;
            collider.shape = GltfLoader::meshConvexHull(asset, asset.meshes[n.meshIndex.value()]);
            collider.material = desc.material;
            static_cast<void>(world.createCollider(body, collider));
            tetras.push_back(body);
        }
        return tetras;
    };

    const auto run = [&](std::array<float, 3>& maxTwistDeg, std::array<Vec3, 3>& finalPos)
    {
        PhysicsWorld world;
        const auto tetras = buildWorld(world);
        std::array<Quaternion, 3> prevQ{};
        maxTwistDeg = {};
        bool allAsleep = false;
        for (int i = 0; i < 900 && !allAsleep; ++i)
        {
            world.step(kFixedDt);
            allAsleep = true;
            for (std::size_t t = 0; t < tetras.size(); ++t)
            {
                const auto tf = world.bodyTransform(tetras[t]);
                REQUIRE(tf.has_value());
                const Quaternion q = tf->rotation();
                const Quaternion dq = Quaternion::normalise(q * prevQ[t].conjugate());
                const float twist =
                    2.0f * std::atan2(dq.y(), dq.w()) * 180.0f / std::numbers::pi_v<float>;
                // Skip the ballistic tumble (steps < 40): a free tumble legitimately rotates.
                // The bound guards the *settle* — contact-driven motion.
                if (i > 40 && !world.sleeping(tetras[t]))
                {
                    maxTwistDeg[t] = std::max(maxTwistDeg[t], std::abs(twist));
                }
                prevQ[t] = q;
                allAsleep = allAsleep && world.sleeping(tetras[t]);
                finalPos[t] = tf->position();
            }
        }
        return allAsleep;
    };

    std::array<float, 3> maxTwist{};
    std::array<Vec3, 3> finalPos{};
    const bool slept = run(maxTwist, finalPos);
    for (std::size_t t = 0; t < 3; ++t)
    {
        INFO("tetra " << t << " max settle twist " << maxTwist[t] << " deg");
        CHECK(maxTwist[t] < 3.0f);
    }
    CHECK(slept); // the pile still comes to rest and sleeps

    // Bit-identical replay: the refresh path must be deterministic.
    std::array<float, 3> maxTwist2{};
    std::array<Vec3, 3> finalPos2{};
    static_cast<void>(run(maxTwist2, finalPos2));
    for (std::size_t t = 0; t < 3; ++t)
    {
        CHECK(finalPos[t].x() == finalPos2[t].x());
        CHECK(finalPos[t].y() == finalPos2[t].y());
        CHECK(finalPos[t].z() == finalPos2[t].z());
    }
}

// Diagnostic sweep (hidden): rotational-tunnelling verification for P9.6. Drops fast-spinning
// bodies (box family incl. the historic (0.3,1,1), a thin plate, and a tetra hull) onto a wide
// static floor and reports each config's minimum body-centre height over 5 s — a tunnel reads as
// minY far below the floor (the P9.5-era violent cases showed minY ≈ −75…−96). A/B protocol:
// run as-is (refresh on), then temporarily set kSubstepRefreshRotation to a huge value (gate
// never fires = pre-P9.6 behaviour), rebuild, rerun, compare, restore.
TEST_CASE("Demos.RotationalTunnellingSweep", "[.][TunnelSweep]")
{
    struct ShapeCase
    {
        const char* name;
        ColliderShape shape;
    };
    const std::array<Vec3, 4> tetraVerts{Vec3{0.4f, 0.4f, 0.4f}, Vec3{0.4f, -0.4f, -0.4f},
                                         Vec3{-0.4f, 0.4f, -0.4f}, Vec3{-0.4f, -0.4f, 0.4f}};
    const std::array<std::uint32_t, 12> tetraIdx{0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2};
    const std::vector<ShapeCase> shapes{
        {"box(0.3,1,1)", BoxShape{Vec3{0.3f, 1.0f, 1.0f}, {}}},
        {"plate(0.5,0.05,0.5)", BoxShape{Vec3{0.5f, 0.05f, 0.5f}, {}}},
        {"tetra(0.4)", buildConvexHull(tetraVerts, tetraIdx)},
    };
    const std::vector<std::pair<const char*, Vec3>> axes{
        {"x", Vec3{1.0f, 0.0f, 0.0f}},
        {"z", Vec3{0.0f, 0.0f, 1.0f}},
        {"diag", Vec3::normalise(Vec3{1.0f, 1.0f, 1.0f})},
    };
    const std::array<float, 3> speeds{10.0f, 20.0f, 40.0f};

    int tunnels = 0;
    for (const ShapeCase& sc : shapes)
    {
        for (const auto& [axisName, axis] : axes)
        {
            for (const float speed : speeds)
            {
                PhysicsWorld world;
                addStaticFloor(world, 30.0f, 0.5f);
                PhysicsBodyDesc desc;
                desc.type = PhysicsBodyType::Dynamic;
                desc.position = {0.0f, 1.5f, 0.0f};
                desc.angularVelocity = axis * speed;
                desc.mass = 1.0f;
                desc.gravityScale = 1.0f;
                desc.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
                const PhysicsBodyHandle body = world.createBody(desc);
                ColliderDesc collider;
                collider.shape = sc.shape;
                collider.material = desc.material;
                static_cast<void>(world.createCollider(body, collider));

                float minY = 1e9f;
                for (int i = 0; i < 300; ++i)
                {
                    world.step(kFixedDt);
                    minY = std::min(minY, world.bodyTransform(body)->position().y());
                }
                const float finalY = world.bodyTransform(body)->position().y();
                const bool tunnelled = minY < -1.0f;
                tunnels += tunnelled ? 1 : 0;
                std::printf("[sweep] %-20s axis=%-4s w=%4.0f  minY=%8.3f finalY=%8.3f%s\n", sc.name,
                            axisName, static_cast<double>(speed), static_cast<double>(minY),
                            static_cast<double>(finalY), tunnelled ? "  <TUNNEL>" : "");
            }
        }
    }
    std::printf("[sweep] tunnelled configs: %d / 27\n", tunnels);
}

// P9.6 regression gate: fast-spinning bodies up to 20 rad/s must never tunnel through the
// floor. Before the mid-step manifold refresh, box(0.3,1,1) at 20 rad/s (about x or the
// diagonal) fell straight through (minY ≈ −45 / −5); with it, every ≤20 rad/s config in the
// sweep rests on the surface. 40 rad/s flat-box spinners (~380 RPM) can still tunnel — a
// single mid-step refresh is ~19° stale at that rate; that is the documented Stage-2
// boundary (per-substep re-detection), not a regression. The hidden [TunnelSweep] variant
// prints the full A/B table including the 40 rad/s cases.
TEST_CASE("Demos.RotationalTunnellingBoundedTo20RadPerSec", "[Demos]")
{
    const std::array<Vec3, 4> tetraVerts{Vec3{0.4f, 0.4f, 0.4f}, Vec3{0.4f, -0.4f, -0.4f},
                                         Vec3{-0.4f, 0.4f, -0.4f}, Vec3{-0.4f, -0.4f, 0.4f}};
    const std::array<std::uint32_t, 12> tetraIdx{0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2};
    const std::vector<ColliderShape> shapes{
        BoxShape{Vec3{0.3f, 1.0f, 1.0f}, {}},
        BoxShape{Vec3{0.5f, 0.05f, 0.5f}, {}},
        buildConvexHull(tetraVerts, tetraIdx),
    };
    const std::vector<Vec3> axes{Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f},
                                 Vec3::normalise(Vec3{1.0f, 1.0f, 1.0f})};

    for (std::size_t s = 0; s < shapes.size(); ++s)
    {
        for (std::size_t a = 0; a < axes.size(); ++a)
        {
            for (const float speed : {10.0f, 20.0f})
            {
                PhysicsWorld world;
                addStaticFloor(world, 30.0f, 0.5f);
                PhysicsBodyDesc desc;
                desc.type = PhysicsBodyType::Dynamic;
                desc.position = {0.0f, 1.5f, 0.0f};
                desc.angularVelocity = axes[a] * speed;
                desc.mass = 1.0f;
                desc.gravityScale = 1.0f;
                desc.material = PhysicsMaterial{.restitution = 0.0f, .friction = 0.5f};
                const PhysicsBodyHandle body = world.createBody(desc);
                ColliderDesc collider;
                collider.shape = shapes[s];
                collider.material = desc.material;
                static_cast<void>(world.createCollider(body, collider));

                float minY = 1e9f;
                for (int i = 0; i < 300; ++i)
                {
                    world.step(kFixedDt);
                    minY = std::min(minY, world.bodyTransform(body)->position().y());
                }
                INFO("shape " << s << " axis " << a << " w=" << speed << " minY=" << minY);
                CHECK(minY > -0.5f); // never tunnelled through the floor
            }
        }
    }
}
