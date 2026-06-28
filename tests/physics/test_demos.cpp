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
#include <numbers>
#include <vector>

#include <fire_engine/core/convex_hull_builder.hpp>
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

TEST_CASE("Demos.FallRest.BoxComesToRestOnFloor", "[Demos]")
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

TEST_CASE("Demos.Restitution.HigherRestitutionBouncesHigher", "[Demos]")
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

TEST_CASE("Demos.Friction.HighFrictionStaysLowFrictionSlides", "[Demos]")
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

TEST_CASE("Demos.Stack.SettlesAndStaysStill", "[Demos]")
{
    // StackDemo.gltf: a tower of three boxes dropped with small gaps settles into a
    // resting stack at centres 0.5, 1.5, 2.5 and stays still (no collapse, no drift)
    // rather than buzzing apart — the warm-started impulse solver's headline win.
    // (Three, not more: a taller tower sits near the fixed-iteration solver's
    // stability margin and quiesces only slowly — see generate.py demo_stack.)
    PhysicsWorld world;
    addStaticFloor(world, 6.0f);
    std::array<PhysicsBodyHandle, 3> boxes{};
    for (int i = 0; i < 3; ++i)
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

TEST_CASE("Demos.Topple.TallBoxTopplesOntoSide", "[Demos]")
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

TEST_CASE("Demos.ConvexHull.PileSettlesAtRest", "[Demos]")
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

TEST_CASE("Demos.Sleep.StackSleepsThenWakesOnImpact", "[Demos]")
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

TEST_CASE("Demos.StaticMesh.BodiesSettleInValley", "[Demos]")
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

    step(world, 480); // 8 s — drop onto the mesh surface and settle

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

TEST_CASE("Demos.Compound.LShapeRestsOnFloor", "[Demos]")
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
