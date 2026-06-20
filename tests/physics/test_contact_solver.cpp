#include <fire_engine/physics/contact_solver.hpp>

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/physics/physics_constants.hpp>

using fire_engine::ContactSolver;
using fire_engine::kVelocityIterations;
using fire_engine::SolverBody;
using fire_engine::SolverContactInput;
using fire_engine::Vec3;

namespace
{

// A single-point contact along `normal`, restitution/friction as given.
SolverContactInput makeContact(Vec3 normal, float penetration, float restitution, float friction)
{
    SolverContactInput c;
    c.bodyA = 0;
    c.bodyB = 1;
    c.normal = normal;
    c.points[0] = Vec3{0.0f, 0.0f, 0.0f};
    c.penetration[0] = penetration;
    c.pointCount = 1;
    c.restitution = restitution;
    c.friction = friction;
    return c;
}

void runVelocity(ContactSolver& solver, std::vector<SolverBody>& bodies,
                 const std::vector<SolverContactInput>& contacts)
{
    solver.prepare(bodies, contacts, 1.0f / 60.0f);
    solver.warmStart(bodies);
    for (int i = 0; i < kVelocityIterations; ++i)
    {
        solver.solveVelocity(bodies);
    }
}

} // namespace

TEST_CASE("ContactSolver.NormalRemovesApproachVelocity", "[ContactSolver]")
{
    // Body A falls onto an immovable body B (normal points B -> A = +y). With no
    // restitution the normal impulse cancels the approaching velocity exactly.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -1.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].positionWeight = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(0.0f).margin(1e-4f));
}

TEST_CASE("ContactSolver.RestitutionBouncesAboveThreshold", "[ContactSolver]")
{
    // Approaching at 2 m/s (above the 1 m/s threshold) with restitution 1 should
    // reverse the velocity to a clean +2 m/s separation.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -2.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].positionWeight = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 1.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(2.0f).margin(1e-3f));
}

TEST_CASE("ContactSolver.RestitutionSuppressedBelowThreshold", "[ContactSolver]")
{
    // Approaching at 0.5 m/s (below threshold) with restitution 1: the bounce bias
    // is suppressed so the body comes to rest instead of buzzing.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -0.5f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].positionWeight = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 1.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(0.0f).margin(1e-4f));
}

TEST_CASE("ContactSolver.FrictionStopsTangentialMotion", "[ContactSolver]")
{
    // A body sliding (x) while pressed down (y) onto a floor, with a friction
    // coefficient large enough that the friction cone (mu * normalImpulse) covers
    // the tangential momentum, comes to a full stop on the surface plane.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {1.0f, -1.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].positionWeight = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 1.0f)});

    CHECK(bodies[0].velocity.approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 1e-3f));
}

TEST_CASE("ContactSolver.FrictionClampedByNormalImpulse", "[ContactSolver]")
{
    // With a small friction coefficient the tangent impulse is clamped to
    // mu * normalImpulse, so only part of the tangential velocity is removed.
    // normalImpulse here is 1 (1 m/s approach, unit mass), so mu=0.3 caps the
    // tangential change at 0.3, leaving 0.7 m/s.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {1.0f, -1.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].positionWeight = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 0.3f)});

    CHECK(bodies[0].velocity.x() == Catch::Approx(0.7f).margin(1e-3f));
    CHECK(bodies[0].velocity.y() == Catch::Approx(0.0f).margin(1e-3f));
}

TEST_CASE("ContactSolver.MassRatioWeightsImpulseByInverseMass", "[ContactSolver]")
{
    // Two dynamic bodies approaching head-on: A is 10x heavier (invMass 0.1) than B
    // (invMass 1). With restitution 0 they reach the momentum-conserving common
    // velocity (-9/11 m/s), and the light body's velocity changes ~10x as much.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -1.0f, 0.0f};
    bodies[0].invMass = 0.1f;
    bodies[0].positionWeight = 0.1f;
    bodies[1].velocity = {0.0f, 1.0f, 0.0f};
    bodies[1].invMass = 1.0f;
    bodies[1].positionWeight = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 0.0f)});

    const float common = -9.0f / 11.0f;
    CHECK(bodies[0].velocity.y() == Catch::Approx(common).margin(1e-3f));
    CHECK(bodies[1].velocity.y() == Catch::Approx(common).margin(1e-3f));

    const float changeA = std::abs(bodies[0].velocity.y() - (-1.0f));
    const float changeB = std::abs(bodies[1].velocity.y() - (1.0f));
    CHECK(changeB == Catch::Approx(10.0f * changeA).margin(1e-2f));
}

TEST_CASE("ContactSolver.PositionPassPushesOutOfPenetration", "[ContactSolver]")
{
    // The split-impulse position pass moves a penetrating dynamic body out along
    // the normal (a fraction of penetration beyond slop), leaving the immovable
    // body in place.
    std::vector<SolverBody> bodies(2);
    bodies[0].position = {0.0f, 0.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].positionWeight = 1.0f;
    bodies[1].position = {0.0f, -1.0f, 0.0f};

    ContactSolver solver;
    solver.prepare(bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.1f, 0.0f, 0.0f)}, 1.0f / 60.0f);
    solver.solvePosition(bodies);

    CHECK(bodies[0].position.y() > 0.0f);                  // pushed out along +y
    CHECK(bodies[0].position.y() < 0.1f);                  // by less than the full penetration
    CHECK(bodies[1].position.y() == Catch::Approx(-1.0f)); // immovable target stays put
}

TEST_CASE("ContactSolver.SpeculativeContactBrakesOvershoot", "[ContactSolver]")
{
    // A speculative gap contact (negative penetration = a 0.05 m gap) lets the body
    // close the gap but no further: a body closing far faster than gap/dt is braked
    // to exactly gap/dt (= 0.05 * 60 = 3 m/s), so it reaches the surface this step
    // without tunnelling through it.
    const float dt = 1.0f / 60.0f;
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -10.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].positionWeight = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, -0.05f, 0.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(-0.05f / dt).margin(1e-3f)); // = -3 m/s
}

TEST_CASE("ContactSolver.SpeculativeContactIgnoresSlowApproach", "[ContactSolver]")
{
    // A body approaching slower than gap/dt won't reach the surface this step, so
    // the speculative contact applies no impulse and leaves its velocity untouched.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -1.0f, 0.0f}; // gap/dt = 3 m/s; 1 m/s won't reach
    bodies[0].invMass = 1.0f;
    bodies[0].positionWeight = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, -0.05f, 0.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(-1.0f).margin(1e-4f)); // unchanged
}
