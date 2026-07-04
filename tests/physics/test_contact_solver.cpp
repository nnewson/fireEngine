#include <fire_engine/physics/contact_solver.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/physics/physics_constants.hpp>

using fire_engine::ContactSolver;
using fire_engine::SolverBody;
using fire_engine::SolverContactInput;
using fire_engine::Vec3;

namespace
{

// Substep dt the unit tests prepare against. The converged velocity of a relax pass is
// independent of h; h only sets the soft-spring rate and the speculative 1/h gap brake,
// so a representative single-step h keeps the gap-brake expectations exactly -gap/h.
constexpr float kTestH = 1.0f / 60.0f;

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

// Drive the velocity constraint to convergence the way the TGS substep loop does:
// warm-start once, then alternate a biased solve with a no-bias relax pass, ending on
// the relax so the soft bias velocity is removed. The bodies are not advanced (these
// are pure velocity-response tests), so the separation stays at its prepared value.
void runVelocity(ContactSolver& solver, std::vector<SolverBody>& bodies,
                 const std::vector<SolverContactInput>& contacts)
{
    solver.prepare(bodies, contacts, kTestH);
    solver.warmStart(bodies);
    for (int i = 0; i < 8; ++i)
    {
        solver.solveVelocity(bodies, /*useBias=*/true);
        solver.solveVelocity(bodies, /*useBias=*/false);
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

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(0.0f).margin(1e-4f));
}

TEST_CASE("ContactSolver.RestitutionBouncesAboveThreshold", "[ContactSolver]")
{
    // Approaching at 2 m/s (above the 1 m/s threshold) with restitution 1 should
    // reverse the velocity to a clean +2 m/s separation. The velocity solve removes
    // the approach; the end-of-step restitution pass adds the rebound.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -2.0f, 0.0f};
    bodies[0].invMass = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 1.0f, 0.0f)});
    solver.applyRestitution(bodies);

    CHECK(bodies[0].velocity.y() == Catch::Approx(2.0f).margin(1e-3f));
}

TEST_CASE("ContactSolver.RestitutionSuppressedBelowThreshold", "[ContactSolver]")
{
    // Approaching at 0.5 m/s (below threshold) with restitution 1: the restitution
    // pass is suppressed so the body comes to rest instead of buzzing.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -0.5f, 0.0f};
    bodies[0].invMass = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 1.0f, 0.0f)});
    solver.applyRestitution(bodies);

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
    bodies[1].velocity = {0.0f, 1.0f, 0.0f};
    bodies[1].invMass = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 0.0f)});

    const float common = -9.0f / 11.0f;
    CHECK(bodies[0].velocity.y() == Catch::Approx(common).margin(1e-3f));
    CHECK(bodies[1].velocity.y() == Catch::Approx(common).margin(1e-3f));

    const float changeA = std::abs(bodies[0].velocity.y() - (-1.0f));
    const float changeB = std::abs(bodies[1].velocity.y() - (1.0f));
    CHECK(changeB == Catch::Approx(10.0f * changeA).margin(1e-2f));
}

TEST_CASE("ContactSolver.FrictionDiskClampLimitsCombinedTangentImpulse", "[ContactSolver]")
{
    // Friction is a 2D disk clamp (P9.3), not two independent scalar axes: a body sliding
    // diagonally on both tangents at once is limited by the *combined* impulse magnitude
    // μ·N, not μ·N per axis. Here the body slides at 1 m/s along each tangent (x and z) with
    // μ = 0.3 and a normal impulse of 1 (1 m/s approach, unit mass), so the total tangential
    // change is capped at 0.3 along the diagonal — leaving ≈ 1 − 0.3/√2 on each axis, not the
    // 0.7 an independent-axis (box) clamp would leave.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {1.0f, -1.0f, 1.0f};
    bodies[0].invMass = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 0.3f)});

    // Total tangential impulse magnitude is μ·N = 0.3, split along the (x,z) diagonal.
    const float perAxis = 1.0f - 0.3f / std::sqrt(2.0f);
    CHECK(bodies[0].velocity.x() == Catch::Approx(perAxis).margin(2e-2f));
    CHECK(bodies[0].velocity.z() == Catch::Approx(perAxis).margin(2e-2f));
    CHECK(bodies[0].velocity.y() == Catch::Approx(0.0f).margin(1e-3f));
}

TEST_CASE("ContactSolver.SoftBiasPushesOutOfPenetration", "[ContactSolver]")
{
    // The soft contact constraint replaces the old split-impulse position pass: a
    // penetrating, resting body is given an outward (separating) bias velocity, which
    // the substep position integration then turns into separation. A single biased
    // solve is enough to surface the push-out velocity.
    std::vector<SolverBody> bodies(2);
    bodies[0].position = {0.0f, 0.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[1].position = {0.0f, -1.0f, 0.0f};

    ContactSolver solver;
    const std::array contacts{makeContact({0.0f, 1.0f, 0.0f}, 0.1f, 0.0f, 0.0f)};
    solver.prepare(bodies, contacts, kTestH);
    solver.warmStart(bodies);
    solver.solveVelocity(bodies, /*useBias=*/true);

    CHECK(bodies[0].velocity.y() > 0.0f);                 // separating (pushed out along +y)
    CHECK(bodies[1].velocity.y() == Catch::Approx(0.0f)); // immovable target unaffected
}

TEST_CASE("ContactSolver.SpeculativeContactBrakesOvershoot", "[ContactSolver]")
{
    // A speculative gap contact (negative penetration = a 0.05 m gap) lets the body
    // close the gap but no further: a body closing far faster than gap/h is braked
    // to exactly gap/h (= 0.05 * 60 = 3 m/s), so it reaches the surface this substep
    // without tunnelling through it. The speculative brake acts on both bias and relax.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -10.0f, 0.0f};
    bodies[0].invMass = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, -0.05f, 0.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(-0.05f / kTestH).margin(1e-3f)); // = -3 m/s
}

TEST_CASE("ContactSolver.SpeculativeContactIgnoresSlowApproach", "[ContactSolver]")
{
    // A body approaching slower than gap/h won't reach the surface this substep, so
    // the speculative contact applies no impulse and leaves its velocity untouched.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -1.0f, 0.0f}; // gap/h = 3 m/s; 1 m/s won't reach
    bodies[0].invMass = 1.0f;

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, -0.05f, 0.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(-1.0f).margin(1e-4f)); // unchanged
}

TEST_CASE("ContactSolver.OffCentreImpulseImpartsAngularVelocity", "[ContactSolver]")
{
    // A body descending onto a contact offset +x from its centre of mass gains spin.
    // invMass 1, isotropic inverse inertia 2; contact at r=(1,0,0), normal +y.
    // Solving the normal constraint to vRel=0 gives velA.y=-2/3 and ωz=+2/3:
    //   J(impulse): velA.y = -1 + J,  ωz = invInertia·(r×P).z = 2·J,
    //   vRel·n = velA.y + (ω×r).y = velA.y + ωz = 0  ⇒  3J = 1.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -1.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].inverseInertiaLocal = {2.0f, 2.0f, 2.0f};

    SolverContactInput c;
    c.bodyA = 0;
    c.bodyB = 1; // static (defaults: invMass 0, inverse inertia 0)
    c.normal = {0.0f, 1.0f, 0.0f};
    c.points[0] = {1.0f, 0.0f, 0.0f};
    c.penetration[0] = 0.0f;
    c.pointCount = 1;

    ContactSolver solver;
    runVelocity(solver, bodies, {c});

    CHECK(bodies[0].velocity.y() == Catch::Approx(-2.0f / 3.0f).margin(1e-3f));
    CHECK(bodies[0].angularVelocity.z() == Catch::Approx(2.0f / 3.0f).margin(1e-3f));
    CHECK(bodies[0].angularVelocity.x() == Catch::Approx(0.0f).margin(1e-4f));
    CHECK(bodies[0].angularVelocity.y() == Catch::Approx(0.0f).margin(1e-4f));
}

TEST_CASE("ContactSolver.CentredContactProducesNoSpin", "[ContactSolver]")
{
    // A contact at the centre of mass (r = 0) has no lever arm, so the result is
    // pure-linear — identical to the P2 solver, with zero angular velocity.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -1.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].inverseInertiaLocal = {2.0f, 2.0f, 2.0f};

    ContactSolver solver;
    runVelocity(solver, bodies, {makeContact({0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 0.0f)});

    CHECK(bodies[0].velocity.y() == Catch::Approx(0.0f).margin(1e-3f));
    CHECK(bodies[0].angularVelocity.approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 1e-5f));
}

// --- Mid-step manifold refresh (P9.6 stage 1) -------------------------------------------

namespace
{

// A single-point contact with explicit body indices / point / key (the refresh tests need
// multiple pairs and non-zero keys — refresh() walks points_ by per-contact unique key runs).
SolverContactInput makeContactAt(int bodyA, int bodyB, Vec3 normal, Vec3 point, float penetration,
                                 float restitution, std::uint64_t key)
{
    SolverContactInput c;
    c.bodyA = bodyA;
    c.bodyB = bodyB;
    c.normal = normal;
    c.points[0] = point;
    c.penetration[0] = penetration;
    c.pointCount = 1;
    c.restitution = restitution;
    c.friction = 0.0f;
    c.key = key;
    return c;
}

} // namespace

TEST_CASE("ContactSolver.RefreshAtUnmovedPoseIsANoOp", "[ContactSolver]")
{
    // Two identical "substeps" with a refresh injected between them (same pose, same manifold)
    // must produce exactly the trajectory of the run without the refresh: rows rebuild
    // identically and the accumulated impulse / restitution state carries over.
    const auto run = [](bool withRefresh)
    {
        std::vector<SolverBody> bodies(2);
        bodies[0].velocity = {0.0f, -2.0f, 0.0f};
        bodies[0].invMass = 1.0f;
        const std::vector<SolverContactInput> contacts{
            makeContactAt(0, 1, {0.0f, 1.0f, 0.0f}, {}, 0.0f, 1.0f, 7ULL)};

        ContactSolver solver;
        solver.prepare(bodies, contacts, kTestH);
        solver.warmStart(bodies);
        solver.solveVelocity(bodies, true);
        solver.solveVelocity(bodies, false);
        if (withRefresh)
        {
            const std::vector<std::uint8_t> flags{1};
            solver.refresh(bodies, contacts, flags);
        }
        solver.warmStart(bodies);
        solver.solveVelocity(bodies, true);
        solver.solveVelocity(bodies, false);
        solver.applyRestitution(bodies);
        return bodies[0].velocity;
    };

    const Vec3 without = run(false);
    const Vec3 with = run(true);
    CHECK(with.x() == without.x());
    CHECK(with.y() == without.y());
    CHECK(with.z() == without.z());
}

TEST_CASE("ContactSolver.RefreshCarriesImpulseAndRestitutionAcrossRotatedPose", "[ContactSolver]")
{
    // A body impacts at 2 m/s (restitution 1); the first "substep" cancels most of the approach
    // and accumulates the impulse. The body then rotates slightly (as the substep integration
    // would) and the manifold is refreshed at the new pose. relVelN0 and maxNormalImpulse must
    // survive the refresh: end-of-step restitution still rebounds to +2 m/s. (If relVelN0 were
    // recomputed at refresh time it would read the already-cancelled approach ≈ 0 and the
    // rebound would vanish.)
    std::vector<SolverBody> bodies(2);
    bodies[0].position = {0.0f, 0.5f, 0.0f}; // COM 0.5 above the contact: a real lever
    bodies[0].velocity = {0.0f, -2.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    bodies[0].inverseInertiaLocal = {2.0f, 2.0f, 2.0f};
    const Vec3 contactPoint{0.0f, 0.0f, 0.0f};
    std::vector<SolverContactInput> contacts{
        makeContactAt(0, 1, {0.0f, 1.0f, 0.0f}, contactPoint, 0.0f, 1.0f, 7ULL)};

    ContactSolver solver;
    solver.prepare(bodies, contacts, kTestH);
    solver.warmStart(bodies);
    solver.solveVelocity(bodies, true);
    solver.solveVelocity(bodies, false);

    // Mid-step: the body has rotated a little; the re-collided manifold finds the contact
    // where the old anchor now tracks to (what a real narrowphase re-run would report).
    bodies[0].orientation = fire_engine::Quaternion::fromAxisAngle({0.0f, 0.0f, 1.0f}, 0.02f);
    const Vec3 tracked =
        bodies[0].position + bodies[0].orientation.rotate(contactPoint - bodies[0].position);
    contacts[0] = makeContactAt(0, 1, {0.0f, 1.0f, 0.0f}, tracked, 0.0f, 1.0f, 7ULL);
    const std::vector<std::uint8_t> flags{1};
    solver.refresh(bodies, contacts, flags);

    solver.warmStart(bodies);
    solver.solveVelocity(bodies, true);
    solver.solveVelocity(bodies, false);
    solver.applyRestitution(bodies);

    CHECK(bodies[0].velocity.y() == Catch::Approx(2.0f).margin(0.1f));
}

TEST_CASE("ContactSolver.RefreshLeavesUnflaggedRowsUntouched", "[ContactSolver]")
{
    // Two independent pairs; only the second is flagged. The first pair's rows are copied
    // verbatim, so its body's trajectory is bit-identical to a run without any refresh.
    const auto run = [](bool withRefresh)
    {
        std::vector<SolverBody> bodies(3);
        bodies[0].velocity = {0.0f, -1.0f, 0.0f};
        bodies[0].invMass = 1.0f;
        bodies[1].position = {5.0f, 0.5f, 0.0f};
        bodies[1].velocity = {0.0f, -3.0f, 0.0f};
        bodies[1].invMass = 1.0f;
        const std::vector<SolverContactInput> contacts{
            makeContactAt(0, 2, {0.0f, 1.0f, 0.0f}, {}, 0.0f, 0.0f, 111ULL),
            makeContactAt(1, 2, {0.0f, 1.0f, 0.0f}, {5.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 222ULL)};

        ContactSolver solver;
        solver.prepare(bodies, contacts, kTestH);
        solver.warmStart(bodies);
        solver.solveVelocity(bodies, true);
        solver.solveVelocity(bodies, false);
        if (withRefresh)
        {
            const std::vector<std::uint8_t> flags{0, 1};
            solver.refresh(bodies, contacts, flags);
        }
        solver.warmStart(bodies);
        solver.solveVelocity(bodies, true);
        solver.solveVelocity(bodies, false);
        return bodies[0].velocity;
    };

    const Vec3 without = run(false);
    const Vec3 with = run(true);
    CHECK(with.x() == without.x());
    CHECK(with.y() == without.y());
    CHECK(with.z() == without.z());
}

TEST_CASE("ContactSolver.RefreshWithEmptyManifoldDropsRows", "[ContactSolver]")
{
    // The re-collide found nothing (the body rotated away mid-step): the pair's rows drop and
    // the solver stops producing impulses — the body keeps its velocity through warm start,
    // solve, and restitution.
    std::vector<SolverBody> bodies(2);
    bodies[0].velocity = {0.0f, -2.0f, 0.0f};
    bodies[0].invMass = 1.0f;
    std::vector<SolverContactInput> contacts{
        makeContactAt(0, 1, {0.0f, 1.0f, 0.0f}, {}, 0.0f, 1.0f, 7ULL)};

    ContactSolver solver;
    solver.prepare(bodies, contacts, kTestH);
    solver.warmStart(bodies);
    solver.solveVelocity(bodies, true);
    solver.solveVelocity(bodies, false);

    contacts[0].pointCount = 0; // mid-step separation
    const std::vector<std::uint8_t> flags{1};
    solver.refresh(bodies, contacts, flags);
    CHECK(solver.empty());

    const Vec3 before = bodies[0].velocity;
    solver.warmStart(bodies);
    solver.solveVelocity(bodies, true);
    solver.solveVelocity(bodies, false);
    solver.applyRestitution(bodies);
    CHECK(bodies[0].velocity.y() == before.y());
}
