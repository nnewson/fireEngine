#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>

#include <fire_engine/math/constants.hpp>
#include <fire_engine/math/quaternion.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/physics/articulation.hpp>
#include <fire_engine/physics/articulation_contact.hpp>
#include <fire_engine/physics/collider_shape.hpp>
#include <fire_engine/physics/physics_world.hpp>
#include <fire_engine/physics/spatial.hpp>

using namespace fire_engine;

namespace
{

// Build a planar two-link revolute chain in the XY plane, each joint about +Z:
//   base (root) → link1 (joint 1m along +X, body 1m further) → link2 (same offsets).
// Returns the articulation with q all zero (call forwardKinematics / set q in the test).
Articulation makeTwoLinkChain()
{
    Articulation arm;
    arm.addRootLink(ArticulationLinkDesc{});

    ArticulationLinkDesc link;
    link.parent = 0;
    link.joint = ArticulationJointType::Revolute;
    link.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
    link.parentToJoint = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    link.jointToChild = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    arm.addLink(link); // index 1

    link.parent = 1;
    arm.addLink(link); // index 2
    return arm;
}

} // namespace

TEST_CASE("Articulation.DofCountTracksRevoluteJoints", "[Articulation]")
{
    Articulation arm;
    arm.addRootLink(ArticulationLinkDesc{});
    CHECK(arm.linkCount() == 1);
    CHECK(arm.dofCount() == 0); // a floating base adds no generalized coordinates here

    ArticulationLinkDesc revolute;
    revolute.parent = 0;
    revolute.joint = ArticulationJointType::Revolute;
    arm.addLink(revolute);
    CHECK(arm.dofCount() == 1);
    CHECK(arm.linkDofOffset(1) == 0);

    ArticulationLinkDesc welded;
    welded.parent = 1;
    welded.joint = ArticulationJointType::Fixed;
    arm.addLink(welded); // a Fixed joint adds no DOF
    CHECK(arm.linkCount() == 3);
    CHECK(arm.dofCount() == 1);
    CHECK(arm.linkDofOffset(2) == -1);
}

TEST_CASE("Articulation.ForwardKinematicsMatchesHandComputedPose", "[Articulation]")
{
    Articulation arm = makeTwoLinkChain();

    // q all zero: the chain lies straight along +X. Link1 body at (2,0,0), link2 at (4,0,0).
    arm.forwardKinematics();
    CHECK(arm.linkWorld(0).translation.approxEqual(Vec3{0.0f, 0.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(1).translation.approxEqual(Vec3{2.0f, 0.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(2).translation.approxEqual(Vec3{4.0f, 0.0f, 0.0f}, 1e-5f));

    // Bend the first joint 90° about +Z; leave the second straight. Hand-computed:
    //   link1 swings to (1,1,0) rotated +90°; link2 to (1,3,0) carrying that rotation.
    arm.q(0, pi / 2.0f);
    arm.forwardKinematics();
    CHECK(arm.linkWorld(1).translation.approxEqual(Vec3{1.0f, 1.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(2).translation.approxEqual(Vec3{1.0f, 3.0f, 0.0f}, 1e-5f));

    // The link's orientation is the accumulated +90° about Z: it maps local +X onto world +Y.
    const Vec3 link1X = arm.linkWorld(1).transformDirection(Vec3{1.0f, 0.0f, 0.0f});
    CHECK(link1X.approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 1e-5f));
}

TEST_CASE("Articulation.FloatingBaseOffsetsEveryLink", "[Articulation]")
{
    Articulation arm = makeTwoLinkChain();
    arm.q(0, pi / 2.0f);

    // Translate the floating base by (10,0,0): every link's world pose shifts by it.
    arm.baseTransform(RigidTransform{Quaternion::identity(), Vec3{10.0f, 0.0f, 0.0f}});
    arm.forwardKinematics();
    CHECK(arm.linkWorld(0).translation.approxEqual(Vec3{10.0f, 0.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(1).translation.approxEqual(Vec3{11.0f, 1.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(2).translation.approxEqual(Vec3{11.0f, 3.0f, 0.0f}, 1e-5f));
}

TEST_CASE("Articulation.FloatingBaseRotationCarriesThroughChain", "[Articulation]")
{
    Articulation arm = makeTwoLinkChain(); // q all zero ⇒ straight along local +X

    // Rotate the base +90° about Z: the straight chain now lies along world +Y.
    arm.baseTransform(
        RigidTransform{Quaternion::fromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, pi / 2.0f), Vec3{}});
    arm.forwardKinematics();
    CHECK(arm.linkWorld(1).translation.approxEqual(Vec3{0.0f, 2.0f, 0.0f}, 1e-5f));
    CHECK(arm.linkWorld(2).translation.approxEqual(Vec3{0.0f, 4.0f, 0.0f}, 1e-5f));
}

TEST_CASE("RigidTransform.ComposeAndInverseRoundTrip", "[Articulation]")
{
    const RigidTransform t{Quaternion::fromAxisAngle(Vec3{0.0f, 1.0f, 0.0f}, 0.7f),
                           Vec3{3.0f, -2.0f, 1.0f}};
    const Vec3 p{1.0f, 2.0f, 3.0f};

    // inverse() undoes the transform; (t * t.inverse()) is the identity map.
    const Vec3 roundTrip = t.inverse().transformPoint(t.transformPoint(p));
    CHECK(roundTrip.approxEqual(p, 1e-5f));

    const RigidTransform composed = t * t.inverse();
    CHECK(composed.transformPoint(p).approxEqual(p, 1e-5f));
}

TEST_CASE("PhysicsWorld.LinkColliderTracksForwardKinematicsInBroadphase", "[Articulation]")
{
    // A collider owned by an articulation link must register in the broadphase and answer
    // spatial queries at the link's forward-kinematics pose — the Phase A gate. The chain:
    // floating base at (0,5,0) → revolute-Z link, joint 1m along +X, body 1m beyond. With
    // q = 0 the link body sits at (2,5,0); bending the joint +90° swings it to (1,6,0).
    PhysicsWorld world;
    const PhysicsArticulationHandle arm = world.createArticulation();
    REQUIRE(world.articulationCount() == 1);

    Articulation* art = world.articulation(arm);
    REQUIRE(art != nullptr);
    art->addRootLink(ArticulationLinkDesc{});

    ArticulationLinkDesc link;
    link.parent = 0;
    link.joint = ArticulationJointType::Revolute;
    link.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
    link.parentToJoint = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    link.jointToChild = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    const int linkIndex = art->addLink(link);
    art->baseTransform(RigidTransform{Quaternion::identity(), Vec3{0.0f, 5.0f, 0.0f}});

    ColliderDesc box;
    box.shape = BoxShape{Vec3{0.4f, 0.4f, 0.4f}};
    const PhysicsColliderHandle collider = world.attachLinkCollider(arm, linkIndex, box);
    REQUIRE(collider.valid());

    // step() runs forward kinematics, then refreshes the link collider's swept bounds.
    world.step(1.0f / 60.0f);

    const auto atRest = world.overlapSphere(Vec3{2.0f, 5.0f, 0.0f}, 0.2f);
    REQUIRE(atRest.size() == 1);
    CHECK(atRest[0].collider == collider);
    // A link collider has no rigid body — its owner is the articulation, so body is null.
    CHECK(!atRest[0].body.valid());
    // Nothing where the link is *not*.
    CHECK(world.overlapSphere(Vec3{1.0f, 6.0f, 0.0f}, 0.2f).empty());

    // Bend the joint +90°: the link (and its collider) swing to (1,6,0); the broadphase
    // tracks it on the next step.
    art->q(0, pi / 2.0f);
    world.step(1.0f / 60.0f);

    const auto bent = world.overlapSphere(Vec3{1.0f, 6.0f, 0.0f}, 0.2f);
    REQUIRE(bent.size() == 1);
    CHECK(bent[0].collider == collider);
    CHECK(world.overlapSphere(Vec3{2.0f, 5.0f, 0.0f}, 0.2f).empty());
}

TEST_CASE("Spatial.CrossProducts", "[Articulation]")
{
    // crossMotion is the familiar 3-D cross in the angular block, with the mixed
    // angular×linear coupling in the linear block; crossForce is its dual.
    const SpatialVector v{Vec3{0.0f, 0.0f, 1.0f}, Vec3{1.0f, 0.0f, 0.0f}};
    const SpatialVector u{Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}};

    const SpatialVector m = crossMotion(v, u);
    // ang = (0,0,1)×(1,0,0) = (0,1,0); lin = (0,0,1)×(0,1,0) + (1,0,0)×(1,0,0) = (-1,0,0)+0
    CHECK(m.angular.approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 1e-6f));
    CHECK(m.linear.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-6f));

    const SpatialVector f = crossForce(v, u);
    // ang = (0,0,1)×(1,0,0) + (1,0,0)×(0,1,0) = (0,1,0)+(0,0,1) = (0,1,1); lin =
    // (0,0,1)×(0,1,0)=(-1,0,0)
    CHECK(f.angular.approxEqual(Vec3{0.0f, 1.0f, 1.0f}, 1e-6f));
    CHECK(f.linear.approxEqual(Vec3{-1.0f, 0.0f, 0.0f}, 1e-6f));
}

TEST_CASE("Spatial.PluckerTransforms", "[Articulation]")
{
    // Pure translation by (1,0,0): a body spinning about +Z at rate 1 has zero linear
    // velocity at the child origin; at the parent origin the linear part becomes
    // r×(E·ω) = (1,0,0)×(0,0,1) = (0,-1,0).
    const RigidTransform t{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    const SpatialMatrix mt = motionTransform(t);
    const SpatialVector out = mt * SpatialVector{Vec3{0.0f, 0.0f, 1.0f}, Vec3{}};
    CHECK(out.angular.approxEqual(Vec3{0.0f, 0.0f, 1.0f}, 1e-6f));
    CHECK(out.linear.approxEqual(Vec3{0.0f, -1.0f, 0.0f}, 1e-6f));

    // Force transform: a pure force (·; 0,1,0) at the child origin produces a torque
    // r×f = (1,0,0)×(0,1,0) = (0,0,1) about the parent origin.
    const SpatialMatrix ft = forceTransform(t);
    const SpatialVector fout = ft * SpatialVector{Vec3{}, Vec3{0.0f, 1.0f, 0.0f}};
    CHECK(fout.angular.approxEqual(Vec3{0.0f, 0.0f, 1.0f}, 1e-6f));
    CHECK(fout.linear.approxEqual(Vec3{0.0f, 1.0f, 0.0f}, 1e-6f));
}

TEST_CASE("Spatial.RigidInertiaMomentum", "[Articulation]")
{
    // COM at the origin: spatial inertia is block-diagonal — momentum = (I·ω ; m·v).
    const Mat3 icom = Mat3::diagonal(Vec3{2.0f, 3.0f, 4.0f});
    const SpatialMatrix centered = spatialInertia(5.0f, Vec3{}, icom);
    const SpatialVector p =
        centered * SpatialVector{Vec3{1.0f, 1.0f, 1.0f}, Vec3{0.0f, 2.0f, 0.0f}};
    CHECK(p.angular.approxEqual(Vec3{2.0f, 3.0f, 4.0f}, 1e-6f)); // I·ω
    CHECK(p.linear.approxEqual(Vec3{0.0f, 10.0f, 0.0f}, 1e-6f)); // m·v

    // Offset COM, pure translation (ω=0): linear momentum m·v with a torque h×v about the
    // offset origin (h = m·com).
    const SpatialMatrix offset = spatialInertia(5.0f, Vec3{1.0f, 0.0f, 0.0f}, icom);
    const SpatialVector po = offset * SpatialVector{Vec3{}, Vec3{0.0f, 1.0f, 0.0f}};
    CHECK(po.linear.approxEqual(Vec3{0.0f, 5.0f, 0.0f}, 1e-6f));  // m·v
    CHECK(po.angular.approxEqual(Vec3{0.0f, 0.0f, 5.0f}, 1e-6f)); // (5,0,0)×(0,1,0)=(0,0,5)

    // Spatial inertia is symmetric: a == aᵀ, d == dᵀ, c == bᵀ.
    CHECK(offset.a.approxEqual(offset.a.transpose(), 1e-6f));
    CHECK(offset.d.approxEqual(offset.d.transpose(), 1e-6f));
    CHECK(offset.c.approxEqual(offset.b.transpose(), 1e-6f));
}

TEST_CASE("SpatialVector.ArithmeticAndDot", "[Articulation]")
{
    const SpatialVector a{Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 2.0f, 0.0f}};
    const SpatialVector b{Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 0.0f, 3.0f}};

    CHECK((a + b).angular.approxEqual(Vec3{1.0f, 1.0f, 0.0f}, 1e-6f));
    CHECK((a + b).linear.approxEqual(Vec3{0.0f, 2.0f, 3.0f}, 1e-6f));
    CHECK((a * 2.0f).linear.approxEqual(Vec3{0.0f, 4.0f, 0.0f}, 1e-6f));

    // Spatial dot pairs angular·angular + linear·linear: here 0 + 0 = 0 (orthogonal parts).
    CHECK(a.dot(b) == 0.0f);
    // Self-dot is the sum of squared parts: 1 + 4 = 5.
    CHECK(a.dot(a) == 5.0f);
}

namespace
{

// A fixed-base single pendulum: a rod of length 2 (COM 1 m out along local +X), revolute
// about +Z at the origin. Rotational inertia about the COM is the thin-rod value m·L²/12.
Articulation singlePendulum()
{
    Articulation a;
    a.baseFixed(true);
    a.addRootLink(ArticulationLinkDesc{}); // root = fixed world anchor
    ArticulationLinkDesc rod;
    rod.parent = 0;
    rod.joint = ArticulationJointType::Revolute;
    rod.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
    rod.mass = 1.0f;
    rod.comLocal = Vec3{1.0f, 0.0f, 0.0f};
    rod.inertiaLocal = Vec3{0.001f, 0.3333f, 0.3333f};
    a.addLink(rod);
    return a;
}

// A fixed-base double pendulum: two unit rods (length 1, COM 0.5 out), each revolute +Z,
// the second jointed at the end of the first.
Articulation doublePendulum()
{
    Articulation a;
    a.baseFixed(true);
    a.addRootLink(ArticulationLinkDesc{});
    ArticulationLinkDesc rod;
    rod.parent = 0;
    rod.joint = ArticulationJointType::Revolute;
    rod.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
    rod.mass = 1.0f;
    rod.comLocal = Vec3{0.5f, 0.0f, 0.0f};
    rod.inertiaLocal = Vec3{0.001f, 0.0833f, 0.0833f};
    a.addLink(rod); // link 1
    rod.parent = 1;
    rod.parentToJoint = RigidTransform{Quaternion::identity(), Vec3{1.0f, 0.0f, 0.0f}};
    a.addLink(rod); // link 2
    return a;
}

} // namespace

TEST_CASE("Articulation.SinglePendulumReleaseAccelerationMatchesAnalytic", "[Articulation]")
{
    // A rod released from horizontal (q = 0) has angular acceleration q̈ = −m·g·L / (I_com +
    // m·L²): the gravity torque m·g·L about the pivot over the pivot inertia. Here
    // −1·9.81·1 / (0.3333 + 1) = −7.358 rad/s². The ABA must reproduce this exactly.
    Articulation a = singlePendulum();
    a.computeAccelerations(Vec3{0.0f, -9.81f, 0.0f});
    CHECK(a.qDDot()[0] == Catch::Approx(-7.358f).margin(0.01f));
}

TEST_CASE("Articulation.SinglePendulumConservesEnergy", "[Articulation]")
{
    // A non-chaotic single pendulum must conserve total mechanical energy under the
    // semi-implicit integrator (drift is O(dt); < 0.1% at dt = 1/8000 over a full swing).
    Articulation a = singlePendulum();
    a.q(0, 1.5f); // released from near-horizontal
    const Vec3 g{0.0f, -9.81f, 0.0f};
    const float iPivot = 0.3333f + 1.0f; // I_com + m·L²
    const float dt = 1.0f / 8000.0f;

    const auto energy = [&]
    {
        a.forwardKinematics();
        const Vec3 com = a.linkWorld(1).transformPoint(Vec3{1.0f, 0.0f, 0.0f});
        return 0.5f * iPivot * a.qDot()[0] * a.qDot()[0] + 9.81f * com.y();
    };

    const float e0 = energy();
    float emin = e0;
    float emax = e0;
    for (int i = 0; i < 8000; ++i)
    {
        a.computeAccelerations(g);
        a.integrate(dt);
        const float e = energy();
        emin = std::min(emin, e);
        emax = std::max(emax, e);
    }
    CHECK((emax - emin) / std::abs(e0) < 0.01f);
}

TEST_CASE("Articulation.DoublePendulumInterLinkCouplingIsStable", "[Articulation]")
{
    // Exercises the inter-link articulated-inertia transform (the single pendulum's only
    // link folds into the *fixed* root, so it never tests link→link coupling). Under zero
    // gravity the energy is pure kinetic, so a bent, moving double pendulum must keep its
    // joint speeds bounded — a runaway would expose a bad coupling transform.
    Articulation a = doublePendulum();
    a.q(1, 0.8f);
    a.qDot(0, 0.5f);
    a.qDot(1, -0.5f);
    const Vec3 zeroG{};
    const float dt = 1.0f / 8000.0f;
    float maxSpeed = 0.0f;
    for (int i = 0; i < 8000; ++i)
    {
        a.computeAccelerations(zeroG);
        a.integrate(dt);
        maxSpeed = std::max(maxSpeed, std::max(std::abs(a.qDot()[0]), std::abs(a.qDot()[1])));
    }
    CHECK(maxSpeed < 2.0f); // started near 0.5; a correct coupling keeps it O(1)
    CHECK(std::isfinite(a.qDot()[0]));
    CHECK(std::isfinite(a.qDot()[1]));
}

TEST_CASE("Articulation.DoublePendulumSettlesUnderDamping", "[Articulation]")
{
    // A chaotic gravity-driven double pendulum under a non-symplectic explicit integrator
    // drifts energy and, undamped at the physics substep rate, diverges — a known limit of
    // explicit integration, not an ABA fault (see the energy-conserving tests above). Passive
    // joint damping (which ragdolls carry) dissipates that: over 6 s at the substep h = 1/480
    // the motion stays finite and bleeds toward rest.
    Articulation a = doublePendulum();
    a.q(0, 1.2f);
    a.q(1, 0.4f);
    const Vec3 g{0.0f, -9.81f, 0.0f};
    const float dt = 1.0f / 480.0f;
    for (int i = 0; i < static_cast<int>(6.0f / dt); ++i)
    {
        a.computeAccelerations(g, 0.2f); // passive damping τ = −0.2·q̇
        a.integrate(dt);
    }
    CHECK(std::isfinite(a.qDot()[0]));
    CHECK(std::isfinite(a.qDot()[1]));
    CHECK(std::abs(a.qDot()[0]) < 5.0f);
    CHECK(std::abs(a.qDot()[1]) < 5.0f);
}

TEST_CASE("Articulation.EffectiveMassAndImpulseResponseMatchAnalytic", "[Articulation]")
{
    // The ConstraintBody seam on a single pendulum: for a tangential impulse at the bob
    // (2 m out), the inverse effective mass is L_bob²/I_pivot and the response is
    // Δq̇ = P·L_bob/I_pivot — the reflected point mass a contact would see.
    Articulation a = singlePendulum();
    a.factorizeArticulatedInertia();
    a.computeLinkVelocities();

    const Vec3 bob{2.0f, 0.0f, 0.0f};
    const Vec3 tangent{0.0f, 1.0f, 0.0f};
    const float iPivot = 0.3333f + 1.0f;

    const float invEff = a.inverseEffectiveMass(1, bob, tangent);
    CHECK(invEff > 0.0f); // a mass response is positive
    CHECK(invEff == Catch::Approx(4.0f / iPivot).margin(1e-3f));

    const float impulse = 0.5f;
    a.applyImpulse(1, bob, tangent * impulse);
    CHECK(a.qDot()[0] == Catch::Approx(impulse * 2.0f / iPivot).margin(1e-3f));
}

TEST_CASE("Articulation.ImpulseThroughLinkageSolvesContactVelocity", "[Articulation]")
{
    // The Phase C go/no-go: a single impulse solved through the articulated response
    // (J = −vₙ / invEffMass) must drive a point's constraint velocity to zero on a *two*-link
    // chain — i.e. the impulse propagates correctly through the inter-link coupling, exactly
    // what a contact normal row needs. A rigid body trivially does this; the articulation
    // must too for the contact solver to treat a link as a constraint body.
    Articulation d = doublePendulum();
    d.q(0, 0.3f);
    d.q(1, -0.5f);
    d.qDot(0, 1.5f);
    d.qDot(1, -2.0f);
    d.factorizeArticulatedInertia();
    d.computeLinkVelocities();

    const Vec3 tip = d.linkWorld(2).transformPoint(Vec3{1.0f, 0.0f, 0.0f});
    const Vec3 normal{0.0f, 1.0f, 0.0f};
    const float vn0 = Vec3::dotProduct(d.pointVelocity(2, tip), normal);
    REQUIRE(std::abs(vn0) > 0.1f); // the tip is actually moving along the constraint

    const float invEff = d.inverseEffectiveMass(2, tip, normal);
    d.applyImpulse(2, tip, normal * (-vn0 / invEff));

    d.computeLinkVelocities();
    const float vn1 = Vec3::dotProduct(d.pointVelocity(2, tip), normal);
    CHECK(std::abs(vn1) < 1e-4f);
}

TEST_CASE("Articulation.SphericalJointMatchesRevoluteInPlane", "[Articulation]")
{
    // A spherical (3-DOF) joint driven only in the plane must reproduce the equivalent
    // revolute joint exactly — same release acceleration (gravity torque about +Z only) and
    // same bob trajectory. Validates the multi-DOF ABA against the trusted 1-DOF path.
    const auto make = [](ArticulationJointType jt)
    {
        Articulation a;
        a.baseFixed(true);
        a.addRootLink(ArticulationLinkDesc{});
        ArticulationLinkDesc rod;
        rod.parent = 0;
        rod.joint = jt;
        rod.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
        rod.mass = 1.0f;
        rod.comLocal = Vec3{1.0f, 0.0f, 0.0f};
        rod.inertiaLocal = Vec3{0.01f, 0.34f, 0.34f};
        a.addLink(rod);
        return a;
    };

    Articulation spherical = make(ArticulationJointType::Spherical);
    spherical.computeAccelerations(Vec3{0.0f, -9.81f, 0.0f});
    // Gravity torque about +Z only: q̈ = (0, 0, −mgL/(I_com+mL²)).
    CHECK(spherical.qDDot()[0] == Catch::Approx(0.0f).margin(1e-4f));
    CHECK(spherical.qDDot()[1] == Catch::Approx(0.0f).margin(1e-4f));
    CHECK(spherical.qDDot()[2] == Catch::Approx(-9.81f / 1.34f).margin(0.01f));

    Articulation revolute = make(ArticulationJointType::Revolute);
    const Vec3 g{0.0f, -9.81f, 0.0f};
    float maxDiff = 0.0f;
    for (int i = 0; i < 4000; ++i)
    {
        spherical.computeAccelerations(g);
        spherical.integrate(1.0f / 2000.0f);
        revolute.computeAccelerations(g);
        revolute.integrate(1.0f / 2000.0f);
        spherical.forwardKinematics();
        revolute.forwardKinematics();
        const Vec3 bs = spherical.linkWorld(1).transformPoint(Vec3{2.0f, 0.0f, 0.0f});
        const Vec3 br = revolute.linkWorld(1).transformPoint(Vec3{2.0f, 0.0f, 0.0f});
        maxDiff = std::max(maxDiff, (bs - br).magnitude());
    }
    CHECK(maxDiff < 1e-3f);
}

TEST_CASE("Articulation.SphericalJointConservesEnergyOutOfPlane", "[Articulation]")
{
    // An out-of-plane precessing spherical pendulum stresses the quaternion integration
    // (a wrong left-vs-right multiply pumps energy badly). Total mechanical energy must hold.
    Articulation a;
    a.baseFixed(true);
    a.addRootLink(ArticulationLinkDesc{});
    ArticulationLinkDesc rod;
    rod.parent = 0;
    rod.joint = ArticulationJointType::Spherical;
    rod.mass = 1.0f;
    rod.comLocal = Vec3{1.0f, 0.0f, 0.0f};
    rod.inertiaLocal = Vec3{0.05f, 0.34f, 0.34f};
    a.addLink(rod);
    a.qDot(1, 3.0f);
    a.qDot(2, -2.0f);

    const Vec3 g{0.0f, -9.81f, 0.0f};
    const float ix = 0.05f;
    const float iyz = 0.34f + 1.0f; // perpendicular inertia about the pivot
    const auto energy = [&]
    {
        a.forwardKinematics();
        const Vec3 com = a.linkWorld(1).transformPoint(Vec3{1.0f, 0.0f, 0.0f});
        const float wx = a.qDot()[0];
        const float wy = a.qDot()[1];
        const float wz = a.qDot()[2];
        return 0.5f * (ix * wx * wx + iyz * wy * wy + iyz * wz * wz) + 9.81f * com.y();
    };

    const float e0 = energy();
    float emin = e0;
    float emax = e0;
    for (int i = 0; i < 16000; ++i)
    {
        a.computeAccelerations(g);
        a.integrate(1.0f / 8000.0f);
        const float e = energy();
        emin = std::min(emin, e);
        emax = std::max(emax, e);
    }
    CHECK((emax - emin) / std::abs(e0) < 0.02f);
}

namespace
{

// Swing angle of link 1's spherical joint about `twistAxis` (the cone-limit measure).
float swingAngle(const Articulation& a, const Vec3& twistAxis)
{
    const Quaternion q = a.jointRotation(1);
    const float d = q.x() * twistAxis.x() + q.y() * twistAxis.y() + q.z() * twistAxis.z();
    const Quaternion twist = Quaternion::normalise(
        Quaternion{twistAxis.x() * d, twistAxis.y() * d, twistAxis.z() * d, q.w()});
    const Quaternion swing = q * twist.conjugate();
    const Vec3 sv{swing.x(), swing.y(), swing.z()};
    const float s = sv.magnitude();
    return 2.0f * std::atan2(s, swing.w());
}

float twistAngle(const Articulation& a, const Vec3& twistAxis)
{
    const Quaternion q = a.jointRotation(1);
    const float d = q.x() * twistAxis.x() + q.y() * twistAxis.y() + q.z() * twistAxis.z();
    return 2.0f * std::atan2(d, q.w());
}

// A spherical joint with a rod hanging along +Z (its twist axis), optionally cone-limited.
Articulation sphericalRod(float swingLimit, float twistLimit, float stiffness, float damping)
{
    Articulation a;
    a.baseFixed(true);
    a.addRootLink(ArticulationLinkDesc{});
    ArticulationLinkDesc rod;
    rod.parent = 0;
    rod.joint = ArticulationJointType::Spherical;
    rod.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
    rod.mass = 1.0f;
    rod.comLocal = Vec3{0.0f, 0.0f, 1.0f};
    rod.inertiaLocal = Vec3{0.34f, 0.34f, 0.05f};
    rod.swingLimit = swingLimit;
    rod.twistLimit = twistLimit;
    rod.limitStiffness = stiffness;
    rod.limitDamping = damping;
    a.addLink(rod);
    return a;
}

} // namespace

TEST_CASE("Articulation.ConeLimitHoldsSphericalSwing", "[Articulation]")
{
    // Gravity pulls a spherical-jointed rod sideways; unconstrained it swings far past a
    // tight cone. The cone-twist limit (a passive restoring torque past swingLimit) must
    // hold it near the cone, while an unlimited joint swings freely well beyond.
    const Vec3 twistAxis{0.0f, 0.0f, 1.0f};
    const Vec3 g{0.0f, -9.81f, 0.0f};
    const float limit = 0.5f;

    Articulation limited = sphericalRod(limit, pi, 400.0f, 10.0f);
    Articulation free = sphericalRod(pi, pi, 0.0f, 0.0f);
    for (int i = 0; i < 3000; ++i)
    {
        limited.computeAccelerations(g, 0.1f);
        limited.integrate(1.0f / 240.0f);
        free.computeAccelerations(g, 0.1f);
        free.integrate(1.0f / 240.0f);
    }
    const float limitedSwing = swingAngle(limited, twistAxis);
    const float freeSwing = swingAngle(free, twistAxis);

    CHECK(std::isfinite(limitedSwing));
    CHECK(limitedSwing < limit + 0.1f);     // held near the cone (soft-limit compliance)
    CHECK(freeSwing > limitedSwing + 0.3f); // the unlimited joint swings much further
}

TEST_CASE("Articulation.TwistLimitArrestsSpin", "[Articulation]")
{
    // A spin about the twist axis must be arrested at ±twistLimit, not run free.
    const Vec3 twistAxis{0.0f, 0.0f, 1.0f};
    Articulation a = sphericalRod(pi, 0.6f, 200.0f, 8.0f);
    a.qDot(2, 4.0f); // spin about the twist axis
    for (int i = 0; i < 2000; ++i)
    {
        a.computeAccelerations(Vec3{}, 0.05f);
        a.integrate(1.0f / 240.0f);
    }
    CHECK(std::abs(twistAngle(a, twistAxis)) < 0.7f); // arrested near the ±0.6 limit
}

TEST_CASE("Articulation.WithinLimitMotionIsUnconstrained", "[Articulation]")
{
    // Inside the cone the limit adds nothing: a gravity-free rod given a small swing keeps
    // rotating freely (no spurious restoring torque below the limit).
    const Vec3 twistAxis{0.0f, 0.0f, 1.0f};
    Articulation a = sphericalRod(1.0f, pi, 400.0f, 10.0f);
    a.qDot(0, 0.3f);
    for (int i = 0; i < 200; ++i)
    {
        a.computeAccelerations(Vec3{}, 0.0f); // no gravity, no global damping
        a.integrate(1.0f / 240.0f);
    }
    // Free rotation at 0.3 rad/s for 200/240 s ≈ 0.25 rad — untouched by the (inactive) limit.
    CHECK(swingAngle(a, twistAxis) == Catch::Approx(0.25f).margin(0.02f));
}

TEST_CASE("Articulation.LinkRestsOnFloorThroughConstraintBody", "[Articulation]")
{
    // The Phase D gate: a fixed-base pendulum whose bob swings down onto a static floor
    // plane must stop AT the surface (no tunnelling) and settle — the full path exercised:
    // ABA free dynamics + the ConstraintBody seam (normal contact) over the TGS substep loop.
    Articulation a;
    a.baseFixed(true);
    a.addRootLink(ArticulationLinkDesc{});
    ArticulationLinkDesc rod;
    rod.parent = 0;
    rod.joint = ArticulationJointType::Revolute;
    rod.jointAxis = Vec3{0.0f, 0.0f, 1.0f};
    rod.mass = 1.0f;
    rod.comLocal = Vec3{1.0f, 0.0f, 0.0f};
    rod.inertiaLocal = Vec3{0.01f, 0.34f, 0.34f};
    a.addLink(rod);
    a.baseTransform(RigidTransform{Quaternion::identity(), Vec3{0.0f, 2.0f, 0.0f}}); // pivot at y=2

    // Bob at the rod tip (2 m out); floor half-space y >= 0.5.
    const float floorY = 0.5f;
    const std::array<ArticulationPlaneContact, 1> contacts{
        ArticulationPlaneContact{1, Vec3{2.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}, floorY, 0.5f}};
    const Vec3 g{0.0f, -9.81f, 0.0f};

    float minBobY = 1e9f;
    for (int i = 0; i < 400; ++i)
    {
        stepArticulationOnPlanes(a, contacts, g, 1.0f / 60.0f, 0.05f);
        a.forwardKinematics();
        minBobY = std::min(minBobY, a.linkWorld(1).transformPoint(Vec3{2.0f, 0.0f, 0.0f}).y());
    }
    a.forwardKinematics();
    const float bobY = a.linkWorld(1).transformPoint(Vec3{2.0f, 0.0f, 0.0f}).y();

    CHECK(minBobY > floorY - 0.01f);                    // never tunnelled through the floor
    CHECK(bobY == Catch::Approx(floorY).margin(0.01f)); // rests on it
    CHECK(std::abs(a.qDot()[0]) < 0.05f);               // came to rest
}
